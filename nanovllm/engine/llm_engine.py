import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


def _run_pd_decode_group(
    config: Config,
    decode_master_global_rank: int,
    decode_worker_events: list,
    decode_in_queue,
    decode_out_queue,
) -> None:
    """Entry point for the decode-group master subprocess in PD mode.

    This function is the ``target`` of a :class:`multiprocessing.Process`.
    It:
    1. Creates a :class:`ModelRunner` for the decode master (local rank 0).
    2. Enters a command loop that is driven by the main (prefill-master)
       process via *decode_in_queue*.

    The *decode worker* subprocesses are spawned **before** this function
    runs (just like regular TP workers).  Their events are collected here
    so that the decode master's ModelRunner can coordinate them via shared
    memory.

    Command protocol (items placed in *decode_in_queue*)
    -----------------------------------------------------
    ``("step", decode_seqs, new_block_ids)``
        Run decode for *decode_seqs*, then put results into
        *decode_out_queue* **before** calling ``receive_kv`` for
        *new_block_ids*.  This ordering lets the prefill-master overlap
        its ``send_kv`` with the decode run.

    ``("exit",)``
        Tear down and return.
    """
    model_runner = ModelRunner(
        config,
        decode_master_global_rank,
        decode_worker_events,
        "decode",
    )

    # Signal to the prefill master that the decode group is initialised
    decode_out_queue.put("ready")

    while True:
        msg = decode_in_queue.get()
        cmd = msg[0]

        if cmd == "step":
            _, decode_seqs, new_block_ids = msg
            # Run decode (may be empty on the very first step)
            if decode_seqs:
                token_ids = model_runner.call("run", decode_seqs, False)
            else:
                token_ids = []
            # Put results into the queue BEFORE receiving KV so the
            # prefill-master can read them as soon as send_kv finishes.
            decode_out_queue.put(token_ids)
            # Receive KV for the newly prefilled sequences
            if new_block_ids:
                model_runner.call("receive_kv", new_block_ids)

        elif cmd == "exit":
            model_runner.call("exit")
            break


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.ps = []
        ctx = mp.get_context("spawn")

        if config.enable_pd_separation:
            self._init_pd(config, ctx)
        else:
            self._init_normal(config, ctx)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self._exited = False
        atexit.register(self.exit)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_normal(self, config: Config, ctx) -> None:
        """Original single-group initialisation path."""
        self.events = []
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.scheduler = Scheduler(config)

    def _init_pd(self, config: Config, ctx) -> None:
        """PD-separation initialisation path.

        Process layout (global ranks)
        --------------------------------
        0 … P-1   : prefill group  (rank 0 = this process / prefill master)
        P … P+D-1 : decode group   (rank P = decode master subprocess)
        """
        from nanovllm.engine.pd_scheduler import PDScheduler

        P = config.num_prefill_ranks
        D = config.num_decode_ranks

        # ---- Prefill workers (global ranks 1 … P-1) ----
        self.prefill_events: list = []
        for i in range(1, P):
            event = ctx.Event()
            process = ctx.Process(
                target=ModelRunner,
                args=(config, i, event, "prefill"),
            )
            process.start()
            self.ps.append(process)
            self.prefill_events.append(event)

        # ---- Decode workers (global ranks P+1 … P+D-1) ----
        decode_worker_events: list = []
        for i in range(P + 1, P + D):
            event = ctx.Event()
            process = ctx.Process(
                target=ModelRunner,
                args=(config, i, event, "decode"),
            )
            process.start()
            self.ps.append(process)
            decode_worker_events.append(event)

        # ---- Decode master (global rank P) ----
        self.decode_in_queue = ctx.Queue()
        self.decode_out_queue = ctx.Queue()
        decode_master = ctx.Process(
            target=_run_pd_decode_group,
            args=(config, P, decode_worker_events, self.decode_in_queue, self.decode_out_queue),
        )
        decode_master.start()
        self.ps.append(decode_master)

        # ---- Prefill master (this process, global rank 0) ----
        self.model_runner = ModelRunner(config, 0, self.prefill_events, "prefill")

        # Wait for the decode master to finish initialising its ModelRunner
        ready = self.decode_out_queue.get()
        assert ready == "ready", f"Unexpected decode-master signal: {ready!r}"

        self.scheduler = PDScheduler(config)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def exit(self):
        if self._exited:
            return
        self._exited = True
        if self.config.enable_pd_separation:
            self.decode_in_queue.put(("exit",))
            self.model_runner.call("exit")
        else:
            self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # ------------------------------------------------------------------
    # Request management
    # ------------------------------------------------------------------

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self):
        if self.config.enable_pd_separation:
            return self._step_pd()
        return self._step_normal()

    def _step_normal(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def _step_pd(self):
        """One PD-separated step.

        Overlap
        -------
        1. Dispatch decode job to decode master (non-blocking queue put).
        2. Run prefill on the prefill GPUs (blocks this thread / GPU 0…P-1).
        3. Send KV cache to decode group via NCCL point-to-point.
           This blocks until the decode master calls ``receive_kv``, which
           it does *after* the decode job is done.
        4. Get decode results (already in the queue by this point).
        5. Bookkeeping.

        In steady state, step 2 (prefill) and the decode run happen
        concurrently on separate GPU sets, giving true PD overlap.
        """
        from nanovllm.engine.pd_scheduler import PDScheduler
        scheduler: PDScheduler = self.scheduler  # type: ignore[assignment]

        prefill_seqs = scheduler.schedule_prefill()
        decode_seqs = scheduler.schedule_decode()

        # Collect block IDs touched during this prefill batch (deduplicated)
        new_block_ids: list[int] = []
        if prefill_seqs:
            new_block_ids = list({bid for seq in prefill_seqs for bid in seq.block_table})

        # 1. Tell decode master what to do (non-blocking)
        self.decode_in_queue.put(("step", decode_seqs, new_block_ids))

        # 2. Run prefill (overlaps with decode on the decode GPUs)
        prefill_token_ids: list[int] = []
        if prefill_seqs:
            prefill_token_ids = self.model_runner.call("run", prefill_seqs, True)
            scheduler.postprocess_prefill(prefill_seqs, prefill_token_ids)

        # 3. Transfer KV cache to decode group; this NCCL send blocks until
        #    the decode master has finished decoding and called receive_kv.
        if new_block_ids:
            self.model_runner.call("send_kv", new_block_ids)

        # 4. Collect decode results (decode master put them before recv_kv)
        decode_token_ids: list[int] = self.decode_out_queue.get()

        # 5. Bookkeeping
        finished_seqs = scheduler.postprocess_decode(decode_seqs, decode_token_ids)
        if prefill_seqs:
            scheduler.complete_transfer(prefill_seqs)

        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in finished_seqs
        ]
        # Report positive tokens for prefill, negative for decode
        if prefill_seqs:
            num_tokens = sum(len(seq) for seq in prefill_seqs)
        else:
            num_tokens = -len(decode_seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
