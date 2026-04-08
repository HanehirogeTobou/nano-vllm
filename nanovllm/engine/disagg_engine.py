"""PD (Prefill-Decode) disaggregation for single-machine multi-GPU setups.

Design
------
On the same machine, KV cache is transferred through a shared
:class:`multiprocessing.Queue` instead of a TCP socket.  Both engines are
meant to run in **separate processes** that share the queue created before
spawning:

    import torch.multiprocessing as mp
    from nanovllm import PrefillEngine, DecodeEngine, SamplingParams

    def run_prefill(model, kv_queue, prompts, sampling_params):
        engine = PrefillEngine(model, kv_queue, nccl_port=2333)
        engine.generate(prompts, sampling_params)

    def run_decode(model, kv_queue, num_seqs):
        engine = DecodeEngine(model, kv_queue, nccl_port=2334)
        return engine.decode_all(num_seqs)

    if __name__ == "__main__":
        ctx = mp.get_context("spawn")
        kv_queue = ctx.Queue()
        p = ctx.Process(target=run_decode, args=(model, kv_queue, len(prompts)))
        p.start()
        run_prefill(model, kv_queue, prompts, SamplingParams(max_tokens=256))
        p.join()

Notes
-----
* Use different ``nccl_port`` values for prefill and decode so their internal
  NCCL process groups do not conflict on the same machine.
* Tensor-parallel size must be identical on both sides.
* KV tensors are moved to CPU before being placed in the queue; they are
  moved back to the decode GPU inside :meth:`DecodeEngine._inject_sequence`.
"""

from __future__ import annotations

import atexit
import queue as _queue
from dataclasses import dataclass, fields
from time import perf_counter

import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

_DURATION_EPSILON = 1e-9  # prevents division by zero in throughput calculations


# ---------------------------------------------------------------------------
# KV transfer payload (replaces KVTransferMeta + TCP framing)
# ---------------------------------------------------------------------------

@dataclass
class _KVPayload:
    seq_id: int
    token_ids: list
    num_prompt_tokens: int
    temperature: float
    max_tokens: int
    ignore_eos: bool


# ---------------------------------------------------------------------------
# Shared helper: build Config + spawn TP workers + create rank-0 ModelRunner
# ---------------------------------------------------------------------------

def _build_engine(
    model: str,
    kwargs: dict,
) -> tuple[Config, ModelRunner, list, AutoTokenizer, Scheduler]:
    config_fields = {field.name for field in fields(Config)}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    config = Config(model, **config_kwargs)

    ps = []
    events = []
    ctx = mp.get_context("spawn")
    for i in range(1, config.tensor_parallel_size):
        event = ctx.Event()
        process = ctx.Process(target=ModelRunner, args=(config, i, event))
        process.start()
        ps.append(process)
        events.append(event)

    model_runner = ModelRunner(config, 0, events)
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    config.eos = tokenizer.eos_token_id
    scheduler = Scheduler(config)
    return config, model_runner, ps, tokenizer, scheduler


# ---------------------------------------------------------------------------
# PrefillEngine
# ---------------------------------------------------------------------------

class PrefillEngine:
    """Prefill-only engine for single-machine PD disaggregation.

    Runs the prefill step, samples the first token, then places the KV cache
    blocks together with sequence metadata into *kv_queue* for the paired
    :class:`DecodeEngine` to consume.

    Parameters
    ----------
    model:
        Path to the model directory.
    kv_queue:
        A :class:`multiprocessing.Queue` shared with the decode process.
        Create it with ``mp.get_context("spawn").Queue()`` before spawning
        either process.
    **kwargs:
        Forwarded to :class:`~nanovllm.config.Config`
        (e.g. ``tensor_parallel_size``, ``nccl_port``, ``max_model_len``).
    """

    def __init__(self, model: str, kv_queue: mp.Queue, **kwargs) -> None:
        config, model_runner, ps, tokenizer, scheduler = _build_engine(model, kwargs)
        self.config = config
        self.model_runner = model_runner
        self.ps = ps
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.kv_queue = kv_queue
        atexit.register(self._exit)

    def _exit(self) -> None:
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> int:
        """Enqueue a generation request and return its sequence id."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self) -> int:
        """Run one prefill batch and transfer resulting KV caches to decode.

        For each sequence in the batch:

        1. Run the prefill forward pass and sample the first token.
        2. Extract the KV blocks from GPU memory (as a CPU tensor).
        3. Put ``(_KVPayload, kv_tensor)`` into *kv_queue*.
        4. Deallocate the blocks on the prefill side.

        Returns:
            Number of sequences transferred in this step.
        """
        if not self.scheduler.waiting:
            return 0

        seqs, is_prefill = self.scheduler.schedule()
        assert is_prefill, "PrefillEngine unexpectedly scheduled a decode batch"

        token_ids = self.model_runner.call("run", seqs, True)

        for seq, token_id in zip(seqs, token_ids):
            # Copy KV blocks to CPU *before* appending the first decode token
            # so the tensor shape matches the allocated block_table.
            kv_data = self.model_runner.kv_cache[:, :, seq.block_table].cpu()

            seq.append_token(token_id)

            payload = _KVPayload(
                seq_id=seq.seq_id,
                token_ids=list(seq.token_ids),
                num_prompt_tokens=seq.num_prompt_tokens,
                temperature=seq.temperature,
                max_tokens=seq.max_tokens,
                ignore_eos=seq.ignore_eos,
            )
            self.kv_queue.put((payload, kv_data))

            # Release blocks so they can be reused for subsequent prefills.
            self.scheduler.block_manager.deallocate(seq)
            seq.status = SequenceStatus.FINISHED
            self.scheduler.running.remove(seq)

        return len(seqs)

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> int:
        """Prefill all *prompts* and send KV caches to the decode engine.

        Returns:
            Total number of sequences transferred.
        """
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Prefilling", dynamic_ncols=True)
        total = 0
        while not self.is_finished():
            n = self.step()
            total += n
            if use_tqdm:
                pbar.update(n)
        if use_tqdm:
            pbar.close()
        # Signal to the decode engine that no more sequences are coming.
        self.kv_queue.put(None)
        return total


# ---------------------------------------------------------------------------
# DecodeEngine
# ---------------------------------------------------------------------------

class DecodeEngine:
    """Decode-only engine for single-machine PD disaggregation.

    Reads ``(_KVPayload, kv_tensor)`` pairs from *kv_queue*, materialises the
    KV data into the local GPU cache, and runs decode iterations until all
    sequences finish.

    Parameters
    ----------
    model:
        Path to the model directory.
    kv_queue:
        The same :class:`multiprocessing.Queue` passed to :class:`PrefillEngine`.
    **kwargs:
        Forwarded to :class:`~nanovllm.config.Config`.
        Use a different ``nccl_port`` than the prefill engine to avoid port
        conflicts (e.g. ``nccl_port=2334``).
    """

    def __init__(self, model: str, kv_queue: mp.Queue, **kwargs) -> None:
        config, model_runner, ps, tokenizer, scheduler = _build_engine(model, kwargs)
        self.config = config
        self.model_runner = model_runner
        self.ps = ps
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.kv_queue = kv_queue

        self._results: dict[int, list[int]] = {}
        self._prefill_done = False

        atexit.register(self._exit)

    def _exit(self) -> None:
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_queue(self, block: bool = False, timeout: float = 0.05) -> None:
        """Pull all available items from *kv_queue* and inject them.

        When *block* is True the call waits up to *timeout* seconds for the
        first item before returning.
        """
        first = True
        while True:
            try:
                item = self.kv_queue.get(block=(block and first), timeout=timeout)
                first = False
                if item is None:
                    self._prefill_done = True
                else:
                    self._inject_sequence(*item)
            except _queue.Empty:
                break

    def _inject_sequence(self, payload: _KVPayload, kv_data: torch.Tensor) -> None:
        """Reconstruct a :class:`Sequence` from a transferred payload.

        The KV data is loaded into the local GPU cache, and the sequence is
        added to the running queue so that decode can begin immediately.
        """
        num_kv_blocks: int = kv_data.shape[2]

        # Reconstruct the Sequence without __init__ so we can reuse the
        # prefill-side seq_id directly (bypassing the global counter).
        seq = object.__new__(Sequence)
        seq.seq_id = payload.seq_id
        seq.status = SequenceStatus.RUNNING
        seq.token_ids = list(payload.token_ids)
        seq.last_token = payload.token_ids[-1]
        seq.num_tokens = len(payload.token_ids)
        seq.num_prompt_tokens = payload.num_prompt_tokens
        seq.num_cached_tokens = 0
        seq.block_table = []
        seq.temperature = payload.temperature
        seq.max_tokens = payload.max_tokens
        seq.ignore_eos = payload.ignore_eos

        self.scheduler.block_manager.allocate_transferred(seq, num_kv_blocks)
        self.model_runner.kv_cache[:, :, seq.block_table] = kv_data.to(
            self.model_runner.kv_cache.device
        )

        # Ensure complete transferred blocks are hashed so that
        # BlockManager.may_append's assertion passes when a new block is
        # needed on the very first decode step.
        #
        # The tricky case: prefill fills exactly k * block_size tokens.
        # After appending d0, len(seq) == k * block_size + 1, so
        # may_append fires the `% block_size == 1` branch which asserts
        # `last_block.hash != -1` before allocating the new block.
        # We pre-hash the transferred blocks here to satisfy that assertion.
        bm = self.scheduler.block_manager
        if len(seq) % bm.block_size == 1:
            h = -1
            for i, block_id in enumerate(seq.block_table):
                toks = list(seq.block(i))
                h = bm.compute_hash(toks, h)
                bm.blocks[block_id].update(h, toks)
                bm.hash_to_block_id[h] = block_id

        self.scheduler.running.append(seq)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> list[tuple[int, list[int]]]:
        """Inject pending transfers and run one decode iteration.

        If the running queue is empty the call blocks briefly waiting for
        new transfers from the prefill engine.

        Returns:
            List of ``(seq_id, completion_token_ids)`` for sequences that
            finished during this step.
        """
        if not self.scheduler.running:
            # Block until at least one sequence arrives (or prefill signals done).
            self._drain_queue(block=True)
        else:
            # Non-blocking drain: pick up any additional ready sequences.
            self._drain_queue(block=False)

        if not self.scheduler.running:
            return []

        seqs, is_prefill = self.scheduler.schedule()
        assert not is_prefill, "DecodeEngine unexpectedly scheduled a prefill batch"

        token_ids = self.model_runner.call("run", seqs, False)
        self.scheduler.postprocess(seqs, token_ids)

        finished = [
            (seq.seq_id, list(seq.completion_token_ids))
            for seq in seqs
            if seq.is_finished
        ]
        for seq_id, tids in finished:
            self._results[seq_id] = tids

        return finished

    def has_pending(self) -> bool:
        """Return True while there are running sequences or unseen transfers."""
        return bool(self.scheduler.running) or not (
            self._prefill_done and self.kv_queue.empty()
        )

    def get_result(self, seq_id: int) -> list[int] | None:
        """Return completion token ids for *seq_id*, or ``None`` if not done."""
        return self._results.get(seq_id)

    def decode_all(self, num_seqs: int, use_tqdm: bool = True) -> dict[int, str]:
        """Run the decode loop until *num_seqs* sequences are finished.

        Parameters
        ----------
        num_seqs:
            Total number of sequences submitted to the prefill engine.
            The loop exits once this many results have been collected.

        Returns:
            Mapping ``{seq_id: decoded_text}`` for every finished sequence.
        """
        if use_tqdm:
            pbar = tqdm(total=num_seqs, desc="Decoding", dynamic_ncols=True)
        prev = 0
        while len(self._results) < num_seqs:
            t = perf_counter()
            finished = self.step()
            if finished and use_tqdm:
                pbar.set_postfix({"Decode": f"{int(len(finished) / max(perf_counter() - t, _DURATION_EPSILON))}tok/s"})
                curr = len(self._results)
                pbar.update(curr - prev)
                prev = curr
        if use_tqdm:
            pbar.close()
        return {
            sid: self.tokenizer.decode(tids)
            for sid, tids in self._results.items()
        }

