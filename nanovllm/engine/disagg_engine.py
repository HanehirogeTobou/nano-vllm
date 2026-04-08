"""Prefill-Decode (PD) disaggregated engines for nano-vllm.

This module provides :class:`PrefillEngine` and :class:`DecodeEngine` that
mirror vLLM's PD disaggregation design:

* **PrefillEngine** handles only the prefill phase.  After each prefill batch
  it extracts the computed KV cache blocks from GPU memory, bundles them with
  the sequence metadata (including the first generated token), and ships
  everything over TCP to a paired :class:`DecodeEngine`.

* **DecodeEngine** listens for incoming KV transfers on a background thread.
  Each received payload is materialised into the local GPU KV cache, and the
  reconstructed sequence is injected directly into the decode scheduler so
  that generation continues seamlessly.

Typical deployment (two separate processes / machines):

    # Process A – prefill
    from nanovllm import PrefillEngine, SamplingParams
    engine = PrefillEngine(model_path, kv_host="host_B", kv_port=29500)
    engine.generate(prompts, SamplingParams(max_tokens=256))

    # Process B – decode
    from nanovllm import DecodeEngine, SamplingParams
    engine = DecodeEngine(model_path, kv_host="0.0.0.0", kv_port=29500)
    results = engine.decode_all()

Notes
-----
* Tensor-parallel size must be identical on both instances.
* For tensor_parallel_size > 1 the KV transfer covers only rank-0's shard;
  cross-rank synchronisation during transfer is not yet supported.
"""

from __future__ import annotations

import atexit
from dataclasses import fields
from time import perf_counter

import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.kv_transfer import KVReceiver, KVSender, KVTransferMeta
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


# ---------------------------------------------------------------------------
# Shared helper: build Config + spawn TP workers + create rank-0 ModelRunner
# ---------------------------------------------------------------------------

def _build_engine(model: str, kwargs: dict) -> tuple[Config, ModelRunner, list, AutoTokenizer, Scheduler]:
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
    """Prefill-only engine.

    Runs the prefill step, then ships the resulting KV cache blocks and
    sequence metadata (including the first generated token) to the paired
    :class:`DecodeEngine` via :class:`~nanovllm.engine.kv_transfer.KVSender`.
    """

    def __init__(self, model: str, kv_host: str, kv_port: int, **kwargs) -> None:
        config, model_runner, ps, tokenizer, scheduler = _build_engine(model, kwargs)
        self.config = config
        self.model_runner = model_runner
        self.ps = ps
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        self.kv_sender = KVSender(kv_host, kv_port)
        self.kv_sender.connect()

        atexit.register(self._exit)

    def _exit(self) -> None:
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
        self.kv_sender.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> None:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self) -> int:
        """Schedule and run one prefill batch.

        For each sequence in the batch:

        1. Run the prefill forward pass and sample the first token.
        2. Extract the KV cache blocks for that sequence from GPU memory.
        3. Send ``(sequence metadata + KV data)`` to the decode engine.
        4. Deallocate the blocks on the prefill side so they can be reused.

        Returns:
            Number of sequences whose KV cache was transferred.
        """
        if not self.scheduler.waiting:
            return 0

        seqs, is_prefill = self.scheduler.schedule()
        assert is_prefill, "PrefillEngine unexpectedly scheduled a decode batch"

        token_ids = self.model_runner.call("run", seqs, True)

        for seq, token_id in zip(seqs, token_ids):
            # Extract KV cache for this sequence's allocated blocks.
            # Shape: [2, num_layers, num_kv_blocks, block_size, kv_heads, head_dim]
            kv_data = self.model_runner.kv_cache[:, :, seq.block_table].cpu()

            # Append the first generated token so the decode side has the
            # complete token_ids = [prompt…, first_decode_token].
            seq.append_token(token_id)

            self.kv_sender.send(
                seq_id=seq.seq_id,
                token_ids=seq.token_ids,
                num_prompt_tokens=seq.num_prompt_tokens,
                temperature=seq.temperature,
                max_tokens=seq.max_tokens,
                ignore_eos=seq.ignore_eos,
                kv_data=kv_data,
            )

            # Release the KV blocks on the prefill side.
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
        """Prefill all *prompts* and transfer KV caches to the decode engine.

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
        return total


# ---------------------------------------------------------------------------
# DecodeEngine
# ---------------------------------------------------------------------------

class DecodeEngine:
    """Decode-only engine.

    A :class:`~nanovllm.engine.kv_transfer.KVReceiver` runs in a background
    thread.  On each :meth:`step` call the engine drains the receiver queue,
    injects the transferred sequences into the running set of the scheduler,
    and executes one decode iteration.
    """

    def __init__(self, model: str, kv_host: str, kv_port: int, **kwargs) -> None:
        config, model_runner, ps, tokenizer, scheduler = _build_engine(model, kwargs)
        self.config = config
        self.model_runner = model_runner
        self.ps = ps
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        self.kv_receiver = KVReceiver(kv_host, kv_port)
        self.kv_receiver.start()

        # seq_id → completion token ids for finished sequences
        self._results: dict[int, list[int]] = {}

        atexit.register(self._exit)

    def _exit(self) -> None:
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
        self.kv_receiver.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_pending(self) -> None:
        """Drain the KV receiver queue and inject sequences into the scheduler."""
        while not self.kv_receiver.queue.empty():
            meta, kv_data = self.kv_receiver.queue.get_nowait()
            self._inject_sequence(meta, kv_data)

    def _inject_sequence(self, meta: KVTransferMeta, kv_data: torch.Tensor) -> None:
        """Reconstruct a :class:`Sequence` from transferred data.

        The KV data (covering the prefill tokens) is loaded into the local GPU
        cache, and the sequence is added to the decode scheduler's running
        queue so that generation can continue immediately.

        Block layout after injection
        ----------------------------
        * ``num_kv_blocks`` blocks are allocated for the prefill KV (positions
          0 .. num_prompt_tokens − 1).
        * The scheduler's ``may_append`` will allocate an extra block if the
          first decode step requires writing to a new block (i.e. when
          ``num_tokens % block_size == 1``).
        """
        num_kv_blocks: int = kv_data.shape[2]

        # Build the Sequence without calling __init__ so that we can assign
        # the prefill-side seq_id directly (bypassing the global counter).
        seq = object.__new__(Sequence)
        seq.seq_id = meta.seq_id
        seq.status = SequenceStatus.RUNNING
        seq.token_ids = list(meta.token_ids)
        seq.last_token = meta.token_ids[-1]
        seq.num_tokens = len(meta.token_ids)
        seq.num_prompt_tokens = meta.num_prompt_tokens
        seq.num_cached_tokens = 0
        seq.block_table = []
        seq.temperature = meta.temperature
        seq.max_tokens = meta.max_tokens
        seq.ignore_eos = meta.ignore_eos

        # Allocate fresh blocks and load the received KV data into the cache.
        self.scheduler.block_manager.allocate_transferred(seq, num_kv_blocks)
        self.model_runner.kv_cache[:, :, seq.block_table] = kv_data.to(
            self.model_runner.kv_cache.device
        )

        self.scheduler.running.append(seq)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> list[tuple[int, list[int]]]:
        """Inject pending transfers, then run one decode iteration.

        Returns:
            List of ``(seq_id, completion_token_ids)`` for sequences that
            finished during this step.
        """
        self._inject_pending()
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
        """Return True while there are running sequences or queued transfers."""
        return bool(self.scheduler.running) or not self.kv_receiver.queue.empty()

    def get_result(self, seq_id: int) -> list[int] | None:
        """Return completion token ids for *seq_id*, or ``None`` if not done."""
        return self._results.get(seq_id)

    def decode_all(self, use_tqdm: bool = True) -> dict[int, str]:
        """Run the decode loop until all pending sequences are finished.

        Returns:
            Mapping ``{seq_id: decoded_text}`` for every finished sequence.
        """
        if use_tqdm:
            pbar = tqdm(desc="Decoding", dynamic_ncols=True)
        decode_throughput = 0.0
        while self.has_pending():
            t = perf_counter()
            finished = self.step()
            if finished and use_tqdm:
                decode_throughput = len(finished) / max(perf_counter() - t, 1e-9)
                pbar.set_postfix({"Decode": f"{int(decode_throughput)}tok/s"})
                pbar.update(len(finished))
        if use_tqdm:
            pbar.close()
        return {
            sid: self.tokenizer.decode(tids)
            for sid, tids in self._results.items()
        }
