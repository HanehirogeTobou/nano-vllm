"""
PDScheduler — scheduler for PD (Prefill-Decode) separation.

Sequence lifecycle in PD mode
------------------------------
  WAITING  →  (prefill batch)  →  TRANSFERRING  →  (KV transfer done)  →  RUNNING  →  FINISHED

Queues
------
waiting       : sequences waiting for prefill allocation
running       : sequences currently being decoded on the decode GPUs
transferring  : sequences whose prefill is done, KV is being (or has
                just been) sent to the decode group; they will move to
                *running* after :meth:`complete_transfer` is called.
"""
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class PDScheduler:

    def __init__(self, config: Config):
        Sequence.block_size = config.kvcache_block_size
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.transferring: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.transferring and not self.running

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    # ------------------------------------------------------------------
    # Prefill scheduling
    # ------------------------------------------------------------------

    def schedule_prefill(self) -> list[Sequence]:
        """Schedule a batch of sequences for prefill.

        Returns the list of sequences to prefill (may be empty).
        """
        scheduled: list[Sequence] = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if (num_batched_tokens + len(seq) > self.max_num_batched_tokens
                    or not self.block_manager.can_allocate(seq)):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            scheduled.append(seq)
        return scheduled

    # ------------------------------------------------------------------
    # Decode scheduling
    # ------------------------------------------------------------------

    def schedule_decode(self) -> list[Sequence]:
        """Schedule a batch of sequences for decode.

        Returns the list of sequences to decode (may be empty if all
        sequences are still in the prefill or transferring stage).
        Sequences that cannot append a new KV block (full KV cache) cause
        another running sequence to be preempted back to *waiting*, just
        like the standard scheduler.
        """
        scheduled: list[Sequence] = []
        num_seqs = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # KV cache is full: preempt the most-recently-added
                # running sequence (or self if the queue is now empty).
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled.append(seq)
        if scheduled:
            self.running.extendleft(reversed(scheduled))
        return scheduled

    def preempt(self, seq: Sequence) -> None:
        """Move a sequence from decode-running back to waiting for re-prefill."""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def postprocess_prefill(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        """Called after the prefill group has processed a batch.

        Appends the first decode token and moves sequences to *transferring*.
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            self.transferring.append(seq)

    def complete_transfer(self, seqs: list[Sequence]) -> None:
        """Move sequences from *transferring* to *running* after KV transfer."""
        transfer_ids = {id(s) for s in seqs}
        self.transferring = deque(s for s in self.transferring if id(s) not in transfer_ids)
        self.running.extend(seqs)

    def postprocess_decode(self, seqs: list[Sequence], token_ids: list[int]) -> list[Sequence]:
        """Called after the decode group has processed a batch.

        Appends the new token, marks finished sequences, and removes them
        from *running*.  Returns the list of finished sequences.
        """
        finished: list[Sequence] = []
        finished_ids: set[int] = set()
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                finished.append(seq)
                finished_ids.add(id(seq))
        if finished_ids:
            self.running = deque(s for s in self.running if id(s) not in finished_ids)
        return finished
