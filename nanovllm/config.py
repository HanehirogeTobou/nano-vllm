import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # PD separation
    enable_pd_separation: bool = False
    num_prefill_ranks: int = 0
    num_decode_ranks: int = 0
    prefill_decode_communication_mode: str = "nccl"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.enable_pd_separation:
            assert self.num_prefill_ranks > 0 and self.num_decode_ranks > 0, \
                "num_prefill_ranks and num_decode_ranks must be > 0 when enable_pd_separation=True"
            assert self.num_prefill_ranks == self.num_decode_ranks, \
                "num_prefill_ranks must equal num_decode_ranks (equal TP size per group required)"
            assert self.num_prefill_ranks + self.num_decode_ranks == self.tensor_parallel_size, \
                "num_prefill_ranks + num_decode_ranks must equal tensor_parallel_size"
            assert self.prefill_decode_communication_mode in ("nccl",), \
                "prefill_decode_communication_mode must be 'nccl'"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
