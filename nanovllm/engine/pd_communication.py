"""
KV-cache communication utilities for PD (Prefill-Decode) separation.

Design
------
All ranks (prefill + decode) belong to a single global NCCL process group.
Two *subgroups* are created for TP all-reduces within each stage:

  * prefill_pg  – global ranks 0 … P-1
  * decode_pg   – global ranks P … P+D-1

KV-cache transfer between the two stages uses point-to-point
``dist.send`` / ``dist.recv`` on the **global** group (group=None →
default group).  Each prefill rank ``i`` corresponds to decode rank
``P + i`` (same local shard index, so P must equal D).

Only the blocks that were written during the current prefill batch need
to be shipped – usually far fewer than the total KV-cache size.
"""
import torch
import torch.distributed as dist


def gather_kv_blocks(
    kv_cache: torch.Tensor,
    block_ids: list[int],
) -> torch.Tensor:
    """Copy the requested KV-cache blocks into a contiguous send buffer.

    Parameters
    ----------
    kv_cache:
        Shape ``[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]``
        on the current CUDA device.
    block_ids:
        Physical block indices to gather (output of ``seq.block_table``).

    Returns
    -------
    Contiguous CUDA tensor of shape
    ``[2, num_layers, len(block_ids), block_size, num_kv_heads, head_dim]``.
    """
    if not block_ids:
        return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)
    # Index along the blocks dimension (dim=2)
    idx = torch.tensor(block_ids, dtype=torch.long, device=kv_cache.device)
    return kv_cache[:, :, idx, :, :, :].contiguous()


def scatter_kv_blocks(
    kv_cache: torch.Tensor,
    block_ids: list[int],
    buf: torch.Tensor,
) -> None:
    """Write a received KV buffer back into the kv_cache at the given block IDs.

    Parameters
    ----------
    kv_cache:
        Same layout as in :func:`gather_kv_blocks`.
    block_ids:
        Physical block indices that correspond to ``buf``'s block axis.
    buf:
        Contiguous CUDA tensor of shape
        ``[2, num_layers, len(block_ids), block_size, num_kv_heads, head_dim]``.
    """
    if not block_ids:
        return
    idx = torch.tensor(block_ids, dtype=torch.long, device=kv_cache.device)
    kv_cache[:, :, idx, :, :, :] = buf


def send_kv_blocks(
    kv_cache: torch.Tensor,
    block_ids: list[int],
    dst_global_rank: int,
) -> None:
    """Send the KV cache for *block_ids* to *dst_global_rank* via NCCL.

    Called by a prefill rank; the matching :func:`recv_kv_blocks` call
    must happen on the corresponding decode rank at the same time.
    """
    buf = gather_kv_blocks(kv_cache, block_ids)
    # Send the number of blocks first so the receiver can allocate
    meta = torch.tensor([len(block_ids)], dtype=torch.int64, device=kv_cache.device)
    dist.send(meta, dst=dst_global_rank)
    if len(block_ids):
        dist.send(buf, dst=dst_global_rank)


def recv_kv_blocks(
    kv_cache: torch.Tensor,
    block_ids: list[int],
    src_global_rank: int,
) -> None:
    """Receive KV-cache blocks from *src_global_rank* and write into kv_cache.

    Called by a decode rank; must be paired with :func:`send_kv_blocks`
    on the corresponding prefill rank.
    """
    meta = torch.empty(1, dtype=torch.int64, device=kv_cache.device)
    dist.recv(meta, src=src_global_rank)
    num_blocks = meta.item()
    assert num_blocks == len(block_ids), (
        f"KV transfer metadata mismatch: expected {len(block_ids)} blocks, "
        f"received {num_blocks}"
    )
    if num_blocks:
        _, num_layers, _, block_size, num_kv_heads, head_dim = kv_cache.shape
        buf = torch.empty(
            2, num_layers, num_blocks, block_size, num_kv_heads, head_dim,
            dtype=kv_cache.dtype, device=kv_cache.device,
        )
        dist.recv(buf, src=src_global_rank)
        scatter_kv_blocks(kv_cache, block_ids, buf)
