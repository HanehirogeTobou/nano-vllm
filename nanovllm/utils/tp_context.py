"""
Tensor-parallel group context.

In normal (non-PD) mode the TP group is ``None``, which makes every
``dist.*`` call fall back to the default process group – exactly the
behaviour the codebase has always had.

In PD-separation mode the model runner calls :func:`set_tp_group` with
the appropriate subgroup (prefill or decode) before constructing the
model, so all collective operations in the model layers transparently use
the correct subset of ranks.

Each worker process is created via ``torch.multiprocessing.spawn`` (with
``start_method="spawn"``), so the global ``_TP_GROUP`` variable is
**per-process** and not shared between workers.  No locking is required.
"""
import torch.distributed as dist


_TP_GROUP = None


def set_tp_group(group) -> None:
    global _TP_GROUP
    _TP_GROUP = group


def get_tp_group():
    """Return the current TP process group (or ``None`` for the default group)."""
    return _TP_GROUP


def get_tp_rank() -> int:
    return dist.get_rank(_TP_GROUP)


def get_tp_size() -> int:
    return dist.get_world_size(_TP_GROUP)
