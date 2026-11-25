"""The P2P mailboxes of the in-kernel TP all-reduces: one buffer in torch
symmetric memory of the TP group, holding one mailbox per mono kernel, mapped
on both ranks; and the TP agreement that precedes it.

Mapping the peer's buffer (``_SymmetricMemory.rendezvous``) imports the peer's
IPC handle, which needs ptrace permission over the peer
process: with ``kernel.yama.ptrace_scope`` 1 a container needs
``--cap-add SYS_PTRACE`` (or the host ``ptrace_scope`` 0)."""

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_CAP_SYS_PTRACE = 19


def tp_agree(ok: bool, group) -> bool:
    """True when every rank of ``group`` passed ``ok`` true. Every rank must call
    it at the same point; ``group`` is a CPU (gloo) process group."""
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def ptrace_problem() -> str:
    """Why this process cannot map a peer's symmetric memory, or "" when it
    can (as far as Yama and the capabilities tell)."""
    try:
        with open("/proc/sys/kernel/yama/ptrace_scope") as f:
            scope = int(f.read().strip())
    except (OSError, ValueError):
        return ""  # no Yama: plain uid checks, same user
    if scope == 0:
        return ""
    if scope >= 3:
        return "kernel.yama.ptrace_scope is 3 (ptrace disabled)"
    cap = 0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("CapEff:"):
                cap = int(line.split()[1], 16)
    if cap & (1 << _CAP_SYS_PTRACE):
        return ""
    return (
        f"kernel.yama.ptrace_scope is {scope} and the process lacks CAP_SYS_PTRACE "
        "(run the container with --cap-add SYS_PTRACE, or set ptrace_scope to 0)"
    )


def _addr(t: torch.Tensor) -> int:
    """Device address of t as the ops' int (int64: USM addresses can be above
    2**63)."""
    p = t.data_ptr()
    return p - (1 << 64) if p >= 1 << 63 else p


class PeerRegion:
    """This rank's mailboxes and the peer's, mapped, one mailbox per name, in
    torch symmetric memory of ``group`` (a GroupCoordinator of two ranks: its
    device group names the symmetric memory, its CPU group agrees on errors).
    The mailboxes start zeroed. The constructor is collective."""

    def __init__(self, names: list, rank: int, group):
        from torch._C._distributed_c10d import _SymmetricMemory

        problem = ptrace_problem()
        if not tp_agree(not problem, group.cpu_group):
            # every rank leaves here together: none waits in the rendezvous
            raise RuntimeError(
                "XPU monokernel: mapping the peer's mailboxes needs ptrace "
                f"permission: {problem or 'the peer rank lacks it'}"
            )
        self.rank = rank
        self.mbox_bytes = int(torch.ops.sgl_kernel.mono_mbox_bytes())
        self.slots = {name: i for i, name in enumerate(names)}
        name = group.device_group.group_name
        dev = torch.device("xpu", torch.xpu.current_device())
        words = len(names) * self.mbox_bytes // 8
        self._buf = _SymmetricMemory.empty_strided_p2p(
            (words * 8,), [1], torch.uint8, dev, name
        )
        self._handle = _SymmetricMemory.rendezvous(self._buf, name)
        if self._handle.world_size != 2:
            raise RuntimeError("XPU monokernel: the TP group must have two ranks")
        me = self._handle.rank
        self._own = self._handle.get_buffer(me, (words,), torch.int64, 0)
        self._peer = self._handle.get_buffer(1 - me, (words,), torch.int64, 0)
        self._own.zero_()
        # the peer may write into the mailboxes only after they were zeroed
        torch.xpu.synchronize(dev)
        self._handle.barrier()
        self.own = _addr(self._own)
        self.peer = _addr(self._peer)
        logger.info(
            "XPU monokernel: P2P mailboxes for %d kernels in symmetric memory",
            len(names),
        )

    def mailbox(self, name: str) -> tuple:
        """(own, peer) addresses of the mailbox of ``name``."""
        off = self.slots[name] * self.mbox_bytes
        return self.own + off, self.peer + off
