"""The TP agreement before the P2P mailboxes of the in-kernel TP all-reduces
are mapped (the mailboxes are the workspace's, ``Qwen35Workspace``: one buffer
in torch symmetric memory of the TP group, one mailbox a kernel).

Mapping the peer's buffer (``_SymmetricMemory.rendezvous``) imports the peer's
IPC handle, which needs ptrace permission over the peer
process: with ``kernel.yama.ptrace_scope`` 1 a container needs
``--cap-add SYS_PTRACE`` (or the host ``ptrace_scope`` 0)."""

import torch
import torch.distributed as dist

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


def require_ptrace(group) -> None:
    """Raises on every rank of ``group`` (a GroupCoordinator: its CPU group
    agrees) when a rank cannot map the peer's symmetric memory, so none waits
    in the rendezvous. Collective."""
    problem = ptrace_problem()
    if not tp_agree(not problem, group.cpu_group):
        raise RuntimeError(
            "XPU monokernel: mapping the peer's mailboxes needs ptrace "
            f"permission: {problem or 'the peer rank lacks it'}"
        )
