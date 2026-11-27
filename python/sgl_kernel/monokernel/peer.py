"""The TP agreement before the P2P mailboxes of the in-kernel TP all-reduces
are mapped (the mailboxes are the workspace's, ``Qwen35Workspace``: one buffer
in torch symmetric memory of the TP group, one mailbox a kernel).

Mapping the peer's buffer (``_SymmetricMemory.rendezvous``) takes the peer's
memory handle with ``pidfd_getfd``, which needs (1) the syscall: Docker's
default seccomp profile allows it only with ``--cap-add SYS_PTRACE``, and (2)
ptrace permission over the peer under Yama: ``kernel.yama.ptrace_scope`` 0 or
1 (at 1 torch declares the peer as its ptracer), or 2 with CAP_SYS_PTRACE."""

import ctypes
import errno
import os

import torch
import torch.distributed as dist

_CAP_SYS_PTRACE = 19
_SYS_PIDFD_OPEN, _SYS_PIDFD_GETFD = 434, 438  # the same on every architecture


def _pidfd_getfd_blocked() -> str:
    """Why pidfd_getfd fails on this process's own descriptor (seccomp, an old
    kernel), or "" when it works. Yama does not restrict a process on itself."""
    libc = ctypes.CDLL(None, use_errno=True)
    pidfd = libc.syscall(_SYS_PIDFD_OPEN, os.getpid(), 0)
    if pidfd < 0:
        e = ctypes.get_errno()
        return f"pidfd_open fails ({errno.errorcode.get(e, e)})"
    try:
        fd = libc.syscall(_SYS_PIDFD_GETFD, pidfd, pidfd, 0)
        if fd < 0:
            e = ctypes.get_errno()
            return (
                f"pidfd_getfd fails ({errno.errorcode.get(e, e)}): a seccomp filter blocks it "
                "(Docker's default profile allows it with --cap-add SYS_PTRACE)"
            )
        os.close(fd)
        return ""
    finally:
        os.close(pidfd)


def tp_agree(ok: bool, group) -> bool:
    """True when every rank of ``group`` passed ``ok`` true. Every rank must call
    it at the same point; ``group`` is a CPU (gloo) process group."""
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def ptrace_problem() -> str:
    """Why this process cannot map a peer's symmetric memory, or "" when it
    can (as far as seccomp, Yama and the capabilities tell)."""
    blocked = _pidfd_getfd_blocked()
    if blocked:
        return blocked
    try:
        with open("/proc/sys/kernel/yama/ptrace_scope") as f:
            scope = int(f.read().strip())
    except (OSError, ValueError):
        return ""  # no Yama: plain uid checks, same user
    if scope <= 1:  # at 1 torch declares the peer as its ptracer
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
        "kernel.yama.ptrace_scope is 2 and the process lacks CAP_SYS_PTRACE "
        "(run the container with --cap-add SYS_PTRACE, or set ptrace_scope to 1 or 0)"
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
