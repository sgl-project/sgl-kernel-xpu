"""The P2P mailboxes of the in-kernel TP all-reduces: one device region per
rank, exported with Level Zero IPC and imported by the peer, holding one mailbox
per mono kernel; and the TP agreement that precedes it."""

import logging
import os
import socket
import tempfile

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# seconds a rank waits for the peer's side of the fd exchange (both ranks are
# at that point together; a rank that died before it must not hang the other)
HANDSHAKE_TIMEOUT_S = 120.0


def tp_agree(ok: bool, group) -> bool:
    """True when every rank of ``group`` passed ``ok`` true. Every rank must call
    it at the same point; ``group`` is a CPU (gloo) process group."""
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def _exchange_fd(
    fd: int, rank: int, group, timeout: float = HANDSHAKE_TIMEOUT_S
) -> int:
    """Swaps the dma-buf fds of the two ranks over a unix socket (SCM_RIGHTS):
    importing the peer's IPC handle needs no ptrace permission this way. Every
    wait is bounded by ``timeout`` seconds; the sockets and the socket file are
    released on every path."""
    paths: list = [None] * dist.get_world_size(group)
    path = srv = None
    try:
        if rank == 0:
            path = os.path.join(
                tempfile.gettempdir(), f"sgl_xpu_mono_{os.getpid()}.sock"
            )
            if os.path.exists(path):
                os.unlink(path)
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.settimeout(timeout)
            srv.bind(path)
            srv.listen(1)
        dist.all_gather_object(paths, path, group=group)
        if srv is not None:
            conn, _ = srv.accept()
        else:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.settimeout(timeout)
            conn.connect(paths[0])
        with conn:
            conn.settimeout(timeout)
            socket.send_fds(conn, [b"f"], [fd])
            _, fds, _, _ = socket.recv_fds(conn, 1, 1)
        return fds[0]
    finally:
        if srv is not None:
            srv.close()
        if path is not None and os.path.exists(path):
            os.unlink(path)


class PeerRegion:
    """This rank's region and the peer's, mapped, one mailbox per name. The
    constructor is collective over the TP CPU group (the handle and fd
    exchange)."""

    def __init__(self, names: list, rank: int, group):
        ops = torch.ops.sgl_kernel
        self.rank = rank
        self.mbox_bytes = int(ops.mono_mbox_bytes())
        self.slots = {name: i for i, name in enumerate(names)}
        error = None
        try:
            self.handle = ops.mono_ipc_create(len(names) * self.mbox_bytes)
            info = ops.mono_ipc_export(self.handle)
        except Exception as e:  # noqa: BLE001
            info, error = None, e
        infos: list = [None] * dist.get_world_size(group)
        dist.all_gather_object(infos, info, group=group)
        if any(i is None for i in infos):
            # every rank leaves here together: no rank waits in the fd exchange
            # for a peer that failed to create its region
            if error is not None:
                raise error
            raise RuntimeError("the peer rank failed to create its mailbox region")
        peer_fd = _exchange_fd(int(info[1]), rank, group)
        mode = ops.mono_ipc_open(self.handle, infos[1 - rank], peer_fd)
        self.own = int(ops.mono_ipc_ptr(self.handle, False))
        self.peer = int(ops.mono_ipc_ptr(self.handle, True))
        logger.info(
            "XPU monokernel: P2P mailboxes for %d kernels (IPC import mode %d)",
            len(names),
            mode,
        )

    def mailbox(self, name: str) -> tuple:
        """(own, peer) addresses of the mailbox of ``name``."""
        off = self.slots[name] * self.mbox_bytes
        return self.own + off, self.peer + off
