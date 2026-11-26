"""Qwen3.5 / Qwen3.6 MoE decode at TP2 (Xe2): one resident kernel per decoder
layer (mixer, MoE with the shared expert, both TP all-reduces inside the kernel
through a P2P mailbox in the peer card's memory), and the MTP drafter's
prologue as its own kernel. The ops are ``torch.ops.sgl_kernel.qwen35_*``; see
src/sycl/models/qwen35/xe20/qwen35_decode.cpp for the descriptor, workspace,
counter and mailbox contracts. The SGLang integration that runs them is
``sgl_kernel.monokernel``."""

from typing import List, Optional, Sequence

import torch

_ops = torch.ops.sgl_kernel

KIND_GDN, KIND_ATTN = 0, 1
MODE_MONO, MODE_SPLIT, MODE_SPLIT_EXTERNAL_AR = 0, 1, 2


def is_available() -> bool:
    """Whether this build has the Qwen3.5 decode kernels (USE_MONO_DECODE, Xe2)."""
    return hasattr(_ops, "qwen35_decode_step")


def is_supported(device: torch.device) -> bool:
    """Whether the kernels run on ``device``: an Xe2 (Battlemage) GPU with at
    least 20 Xe cores. A launch keeps one work-group a Xe core resident: 32 on
    a GPU with 32 or more Xe cores (Arc Pro B70), else one a Xe core (Arc Pro
    B60: 20), each running the parts of several of the 32 (functional, not
    tuned)."""
    if not is_available() or device.type != "xpu":
        return False
    index = device.index if device.index is not None else torch.xpu.current_device()
    return bool(_ops.qwen35_supported(index))


def work_groups(device: torch.device) -> int:
    """The work-groups of a launch on ``device`` (32, or the Xe cores of a
    smaller Xe2 GPU; see force_work_groups)."""
    index = device.index if device.index is not None else torch.xpu.current_device()
    return int(_ops.qwen35_work_groups(index))


def force_work_groups(n: int) -> None:
    """Test hook: launch every kernel on ``n`` work-groups (20 .. 32), e.g. 20
    on an Arc Pro B70 to run the kernels of a 20-core GPU; 0: the device's.
    Process-wide; set it the same on both ranks, between steps."""
    _ops.qwen35_force_work_groups(int(n))


def max_context() -> int:
    """Longest context (cached tokens a request) the attention kernels support."""
    return int(_ops.qwen35_max_context())


def qwen35_sizes() -> dict:
    ws, cnt, health, mbox, desc = _ops.qwen35_sizes()
    return dict(
        workspace_bytes=ws,
        counter_words=cnt,
        health_words=health,
        mailbox_words=mbox,
        desc_bytes=desc,
    )


def qwen35_mailbox_never() -> int:
    """The initial value of every mailbox word (it matches no tag)."""
    return int(_ops.qwen35_mailbox_never())


class Qwen35Workspace:
    """Device state of the decode kernels of one rank: the scratch workspace,
    the counters (one row a kernel), the health words, and P2P mailboxes for
    ``n_mailboxes`` kernels (one each) in torch symmetric memory of
    ``group_name`` (two ranks). Construction is collective; mapping the peer's
    memory needs ptrace permission over the peer process (with
    ``kernel.yama.ptrace_scope`` 1: ``--cap-add SYS_PTRACE``)."""

    def __init__(
        self,
        group_name: str,
        n_mailboxes: int,
        device: torch.device,
        deadline_s: float = 2.0,
    ):
        from torch._C._distributed_c10d import _SymmetricMemory

        s = qwen35_sizes()
        self.device = device
        self.n = n_mailboxes
        self.mailbox_words = s["mailbox_words"]
        words = n_mailboxes * self.mailbox_words
        hw = s["health_words"]
        # the mailboxes, then the health words (the peer's waits flag a failure
        # in them)
        self._buf = _SymmetricMemory.empty_strided_p2p(
            (words * 8 + hw * 4,), [1], torch.uint8, device, group_name
        )
        self._handle = _SymmetricMemory.rendezvous(self._buf, group_name)
        if self._handle.world_size != 2:
            raise ValueError("Qwen35Workspace: the group must have two ranks")
        self.rank = self._handle.rank
        self._own = self._handle.get_buffer(self.rank, (words,), torch.int64, 0)
        self._peer = self._handle.get_buffer(1 - self.rank, (words,), torch.int64, 0)
        self.workspace = torch.zeros(
            s["workspace_bytes"], dtype=torch.uint8, device=device
        )
        self.counters = torch.zeros(
            (n_mailboxes, s["counter_words"]), dtype=torch.int32, device=device
        )
        self.health = self._handle.get_buffer(self.rank, (hw,), torch.int32, words * 2)
        self.peer_health = self._handle.get_buffer(
            1 - self.rank, (hw,), torch.int32, words * 2
        )
        self.health_host = torch.zeros(
            s["health_words"], dtype=torch.int32
        ).pin_memory()
        self._reset(deadline_s)

    def _reset(self, deadline_s: float) -> None:
        self.deadline_s = float(deadline_s)
        self._own.fill_(qwen35_mailbox_never())
        _ops.qwen35_init_counters(
            self.counters,
            self.health,
            self.health_host,
            self.deadline_s,
            self.peer_health,
        )
        # the peer may write into the mailboxes only after they were filled
        torch.xpu.synchronize(self.device)
        self._handle.barrier()

    def set_deadline(self, deadline_s: float) -> None:
        """Restarts the kernels' state (mailboxes, tags, health) with the wait
        deadline ``deadline_s``. Collective, between steps: no kernel of either
        rank may be running (the tags restart, so the old mailbox words are
        cleared too)."""
        torch.xpu.synchronize(self.device)
        self._handle.barrier()
        self._reset(deadline_s)

    def mailbox(self, i: int):
        """(own, peer) int64 views of mailbox i."""
        w = self.mailbox_words
        return self._own[i * w : (i + 1) * w], self._peer[i * w : (i + 1) * w]

    def failed(self) -> bool:
        """A wait gave up (host copy of the health words; no device sync)."""
        return int(self.health_host[0]) != 0


def qwen35_prepare_layer(attn: bool, tensors: Sequence[torch.Tensor]) -> list:
    """The tensor list of qwen35_layer_desc from a layer's SGLang tensors (the
    order is in qwen35_decode.cpp): the scales as fp32 row-major copies where
    needed, in_proj_ba in fp16 appended for a GatedDeltaNet layer. Keep the
    list alive as long as the descriptor."""
    return list(_ops.qwen35_prepare_layer(attn, list(tensors)))


def qwen35_layer_desc(
    attn: bool,
    tensors: Sequence[torch.Tensor],
    mailbox: torch.Tensor,
    peer_mailbox: torch.Tensor,
) -> torch.Tensor:
    """A decoder layer's descriptor (CPU uint8) from qwen35_prepare_layer's list."""
    return _ops.qwen35_layer_desc(attn, list(tensors), mailbox, peer_mailbox)


def qwen35_mtp_desc(
    tensors: Sequence[torch.Tensor],
    mailbox: torch.Tensor,
    peer_mailbox: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """The MTP prologue's descriptor: [fc, pre_fc_norm_embedding, pre_fc_norm_hidden]."""
    return _ops.qwen35_mtp_desc(list(tensors), mailbox, peer_mailbox, rank)


def qwen35_bind_gdn_caches(
    desc: torch.Tensor,
    conv: torch.Tensor,
    ssm: torch.Tensor,
    inter_ssm: Optional[torch.Tensor] = None,
    inter_conv: Optional[torch.Tensor] = None,
) -> None:
    _ops.qwen35_bind_gdn_caches(desc, conv, ssm, inter_ssm, inter_conv)


def qwen35_bind_attn_caches(
    desc: torch.Tensor, k: torch.Tensor, v: torch.Tensor, page_size: int
) -> None:
    _ops.qwen35_bind_attn_caches(desc, k, v, page_size)


def qwen35_decode_step(
    ws: Qwen35Workspace,
    layers: torch.Tensor,
    l0: int,
    l1: int,
    hidden: torch.Tensor,
    residual: torch.Tensor,
    hid_out: torch.Tensor,
    res_out: torch.Tensor,
    positions: torch.Tensor,
    tokens: int = 1,
    state_idx: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    out_loc: Optional[torch.Tensor] = None,
    counters: Optional[torch.Tensor] = None,
    mode: int = MODE_MONO,
    i0: int = 0,
    i1: int = 0,
    ar: Optional[torch.Tensor] = None,
) -> bool:
    """One decode (tokens 1) or verify (tokens >= 2 a request) step of layers
    [l0, l1) of ``layers`` (CPU uint8 [L, desc_bytes]); counters: the rows of
    the table's kernels (default ``ws.counters``). mode 0: one kernel a layer;
    1: one kernel a segment of a layer (in-kernel all-reduces); 2: the plan
    items [i0, i1) of one kernel a segment with the all-reduces left to the
    caller on ``ar`` ([>= rows, 2048] fp32, summed over the ranks in place)
    between the ranges qwen35_split_ar_slots() gives. Modes 1 and 2 are bitwise
    references of mode 0. False when the shape has no kernel (nothing
    launched)."""
    return bool(
        _ops.qwen35_decode_step(
            mode,
            i0,
            i1,
            ar,
            layers,
            l0,
            l1,
            hidden,
            residual,
            hid_out,
            res_out,
            positions,
            ws.workspace,
            ws.counters if counters is None else counters,
            tokens,
            state_idx,
            page_table,
            seq_lens,
            out_loc,
        )
    )


def qwen35_split_ar_slots(layers: torch.Tensor, l0: int, l1: int) -> List[int]:
    """The plan items of a mode-2 step of layers [l0, l1) where the caller
    all-reduces ``ar``, then the plan length."""
    return list(_ops.qwen35_split_ar_slots(layers, l0, l1))


def qwen35_mtp_prologue(
    ws: Qwen35Workspace,
    desc: torch.Tensor,
    embed: torch.Tensor,
    hidden: torch.Tensor,
    out: torch.Tensor,
    counter_row: int,
) -> bool:
    """The MTP prologue (kernel ``counter_row`` of ws): out = fc([norm(embed);
    norm(hidden)]). False for a row count without a kernel."""
    return bool(
        _ops.qwen35_mtp_prologue(
            desc, embed, hidden, out, ws.workspace, ws.counters[counter_row]
        )
    )


def qwen35_fault_inject(
    ws: Qwen35Workspace, row: int, delta: int = 7, arrivals: bool = False
) -> None:
    """Test hook: shifts kernel ``row``'s mailbox tags on this rank (the peer's
    waits and then this rank's give up), or with ``arrivals`` sets its stage
    arrival counters back (its waits for this card's other work-groups give
    up)."""
    _ops.qwen35_fault_inject(ws.counters[row], delta, arrivals)


def stack_descs(descs: List[torch.Tensor]) -> torch.Tensor:
    """The layer table (CPU uint8 [L, desc_bytes]) of the descriptors."""
    return torch.stack(list(descs))
