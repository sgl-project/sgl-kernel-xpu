# SPDX-License-Identifier: Apache-2.0
"""Caller-owned state for the MiniCPM5 Xe2 decode kernel."""

from collections.abc import Callable, Sequence
from enum import IntEnum

import torch
from torch.distributed import _symmetric_memory as symm_mem

_HIDDEN, _HEAD_DIM, _ROWS, _EXPERTS, _TOPK = 2048, 128, 8, 160, 16
_CONTROL_WORDS = 16 * 16 + 16 + _ROWS * 4
_DEADLINE_MILLIS_WORD = 16 * 16 + 6
_TP2_CONTROL_BANKS = 1 + 3 * _ROWS
_MAILBOX_WORDS = 2 * (_ROWS * (_ROWS + 1) // 2) * _HIDDEN
_MAILBOX_NEVER = 0x00000000FFFF0000


class _Part(IntEnum):
    FULL = 0
    ATTENTION = 1
    FEED_FORWARD = 2
    ATTENTION_PARTIAL = 3
    FEED_FORWARD_PARTIAL = 4


def is_available() -> bool:
    return hasattr(torch.ops.sgl_kernel, "minicpm5_decode_step")


def work_groups(device: torch.device) -> int:
    if device.type != "xpu" or not is_available():
        return 0
    index = torch.xpu.current_device() if device.index is None else device.index
    return torch.ops.sgl_kernel.minicpm5_work_groups(index)


class LayerTable:
    """Own the tensors referenced by the CPU uint8 layer descriptors."""

    def __init__(
        self,
        weights: Sequence[Sequence[torch.Tensor]],
        dense: Sequence[bool],
        tp: int,
        *,
        draft: bool = False,
    ):
        if not weights or len(weights) != len(dense):
            raise ValueError("Expected a nonempty MiniCPM5 layer list")
        self.weights = tuple(tuple(layer) for layer in weights)
        self.descriptors = torch.stack(
            [
                (
                    torch.ops.sgl_kernel.minicpm5_draft_layer_desc(layer, tp)
                    if draft
                    else torch.ops.sgl_kernel.minicpm5_layer_desc(layer, tp, is_dense)
                )
                for layer, is_dense in zip(weights, dense)
            ]
        )


class Workspace:
    """State shared by launches on a single in-order XPU stream."""

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        layers: int,
        tp: int,
        group=None,
        groups: int | None = None,
        deadline_s: float = 2.0,
    ):
        supported = work_groups(device)
        self.groups = supported if groups is None else groups
        if not supported or self.groups not in (16, supported):
            raise ValueError(
                f"Unsupported MiniCPM5 launch on {device}: {self.groups} work-groups"
            )
        if tp not in (1, 2) or (tp == 1) != (group is None) or layers <= 0:
            raise ValueError("MiniCPM5 mono requires TP1 or a TP2 process group")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("MiniCPM5 mono requires FP16 or BF16 activations")
        self.tp, self.device, self.dtype = tp, device, dtype
        q_heads, inter = 32 // tp, 512 // tp
        q_size = q_heads * _HEAD_DIM
        shapes = (
            ((_ROWS, _HIDDEN), dtype),
            ((_ROWS, 2 * (q_size + 2 // tp * _HEAD_DIM)), dtype),
            ((_ROWS, q_size), dtype),
            ((_ROWS, 32 * tp, q_heads, _HEAD_DIM + 2), torch.float32),
            ((_ROWS, (_TOPK + 1) * inter), dtype),
            ((_ROWS, _EXPERTS), torch.float32),
            ((_ROWS, _TOPK), torch.float32),
            ((_ROWS, _TOPK), torch.int32),
            ((_ROWS, _TOPK, 2 * inter, 8), torch.float32),
            ((_ROWS, _TOPK + 1, _HIDDEN), dtype),
        )
        self.scratch = [
            torch.empty(shape, device=device, dtype=kind) for shape, kind in shapes
        ]
        self.counters = torch.zeros(
            (layers, _TP2_CONTROL_BANKS if tp == 2 else 1, _CONTROL_WORDS),
            device=device,
            dtype=torch.int32,
        )
        if not 0.0 < deadline_s <= 2.0:
            raise ValueError("MiniCPM5 decode timeout must be in (0, 2] seconds")
        self.counters[..., _DEADLINE_MILLIS_WORD].fill_(
            max(1, round(deadline_s * 1000))
        )
        self.health = torch.zeros((1,), device=device, dtype=torch.int32)
        self.host_health = torch.zeros((1,), dtype=torch.int32, pin_memory=True)
        self.local_mailbox = self.peer_mailbox = self.handle = None
        if tp == 2:
            if torch.distributed.get_world_size(group) != 2:
                raise ValueError(
                    "MiniCPM5 mono requires a two-rank symmetric memory group"
                )
            rank = torch.distributed.get_rank(group)
            shape = (layers, _MAILBOX_WORDS)
            self.local_mailbox = symm_mem.empty(
                *shape, dtype=torch.int64, device=device
            )
            self.local_mailbox.fill_(_MAILBOX_NEVER)
            torch.xpu.synchronize(device)
            self.handle = symm_mem.rendezvous(self.local_mailbox, group)
            self.peer_mailbox = self.handle.get_buffer(1 - rank, shape, torch.int64)

    def check_health(self) -> None:
        if self.host_health[0].item():
            raise RuntimeError(
                "MiniCPM5 mono decode failed a device hand-off or KV metadata check"
            )

    def decode_step(
        self,
        table: LayerTable,
        l0: int,
        l1: int,
        *,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        slots: torch.Tensor,
        blocks: torch.Tensor,
        lengths: torch.Tensor,
        rope: torch.Tensor,
        keys: Sequence[torch.Tensor],
        values: Sequence[torch.Tensor],
        mode: int = 0,
        all_reduce: Callable[[torch.Tensor], None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run [l0, l1) with fused layers, split layers or external FP32 reduction.

        With mode 1, attention and feed-forward launch separately. With mode 2,
        the caller's in-place all_reduce operates on each TP2 FP32 partial.
        Each TP2 row width and launch part has its own mailbox epoch.
        """
        if mode not in (0, 1, 2) or not 0 <= l0 < l1 <= len(table.weights):
            raise ValueError("Invalid MiniCPM5 decode mode or layer interval")
        if mode == 2 and self.tp == 2 and all_reduce is None:
            raise ValueError("TP2 split mode 2 requires an in-place FP32 all-reduce")
        buffers = hidden.new_empty((2, *hidden.shape))
        residuals = torch.empty_like(buffers)

        def launch(first, last, hidden, residual, output, res_out, part, ar=None):
            torch.ops.sgl_kernel.minicpm5_decode_step(
                self.groups,
                self.tp,
                first,
                last,
                table.descriptors,
                hidden,
                residual,
                positions,
                slots,
                blocks,
                lengths,
                rope,
                keys,
                values,
                self.scratch,
                self.counters,
                self.health,
                self.host_health,
                self.local_mailbox,
                self.peer_mailbox,
                output,
                res_out,
                part,
                ar,
            )

        if mode == 0:
            launch(l0, l1, hidden, residual, buffers, residuals, _Part.FULL)
            index = (l1 - l0 - 1) % 2
            return buffers[index], residuals[index]

        # A separate pair avoids aliasing the saved residual of the preceding
        # layer with the output of the next attention kernel.
        next_buffers = torch.empty_like(buffers)
        next_residuals = torch.empty_like(residuals)
        use_external = mode == 2 and self.tp == 2
        ar = (
            hidden.new_empty(hidden.shape, dtype=torch.float32)
            if use_external
            else None
        )
        for layer in range(l0, l1):
            if use_external:
                launch(
                    layer,
                    layer + 1,
                    hidden,
                    residual,
                    buffers,
                    residuals,
                    _Part.ATTENTION_PARTIAL,
                    ar,
                )
                all_reduce(ar)
                summed = (
                    hidden
                    if residual is None
                    else (hidden.float() + residual.float()).to(hidden.dtype)
                )
                residuals[0].copy_(summed.float() + ar.to(hidden.dtype).float())
                launch(
                    layer,
                    layer + 1,
                    hidden,
                    residual,
                    buffers,
                    residuals,
                    _Part.FEED_FORWARD_PARTIAL,
                    ar,
                )
                all_reduce(ar)
                buffers[0].copy_(ar)
            else:
                launch(
                    layer,
                    layer + 1,
                    hidden,
                    residual,
                    buffers,
                    residuals,
                    _Part.ATTENTION,
                )
                launch(
                    layer,
                    layer + 1,
                    hidden,
                    residual,
                    buffers,
                    residuals,
                    _Part.FEED_FORWARD,
                )
            hidden, residual = buffers[0], residuals[0]
            buffers, next_buffers = next_buffers, buffers
            residuals, next_residuals = next_residuals, residuals
        return hidden, residual
