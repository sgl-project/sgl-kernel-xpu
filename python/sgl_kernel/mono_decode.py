"""Python side of the mono decode layer kernels (Xe2): one resident kernel per
Qwen3.5 / Qwen3.6 MoE decoder layer at TP2, the TP all-reduces done inside the
kernel through a P2P mailbox in the peer card's memory. The ops are
``torch.ops.sgl_kernel.mono_*``; the SGLang integration that runs them is
``sgl_kernel.monokernel``."""

from typing import Optional

import torch


def is_available() -> bool:
    """Whether this build has the mono decode kernels (USE_MONO_DECODE, Xe2)."""
    return hasattr(torch.ops.sgl_kernel, "mono_decode_layer")


def supports(device: torch.device) -> bool:
    """Whether the kernels run on ``device``: an Xe2 (Battlemage) GPU with at
    least 32 Xe cores (the 32 work-groups of a layer kernel stay resident)."""
    if not is_available() or device.type != "xpu":
        return False
    index = device.index if device.index is not None else torch.xpu.current_device()
    return bool(torch.ops.sgl_kernel.mono_decode_supported(index))


def max_context() -> int:
    """Longest context (cached tokens a request) the attention kernels support."""
    return int(torch.ops.sgl_kernel.mono_max_context())


def health(device: Optional[int] = None) -> torch.Tensor:
    """The device's health words (int32 [16], device memory): [0] FAILED (sticky:
    a kernel gave up an in-kernel wait for the peer, its outputs are NaN), [1]
    the waits that gave up, [2] the peer reported a failure, and of the first
    wait that gave up [3] its tag, [4] / [5] the word it last read, [6] the
    word's offset in the mailbox region, [7] the low half of the address of the
    layer's counters."""
    return torch.ops.sgl_kernel.mono_health(-1 if device is None else device)


def health_host(device: Optional[int] = None) -> torch.Tensor:
    """A CPU view of the host copy of health words 0 .. 7 (host memory): the
    wait that marks the device failed writes it, [0] last, so a host thread can
    poll [0] without touching the GPU."""
    return torch.ops.sgl_kernel.mono_health_host(-1 if device is None else device)


def set_deadline(seconds: float) -> None:
    """Deadline of every in-kernel wait for the peer card (default 5 s)."""
    torch.ops.sgl_kernel.mono_set_deadline(float(seconds))
