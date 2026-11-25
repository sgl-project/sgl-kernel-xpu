"""Test-side helpers of the mono decode kernel tests: the fixtures of
mono_decode_fixtures (random layers, the torch reference, the caller-owned
state of the qwen35 ops) and the pytest skip markers."""

import pytest
import torch
from mono_decode_fixtures import *  # noqa: F401,F403
from mono_decode_fixtures import _gemma_rms, _ops, _q35  # noqa: F401


def mono_available(device: int = 0) -> bool:
    try:
        q35 = _q35()
    except Exception:
        return False
    return (
        q35.is_available()
        and torch.xpu.is_available()
        and torch.xpu.device_count() > device
        and q35.is_supported(torch.device("xpu", device))
    )


requires_mono = pytest.mark.skipif(
    not mono_available(),
    reason="needs the qwen35 decode kernels and an Xe2 GPU with 32 Xe cores",
)
requires_two_cards = pytest.mark.skipif(
    not (mono_available(0) and mono_available(1)),
    reason="needs two Xe2 GPUs with 32 Xe cores",
)
