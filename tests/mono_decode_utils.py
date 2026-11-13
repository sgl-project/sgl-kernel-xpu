"""Test-side helpers of the mono decode kernel tests: the fixtures of
mono_decode_fixtures (random layers, the torch reference, mailboxes) and the
pytest skip markers."""

import pytest
import torch
from mono_decode_fixtures import *  # noqa: F401,F403
from mono_decode_fixtures import _gemma_rms, _ops  # noqa: F401


def mono_available(device: int = 0) -> bool:
    try:
        ops = _ops()
    except Exception:
        return False
    return (
        hasattr(ops, "mono_decode_layer")
        and torch.xpu.is_available()
        and torch.xpu.device_count() > device
        and bool(ops.mono_decode_supported(device))
    )


requires_mono = pytest.mark.skipif(
    not mono_available(),
    reason="needs the mono decode kernels and an Xe2 GPU with 32 Xe cores",
)
requires_two_cards = pytest.mark.skipif(
    not (mono_available(0) and mono_available(1)),
    reason="needs two Xe2 GPUs with 32 Xe cores",
)
