"""The single XPU-architecture probe for the test suite.

Every arch decision in tests/benchmarks flows through here — `current_arch()` is
the ONLY place that touches the live device. Tests declare support with the
`@pytest.mark.arch(...)` marker (gated in ``conftest.py``); standalone scripts
use the imperative twin ``require_arch(...)``.
"""

import functools
from enum import Enum

import torch


class Arch(str, Enum):
    """XPU arch buckets. ``str`` mixin ⇒ members compare equal to their tag
    string, so ``Arch.XE20 == "xe20"`` and marker args can be plain strings."""

    XE20 = "xe20"  # BMG  (compute capability major 2)
    XE35 = "xe35"  # CRI  (compute capability 3.5)


#: All arch tags the harness understands (used to catch typos in markers).
KNOWN_ARCHS = frozenset(a.value for a in Arch)


@functools.lru_cache(maxsize=1)
def current_arch():
    """Return the live XPU arch tag (``"xe20"`` / ``"xe35"``), or
    ``None`` when no XPU is present or the device isn't a recognized arch.

    Cached: the running device does not change within a process.
    """
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return None
    device = torch.xpu.current_device()
    major, minor = torch.ops.sgl_kernel.query_device.default(device)
    if major == 2:
        return Arch.XE20.value
    if major == 3 and minor == 5:
        return Arch.XE35.value
    return None


def supports(*archs):
    """True if the live device is one of ``archs`` (arch tags or ``Arch``)."""
    return current_arch() in {a.value if isinstance(a, Arch) else a for a in archs}


def require_arch(*archs):
    """Imperative twin of ``@pytest.mark.arch`` for standalone scripts.

    Skips (under pytest) or raises (bare ``python test_x.py``) when the live
    device is not one of ``archs``.
    """
    if supports(*archs):
        return
    wanted = sorted(str(a) for a in archs)
    reason = f"op supports {wanted}; running device is {current_arch()}"
    try:
        import pytest

        pytest.skip(reason, allow_module_level=True)
    except ImportError:
        raise RuntimeError(reason)
