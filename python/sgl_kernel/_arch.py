"""Architecture detection and per-architecture backend selection.

The wheel ships one compiled backend per Intel GPU architecture (``_xe20`` for
Xe2/BMG, ``_xe35`` for Xe3P/CRI), because ``SYCL_INTEL_TARGET`` is a
compile-time switch that reaches host code in ~37% of the common kernel
sources. Exactly one backend is imported per process.

The architecture cannot be read from ``torch.ops.sgl_kernel.query_device``
here: that op lives inside the backend we are trying to choose. Instead we load
``libsgl_arch_probe.so``, which is built independently of ``SYCL_INTEL_TARGET``
and exposes the same SYCL query behind a C entry point.
"""

import ctypes
import os
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent

# (major, minor) from the probe -> backend subpackage.
_BACKENDS = {
    (2, 0): "_xe20",  # Xe2  (BMG)
    (3, 5): "_xe35",  # Xe3P (CRI)
}

_PROBE_LIB = "libsgl_arch_probe.so"

# Escape hatch for debugging and for forcing a backend in CI.
_OVERRIDE_ENV = "SGL_KERNEL_ARCH"


def _has_backend(name):
    # The stub __init__.py ships in every wheel, so only the payload proves presence.
    return any((_PKG_DIR / name).glob("*.so"))


def _available_backends():
    return sorted(b for b in set(_BACKENDS.values()) if _has_backend(b))


def probe_capability():
    """Return ``(major, minor)`` for the current XPU device via the probe library."""
    lib_path = _PKG_DIR / _PROBE_LIB
    if not lib_path.exists():
        raise RuntimeError(
            f"architecture probe {lib_path} is missing; the wheel is incomplete"
        )

    lib = ctypes.CDLL(str(lib_path))
    fn = lib.sgl_probe_device_capability
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]

    major = ctypes.c_int64()
    minor = ctypes.c_int64()
    rc = fn(-1, ctypes.byref(major), ctypes.byref(minor))
    if rc != 0:
        raise RuntimeError(
            f"could not determine the XPU architecture (probe returned {rc}). "
            "The device is either unavailable or not supported by this build."
        )
    return major.value, minor.value


def select_backend():
    """Return the name of the backend subpackage to import for this device."""
    forced = os.environ.get(_OVERRIDE_ENV)
    if forced:
        if forced not in set(_BACKENDS.values()):
            raise RuntimeError(
                f"{_OVERRIDE_ENV}={forced!r} is not one of {sorted(set(_BACKENDS.values()))}"
            )
        return forced

    capability = probe_capability()
    backend = _BACKENDS.get(capability)
    if backend is None:
        raise RuntimeError(
            f"XPU device capability {capability} has no backend in this wheel "
            f"(available: {_available_backends()})"
        )
    if not _has_backend(backend):
        raise RuntimeError(
            f"device capability {capability} needs the {backend!r} backend, which this wheel "
            f"does not contain (available: {_available_backends()}). "
            "This is a single-architecture build."
        )
    return backend


def load_backend():
    """Import the architecture-appropriate ``common_ops`` extension.

    Importing it is what registers the ``sgl_kernel::*`` operators with torch;
    the returned module is not otherwise used.
    """
    backend = select_backend()
    import importlib

    module = importlib.import_module(f"{__package__}.{backend}.common_ops")
    # Let `import sgl_kernel.common_ops` keep working regardless of backend.
    sys.modules[f"{__package__}.common_ops"] = module
    return module
