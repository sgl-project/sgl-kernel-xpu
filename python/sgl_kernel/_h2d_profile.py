"""Opt-in global H2D transfer timing.

Enabled by setting the env var ``SGL_H2D_PROFILE=1`` before ``import sgl_kernel``.
When enabled, every ``cpu_tensor.to(<non_cpu_device>[, dtype])`` call prints a
single-line perf record that mirrors the shape of the C++ ``SGL_KERNEL_PERF_SCOPE``
output, so downstream log-parsers can treat both uniformly:

    h2d perf(xpu_event): time=X.XXX ms (X.XXX us), bandwidth=Y.YY GB/s, \
shape=(...) dtype=torch.bfloat16

The timing uses ``torch.xpu.Event`` when XPU is available (device-observable);
otherwise it falls back to host wall-clock. Non-H2D ``.to()`` calls -- dtype-only
casts, cpu->cpu moves, xpu->xpu moves -- are passed through unchanged and do not
print.

Note: enabling this flag forces an ``xpu.synchronize()`` on every H2D call to
resolve the timing events. If the caller relied on ``non_blocking=True`` for
async uploads, that async is lost while profiling is on. This is inherent to
"print the time right after the copy" and matches how ``torch.xpu.Event`` is
normally used.
"""

import time

import torch

_orig_to = torch.Tensor.to


def _to_timed(self, *args, **kwargs):
    # Fast-path: not from CPU -- can't be H2D.
    if self.device.type != "cpu":
        return _orig_to(self, *args, **kwargs)

    # Snapshot bytes and shape/dtype before the call so we don't rely on the
    # result (it may share storage in edge cases).
    src_bytes = self.numel() * self.element_size()
    src_shape = tuple(self.shape)
    src_dtype = self.dtype

    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()

    if xpu_available:
        start_evt = torch.xpu.Event(enable_timing=True)
        end_evt = torch.xpu.Event(enable_timing=True)
        start_evt.record()
        result = _orig_to(self, *args, **kwargs)
        end_evt.record()
    else:
        host_start = time.time()
        result = _orig_to(self, *args, **kwargs)
        host_end = time.time()

    # Post-hoc device check: skips dtype-only casts, cpu->cpu, etc.
    if result.device.type == "cpu":
        return result

    if xpu_available:
        torch.xpu.synchronize()
        elapsed_ms = start_evt.elapsed_time(end_evt)
        source = "xpu_event"
    else:
        elapsed_ms = (host_end - host_start) * 1e3
        source = "host_wall"

    gbps = (src_bytes / 1e9) / (elapsed_ms / 1e3) if elapsed_ms > 0 else 0.0
    print(
        f"h2d perf({source}): time={elapsed_ms:.6f} ms ({elapsed_ms * 1e3:.3f} us), "
        f"bandwidth={gbps:.6f} GB/s, shape={src_shape} dtype={src_dtype}",
        flush=True,
    )
    return result


def install() -> None:
    """Replace ``torch.Tensor.to`` with the timing wrapper. Idempotent."""
    if getattr(torch.Tensor.to, "_sgl_h2d_profile_installed", False):
        return
    _to_timed._sgl_h2d_profile_installed = True
    torch.Tensor.to = _to_timed
