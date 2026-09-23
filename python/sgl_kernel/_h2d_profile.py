"""Opt-in global cross-device transfer timing (H2D and D2H) + end-of-run summary.

Enabled by setting the env var ``SGL_H2D_PROFILE=1`` before ``import sgl_kernel``.
When enabled, every cross-device ``.to()`` call:

  1. Prints a single-line perf record (inline) whose shape mirrors the C++
     ``SGL_KERNEL_PERF_SCOPE`` output. The prefix distinguishes direction:

         h2d perf(xpu_event): time=X.XXX ms (X.XXX us), bandwidth=Y.YY GB/s, shape=(...) dtype=...
         d2h perf(xpu_event): time=X.XXX ms (X.XXX us), bandwidth=Y.YY GB/s, shape=(...) dtype=...

  2. Records the transfer for a per-process summary that fires via ``atexit``
     just before the interpreter exits, e.g.:

         =========== SGL_H2D_PROFILE summary ===========
         H2D: 6 transfers, total 0.443 ms, 0.528 MB, agg 1.191 GB/s
             #1  0.058 ms   0.004 MB    0.070 GB/s  (1, 16, 128)     torch.bfloat16
             ...
         D2H: 1 transfers, total 0.062 ms, 0.004 MB, agg 0.066 GB/s
             #1  0.062 ms   0.004 MB    0.066 GB/s  (1, 16, 128)     torch.bfloat16
         ================================================

- Non-transfer ``.to()`` calls (dtype-only casts, cpu->cpu, xpu->xpu) pass
  through unchanged and do not print.
- Timing uses ``torch.xpu.Event`` when XPU is available (device-observable);
  otherwise falls back to host wall-clock.
- The C++ kernel perf-lines (``mha_fwd perf(host_wall): ...``) are emitted
  inline by the C++ ``SGL_KERNEL_PERF_SCOPE`` destructor and are not
  aggregated here -- Python cannot intercept C++ printf. Grep them from the
  same log with ``grep '<op_name> perf('`` if you need a summary.

Note: enabling this flag forces ``xpu.synchronize()`` on every timed call to
resolve the timing events. non_blocking=True async transfers are effectively
serialized while profiling is on.
"""

import atexit
import time
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class _Transfer:
    direction: str  # "h2d" or "d2h"
    time_ms: float
    bytes_: int
    shape: tuple
    dtype: torch.dtype
    source: str  # "xpu_event" or "host_wall"


_transfers: List[_Transfer] = []
_orig_to = torch.Tensor.to


def _to_timed(self, *args, **kwargs):
    src_type = self.device.type

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

    dst_type = result.device.type

    if src_type == "cpu" and dst_type != "cpu":
        direction = "h2d"
    elif src_type != "cpu" and dst_type == "cpu":
        direction = "d2h"
    else:
        return result

    if xpu_available:
        torch.xpu.synchronize()
        elapsed_ms = start_evt.elapsed_time(end_evt)
        source = "xpu_event"
    else:
        elapsed_ms = (host_end - host_start) * 1e3
        source = "host_wall"

    gbps = (src_bytes / 1e9) / (elapsed_ms / 1e3) if elapsed_ms > 0 else 0.0

    # Inline print (matches the C++ SGL_KERNEL_PERF_SCOPE format shape)
    print(
        f"{direction} perf({source}): time={elapsed_ms:.6f} ms ({elapsed_ms * 1e3:.3f} us), "
        f"bandwidth={gbps:.6f} GB/s, shape={src_shape} dtype={src_dtype}",
        flush=True,
    )
    # Record for the end-of-run summary
    _transfers.append(
        _Transfer(direction, elapsed_ms, src_bytes, src_shape, src_dtype, source)
    )
    return result


def _print_summary() -> None:
    if not _transfers:
        return
    h2d = [t for t in _transfers if t.direction == "h2d"]
    d2h = [t for t in _transfers if t.direction == "d2h"]

    def _section(title: str, xs: List[_Transfer]) -> None:
        if not xs:
            return
        total_ms = sum(t.time_ms for t in xs)
        total_bytes = sum(t.bytes_ for t in xs)
        total_s = total_ms / 1e3
        agg_gbps = (total_bytes / 1e9) / total_s if total_s > 0 else 0.0
        print(
            f"{title}: {len(xs)} transfers, total {total_ms:.6f} ms, "
            f"{total_bytes / 1e6:.3f} MB, agg {agg_gbps:.6f} GB/s",
            flush=True,
        )
        for i, t in enumerate(xs, 1):
            per_gbps = (t.bytes_ / 1e9) / (t.time_ms / 1e3) if t.time_ms > 0 else 0.0
            print(
                f"    #{i:<3d} {t.time_ms:>10.6f} ms  {t.bytes_ / 1e6:>10.3f} MB  "
                f"{per_gbps:>10.6f} GB/s  shape={t.shape} dtype={t.dtype}  src={t.source}",
                flush=True,
            )

    print("=========== SGL_H2D_PROFILE summary ===========", flush=True)
    _section("H2D", h2d)
    _section("D2H", d2h)
    print("===============================================", flush=True)


def install() -> None:
    """Replace ``torch.Tensor.to`` with the timing wrapper. Idempotent."""
    if getattr(torch.Tensor.to, "_sgl_h2d_profile_installed", False):
        return
    _to_timed._sgl_h2d_profile_installed = True
    torch.Tensor.to = _to_timed
    atexit.register(_print_summary)
