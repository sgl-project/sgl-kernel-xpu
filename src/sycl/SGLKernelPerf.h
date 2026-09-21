#pragma once

// Kernel-perf reporting helper. Emits the same printf shape as PR#269's
// flash_attentionXe35_common.hpp (GPU_Clock timing + FLOPS/BYTES/us/ms/TFLOPS/
// GFLOPS/GB/s line, plus unavailability fallback). Consolidated into a single
// helper so each instrumented kernel entry-point stays focused on its op-specific
// FLOPS/BYTES formulas.
//
// Usage pattern (mirrors the PR#269 wrap):
//
//   #if defined(CUTLASS_SYCL_PROFILING_ENABLED)
//     auto profiling_queue = at::xpu::getCurrentXPUStream().queue();
//     GPU_Clock timer;
//     timer.start();
//   #endif
//     <existing kernel-launch code>
//   #if defined(CUTLASS_SYCL_PROFILING_ENABLED)
//     ::sglkernel::report_kernel_perf("my_op", profiling_queue, timer, bytes, flops);
//   #endif

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)

#include <chrono>
#include <cstdio>
#include <cutlass/util/GPU_Clock.hpp>
#include <sycl/sycl.hpp>

namespace sglkernel {

// Report per-kernel perf: prefer GPU_Clock (SYCL event elapsed time in CUTLASS's
// EventManager) when non-zero; otherwise fall back to host wall-clock time
// measured across queue.wait(). The fallback makes the perf line meaningful for
// any kernel whose SYCL events are not captured by EventManager -- e.g. raw
// q.submit() call sites that don't go through the sycl_kernel_submit() /
// common.h::launch<> helpers -- so the reporting is generic: every instrumented
// kernel prints a real number without needing custom event wiring at its launch
// site.
inline void
report_kernel_perf(const char* op_name, ::sycl::queue& queue, GPU_Clock& timer, double bytes, double flops) {
  const auto host_start = std::chrono::high_resolution_clock::now();
  queue.wait();
  const auto host_end = std::chrono::high_resolution_clock::now();
  const double host_elapsed_s = std::chrono::duration<double>(host_end - host_start).count();

  const double sycl_elapsed_s = timer.seconds();
  const bool used_sycl = sycl_elapsed_s > 0.0;
  const double elapsed_s = used_sycl ? sycl_elapsed_s : host_elapsed_s;
  const double tflops = elapsed_s > 0.0 ? (flops * 1e-12) / elapsed_s : 0.0;
  const double gbps = elapsed_s > 0.0 ? (bytes * 1e-9) / elapsed_s : 0.0;

  ::printf(
      "%s perf(%s): time=%.9f ms (%.3f us), bandwidth=%.6f GB/s, "
      "compute=%.6f TFLOPS (%.6f GFLOPS)\n",
      op_name,
      used_sycl ? "gpu_clock" : "host_wall",
      elapsed_s * 1000.0,
      elapsed_s * 1e6,
      gbps,
      tflops,
      tflops * 1e3);

  if (elapsed_s <= 0.0) {
    ::printf(
        "%s perf: unavailable (both gpu_clock and host_wall reported 0). Check SYCL profiling "
        "event path and CUTLASS_SYCL_PROFILING_ENABLED build flag.\n",
        op_name);
  }
}

}  // namespace sglkernel

#endif  // CUTLASS_SYCL_PROFILING_ENABLED
