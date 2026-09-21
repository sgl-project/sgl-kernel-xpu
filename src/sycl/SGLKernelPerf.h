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

#include <cstdio>
#include <cutlass/util/GPU_Clock.hpp>
#include <sycl/sycl.hpp>

namespace sglkernel {

inline void
report_kernel_perf(const char* op_name, ::sycl::queue& queue, GPU_Clock& timer, double bytes, double flops) {
  queue.wait();
  const double elapsed_s = timer.seconds();
  const double tflops = elapsed_s > 0.0 ? (flops * 1e-12) / elapsed_s : 0.0;
  const double gbps = elapsed_s > 0.0 ? (bytes * 1e-9) / elapsed_s : 0.0;

  ::printf(
      "%s perf(gpu_clock): time=%.9f ms (%.3f us), bandwidth=%.6f GB/s, "
      "compute=%.6f TFLOPS (%.3f GFLOPS)\n",
      op_name,
      elapsed_s * 1000.0,
      elapsed_s * 1e6,
      gbps,
      tflops,
      tflops * 1e3);

  if (elapsed_s <= 0.0) {
    ::printf(
        "%s perf(gpu_clock): unavailable (reported 0). Check SYCL profiling event path and "
        "CUTLASS_SYCL_PROFILING_ENABLED build flag.\n",
        op_name);
  }
}

}  // namespace sglkernel

#endif  // CUTLASS_SYCL_PROFILING_ENABLED
