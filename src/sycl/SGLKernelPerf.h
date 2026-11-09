#pragma once

// Kernel-perf reporting helper. Emits the same printf shape as PR#269's
// flash_attentionXe35_common.hpp (GPU_Clock timing + FLOPS/BYTES/us/ms/TFLOPS/
// GFLOPS/GB/s line, plus unavailability fallback). Consolidated into a single
// helper so each instrumented kernel entry-point stays focused on its op-specific
// FLOPS/BYTES formulas.
//
// The header is self-guarding (its own body is under `#if
// CUTLASS_SYCL_PROFILING_ENABLED` with a no-op `#else`), so callers include it
// unconditionally.
//
// Call-site pattern: wrap the profiling-only prologue (bytes/flops locals,
// queue lookup, scope macro) in a single `#if defined(CUTLASS_SYCL_PROFILING_ENABLED)`
// block so non-profiling builds pay zero cost -- no arithmetic, no queue
// lookup. Kernel-launch code stays outside the block and runs in both builds.
//
//   #include "SGLKernelPerf.h"
//   ...
//   #if defined(CUTLASS_SYCL_PROFILING_ENABLED)
//     const double bytes = ...;
//     const double flops = ...;
//     auto profiling_queue = at::xpu::getCurrentXPUStream().queue();
//     SGL_KERNEL_PERF_SCOPE("my_op", profiling_queue, bytes, flops);
//   #endif
//     <existing kernel-launch code>
//
// The scope object starts a GPU_Clock at construction and calls
// report_kernel_perf() at destruction. It measures correctly only when the
// target submit is the last queue operation in its scope (GPU_Clock::stop() ---
// invoked inside seconds() --- records the queue's last event at destruction).
//
// For explicit timing boundaries, pair BEGIN/END once using a named handle.
// BEGIN does not report at scope exit; END reports at its call site. Use the
// actual launch queue, which must outlive the handle. Keep formulas and both
// calls under `#if` so non-profiling builds do no profiling-only work.
//
//   #if defined(CUTLASS_SYCL_PROFILING_ENABLED)
//     SGL_KERNEL_PERF_BEGIN(first_perf, "first_op", queue);
//   #endif
//     <first submit>
//   #if defined(CUTLASS_SYCL_PROFILING_ENABLED)
//     const double bytes_a = ...;
//     const double flops_a = ...;
//     SGL_KERNEL_PERF_END(first_perf, bytes_a, flops_a);
//     SGL_KERNEL_PERF_BEGIN(second_perf, "second_op", queue);
//   #endif
//     <second submit>
//   #if defined(CUTLASS_SYCL_PROFILING_ENABLED)
//     const double bytes_b = ...;
//     const double flops_b = ...;
//     SGL_KERNEL_PERF_END(second_perf, bytes_b, flops_b);
//   #endif

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)

#include <chrono>
#include <cstdio>
#include <cutlass/util/GPU_Clock.hpp>
#include <sycl/sycl.hpp>

namespace sglkernel {

// Report per-kernel perf: prefer GPU_Clock (SYCL event elapsed time in CUTLASS's
// EventManager) when non-zero; otherwise fall back to host wall-clock time
// measured across queue.wait(). This fallback measures only the remaining wait,
// not the full launch interval, for events not captured by EventManager (e.g.
// raw q.submit() calls). Reporting synchronizes the queue in profiling builds.
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

// Explicit timer: report only when finish() is called, not at scope exit.
struct KernelPerfTimer {
  const char* op_name;
  ::sycl::queue& queue;
  GPU_Clock timer;

  KernelPerfTimer(const char* op, ::sycl::queue& q) : op_name(op), queue(q) {
    timer.start();
  }

  void finish(double bytes, double flops) {
    report_kernel_perf(op_name, queue, timer, bytes, flops);
  }

  KernelPerfTimer(const KernelPerfTimer&) = delete;
  KernelPerfTimer& operator=(const KernelPerfTimer&) = delete;
};

// RAII scope: start a GPU_Clock at construction, call report_kernel_perf() at
// destruction. The captured queue reference must outlive the scope. See the
// header comment for when to prefer this over the explicit two-call pattern.
struct KernelPerfScope {
  const char* op_name;
  ::sycl::queue& queue;
  double bytes;
  double flops;
  GPU_Clock timer;

  KernelPerfScope(const char* op, ::sycl::queue& q, double b, double f) : op_name(op), queue(q), bytes(b), flops(f) {
    timer.start();
  }

  ~KernelPerfScope() {
    report_kernel_perf(op_name, queue, timer, bytes, flops);
  }

  KernelPerfScope(const KernelPerfScope&) = delete;
  KernelPerfScope& operator=(const KernelPerfScope&) = delete;
};

}  // namespace sglkernel

// Two-level concat: `foo##__LINE__` doesn't expand __LINE__ (and identifiers
// containing `__` are reserved to the implementation, hence the non-reserved
// `sgl_perf_scope_` prefix).
#define SGL_PERF_CAT_(a, b) a##b
#define SGL_PERF_CAT(a, b) SGL_PERF_CAT_(a, b)
#define SGL_KERNEL_PERF_SCOPE(op, queue, bytes, flops) \
  ::sglkernel::KernelPerfScope SGL_PERF_CAT(sgl_perf_scope_, __LINE__)((op), (queue), (bytes), (flops))
#define SGL_KERNEL_PERF_BEGIN(handle, op, queue) ::sglkernel::KernelPerfTimer handle((op), (queue))
#define SGL_KERNEL_PERF_END(handle, bytes, flops) (handle).finish((bytes), (flops))

#else  // !CUTLASS_SYCL_PROFILING_ENABLED

// Consume each argument so call-site locals used only to compute bytes/flops
// don't trip -Wunused-variable in non-profiling builds. The comma expression
// evaluates each cast then discards the result; the compiler drops the whole
// thing.
#define SGL_KERNEL_PERF_SCOPE(op, queue, bytes, flops) ((void)(op), (void)&(queue), (void)(bytes), (void)(flops))
#define SGL_KERNEL_PERF_BEGIN(handle, op, queue) ((void)0)
#define SGL_KERNEL_PERF_END(handle, bytes, flops) ((void)0)

#endif  // CUTLASS_SYCL_PROFILING_ENABLED
