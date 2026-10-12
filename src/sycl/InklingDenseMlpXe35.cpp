#define SYCL_INTEL_TARGET 35

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>

#include "SGLKernelPerf.h"
#include "Utils.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "sgl_kernel_export.h"
#include "sycl/kernels/inkling_dense_mlp/xe_swiglu_epilogue.hpp"

namespace {

using namespace cute;

using Element = cutlass::bfloat16_t;
using Accumulator = float;
using MatrixLayout = cutlass::layout::RowMajor;
using MmaAtom = MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>;

template <int TileM, int TileN, int SgM, int SgN, bool Persistent, int OutBlocksPerStore = 1>
struct Config {
  using TileShape = Shape<Int<TileM>, Int<TileN>, Int<32>>;
  using SubgroupLayout = Layout<Shape<Int<SgM>, Int<SgN>, _1>, Stride<Int<SgN>, _1, _0>>;
  static constexpr bool kPersistent = Persistent;
  static constexpr int kOutBlocksPerStore = OutBlocksPerStore;
};

// Tile policies for small, medium, and large M. The large-M policies use
// persistent scheduling to distribute prefill work across the device.
using GateUpM16 = Config<16, 256, 1, 4, false>;
using GateUpM32 = Config<32, 128, 1, 4, false>;
using GateUpM192 = Config<128, 64, 4, 2, false>;
using GateUpM1024 = Config<128, 256, 4, 4, false, 2>;
using GateUpPrefill = Config<512, 128, 16, 2, true, 2>;

using DownM32 = Config<32, 128, 1, 4, false>;
using DownM192 = Config<128, 128, 4, 2, false>;
using DownM1024 = Config<128, 64, 4, 2, false>;
using DownPrefill = Config<512, 128, 16, 2, true>;

template <class Config_>
struct Mainloop {
  using TileShape = typename Config_::TileShape;
  using TiledMma = typename TiledMMAHelper<MmaAtom, Layout<TileShape>, typename Config_::SubgroupLayout>::TiledMMA;
  using KernelSchedule =
      cute::conditional_t<Config_::kPersistent, cutlass::gemm::KernelXeCooperative, cutlass::gemm::KernelXe>;
  using Policy = cutlass::gemm::MainloopXeL1Staged<2, KernelSchedule>;
  using Scheduler = cute::conditional_t<Config_::kPersistent, cutlass::gemm::StreamKScheduler, void>;
  using Collective = cutlass::gemm::collective::CollectiveMma<
      Policy,
      TileShape,
      Element,
      cutlass::gemm::TagToStrideA_t<MatrixLayout>,
      Element,
      cutlass::gemm::TagToStrideB_t<MatrixLayout>,
      TiledMma,
      void,
      void,
      void,
      cute::identity,
      void,
      void,
      void,
      cute::identity>;
};

template <class Config_>
struct GateUpTypes {
  using MainloopTypes = Mainloop<Config_>;
  using TileShape = typename MainloopTypes::TileShape;
  using Epilogue = cutlass::epilogue::collective::XeSwiGluGateUpEpilogue<
      TileShape,
      typename Config_::SubgroupLayout,
      MmaAtom,
      Accumulator,
      Element,
      cutlass::gemm::TagToStrideC_t<MatrixLayout>,
      cutlass::epilogue::collective::SwiGluFn,
      Config_::kOutBlocksPerStore>;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      typename MainloopTypes::Collective,
      Epilogue,
      typename MainloopTypes::Scheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <class Config_>
struct DownTypes {
  using MainloopTypes = Mainloop<Config_>;
  using TileShape = typename MainloopTypes::TileShape;
  using TiledMma = typename MainloopTypes::TiledMma;
  using EpiloguePolicy = cutlass::epilogue::IntelXeGeneric;
  using EpilogueOp = cutlass::epilogue::fusion::
      LinearCombination<Element, float, Accumulator, Accumulator, cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks = cutlass::epilogue::fusion::
      FusionCallbacks<EpiloguePolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;
  using Epilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpiloguePolicy,
      TileShape,
      void,
      Accumulator,
      cutlass::gemm::TagToStrideC_t<MatrixLayout>,
      Element,
      cutlass::gemm::TagToStrideC_t<MatrixLayout>,
      FusionCallbacks,
      void,
      void>;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      typename MainloopTypes::Collective,
      Epilogue,
      typename MainloopTypes::Scheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

void check_matrix(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_xpu(), name, " must be on XPU");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bfloat16");
  TORCH_CHECK(tensor.dim() == 2, name, " must be two-dimensional");
  TORCH_CHECK(tensor.stride(1) == 1, name, " must have unit inner stride");
}

void check_extent(int64_t value, const char* name) {
  TORCH_CHECK(value <= std::numeric_limits<int>::max(), name, " exceeds the 32-bit CUTLASS problem-size limit");
}

template <class Config_>
void run_gate_up(
    const torch::Tensor& input, const torch::Tensor& packed_weight, torch::Tensor& output, sycl::queue& queue) {
  using Types = GateUpTypes<Config_>;
  using Kernel = typename Types::Kernel;
  using Gemm = typename Types::Gemm;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideD = typename Kernel::StrideD;

  const int m = static_cast<int>(input.size(0));
  const int k = static_cast<int>(input.size(1));
  const int n = static_cast<int>(packed_weight.size(1));
  typename Kernel::ProblemShape problem{m, n, k, 1};
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n / 2, 1));
  get<0>(stride_a) = input.stride(0);
  get<2>(stride_a) = input.stride(0) * m;
  get<1>(stride_b) = packed_weight.stride(0);
  get<2>(stride_b) = packed_weight.stride(0) * k;
  get<0>(stride_d) = output.stride(0);
  get<2>(stride_d) = output.stride(0) * m;

  typename Kernel::MainloopArguments mainloop{
      reinterpret_cast<Element*>(input.data_ptr()),
      stride_a,
      reinterpret_cast<Element*>(packed_weight.data_ptr()),
      stride_b};
  typename Kernel::EpilogueArguments epilogue{reinterpret_cast<Element*>(output.data_ptr()), stride_d};

  cutlass::KernelHardwareInfo hardware;
  hardware.device_id = input.get_device();
  hardware.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hardware.device_id);
  auto arguments = [&]() {
    if constexpr (Config_::kPersistent) {
      using Params = cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams;
      return typename Kernel::Arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          problem,
          mainloop,
          epilogue,
          hardware,
          {1, Params::DecompositionMode::SplitK}};
    } else {
      return typename Kernel::Arguments{cutlass::gemm::GemmUniversalMode::kGemm, problem, mainloop, epilogue, hardware};
    }
  }();

  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess, "Inkling gate/up shape is unsupported");
  const int64_t workspace_size = static_cast<int64_t>(Gemm::get_workspace_size(arguments));
  auto workspace =
      workspace_size == 0 ? torch::Tensor() : torch::empty({workspace_size}, input.options().dtype(torch::kUInt8));
  void* workspace_ptr = workspace_size == 0 ? nullptr : workspace.data_ptr();

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  const double flops = 2.0 * m * n * k;
  const double bytes = sizeof(Element) * (double(m) * k + double(k) * n + double(m) * (n / 2));
  SGL_KERNEL_PERF_SCOPE("inkling_gate_up", queue, bytes, flops);
#endif
  CUTLASS_CHECK(gemm.run(arguments, workspace_ptr, &queue));
}

template <class Config_>
void run_down(
    const torch::Tensor& input, const torch::Tensor& packed_weight, torch::Tensor& output, sycl::queue& queue) {
  using Types = DownTypes<Config_>;
  using Kernel = typename Types::Kernel;
  using Gemm = typename Types::Gemm;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;

  const int m = static_cast<int>(input.size(0));
  const int k = static_cast<int>(input.size(1));
  const int n = static_cast<int>(packed_weight.size(1));
  typename Kernel::ProblemShape problem{m, n, k, 1};
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, 1));
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));
  get<0>(stride_a) = input.stride(0);
  get<2>(stride_a) = input.stride(0) * m;
  get<1>(stride_b) = packed_weight.stride(0);
  get<2>(stride_b) = packed_weight.stride(0) * k;
  get<0>(stride_c) = output.stride(0);
  get<2>(stride_c) = output.stride(0) * m;
  get<0>(stride_d) = output.stride(0);
  get<2>(stride_d) = output.stride(0) * m;

  typename Kernel::MainloopArguments mainloop{
      reinterpret_cast<Element*>(input.data_ptr()),
      stride_a,
      reinterpret_cast<Element*>(packed_weight.data_ptr()),
      stride_b};
  typename Kernel::EpilogueArguments epilogue{
      {1.0f, 0.0f}, nullptr, stride_c, reinterpret_cast<Element*>(output.data_ptr()), stride_d};

  cutlass::KernelHardwareInfo hardware;
  hardware.device_id = input.get_device();
  hardware.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hardware.device_id);
  auto arguments = [&]() {
    if constexpr (Config_::kPersistent) {
      using Params = cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams;
      return typename Kernel::Arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          problem,
          mainloop,
          epilogue,
          hardware,
          {1, Params::DecompositionMode::SplitK}};
    } else {
      return typename Kernel::Arguments{cutlass::gemm::GemmUniversalMode::kGemm, problem, mainloop, epilogue, hardware};
    }
  }();

  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess, "Inkling down shape is unsupported");
  const int64_t workspace_size = static_cast<int64_t>(Gemm::get_workspace_size(arguments));
  auto workspace =
      workspace_size == 0 ? torch::Tensor() : torch::empty({workspace_size}, input.options().dtype(torch::kUInt8));
  void* workspace_ptr = workspace_size == 0 ? nullptr : workspace.data_ptr();

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  const double flops = 2.0 * m * n * k;
  const double bytes = sizeof(Element) * (double(m) * k + double(k) * n + double(m) * n);
  SGL_KERNEL_PERF_SCOPE("inkling_down", queue, bytes, flops);
#endif
  CUTLASS_CHECK(gemm.run(arguments, workspace_ptr, &queue));
}

void validate_gate_up(const torch::Tensor& input, const torch::Tensor& packed_weight, const torch::Tensor& output) {
  check_matrix(input, "input");
  check_matrix(packed_weight, "packed_weight");
  check_matrix(output, "output");
  TORCH_CHECK(input.device() == packed_weight.device(), "input and packed_weight must be on the same XPU");
  TORCH_CHECK(input.device() == output.device(), "input and output must be on the same XPU");
  TORCH_CHECK(input.size(1) == packed_weight.size(0), "input K must equal packed_weight K");
  TORCH_CHECK(packed_weight.size(1) % 32 == 0, "packed gate/up N must be divisible by 32");
  TORCH_CHECK(input.stride(0) >= input.size(1), "input row stride is invalid");
  TORCH_CHECK(packed_weight.stride(0) >= packed_weight.size(1), "packed_weight row stride is invalid");
  const int64_t out_columns = packed_weight.size(1) / 2;
  TORCH_CHECK(output.size(0) == input.size(0) && output.size(1) == out_columns, "gate/up output shape mismatch");
  TORCH_CHECK(output.stride(0) >= out_columns, "gate/up output row stride is invalid");
  check_extent(input.size(0), "M");
  check_extent(input.size(1), "K");
  check_extent(packed_weight.size(1), "N");
}

void validate_down(const torch::Tensor& input, const torch::Tensor& packed_weight, const torch::Tensor& output) {
  check_matrix(input, "input");
  check_matrix(packed_weight, "packed_weight");
  check_matrix(output, "output");
  TORCH_CHECK(input.device() == packed_weight.device(), "input and packed_weight must be on the same XPU");
  TORCH_CHECK(input.device() == output.device(), "input and output must be on the same XPU");
  TORCH_CHECK(input.size(1) == packed_weight.size(0), "input K must equal packed_weight K");
  TORCH_CHECK(input.stride(0) >= input.size(1), "down input row stride is invalid");
  TORCH_CHECK(packed_weight.stride(0) >= packed_weight.size(1), "packed_weight row stride is invalid");
  TORCH_CHECK(output.size(0) == input.size(0) && output.size(1) == packed_weight.size(1), "down output shape mismatch");
  TORCH_CHECK(output.is_contiguous(), "down output must be contiguous");
  check_extent(input.size(0), "M");
  check_extent(input.size(1), "K");
  check_extent(packed_weight.size(1), "N");
}

void dispatch_gate_up(const torch::Tensor& input, const torch::Tensor& packed_weight, torch::Tensor& output) {
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  const int64_t m = input.size(0);
  if (m <= 16) {
    run_gate_up<GateUpM16>(input, packed_weight, output, queue);
  } else if (m <= 32) {
    run_gate_up<GateUpM32>(input, packed_weight, output, queue);
  } else if (m <= 192) {
    run_gate_up<GateUpM192>(input, packed_weight, output, queue);
  } else if (m <= 1024) {
    run_gate_up<GateUpM1024>(input, packed_weight, output, queue);
  } else {
    run_gate_up<GateUpPrefill>(input, packed_weight, output, queue);
  }
}

void dispatch_down(const torch::Tensor& input, const torch::Tensor& packed_weight, torch::Tensor& output) {
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  const int64_t m = input.size(0);
  if (m <= 32) {
    run_down<DownM32>(input, packed_weight, output, queue);
  } else if (m <= 192) {
    run_down<DownM192>(input, packed_weight, output, queue);
  } else if (m <= 1024) {
    run_down<DownM1024>(input, packed_weight, output, queue);
  } else {
    run_down<DownPrefill>(input, packed_weight, output, queue);
  }
}

}  // namespace

SGL_KERNEL_EXPORT void
inkling_gate_up_out(const torch::Tensor& input, const torch::Tensor& packed_weight, torch::Tensor& output) {
  validate_gate_up(input, packed_weight, output);
  if (input.size(0) == 0 || packed_weight.size(1) == 0) {
    return;
  }
  TORCH_CHECK(input.size(1) > 0, "K must be greater than zero");
  dispatch_gate_up(input, packed_weight, output);
}

SGL_KERNEL_EXPORT torch::Tensor inkling_gate_up(const torch::Tensor& input, const torch::Tensor& packed_weight) {
  check_matrix(input, "input");
  check_matrix(packed_weight, "packed_weight");
  TORCH_CHECK(packed_weight.size(1) % 2 == 0, "packed gate/up N must be even");
  const int64_t out_columns = packed_weight.size(1) / 2;
  auto output = torch::empty_strided({input.size(0), out_columns}, {out_columns + 64, 1}, input.options());
  inkling_gate_up_out(input, packed_weight, output);
  return output;
}

SGL_KERNEL_EXPORT void
inkling_down_out(const torch::Tensor& input, const torch::Tensor& packed_weight, torch::Tensor& output) {
  validate_down(input, packed_weight, output);
  if (input.size(0) == 0 || packed_weight.size(1) == 0) {
    return;
  }
  TORCH_CHECK(input.size(1) > 0, "K must be greater than zero");
  dispatch_down(input, packed_weight, output);
}

SGL_KERNEL_EXPORT torch::Tensor inkling_down(const torch::Tensor& input, const torch::Tensor& packed_weight) {
  check_matrix(input, "input");
  check_matrix(packed_weight, "packed_weight");
  auto output = torch::empty({input.size(0), packed_weight.size(1)}, input.options());
  inkling_down_out(input, packed_weight, output);
  return output;
}

#undef SYCL_INTEL_TARGET
