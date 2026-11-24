/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief NVFP4 weight-only-quantized MXFP8 grouped GEMM for MoE on Intel XPU (xe35).

    MXFP8 activations: B arrives as NVFP4 (packed E2M1 payload) and is
    requantized to MXFP8 in registers by a hand-written, lane-local SYCL kernel
    (nvfp4_woq_mxfp8_mainloop.hpp: numerics, register layouts, tile menu); one
    launch per call, no metadata/transpose prep kernel.  src/sycl/nvfp4_woq.cpp
    holds the one-time scale prep that produces the (c, sfmx) pair it consumes.

    BF16 activations: CUTLASS grouped GEMM with nvfp4_woq_bf16_mainloop.hpp.

    NOTE: unlike blockwise_moe_mxfp4.cpp this TU does NOT define
    CUTE_ENABLE_XE_BLOCK_2D_ASSERT (device asserts fire per work-item on
    ragged-M shapes this code handles by construction).
*/

// clang-format off
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <torch/extension.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_xe.hpp>
#include <cute/util/compat.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "moe_group_gemm_helper.hpp"
#include "nvfp4_woq_bf16_mainloop.hpp"
#include "nvfp4_woq_mxfp8_mainloop.hpp"

using namespace cute;
using namespace cutlass::gemm;

class NvFp4WoqBf16BuildMetadata;

namespace at::native::xpu {

using NvFp4GroupedProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using NvFp4UnderlyingProblemShapeType = typename NvFp4GroupedProblemShape::UnderlyingProblemShape;

int cached_device_multiprocessor_count(int device_id) {
  static std::mutex cache_mutex;
  static std::unordered_map<int, int> cache;
  thread_local bool thread_cache_valid = false;
  thread_local int thread_device_id = 0;
  thread_local int thread_count = 0;

  if (thread_cache_valid && thread_device_id == device_id) {
    return thread_count;
  }
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = cache.find(device_id);
  if (it != cache.end()) {
    thread_count = it->second;
  } else {
    thread_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
    cache.emplace(device_id, thread_count);
  }
  thread_device_id = device_id;
  thread_cache_valid = true;
  return thread_count;
}

// NVFP4-B x MXFP8-A grouped GEMM: see nvfp4_woq_mxfp8_mainloop.hpp.
//
// Tile menu (measured on CRI, see the sweep notes in the agent report).
//
//   m8 / m16 / m32  one 1x4 subgroup row per workgroup, SG tile SG_M x 16,
//                   128-wide k steps.  B is converted once per subgroup, i.e.
//                   once per SG_M rows; that is the right trade while the GEMM
//                   is memory bound (few rows per expert).  The 16-column
//                   subgroup keeps each subgroup's L1 working set small: with
//                   32 subgroups per Xe-core, 64-column subgroups thrash L1 and
//                   lose up to 4x at K = 2048.
//   m128 / m256     4x4 subgroups of 32x64 / 64x64 sharing converted B through
//                   a double-buffered SLM stage (split barrier per 64-wide k
//                   step), so B is converted once per 128 / 256 rows.  Needed
//                   once the conversion, not memory, bounds the kernel.
//
// GRF budget: this TU is compiled with -cl-intel-512-GRF-per-thread (see
// src/BuildOnLinux.cmake), i.e. 4 hardware threads per EU, 32 per Xe-core.
//
//                                                       BM   BN SGM SGN  KT   CB PF SHARE
using NvFp4WoqMxfp8M8Kernel   = sgl_xpu::NvFp4WoqMxfp8Kernel<  8,  64, 1, 4, 128, 512, 2>;
using NvFp4WoqMxfp8M16Kernel  = sgl_xpu::NvFp4WoqMxfp8Kernel< 16,  64, 1, 4, 128, 512, 2>;
using NvFp4WoqMxfp8M32Kernel  = sgl_xpu::NvFp4WoqMxfp8Kernel< 32,  64, 1, 4, 128, 512, 2>;
using NvFp4WoqMxfp8M128Kernel = sgl_xpu::NvFp4WoqMxfp8Kernel<128, 256, 4, 4,  64,  64, 2, true>;
using NvFp4WoqMxfp8M256Kernel = sgl_xpu::NvFp4WoqMxfp8Kernel<256, 256, 4, 4,  64,  64, 2, true>;

enum class NvFp4WoqMxfp8Tile { M8, M16, M32, M128, M256 };

constexpr int kNvFp4WoqThreadsPerXeCore = 32;  // 8 EUs x 4 threads at 512 GRF

/// Host-side tile choice from the flat row count only (problem_sizes stays on
/// the device, so no D2H sync): rows / E approximates rows per active expert
/// from above once most experts are active, and from below at decode where
/// most are empty.  Thresholds fitted on qwen3-30b, gpt-oss-120b and dsv3-ep8
/// (up and down projections, T = 1 .. 4096).
NvFp4WoqMxfp8Tile select_nvfp4_woq_mxfp8_tile(int64_t total_rows, int num_groups, int k) {
  const int64_t rows_per_group = (total_rows + num_groups - 1) / num_groups;
  if (rows_per_group <= 8) return NvFp4WoqMxfp8Tile::M8;
  // Long-K experts with ~17 rows prefer two 16-row tiles over one 32-row tile.
  if (rows_per_group <= 16 || (rows_per_group <= 24 && k >= 4096)) return NvFp4WoqMxfp8Tile::M16;
  if (rows_per_group <= 48) return NvFp4WoqMxfp8Tile::M32;
  if (rows_per_group <= 384) return NvFp4WoqMxfp8Tile::M128;
  return NvFp4WoqMxfp8Tile::M256;
}

/// Optional override for benchmarking / tests: SGLANG_NVFP4_WOQ_MXFP8_TILE in
/// {m8, m16, m32, m128, m256}.
bool nvfp4_woq_mxfp8_tile_override(NvFp4WoqMxfp8Tile& tile) {
  const char* value = std::getenv("SGLANG_NVFP4_WOQ_MXFP8_TILE");
  if (value == nullptr || value[0] == '\0') return false;
  static const std::pair<const char*, NvFp4WoqMxfp8Tile> names[] = {
      {"m8", NvFp4WoqMxfp8Tile::M8},     {"m16", NvFp4WoqMxfp8Tile::M16},
      {"m32", NvFp4WoqMxfp8Tile::M32},   {"m128", NvFp4WoqMxfp8Tile::M128},
      {"m256", NvFp4WoqMxfp8Tile::M256}};
  for (auto const& [name, t] : names) {
    if (std::strcmp(value, name) == 0) {
      tile = t;
      return true;
    }
  }
  TORCH_CHECK(false, "SGLANG_NVFP4_WOQ_MXFP8_TILE must be one of m8/m16/m32/m128/m256; got '",
              value, "'");
  return false;
}

template <class Kernel>
void launch_nvfp4_woq_mxfp8(sgl_xpu::NvFp4WoqMxfp8Params const& params, sycl::queue& queue,
                            int xe_cores) {
  // Upper bound on the tile count without reading problem_sizes: an expert with
  // m > 0 rows has ceil(m / BM) <= m / BM + 1 m-tiles, and at most
  // min(E, total_m) experts are non-empty.
  const int64_t n_tiles = (params.n + Kernel::BLK_N - 1) / Kernel::BLK_N;
  const int64_t m_tiles_bound =
      params.total_m / Kernel::BLK_M +
      std::min<int64_t>(params.num_experts, params.total_m);
  const int64_t tiles_bound = std::max<int64_t>(1, m_tiles_bound * n_tiles);
  // Persistent grid: as many workgroups as fit at 512 GRF (no more, or the
  // round-robin tile assignment leaves the queued ones a tail of work).
  const int64_t wgs_per_core =
      std::max(1, kNvFp4WoqThreadsPerXeCore / (Kernel::WG_SIZE / 16));
  const int64_t groups = std::min<int64_t>(tiles_bound, wgs_per_core * xe_cores);
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint32_t, 1> slm;
    if constexpr (Kernel::SHARE) {
      slm = sycl::local_accessor<uint32_t, 1>(Kernel::SLM_DWORDS, cgh);
    }
    cgh.parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(groups) * Kernel::WG_SIZE, Kernel::WG_SIZE),
        Kernel{params, slm});
  });
}

constexpr char NvFp4WoqQuantizeInKLoopEnv[] =
    "SGLANG_NVFP4_WOQ_QUANTIZE_IN_K_LOOP";

bool nvfp4_woq_quantize_in_k_loop_enabled() {
  const char* value = std::getenv(NvFp4WoqQuantizeInKLoopEnv);
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0 ||
      std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0 ||
      std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0) {
    return false;
  }
  if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
      std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "on") == 0 ||
      std::strcmp(value, "ON") == 0) {
    return true;
  }
  TORCH_CHECK(
      false,
      NvFp4WoqQuantizeInKLoopEnv,
      " must be one of 0/1, false/true, or off/on; got '",
      value,
      "'");
  return false;
}

// NVFP4-B x BF16-A: B is converted and scaled to BF16 in registers, then a
// regular BF16 DPAS performs the matrix multiply with FP32 accumulation.
template <class TileShape_, class ThreadLayout_>
struct NvFp4WoqBf16TypesBase {
  using ElementInputA = cutlass::bfloat16_t;
  using ElementInputB = cutlass::float_e2m1_t;
  using ElementScale = uint8_t;
  using ElementMult = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementOutput = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using TileShape = TileShape_;
  using ThreadLayout = ThreadLayout_;
  using TiledMma = typename cute::TiledMMAHelper<
      cute::MMA_Atom<cute::XE_DPAS_TT<8, float, ElementInputA>>,
      cute::Layout<TileShape>,
      ThreadLayout>::TiledMMA;

  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy =
      sgl_xpu::MainloopXeNvFp4WoqBf16Group<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(cute::tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC*>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD*>,
      FusionCallBacks,
      XE_2D_U32x8x16_LD_N,
      void,
      void,
      XE_2D_U32x8x16_ST_N,
      void,
      void>;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementInputA,
      cutlass::gemm::TagToStrideA_t<LayoutA*>,
      ElementInputB,
      cutlass::gemm::TagToStrideB_t<LayoutB*>,
      TiledMma,
      void,
      void,
      void,
      cute::identity,
      void,
      void,
      void,
      cute::identity>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      NvFp4GroupedProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::GroupScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::InternalStrideA;
  using StrideB = typename GemmKernel::InternalStrideB;
  using StrideC = typename GemmKernel::InternalStrideC;
  using StrideD = typename GemmKernel::InternalStrideD;
};

using NvFp4WoqBf16M8N128Types = NvFp4WoqBf16TypesBase<
    Shape<_8, _128, _32>,
    cute::Layout<cute::Shape<_1, _4, _1>,
                 cute::Stride<_4, _1, _0>>>;
using NvFp4WoqBf16M8N256Types = NvFp4WoqBf16TypesBase<
    Shape<_8, _256, _32>,
    cute::Layout<cute::Shape<_1, _8, _1>,
                 cute::Stride<_8, _1, _0>>>;
using NvFp4WoqBf16M32N256Types = NvFp4WoqBf16TypesBase<
    Shape<_32, _256, _32>,
    cute::Layout<cute::Shape<_1, _8, _1>,
                 cute::Stride<_8, _1, _0>>>;
using NvFp4WoqBf16M64N256Types = NvFp4WoqBf16TypesBase<
    Shape<_64, _256, _32>,
    cute::Layout<cute::Shape<_2, _8, _1>,
                 cute::Stride<_8, _1, _0>>>;
using NvFp4WoqBf16Types = NvFp4WoqBf16TypesBase<
    Shape<_256, _256, _32>,
    cute::Layout<cute::Shape<_8, _4, _1>,
                 cute::Stride<_4, _1, _0>>>;

enum class NvFp4WoqBf16Tile {
  M8N128,
  M8N256,
  M32N256,
  M64N256,
  M256N256,
};

NvFp4WoqBf16Tile select_nvfp4_woq_bf16_tile(
    int64_t total_rows, int num_groups, int n, int sm_count) {
  const int64_t rows_per_group =
      (total_rows + num_groups - 1) / num_groups;
  if (rows_per_group <= 8) {
    const int64_t m8n256_workgroups =
        static_cast<int64_t>(num_groups) * cute::ceil_div(n, 256);
    return m8n256_workgroups < sm_count ? NvFp4WoqBf16Tile::M8N128
                                       : NvFp4WoqBf16Tile::M8N256;
  }
  if (rows_per_group <= 32) return NvFp4WoqBf16Tile::M32N256;
  if (rows_per_group <= 64) return NvFp4WoqBf16Tile::M64N256;
  return NvFp4WoqBf16Tile::M256N256;
}

template <class Types>
void run_nvfp4_woq_bf16_gemm(
    int num_groups,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& ptr_table,
    const torch::Tensor& cb_ptrs,
    const torch::Tensor& cb_ld,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_cd,
    const torch::Tensor& workspace,
    int sm_count,
    at::Device device) {
  using Bf16Gemm = typename Types::Gemm;
  using Bf16GemmKernel = typename Types::GemmKernel;

  auto row = [&](int i) { return ptr_table[i].data_ptr(); };

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = static_cast<int>(device.index());
  hw_info.sm_count = sm_count;

  auto stream = at::xpu::getCurrentXPUStream(device.index());
  sycl::queue& queue = stream.queue();

  typename Bf16Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::
      PersistentTileSchedulerXeGroup<NvFp4GroupedProblemShape>::RasterOrderOptions;
  typename Bf16GemmKernel::Arguments gemm_args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      typename Bf16GemmKernel::ProblemShape{
          num_groups,
          reinterpret_cast<NvFp4UnderlyingProblemShapeType*>(
              problem_sizes.data_ptr<int32_t>()),
          nullptr},
      typename Bf16GemmKernel::MainloopArguments{
          reinterpret_cast<typename Types::ElementInputA const**>(row(0)),
          reinterpret_cast<typename Types::StrideA*>(stride_a.data_ptr<int64_t>()),
          reinterpret_cast<typename Types::ElementInputB const**>(row(1)),
          reinterpret_cast<typename Types::StrideB*>(stride_b.data_ptr<int64_t>()),
          reinterpret_cast<typename Types::ElementScale const**>(row(3)),
          reinterpret_cast<typename Types::ElementMult const**>(cb_ptrs.data_ptr()),
          cb_ld.data_ptr<int64_t>()},
      typename Bf16GemmKernel::EpilogueArguments{
          fusion_args,
          reinterpret_cast<typename Types::ElementOutput const**>(row(2)),
          reinterpret_cast<typename Types::StrideC*>(stride_cd.data_ptr<int64_t>()),
          reinterpret_cast<typename Types::ElementOutput**>(row(2)),
          reinterpret_cast<typename Types::StrideD*>(stride_cd.data_ptr<int64_t>())},
      hw_info,
      typename Bf16GemmKernel::TileSchedulerArguments{
          1, RasterOrderOptions::AlongN}};

  Bf16Gemm gemm_op;
  const size_t workspace_size = Bf16Gemm::get_workspace_size(gemm_args);
  TORCH_CHECK(
      static_cast<size_t>(workspace.numel()) >= workspace_size,
      "nvfp4_woq_bf16_grouped_mm: workspace insufficient, need ",
      workspace_size,
      " bytes");
  TORCH_CHECK(
      gemm_op.initialize(gemm_args, workspace.data_ptr()) ==
          cutlass::Status::kSuccess,
      "nvfp4_woq_bf16_grouped_mm: failed to initialize");
  TORCH_CHECK(
      gemm_op.run(&queue) == cutlass::Status::kSuccess,
      "nvfp4_woq_bf16_grouped_mm: failed to run");
}

}  // namespace at::native::xpu

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
//
// Flat-2D ragged-M layout, matching fp8_blockwise_scaled_grouped_mm's
// on-device-prep path: A / scales_a / output are 2D with per-expert row
// offsets in expert_offsets, weights are 3D per expert.
void nvfp4_woq_mxfp8_scaled_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  constexpr int BS = 32;   // MXFP8 / bdpas block
  constexpr int MBS = 16;  // NVFP4 block
  constexpr int BLK_K = 64;

  TORCH_CHECK(a.device().is_xpu(), "nvfp4_woq: A must be on XPU");
  const auto device = a.device();
  TORCH_CHECK(b.device() == device, "nvfp4_woq: B must be on the same XPU device as A");
  TORCH_CHECK(output.device() == device, "nvfp4_woq: output must be on the same XPU device as A");
  TORCH_CHECK(scales_a.device() == device, "nvfp4_woq: scales_a must be on the same XPU device as A");
  TORCH_CHECK(sfmx_b.device() == device, "nvfp4_woq: sfmx_b must be on the same XPU device as A");
  TORCH_CHECK(c_b.device() == device, "nvfp4_woq: c_b must be on the same XPU device as A");
  TORCH_CHECK(
      problem_sizes.device() == device,
      "nvfp4_woq: problem_sizes must be on the same XPU device as A");
  TORCH_CHECK(
      expert_offsets.device() == device,
      "nvfp4_woq: expert_offsets must be on the same XPU device as A");
  TORCH_CHECK(
      workspace.device() == device,
      "nvfp4_woq: workspace must be on the same XPU device as A");

  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn, "nvfp4_woq: A must be float8_e4m3fn (MXFP8)");
  TORCH_CHECK(b.scalar_type() == torch::kUInt8, "nvfp4_woq: B must be uint8 (packed E2M1)");
  TORCH_CHECK(scales_a.scalar_type() == torch::kUInt8, "nvfp4_woq: scales_a must be uint8 (UE8M0)");
  TORCH_CHECK(sfmx_b.scalar_type() == torch::kUInt8, "nvfp4_woq: sfmx_b must be uint8 (UE8M0)");
  TORCH_CHECK(c_b.scalar_type() == torch::kBFloat16, "nvfp4_woq: c_b must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "nvfp4_woq: output must be float32");

  TORCH_CHECK(a.dim() == 2, "nvfp4_woq: A must be flat 2D (sum_m, K)");
  TORCH_CHECK(scales_a.dim() == 2, "nvfp4_woq: scales_a must be flat 2D (sum_m, K/32)");
  TORCH_CHECK(output.dim() == 2, "nvfp4_woq: output must be flat 2D (sum_m, N)");
  TORCH_CHECK(b.dim() == 3, "nvfp4_woq: B must be 3D (E, N, K/2)");
  TORCH_CHECK(sfmx_b.dim() == 3, "nvfp4_woq: sfmx_b must be 3D (E, K/32, N)");
  TORCH_CHECK(c_b.dim() == 3, "nvfp4_woq: c_b must be 3D (E, N, K/16)");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && scales_a.is_contiguous() &&
                  sfmx_b.is_contiguous() && c_b.is_contiguous() && output.is_contiguous(),
              "nvfp4_woq: all tensors must be contiguous");
  TORCH_CHECK(problem_sizes.dim() == 2 && problem_sizes.size(1) == 3,
              "nvfp4_woq: problem_sizes must be (E, 3) int32");
  TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32 && expert_offsets.scalar_type() == torch::kInt32,
              "nvfp4_woq: problem_sizes / expert_offsets must be int32");

  const int E = static_cast<int>(expert_offsets.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(output.size(1));
  TORCH_CHECK(E > 0, "nvfp4_woq: expert count must be positive");
  TORCH_CHECK(problem_sizes.size(0) == E, "nvfp4_woq: problem_sizes / expert_offsets expert mismatch");
  TORCH_CHECK(K % BLK_K == 0, "nvfp4_woq: K must be a multiple of ", BLK_K, " (got ", K, ")");
  TORCH_CHECK(N % 4 == 0, "nvfp4_woq: N must be 4-byte aligned for the MN-major scale load (got ", N, ")");

  const int scale_k = K / BS;
  TORCH_CHECK(scales_a.size(1) == scale_k, "nvfp4_woq: scales_a must be (*, K/32)");
  TORCH_CHECK(b.size(0) == E && b.size(1) == N && b.size(2) == K / 2,
              "nvfp4_woq: B must be (E, N, K/2)");
  TORCH_CHECK(sfmx_b.size(0) == E && sfmx_b.size(1) == scale_k && sfmx_b.size(2) == N,
              "nvfp4_woq: sfmx_b must be (E, K/32, N) MN-major");
  TORCH_CHECK(c_b.size(0) == E && c_b.size(1) == N && c_b.size(2) == K / MBS,
              "nvfp4_woq: c_b must be (E, N, K/16) row-major");

  // Every per-expert B-scale slice becomes the base address of an Xe block-2D
  // load, and those must be 64-byte aligned. Torch's allocator aligns the
  // tensor itself, so the requirement lands on the per-expert stride.
  TORCH_CHECK(sfmx_b.stride(0) % 64 == 0,
              "nvfp4_woq: N * (K/32) must be a multiple of 64 so each expert's B-scale slice is "
              "64-byte aligned for the Xe block-2D load; got N=", N, " K=", K,
              " (per-expert stride ", sfmx_b.stride(0), " bytes)");

  const int64_t total_m = a.size(0);
  TORCH_CHECK(total_m > 0, "nvfp4_woq: flat A must have at least one row");
  TORCH_CHECK(output.size(0) == total_m && scales_a.size(0) == total_m,
              "nvfp4_woq: A, scales_a and output flat row counts must match");
  TORCH_CHECK(total_m < (int64_t{1} << 24) && static_cast<int64_t>(E) * N < (int64_t{1} << 24),
              "nvfp4_woq: row counts exceed the Xe block-2D surface height limit");
  (void)workspace;

  sgl_xpu::NvFp4WoqMxfp8Params params{
      problem_sizes.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      reinterpret_cast<const uint8_t*>(a.data_ptr()),
      scales_a.data_ptr<uint8_t>(),
      b.data_ptr<uint8_t>(),
      sfmx_b.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(c_b.data_ptr()),
      output.data_ptr<float>(),
      E,
      N,
      K,
      static_cast<int>(total_m)};

  // The lane-local kernel converts each K=32 MMA step right before its bdpas,
  // so the legacy "quantize in k loop" switch selects the same kernel; it is
  // still parsed so that invalid values keep failing loudly.
  (void)at::native::xpu::nvfp4_woq_quantize_in_k_loop_enabled();

  auto tile = at::native::xpu::select_nvfp4_woq_mxfp8_tile(total_m, E, K);
  at::native::xpu::nvfp4_woq_mxfp8_tile_override(tile);
  auto& queue = at::xpu::getCurrentXPUStream(device.index()).queue();
  const int xe_cores =
      at::native::xpu::cached_device_multiprocessor_count(static_cast<int>(device.index()));
  using at::native::xpu::NvFp4WoqMxfp8Tile;
  switch (tile) {
    case NvFp4WoqMxfp8Tile::M8:
      at::native::xpu::launch_nvfp4_woq_mxfp8<at::native::xpu::NvFp4WoqMxfp8M8Kernel>(
          params, queue, xe_cores);
      break;
    case NvFp4WoqMxfp8Tile::M16:
      at::native::xpu::launch_nvfp4_woq_mxfp8<at::native::xpu::NvFp4WoqMxfp8M16Kernel>(
          params, queue, xe_cores);
      break;
    case NvFp4WoqMxfp8Tile::M32:
      at::native::xpu::launch_nvfp4_woq_mxfp8<at::native::xpu::NvFp4WoqMxfp8M32Kernel>(
          params, queue, xe_cores);
      break;
    case NvFp4WoqMxfp8Tile::M128:
      at::native::xpu::launch_nvfp4_woq_mxfp8<at::native::xpu::NvFp4WoqMxfp8M128Kernel>(
          params, queue, xe_cores);
      break;
    case NvFp4WoqMxfp8Tile::M256:
      at::native::xpu::launch_nvfp4_woq_mxfp8<at::native::xpu::NvFp4WoqMxfp8M256Kernel>(
          params, queue, xe_cores);
      break;
  }
}

void nvfp4_woq_bf16_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfmx_b,
    const torch::Tensor& c_b,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  using Types = at::native::xpu::NvFp4WoqBf16Types;
  constexpr int BLK_K = cute::get<2>(Types::TileShape{});

  TORCH_CHECK(a.device().is_xpu(), "nvfp4_woq_bf16: A must be on XPU");
  const auto device = a.device();
  TORCH_CHECK(b.device() == device, "nvfp4_woq_bf16: B must be on the same XPU device as A");
  TORCH_CHECK(output.device() == device, "nvfp4_woq_bf16: output must be on the same XPU device as A");
  TORCH_CHECK(sfmx_b.device() == device, "nvfp4_woq_bf16: sfmx_b must be on the same XPU device as A");
  TORCH_CHECK(c_b.device() == device, "nvfp4_woq_bf16: c_b must be on the same XPU device as A");
  TORCH_CHECK(
      problem_sizes.device() == device,
      "nvfp4_woq_bf16: problem_sizes must be on the same XPU device as A");
  TORCH_CHECK(
      expert_offsets.device() == device,
      "nvfp4_woq_bf16: expert_offsets must be on the same XPU device as A");
  TORCH_CHECK(
      workspace.device() == device,
      "nvfp4_woq_bf16: workspace must be on the same XPU device as A");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "nvfp4_woq_bf16: A must be bfloat16");
  TORCH_CHECK(b.scalar_type() == torch::kUInt8, "nvfp4_woq_bf16: B must be uint8 (packed E2M1)");
  TORCH_CHECK(sfmx_b.scalar_type() == torch::kUInt8, "nvfp4_woq_bf16: sfmx_b must be uint8 (UE8M0)");
  TORCH_CHECK(c_b.scalar_type() == torch::kBFloat16, "nvfp4_woq_bf16: c_b must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "nvfp4_woq_bf16: output must be float32");
  TORCH_CHECK(a.dim() == 2, "nvfp4_woq_bf16: A must be flat 2D (sum_m, K)");
  TORCH_CHECK(output.dim() == 2, "nvfp4_woq_bf16: output must be flat 2D (sum_m, N)");
  TORCH_CHECK(b.dim() == 3, "nvfp4_woq_bf16: B must be 3D (E, N, K/2)");
  TORCH_CHECK(sfmx_b.dim() == 3, "nvfp4_woq_bf16: sfmx_b must be 3D (E, K/32, N)");
  TORCH_CHECK(c_b.dim() == 3, "nvfp4_woq_bf16: c_b must be 3D (E, N, K/16)");
  TORCH_CHECK(
      a.is_contiguous() && b.is_contiguous() && sfmx_b.is_contiguous() &&
          c_b.is_contiguous() && output.is_contiguous(),
      "nvfp4_woq_bf16: all tensors must be contiguous");
  TORCH_CHECK(
      problem_sizes.dim() == 2 && problem_sizes.size(1) == 3,
      "nvfp4_woq_bf16: problem_sizes must be (E, 3) int32");
  TORCH_CHECK(
      problem_sizes.scalar_type() == torch::kInt32 &&
          expert_offsets.scalar_type() == torch::kInt32,
      "nvfp4_woq_bf16: problem_sizes / expert_offsets must be int32");

  const int E = static_cast<int>(expert_offsets.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(output.size(1));
  TORCH_CHECK(E > 0, "nvfp4_woq_bf16: expert count must be positive");
  TORCH_CHECK(
      problem_sizes.size(0) == E,
      "nvfp4_woq_bf16: problem_sizes / expert_offsets expert mismatch");
  TORCH_CHECK(
      output.size(0) == a.size(0),
      "nvfp4_woq_bf16: A and output flat row counts must match");
  TORCH_CHECK(
      K % BLK_K == 0,
      "nvfp4_woq_bf16: K must be a multiple of ",
      BLK_K,
      " (got ",
      K,
      ")");
  TORCH_CHECK(
      b.size(0) == E && b.size(1) == N && b.size(2) == K / 2,
      "nvfp4_woq_bf16: B must be (E, N, K/2)");
  TORCH_CHECK(
      sfmx_b.size(0) == E && sfmx_b.size(1) == K / 32 &&
          sfmx_b.size(2) == N,
      "nvfp4_woq_bf16: sfmx_b must be (E, K/32, N)");
  TORCH_CHECK(
      c_b.size(0) == E && c_b.size(1) == N && c_b.size(2) == K / 16,
      "nvfp4_woq_bf16: c_b must be (E, N, K/16)");

  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto ptr_table = torch::empty({4, E}, opts_i64);
  auto cb_ptrs = torch::empty({E}, opts_i64);
  auto cb_ld = torch::empty({E}, opts_i64);
  auto stride_a = torch::empty({E}, opts_i64);
  auto stride_b = torch::empty({E}, opts_i64);
  auto stride_cd = torch::empty({E}, opts_i64);

  const int32_t* offsets_ptr = expert_offsets.data_ptr<int32_t>();
  int64_t* ptrs = ptr_table.data_ptr<int64_t>();
  int64_t* cb_ptrs_ptr = cb_ptrs.data_ptr<int64_t>();
  int64_t* cb_ld_ptr = cb_ld.data_ptr<int64_t>();
  int64_t* stride_a_ptr = stride_a.data_ptr<int64_t>();
  int64_t* stride_b_ptr = stride_b.data_ptr<int64_t>();
  int64_t* stride_cd_ptr = stride_cd.data_ptr<int64_t>();
  const int64_t a_base = reinterpret_cast<int64_t>(a.data_ptr());
  const int64_t b_base = reinterpret_cast<int64_t>(b.data_ptr());
  const int64_t out_base = reinterpret_cast<int64_t>(output.data_ptr());
  const int64_t sfmx_base = reinterpret_cast<int64_t>(sfmx_b.data_ptr());
  const int64_t cb_base = reinterpret_cast<int64_t>(c_b.data_ptr());
  const int64_t a_row_bytes = static_cast<int64_t>(K) * a.element_size();
  const int64_t out_row_bytes = static_cast<int64_t>(N) * output.element_size();
  const int64_t b_expert_bytes = b.stride(0) * b.element_size();
  const int64_t sfmx_expert_bytes = sfmx_b.stride(0) * sfmx_b.element_size();
  const int64_t cb_expert_bytes = c_b.stride(0) * c_b.element_size();
  const int64_t cb_ld_value = c_b.stride(1);

  auto& queue = at::xpu::getCurrentXPUStream(device.index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<NvFp4WoqBf16BuildMetadata>(
        sycl::range<1>(static_cast<size_t>(E)), [=](sycl::id<1> id) {
          const int e = static_cast<int>(id[0]);
          const int64_t ei = static_cast<int64_t>(e);
          const int64_t row = static_cast<int64_t>(offsets_ptr[e]);
          ptrs[0 * E + e] = a_base + row * a_row_bytes;
          ptrs[1 * E + e] = b_base + ei * b_expert_bytes;
          ptrs[2 * E + e] = out_base + row * out_row_bytes;
          ptrs[3 * E + e] = sfmx_base + ei * sfmx_expert_bytes;
          cb_ptrs_ptr[e] = cb_base + ei * cb_expert_bytes;
          cb_ld_ptr[e] = cb_ld_value;
          stride_a_ptr[e] = K;
          stride_b_ptr[e] = K;
          stride_cd_ptr[e] = N;
        });
  });
  // Order the preparation writes before the direct CUTLASS submission.
  queue.ext_oneapi_submit_barrier();

  const int sm_count = at::native::xpu::cached_device_multiprocessor_count(
      static_cast<int>(device.index()));
  const auto tile = at::native::xpu::select_nvfp4_woq_bf16_tile(
      a.size(0), E, N, sm_count);

#define RUN_BF16_GEMM(TYPES)                                                \
  at::native::xpu::run_nvfp4_woq_bf16_gemm<TYPES>(                         \
      E,                                                                    \
      problem_sizes,                                                        \
      ptr_table,                                                            \
      cb_ptrs,                                                              \
      cb_ld,                                                                \
      stride_a,                                                             \
      stride_b,                                                             \
      stride_cd,                                                            \
      workspace,                                                            \
      sm_count,                                                             \
      device)

  switch (tile) {
    case at::native::xpu::NvFp4WoqBf16Tile::M8N128:
      RUN_BF16_GEMM(at::native::xpu::NvFp4WoqBf16M8N128Types);
      break;
    case at::native::xpu::NvFp4WoqBf16Tile::M8N256:
      RUN_BF16_GEMM(at::native::xpu::NvFp4WoqBf16M8N256Types);
      break;
    case at::native::xpu::NvFp4WoqBf16Tile::M32N256:
      RUN_BF16_GEMM(at::native::xpu::NvFp4WoqBf16M32N256Types);
      break;
    case at::native::xpu::NvFp4WoqBf16Tile::M64N256:
      RUN_BF16_GEMM(at::native::xpu::NvFp4WoqBf16M64N256Types);
      break;
    case at::native::xpu::NvFp4WoqBf16Tile::M256N256:
      RUN_BF16_GEMM(at::native::xpu::NvFp4WoqBf16Types);
      break;
  }

#undef RUN_BF16_GEMM
}
