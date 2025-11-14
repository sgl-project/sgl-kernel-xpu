/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Flash Attention execution engine
*/

#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "kernels/chunk_prefill/fmha_fusion.hpp"
#include "kernels/chunk_prefill/tile_scheduler_chunk_prefill.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "kernels/chunk_prefill/xe_chunk_prefill.hpp"
#include "kernels/chunk_prefill/xe_flash_attn_chunk_prefill_epilogue.hpp"
#include "kernels/chunk_prefill/xe_flash_attn_chunk_prefill_softmax_epilogue.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/initialize_block.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/sycl_tensor_fill.h"


namespace runner {
namespace flash_attention {

using namespace cute;

using MMAOperationBF16 = cute::XE_8x16x16_F32BF16BF16F32_TT;
using MMAOperationFP16 = cute::XE_8x16x16_F32F16F16F32_TT;

struct Shape_h64 {
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>;
};

struct Shape_h96 {
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
};

struct Shape_h128 {
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>;
};

struct Shape_h192 {
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
};

// Add a template-based selector for MMA operations
template<typename ElementType>
struct MMAOperationSelector;

template<>
struct MMAOperationSelector<cutlass::bfloat16_t> {
    using type = MMAOperationBF16;
};

template<>
struct MMAOperationSelector<cutlass::half_t> {
    using type = MMAOperationFP16;
};

// Add a template-based selector for shapes
template<int HeadSize>
struct ShapeSelector;

template<>
struct ShapeSelector<64> {
    using type = Shape_h64;
};

template<>
struct ShapeSelector<96> {
    using type = Shape_h96;
};

template<>
struct ShapeSelector<128> {
    using type = Shape_h128;
};

template<>
struct ShapeSelector<192> {
    using type = Shape_h192;
};


/////////////////////////////////////////////////////////////////////
  template <int input_bits, int output_bits> struct TiledCopyConfig;

  template <> struct TiledCopyConfig<8, 32> {
    using GmemTiledCopyQ = cute::XE_2D_U8x8x32_LD_N;
    using GmemTiledCopyK = cute::XE_2D_U8x16x16_LD_T;
    using GmemTiledCopyV = cute::XE_2D_U8x32x32_LD_V;
    using GmemTiledCopyO = cute::XE_2D_U32x8x16_ST_N;
  };

  template <> struct TiledCopyConfig<8, 8> {
    using GmemTiledCopyQ = cute::XE_2D_U8x8x32_LD_N;
    using GmemTiledCopyK = cute::XE_2D_U8x16x16_LD_T;
    using GmemTiledCopyV = cute::XE_2D_U8x32x32_LD_V;
    using GmemTiledCopyO = cute::XE_2D_U8x8x16_ST_N;
  };

  template <> struct TiledCopyConfig<16, 32> {
    using GmemTiledCopyQ = cute::XE_2D_U16x8x32_LD_N;
    using GmemTiledCopyK = cute::XE_2D_U16x16x16_LD_T;
    using GmemTiledCopyV = cute::XE_2D_U16x16x32_LD_V;
    using GmemTiledCopyO = cute::XE_2D_U32x8x16_ST_N;
  };

  template <> struct TiledCopyConfig<16, 16> {
    using GmemTiledCopyQ = cute::XE_2D_U16x8x32_LD_N;
    using GmemTiledCopyK = cute::XE_2D_U16x16x16_LD_T;
    using GmemTiledCopyV = cute::XE_2D_U16x16x32_LD_V;
    using GmemTiledCopyO = cute::XE_2D_U16x8x16_ST_N;
  };
  
  template <class, class> class convert_fp8_to_fp16_name;

  template <typename SrcT, typename DstT>
  void convert_fp8_to_fp16(const SrcT* d_src, DstT* d_dst, size_t size) {
    compat::get_default_queue().parallel_for<convert_fp8_to_fp16_name<SrcT, DstT>>(size, [=](auto indx) {
      d_dst[indx] = static_cast<DstT>(d_src[indx]);
    }).wait();
  }


/////////////////////////////////////////////////////////////////////

using LayoutQ = cutlass::layout::RowMajor;
using LayoutK = cutlass::layout::ColumnMajor;
using LayoutV = cutlass::layout::RowMajor;
using LayoutO = cutlass::layout::RowMajor;

/////////////////////////////////////////////////////////////////////

template<typename ElementInputType, typename ElementAccumulatorType, typename ElementOutputType,  
        typename TileShapeQK, typename TileShapePV, typename TileShapeOutput, typename SubgroupLayout, 
        typename MMAOperation, int PipelineStages, bool CausalMask, bool isVarLen, bool PagedKV, 
        bool LocalMask = false, bool Sink = false, typename ElementSink = bfloat16_t, class Scheduler= void>
struct XE_Flash_Attention {
  using ElementAccumulator = ElementAccumulatorType;
  using ElementComputeEpilogue = ElementAccumulatorType;
  using ElementInputQ = ElementInputType;
  using ElementInputKV = ElementInputType;
  using ElementOutput = ElementOutputType;

  using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int, int>;
  using ProblemShapeVarlen = cute::tuple<int, int, int, cutlass::fmha::collective::VariableLength,
                                         cutlass::fmha::collective::VariableLength, cutlass::fmha::collective::VariableLength, int, int>;
  using ProblemShapeType = std::conditional_t<isVarLen, ProblemShapeVarlen, ProblemShapeRegular>;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using GmemTiledCopyQ = typename TiledCopyConfig<cute::sizeof_bits_v<ElementInputQ>, cute::sizeof_bits_v<ElementOutput>>::GmemTiledCopyQ;
  using GmemTiledCopyK = typename TiledCopyConfig<cute::sizeof_bits_v<ElementInputKV>, cute::sizeof_bits_v<ElementOutput>>::GmemTiledCopyK;
  using GmemTiledCopyV = typename TiledCopyConfig<cute::sizeof_bits_v<ElementInputKV>, cute::sizeof_bits_v<ElementOutput>>::GmemTiledCopyV;
  using GmemTiledCopyStore = typename TiledCopyConfig<cute::sizeof_bits_v<ElementInputQ>, cute::sizeof_bits_v<ElementOutput>>::GmemTiledCopyO;
  using CollectiveEpilogue = cutlass::flash_attention::collective::FlashChunkPrefillEpilogue<
        Sink,
        EpilogueDispatchPolicy,
        MMAOperation,
        TileShapeOutput,
        SubgroupLayout,
        ElementComputeEpilogue,
        ElementOutput, 
        cutlass::gemm::TagToStrideC_t<LayoutO>,
        ElementOutput,
        GmemTiledCopyStore,
        ElementSink>;
  using CollectiveSoftmaxEpilogue = cutlass::flash_attention::collective::FlashChunkPrefillSoftmaxEpilogue<
        CausalMask, LocalMask, EpilogueDispatchPolicy, ElementAccumulator>;

  // Mainloop
  using CollectiveMainloop = cutlass::flash_attention::collective::FlashChunkPrefillMma<
        GEMMDispatchPolicy,
        ProblemShapeType,
        ElementInputQ,
        cutlass::gemm::TagToStrideA_t<LayoutQ>,
        ElementInputKV,
        cutlass::gemm::TagToStrideB_t<LayoutK>,
        ElementInputKV,
        cutlass::gemm::TagToStrideB_t<LayoutV>,
        MMAOperation,
        TileShapeQK,
        TileShapePV,
        SubgroupLayout,
        GmemTiledCopyQ, // Q
        GmemTiledCopyK, // K
        GmemTiledCopyV, // V,
        CausalMask,
        LocalMask,
        PagedKV>;

  using Kernel = cutlass::flash_attention::kernel::FMHAPrefillChunk<ProblemShapeType, CollectiveMainloop,
                                                      CollectiveSoftmaxEpilogue, CollectiveEpilogue, Scheduler>;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel>
class KernelCur {};

namespace detail {

template <typename FlashAttentionKernel>
struct EngineImpl {

  using StrideQ = typename FlashAttentionKernel::StrideQ;
  using StrideK = typename FlashAttentionKernel::StrideK;
  using StrideV = typename FlashAttentionKernel::StrideV;
  using StrideO = typename FlashAttentionKernel::StrideO;

  using ElementQ = typename FlashAttentionKernel::ElementQ;
  using ElementK = typename FlashAttentionKernel::ElementK;
  using ElementV = typename FlashAttentionKernel::ElementV;
  using ElementAcc = typename FlashAttentionKernel::ElementAccumulator;
  using ElementSink = typename FlashAttentionKernel::ElementSink;

  using CollectiveMainloop = typename FlashAttentionKernel::CollectiveMainloop;
  using CollectiveEpilogue = typename FlashAttentionKernel::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename FlashAttentionKernel::ProblemShape;
  static constexpr bool HasCausalMask = CollectiveMainloop::CausalMask;
  static constexpr bool isVarLen = CollectiveMainloop::is_var_len;

  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideK stride_K_cache;
  StrideV stride_V_cache;
  StrideO stride_O;

  std::vector<int> cumulative_seqlen_q;
  std::vector<int> cumulative_seqlen_kv;
  cutlass::DeviceAllocation<int> device_cumulative_seqlen_q;
  cutlass::DeviceAllocation<int> device_cumulative_seqlen_kv;
  cutlass::DeviceAllocation<ElementQ> block_Q;
  cutlass::DeviceAllocation<ElementK> block_K;
  cutlass::DeviceAllocation<ElementV> block_V;
  cutlass::DeviceAllocation<ElementOutput> block_O;
  cutlass::DeviceAllocation<ElementOutput> block_ref_O;

  //
  // Methods
  //
  template <typename T>
  static constexpr bool is_fp8_v = cute::is_any_of_v<T, cute::float_e5m2_t, cute::float_e4m3_t>;

  template <typename Tin> inline auto in_memory(cutlass::DeviceAllocation<Tin>& in) {
    using outType = cute::conditional_t<is_fp8_v<Tin>, half_t, Tin>;
    if constexpr(is_fp8_v<Tin>) {
      cutlass::DeviceAllocation<outType> out(in.size());
      convert_fp8_to_fp16<Tin, outType>(in.get(), out.get(), in.size());
      return out;
    } else { 
      return in;
    };
  }

  template <typename Params,class ProblemShape>
  auto initialize_varlen(const Params& params, ProblemShape& problem_size) {
    ProblemShape problem_size_for_init = problem_size;
    get<0>(problem_size_for_init) = 1;  // concentrated batch
    get<3>(problem_size_for_init) = params.total_q;
    get<4>(problem_size_for_init) = params.total_knew;
    get<5>(problem_size_for_init) = params.total_k;

    ProblemShapeType problem_size_for_launch;

    get<0>(problem_size_for_launch) = get<0>(problem_size);
    get<1>(problem_size_for_launch) = get<1>(problem_size);
    get<2>(problem_size_for_launch) = get<2>(problem_size);
    get<3>(problem_size_for_launch) = cutlass::fmha::collective::VariableLength{params.seqlen_q, params.total_q};
    get<4>(problem_size_for_launch) = cutlass::fmha::collective::VariableLength{params.seqlen_knew, params.total_knew};
    get<5>(problem_size_for_launch) = cutlass::fmha::collective::VariableLength{params.seqlen_k, params.total_k};
    get<6>(problem_size_for_launch) = get<6>(problem_size);
    get<7>(problem_size_for_launch) = get<7>(problem_size);

    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  template <typename Params>
  ProblemShapeType initialize(const Params& params) {
    auto problem_shape_in = cute::make_tuple(
        params.b,    // batch
        params.h,    // num_heads_q
        params.h_k,  // num_heads_kv
        params.seqlen_q,
        params.seqlen_knew,
        params.seqlen_k,
        params.d,
        params.dv);

    ProblemShapeType problem_shape;
    decltype(problem_shape_in) problem_size;

    if constexpr (isVarLen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(params, problem_shape_in);
      problem_size = problem_shape_init;
      problem_shape = problem_shape_launch;
    } else {
      problem_size = problem_shape_in;
      problem_shape = problem_shape_in;
    }

    auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo] =
        problem_size;
    auto group_q_size = num_heads_q / num_heads_kv;
    auto group_q_num = num_heads_q / group_q_size;

    stride_Q =
        cutlass::make_cute_packed_stride(StrideQ{}, cute::make_shape(seq_len_qo, num_heads_q * head_size_qk, batch));
    stride_K =
        cutlass::make_cute_packed_stride(StrideK{}, cute::make_shape(seq_len_kv, num_heads_kv * head_size_qk, batch));
    stride_V =
        cutlass::make_cute_packed_stride(StrideV{}, cute::make_shape(head_size_vo * num_heads_kv, seq_len_kv, batch));

    stride_K_cache = cutlass::make_cute_packed_stride(
        StrideK{}, cute::make_shape(seq_len_kv_cache, num_heads_kv * head_size_qk, batch));
    stride_V_cache = cutlass::make_cute_packed_stride(
        StrideV{}, cute::make_shape(head_size_vo * head_size_qk, seq_len_kv_cache, batch * num_heads_kv));
    stride_O = cutlass::make_cute_packed_stride(
        StrideQ{}, cute::make_shape(seq_len_qo * group_q_size, group_q_num * head_size_vo, batch));

    if constexpr (isVarLen) {
      get<3>(problem_shape).cumulative_length = params.cu_seqlens_q;
      get<4>(problem_shape).cumulative_length = params.cu_seqlens_knew;
      get<5>(problem_shape).cumulative_length = params.cu_seqlens_k;
    }

    return problem_shape;
  }

  bool sufficient() {
    // check device properties
    // Currently, we assume that all Intel Xe devices support Flash Attention
    return true;
  }

  template<typename Params>
  bool run(Params params)
  {
    // Fail test if insufficient device
    if (!sufficient()) {
      CUTLASS_TRACE_HOST("EngineImpl::run: Test failed due to insufficient device");
      std::cout << "Test failed due to insufficient device." << std::endl;
      return false;
    }
    ProblemShapeType problem_size = this->initialize(params);
    //
    // Initialize the Flash attention operator
    //
    cutlass::KernelHardwareInfo hw_info;

    typename FlashAttentionKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {// static_cast<const ElementQ*>(params.q_ptr),
         static_cast<const ElementQ*>(params.q_ptr),
         stride_Q,
         //  static_cast<const ElementK*>(params.knew_ptr),
         //  stride_K,
         //  static_cast<const ElementV*>(params.vnew_ptr),
         //  stride_V,
         static_cast<const ElementV*>(params.k_ptr),
         stride_K_cache,
         static_cast<const ElementV*>(params.v_ptr),
         stride_V_cache,
         params.page_table,
         params.page_size,
         params.max_num_pages_per_seq,
         params.window_size_left,
         params.window_size_right},
        {(ElementQ)params.scale_softmax},
        {static_cast<const ElementOutput*>(params.o_ptr),
         stride_O,
         static_cast<const ElementSink*>(params.sink_softmax)},
        hw_info};

    size_t workspace_size = FlashAttentionKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (!FlashAttentionKernel::can_implement(arguments)) {
      std::cerr << "This case is not supported." << "\n";
      return false;
    }

    //
    // Run Flash attention
    //
    auto params_kernel = FlashAttentionKernel::to_underlying_arguments(arguments, workspace.get());
    auto const block = FlashAttentionKernel::get_block_shape();
    auto const grid = FlashAttentionKernel::get_grid_shape(params_kernel);

    // configure smem size and carveout
    int smem_size = FlashAttentionKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    using namespace compat::experimental;
    compat::experimental::launch_properties launch_props{
        sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<FlashAttentionKernel::DispatchPolicy::SubgroupSize>
    };
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};

    sycl::ext::oneapi::experimental::launch_config config(policy.get_range(), policy.get_launch_properties());
    auto cgf = [&](::sycl::handler& cgh) {
      auto KernelFunctor =
          compat::experimental::detail::build_kernel_functor<cutlass::device_kernel<FlashAttentionKernel>>(
              cgh, policy, params_kernel);
      sycl::ext::oneapi::experimental::detail::
          LaunchConfigAccess<sycl::nd_range<3>, decltype(policy.get_launch_properties())>
              ConfigAccess(config);
      cgh.parallel_for<KernelCur<FlashAttentionKernel>>(
          ConfigAccess.getRange(), ConfigAccess.getProperties(), KernelFunctor);
    };
    auto stream = at::xpu::getCurrentXPUStream();
    auto q = stream.queue();
    q.submit(cgf);

    try {
      compat::wait_and_throw();
    } catch (std::exception const &e) {
      std::cerr << "Error at Kernel Sync: " << e.what() << "\n";
      return false;
    }
    return true;
  }
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename FlashAttention
>
struct Engine3x {
  detail::EngineImpl<FlashAttention> impl_;

  //
  // Methods
  //
  Engine3x() : impl_() {}

  template <typename Params>
  bool run(Params params) {
    return impl_.run(params);
  }
};

template <typename FlashAttention, typename Params>
void RunFlashAttention(Params params, std::string config="default") {
  Engine3x<FlashAttention> engine;
  bool passed = true;
  try {
    passed = engine.run(params);
  }
  catch (std::exception const& e) {
    std::cerr << "Executing: engine.run {"
      << "batch: " << params.b << ", num_heads_q: " << params.h << ", num_heads_kv: " << params.h_k
      << ", seq_len_qo: " << params.seqlen_q << ", seq_len_kv: " << params.seqlen_k << ", seq_len_knew: " << params.seqlen_knew
      << ", head_size_vo: " << params.dv << ", head_size_qk: " << params.d
      << "} threw an exception: " << e.what() << "\n";
    throw;
  }
  catch (...) {
    std::cerr << "Executing: engine.run {"
      << "batch: " << params.b << ", num_heads_q: " << params.h << ", num_heads_kv: " << params.h_k
      << ", seq_len_qo: " << params.seqlen_q << ", seq_len_kv: " << params.seqlen_k << ", seq_len_knew: " << params.seqlen_knew
      << ", head_size_vo: " << params.dv << ", head_size_qk: " << params.d
      << "} threw an exception (unknown)" << "\n";
    throw;
  }
  return;
}

} // namespace flash_attention
} // namespace runner

/////////////////////////////////////////////////////////////////////////////////////////////////
