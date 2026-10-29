#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdio>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/kernel_hardware_info.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <limits>
#include <optional>
#include <type_traits>

#include "comm/common.h"
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
#include <cutlass/util/sycl_event_manager.hpp>
#endif
#include "kernels/flash_attention_v2/xe35/collective/xe_fmha_fwd_epilogue.hpp"
#include "kernels/flash_attention_v2/xe35/collective/xe_fmha_fwd_mainloop.hpp"
#include "kernels/flash_attention_v2/xe35/kernel/xe_fmha_fwd_kernel.hpp"
#include "kernels/flash_attention_v2/xe35/kernel/xe_reduce_split_k.hpp"
#include "kernels/flash_attention_v2/xe35/kernel/xe_tile_scheduler.hpp"

#define SYCL_INTEL_TARGET 35

using namespace cute;

namespace xe35_fmha {

namespace detail {
struct SplitKVCache {
  at::Tensor temp_out;
  at::Tensor exp_sums;
  at::Tensor max_logits;

  void
  ensure(int64_t out_elts, int64_t lse_elts, const at::TensorOptions& out_opts, const at::TensorOptions& lse_opts) {
    if (!temp_out.defined() || temp_out.numel() < out_elts || temp_out.dtype() != out_opts.dtype() ||
        temp_out.device() != out_opts.device()) {
      temp_out = at::empty({out_elts}, out_opts);
    }
    if (!exp_sums.defined() || exp_sums.numel() < lse_elts || exp_sums.dtype() != lse_opts.dtype() ||
        exp_sums.device() != lse_opts.device()) {
      exp_sums = at::empty({lse_elts}, lse_opts);
    }
    if (!max_logits.defined() || max_logits.numel() < lse_elts || max_logits.dtype() != lse_opts.dtype() ||
        max_logits.device() != lse_opts.device()) {
      max_logits = at::empty({lse_elts}, lse_opts);
    }
  }
};

inline SplitKVCache& get_split_kv_cache() {
  thread_local SplitKVCache cache;
  return cache;
}
}  // namespace detail

template <class Element, class OutputElement = Element>
struct Xe3FmhaConfig {
  using ElementInput = Element;
  using ElementOutput = OutputElement;
  using ElementScale = float;

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;
  using StrideScale = Stride<_1, int, int, int>;

  template <typename T, typename S>
  static auto make_dummy_tensor(T val, S stride) {
    return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<S>>(1), stride));
  }

  template <
      bool Causal,
      bool LocalMask,
      bool Sink,
      int PipelineStages,
      bool GqaFusion,
      bool Paged,
      class TileShapeQK,
      class TileShapePV,
      class TileShapeOut,
      class SubgroupLayoutQK>
  static void
  run(const at::Tensor& q_dense,
      const at::Tensor& k_pool,
      const at::Tensor& v_pool,
      const at::Tensor& page_table,
      int num_pages_per_seq,
      const at::Tensor& cu_seqlens_q,
      const at::Tensor& cu_seqlens_k_cache,
      at::Tensor& out_dense,
      int batch,
      int num_heads_q,
      int num_heads_kv,
      int max_seqlen_q,
      int max_seqlen_kv_cache,
      int head_dim,
      int page_size,
      const float* q_descale,
      const float* k_descale,
      const float* v_descale,
      float softmax_scale,
      int window_size_left,
      int window_size_right,
      const ElementInput* sm_sink,
      int num_kv_splits = 1) {
    static constexpr bool BlockScale = false;
    static constexpr bool F8kvF16mma = false;
    static constexpr bool PerTensorScale = sizeof_bits_v<ElementInput> == 8;
    // Paged KV is always cached; non-paged runs as plain attention (CachedKV=false)
    // with the KV placed in the "new" slot, which is the upstream-tested path.
    static constexpr bool CachedKV = Paged;
    // Paged (PagedKV=true) uses page_table indirection; non-paged (Paged=false)
    // reads contiguous ragged KV addressed by the cumulative cu_seqlens_k_cache.
    static constexpr bool PagedKV = Paged;

    constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
    using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementInput>;
    using MMAOperationPV = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementInput>;
    using SubgroupLayoutPV = decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV =
        typename TiledMMAHelper<MMA_Atom<MMAOperationPV>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

    static_assert(
        get<0>(TileShapeOut{}) == get<0>(TileShapePV{}), "Output tile and P*V tile must match on Q dimension");
    static constexpr int VTiles = get<1>(TileShapeOut{}) / get<1>(TileShapePV{});

    using TensorQ = decltype(make_dummy_tensor(ElementInput{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementInput{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementInput{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementOutput{}, StrideO{}));
    using TensorScaleQ = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));
    using TensorScaleK = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));
    using TensorScaleV = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));

    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy,
        Causal,
        LocalMask,
        BlockScale,
        F8kvF16mma,
        PerTensorScale,
        CachedKV,
        PagedKV,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        TensorScaleQ,
        TensorScaleK,
        TensorScaleV,
        TensorK,
        TensorV,
        void,
        void,
        void,
        void,
        void>;

    using CollectiveEpilogue =
        cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOut, TensorO, void, Sink>;
    using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<true>;
    using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false, Causal, GqaFusion>;
    using FMHAKernel =
        cutlass::fmha::kernel::XeFMHAFwdKernel<ProblemShape, CollectiveMainloop, CollectiveEpilogue, Scheduler>;
    using SplitScheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false, Causal, true>;
    using SplitKernel = cutlass::fmha::kernel::
        XeFMHAFwdKernel<ProblemShape, CollectiveMainloop, CollectiveEpilogue, SplitScheduler, true>;
    using ReduceKernel = cutlass::reduction::kernel::
        ReduceSplitK<ProblemShape, cutlass::fmha::kernel::XeReduceSplitKTileScheduler, SplitKernel>;

    const int q_seq_stride = static_cast<int>(q_dense.stride(0));
    const int q_head_stride = static_cast<int>(q_dense.stride(1));
    const int kv_seq_stride = static_cast<int>(k_pool.stride(0));
    const int kv_head_stride = static_cast<int>(k_pool.stride(1));
    const int o_seq_stride = static_cast<int>(out_dense.stride(0));
    const int o_head_stride = static_cast<int>(out_dense.stride(1));
    typename FMHAKernel::StrideQ stride_q{q_seq_stride, {}, q_head_stride, 0};
    typename FMHAKernel::StrideK stride_k{kv_seq_stride, {}, kv_head_stride, 0};
    typename FMHAKernel::StrideV stride_v{{}, kv_seq_stride, kv_head_stride, 0};
    typename FMHAKernel::StrideO stride_o{o_seq_stride, {}, o_head_stride, 0};

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // page_table is only valid in the paged case; keep it null otherwise so the
    // non-paged mainloop never dereferences an absent table.
    const int* page_table_ptr = nullptr;
    if constexpr (PagedKV) {
      page_table_ptr = static_cast<const int*>(page_table.data_ptr());
    }

    // Plain attention (CachedKV=false) carries the KV in the "new" slot; paged
    // carries it in the "cache" slot. The unused slot stays empty.
    int* const cu_kv = cu_seqlens_k_cache.data_ptr<int>();
    const int kv_total = static_cast<int>(k_pool.size(0));
    const int new_max = Paged ? 0 : max_seqlen_kv_cache;
    const int new_total = Paged ? 0 : kv_total;
    int* const new_cu = Paged ? nullptr : cu_kv;
    const int cache_max = Paged ? max_seqlen_kv_cache : 0;
    const int cache_total = Paged ? kv_total : 0;
    int* const cache_cu = Paged ? cu_kv : nullptr;

    typename FMHAKernel::Arguments arguments{
        {
            {batch,
             num_heads_q,
             num_heads_kv,
             {max_seqlen_q, static_cast<int>(q_dense.size(0)), cu_seqlens_q.data_ptr<int>(), nullptr},
             {new_max, new_total, new_cu, nullptr},
             {cache_max, cache_total, cache_cu, nullptr},
             head_dim,
             head_dim},
            static_cast<const ElementInput*>(q_dense.data_ptr()),
            stride_q,
            static_cast<const ElementInput*>(k_pool.data_ptr()),
            stride_k,
            static_cast<const ElementInput*>(v_pool.data_ptr()),
            stride_v,
            static_cast<ElementOutput*>(out_dense.data_ptr()),
            stride_o,
            q_descale,
            {},
            k_descale,
            {},
            v_descale,
            {},
            1.0f,
            1.0f,
            1.0f,
            32,
            static_cast<const ElementInput*>(k_pool.data_ptr()),
            stride_k,
            static_cast<const ElementInput*>(v_pool.data_ptr()),
            stride_v,
            sm_sink,
        },
        {
            softmax_scale,
            page_table_ptr,
            page_size,
            num_pages_per_seq,
            static_cast<int>(k_pool.size(0)),
            window_size_left,
            window_size_right,
        },
        {},
        hw_info,
    };

    if constexpr (!Sink) {
      if (num_kv_splits > 1) {
        TORCH_CHECK(
            num_kv_splits <= SplitKernel::max_num_kv_splits,
            "flash_attentionXe35: num_kv_splits exceeds the split-KV kernel limit");
        const auto shape = arguments.kernel.shape;
        const int64_t total_out_elts = static_cast<int64_t>(q_dense.size(0)) * num_heads_q * head_dim * num_kv_splits;
        const int64_t total_lse_elts = static_cast<int64_t>(q_dense.size(0)) * num_heads_q * num_kv_splits;
        auto& cache = detail::get_split_kv_cache();
        cache.ensure(total_out_elts, total_lse_elts, out_dense.options(), q_dense.options().dtype(at::kFloat));
        auto temp_out = cache.temp_out.narrow(0, 0, total_out_elts);
        auto exp_sums = cache.exp_sums.narrow(0, 0, total_lse_elts);
        auto max_logits = cache.max_logits.narrow(0, 0, total_lse_elts);

        auto split_stride_o = cutlass::make_cute_packed_stride(
            typename SplitKernel::StrideO{},
            cute::make_shape(max_seqlen_q, head_dim, num_heads_q * num_kv_splits, batch));
        auto split_stride_lse = cutlass::make_cute_packed_stride(
            typename SplitKernel::StrideO{}, cute::make_shape(max_seqlen_q, num_kv_splits, num_heads_q, batch));

        typename SplitKernel::Arguments split_arguments{
            {shape,
             static_cast<const ElementInput*>(q_dense.data_ptr()),  // Q
             stride_q,                                              // dQ
             static_cast<const ElementInput*>(k_pool.data_ptr()),   // K
             stride_k,                                              // dK
             static_cast<const ElementInput*>(v_pool.data_ptr()),   // V
             stride_v,                                              // dV
             static_cast<ElementOutput*>(temp_out.data_ptr()),      // O
             stride_o,                                              // dO
             q_descale,                                             // scaleQ
             {},                                                    // dScaleQ
             k_descale,                                             // scaleK
             {},                                                    // dScaleK
             v_descale,                                             // scaleV
             {},                                                    // dScaleV
             1.0f,                                                  // scale_q
             1.0f,                                                  // scale_k
             1.0f,                                                  // scale_v
             32,                                                    // group_size
             static_cast<const ElementInput*>(k_pool.data_ptr()),   // K_cache
             stride_k,                                              // dK_cache
             static_cast<const ElementInput*>(v_pool.data_ptr()),   // V_cache
             stride_v,                                              // dV_cache
             nullptr,                                               // sm_sink
             static_cast<ElementOutput*>(temp_out.data_ptr()),      // Oaccum
             split_stride_o,                                        // dOaccum
             exp_sums.data_ptr<float>(),                            // exp_sums
             split_stride_lse,                                      // dExp_sums
             max_logits.data_ptr<float>(),                          // max_logits
             split_stride_lse},                                     // dMax_logits
            arguments.mainloop,
            arguments.epilogue,
            hw_info,
            num_kv_splits};
        typename ReduceKernel::Arguments reduce_arguments{
            {shape,
             static_cast<ElementOutput*>(out_dense.data_ptr()),
             stride_o,
             static_cast<const ElementOutput*>(temp_out.data_ptr()),
             stride_o,
             exp_sums.data_ptr<float>(),
             stride_o,
             max_logits.data_ptr<float>(),
             stride_o,
             window_size_left,
             1.0f,
             v_descale},
            hw_info,
            num_kv_splits};

        typename SplitKernel::Params split_params = SplitKernel::to_underlying_arguments(split_arguments, nullptr);
        typename ReduceKernel::Params reduce_params = ReduceKernel::to_underlying_arguments(reduce_arguments, nullptr);
        namespace syclex = sycl::ext::oneapi::experimental;
        namespace intelex = sycl::ext::intel::experimental;
        auto launch_split = [&](auto kernel_tag, auto params) {
          using Kernel = decltype(kernel_tag);
          compat::dim3 const block = Kernel::get_block_shape();
          compat::dim3 const grid = Kernel::get_grid_shape(params);
          compat::experimental::launch_properties launch_props{
              syclex::work_group_scratch_size(Kernel::SharedStorageSize)};
          compat::experimental::kernel_properties kernel_props{
              syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<512>};
          compat::experimental::launch_policy policy{
              compat::dim3(grid.x, grid.y, grid.z),
              compat::dim3(block.x, block.y, block.z),
              launch_props,
              kernel_props};
          return compat::experimental::launch<cutlass::device_kernel<Kernel>, Kernel, false>(
              policy, at::xpu::getCurrentXPUStream().queue(), params);
        };
        launch_split(SplitKernel{}, split_params);
        launch_split(ReduceKernel{}, reduce_params);
        return;
      }
    }

    TORCH_CHECK(FMHAKernel::can_implement(arguments), "flash_attentionXe35: invalid cutlass FMHA arguments");

    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    // Zero-init: the split-reduction workspace is otherwise read while partially
    // uninitialized, giving non-deterministic results across launches.
    at::Tensor workspace = at::zeros({static_cast<long>(workspace_size)}, q_dense.options().dtype(at::kByte));
    auto* workspace_ptr = workspace_size > 0 ? workspace.data_ptr() : nullptr;
    auto status = FMHAKernel::initialize_workspace(arguments, workspace_ptr);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "flash_attentionXe35: FMHA workspace initialization failed");

    auto params = FMHAKernel::to_underlying_arguments(arguments, workspace_ptr);

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    compat::dim3 const block = FMHAKernel::get_block_shape();
    compat::dim3 const grid = FMHAKernel::get_grid_shape(params);
    const int smem_size = FMHAKernel::SharedStorageSize;
    compat::experimental::launch_properties launch_props{syclex::work_group_scratch_size(smem_size)};
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<512>};
    compat::experimental::launch_policy policy{
        compat::dim3(grid.x, grid.y, grid.z), compat::dim3(block.x, block.y, block.z), launch_props, kernel_props};
    sycl::queue& queue = at::xpu::getCurrentXPUStream().queue();
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
    auto event = compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, queue, params);
    EventManager::getInstance().addEvent(event);
#else
    compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel, false>(policy, queue, params);
#endif
  }
};

template <int HeadDim, bool UseDecode, bool Paged, class Element = cutlass::bfloat16_t, class OutputElement = Element>
void dispatch_xe3_kernel(
    const at::Tensor& q_dense,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_table,
    int num_pages_per_seq,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k_cache,
    at::Tensor& out_dense,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int max_seqlen_q,
    int max_seqlen_kv_cache,
    int page_size,
    const float* q_descale,
    const float* k_descale,
    const float* v_descale,
    float softmax_scale,
    bool is_causal,
    bool is_local,
    int window_size_left,
    int window_size_right,
    const cutlass::bfloat16_t* sm_sink,
    bool use_sink,
    int num_kv_splits = 1) {
  const int gqa_group = num_heads_kv > 0 ? num_heads_q / num_heads_kv : 1;
  const int total_rows = gqa_group * max_seqlen_q;

  // Paged is a compile-time parameter so each translation unit only instantiates
  // one KV-layout variant; paged and non-paged are generated as separate .cpp.
#define XE35_INVOKE(CAUSAL, LOCAL, SINKV, STAGES, GQA, QK, PV, OUT, SGL, SINKPTR)                                  \
  Xe3FmhaConfig<Element, OutputElement>::template run<CAUSAL, LOCAL, SINKV, STAGES, GQA, Paged, QK, PV, OUT, SGL>( \
      q_dense,                                                                                                     \
      k_pool,                                                                                                      \
      v_pool,                                                                                                      \
      page_table,                                                                                                  \
      num_pages_per_seq,                                                                                           \
      cu_seqlens_q,                                                                                                \
      cu_seqlens_k_cache,                                                                                          \
      out_dense,                                                                                                   \
      batch,                                                                                                       \
      num_heads_q,                                                                                                 \
      num_heads_kv,                                                                                                \
      max_seqlen_q,                                                                                                \
      max_seqlen_kv_cache,                                                                                         \
      HeadDim,                                                                                                     \
      page_size,                                                                                                   \
      q_descale,                                                                                                   \
      k_descale,                                                                                                   \
      v_descale,                                                                                                   \
      softmax_scale,                                                                                               \
      window_size_left,                                                                                            \
      window_size_right,                                                                                           \
      SINKPTR,                                                                                                     \
      num_kv_splits)

#define XE35_RUN(CAUSAL, LOCAL, STAGES, GQA, QK, PV, OUT, SGL)                     \
  do {                                                                             \
    if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {                  \
      if (use_sink)                                                                \
        XE35_INVOKE(CAUSAL, LOCAL, true, STAGES, GQA, QK, PV, OUT, SGL, sm_sink);  \
      else                                                                         \
        XE35_INVOKE(CAUSAL, LOCAL, false, STAGES, GQA, QK, PV, OUT, SGL, nullptr); \
    } else {                                                                       \
      TORCH_CHECK(!use_sink, "flash_attentionXe35: FP8 does not support sinks");   \
      XE35_INVOKE(CAUSAL, LOCAL, false, STAGES, GQA, QK, PV, OUT, SGL, nullptr);   \
    }                                                                              \
  } while (false)

  if constexpr (HeadDim == 256) {
    if constexpr (UseDecode) {
      using ShapeQK = Shape<_128, _64, _64>;
      using ShapePV = Shape<_128, _64, _64>;
      using ShapeOut = Shape<_128, _256>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    }

    if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
      using ShapeQK = Shape<_128, _64, _64>;
      using ShapePV = Shape<_128, _32, _64>;
      using ShapeOut = Shape<_128, _256>;
      using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    } else {
      // FP8 HeadDim=256: 8-subgroup architecture (128x64x64 QK and 128x64x64 PV)
      // Maintains 100% occupancy without register spilling, delivering 380+ TFLOPS on CRI
      using ShapeQK = Shape<_128, _64, _64>;
      using ShapePV = Shape<_128, _64, _64>;
      using ShapeOut = Shape<_128, _256>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;

      // Adaptive low-wave tile for short Q (<=1024) to ensure sufficient workgroups
      using ShapeQK_Small = Shape<_64, _64, _64>;
      using ShapePV_Small = Shape<_64, _64, _64>;
      using ShapeOut_Small = Shape<_64, _256>;
      using SubgroupLayoutQK_Small = Layout<Shape<_8, _1, _1>>;

      const int num_xe_cores = 32;
      const int num_q_tiles_128 = (max_seqlen_q + 127) / 128;
      const int total_wgs = num_q_tiles_128 * num_heads_q * batch;
      const bool use_small = (total_wgs < 2 * num_xe_cores) && (max_seqlen_q <= 1024);

      if (is_local) {
        if (use_small)
          XE35_RUN(false, true, 2, false, ShapeQK_Small, ShapePV_Small, ShapeOut_Small, SubgroupLayoutQK_Small);
        else
          XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      } else if (is_causal) {
        if (use_small)
          XE35_RUN(true, false, 2, false, ShapeQK_Small, ShapePV_Small, ShapeOut_Small, SubgroupLayoutQK_Small);
        else
          XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      } else {
        if (use_small)
          XE35_RUN(false, false, 2, false, ShapeQK_Small, ShapePV_Small, ShapeOut_Small, SubgroupLayoutQK_Small);
        else
          XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      }
      return;
    }
  }

  if constexpr (HeadDim == 128 || ((HeadDim == 72 || HeadDim == 96) && !Paged)) {
    if constexpr (UseDecode) {
      using ShapeQK = Shape<_128, _64, _64>;
      using ShapePV = Shape<_128, _64, _64>;
      using ShapeOut = Shape<_128, _128>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    }

    if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
      using ShapeQK = Shape<_128, _64, _64>;
      using ShapePV = Shape<_128, _64, _64>;
      using ShapeOut = Shape<_128, _128>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local) {
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      } else if (is_causal) {
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      } else {
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      }
      return;
    } else {
      // FP8 HeadDim=128: reference CRI configuration from cutlass (06_xe_fmha_fwd_prefill_hdim128.cpp)
      // 32-wide V tiles halve per-subgroup V fragments, reducing register pressure on CRI
      using ShapeQK = Shape<_256, _64, _128>;
      using ShapePV = Shape<_256, _32, _64>;
      using ShapeOut = Shape<_256, _128>;
      using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;

      // Adaptive low-wave tile for short Q or few heads (ensures >=2 waves for good saturation)
      using ShapeQK_Small = Shape<_128, _64, _128>;
      using ShapePV_Small = Shape<_128, _32, _64>;
      using ShapeOut_Small = Shape<_128, _128>;
      using SubgroupLayoutQK_Small = Layout<Shape<_8, _1, _1>>;

      const int num_xe_cores = 32;
      const int num_q_tiles_256 = (max_seqlen_q + 255) / 256;
      const int total_wgs_256 = num_q_tiles_256 * num_heads_q * batch;
      const bool use_small = (total_wgs_256 < 2 * num_xe_cores) && (max_seqlen_q <= 2048);

      if (is_local) {
        if (use_small)
          XE35_RUN(false, true, 2, false, ShapeQK_Small, ShapePV_Small, ShapeOut_Small, SubgroupLayoutQK_Small);
        else
          XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      } else if (is_causal) {
        if (use_small)
          XE35_RUN(true, false, 2, false, ShapeQK_Small, ShapePV_Small, ShapeOut_Small, SubgroupLayoutQK_Small);
        else
          XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      } else {
        if (use_small)
          XE35_RUN(false, false, 2, false, ShapeQK_Small, ShapePV_Small, ShapeOut_Small, SubgroupLayoutQK_Small);
        else
          XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      }
      return;
    }
  }

  if constexpr (HeadDim == 64) {
    if constexpr (UseDecode) {
      if (use_sink) {
        using SinkShapeQK = Shape<_128, _64, _32>;
        using SinkShapePV = Shape<_128, _32, _64>;
        using SinkShapeOut = Shape<_128, _64>;
        using SinkSubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
        if (is_local) {
          XE35_RUN(false, true, 2, false, SinkShapeQK, SinkShapePV, SinkShapeOut, SinkSubgroupLayoutQK);
        } else {
          XE35_RUN(false, false, 2, false, SinkShapeQK, SinkShapePV, SinkShapeOut, SinkSubgroupLayoutQK);
        }
        return;
      }
      using ShapeQK = Shape<_128, _64, _32>;
      using ShapePV = Shape<_128, _32, _64>;
      using ShapeOut = Shape<_128, _64>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    }

    using ShapeQK = Shape<_128, _64, _32>;
    using ShapePV = Shape<_128, _32, _64>;
    using ShapeOut = Shape<_128, _64>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    if (is_local) {
      XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    } else if (is_causal) {
      XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    } else {
      XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    }
    return;
  }

#undef XE35_RUN
#undef XE35_INVOKE

  TORCH_CHECK(
      false,
      "flash_attentionXe35: only head_dim 64, 72 and 96 (non-paged), 128, and 256 are supported in Xe3 cutlass path");
}

}  // namespace xe35_fmha

using Xe35DispatchFn = void(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    int,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    int,
    int,
    int,
    int,
    int,
    int,
    const float*,
    const float*,
    const float*,
    float,
    bool,
    bool,
    int,
    int,
    const cutlass::bfloat16_t*,
    bool,
    int);
#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdio>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/kernel_hardware_info.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <optional>
#include <vector>

#include "comm/common.h"
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
#include <cutlass/util/sycl_event_manager.hpp>
#endif
#include "kernels/flash_attention_v2/xe35/collective/xe_fmha_fwd_epilogue.hpp"
#include "kernels/flash_attention_v2/xe35/collective/xe_fmha_fwd_mainloop.hpp"
#include "kernels/flash_attention_v2/xe35/kernel/xe_fmha_fwd_kernel.hpp"
#include "kernels/flash_attention_v2/xe35/kernel/xe_tile_scheduler.hpp"

#define SYCL_INTEL_TARGET 35

using namespace cute;

namespace {
sycl::queue& xe35_profiling_queue() {
  static sycl::queue queue = [] {
    sycl::queue torch_queue = at::xpu::getCurrentXPUStream().queue();
    return sycl::queue(torch_queue.get_context(), torch_queue.get_device(), sycl::property::queue::enable_profiling{});
  }();
  return queue;
}
}  // namespace

template <class Element>
struct Xe3FmhaConfig {
  using ElementInput = Element;
  using ElementOutput = Element;
  using ElementScale = float;

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;
  using StrideScale = Stride<_1, int, int, int>;

  template <typename T, typename S>
  static auto make_dummy_tensor(T val, S stride) {
    return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<S>>(1), stride));
  }

  template <
      bool Causal,
      bool LocalMask,
      bool Sink,
      int PipelineStages,
      bool GqaFusion,
      class TileShapeQK,
      class TileShapePV,
      class TileShapeOut,
      class SubgroupLayoutQK>
  static void
  run(const at::Tensor& q_dense,
      const at::Tensor& k_pool,
      const at::Tensor& v_pool,
      const at::Tensor& page_table,
      int num_pages_per_seq,
      const at::Tensor& cu_seqlens_q,
      const at::Tensor& cu_seqlens_k_cache,
      at::Tensor& out_dense,
      int batch,
      int num_heads_q,
      int num_heads_kv,
      int max_seqlen_q,
      int max_seqlen_kv_cache,
      int head_dim,
      int page_size,
      float softmax_scale,
      int window_size_left,
      int window_size_right,
      const ElementInput* sm_sink) {
    static constexpr bool BlockScale = false;
    static constexpr bool F8kvF16mma = false;
    static constexpr bool PerTensorScale = false;
    static constexpr bool CachedKV = true;
    static constexpr bool PagedKV = true;

    constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
    using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementInput>;
    using MMAOperationPV = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementInput>;
    using SubgroupLayoutPV = decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV =
        typename TiledMMAHelper<MMA_Atom<MMAOperationPV>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

    static_assert(
        get<0>(TileShapeOut{}) == get<0>(TileShapePV{}), "Output tile and P*V tile must match on Q dimension");
    static constexpr int VTiles = get<1>(TileShapeOut{}) / get<1>(TileShapePV{});

    using TensorQ = decltype(make_dummy_tensor(ElementInput{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementInput{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementInput{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementOutput{}, StrideO{}));
    using TensorScaleQ = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));
    using TensorScaleK = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));
    using TensorScaleV = decltype(make_dummy_tensor(ElementScale{}, StrideScale{}));

    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy,
        Causal,
        LocalMask,
        BlockScale,
        F8kvF16mma,
        PerTensorScale,
        CachedKV,
        PagedKV,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        TensorScaleQ,
        TensorScaleK,
        TensorScaleV,
        TensorK,
        TensorV,
        void,
        void,
        void,
        void,
        void>;

    using CollectiveEpilogue =
        cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOut, TensorO, void, Sink>;
    using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<true>;
    using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false, Causal, GqaFusion>;
    using FMHAKernel =
        cutlass::fmha::kernel::XeFMHAFwdKernel<ProblemShape, CollectiveMainloop, CollectiveEpilogue, Scheduler>;

    const int q_seq_stride = static_cast<int>(q_dense.stride(0));
    const int q_head_stride = static_cast<int>(q_dense.stride(1));
    const int kv_seq_stride = static_cast<int>(k_pool.stride(0));
    const int kv_head_stride = static_cast<int>(k_pool.stride(1));
    const int o_seq_stride = static_cast<int>(out_dense.stride(0));
    const int o_head_stride = static_cast<int>(out_dense.stride(1));
    typename FMHAKernel::StrideQ stride_q{q_seq_stride, {}, q_head_stride, 0};
    typename FMHAKernel::StrideK stride_k{kv_seq_stride, {}, kv_head_stride, 0};
    typename FMHAKernel::StrideV stride_v{{}, kv_seq_stride, kv_head_stride, 0};
    typename FMHAKernel::StrideO stride_o{o_seq_stride, {}, o_head_stride, 0};

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    typename FMHAKernel::Arguments arguments{
        {
            {batch,
             num_heads_q,
             num_heads_kv,
             {max_seqlen_q, static_cast<int>(q_dense.size(0)), cu_seqlens_q.data_ptr<int>(), nullptr},
             {0, 0, nullptr, nullptr},
             {max_seqlen_kv_cache, static_cast<int>(k_pool.size(0)), cu_seqlens_k_cache.data_ptr<int>(), nullptr},
             head_dim,
             head_dim},
            static_cast<const ElementInput*>(q_dense.data_ptr()),
            stride_q,
            static_cast<const ElementInput*>(k_pool.data_ptr()),
            stride_k,
            static_cast<const ElementInput*>(v_pool.data_ptr()),
            stride_v,
            static_cast<ElementOutput*>(out_dense.data_ptr()),
            stride_o,
            nullptr,
            {},
            nullptr,
            {},
            nullptr,
            {},
            1.0f,
            1.0f,
            1.0f,
            32,
            static_cast<const ElementInput*>(k_pool.data_ptr()),
            stride_k,
            static_cast<const ElementInput*>(v_pool.data_ptr()),
            stride_v,
            sm_sink,
        },
        {
            softmax_scale,
            static_cast<const int*>(page_table.data_ptr()),
            page_size,
            num_pages_per_seq,
            static_cast<int>(k_pool.size(0)),
            window_size_left,
            window_size_right,
        },
        {},
        hw_info,
    };

    TORCH_CHECK(FMHAKernel::can_implement(arguments), "flash_attentionXe35: invalid cutlass FMHA arguments");

    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    at::Tensor workspace = at::empty({static_cast<long>(workspace_size)}, q_dense.options().dtype(at::kByte));

    auto* workspace_ptr = workspace_size > 0 ? workspace.data_ptr() : nullptr;
    auto status = FMHAKernel::initialize_workspace(arguments, workspace_ptr);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "flash_attentionXe35: FMHA workspace initialization failed");

    auto params = FMHAKernel::to_underlying_arguments(arguments, workspace_ptr);

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    compat::dim3 const block = FMHAKernel::get_block_shape();
    compat::dim3 const grid = FMHAKernel::get_grid_shape(params);
    const int smem_size = FMHAKernel::SharedStorageSize;
    compat::experimental::launch_properties launch_props{syclex::work_group_scratch_size(smem_size)};
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<512>};
    compat::experimental::launch_policy policy{
        compat::dim3(grid.x, grid.y, grid.z), compat::dim3(block.x, block.y, block.z), launch_props, kernel_props};
    sycl::queue& profiling_queue = xe35_profiling_queue();
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
    auto event =
        compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, profiling_queue, params);
    EventManager::getInstance().addEvent(event);
#else
    compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel, false>(
        policy, profiling_queue, params);
#endif
  }
};

template <int HeadDim, bool UseDecode>
void dispatch_xe3_kernel(
    const at::Tensor& q_dense,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_table,
    int num_pages_per_seq,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k_cache,
    at::Tensor& out_dense,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int max_seqlen_q,
    int max_seqlen_kv_cache,
    int page_size,
    float softmax_scale,
    bool is_causal,
    bool is_local,
    int window_size_left,
    int window_size_right,
    const cutlass::bfloat16_t* sm_sink,
    bool use_sink,
    int num_kv_splits = 1) {
  const int gqa_group = num_heads_kv > 0 ? num_heads_q / num_heads_kv : 1;
  const int total_rows = gqa_group * max_seqlen_q;
  const bool use_decode = UseDecode;

#define XE35_RUN(CAUSAL, LOCAL, STAGES, GQA, QK, PV, OUT, SGL)                                               \
  do {                                                                                                       \
    if (use_sink) {                                                                                          \
      Xe3FmhaConfig<cutlass::bfloat16_t>::template run<CAUSAL, LOCAL, true, STAGES, GQA, QK, PV, OUT, SGL>(  \
          q_dense,                                                                                           \
          k_pool,                                                                                            \
          v_pool,                                                                                            \
          page_table,                                                                                        \
          num_pages_per_seq,                                                                                 \
          cu_seqlens_q,                                                                                      \
          cu_seqlens_k_cache,                                                                                \
          out_dense,                                                                                         \
          batch,                                                                                             \
          num_heads_q,                                                                                       \
          num_heads_kv,                                                                                      \
          max_seqlen_q,                                                                                      \
          max_seqlen_kv_cache,                                                                               \
          HeadDim,                                                                                           \
          page_size,                                                                                         \
          softmax_scale,                                                                                     \
          window_size_left,                                                                                  \
          window_size_right,                                                                                 \
          sm_sink);                                                                                          \
    } else {                                                                                                 \
      Xe3FmhaConfig<cutlass::bfloat16_t>::template run<CAUSAL, LOCAL, false, STAGES, GQA, QK, PV, OUT, SGL>( \
          q_dense,                                                                                           \
          k_pool,                                                                                            \
          v_pool,                                                                                            \
          page_table,                                                                                        \
          num_pages_per_seq,                                                                                 \
          cu_seqlens_q,                                                                                      \
          cu_seqlens_k_cache,                                                                                \
          out_dense,                                                                                         \
          batch,                                                                                             \
          num_heads_q,                                                                                       \
          num_heads_kv,                                                                                      \
          max_seqlen_q,                                                                                      \
          max_seqlen_kv_cache,                                                                               \
          HeadDim,                                                                                           \
          page_size,                                                                                         \
          softmax_scale,                                                                                     \
          window_size_left,                                                                                  \
          window_size_right,                                                                                 \
          nullptr);                                                                                          \
    }                                                                                                        \
  } while (false)

  if constexpr (HeadDim == 128) {
    if constexpr (UseDecode) {
      using ShapeQK = Shape<_128, _64, _64>;
      using ShapePV = Shape<_128, _64, _64>;
      using ShapeOut = Shape<_128, _128>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    }

    using ShapeQK = Shape<_128, _64, _64>;
    using ShapePV = Shape<_128, _64, _64>;
    using ShapeOut = Shape<_128, _128>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    if (is_local)
      XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else if (is_causal)
      XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else
      XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    return;
  }

  if constexpr (HeadDim == 64) {
    if constexpr (UseDecode) {
      if (use_sink) {
        using SinkShapeQK = Shape<_128, _64, _32>;
        using SinkShapePV = Shape<_128, _32, _64>;
        using SinkShapeOut = Shape<_128, _64>;
        using SinkSubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
        if (is_local)
          XE35_RUN(false, true, 2, false, SinkShapeQK, SinkShapePV, SinkShapeOut, SinkSubgroupLayoutQK);
        else
          XE35_RUN(false, false, 2, false, SinkShapeQK, SinkShapePV, SinkShapeOut, SinkSubgroupLayoutQK);
        return;
      }
      using ShapeQK = Shape<_128, _64, _32>;
      using ShapePV = Shape<_128, _32, _64>;
      using ShapeOut = Shape<_128, _64>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      if (is_local)
        XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else if (is_causal)
        XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      else
        XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
      return;
    }

    using ShapeQK = Shape<_128, _64, _32>;
    using ShapePV = Shape<_128, _32, _64>;
    using ShapeOut = Shape<_128, _64>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    if (is_local)
      XE35_RUN(false, true, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else if (is_causal)
      XE35_RUN(true, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    else
      XE35_RUN(false, false, 2, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK);
    return;
  }

  TORCH_CHECK(false, "flash_attentionXe35: only head_dim 64 and 128 are supported in Xe3 cutlass path");
}

#undef XE35_RUN
