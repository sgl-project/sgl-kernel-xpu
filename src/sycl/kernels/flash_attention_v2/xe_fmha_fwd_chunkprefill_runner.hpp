#pragma once

#include <cstdlib>
#include <vector>

#include "sycl/kernels/flash_attention_v2/xe_fmha_fwd_prefill_runner.hpp"
#include "sycl/kernels/flash_attention_v2/kernel/xe_reduce_split_k.hpp"

namespace cutlass::fmha::kernel {

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeFMHAChunkPrefillSplitKVKernel {
 public:
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len = cutlass::fmha::collective::is_variable_length_v<typename ProblemShape::SeqLenType>;

  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using SubgroupLayoutQK = typename CollectiveMainloop::SubgroupLayoutQK;
  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;
  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));
  using SGPerWG = typename CollectiveMainloop::SGPerWG;
  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using ElementA = typename CollectiveMainloop::ElementA;

  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));
  using ElementLSE = typename CollectiveEpilogue::ElementLSE;

  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);
  static constexpr int max_num_kv_splits = SGPerWG::value * intel::sg_size;

  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q;
    StrideQ dQ;
    const ElementK* K;
    StrideK dK;
    const ElementV* V;
    StrideV dV;
    ElementO* O;
    StrideO dO;
    const ElementK* K_cache = nullptr;
    StrideK dK_cache{};
    const ElementV* V_cache = nullptr;
    StrideV dV_cache{};
    ElementO* Oaccum = nullptr;
    StrideO dOaccum{};
    ElementLSE* exp_sums = nullptr;
    StrideO dExp_sums{};
    ElementLSE* max_logits = nullptr;
    StrideO dMax_logits{};
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = 1;
  };

  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  static size_t align_up(size_t offset, size_t alignment) {
    return (offset + alignment - 1) / alignment * alignment;
  }

  static int q_capacity(ProblemShape const& shape) {
    if constexpr (is_var_len) {
      return shape.batch * shape.seq_len_qo.max_length;
    } else {
      return shape.batch * shape.seq_len_qo;
    }
  }

  static size_t counter_workspace_size() {
    if constexpr (is_same_v<TileScheduler, XeFMHASplitKVDynamicPersistentTileScheduler>) {
      return sizeof(int);
    }
    return 0;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    KernelParams kernel = args.kernel;
    size_t workspace_offset = counter_workspace_size();
    workspace_offset = align_up(workspace_offset, alignof(ElementO));
    int q_tokens = q_capacity(args.kernel.shape);
    size_t oaccum_count = size_t(q_tokens) * args.kernel.shape.num_heads_q * args.num_kv_splits *
        args.kernel.shape.head_size_vo;
    if (kernel.Oaccum == nullptr) {
      kernel.Oaccum = reinterpret_cast<ElementO*>(reinterpret_cast<uint8_t*>(workspace) + workspace_offset);
    }
    workspace_offset += oaccum_count * sizeof(ElementO);
    workspace_offset = align_up(workspace_offset, alignof(ElementLSE));
    size_t lse_count = size_t(q_tokens) * args.kernel.shape.num_heads_q * args.num_kv_splits;
    if (kernel.exp_sums == nullptr) {
      kernel.exp_sums = reinterpret_cast<ElementLSE*>(reinterpret_cast<uint8_t*>(workspace) + workspace_offset);
    }
    workspace_offset += lse_count * sizeof(ElementLSE);
    workspace_offset = align_up(workspace_offset, alignof(ElementLSE));
    if (kernel.max_logits == nullptr) {
      kernel.max_logits = reinterpret_cast<ElementLSE*>(reinterpret_cast<uint8_t*>(workspace) + workspace_offset);
    }

    auto batch_dim = is_var_len ? 1 : args.kernel.shape.batch;
    int seq_len_qo = is_var_len ? args.kernel.shape.seq_len_qo.max_length : args.kernel.shape.seq_len_qo;
    auto shape_Oaccum = make_shape(
        seq_len_qo, args.kernel.shape.head_size_vo, args.kernel.shape.num_heads_q * args.num_kv_splits, batch_dim);
    auto shape_lse = make_shape(seq_len_qo, args.num_kv_splits, args.kernel.shape.num_heads_q, batch_dim);
    kernel.dOaccum = cutlass::make_cute_packed_stride(StrideO{}, shape_Oaccum);
    kernel.dExp_sums = cutlass::make_cute_packed_stride(StrideO{}, shape_lse);
    kernel.dMax_logits = cutlass::make_cute_packed_stride(StrideO{}, shape_lse);

    auto scheduler = TileScheduler::template to_underlying_arguments<SGPerWG::value>(
        args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits);
    if constexpr (is_same_v<TileScheduler, XeFMHASplitKVDynamicPersistentTileScheduler>) {
      scheduler.tile_counter = reinterpret_cast<int*>(workspace);
    }
    return {
        kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
        scheduler};
  }

  static bool can_implement(Arguments const& args) {
    return args.num_kv_splits > 0 && args.num_kv_splits <= max_num_kv_splits &&
        CollectiveMainloop::can_implement(args.mainloop) && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const& args) {
    int q_tokens = q_capacity(args.kernel.shape);
    size_t workspace_size = counter_workspace_size();
    workspace_size = align_up(workspace_size, alignof(ElementO));
    workspace_size += size_t(q_tokens) * args.kernel.shape.num_heads_q * args.num_kv_splits *
        args.kernel.shape.head_size_vo * sizeof(ElementO);
    workspace_size = align_up(workspace_size, alignof(ElementLSE));
    workspace_size += size_t(q_tokens) * args.kernel.shape.num_heads_q * args.num_kv_splits * sizeof(ElementLSE);
    workspace_size = align_up(workspace_size, alignof(ElementLSE));
    workspace_size += size_t(q_tokens) * args.kernel.shape.num_heads_q * args.num_kv_splits * sizeof(ElementLSE);
    return int(workspace_size);
  }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    if constexpr (is_same_v<TileScheduler, XeFMHASplitKVDynamicPersistentTileScheduler>) {
      compat::fill(reinterpret_cast<int*>(workspace), 0, 1);
    }
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  Shape<int, int, int> get_sequence_length_shape(ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      return cutlass::fmha::collective::apply_variable_length(
          Shape<VariableLength, VariableLength, VariableLength>{
              problem_shape.seq_len_qo, problem_shape.seq_len_kv, problem_shape.seq_len_kv_cache},
          batch);
    } else {
      return Shape<int, int, int>{problem_shape.seq_len_qo, problem_shape.seq_len_kv, problem_shape.seq_len_kv_cache};
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    auto& p = params.kernel;
    ProblemShape const& s = p.shape;
    using SeqLenQ = remove_cvref_t<decltype(s.seq_len_qo)>;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());
    int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));

    auto cS = make_identity_tensor(take<0, 2>(TiledMMAQK{}.tile_mnk()));
    auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
    auto q_offset_wi = get<0>(tScS(0));
    auto q_offset_sg = group_broadcast(sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);

    TileScheduler tile_scheduler{params.scheduler};
    int num_kv_splits = params.scheduler.num_kv_splits_;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_q, idx_b, blk_k] = tile_scheduler.template get_block_coord<SeqLenQ>();
      auto blk_qv = make_coord(blk_q, blk_v);
      int head = head_q / head_group_q;

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv, seq_len_kv_cache] = sequence_length_shape;
      if (blk_q * get<0>(TileShapeQK{}) >= seq_len_qo) continue;

      auto offset = cute::min(seq_len_qo, seq_len_kv);
      auto discard_seq_coord = seq_len_qo - offset;
      auto full_tile_offset = seq_len_kv - offset;
      int seq_coord = cute::min(seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));

      if (CollectiveMainloop::CausalMask && seq_coord < discard_seq_coord) continue;
      const int seq_len_new = CollectiveMainloop::CausalMask
          ? full_tile_offset + cute::min(seq_len_kv, seq_coord - discard_seq_coord) + q_sg_tile
          : seq_len_kv;
      const int seq_len = seq_len_new + seq_len_kv_cache;
      const int k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));
      int num_blocks_per_split = cute::ceil_div(k_blocks, num_kv_splits);
      int start_blk = blk_k * num_blocks_per_split;
      int end_blk = cute::min(k_blocks, start_blk + num_blocks_per_split);
      if (start_blk >= end_blk) continue;

      int offset_q = 0, offset_k = 0, offset_v = 0, offset_o = 0;
      int offset_k_cache = 0, offset_v_cache = 0;
      int offset_exp_sums = 0, offset_max_logits = 0;
      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;
        auto kv_cumulative = s.seq_len_kv.cumulative_length;
        offset_q = s.num_heads_q * s.head_size_qk * qo_cumulative[idx_b];
        offset_k = s.num_heads_kv * s.head_size_qk * kv_cumulative[idx_b];
        offset_v = s.num_heads_kv * s.head_size_vo * kv_cumulative[idx_b];
        offset_o = s.num_heads_q * s.head_size_vo * num_kv_splits * qo_cumulative[idx_b];
        offset_exp_sums = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];
        offset_max_logits = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];
        if (s.seq_len_kv_cache.cumulative_length) {
          auto kv_cumulative_cache = s.seq_len_kv_cache.cumulative_length;
          offset_k_cache = s.num_heads_kv * s.head_size_qk * kv_cumulative_cache[idx_b];
          offset_v_cache = s.num_heads_kv * s.head_size_vo * kv_cumulative_cache[idx_b];
        }
      }

      auto batch_dim = is_var_len ? 1 : s.batch;
      auto shape_Q = make_shape(seq_len_qo, s.head_size_qk, s.num_heads_q, batch_dim);
      auto shape_K = make_shape(seq_len_kv, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V = make_shape(s.head_size_vo, seq_len_kv, s.num_heads_kv, batch_dim);
      auto shape_O = make_shape(seq_len_qo, s.head_size_vo, s.num_heads_q * num_kv_splits, batch_dim);
      auto shape_exp_sums = make_shape(seq_len_qo, num_kv_splits, s.num_heads_q, batch_dim);
      auto shape_max_logits = make_shape(seq_len_qo, num_kv_splits, s.num_heads_q, batch_dim);
      auto shape_K_cache = make_shape(seq_len_kv_cache, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V_cache = make_shape(s.head_size_vo, seq_len_kv_cache, s.num_heads_kv, batch_dim);

      auto dcQ = const_cast<ElementQ*>(p.Q + offset_q);
      auto dcK = const_cast<ElementK*>(p.K + offset_k);
      auto dcV = const_cast<ElementV*>(p.V + offset_v);
      auto dcK_cache = const_cast<ElementK*>(p.K_cache + offset_k_cache);
      auto dcV_cache = const_cast<ElementV*>(p.V_cache + offset_v_cache);
      auto ptrO = p.Oaccum + offset_o;
      auto ptrExp_sums = p.exp_sums + offset_exp_sums;
      auto ptrMax_logits = p.max_logits + offset_max_logits;

      auto stride_q = is_var_len ? cutlass::make_cute_packed_stride(StrideQ{}, shape_Q) : p.dQ;
      auto stride_k = is_var_len ? cutlass::make_cute_packed_stride(StrideK{}, shape_K) : p.dK;
      auto stride_v = is_var_len ? cutlass::make_cute_packed_stride(StrideV{}, shape_V) : p.dV;
      auto stride_o = cutlass::make_cute_packed_stride(StrideO{}, shape_O);
      auto stride_k_cache = is_var_len ? cutlass::make_cute_packed_stride(StrideK{}, shape_K_cache) : p.dK_cache;
      auto stride_v_cache = is_var_len ? cutlass::make_cute_packed_stride(StrideV{}, shape_V_cache) : p.dV_cache;
      auto stride_exp_sums = cutlass::make_cute_packed_stride(StrideO{}, shape_exp_sums);
      auto stride_max_logits = cutlass::make_cute_packed_stride(StrideO{}, shape_max_logits);

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, stride_q));
      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, stride_k));
      Tensor V = make_tensor(make_gmem_ptr(dcV), make_layout(shape_V, stride_v));
      Tensor K_cache = make_tensor(make_gmem_ptr(dcK_cache), make_layout(shape_K_cache, stride_k_cache));
      Tensor V_cache = make_tensor(make_gmem_ptr(dcV_cache), make_layout(shape_V_cache, stride_v_cache));
      Tensor O = make_tensor(make_gmem_ptr(ptrO), make_layout(shape_O, stride_o));
      Tensor exp_sums = make_tensor(make_gmem_ptr(ptrExp_sums), make_layout(shape_exp_sums, stride_exp_sums));
      Tensor max_logits = make_tensor(make_gmem_ptr(ptrMax_logits), make_layout(shape_max_logits, stride_max_logits));

      FragA tArA;
      FragARow tA_max, tA_sum;
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);

      int l_coord = is_var_len ? 0 : idx_b;
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      mainloop(
          Q(_, _, head_q, l_coord),
          K(_, _, head, l_coord),
          V(_, _, head, l_coord),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          start_blk,
          end_blk,
          k_blocks,
          end_blk,
          thr_id,
          seq_len,
          seq_len_kv_cache,
          idx_b,
          full_tile_offset,
          discard_seq_coord,
          K_cache(_, _, head, l_coord),
          V_cache(_, _, head, l_coord));

      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      epilogue(
          O(_, _, blk_k * s.num_heads_q + head_q, l_coord),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          thr_id,
          exp_sums(_, _, head_q, l_coord),
          max_logits(_, _, head_q, l_coord),
          blk_k);
    }
  }
};

}  // namespace cutlass::fmha::kernel

namespace chunkprefill {
struct Arguments : public prefill::Arguments {
  int num_kv_splits = 1;
};

inline int round_up_headdim(int head_size) {
  if (head_size <= 64) return 64;
  if (head_size <= 96) return 96;
  if (head_size <= 128) return 128;
  if (head_size <= 192) return 192;
  if (head_size <= 256) return 256;
  return 512;
}

enum class ChunkPrefillSchedulerVariant : int {
  StaticPersistent = 1,
  DynamicPersistent = 2,
  SplitKVStaticPersistent = 3,
  SplitKVDynamicPersistent = 4,
};

inline ChunkPrefillSchedulerVariant get_scheduler_variant() {
  const char* env = std::getenv("SGL_CHUNKPREFILL_SCHEDULER");
  if (env == nullptr) {
    return ChunkPrefillSchedulerVariant::StaticPersistent;
  }
  int value = std::atoi(env);
  if (value < 1 || value > 4) {
    return ChunkPrefillSchedulerVariant::StaticPersistent;
  }
  return static_cast<ChunkPrefillSchedulerVariant>(value);
}

template <int HEAD_DIM>
struct FmhaChunkPrefillStaticRunner {
  void operator()(const Arguments& params) const;
};

template <int HEAD_DIM>
struct FmhaChunkPrefillDynamicRunner {
  void operator()(const Arguments& params) const;
};

template <int HEAD_DIM>
struct FmhaChunkPrefillSplitKVStaticRunner {
  void operator()(const Arguments& params) const;
};

template <int HEAD_DIM>
struct FmhaChunkPrefillSplitKVDynamicRunner {
  void operator()(const Arguments& params) const;
};

template <class FMHAKernel, class ReduceSplitKernel, bool isVarLen>
struct SplitKVChunkPrefillRunner {
  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;
  using ElementLSE = typename FMHAKernel::ElementLSE;

  cutlass::Status run(const Arguments& params, const cutlass::KernelHardwareInfo& hw_info) {
    prefill::PrefillRunner<FMHAKernel, isVarLen> base;
    auto shape = base.initialize(params);
    bool direct_output = params.seqlen_k <= 0 || params.seqlen_knew <= 0 || params.seqlen_knew == 0;
    direct_output = params.seqlen_k > 0 && params.seqlen_knew == 0 && params.h > 0;

    typename FMHAKernel::Arguments arguments{
        {
            shape,
            static_cast<const ElementQ*>(params.q_ptr),
            base.stride_Q,
            nullptr,
            base.stride_K,
            nullptr,
            base.stride_V,
            static_cast<ElementO*>(params.o_ptr),
            base.stride_O,
            static_cast<const ElementK*>(params.k_ptr),
            base.stride_K_cache,
            static_cast<const ElementV*>(params.v_ptr),
            base.stride_V_cache,
            direct_output ? static_cast<ElementO*>(params.o_ptr) : nullptr,
            base.stride_O,
            nullptr,
            {},
            nullptr,
            {},
        },
        {params.softmax_scale, params.page_table, params.page_size, params.max_num_pages_per_seq},
        {},
        hw_info,
        params.num_kv_splits};

    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    torch::Tensor workspace = torch::empty(workspace_size, params.tensor_opts);

    if (!FMHAKernel::can_implement(arguments)) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    FMHAKernel::initialize_workspace(arguments, workspace.data_ptr());
    auto kernel_params = FMHAKernel::to_underlying_arguments(arguments, workspace.data_ptr());
    launch<FMHAKernel>(kernel_params);

    if (params.num_kv_splits > 1) {
      typename ReduceSplitKernel::Arguments reduce_arg{
          {
              shape,
              static_cast<ElementO*>(params.o_ptr),
              base.stride_O,
              kernel_params.kernel.Oaccum,
              kernel_params.kernel.dOaccum,
              kernel_params.kernel.exp_sums,
              kernel_params.kernel.dExp_sums,
              kernel_params.kernel.max_logits,
              kernel_params.kernel.dMax_logits,
              params.window_size_left,
          },
          hw_info,
          params.num_kv_splits};
      auto reduce_params = ReduceSplitKernel::to_underlying_arguments(reduce_arg, nullptr);
      launch<ReduceSplitKernel>(reduce_params);
    }
    return cutlass::Status::kSuccess;
  }
};

template <
    bool Causal,
    bool LocalMask_,
    bool Sink,
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    typename SubgroupLayoutPV_ = void,
    int PipelineStages = 2,
    typename ElementQ = bfloat16_t,
    typename ElementK = bfloat16_t,
    typename ElementV = bfloat16_t,
    typename ElementO = bfloat16_t,
    typename MMAOperation_ = void,
    typename StrideQ = Stride<int, _1, int, int>,
    typename StrideK = Stride<int, _1, int, int>,
    typename StrideV = Stride<_1, int, int, int>,
    typename StrideO = Stride<int, _1, int, int>,
    typename GmemTiledCopyQ = void,
    typename GmemTiledCopyK = void,
    typename GmemTiledCopyV = void,
    typename GmemTiledCopyO = void>
struct ChunkPrefillSplitKVConfig {
  static constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = cute::conditional_t<
      is_void_v<MMAOperation_>,
      typename cute::conditional_t<
          cute::is_same_v<ElementQ, cutlass::float_e5m2_t> || cute::is_same_v<ElementQ, cutlass::float_e4m3_t>,
          XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, half_t>,
          XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>>,
      MMAOperation_>;
  using SubgroupLayoutPV = cute::conditional_t<
      is_void_v<SubgroupLayoutPV_>,
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})),
      SubgroupLayoutPV_>;

  template <bool isVarLen, bool CachedKV, bool PagedKV, class Scheduler>
  static void run(const Arguments& params) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;
    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ = decltype(make_dummy_tensor(ElementQ{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementK{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementV{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));
    using TensorLSE = decltype(make_dummy_tensor(float{}, StrideO{}));

    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy,
        Causal,
        CachedKV,
        PagedKV,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        TensorK,
        TensorV,
        GmemTiledCopyQ,
        GmemTiledCopyK,
        GmemTiledCopyV,
        GmemTiledCopyK,
        GmemTiledCopyV>;

    using CollectiveEpilogue =
        cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO, TensorLSE>;
    using FMHAKernel = cutlass::fmha::kernel::XeFMHAChunkPrefillSplitKVKernel<
        ProblemShapeType,
        CollectiveMainloop,
        CollectiveEpilogue,
        Scheduler>;
    using ReduceSplitKernel = cutlass::reduction::kernel::
        ReduceSplitK<ProblemShapeType, cutlass::fmha::kernel::XeReduceSplitKTileScheduler, FMHAKernel>;

    SplitKVChunkPrefillRunner<FMHAKernel, ReduceSplitKernel, isVarLen> runner;
    runner.run(params, hw_info);
  }
};

}  // namespace chunkprefill
