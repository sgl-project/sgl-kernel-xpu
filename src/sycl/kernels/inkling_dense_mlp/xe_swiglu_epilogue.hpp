// Copyright (C) 2026 Intel Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Fused gate/up + SwiGLU epilogue for the Inkling dense MLP.
//
// The source weight is repacked in 16-column blocks:
//   [gate block 0, up block 0, gate block 1, up block 1, ...].
// This places both inputs for one output block in the same subgroup, so the
// activation is register-local and only the MxI result reaches memory.

#pragma once

#include <sycl/sycl.hpp>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"

namespace cutlass::epilogue::collective {

template <class ElementOutput_, class ElementCompute_, class ElementAccumulator_>
struct SwiGluEpilogueOp {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementAccumulator = ElementAccumulator_;
  static constexpr cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
};

template <class ElementCompute>
struct SwiGluFn {
  CUTLASS_DEVICE ElementCompute operator()(ElementCompute gate, ElementCompute up) const {
    cutlass::epilogue::thread::SiLu<ElementCompute> silu;
    return silu(gate) * up;
  }
};

template <
    class WGTileMNK_,
    class SubgroupLayout_,
    class MmaAtom_,
    class ElementAccumulator_,
    class ElementD_,
    class StrideD_,
    template <class> class ActivationFn_ = SwiGluFn,
    int OutBlocksPerStore_ = 1>
class XeSwiGluGateUpEpilogue {
 public:
  using DispatchPolicy = IntelXeXMX16;
  using WGTileMNK = WGTileMNK_;
  using SubgroupLayout = SubgroupLayout_;
  using MmaAtom = MmaAtom_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementAccumulator_;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using ElementOutput = ElementD;
  using ElementC = void;
  using StrideC = StrideD_;
  using NonVoidElementC = ElementD;
  using ThreadEpilogueOp = SwiGluEpilogueOp<ElementD, ElementCompute, ElementAccumulator>;

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;
  static constexpr int CopyBitsD = cute::sizeof_bits_v<ElementD>;
  static constexpr int kOutBlocksPerStore = OutBlocksPerStore_;
  static constexpr int kSubgroupsN = cute::size<1>(SubgroupLayout{});
  static constexpr int kSgN = cute::size<1>(WGTileMNK{}) / kSubgroupsN;

  using HalfWGTileMNK = cute::Shape<
      decltype(cute::get<0>(WGTileMNK{})),
      decltype(cute::get<1>(WGTileMNK{}) / cute::_2{}),
      decltype(cute::get<2>(WGTileMNK{}))>;
  using HalfTiledMma = typename cute::TiledMMAHelper<MmaAtom, cute::Layout<HalfWGTileMNK>, SubgroupLayout>::TiledMMA;

  static_assert(
      kSgN % (32 * kOutBlocksPerStore) == 0, "SwiGLU pairing requires SG_N to be divisible by 32*OutBlocksPerStore");

  using MMATile = decltype(cute::take<0, 2>(typename HalfTiledMma::AtomShape_MNK{}));
  using EpilogueTile = cute::
      Shape<decltype(cute::get<0>(MMATile{})), decltype(cute::get<1>(MMATile{}) * cute::Int<kOutBlocksPerStore>{})>;
  using MmaPerEpi = cute::Shape<cute::_1, cute::Int<kOutBlocksPerStore>>;

  using GmemTiledCopyC = cute::XE_LOAD_2D<CopyBitsD, 8, 512 / CopyBitsD>;
  using GmemTiledCopyD = cute::XE_STORE_2D<CopyBitsD, 8, 512 / CopyBitsD>;
  using ActualCopyOpD = cute::XE_STORE_2D<
      CopyBitsD,
      cute::gcd(8, cute::get<0>(EpilogueTile{})),
      cute::gcd(512 / CopyBitsD, cute::get<1>(EpilogueTile{}))>;

  using TensorD = decltype(cute::make_tensor(
      cute::make_gmem_ptr(static_cast<ElementD*>(nullptr)), cute::Layout<cute::Shape<int, int, int>, StrideD>{}));

  struct SharedStorage {};
  using TensorStorage = SharedStorage;

  struct Arguments {
    ElementD* ptr_D;
    StrideD dD;
  };

  struct Params {
    TensorD mD;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const& shape, Arguments const& args, void* workspace) {
    (void)workspace;
    auto problem_shape_mnkl = cute::append<4>(shape, 1);
    auto [m, n, k, l] = problem_shape_mnkl;
    auto output =
        cute::make_tensor(cute::make_gmem_ptr(args.ptr_D), cute::make_layout(cute::make_shape(m, n / 2, l), args.dD));
    return {output};
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const&,
      Arguments const&,
      void*,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement(ProblemShape const& shape, Arguments const& args) {
    auto problem_shape_mnkl = cute::append<4>(shape, 1);
    auto [m, n, k, l] = problem_shape_mnkl;
    constexpr int min_aligned_elements_d = 128 / cute::sizeof_bits_v<ElementD>;
    return n % 32 == 0 &&
           cutlass::detail::check_alignment<min_aligned_elements_d>(cute::make_shape(m, n / 2, l), args.dD);
  }

  CUTLASS_HOST_DEVICE
  XeSwiGluGateUpEpilogue(Params const& params_, SharedStorage const&) : params(params_) {}

  CUTLASS_DEVICE bool is_producer_load_needed() const {
    return false;
  }

  template <class ProblemShapeMNKL, class TileShapeMNK, class TileCoordMNKL, class Accumulator, class TiledMMA>
  CUTLASS_DEVICE void operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK,
      TileCoordMNKL tile_coord_mnkl,
      Accumulator accumulators,
      TiledMMA,
      int thread_idx) {
    using namespace cute;

    static constexpr int kFragmentsN = size<2>(Accumulator{});
    static_assert(kFragmentsN % (2 * kOutBlocksPerStore) == 0);
    static constexpr int kOutputFragmentsN = kFragmentsN / 2;
    static constexpr int kStoresN = kOutputFragmentsN / kOutBlocksPerStore;
    static constexpr int kFragmentSize = size<0>(Accumulator{});
    static constexpr int kEpilogueTilesM = size<1>(Accumulator{});

    auto batch_idx = get<3>(tile_coord_mnkl);
    auto m = get<0>(problem_shape_mnkl);
    auto n_out = get<1>(problem_shape_mnkl) / 2;

    auto coord_d = make_identity_tensor(make_shape(m, n_out));
    auto global_d = local_tile(coord_d, take<0, 2>(HalfWGTileMNK{}), take<0, 2>(tile_coord_mnkl));
    auto thread_mma = HalfTiledMma{}.get_slice(thread_idx);
    auto thread_coord_d = thread_mma.partition_C(global_d);

    auto subgroup_value_coord =
        prepend(flat_divide(remove<0>(thread_coord_d.layout()), MmaPerEpi{}), get<0>(thread_coord_d.layout()));
    auto global_d_epilogue_layout = append(
        append(make_identity_layout(EpilogueTile{}), get<3>(subgroup_value_coord)), get<4>(subgroup_value_coord));
    auto global_d_epilogue = make_tensor(thread_coord_d.data(), global_d_epilogue_layout);

    auto copy_d = make_block_2d_copy(ActualCopyOpD{}, params.mD(_, _, batch_idx));
    auto thread_copy_d = copy_d.get_slice(thread_idx % intel::sg_size);
    auto thread_global_d = thread_copy_d.partition_D(global_d_epilogue);
    auto thread_register_d = thread_copy_d.partition_sg_fragment_S(global_d_epilogue(_, _, 0, 0));

    using AccTVLayout = decltype(thread_mma.partition_sg_fragment_C(global_d).tv_layout());
    auto compute_tv = make_layout(get<0>(AccTVLayout{}), subgroup_value_coord(_, _, _, _0{}, _0{}));
    auto thread_register_d_wi = make_fragment_like<ElementD>(subgroup_value_coord(_, _, _, _0{}, _0{}).shape());
    auto thread_register_d_compute = make_subgroup_tensor(thread_register_d_wi, compute_tv);

    ActivationFn_<ElementCompute> activation;
    NumericConverter<ElementD, ElementCompute, ThreadEpilogueOp::RoundStyle> convert_output;

    CUTLASS_PRAGMA_UNROLL
    for (int epilogue_m = 0; epilogue_m < kEpilogueTilesM; ++epilogue_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int epilogue_n = 0; epilogue_n < kStoresN; ++epilogue_n) {
        CUTLASS_PRAGMA_UNROLL
        for (int block = 0; block < kOutBlocksPerStore; ++block) {
          const int accumulator_n = 2 * (epilogue_n * kOutBlocksPerStore + block);
          CUTLASS_PRAGMA_UNROLL
          for (int epilogue_v = 0; epilogue_v < kFragmentSize; ++epilogue_v) {
            const ElementCompute gate = ElementCompute(accumulators(epilogue_v, epilogue_m, accumulator_n));
            const ElementCompute up = ElementCompute(accumulators(epilogue_v, epilogue_m, accumulator_n + 1));
            thread_register_d_wi(epilogue_v, _0{}, block) = convert_output(activation(gate, up));
          }
        }
        reorder(thread_register_d_compute, thread_register_d);
        copy(copy_d, thread_register_d, thread_global_d(_, _, _, epilogue_m, epilogue_n));
      }
    }
  }

 private:
  Params const& params;
};

}  // namespace cutlass::epilogue::collective
