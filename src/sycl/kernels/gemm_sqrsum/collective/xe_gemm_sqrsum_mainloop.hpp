#pragma once

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

namespace cutlass::gemm_sqrsum {

template <int Stages>
class XeDefault {};

}  // namespace cutlass::gemm_sqrsum

namespace cutlass::gemm_sqrsum::collective {
using namespace cute;

template <
    class DispatchPolicy_,
    class TiledMMA_,
    class TensorA_,
    class TensorB_,
    class TiledCopyA_ = void,
    class TiledCopyB_ = void>
struct XeGemmSqrSumMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

template <int Stages, class TiledMMA_, class TensorA_, class TensorB_, class TiledCopyA_, class TiledCopyB_>
struct XeGemmSqrSumMainloop<XeDefault<Stages>, TiledMMA_, TensorA_, TensorB_, TiledCopyA_, TiledCopyB_> {
  using TiledMMA = TiledMMA_;
  using TileShape = decltype(TiledMMA{}.tile_mnk());

  static constexpr auto BLK_M = get<0>(TileShape{});
  static constexpr auto BLK_N = get<1>(TileShape{});
  static constexpr auto BLK_K = get<2>(TileShape{});

  using SubgroupLayout = decltype(TiledMMA{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMA::ThrLayoutVMNK{}))));

  static constexpr int NumSubgroups = SGPerWG::value;

  using TensorA = TensorA_;
  using TensorB = TensorB_;

  using ElementA = typename TensorA_::value_type;
  using ElementB = typename TensorB_::value_type;

  using ElementMMA = typename TiledMMA::ValTypeA;

  using TensorA2D = decltype(TensorA_{}(append<rank_v<TensorA_>>(make_coord(_, _), 0)));
  using TensorB2D = decltype(TensorB_{}(append<rank_v<TensorB_>>(make_coord(_, _), 0)));

  using TiledCopyA =
      conditional_t<is_void_v<TiledCopyA_>, decltype(make_block_2d_copy_A(TiledMMA{}, TensorA2D{})), TiledCopyA_>;
  using TiledCopyB =
      conditional_t<is_void_v<TiledCopyB_>, decltype(make_block_2d_copy_B(TiledMMA{}, TensorB2D{})), TiledCopyB_>;

  using FragGemm = decltype(partition_fragment_C(TiledMMA{}, select<0, 1>(TileShape{})));
  using ElementAccum = typename TiledMMA::ValTypeD;
  using ElementC = ElementAccum;

  using FragSqrSum = decltype(partition_fragment_C(TiledMMA{}, select<0, 1>(TileShape{})));
  using ElementSqrSum = float;

  struct Arguments {};
  using Params = Arguments;
  struct SharedStorage {};

  Params params;
  SharedStorage& shared;

  XeGemmSqrSumMainloop(Params const& params_, SharedStorage& shared_) : params(params_), shared(shared_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return Params{};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  template <typename TensorA2D_runtime, typename TensorB2D_runtime, typename MNCoord>
  CUTLASS_DEVICE void operator()(
      TensorA2D_runtime const& A_2D,
      TensorB2D_runtime const& B_2D,
      FragGemm& tC,
      FragSqrSum& tSqrSum,
      MNCoord blk_mn,
      int thr_id,
      int k_tile_begin,
      int k_tile_end) {
    using namespace sycl::ext::oneapi::this_work_item;

    TiledMMA mma{};
    auto wg_tile = mma.tile_mnk();
    int wg_m = int(get<0>(blk_mn));
    int wg_n = int(get<1>(blk_mn));

    Tensor cA = make_identity_tensor(A_2D.shape());
    Tensor cB = make_identity_tensor(B_2D.shape());

    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));

    auto copy_a = make_block_2d_copy_A(mma, A_2D);
    auto copy_b = make_block_2d_copy_B(mma, B_2D);

    auto thr_mma = mma.get_slice(thr_id);
    auto thr_copy_a = copy_a.get_slice(thr_id);
    auto thr_copy_b = copy_b.get_slice(thr_id);

    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    auto tCrAsq = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrBones = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgB = prefetch_b.get_slice(thr_id).partition_S(gB);

    constexpr int prefetch_dist = Stages;
    constexpr int barrier_scope = 2;
    int k_tile_prefetch = k_tile_begin;

    clear(tC);
    clear(tSqrSum);

    CUTE_UNROLL
    for (int p = 0; p < prefetch_dist; p++, k_tile_prefetch++) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    for (int k_tile = k_tile_begin; k_tile < k_tile_end; k_tile++, k_tile_prefetch++) {
      barrier_arrive(barrier_scope);

      copy(copy_a, tAgA(_, _, _, k_tile), tArA);
      copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);

      cute::gemm(mma, tCrA, tCrB, tC);

      constexpr int n_a = decltype(size(tCrAsq.tensor()))::value;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < n_a; i++) {
        ElementMMA a_in = tCrA(i);
        tCrAsq(i) = static_cast<ElementMMA>(a_in * a_in);
      }
      constexpr int n_b = decltype(size(tCrBones.tensor()))::value;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < n_b; i++) {
        tCrBones(i) = static_cast<ElementMMA>(1);
      }
      cute::gemm(mma, tCrAsq, tCrBones, tSqrSum);

      barrier_wait(barrier_scope);
    }
  }
};

}  // namespace cutlass::gemm_sqrsum::collective
