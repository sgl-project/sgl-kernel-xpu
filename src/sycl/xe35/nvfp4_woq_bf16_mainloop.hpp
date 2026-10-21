/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief NVFP4 weight-only-quantized BF16 mainloop for Xe3P.

    A remains BF16. B is loaded as packed E2M1, converted to BF16 in registers,
    multiplied by its full per-16 NVFP4 dequantization scale, and consumed by a
    BF16 x BF16 DPAS instruction with FP32 accumulation.

    The one-time NVFP4 prep stores a split scale representation:

      c[n, k/16]       = bf16(full_scale * 2^-kexp)
      sfmx[k/32, n]    = kexp + 127

    This mainloop reconstructs the BF16 multiplier as c * 2^kexp. The
    multiplication by a power of two preserves c's BF16 significand, so no
    additional persistent scale tensor is needed for this compute path.
*/

#pragma once

// clang-format off
#include <cstdint>

#include <sycl/sycl.hpp>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_xe.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "sycl/xe35/nvfp4/collective/moe_array_mma.hpp"
#include "sycl/xe35/nvfp4/payload_compat.hpp"

namespace sgl_xpu_xe35 {

template <int Stages_, class KernelSchedule = cutlass::gemm::KernelXe>
struct MainloopXeNvFp4WoqBf16_xe35
    : cutlass::gemm::MainloopXeL1Staged<Stages_, KernelSchedule> {};

template <int Stages_, class KernelSchedule = cutlass::gemm::KernelMoEArrayCooperative_xe35>
struct MainloopXeNvFp4WoqBf16Group_xe35
    : cutlass::gemm::MainloopXeL1Staged<Stages_, KernelSchedule> {};

}  // namespace sgl_xpu_xe35

namespace cutlass::gemm::collective {

using namespace cute;

template <
    int Stages,
    class KernelSchedule,
    class TileShape_,
    class ElementA_,
    class StrideA_,
    class ElementB_,
    class StrideB_,
    class TiledMma_,
    class GmemTiledCopyA_,
    class SmemLayoutAtomA_,
    class SmemCopyAtomA_,
    class TransformA_,
    class GmemTiledCopyB_,
    class SmemLayoutAtomB_,
    class SmemCopyAtomB_,
    class TransformB_>
struct CollectiveMma<
    sgl_xpu_xe35::MainloopXeNvFp4WoqBf16_xe35<Stages, KernelSchedule>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_> {
 public:
  using DispatchPolicy = sgl_xpu_xe35::MainloopXeNvFp4WoqBf16_xe35<Stages, KernelSchedule>;
  using WorkgroupTileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = cute::remove_pointer_t<StrideA_>;
  using ElementB = ElementB_;
  using StrideB = cute::remove_pointer_t<StrideB_>;
  using TiledMma = TiledMma_;
  using ElementBMma = typename TiledMma::ValTypeB;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using ElementScale = uint8_t;
  using ElementMult = cutlass::bfloat16_t;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static_assert(cute::is_same_v<ElementA, cutlass::bfloat16_t>,
                "NVFP4 BF16 mainloop expects BF16 A");
  static_assert(cute::is_same_v<ElementB, cutlass::float_e2m1_t>,
                "NVFP4 BF16 mainloop expects packed E2M1 B");
  static_assert(cute::is_same_v<ElementBMma, cutlass::bfloat16_t>,
                "NVFP4 BF16 mainloop requires a BF16 MMA-B fragment");
  static_assert(std::is_same_v<TransformA_, cute::identity> &&
                    std::is_same_v<TransformB_, cute::identity>,
                "Operand transforms are not supported");

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;
  using MmaAtomShape = typename TiledMma::AtomShape_MNK;
  static constexpr int BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr int BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr int BLK_K = get<2>(WorkgroupTileShape{});
  static constexpr int SG_NUMS_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int SG_NUMS_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int SG_NUMS_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int SG_M = ceil_div(BLK_M, SG_NUMS_M);
  static constexpr int SG_N = ceil_div(BLK_N, SG_NUMS_N);
  static constexpr int SG_K = ceil_div(BLK_K, SG_NUMS_K);
  using SubgroupTileShape = Shape<C<SG_M>, C<SG_N>, C<SG_K>>;
  static constexpr auto Num_SGs = SG_NUMS_M * SG_NUMS_N * SG_NUMS_K;
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  static constexpr int NvFp4GroupK = 16;
  static constexpr int MxFp8GroupK = 32;
  static_assert(SG_NUMS_K == 1, "NVFP4 BF16 scale mapping assumes one subgroup along K");
  static_assert(BLK_K % MxFp8GroupK == 0, "BLK_K must preserve complete prepared scale groups");

  template <class Element, class Stride>
  using TensorType = decltype(make_tensor(
      make_gmem_ptr(static_cast<Element const*>(nullptr)),
      make_layout(make_shape(int{}, int{}, int{}), Stride{})));

  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementScale const* ptr_SF = nullptr;  // (K/32, N), MN-major
    ElementMult const* ptr_CB = nullptr;   // (N, K/16), row-major
    int64_t ldCB = 0;
  };

  struct Params {
    TensorType<ElementA, StrideA> mA_mkl;
    TensorType<ElementB, StrideB> mB_nkl;
    ElementScale const* ptr_SF;
    ElementMult const* ptr_CB;
    int64_t ldCB;
  };

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void)workspace;
    auto [M, N, K, L] = problem_shape;
    auto mA_mkl = make_tensor(
        make_gmem_ptr(args.ptr_A), make_layout(make_shape(M, K, L), args.dA));
    auto mB_nkl = make_tensor(
        make_gmem_ptr(args.ptr_B), make_layout(make_shape(N, K, L), args.dB));
    return Params{mA_mkl, mB_nkl, args.ptr_SF, args.ptr_CB, args.ldCB};
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape problem_shapes, Arguments const& args) {
    auto [M, N, K, L] = append<4>(problem_shapes, 1);
    constexpr int copy_alignment_bits = 128;
    constexpr int min_aligned_elements_A = copy_alignment_bits / sizeof_bits<ElementA>::value;
    constexpr int min_aligned_elements_B = copy_alignment_bits / sizeof_bits<ElementB>::value;
    bool implementable =
        cutlass::detail::check_alignment<min_aligned_elements_A>(
            cute::make_shape(M, K, L), args.dA) &&
        cutlass::detail::check_alignment<min_aligned_elements_B>(
            cute::make_shape(N, K, L), args.dB) &&
        K % BLK_K == 0;
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: NVFP4 BF16 alignment requirements not met.\n");
    }
    return implementable;
  }

  CUTLASS_DEVICE static float ue8m0_to_float(ElementScale value) {
    // Prepared exponents cover [-127, 127]. IEEE's exponent field directly
    // represents all but -127, which is the exactly representable subnormal
    // 2^-127.
    const uint32_t bits = value == 0 ? 0x00400000u : static_cast<uint32_t>(value) << 23;
    return sycl::bit_cast<float>(bits);
  }

  template <class FragB, class CoordFrag>
  CUTLASS_DEVICE static void apply_nvfp4_scales(
      FragB& frag,
      CoordFrag const& coords,
      ElementScale const* sfmx,
      ElementMult const* cb,
      int64_t cb_ld,
      int n_extent,
      int n_tile_base,
      int k_tile_base) {
    using FragLayout = typename FragB::layout_type;
    constexpr int V = decltype(size<0>(FragLayout{}))::value;
    constexpr int N_IT = decltype(size<1>(FragLayout{}))::value;
    constexpr int K_IT = decltype(size<2>(FragLayout{}))::value;
    static_assert(V == NvFp4GroupK,
                  "BF16 MMA-B must expose 16 values per atom iteration");
    static_assert(decltype(stride<1>(FragLayout{}))::value == V,
                  "MMA-B fragment must be compact in N iteration");
    static_assert(decltype(stride<2>(FragLayout{}))::value == V * N_IT,
                  "MMA-B fragment must be compact in K iteration");

    CUTLASS_PRAGMA_UNROLL
    for (int ki = 0; ki < K_IT; ++ki) {
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < N_IT; ++ni) {
        const int base = V * (ni + N_IT * ki);

        // A BF16 DPAS B vector alternates between two N rows. Its even and
        // odd values each cover eight K positions in the same K/16 group.
        auto coord0 = coords(base);
        auto coord1 = coords(base + 1);
        const int n0 = n_tile_base + get<0>(coord0);
        const int n1 = n_tile_base + get<0>(coord1);
        const int k0 = k_tile_base + get<1>(coord0);
        const int k1 = k_tile_base + get<1>(coord1);

        ElementMult scale0(0.0f);
        ElementMult scale1(0.0f);
        if (n0 < n_extent) {
          const float c = static_cast<float>(
              cb[static_cast<int64_t>(n0) * cb_ld + k0 / NvFp4GroupK]);
          const float power_of_two_xe35 = ue8m0_to_float(
              sfmx[static_cast<int64_t>(k0 / MxFp8GroupK) * n_extent + n0]);
          scale0 = ElementMult(c * power_of_two_xe35);
        }
        if (n1 < n_extent) {
          const float c = static_cast<float>(
              cb[static_cast<int64_t>(n1) * cb_ld + k1 / NvFp4GroupK]);
          const float power_of_two_xe35 = ue8m0_to_float(
              sfmx[static_cast<int64_t>(k1 / MxFp8GroupK) * n_extent + n1]);
          scale1 = ElementMult(c * power_of_two_xe35);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < V; ++v) {
          frag(base + v) = frag(base + v) * ((v & 1) ? scale1 : scale0);
        }
      }
    }
  }

  template <
      class FrgTensorD,
      class TensorA,
      class TensorB,
      class FrgTensorC,
      class KTileIterator,
      class BlkCoord>
  CUTLASS_DEVICE void operator()(
      FrgTensorD& accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const& src_accum,
      KTileIterator k_tile_iter,
      int k_tile_count,
      BlkCoord const& blk_coord,
      int const& K_start,
      int thread_idx,
      Params const& mainloop) {
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident");
    (void)src_accum;

    auto batch_idx = get<3>(blk_coord);
    auto copy_a =
        get_block_2d_copy_A<GmemTiledCopyA>(TiledMma{}, mainloop.mA_mkl(_, _, batch_idx));
    auto copy_b =
        get_block_2d_copy_B<GmemTiledCopyB>(TiledMma{}, mainloop.mB_nkl(_, _, batch_idx));
    auto thr_copy_a = copy_a.get_slice(thread_idx);
    auto thr_copy_b = copy_b.get_slice(thread_idx);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    auto coord_tile =
        make_identity_tensor(make_shape(Int<BLK_N>{}, Int<BLK_K>{}));
    auto tCrB_coord = thr_mma.partition_B(coord_tile);

    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
    auto pAgA = prefetch_a.get_slice(thread_idx).partition_S(gA);
    auto pBgB = prefetch_b.get_slice(thread_idx).partition_S(gB);

    const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));
    auto k_tile_delta = make_coord(_0{}, Int<BLK_K>{});
    auto prepared_a = sgl_xpu_xe35::prepare_payloads_xe35(copy_a, tAgA(_, _, _, k_start_idx));
    auto prepared_b = sgl_xpu_xe35::prepare_payloads_xe35(copy_b, tBgB(_, _, _, k_start_idx));
    auto prepared_pa =
        sgl_xpu_xe35::prepare_payloads_xe35(prefetch_a, pAgA(_, _, _, k_start_idx));
    auto prepared_pb =
        sgl_xpu_xe35::prepare_payloads_xe35(prefetch_b, pBgB(_, _, _, k_start_idx));

    int prefetch_k = k_start_idx;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; ++i, ++prefetch_k) {
      sgl_xpu_xe35::prefetch_payloads_xe35(prefetch_a, prepared_pa);
      sgl_xpu_xe35::prefetch_payloads_xe35(prefetch_b, prepared_pb);
      prepared_pa += k_tile_delta;
      prepared_pb += k_tile_delta;
    }

    const int n_extent = size<0>(mainloop.mB_nkl.shape());
    const int n_tile_base = get<1>(blk_coord) * BLK_N;
    for (int k_tile = k_start_idx;
         k_tile < k_tile_count + k_start_idx;
         ++k_tile, ++prefetch_k) {
      barrier_arrive(ScopeWorkgroup);
      sgl_xpu_xe35::copy_payloads_xe35(copy_a, prepared_a, tArA);
      sgl_xpu_xe35::copy_payloads_xe35(copy_b, prepared_b, tBrB);
      prepared_a += k_tile_delta;
      prepared_b += k_tile_delta;

      if (prefetch_k < k_tile_count + k_start_idx) {
        sgl_xpu_xe35::prefetch_payloads_xe35(prefetch_a, prepared_pa);
        sgl_xpu_xe35::prefetch_payloads_xe35(prefetch_b, prepared_pb);
        prepared_pa += k_tile_delta;
        prepared_pb += k_tile_delta;
      }

      reorder(tArA, tCrA);
      // CUTLASS has native Xe reorder atoms for E2M1 -> BF16. This converts
      // and permutes directly from the packed-copy fragment to MMA-B.
      reorder(tBrB, tCrB);
      apply_nvfp4_scales(
          tCrB,
          tCrB_coord,
          mainloop.ptr_SF,
          mainloop.ptr_CB,
          mainloop.ldCB,
          n_extent,
          n_tile_base,
          k_tile * BLK_K);
      cute::gemm(tiled_mma, tCrA, tCrB, accum);
      barrier_wait(ScopeWorkgroup);
    }
  }
};

template <
    int Stages,
    class Schedule,
    class TileShape_,
    class ElementA_,
    class StrideA_,
    class ElementB_,
    class StrideB_,
    class TiledMma_,
    class GmemTiledCopyA_,
    class SmemLayoutAtomA_,
    class SmemCopyAtomA_,
    class TransformA_,
    class GmemTiledCopyB_,
    class SmemLayoutAtomB_,
    class SmemCopyAtomB_,
    class TransformB_>
struct CollectiveMma<
    sgl_xpu_xe35::MainloopXeNvFp4WoqBf16Group_xe35<Stages, Schedule>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
    : public CollectiveMma<
          sgl_xpu_xe35::MainloopXeNvFp4WoqBf16_xe35<Stages, cutlass::gemm::KernelXe>,
          TileShape_,
          ElementA_,
          StrideA_,
          ElementB_,
          StrideB_,
          TiledMma_,
          GmemTiledCopyA_,
          SmemLayoutAtomA_,
          SmemCopyAtomA_,
          TransformA_,
          GmemTiledCopyB_,
          SmemLayoutAtomB_,
          SmemCopyAtomB_,
          TransformB_> {
 public:
  using Base = CollectiveMma<
      sgl_xpu_xe35::MainloopXeNvFp4WoqBf16_xe35<Stages, cutlass::gemm::KernelXe>,
      TileShape_,
      ElementA_,
      StrideA_,
      ElementB_,
      StrideB_,
      TiledMma_,
      GmemTiledCopyA_,
      SmemLayoutAtomA_,
      SmemCopyAtomA_,
      TransformA_,
      GmemTiledCopyB_,
      SmemLayoutAtomB_,
      SmemCopyAtomB_,
      TransformB_>;
  using BaseArguments = typename Base::Arguments;
  using DispatchPolicy =
      sgl_xpu_xe35::MainloopXeNvFp4WoqBf16Group_xe35<Stages, Schedule>;
  using ElementA = typename Base::ElementA;
  using ElementB = typename Base::ElementB;
  using ElementScale = typename Base::ElementScale;
  using ElementMult = typename Base::ElementMult;
  using StrideA = StrideA_;
  using StrideB = StrideB_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;

  struct Arguments {
    ElementA const* ptr_A;
    ElementB const* ptr_B;
    ElementScale const* ptr_SF = nullptr;
    ElementMult const* ptr_CB = nullptr;
  };

  using Params = Arguments;

  CollectiveMma() = default;

  static constexpr Params to_underlying_arguments(Arguments const& args) {
    return args;
  }

  template <typename ProblemShapeMNKL>
  CUTLASS_DEVICE static constexpr BaseArguments to_base_arguments(
      Arguments const& args,
      int group,
      ProblemShapeMNKL const& problem_shape,
      int64_t expert_first_token_offset,
      int64_t /*expert_first_scale_offset*/) {
    const int32_t M = get<0>(problem_shape);
    const int32_t N = get<1>(problem_shape);
    const int32_t K = get<2>(problem_shape);

    auto ptr_A = args.ptr_A + expert_first_token_offset * K;
    auto ptr_B =
        args.ptr_B + static_cast<int64_t>(group) * N * K / 2;
    auto ptr_SF =
        args.ptr_SF + static_cast<int64_t>(group) * N * (K / Base::MxFp8GroupK);
    auto ptr_CB =
        args.ptr_CB + static_cast<int64_t>(group) * N * (K / Base::NvFp4GroupK);
    auto dA = cutlass::make_cute_packed_stride(
        InternalStrideA{}, cute::make_shape(M, K, 1));
    auto dB = cutlass::make_cute_packed_stride(
        InternalStrideB{}, cute::make_shape(N, K, 1));

    return BaseArguments{
        ptr_A,
        dA,
        ptr_B,
        dB,
        ptr_SF,
        ptr_CB,
        K / Base::NvFp4GroupK};
  }

  template <class ProblemShape>
  static bool can_implement(int64_t, int64_t K, Arguments const&) {
    return K % Base::BLK_K == 0;
  }
};

}  // namespace cutlass::gemm::collective
