/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief Fused NVFP4 weight-only-quantized MXFP8 block-scaled mainloop for Xe3P.

    Xe3P's `bdpas` block-scaled MMA only accepts UE8M0 scale factors with a
    fixed block size of 32.  NVFP4 (E2M1 payload, E4M3 scales per 16, one FP32
    per-tensor global scale) is therefore emulated: B is fetched from memory as
    4-bit NVFP4, dequantized to BF16 and requantized to MXFP8 **in registers**,
    and the MMA runs as a plain MXFP8 x MXFP8 `bdpas`.  A is already MXFP8.

    Weight traffic stays at NVFP4 density (0.5 B/element) — the point of doing
    this in the mainloop rather than materializing MXFP8 weights, which would
    double weight footprint and bandwidth.

    Numerics.  A one-time host-side prep pass (nvfp4_woq_prep_xe35, see
    src/sycl/nvfp4_woq.cpp) splits the NVFP4 scales into

      sfmx (N, K/32) UE8M0   the requantization scale 2^k, k minimal such that
                             448 * 2^k >= 6 * max(s_2j, s_2j+1)
      c    (N, K/16) BF16    per-16 multiplier gs_dq * sf16_j * 2^-k

    so the mainloop only has to evaluate, per element,

      w_mx = e4m3_rne(float(c[n][k/16]) * float(e2m1_val))

    and hand `sfmx` to `bdpas` unchanged through the ordinary block-scaled
    scale-load path.  `6` is the largest E2M1 magnitude, so the bound is
    conservative and requantization never clips; NVFP4 quantizers set
    sf16 = block_amax / 6, which makes it tight.

    Layout notes.
      * B data is loaded with a *shadow* TiledMMA built on the E2M1 atom over
        the same workgroup tile, because the 4-bit block-2D load and fragment
        differ from the E4M3 ones.  A single cute::reorder moves the payload
        from the E2M1 copy layout straight into the E4M3 MMA-B fragment layout
        (which needs cross-lane movement — reorder handles it).
      * `c` is indexed directly from the E4M3 MMA-B fragment coordinates.  For
        XE_DPAS_TT with 8-bit B, fragment element (v, n_iter, k_iter) in lane t
        holds
            n = 16*n_iter + 4*(v%4) + t/4
            k = 32*k_iter + (t%4)   + 4*(v/4)
        so the 16-block index k/16 == 2*k_iter + v/16 is lane-independent and
        the N index depends on the lane only through the fixed offset t/4.
        `c` is kept row-major (N, K/16) so a lane's K/16 entries for one N are
        contiguous: one short vector load per N slot per k-tile.
*/

#pragma once

// clang-format off
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_xe.hpp>
#include <cute/util/compat/memory.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/collective/xe_common_blockscaled_mxfp.hpp"

#include "sycl/xe35/nvfp4/collective/moe_array_mma.hpp"
#include "xe_subbyte_upconvert.hpp"

namespace sgl_xpu_xe35 {

enum class NvFp4WideCache_xe35 {
  None,
  L1,
  Slm,
};

/// Single-GEMM dispatch policy for the NVFP4 -> MXFP8 WoQ mainloop.
template <
    int Stages_,
    bool QuantizeInKLoop_ = false,
    NvFp4WideCache_xe35 CacheMode_ = NvFp4WideCache_xe35::None,
    class KernelSchedule = cutlass::gemm::KernelXe>
struct MainloopXeNvFp4WoqMxFp8_xe35
    : cutlass::gemm::MainloopIntelXeXMX16<Stages_, KernelSchedule> {
  static constexpr bool QuantizeInKLoop = QuantizeInKLoop_;
  static constexpr NvFp4WideCache_xe35 CacheMode = CacheMode_;
};

/// Grouped (ptr-array) dispatch policy; `Base` above does the work.
template <
    int Stages_,
    bool QuantizeInKLoop_ = false,
    NvFp4WideCache_xe35 CacheMode_ = NvFp4WideCache_xe35::None,
    class KernelSchedule = cutlass::gemm::KernelMoEArrayCooperative_xe35>
struct MainloopXeNvFp4WoqMxFp8Group_xe35
    : cutlass::gemm::MainloopIntelXeXMX16<Stages_, KernelSchedule> {
  static constexpr bool QuantizeInKLoop = QuantizeInKLoop_;
  static constexpr NvFp4WideCache_xe35 CacheMode = CacheMode_;
};

/// Extract the warp (subgroup) layout from an already-built TiledMMA so the
/// shadow E2M1 TiledMMA can be tiled identically.
template <class T>
struct tiled_mma_warp_layout;
template <class Atom, class WarpLayout_, class Permutation_>
struct tiled_mma_warp_layout<cute::TiledMMA<Atom, WarpLayout_, Permutation_>> {
  using type = WarpLayout_;
};

}  // namespace sgl_xpu_xe35

namespace cutlass::gemm::collective {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
// Single-GEMM (base) specialization
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  bool QuantizeInKLoop,
  sgl_xpu_xe35::NvFp4WideCache_xe35 CacheMode,
  class KernelSchedule,
  class TileShape_,
  class ElementPairA_,
  class StridePairA_,
  class ElementPairB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    sgl_xpu_xe35::MainloopXeNvFp4WoqMxFp8_xe35<
        Stages, QuantizeInKLoop, CacheMode, KernelSchedule>,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
 public:
  using DispatchPolicy =
      sgl_xpu_xe35::MainloopXeNvFp4WoqMxFp8_xe35<
          Stages, QuantizeInKLoop, CacheMode, KernelSchedule>;
  using WorkgroupTileShape = TileShape_;

  using TiledMma = TiledMma_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;

  // A: MXFP8 (E4M3 payload, UE8M0 per-32 scales).
  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using StrideA  = cute::remove_pointer_t<remove_cvref_t<decltype(get<0>(StridePairA{}))>>;
  // B: NVFP4 payload (E2M1) with UE8M0 per-32 requantization scales.
  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB  = cute::remove_pointer_t<remove_cvref_t<decltype(get<0>(StridePairB{}))>>;

  using ElementSF     = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using ElementScaleA = ElementSF;
  using ElementScaleB = ElementSF;
  using StrideScaleA  = cute::remove_pointer_t<remove_cvref_t<decltype(get<1>(StridePairA{}))>>;
  using StrideScaleB  = cute::remove_pointer_t<remove_cvref_t<decltype(get<1>(StridePairB{}))>>;

  // Per-16 multiplier: BF16, row-major (N, K/16).
  using ElementMult = cutlass::bfloat16_t;

  using ElementAMma = typename TiledMma::ValTypeA;   // E4M3 — what bdpas sees
  using ElementAccumulator = typename TiledMma::ValTypeC;

  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;

  using GmemTiledCopyA      = typename std::tuple_element<0, GmemTiledCopyPairA_>::type;
  using GmemTiledCopyScaleA = typename std::tuple_element<1, GmemTiledCopyPairA_>::type;
  using GmemTiledCopyB      = typename std::tuple_element<0, GmemTiledCopyPairB_>::type;
  using GmemTiledCopyScaleB = typename std::tuple_element<1, GmemTiledCopyPairB_>::type;

  static_assert(cute::is_same_v<ElementAMma, cutlass::float_e4m3_t>,
                "NVFP4 WoQ mainloop targets an E4M3 (MXFP8) bdpas MMA");
  static_assert(cute::is_same_v<ElementB, cutlass::float_e2m1_t>,
                "NVFP4 WoQ mainloop expects a packed E2M1 B operand");
  static_assert(sizeof_bits_v<ElementSF> == 8, "Block scales must be 8-bit (UE8M0)");
  static_assert(std::is_same_v<TransformA_, cute::identity> && std::is_same_v<TransformB_, cute::identity>,
                "Operand transforms are not supported");

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;

  static constexpr int BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr int BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr int BLK_K = get<2>(WorkgroupTileShape{});

  static constexpr int SG_NUMS_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int SG_NUMS_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int SG_NUMS_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());

  static constexpr int MMA_K = get<2>(typename TiledMma::Shape_MNK{});

  static constexpr int SG_M = ceil_div(BLK_M, SG_NUMS_M);
  static constexpr int SG_N = ceil_div(BLK_N, SG_NUMS_N);
  static constexpr int SG_K = ceil_div(BLK_K, SG_NUMS_K);
  using SubgroupTileShape = Shape<C<SG_M>, C<SG_N>, C<SG_K>>;

  static constexpr int GroupK = 32;              // MXFP8 / bdpas block size
  static constexpr int MultGroupK = 16;          // NVFP4 block size
  static constexpr int KG_PER_TILE = SG_K / MultGroupK;

  static_assert(SG_K >= GroupK, "Xe block-scaled MMA requires SG_K >= 32");
  static_assert(SG_K % MultGroupK == 0, "SG_K must be a multiple of the NVFP4 block size");
  static_assert(SG_NUMS_K == 1,
                "NVFP4 WoQ mainloop indexes the per-16 multiplier assuming one subgroup along K");
  static_assert(BLK_K % MultGroupK == 0, "BLK_K must be a multiple of 16");
  // requantize_B's `2 * k_it + (v >> 4)` derivation of the 16-block index, and
  // its `4 * n_it + (v & 3)` N slot, both assume the 8-bit XE_DPAS_TT B layout
  // (MMA_K = 32, atom N = 16) against NVFP4's block size of 16.
  static_assert(MMA_K == 2 * MultGroupK, "requantize_B assumes MMA_K == 32 with 16-element blocks");
  static_assert(GroupK == 2 * MultGroupK, "MXFP8 block must be exactly two NVFP4 blocks");

  static constexpr auto Num_SGs = SG_NUMS_M * SG_NUMS_N * SG_NUMS_K;
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  // 2D block loads need a 4-byte aligned surface width; callers pad the scale
  // buffers' M/N extent up to this.
  static constexpr int ScaleAlignElems = cute::ceil_div(4, (int)(sizeof_bits_v<ElementSF> / 8));

  template <class Element, class Stride>
  using TensorType = decltype(make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                                          make_layout(make_shape(int{}, int{}, int{}), Stride{})));

  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementScaleA const* ptr_SA = nullptr;
    StrideScaleA dSA{};
    ElementScaleB const* ptr_SB = nullptr;
    StrideScaleB dSB{};
    ElementMult const* ptr_CB = nullptr;  // (N, K/16) row-major BF16
    int64_t ldCB = 0;                     // row stride of ptr_CB, in elements
    ElementMult* ptr_WideCache = nullptr; // Per-workgroup L1-backed BF16 scratch
  };

  struct Params {
    TensorType<ElementA, StrideA> mA_mkl;
    TensorType<ElementB, StrideB> mB_nkl;
    TensorType<ElementScaleA, StrideScaleA> mAscale;
    TensorType<ElementScaleB, StrideScaleB> mBscale;
    ElementMult const* ptr_CB;
    int64_t ldCB;
    ElementMult* ptr_WideCache;
  };

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void)workspace;
    auto [M, N, K, L] = problem_shape;

    auto mA_mkl = make_tensor(make_gmem_ptr(args.ptr_A), make_layout(make_shape(M, K, L), args.dA));
    auto mB_nkl = make_tensor(make_gmem_ptr(static_cast<ElementB const*>(args.ptr_B)),
                              make_layout(make_shape(N, K, L), args.dB));
    auto scale_k = cute::ceil_div(K, GroupK);
    auto mAscale = make_tensor(make_gmem_ptr(static_cast<ElementScaleA const*>(args.ptr_SA)),
                               make_layout(make_shape(M, scale_k, L), args.dSA));
    auto mBscale = make_tensor(make_gmem_ptr(static_cast<ElementScaleB const*>(args.ptr_SB)),
                               make_layout(make_shape(N, scale_k, L), args.dSB));
    return Params{
        mA_mkl,
        mB_nkl,
        mAscale,
        mBscale,
        args.ptr_CB,
        args.ldCB,
        args.ptr_WideCache};
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape problem_shapes, Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shapes, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    bool implementable = true;
    constexpr int copy_alignment_bits = 128;
    constexpr int min_aligned_elements_A = copy_alignment_bits / sizeof_bits<ElementA>::value;
    implementable &= cutlass::detail::check_alignment<min_aligned_elements_A>(cute::make_shape(M, K, L), args.dA);
    constexpr int min_aligned_elements_B = copy_alignment_bits / sizeof_bits<ElementB>::value;
    implementable &= cutlass::detail::check_alignment<min_aligned_elements_B>(cute::make_shape(N, K, L), args.dB);
    implementable &= (M % ScaleAlignElems == 0) && (N % ScaleAlignElems == 0);
    implementable &= (K % BLK_K == 0);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: NVFP4 WoQ alignment requirements not met.\n");
    }
    return implementable;
  }

  /// Maps an E4M3 MMA-B fragment's flat index to its per-16 multiplier.
  ///
  /// The fragment is compact over (V=32, N_ITER, K_ITER) with V flat across the
  /// atom's nested (4, 8) value mode, so flat index i decomposes as
  ///   v = i % 32,  n_it = (i / 32) % N_ITER,  k_it = i / (32 * N_ITER)
  /// (checked by assert_b_fragment_layout below), and per the layout note at the
  /// top of this file the N slot advances with v % 4 while the 16-block index is
  /// 2 * k_it + v / 16, which is lane-independent.
  template <int NSlots, int KGroups, int N_ITER>
  struct PerBlockScale {
    float const (*cvals)[KGroups];

    CUTLASS_DEVICE float operator()(int i) const {
      const int v = i % 32;
      const int n_it = (i / 32) % N_ITER;
      const int k_it = i / (32 * N_ITER);
      return cvals[4 * n_it + (v & 3)][2 * k_it + (v >> 4)];
    }
  };

  /// Scale lookup for one 32-wide MMA-K slice (two NVFP4 scale blocks).
  template <int NSlots, int N_ITER>
  struct PerMmaScale {
    float const (*cvals)[2];

    CUTLASS_DEVICE float operator()(int i) const {
      const int v = i % 32;
      const int n_it = (i / 32) % N_ITER;
      return cvals[4 * n_it + (v & 3)][v >> 4];
    }
  };

  /// Fail loudly if a tile or atom change invalidates PerBlockScale's decode.
  template <class Frag>
  CUTLASS_DEVICE static void assert_b_fragment_layout(Frag const&) {
    using FragLayout = typename Frag::layout_type;
    constexpr int V = decltype(size<0>(FragLayout{}))::value;
    constexpr int N_IT = decltype(size<1>(FragLayout{}))::value;
    static_assert(V == 32, "E4M3 MMA-B atom must hold 32 values per lane");
    static_assert(decltype(stride<1>(FragLayout{}))::value == 32,
                  "MMA-B fragment must be compact: N-iteration stride must be 32");
    static_assert(decltype(stride<2>(FragLayout{}))::value == 32 * N_IT,
                  "MMA-B fragment must be compact: K-iteration stride must be 32 * N_ITER");
  }

  /// Requantize one K32 slice from a widened row-major BF16 cache tile.
  template <class WidePtr, class FragB, class CoordFrag>
  CUTLASS_DEVICE static void requantize_cached_b_k(
      WidePtr wide_cache,
      FragB& frag,
      CoordFrag const& coords,
      int mma_k) {
    using DstType = typename FragB::value_type;
    constexpr int N_IT = decltype(size<1>(typename FragB::layout_type{}))::value;
    constexpr int kMmaValues = 32 * N_IT;
    const int fragment_base = mma_k * kMmaValues;

    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < N_IT; ++ni) {
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < 32; v += 2) {
        const int idx = fragment_base + 32 * ni + v;
#ifdef __SYCL_DEVICE_ONLY__
        sycl::detail::float16_vec2 values;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 2; ++j) {
          auto coord = coords(idx + j);
          const int n = get<0>(coord);
          const int k = get<1>(coord);
          values[j] = static_cast<_Float16>(
              static_cast<float>(wide_cache[n * BLK_K + k]));
        }
        auto converted =
            __builtin_spirv_ClampConvertFP16ToE4M3INTEL(values);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 2; ++j) {
          frag(idx + j) = DstType::bitcast(converted[j]);
        }
#else
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 2; ++j) {
          auto coord = coords(idx + j);
          const int n = get<0>(coord);
          const int k = get<1>(coord);
          frag(idx + j) =
              DstType(static_cast<float>(wide_cache[n * BLK_K + k]));
        }
#endif
      }
    }
  }

  /// Perform a subgroup-scoped matrix multiply-accumulate.
  template <class FrgTensorD, class TensorA, class TensorB, class FrgTensorC, class KTileIterator, class BlkCoord>
  CUTLASS_DEVICE void
  operator()(
      FrgTensorD& accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const& src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      BlkCoord const& blk_coord,
      int const& K_start,
      int thread_idx,
      Params const& mainloop) {
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    (void)src_accum;

    auto batch_idx = get<3>(blk_coord);

    // Both copies are built from the E4M3 TiledMMA. For B the block-2D
    // selector sees a 4-bit memory type against an 8-bit register type
    // (block_2d_selector<MemType, RegType>) and picks a resizing 4-bit load
    // whose destination fragment is derived from the E4M3 MMA-B layout, which
    // keeps the relayout below cheap.
    auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(TiledMma{}, mainloop.mA_mkl(_, _, batch_idx));
    auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(TiledMma{}, mainloop.mB_nkl(_, _, batch_idx));

    auto thr_copy_a = copy_a.get_slice(thread_idx);
    auto thr_copy_b = copy_b.get_slice(thread_idx);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));   // E4M3, MMA-B layout
    auto tCrBWideK = make_tensor<ElementMult>(tCrB(_, _, 0).layout());
    auto b_coord_tile =
        make_identity_tensor(make_shape(Int<BLK_N>{}, Int<BLK_K>{}));
    auto tCrBCoord = thr_mma.partition_B(b_coord_tile);

    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
    auto pAgA = prefetch_a.get_slice(thread_idx).partition_S(gA);
    auto pBgB = prefetch_b.get_slice(thread_idx).partition_S(gB);

    using GemmIterM = Int<decltype(size<1>(tCrA.shape()))::value>;
    using GemmIterN = Int<decltype(size<1>(tCrB.shape()))::value>;
    using GemmIterK = Int<decltype(size<2>(tCrB.shape()))::value>;
    static constexpr int N_ITER = GemmIterN::value;
    static constexpr int N_SLOTS = 4 * N_ITER;
    using TileBScale = PerBlockScale<N_SLOTS, KG_PER_TILE, N_ITER>;
    using MmaBScale = PerMmaScale<N_SLOTS, N_ITER>;
    assert_b_fragment_layout(tCrB);
    static_assert(KG_PER_TILE == 2 * GemmIterK::value,
                  "Each E4M3 MMA-K step must consume two NVFP4 scale blocks");

    auto [m_idx, n_idx, k_idx, l_idx] = blk_coord;
    const int m_coord = m_idx * BLK_M + (get_sub_group_id() / SG_NUMS_N) * SG_M;
    const int n_coord = n_idx * BLK_N + (get_sub_group_id() % SG_NUMS_N) * SG_N;
    const int l_coord = l_idx;
    const int lane_id = thread_idx % SubgroupSize;
    const bool wide_cache_producer =
        get_sub_group_id() < static_cast<uint32_t>(SG_NUMS_N);

    ElementMult* wide_cache = mainloop.ptr_WideCache;
    if constexpr (DispatchPolicy::CacheMode ==
                  sgl_xpu_xe35::NvFp4WideCache_xe35::L1) {
      const int workgroup_idx =
          BlockIdxX() + BlockIdxY() * GridDimX();
      wide_cache +=
          static_cast<int64_t>(workgroup_idx) * BLK_N * BLK_K;
    } else if constexpr (DispatchPolicy::CacheMode ==
                         sgl_xpu_xe35::NvFp4WideCache_xe35::Slm) {
      wide_cache =
          compat::local_mem<ElementMult[BLK_N * BLK_K]>();
    }

    const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));
    constexpr int k_reload_factor = cute::max(GroupK / BLK_K, 1);

    auto [tiled_copy_scaleA, copy_iter_scaleA, fragment_scaleA] =
        make_scaled_copy<GmemTiledCopyScaleA, ElementScaleA, SG_M, SG_K, GroupK>(
            mainloop.mAscale, m_coord, l_coord, k_tile_count);
    auto [tiled_copy_scaleB, copy_iter_scaleB, fragment_scaleB] =
        make_scaled_copy<GmemTiledCopyScaleB, ElementScaleB, SG_N, SG_K, GroupK>(
            mainloop.mBscale, n_coord, l_coord, k_tile_count);
    auto [scale_m_offsets, scale_n_offsets, scale_ak_offsets, scale_bk_offsets] =
        make_scaled_offsets<GemmIterM::value, GemmIterN::value, GemmIterK::value, MMA_K, GroupK,
                            typename decltype(tiled_copy_scaleA)::BlockShape,
                            typename decltype(tiled_copy_scaleB)::BlockShape>();
    auto [tiled_prefetch_scaleA, prefetch_iter_scaleA] =
        make_scaled_prefetch<decltype(tiled_copy_scaleA), SG_M, SG_K, GroupK>(
            tiled_copy_scaleA, m_coord, l_coord, k_tile_count);
    auto [tiled_prefetch_scaleB, prefetch_iter_scaleB] =
        make_scaled_prefetch<decltype(tiled_copy_scaleB), SG_N, SG_K, GroupK>(
            tiled_copy_scaleB, n_coord, l_coord, k_tile_count);

    using scaleA_vec_t = intel::vector_t<ElementScaleA, decltype(size(fragment_scaleA))::value>;
    using scaleB_vec_t = intel::vector_t<ElementScaleB, decltype(size(fragment_scaleB))::value>;

    // Per-16 multiplier addressing. Slot s covers fragment N index
    //   n = n_coord + 16*(s/4) + 4*(s%4) + lane/4,
    // and the lane's KG_PER_TILE entries for that N are contiguous in the
    // row-major (N, K/16) buffer.
    // Row offsets are clamped into range so the loads below are always legal;
    // out-of-range N columns then get a zero multiplier. The epilogue masks
    // those columns anyway, but keeping them finite avoids poisoning anything.
    const int N_extent = size<0>(mainloop.mB_nkl.shape());
    ElementMult const* cb_ptr = mainloop.ptr_CB;
    const int64_t cb_ld = mainloop.ldCB;
    int64_t c_row_base[N_SLOTS];
    bool c_row_oob[N_SLOTS];
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < N_SLOTS; ++s) {
      const int n = n_coord + 16 * (s / 4) + 4 * (s % 4) + (lane_id / 4);
      c_row_oob[s] = (n >= N_extent);
      c_row_base[s] = static_cast<int64_t>(cute::min(n, N_extent - 1)) * cb_ld;
    }

    int prefetch_k = k_start_idx;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prefetch_k++) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      if constexpr (DispatchPolicy::CacheMode ==
                    sgl_xpu_xe35::NvFp4WideCache_xe35::None) {
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      } else if (wide_cache_producer) {
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }
      prefetch(tiled_prefetch_scaleA, prefetch_iter_scaleA(_, _, _, prefetch_k / k_reload_factor));
      prefetch(tiled_prefetch_scaleB, prefetch_iter_scaleB(_, _, _, prefetch_k / k_reload_factor));
    }

    for (int k_tile = k_start_idx; k_tile < k_tile_count + k_start_idx; k_tile++, prefetch_k++) {
      copy(copy_a, tAgA(_, _, _, k_tile), tArA);
      if constexpr (DispatchPolicy::CacheMode ==
                    sgl_xpu_xe35::NvFp4WideCache_xe35::None) {
        copy(copy_b, tBgB(_, _, _, k_tile), tBrB);
      } else if (wide_cache_producer) {
        copy(copy_b, tBgB(_, _, _, k_tile), tBrB);
      }
      copy(tiled_copy_scaleA, copy_iter_scaleA(_, _, _, k_tile / k_reload_factor), fragment_scaleA);
      copy(tiled_copy_scaleB, copy_iter_scaleB(_, _, _, k_tile / k_reload_factor), fragment_scaleB);

      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      if constexpr (DispatchPolicy::CacheMode ==
                    sgl_xpu_xe35::NvFp4WideCache_xe35::None) {
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      } else if (wide_cache_producer) {
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }
      prefetch(tiled_prefetch_scaleA, prefetch_iter_scaleA(_, _, _, prefetch_k / k_reload_factor));
      prefetch(tiled_prefetch_scaleB, prefetch_iter_scaleB(_, _, _, prefetch_k / k_reload_factor));

      reorder(tArA, tCrA);

      Tensor scaleA = make_tensor(recast<scaleA_vec_t>(fragment_scaleA).data(),
                                  make_layout(Shape<_1, GemmIterM, _1>{}, Stride<_1, _0, _0>{}));
      Tensor scaleB = make_tensor(recast<scaleB_vec_t>(fragment_scaleB).data(),
                                  make_layout(Shape<_1, GemmIterN, _1>{}, Stride<_1, _0, _0>{}));

      if constexpr (DispatchPolicy::CacheMode !=
                    sgl_xpu_xe35::NvFp4WideCache_xe35::None) {
        // Only one row of N subgroups decodes B. The resulting BF16 tile is
        // shared by every M subgroup through either L1-backed scratch or SLM.
        if (wide_cache_producer) {
          CUTLASS_PRAGMA_UNROLL
          for (int mma_k = 0; mma_k < GemmIterK::value; ++mma_k) {
            const int64_t kg_base =
                static_cast<int64_t>(k_tile) * KG_PER_TILE + 2 * mma_k;
            float cvals[N_SLOTS][2];
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < N_SLOTS; ++s) {
              ElementMult const* row =
                  cb_ptr + c_row_base[s] + kg_base;
              CUTLASS_PRAGMA_UNROLL
              for (int j = 0; j < 2; ++j) {
                cvals[s][j] =
                    c_row_oob[s] ? 0.0f : static_cast<float>(row[j]);
              }
            }

            sgl_xpu_xe35::reorder_upconvert_mma_b_k_wide_xe35(
                tBrB,
                tCrBWideK,
                mma_k,
                MmaBScale{&cvals[0]});
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tCrBWideK); ++i) {
              const int fragment_idx = i + mma_k * 32 * N_ITER;
              auto coord = tCrBCoord(fragment_idx);
              const int n = get<0>(coord);
              const int k = get<1>(coord);
              wide_cache[n * BLK_K + k] = tCrBWideK(i);
            }
          }
        }

        if constexpr (DispatchPolicy::CacheMode ==
                      sgl_xpu_xe35::NvFp4WideCache_xe35::Slm) {
          barrier_arrive(
              ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
          barrier_wait(
              ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
        } else {
          sycl::atomic_fence(
              sycl::memory_order::release,
              sycl::memory_scope::work_group);
          barrier_arrive(
              ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
          barrier_wait(
              ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
          sycl::atomic_fence(
              sycl::memory_order::acquire,
              sycl::memory_scope::work_group);
        }

        auto mma_a =
            make_zip_tensor(tCrA, scaleA, scale_m_offsets, scale_ak_offsets);
        auto mma_b =
            make_zip_tensor(tCrB, scaleB, scale_n_offsets, scale_bk_offsets);
        CUTLASS_PRAGMA_UNROLL
        for (int mma_k = 0; mma_k < GemmIterK::value; ++mma_k) {
          requantize_cached_b_k(
              wide_cache, tCrB, tCrBCoord, mma_k);
          cute::gemm(
              tiled_mma, mma_a(_, _, mma_k), mma_b(_, _, mma_k), accum);
        }

        // The producer subgroups may overwrite the cache on the next K tile
        // only after every consumer has completed its final cache load.
        if constexpr (DispatchPolicy::CacheMode ==
                      sgl_xpu_xe35::NvFp4WideCache_xe35::Slm) {
          barrier_arrive(
              ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
          barrier_wait(
              ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
        } else {
          sycl::atomic_fence(
              sycl::memory_order::release,
              sycl::memory_scope::work_group);
          barrier_arrive(
              ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
          barrier_wait(
              ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
          sycl::atomic_fence(
              sycl::memory_order::acquire,
              sycl::memory_scope::work_group);
        }
      } else if constexpr (DispatchPolicy::QuantizeInKLoop) {
        auto mma_a =
            make_zip_tensor(tCrA, scaleA, scale_m_offsets, scale_ak_offsets);
        auto mma_b =
            make_zip_tensor(tCrB, scaleB, scale_n_offsets, scale_bk_offsets);

        // Convert one 32-wide MMA-K slice immediately before its BDPAS. This
        // avoids keeping the whole converted B tile and all four per-16
        // multipliers live together across the inner K loop.
        CUTLASS_PRAGMA_UNROLL
        for (int mma_k = 0; mma_k < GemmIterK::value; ++mma_k) {
          const int64_t kg_base =
              static_cast<int64_t>(k_tile) * KG_PER_TILE + 2 * mma_k;
          float cvals[N_SLOTS][2];
          CUTLASS_PRAGMA_UNROLL
          for (int s = 0; s < N_SLOTS; ++s) {
            ElementMult const* row = cb_ptr + c_row_base[s] + kg_base;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 2; ++j) {
              cvals[s][j] =
                  c_row_oob[s] ? 0.0f : static_cast<float>(row[j]);
            }
          }

          sgl_xpu_xe35::reorder_upconvert_mma_b_k_xe35(
              tBrB, tCrB, mma_k, MmaBScale{&cvals[0]});
          cute::gemm(
              tiled_mma, mma_a(_, _, mma_k), mma_b(_, _, mma_k), accum);
        }
      } else {
        // Convert the complete K tile before entering the MMA K loop.
        const int64_t kg_base =
            static_cast<int64_t>(k_tile) * KG_PER_TILE;
        float cvals[N_SLOTS][KG_PER_TILE];
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < N_SLOTS; ++s) {
          ElementMult const* row = cb_ptr + c_row_base[s] + kg_base;
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < KG_PER_TILE; ++j) {
            cvals[s][j] =
                c_row_oob[s] ? 0.0f : static_cast<float>(row[j]);
          }
        }

        sgl_xpu_xe35::reorder_upconvert_xe35(
            tBrB, tCrB, TileBScale{&cvals[0]});
        cute::gemm(
            tiled_mma,
            make_zip_tensor(tCrA, scaleA, scale_m_offsets, scale_ak_offsets),
            make_zip_tensor(tCrB, scaleB, scale_n_offsets, scale_bk_offsets),
            accum);
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Grouped specialization for the shared persistent MoE kernel.
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  bool QuantizeInKLoop,
  sgl_xpu_xe35::NvFp4WideCache_xe35 CacheMode,
  class Schedule,
  class TileShape_,
  class ElementPairA_,
  class StridePairA_,
  class ElementPairB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    sgl_xpu_xe35::MainloopXeNvFp4WoqMxFp8Group_xe35<
        Stages, QuantizeInKLoop, CacheMode, Schedule>,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
  : public CollectiveMma<
        sgl_xpu_xe35::MainloopXeNvFp4WoqMxFp8_xe35<
            Stages,
            QuantizeInKLoop,
            CacheMode,
            cutlass::gemm::KernelXe>,
        TileShape_, ElementPairA_, StridePairA_, ElementPairB_, StridePairB_, TiledMma_,
        GmemTiledCopyPairA_, SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_,
        GmemTiledCopyPairB_, SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_>
{
 public:
  using Base = CollectiveMma<
      sgl_xpu_xe35::MainloopXeNvFp4WoqMxFp8_xe35<
          Stages,
          QuantizeInKLoop,
          CacheMode,
          cutlass::gemm::KernelXe>,
      TileShape_, ElementPairA_, StridePairA_, ElementPairB_, StridePairB_, TiledMma_,
      GmemTiledCopyPairA_, SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_,
      GmemTiledCopyPairB_, SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_>;

  using DispatchPolicy = sgl_xpu_xe35::MainloopXeNvFp4WoqMxFp8Group_xe35<
      Stages, QuantizeInKLoop, CacheMode, Schedule>;
  using BaseArguments = typename Base::Arguments;

  using ElementA = typename Base::ElementA;
  using ElementB = typename Base::ElementB;
  using ElementScaleA = typename Base::ElementScaleA;
  using ElementScaleB = typename Base::ElementScaleB;
  using ElementMult = typename Base::ElementMult;

  using StrideA = remove_cvref_t<decltype(get<0>(StridePairA_{}))>;
  using StrideB = remove_cvref_t<decltype(get<0>(StridePairB_{}))>;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;
  using StrideScaleA = remove_cvref_t<decltype(get<1>(StridePairA_{}))>;
  using StrideScaleB = remove_cvref_t<decltype(get<1>(StridePairB_{}))>;
  using InternalStrideScaleA = cute::remove_pointer_t<StrideScaleA>;
  using InternalStrideScaleB = cute::remove_pointer_t<StrideScaleB>;

  struct Arguments {
    ElementA const* ptr_A;
    ElementB const* ptr_B;
    ElementScaleA const* ptr_SA = nullptr;
    ElementScaleB const* ptr_SB = nullptr;
    ElementMult const* ptr_CB = nullptr;
    ElementMult* ptr_WideCache = nullptr;
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
      int64_t expert_first_scale_offset) {
    const int32_t M = get<0>(problem_shape);
    const int32_t N = get<1>(problem_shape);
    const int32_t K = get<2>(problem_shape);
    const int32_t scale_k = cute::ceil_div(K, Base::GroupK);
    const int32_t scale_m = (M + 3) & ~int32_t(3);

    auto ptr_A = args.ptr_A + expert_first_token_offset * K;
    auto ptr_B =
        args.ptr_B + static_cast<int64_t>(group) * N * K / 2;
    auto ptr_SA =
        args.ptr_SA + expert_first_scale_offset * scale_k;
    auto ptr_SB =
        args.ptr_SB + static_cast<int64_t>(group) * N * scale_k;
    auto ptr_CB =
        args.ptr_CB + static_cast<int64_t>(group) * N * (K / Base::MultGroupK);

    auto dA = cutlass::make_cute_packed_stride(
        InternalStrideA{}, cute::make_shape(M, K, 1));
    auto dB = cutlass::make_cute_packed_stride(
        InternalStrideB{}, cute::make_shape(N, K, 1));
    auto dSA = cutlass::make_cute_packed_stride(
        InternalStrideScaleA{}, cute::make_shape(scale_m, scale_k, 1));
    auto dSB = cutlass::make_cute_packed_stride(
        InternalStrideScaleB{}, cute::make_shape(N, scale_k, 1));

    return BaseArguments{
        ptr_A,
        dA,
        ptr_B,
        dB,
        ptr_SA,
        dSA,
        ptr_SB,
        dSB,
        ptr_CB,
        K / Base::MultGroupK,
        args.ptr_WideCache};
  }

  template <class ProblemShape>
  static bool can_implement(int64_t N, int64_t K, Arguments const&) {
    bool implementable =
        K % Base::BLK_K == 0 && N % Base::ScaleAlignElems == 0;
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: NVFP4 WoQ grouped alignment requirements not met.\n");
    }
    return implementable;
  }
};

}  // namespace cutlass::gemm::collective
