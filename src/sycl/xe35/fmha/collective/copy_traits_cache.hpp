#pragma once
#include <cute/atom/copy_traits_xe_2d.hpp>

namespace cute {
enum class XeStoreCachePolicy_xe35 {
  kDefault,
  kUC_UC_UC,
  kUC_UC_WB,
  kUC_WB_UC,
  kUC_WB_WB,
  kWT_UC_UC,
  kWT_UC_WB,
  kWT_WB_UC,
  kWT_WB_WB,
  kWB_UC_UC,
  kWB_UC_WB,
  kWB_WB_UC,
  kST_UC_UC,
  kST_UC_WB,
  kST_WB_UC
};

#define CUTE_XE_STORE_2D_ASM_XE35(CACHE_SUFFIX)                                      \
  asm (                                                                         \
    "lsc_store_block2d.ugm" CACHE_SUFFIX                                        \
    " (M1, 1) flat[%1+(0,0)] %0:d%2.%3x%4nn"                                    \
      :: "rw"(sv), "rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height)         \
  )

template <int Bits, int Height, int Width, XeStoreCachePolicy_xe35 CachePolicy = XeStoreCachePolicy_xe35::kDefault>
struct XE_STORE_2D_xe35 : XE_Copy_Op_2D_Base<Bits, Height, Width>
{
  static_assert(Height <= 8, "Height exceeds hardware limits");
  static constexpr XeStoreCachePolicy_xe35 StoreCachePolicy = CachePolicy;

  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int *payload, const T *src) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    using namespace intel;
    auto &sv = *reinterpret_cast<const storage_vector_t<T, Width * Height * Bits / sg_size>*>(src);
    if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kDefault) {
      CUTE_XE_STORE_2D_ASM_XE35("");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kUC_UC_UC) {
      CUTE_XE_STORE_2D_ASM_XE35(".uc.uc.uc");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kUC_UC_WB) {
      CUTE_XE_STORE_2D_ASM_XE35(".uc.uc.wb");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kUC_WB_UC) {
      CUTE_XE_STORE_2D_ASM_XE35(".uc.wb.uc");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kUC_WB_WB) {
      CUTE_XE_STORE_2D_ASM_XE35(".uc.wb.wb");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kWT_UC_UC) {
      CUTE_XE_STORE_2D_ASM_XE35(".wt.uc.uc");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kWT_UC_WB) {
      CUTE_XE_STORE_2D_ASM_XE35(".wt.uc.wb");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kWT_WB_UC) {
      CUTE_XE_STORE_2D_ASM_XE35(".wt.wb.uc");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kWT_WB_WB) {
      CUTE_XE_STORE_2D_ASM_XE35(".wt.wb.wb");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kWB_UC_UC) {
      CUTE_XE_STORE_2D_ASM_XE35(".wb.uc.uc");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kWB_UC_WB) {
      CUTE_XE_STORE_2D_ASM_XE35(".wb.uc.wb");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kWB_WB_UC) {
      CUTE_XE_STORE_2D_ASM_XE35(".wb.wb.uc");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kST_UC_UC) {
      CUTE_XE_STORE_2D_ASM_XE35(".st.uc.uc");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kST_UC_WB) {
      CUTE_XE_STORE_2D_ASM_XE35(".st.uc.wb");
    } else if constexpr (CachePolicy == XeStoreCachePolicy_xe35::kST_WB_UC) {
      CUTE_XE_STORE_2D_ASM_XE35(".st.wb.uc");
    } else {
      static_assert(dependent_false<integral_constant<XeStoreCachePolicy_xe35, CachePolicy>>,
                    "Unsupported Xe store cache policy");
    }
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }
};

#undef CUTE_XE_STORE_2D_ASM_XE35
template <class XMode, class YMode, typename ValType, typename TiledStrides,
      int CopyBits, int Height, int Width, XeStoreCachePolicy_xe35 CachePolicy>
struct Copy_Traits<XE_STORE_2D_xe35<CopyBits, Height, Width, CachePolicy>, XMode, YMode, ValType, TiledStrides>
  : Xe2DTraitsBase<XE_STORE_2D_xe35<CopyBits, Height, Width, CachePolicy>, XMode, YMode, ValType, TiledStrides>
{
  // (src-thr, src-val) -> (x, y)
  using SrcLayout = XeInterleavedLayout<Layout<Shape<Int<Width>, Int<Height>>>,
                                        CopyBits,
                                        sizeof_bits_v<ValType>>;

  using RefLayout = SrcLayout;
  using DstLayout = decltype(replace<0>(RefLayout{}, Layout<Shape<intel::_SGSize>, Stride<_0>>{}));

  using Op = XE_STORE_2D_xe35<CopyBits, Height, Width, CachePolicy>;
  using Super = Xe2DTraitsBase<Op, XMode, YMode, ValType, TiledStrides>;
  using Traits = typename Super::Traits;  // a.k.a. this class
  using ThrID = typename Super::ThrID;

  using Super::Super;

  // Execution.
  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_DEVICE friend constexpr void
  copy_unpack(Traits const&                   traits,
              Tensor<SEngine, SLayout> const& src,
              Tensor<DEngine, DLayout> &      dst) {
    using SType = typename SEngine::value_type;
    using DType = typename DEngine::value_type;
    using SrcLayout = typename Traits::SrcLayout;
    using DstLayout = typename Traits::DstLayout;
    constexpr auto SBits = sizeof_bits_v<SType>;

    static_assert(is_counting_layout_v<DLayout>, "Destination tensor must be a coordinate tensor.");
    static_assert(is_rmem_v<SEngine>, "Source tensor must be in registers.");
    static_assert(size(SLayout{}) * SBits == size<1>(SrcLayout{}),
                  "Source tensor size does not match copy atom size.");
    static_assert(size(DLayout{}) * SBits == size<1>(DstLayout{}),
                  "Destination tensor size does not match copy atom size.");

    traits.template update_payload<SBits>(dst.data().coord_);
    Op::copy(traits.payload, recast_ptr<int_byte_t<bits_to_bytes(Super::ValBits)>>(raw_pointer_cast(src.data())));
  }
};
template<int Bits, int Height, int Width, XeStoreCachePolicy_xe35 Policy>
struct is_xe_block_2d_atom<XE_STORE_2D_xe35<Bits,Height,Width,Policy>> : std::true_type {};
template <typename MemType, typename RegType, bool Store = false,
          XeStoreCachePolicy_xe35 StoreCachePolicy = XeStoreCachePolicy_xe35::kDefault,
          typename CoordLayout, typename GlobalStride>
CUTE_HOST_DEVICE
constexpr auto
block_2d_selector_xe35(CoordLayout const&, GlobalStride const&)
{
  static_assert(is_static_v<CoordLayout>, "Coordinate layout must be static");

  auto layout = coalesce(CoordLayout{});
  GlobalStride gstride{};

  // Determine size of copy.
  constexpr int MemBits = sizeof_bits_v<MemType>;
  constexpr int RegBits = sizeof_bits_v<RegType>;

  // Determine which kind of block 2D message to use (regular/VNNI/transpose)
  constexpr auto kind = block_2d_transform_selector<MemBits, RegBits, Store>(layout, gstride);

  // Strip off VNNI mode if present.
  constexpr auto slayout = get<0>(strip_vnni_subbyte<RegBits>(layout));

  constexpr auto x_mode = find_x_mode(gstride);
  constexpr auto y_mode = find_y_mode(gstride);

  constexpr int min_large_block = cute::min(256 / RegBits, 16);
  constexpr bool resize = (MemBits != RegBits);

  auto shape = atuple_coshape(layout);

  if constexpr (kind != Block2DTransform::T) {
    constexpr int CopyBits = cute::max(8, cute::min(64, MemBits));

    // Determine block width.
    // Get innermost stride in x dimension that is >= 1/2 GRF
    //   Block width = highest power of 2 divisor (up to 64b)
    //   Width = highest power of 2 divisor of full tile's width, up to 64b and 4x block width
    constexpr int max_w = 64 * 8 / MemBits;
    constexpr int x_stride = get_block_size<x_mode(), min_large_block>(slayout);
    constexpr int block_width = cute::gcd(max_w, x_stride);
    constexpr int load_width = cute::gcd(cute::min(max_w, 4 * block_width),
                                          get<x_mode()>(shape));
    constexpr int width = Store ? block_width : load_width;
    constexpr int block_cwidth = block_width * MemBits / CopyBits;
    constexpr int cwidth = width * MemBits / CopyBits;

    // Determine block height.
    // Get innermost stride in H dimension, besides VNNI stride if VNNI.
    // However, if data resizing will occur, choose full tile height, up to block height limit.
    //   (Rationale: we are already moving data, so layouts don't need to match)
    constexpr int y_stride = get_block_size<y_mode()>(slayout);
    constexpr int max_h = Store ? 8 : 32;
#if (SYCL_INTEL_TARGET == 35)
    constexpr int height = Store ? cute::gcd(resize ? get<y_mode()>(shape) : y_stride, max_h) :
                                   cute::gcd(get<y_mode()>(shape), max_h);
#else
    constexpr int height = cute::gcd(resize ? get<y_mode()>(shape) : y_stride, max_h);
#endif

    if constexpr (Store)
      return XE_STORE_2D_xe35    <CopyBits, height, cwidth, StoreCachePolicy>{};
    else if constexpr (kind == Block2DTransform::V)
      return XE_LOAD_2D_VNNI<CopyBits, height, cwidth, block_cwidth>{};
    else
      return XE_LOAD_2D     <CopyBits, height, cwidth, block_cwidth>{};
  } else {
    // Similar process for transposing copies, but with width/height reversed.
    constexpr int CopyBits = cute::max(32, cute::min(64, MemBits));

    constexpr int y_stride = get_block_size<y_mode(), min_large_block>(slayout);
    constexpr int height = cute::gcd(32, y_stride);

    // Transpose Width limits (in CopyBits-element units):
    //   d32: <=8 on Xe2; <=16 on Xe3P+.
    //   d64: <=4 on Xe2; <=8 on Xe3P+.
#if (SYCL_INTEL_TARGET == 35)
    constexpr int max_cwidth_d32 = 16;
    constexpr int max_cwidth_d64 = 8;
#else
    constexpr int max_cwidth_d32 = 8;
    constexpr int max_cwidth_d64 = 4;
#endif
    constexpr int max_cwidth = (CopyBits == 64) ? max_cwidth_d64 : max_cwidth_d32;
    constexpr int max_w = max_cwidth * CopyBits / MemBits;
    constexpr int width = cute::gcd(get<x_mode()>(shape), max_w);
    constexpr int cwidth = width * MemBits / CopyBits;

    return XE_LOAD_2D_TRANSPOSE<CopyBits, height, cwidth>{};
  }
}
template <class ValType, class TiledMMA, class... Strides>
CUTE_HOST_DEVICE auto make_block_2d_copy_A_xe35(TiledMMA const& mma, Stride<Strides...> const& gstride) {
  using MMAType = typename TiledMMA::ValTypeA;
  auto coordinates = make_identity_tensor(select<0,2>(mma.tile_mnk()));
  auto operation = block_2d_selector_xe35<ValType, MMAType>(mma.get_slice(0).atom_partition_A(coordinates).layout(), gstride);
  return make_block_2d_copy_A<ValType>(operation, mma, gstride);
}
template <class TiledMMA, class GEngine, class GLayout>
CUTE_HOST_DEVICE auto make_block_2d_copy_A_xe35(TiledMMA const& mma, Tensor<GEngine, GLayout> const& gmem) {
  using ValType = typename GEngine::value_type;
  return make_block_2d_copy_A_xe35<ValType>(mma, gmem.stride()).with(gmem);
}
template <class ValType, class TiledMMA, class... Strides>
CUTE_HOST_DEVICE auto make_block_2d_copy_B_xe35(TiledMMA const& mma, Stride<Strides...> const& gstride) {
  using MMAType = typename TiledMMA::ValTypeB;
  auto coordinates = make_identity_tensor(select<1,2>(mma.tile_mnk()));
  auto operation = block_2d_selector_xe35<ValType, MMAType>(mma.get_slice(0).atom_partition_B(coordinates).layout(), gstride);
  return make_block_2d_copy_B<ValType>(operation, mma, gstride);
}
template <class TiledMMA, class GEngine, class GLayout>
CUTE_HOST_DEVICE auto make_block_2d_copy_B_xe35(TiledMMA const& mma, Tensor<GEngine, GLayout> const& gmem) {
  using ValType = typename GEngine::value_type;
  return make_block_2d_copy_B_xe35<ValType>(mma, gmem.stride()).with(gmem);
}
template <XeStoreCachePolicy_xe35 Policy, class ValType, class TiledMMA, class... Strides>
CUTE_HOST_DEVICE auto make_block_2d_copy_D_xe35(TiledMMA const& mma, Stride<Strides...> const& gstride) {
  using MMAType = typename TiledMMA::ValTypeD;
  auto coordinates = make_identity_tensor(select<0,1>(mma.tile_mnk()));
  auto operation = block_2d_selector_xe35<ValType, MMAType, true, Policy>(mma.get_slice(0).atom_partition_C(coordinates).layout(), gstride);
  return make_block_2d_copy_CD<ValType>(operation, mma, gstride);
}
template <XeStoreCachePolicy_xe35 Policy, class TiledMMA, class GEngine, class GLayout>
CUTE_HOST_DEVICE auto make_block_2d_copy_D_xe35(TiledMMA const& mma, Tensor<GEngine, GLayout> const& gmem) {
  using ValType = typename GEngine::value_type;
  return make_block_2d_copy_D_xe35<Policy, ValType>(mma, gmem.stride()).with(gmem);
}
template <XeStoreCachePolicy_xe35 Policy, class TiledMMA, class SubtileTVCoordLayout, class SubtileSGLayout, class GEngine, class GLayout>
CUTE_HOST_DEVICE auto make_block_2d_copy_D_subtiled_xe35(TiledMMA const& mma, SubtileTVCoordLayout const& tv_layout, SubtileSGLayout const& sg_layout, Tensor<GEngine, GLayout> const& gmem) {
  using ValType = typename GEngine::value_type;
  using MMAType = typename TiledMMA::ValTypeD;
  auto operation = block_2d_selector_xe35<ValType, MMAType, true, Policy>(tv_layout, gmem.stride());
  return make_block_2d_copy_CD_subtiled<ValType>(operation, mma, atuple_coshape(tv_layout), sg_layout, gmem.stride()).with(gmem);
}
} 
