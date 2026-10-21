/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief Sub-byte -> wider-type subgroup fragment conversion for Xe, with
           optional relayout and a fused per-element scale.

    Why this exists: `cute::reorder` cannot upconvert a sub-byte fragment. For an
    E2M1 (or INT4) source whose relayout isn't classified UU / UV / VU /
    VV-with-atom, `ReorderDispatchConvertRelayout` re-enters the same
    instantiation — it picks `NewSrcType = upcast_subbyte_t<E2M1> = half_t` and
    recurses on the identical `(E2M1 -> half, same layout)` call — and the SYCL
    front end rejects it with

        reorder.hpp:385: error: SYCL kernel cannot call a recursive function

    The asymmetry is deliberate upstream: sycl-tla ships optimized `Xe_Reorder`
    atoms only for the *quantize* direction (bf16/half -> E2M1/INT4, see
    cute/arch/reorder_xe35.hpp); dequantize falls back to the generic UU path,
    which only covers relayouts that are already unit-stride.

    Adding an `Xe_Reorder<Generic, ...>` specialization would not fix it: the
    atom-based `reorder_impl` assumes a register-to-register mapping with no
    permutation, so it is only sound for UU-class relayouts. The fix instead
    splits the job along the line where each half *is* well supported:

      1. the type conversion is index-preserving (a 4-bit copy fragment already
         describes work-item ownership at element granularity), so do it here;
      2. what remains is a same-type relayout, which `cute::reorder` handles.

    When the two fragments already share a TV layout, step 2 drops out.  The
    NVFP4 MMA-B pairing cannot share a layout: each loaded byte belongs to one
    lane, while its two nibbles belong to different MMA lanes.  Its fallback is
    therefore a fused subgroup gather that converts and scales directly into
    MMA-B, without allocating a wider staging fragment.

    Conversion goes through FP32.  The relayout path decodes E2M1 directly to
    binary16 bits and uses Xe's packed half-to-E4M3 builtin after the multiply,
    avoiding the large scalar conversion expansion produced by the generic
    CUTLASS types.
*/

#pragma once

// clang-format off
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>
#include <sycl/ext/oneapi/experimental/float_4bit/types.hpp>

#include <cute/tensor.hpp>
#include <cute/algorithm/reorder.hpp>

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

namespace sgl_xpu_xe35 {

using namespace cute;

/// Identity scale: leaves the converted value untouched.
struct NoScale_xe35 {
  CUTLASS_DEVICE float operator()(int) const { return 1.0f; }
};

namespace detail {

/// Convert `Count` sub-byte elements starting at `base` into `dst`, scaling each
/// by `scale(base + j)`. Src storage is assumed contiguous in the fragment's
/// flat index, which holds for the compact register fragments cute builds.
template <int Chunk, class Src, class Dst, class Scale>
CUTLASS_DEVICE void
convert_chunked(Src const& src, Dst& dst, Scale const& scale) {
  using SrcType = typename Src::value_type;
  using DstType = typename Dst::value_type;
  constexpr int kSize = cute::size_v<typename Src::layout_type>;
  static_assert(kSize == cute::size_v<typename Dst::layout_type>,
                "Source and destination fragments must hold the same element count");
  static_assert(kSize % Chunk == 0, "Fragment size must be a whole number of chunks");

  using SrcArray = cutlass::Array<SrcType, Chunk>;
  using MidArray = cutlass::Array<float, Chunk>;
  using DstArray = cutlass::Array<DstType, Chunk>;
  cutlass::NumericArrayConverter<float, SrcType, Chunk> to_float;
  cutlass::NumericArrayConverter<DstType, float, Chunk> from_float;

  auto const* src_raw = reinterpret_cast<SrcArray const*>(raw_pointer_cast(src.data()));
  auto* dst_raw = reinterpret_cast<DstArray*>(raw_pointer_cast(dst.data()));

  CUTLASS_PRAGMA_UNROLL
  for (int c = 0; c < kSize / Chunk; ++c) {
    MidArray mid = to_float(src_raw[c]);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Chunk; ++j) {
      mid[j] *= scale(c * Chunk + j);
    }
    dst_raw[c] = from_float(mid);
  }
}

/// Compress eight selected nibbles from byte lanes into one packed E2M1 word.
CUTLASS_DEVICE uint32_t
pack_selected_nibbles(uint64_t bytes, int nibble_shift) {
  uint64_t x = (bytes >> nibble_shift) & 0x0f0f0f0f0f0f0f0full;
  x = (x | (x >> 4)) & 0x00ff00ff00ff00ffull;
  x = (x | (x >> 8)) & 0x0000ffff0000ffffull;
  return static_cast<uint32_t>(x | (x >> 16));
}

#ifdef __SYCL_DEVICE_ONLY__
CUTLASS_DEVICE sycl::detail::float16_vec2
mul_fp16x2_visa(
    sycl::detail::float16_vec2 values,
    sycl::detail::float16_vec2 scales) {
  uint32_t value_bits = sycl::bit_cast<uint32_t>(values);
  uint32_t scale_bits = sycl::bit_cast<uint32_t>(scales);
  asm volatile(
      "{\n"
      ".decl VALUE_HF v_type=G type=HF num_elts=32 alias=<%0,0>\n"
      ".decl SCALE_HF v_type=G type=HF num_elts=32 alias=<%1,0>\n"
      "mul (M1_NM, 32) VALUE_HF(0,0)<1> "
      "VALUE_HF(0,0)<1;1,0> SCALE_HF(0,0)<1;1,0>\n"
      "}\n"
      : "+rw"(value_bits)
      : "rw"(scale_bits));
  return sycl::bit_cast<sycl::detail::float16_vec2>(value_bits);
}
#endif

template <class Dst, class Scale>
CUTLASS_DEVICE void
convert_native_e2m1x8(
    uint32_t packed,
    Dst& dst,
    Scale const& scale,
    int r,
    int s,
    int parity,
    int c_base) {
  using DstType = typename Dst::value_type;

#ifdef __SYCL_DEVICE_ONLY__
  // E2M1 and the prepared BF16 multiplier have at most 2 and 8 significant
  // bits, respectively, and prep bounds the multiplier by 448 / 6. Both the
  // multiplier and product therefore fit exactly in binary16. Values below
  // binary16's range are also below E4M3's zero-rounding threshold after the
  // largest E2M1 multiplier, so converting the scale before multiplication
  // preserves the E4M3 result while avoiding FP32 promotion.
  CUTLASS_PRAGMA_UNROLL
  for (int pair = 0; pair < 4; ++pair) {
    sycl::detail::fp4_uint8_vec1 packed_pair{
        static_cast<uint8_t>(packed >> (8 * pair))};
    auto decoded =
        __builtin_spirv_ConvertE2M1ToFP16INTEL(packed_pair);
    sycl::detail::float16_vec2 scales;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int local = 2 * pair + j;
      const int c = c_base + (local >> 2);
      const int p = local & 3;
      const int q = parity + 2 * c;
      const int dst_idx = p + 4 * q + 32 * r + 128 * s;
      scales[j] = static_cast<_Float16>(scale(dst_idx));
    }
    auto scaled = detail::mul_fp16x2_visa(decoded, scales);

    auto converted = __builtin_spirv_ClampConvertFP16ToE4M3INTEL(scaled);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int local = 2 * pair + j;
      const int c = c_base + (local >> 2);
      const int p = local & 3;
      const int q = parity + 2 * c;
      const int dst_idx = p + 4 * q + 32 * r + 128 * s;
      dst(dst_idx) = DstType::bitcast(converted[j]);
    }
  }
#else
  using SrcType = cutlass::float_e2m1_t;
  CUTLASS_PRAGMA_UNROLL
  for (int local = 0; local < 8; ++local) {
    const int c = c_base + (local >> 2);
    const int p = local & 3;
    const int q = parity + 2 * c;
    const int dst_idx = p + 4 * q + 32 * r + 128 * s;
    SrcType value = SrcType::bitcast((packed >> (4 * local)) & 0xf);
    dst(dst_idx) =
        static_cast<DstType>(static_cast<float>(value) * scale(dst_idx));
  }
#endif
}

/// Decode eight selected E2M1 values into a widened BF16 cache fragment.
///
/// Unlike convert_native_e2m1x8, this deliberately stops before the E4M3
/// conversion.  The widened fragment can be written to L1-backed scratch or
/// SLM once per workgroup B tile and requantized by every M subgroup.
template <class Dst, class Scale>
CUTLASS_DEVICE void
convert_native_e2m1x8_wide(
    uint32_t packed,
    Dst& dst,
    Scale const& scale,
    int r,
    int s,
    int parity,
    int c_base) {
  using DstType = typename Dst::value_type;

#ifdef __SYCL_DEVICE_ONLY__
  CUTLASS_PRAGMA_UNROLL
  for (int pair = 0; pair < 4; ++pair) {
    sycl::detail::fp4_uint8_vec1 packed_pair{
        static_cast<uint8_t>(packed >> (8 * pair))};
    auto decoded =
        __builtin_spirv_ConvertE2M1ToFP16INTEL(packed_pair);
    sycl::detail::float16_vec2 scales;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int local = 2 * pair + j;
      const int c = c_base + (local >> 2);
      const int p = local & 3;
      const int q = parity + 2 * c;
      const int dst_idx = p + 4 * q + 32 * r + 128 * s;
      scales[j] = static_cast<_Float16>(scale(dst_idx));
    }
    auto scaled = detail::mul_fp16x2_visa(decoded, scales);

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 2; ++j) {
      const int local = 2 * pair + j;
      const int c = c_base + (local >> 2);
      const int p = local & 3;
      const int q = parity + 2 * c;
      const int dst_idx = p + 4 * q + 32 * r + 128 * s;
      dst(dst_idx) = DstType(static_cast<float>(scaled[j]));
    }
  }
#else
  using SrcType = cutlass::float_e2m1_t;
  CUTLASS_PRAGMA_UNROLL
  for (int local = 0; local < 8; ++local) {
    const int c = c_base + (local >> 2);
    const int p = local & 3;
    const int q = parity + 2 * c;
    const int dst_idx = p + 4 * q + 32 * r + 128 * s;
    SrcType value = SrcType::bitcast((packed >> (4 * local)) & 0xf);
    dst(dst_idx) =
        DstType(static_cast<float>(value) * scale(dst_idx));
  }
#endif
}

/// Fused relayout, upconversion, and scaling for the Xe3P NVFP4 MMA-B tile.
///
/// The native D32 transpose load owns source coordinates as
///
///   n = lane/4 + 4*b + 16*d,  k = 2*(lane%4) + a + 8*c,
///
/// while E4M3 MMA-B owns them as
///
///   n = lane/4 + 4*p + 16*r,  k = lane%4 + 4*q + 32*s.
///
/// For fixed (q,r,s), all lanes need the same four-byte source span. Selecting
/// a 16-byte span handles four q values of one parity at once, reducing the
/// whole 64x64 subgroup tile to 16 cross-lane selects. Each receiver then takes
/// the low or high nibble selected by its MMA K lane and writes MMA-B directly.
template <class Src, class Dst, class Scale>
CUTLASS_DEVICE void
convert_relayout_mma_b(Src const& src, Dst& dst, Scale const& scale) {
  using SrcTv = remove_cvref_t<decltype(src.tv_layout())>;
  using DstTv = remove_cvref_t<decltype(dst.tv_layout())>;

  using ExpectedSrcTv = decltype(make_layout(
      make_shape(make_shape(_4{}, _4{}),
                 make_shape(make_shape(_2{}, _4{}, _8{}), _4{}),
                 make_shape(_1{}, _1{})),
      make_stride(make_stride(_2{} * E<1>{}, E<0>{}),
                  make_stride(make_stride(E<1>{}, _4{} * E<0>{}, _8{} * E<1>{}),
                              _16{} * E<0>{}),
                  make_stride(_0{}, _0{}))));
  using ExpectedDstTv = decltype(make_layout(
      make_shape(make_shape(_4{}, _4{}),
                 make_shape(make_shape(_4{}, _8{}), make_shape(_4{}, _2{}))),
      make_stride(make_stride(E<1>{}, E<0>{}),
                  make_stride(make_stride(_4{} * E<0>{}, _4{} * E<1>{}),
                              make_stride(_16{} * E<0>{}, _32{} * E<1>{})))));

  static_assert(cute::is_same_v<decltype(project_strides(SrcTv{})),
                                decltype(project_strides(ExpectedSrcTv{}))>,
                "NVFP4 copy fragment layout changed; update the fused MMA-B mapping");
  static_assert(cute::is_same_v<decltype(project_strides(DstTv{})),
                                decltype(project_strides(ExpectedDstTv{}))>,
                "E4M3 MMA-B fragment layout changed; update the fused mapping");
  static_assert(cute::size_v<typename Src::layout_type> == 256 &&
                cute::size_v<typename Dst::layout_type> == 256,
                "Fused NVFP4 mapping requires 256 values per lane");

  auto const* packed = reinterpret_cast<intel::uchar16 const*>(raw_pointer_cast(src.data()));
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  const int lane = static_cast<int>(sg.get_local_linear_id());
  const int lane_n = lane >> 2;
  const int lane_k = lane & 3;
  const int nibble_shift = (lane_k & 1) * 4;

  CUTLASS_PRAGMA_UNROLL
  for (int r = 0; r < 4; ++r) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < 2; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int parity = 0; parity < 2; ++parity) {
        const int src_lane = 4 * lane_n + 2 * parity + (lane_k >> 1);
        intel::uchar16 remote = sycl::select_from_group(sg, packed[2 * r + s], src_lane);
        intel::ulong2 words = sycl::bit_cast<intel::ulong2>(remote);
        detail::convert_native_e2m1x8(
            detail::pack_selected_nibbles(words[0], nibble_shift),
            dst, scale, r, s, parity, 0);
        detail::convert_native_e2m1x8(
            detail::pack_selected_nibbles(words[1], nibble_shift),
            dst, scale, r, s, parity, 2);
      }
    }
  }
}

/// Convert one 32-wide MMA-K slice of the Xe3P NVFP4 MMA-B tile.
///
/// Keeping conversion adjacent to the consuming BDPAS shortens the live range
/// of both the converted E4M3 values and their per-16 multipliers. `dst` keeps
/// the full fragment layout so the mapping remains identical to
/// convert_relayout_mma_b; only the selected 128-value K slice is written.
template <class Src, class Dst, class Scale>
CUTLASS_DEVICE void
convert_relayout_mma_b_k(
    Src const& src, Dst& dst, Scale const& scale, int k_step) {
  using SrcTv = remove_cvref_t<decltype(src.tv_layout())>;
  using DstTv = remove_cvref_t<decltype(dst.tv_layout())>;

  using ExpectedSrcTv = decltype(make_layout(
      make_shape(make_shape(_4{}, _4{}),
                 make_shape(make_shape(_2{}, _4{}, _8{}), _4{}),
                 make_shape(_1{}, _1{})),
      make_stride(make_stride(_2{} * E<1>{}, E<0>{}),
                  make_stride(make_stride(E<1>{}, _4{} * E<0>{}, _8{} * E<1>{}),
                              _16{} * E<0>{}),
                  make_stride(_0{}, _0{}))));
  using ExpectedDstTv = decltype(make_layout(
      make_shape(make_shape(_4{}, _4{}),
                 make_shape(make_shape(_4{}, _8{}), make_shape(_4{}, _2{}))),
      make_stride(make_stride(E<1>{}, E<0>{}),
                  make_stride(make_stride(_4{} * E<0>{}, _4{} * E<1>{}),
                              make_stride(_16{} * E<0>{}, _32{} * E<1>{})))));

  static_assert(cute::is_same_v<decltype(project_strides(SrcTv{})),
                                decltype(project_strides(ExpectedSrcTv{}))>);
  static_assert(cute::is_same_v<decltype(project_strides(DstTv{})),
                                decltype(project_strides(ExpectedDstTv{}))>);
  static_assert(cute::size_v<typename Src::layout_type> == 256 &&
                cute::size_v<typename Dst::layout_type> == 256);

  auto const* packed =
      reinterpret_cast<intel::uchar16 const*>(raw_pointer_cast(src.data()));
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  const int lane = static_cast<int>(sg.get_local_linear_id());
  const int lane_n = lane >> 2;
  const int lane_k = lane & 3;
  const int nibble_shift = (lane_k & 1) * 4;

  CUTLASS_PRAGMA_UNROLL
  for (int r = 0; r < 4; ++r) {
    CUTLASS_PRAGMA_UNROLL
    for (int parity = 0; parity < 2; ++parity) {
      const int src_lane = 4 * lane_n + 2 * parity + (lane_k >> 1);
      intel::uchar16 remote =
          sycl::select_from_group(sg, packed[2 * r + k_step], src_lane);
      intel::ulong2 words = sycl::bit_cast<intel::ulong2>(remote);
      detail::convert_native_e2m1x8(
          detail::pack_selected_nibbles(words[0], nibble_shift),
          dst, scale, r, k_step, parity, 0);
      detail::convert_native_e2m1x8(
          detail::pack_selected_nibbles(words[1], nibble_shift),
          dst, scale, r, k_step, parity, 2);
    }
  }
}

/// Fused E2M1 relayout/dequantization into a BF16 MMA-coordinate fragment.
template <class Src, class Dst, class Scale>
CUTLASS_DEVICE void
convert_relayout_mma_b_wide(
    Src const& src, Dst& dst, Scale const& scale) {
  static_assert(cute::size_v<typename Src::layout_type> ==
                cute::size_v<typename Dst::layout_type>);
  constexpr int kValues = cute::size_v<typename Dst::layout_type>;
  static_assert(kValues % 128 == 0,
                "NVFP4 widened MMA-B fragment must contain complete K32 slices");
  constexpr int kKSteps = kValues / 128;

  auto const* packed =
      reinterpret_cast<intel::uchar16 const*>(raw_pointer_cast(src.data()));
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  const int lane = static_cast<int>(sg.get_local_linear_id());
  const int lane_n = lane >> 2;
  const int lane_k = lane & 3;
  const int nibble_shift = (lane_k & 1) * 4;

  CUTLASS_PRAGMA_UNROLL
  for (int r = 0; r < 4; ++r) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kKSteps; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int parity = 0; parity < 2; ++parity) {
        const int src_lane = 4 * lane_n + 2 * parity + (lane_k >> 1);
        intel::uchar16 remote =
            sycl::select_from_group(sg, packed[kKSteps * r + s], src_lane);
        intel::ulong2 words = sycl::bit_cast<intel::ulong2>(remote);
        detail::convert_native_e2m1x8_wide(
            detail::pack_selected_nibbles(words[0], nibble_shift),
            dst, scale, r, s, parity, 0);
        detail::convert_native_e2m1x8_wide(
            detail::pack_selected_nibbles(words[1], nibble_shift),
            dst, scale, r, s, parity, 2);
      }
    }
  }
}

/// Decode one K32 slice into a compact 128-value/lane BF16 fragment.
template <class Src, class Dst, class Scale>
CUTLASS_DEVICE void
convert_relayout_mma_b_k_wide(
    Src const& src,
    Dst& dst,
    Scale const& scale,
    int k_step) {
  constexpr int kSrcValues = cute::size_v<typename Src::layout_type>;
  constexpr int kDstValues = cute::size_v<typename Dst::layout_type>;
  static_assert(kSrcValues % 128 == 0);
  static_assert(kDstValues == 128);
  constexpr int kKSteps = kSrcValues / 128;

  auto const* packed =
      reinterpret_cast<intel::uchar16 const*>(raw_pointer_cast(src.data()));
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  const int lane = static_cast<int>(sg.get_local_linear_id());
  const int lane_n = lane >> 2;
  const int lane_k = lane & 3;
  const int nibble_shift = (lane_k & 1) * 4;

  CUTLASS_PRAGMA_UNROLL
  for (int r = 0; r < 4; ++r) {
    CUTLASS_PRAGMA_UNROLL
    for (int parity = 0; parity < 2; ++parity) {
      const int src_lane = 4 * lane_n + 2 * parity + (lane_k >> 1);
      intel::uchar16 remote = sycl::select_from_group(
          sg, packed[kKSteps * r + k_step], src_lane);
      intel::ulong2 words = sycl::bit_cast<intel::ulong2>(remote);
      detail::convert_native_e2m1x8_wide(
          detail::pack_selected_nibbles(words[0], nibble_shift),
          dst, scale, r, 0, parity, 0);
      detail::convert_native_e2m1x8_wide(
          detail::pack_selected_nibbles(words[1], nibble_shift),
          dst, scale, r, 0, parity, 2);
    }
  }
}

}  // namespace detail

/// True when two subgroup fragments already agree on (thread, value) -> coord,
/// so a conversion by flat index needs no relayout.
template <class SrcTv, class DstTv>
static constexpr bool tv_layouts_match_v =
    cute::is_same_v<decltype(cute::project_strides(SrcTv{})),
                    decltype(cute::project_strides(DstTv{}))>;

/// Upconvert a sub-byte subgroup fragment into a wider-type one, relaying out if
/// their TV layouts differ, and scaling each value (in FP32) by `scale(i)` where
/// `i` is the *destination* fragment's flat index.
///
/// See the file header for why cute::reorder cannot do this directly.
template <int Chunk = 8, class Src, class Dst, class Scale = NoScale_xe35>
CUTLASS_DEVICE void
reorder_upconvert_xe35(Src const& src, Dst& dst, Scale const& scale = {}) {
  using SrcType = typename Src::value_type;
  using DstType = typename Dst::value_type;
  static_assert(sizeof_bits_v<SrcType> < 8, "Source must be a sub-byte type");
  static_assert(sizeof_bits_v<DstType> >= 8, "Destination must be at least a byte wide");

  if constexpr (tv_layouts_match_v<decltype(src.tv_layout()), decltype(dst.tv_layout())>) {
    // No relayout: flat indices already agree, so convert and scale in one pass
    // and keep the staging fragment out of the register budget.
    detail::convert_chunked<Chunk>(src, dst, scale);
  } else {
    detail::convert_relayout_mma_b(src, dst, scale);
  }
}

/// Fused relayout/upconvert for one MMA-K step of the NVFP4 Xe MMA-B layout.
template <class Src, class Dst, class Scale = NoScale_xe35>
CUTLASS_DEVICE void
reorder_upconvert_mma_b_k_xe35(
    Src const& src, Dst& dst, int k_step, Scale const& scale = {}) {
  using SrcType = typename Src::value_type;
  using DstType = typename Dst::value_type;
  static_assert(sizeof_bits_v<SrcType> < 8, "Source must be a sub-byte type");
  static_assert(sizeof_bits_v<DstType> >= 8, "Destination must be at least a byte wide");
  detail::convert_relayout_mma_b_k(src, dst, scale, k_step);
}

/// Fused relayout/dequantization into a widened BF16 cache fragment.
template <class Src, class Dst, class Scale = NoScale_xe35>
CUTLASS_DEVICE void
reorder_upconvert_mma_b_wide(
    Src const& src, Dst& dst, Scale const& scale = {}) {
  using SrcType = typename Src::value_type;
  using DstType = typename Dst::value_type;
  static_assert(sizeof_bits_v<SrcType> < 8, "Source must be a sub-byte type");
  static_assert(cute::is_same_v<DstType, cutlass::bfloat16_t>,
                "Widened NVFP4 cache must store BF16");
  detail::convert_relayout_mma_b_wide(src, dst, scale);
}

/// Fused relayout/dequantization of one K32 slice into BF16.
template <class Src, class Dst, class Scale = NoScale_xe35>
CUTLASS_DEVICE void
reorder_upconvert_mma_b_k_wide_xe35(
    Src const& src,
    Dst& dst,
    int k_step,
    Scale const& scale = {}) {
  using SrcType = typename Src::value_type;
  using DstType = typename Dst::value_type;
  static_assert(sizeof_bits_v<SrcType> < 8, "Source must be a sub-byte type");
  static_assert(cute::is_same_v<DstType, cutlass::bfloat16_t>,
                "Widened NVFP4 cache must store BF16");
  detail::convert_relayout_mma_b_k_wide(src, dst, scale, k_step);
}

}  // namespace sgl_xpu_xe35
