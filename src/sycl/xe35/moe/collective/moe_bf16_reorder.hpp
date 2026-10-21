#pragma once

#include "cute/algorithm/reorder.hpp"
#include "cute/arch/reorder_xe.hpp"

#if defined(SYCL_INTEL_TARGET) && SYCL_INTEL_TARGET == 35

namespace sgl_kernel::vllm_grouped_gemm_xe35 {

struct MoeBf16ReorderAtom_xe35 {
  using Stage = cute::intel::uint2;

  CUTE_HOST_DEVICE static void reorder_stage_xe35(
      float const& source_first, float const& source_second, Stage& destination) {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm(
        "{\n"
        ".decl TMP_UD v_type=G type=UD num_elts=32 alias=<%0,0>\n"
        ".decl IN_UD0 v_type=G type=UD num_elts=16 alias=<%1,0>\n"
        ".decl IN_UD1 v_type=G type=UD num_elts=16 alias=<%2,0>\n"
        "add (M1_NM, 16) TMP_UD(0,0)<1> IN_UD0(0,0)<1;1,0> 0x8000:uw\n"
        "add (M1_NM, 16) TMP_UD(1,0)<1> IN_UD1(0,0)<1;1,0> 0x8000:uw\n"
        "}\n"
        : "=rw"(destination)
        : "rw"(source_first), "rw"(source_second));
#else
    CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }

  CUTE_HOST_DEVICE static void reorder_pack_xe35(Stage const& source, cute::intel::ushort2& destination) {
#if defined(CUTE_ARCH_REORDER_XE_ENABLED)
    asm(
        "{\n"
        ".decl OUT_UW v_type=G type=UW num_elts=32 alias=<%0,0>\n"
        ".decl TMP_UD v_type=G type=UD num_elts=32 alias=<%1,0>\n"
        ".decl TMP_UW v_type=G type=UW num_elts=64 alias=<TMP_UD,0>\n"
        "mov (M1_NM, 32) OUT_UW(0,0)<1> TMP_UW(0,1)<2;1,0>\n"
        "}\n"
        : "=rw"(destination)
        : "rw"(source));
#else
    CUTE_INVALID_CONTROL_PATH("Not Xe");
#endif
  }
};

template <int DestinationValue, int SourceValue, int Values, class ValueLayout>
struct MoeReorderFindSource_xe35 {
  static constexpr int mapped = decltype(ValueLayout{}(cute::Int<SourceValue>{}))::value;
  static constexpr int value = mapped == DestinationValue
      ? SourceValue
      : MoeReorderFindSource_xe35<DestinationValue, SourceValue + 1, Values, ValueLayout>::value;
};

template <int DestinationValue, int Values, class ValueLayout>
struct MoeReorderFindSource_xe35<DestinationValue, Values, Values, ValueLayout> {
  static constexpr int value = -1;
};

template <int DestinationValue, int Offset, int ChunkValues, int Values, class ValueLayout>
struct MoeReorderChunkSupported_xe35 {
  static constexpr bool value =
      MoeReorderFindSource_xe35<DestinationValue + Offset, 0, Values, ValueLayout>::value >= 0 &&
      MoeReorderChunkSupported_xe35<DestinationValue, Offset + 1, ChunkValues, Values, ValueLayout>::value;
};

template <int DestinationValue, int ChunkValues, int Values, class ValueLayout>
struct MoeReorderChunkSupported_xe35<DestinationValue, ChunkValues, ChunkValues, Values, ValueLayout> {
  static constexpr bool value = true;
};

template <int DestinationValue, int ChunkValues, int Values, class ValueLayout>
struct MoeReorderLayoutSupported_xe35 {
  static constexpr bool value = DestinationValue + ChunkValues <= Values &&
      MoeReorderChunkSupported_xe35<DestinationValue, 0, ChunkValues, Values, ValueLayout>::value &&
      MoeReorderLayoutSupported_xe35<DestinationValue + ChunkValues, ChunkValues, Values, ValueLayout>::value;
};

template <int ChunkValues, int Values, class ValueLayout>
struct MoeReorderLayoutSupported_xe35<Values, ChunkValues, Values, ValueLayout> {
  static constexpr bool value = true;
};

template <int DestinationValue, int Values, class ValueLayout, class SourceTensor, int Chunks>
CUTE_HOST_DEVICE void moe_reorder_stage_xe35(
    SourceTensor const& source, MoeBf16ReorderAtom_xe35::Stage (&staged)[Chunks]) {
  if constexpr (DestinationValue < Values) {
    MoeBf16ReorderAtom_xe35::reorder_stage_xe35(
        source(MoeReorderFindSource_xe35<DestinationValue, 0, Values, ValueLayout>::value),
        source(MoeReorderFindSource_xe35<DestinationValue + 1, 0, Values, ValueLayout>::value),
        staged[DestinationValue / 2]);
    moe_reorder_stage_xe35<DestinationValue + 2, Values, ValueLayout>(source, staged);
  }
}

template <int DestinationValue, int Values, class DestinationTensor, int Chunks>
CUTE_HOST_DEVICE void moe_reorder_store_xe35(
    MoeBf16ReorderAtom_xe35::Stage const (&staged)[Chunks], DestinationTensor& destination) {
  if constexpr (DestinationValue < Values) {
    MoeBf16ReorderAtom_xe35::reorder_pack_xe35(
        staged[DestinationValue / 2], reinterpret_cast<cute::intel::ushort2&>(destination(DestinationValue)));
    moe_reorder_store_xe35<DestinationValue + 2, Values>(staged, destination);
  }
}

template <class SourceEngine, class SourceLayoutWI, class SourceLayout,
          class DestinationEngine, class DestinationLayoutWI, class DestinationLayout>
CUTE_HOST_DEVICE void reorder_moe_output_xe35(
    cute::SubgroupTensor<SourceEngine, SourceLayoutWI, SourceLayout> const& source,
    cute::SubgroupTensor<DestinationEngine, DestinationLayoutWI, DestinationLayout>& destination) {
  using SourceType = typename SourceEngine::element_type;
  using DestinationType = typename DestinationEngine::element_type;
  auto source_layout = cute::detail::subbyte_sg_tv_swizzle<SourceType>(cute::project_strides(source.tv_layout()));
  auto destination_layout = cute::detail::subbyte_sg_tv_swizzle<DestinationType>(cute::project_strides(destination.tv_layout()));
  auto implementation = cute::choose_xe_reorder_impl<SourceType, DestinationType>(source_layout, destination_layout);
  constexpr bool has_broadcast = cute::size(DestinationLayoutWI{}) > cute::size(SourceLayoutWI{});
  if constexpr (cute::is_same_v<decltype(implementation),
                    cute::Xe_Reorder<cute::ReorderKind::UU, float, cutlass::bfloat16_t>> && !has_broadcast) {
    using SubgroupSize = cute::intel::_SGSize;
    constexpr int values = cute::size(decltype(source_layout){}) / cute::size<0>(decltype(source_layout){});
    static_assert(SubgroupSize::value == 16);
    auto mapping = cute::coalesce(cute::composition(cute::right_inverse(destination_layout), source_layout));
    auto value_layout = cute::composition(
        cute::composition(cute::Layout<cute::Shape<SubgroupSize, cute::Int<values>>, cute::Stride<cute::_0, cute::_1>>{}, mapping),
        cute::Layout<cute::Shape<cute::_1, cute::Int<values>>, cute::Stride<cute::_0, SubgroupSize>>{});
    if constexpr (values % 2 == 0) {
      if constexpr (MoeReorderLayoutSupported_xe35<0, 2, values, decltype(value_layout)>::value) {
        MoeBf16ReorderAtom_xe35::Stage staged[values / 2];
        moe_reorder_stage_xe35<0, values, decltype(value_layout)>(source, staged);
        moe_reorder_store_xe35<0, values>(staged, destination);
        return;
      }
    }
  }
  cute::reorder(source, destination);
}

}

#endif
