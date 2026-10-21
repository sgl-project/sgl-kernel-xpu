#pragma once
#include <cute/atom/copy_traits_xe_2d.hpp>

namespace cute {
template <int N, class BaseT = void>
struct Xe2DPreparedPayloads_xe35 {
  int* payloads[N];

  template <int Field>
  CUTE_DEVICE static void add_payload_field_xe35(int* payload, int32_t delta) {
    static_assert(Field == 5 || Field == 6, "Only block X/Y payload fields can be updated.");
#ifdef __SYCL_DEVICE_ONLY__
    asm volatile(
        "{\n"
        ".decl PAYLOAD_D v_type=G type=D num_elts=8 alias=<%0,0>\n"
        ".decl DELTA_D v_type=G type=D num_elts=1 alias=<%1,0>\n"
        "add (M1_NM, 1) PAYLOAD_D(0,%2)<1> PAYLOAD_D(0,%2)<0;1,0> DELTA_D(0,0)<0;1,0>\n"
        "}\n"
        :
        : "rw.u"(payload), "rw.u"(delta), "P"(Field));
#else
    (void) payload;
    (void) delta;
    CUTE_INVALID_CONTROL_PATH("Xe 2D payload updates are only available on SYCL device.");
#endif
  }

  // Advance all N payloads by `delta` in the copy's coordinate space.
  //
  // get<> preserves per-element type: static elements (e.g. _0, C<8>) yield
  // integral_constant types whose arithmetic the compiler folds into immediates.
  // Zero-valued dimensions are elided entirely via if-constexpr on is_constant_v.
  template <class Coord>
  CUTE_DEVICE Xe2DPreparedPayloads_xe35& operator+=(Coord const& delta) {
    using Op = typename BaseT::CopyOp;
    static_assert(!BaseT::nontrivial_tiled_strides,
                  "Payload updates are only supported for modes handled by block 2D messages.");
#ifdef __SYCL_DEVICE_ONLY__
    auto dx = get<decltype(BaseT::get_x_mode())::value>(delta);
    auto dy = get<decltype(BaseT::get_y_mode())::value>(delta);
    CUTE_UNROLL
    for (int i = 0; i < N; ++i) {
      if constexpr (!is_constant_v<0, decltype(dx)>) {
        add_payload_field_xe35<5>(payloads[i], int32_t(dx) * BaseT::ValBits / Op::CopyBits);
      }
      if constexpr (!is_constant_v<0, decltype(dy)>) {
        add_payload_field_xe35<6>(payloads[i], int32_t(dy));
      }
    }
#else
    (void) delta;
    CUTE_INVALID_CONTROL_PATH("Xe 2D multi-payload copies are only available on SYCL device.");
#endif
    return *this;
  }

  // Convenience: treat a compile-time constant as a Y-only step.
  // Preserves static type so the inner operator+= takes the compile-time path.
  // Usage: prepared += C<SG_K>{};
  template <int Step>
  CUTE_DEVICE Xe2DPreparedPayloads_xe35& operator+=(C<Step>) {
    return *this += make_coord(_0{}, C<Step>{});
  }

  // Convenience: treat a runtime int as a Y-only step.
  CUTE_DEVICE Xe2DPreparedPayloads_xe35& operator+=(int step) {
    return *this += make_coord(_0{}, step);
  }
};

namespace detail_payloads_xe35 {

template <class Op, class XMode, class YMode, typename ValType, typename TiledStrides>
CUTE_HOST_DEVICE
Xe2DTraitsBase<Op, XMode, YMode, ValType, TiledStrides> const&
as_xe2d_base_xe35(Xe2DTraitsBase<Op, XMode, YMode, ValType, TiledStrides> const& o) {
  return o;
}

// Recursively create one payload per copy atom, traversing via nullspace.
template <int NumValDst, int N, class BaseT,
          class SEngine, class SLayout>
CUTE_DEVICE void
prepare_payloads_xe35(BaseT const& base,
                      Tensor<SEngine, SLayout> const& src,
                      Xe2DPreparedPayloads_xe35<N, BaseT>& payloads,
                      int& idx)
{
  using Op    = typename BaseT::CopyOp;
  constexpr int ValBits = BaseT::ValBits;

  if constexpr (SLayout::rank == 1 && decltype(size(SLayout{}))::value == NumValDst) {
#ifdef __SYCL_DEVICE_ONLY__
    using XMode = decltype(BaseT::get_x_mode());
    using YMode = decltype(BaseT::get_y_mode());
    auto coord = src.data().coord_;
    int32_t x = get<XMode::value>(coord) * ValBits / Op::CopyBits;
    int32_t y = get<YMode::value>(coord);
    uint64_t bp = base.base_ptr;
    if constexpr (BaseT::nontrivial_tiled_strides) {
      auto off = inner_product(coord, base.tiled_strides);
      bp += (off * ValBits) >> 3;
    }
    payloads.payloads[idx] =
        __builtin_IB_subgroup_copyBlock2DAddressPayload(base.payload);
        __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(payloads.payloads[idx], x);
        __builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(payloads.payloads[idx], y);
    if constexpr (BaseT::nontrivial_tiled_strides) {
      __builtin_IB_subgroup_setBlock2DAddressPayloadBase(payloads.payloads[idx], bp);
    }
#else
    CUTE_INVALID_CONTROL_PATH("Xe 2D multi-payload copies are only available on SYCL device.");
#endif
    ++idx;
  } else if constexpr (SLayout::rank > 1) {
    constexpr int R = SLayout::rank;
    auto src_v = group_modes<1, R>(src);
    auto src_null = nullspace(layout<1>(src_v));
    auto src_n = zipped_divide(src_v, make_tile(shape<0>(src_v), src_null));
    auto src_c = src_n(make_coord(_, Int<0>{}), make_coord(Int<0>{}, _));
    constexpr int Rest = decltype(size<1>(src_c))::value;
    CUTE_UNROLL
    for (int i = 0; i < Rest; ++i) {
      auto s_i = src_c(_, i);
      prepare_payloads_xe35<NumValDst>(base, s_i, payloads, idx);
    }
  } else {
    static_assert(is_tuple<decltype(shape<0>(SLayout{}))>::value,
                  "Cannot peel further: V mode is atomic but size does not match NumValDst");
    auto s_sub = tensor<0>(src);
    prepare_payloads_xe35<NumValDst>(base, s_sub, payloads, idx);
  }
}

// Recursively issue one load per copy atom, traversing via nullspace.
template <int NumValDst, int N, class BaseT,
          class DEngine, class DLayout>
CUTE_DEVICE void
copy_payloads_xe35(BaseT const& base,
                   Tensor<DEngine, DLayout>& dst,
                   Xe2DPreparedPayloads_xe35<N, BaseT> const& payloads,
                   int& idx)
{
  using Op    = typename BaseT::CopyOp;
  constexpr int ValBits = BaseT::ValBits;

  if constexpr (DLayout::rank == 1 && decltype(size(DLayout{}))::value == NumValDst) {
#ifdef __SYCL_DEVICE_ONLY__
    using ValT = int_byte_t<bits_to_bytes(ValBits)>;
    Op::copy(payloads.payloads[idx],
             const_cast<ValT*>(recast_ptr<ValT>(raw_pointer_cast(dst.data()))));
#else
    CUTE_INVALID_CONTROL_PATH("Xe 2D multi-payload copies are only available on SYCL device.");
#endif
    ++idx;
  } else if constexpr (DLayout::rank > 1) {
    constexpr int R = DLayout::rank;
    auto dst_v = group_modes<1, R>(dst);
    auto dst_null = nullspace(layout<1>(dst_v));
    auto dst_n = zipped_divide(dst_v, make_tile(shape<0>(dst_v), dst_null));
    auto dst_c = dst_n(make_coord(_, Int<0>{}), make_coord(Int<0>{}, _));
    constexpr int Rest = decltype(size<1>(dst_c))::value;
    CUTE_UNROLL
    for (int i = 0; i < Rest; ++i) {
      auto d_i = dst_c(_, i);
      copy_payloads_xe35<NumValDst>(base, d_i, payloads, idx);
    }
  } else {
    static_assert(is_tuple<decltype(shape<0>(DLayout{}))>::value,
                  "Cannot peel further: V mode is atomic but size does not match NumValDst");
    auto d_sub = tensor<0>(dst);
    copy_payloads_xe35<NumValDst>(base, d_sub, payloads, idx);
  }
}

} // namespace detail

// TODO: Fold these multi-payload API into the generic cute::copy / cute::prefetch_xe35 paths
template <class TiledCopy,
          class Engine, class Layout>
CUTE_DEVICE auto
prepare_payloads_xe35(TiledCopy const& tiled_copy,
                  Tensor<Engine, Layout> const& tensor)
{
  static_assert(is_counting_layout_v<Layout>, "tensor must be a coordinate tensor.");
  auto const& base = detail_payloads_xe35::as_xe2d_base_xe35(tiled_copy);
  using BaseT = remove_cvref_t<decltype(base)>;
  constexpr int NumValDst = decltype(size<1>(typename BaseT::Traits::DstLayout{}))::value / BaseT::ValBits;
  constexpr int Total     = decltype(size(Layout{}))::value;
  static_assert(Total % NumValDst == 0,
                "src tensor size is not a multiple of per-atom size");
  constexpr int N = Total / NumValDst;

  Xe2DPreparedPayloads_xe35<N, BaseT> prepared{};
  int idx = 0;
  detail_payloads_xe35::prepare_payloads_xe35<NumValDst>(base, tensor, prepared, idx);
  return prepared;
}

template <class TiledCopy, int N, class BaseT, class Engine, class Layout>
CUTE_DEVICE void
copy_payloads_xe35(TiledCopy const& tiled_copy,
     Xe2DPreparedPayloads_xe35<N, BaseT> const& prepared,
     Tensor<Engine, Layout>& tensor)
{
  static_assert(is_rmem_v<Engine>, "tensor must be in registers.");
  auto const& base = detail_payloads_xe35::as_xe2d_base_xe35(tiled_copy);
  constexpr int NumValDst = decltype(size<1>(typename BaseT::Traits::DstLayout{}))::value / BaseT::ValBits;

  int idx = 0;
  detail_payloads_xe35::copy_payloads_xe35<NumValDst>(base, tensor, prepared, idx);
}

template <class TiledCopy, int N, class BaseT>
CUTE_DEVICE void
prefetch_xe35(TiledCopy const& tiled_copy,
                       Xe2DPreparedPayloads_xe35<N, BaseT> const& prepared)
{
  auto const& base = detail_payloads_xe35::as_xe2d_base_xe35(tiled_copy);
  using Op = typename BaseT::CopyOp;
#ifdef __SYCL_DEVICE_ONLY__
  CUTE_UNROLL
  for (int i = 0; i < N; ++i) {
    Op::copy(prepared.payloads[i]);
  }
#else
  (void) base;
  (void) prepared;
  CUTE_INVALID_CONTROL_PATH("Xe 2D multi-payload copies are only available on SYCL device.");
#endif
}


template <class TiledCopy, class Engine, class Layout>
CUTE_DEVICE void prefetch_xe35(TiledCopy const& operation, Tensor<Engine, Layout> const& tensor) {
  prefetch(operation, tensor);
}
}
