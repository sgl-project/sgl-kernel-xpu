// BF16 dense gate/up GEMM reusing the Xe2 grouped-GEMM mainloop.
// Dense problems have uniform tiles: static scheduling needs neither an
// expert-row-count tensor nor an atomic task counter or local-memory barrier.
#pragma once
#include "../../moe/xe20/bf16/moe_kernel.hpp"
#include "bf16_gelu_tanh_lut.hpp"

namespace sgl::gemm {

// Large-prefill specialization can reuse each weight tile across M tiles.
// Keep the previous traversal by default for all existing instantiations.
template <class Base, class MMA, class Element, bool MFast = false>
struct Bf16DenseGeluTanhMulLut {
  struct Params {
    const Element* a;
    const Element* w;
    Element* out;
    const Element* lut;
    int m, n, k, ld_b;
    MMA mma;
  };

  void operator()(const Params& p, sycl::nd_item<3> item) {
    using namespace cute;
    auto tile = p.mma.tile_mnk();
    const int tiles_m = (p.m + get<0>(tile) - 1) / get<0>(tile);
    const int tiles_n = (p.n / 2 + get<1>(tile) - 1) / get<1>(tile);
    auto a = make_tensor(
        make_gmem_ptr(const_cast<Element*>(p.a)), make_layout(make_shape(p.m, p.k), make_stride(p.k, _1{})));
    Base base;
    auto b = base.make_B_tensors(const_cast<Element*>(p.w), p.n, p.k, p.ld_b);
    auto out = base.make_D_tensors(p.out, 0, p.m, p.n);
    auto bias = base.make_Bias_tensors(nullptr, p.n);
    for (int task = item.get_group_linear_id(); task < tiles_m * tiles_n; task += item.get_group_range(1)) {
      typename Base::CollectiveMainloop mainloop;
      const int tile_m = MFast ? task % tiles_m : task / tiles_n;
      const int tile_n = MFast ? task / tiles_m : task % tiles_n;
      auto coordinate = make_coord(tile_m, tile_n, tile_n, 0);
      mainloop(
          a,
          get<0>(b),
          get<1>(b),
          out,
          coordinate,
          p.mma,
          int(item.get_local_linear_id()),
          get<0>(bias),
          get<1>(bias),
          1.702f,
          7.0f,
          Bf16GeluTanhLut{p.lut});
    }
  }
};

}  // namespace sgl::gemm
