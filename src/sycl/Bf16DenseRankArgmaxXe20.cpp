#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "kernels/gemm/xe20/bf16_dense_rank_argmax.hpp"
#include "sgl_kernel_export.h"

namespace {
class Bf16DenseRankPartialsXe20Name;
class Bf16DenseRankArgmaxXe20Name;
}  // namespace

SGL_KERNEL_EXPORT torch::Tensor bf16_dense_rank_argmax_xe20(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& comparison_ranks) {
  constexpr auto name = "bf16_dense_rank_argmax_xe20";
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, name, " expects rank-two input and weight");
  TORCH_CHECK(
      input.is_xpu() && weight.device() == input.device() && comparison_ranks.device() == input.device(),
      name,
      " expects tensors on one XPU device");
  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
      name,
      " expects BF16 input and weight");
  TORCH_CHECK(
      comparison_ranks.scalar_type() == at::kUInt16 && comparison_ranks.dim() == 1 && comparison_ranks.numel() == 65536,
      name,
      " expects 65536 uint16 comparison ranks");
  TORCH_CHECK(
      input.is_contiguous() && weight.is_contiguous() && comparison_ranks.is_contiguous(),
      name,
      " expects contiguous tensors");
  TORCH_CHECK(
      reinterpret_cast<std::uintptr_t>(input.data_ptr()) % 64 == 0 &&
          reinterpret_cast<std::uintptr_t>(weight.data_ptr()) % 64 == 0,
      name,
      " requires 64-byte aligned input and weight base pointers");
  TORCH_CHECK(input.size(1) == weight.size(1), name, " K mismatch");
  const int64_t m = input.size(0), n = weight.size(0), k = input.size(1);
  TORCH_CHECK(
      k > 0 && k % 32 == 0 && n > 0 && n % 128 == 0,
      name,
      " requires positive K divisible by 32 and positive N divisible by 128");
  // The pinned Xe2 block-2D descriptor stores heights and byte widths/pitches
  // in 24 bits. Both input and weight have BF16 row pitch 2*K.
  TORCH_CHECK(
      m <= 0xFFFFFF && n <= 0xFFFFFF && k <= 0xFFFFFF / 2,
      name,
      " exceeds the Xe2 block-2D height or BF16 byte-pitch limit");
  constexpr int64_t max_int = std::numeric_limits<int>::max();
  TORCH_CHECK(
      m <= max_int - 127 && n <= max_int - 127 && k <= max_int / 2,
      name,
      " dimensions exceed the supported integer range");
  TORCH_CHECK(m * k <= max_int && n * k <= max_int, name, " input or weight element count exceeds int32 indexing");
  const int64_t tiles_n = n / 128, tiles = ((m + 127) / 128) * tiles_n;
  TORCH_CHECK(tiles <= max_int, name, " tile count exceeds int32");
  c10::DeviceGuard guard(input.device());
  auto output = at::empty({m}, input.options().dtype(at::kLong));
  if (m == 0) return output;
  auto partial = at::empty({m, tiles_n}, input.options().dtype(at::kUInt32));

  using namespace cute;
  using Element = cutlass::bfloat16_t;
  using Tile = Shape<_128, _128, _32>;
  using SG = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  auto dummy =
      make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), make_layout(make_shape(1, 1), make_stride(1, _1{})));
  auto dummy_bias =
      make_tensor(make_gmem_ptr(static_cast<float*>(nullptr)), make_layout(make_shape(1), make_stride(_1{})));
  using Tensor = decltype(dummy);
  using Bias = decltype(dummy_bias);
  using MMA = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, Element>>, Layout<Tile>, SG>::TiledMMA;
  using Base =
      MoE::MoEGEMM<Tile, SG, Tensor, Tensor, Tensor, Bias, MMA, MoE::ActivationType::GELU, false, false, Element>;
  using Kernel = sgl::gemm::Bf16DenseRankPartials<Base, MMA, Element>;
  typename Kernel::Params params{
      static_cast<const Element*>(input.data_ptr()),
      static_cast<const Element*>(weight.data_ptr()),
      comparison_ranks.data_ptr<std::uint16_t>(),
      partial.data_ptr<std::uint32_t>(),
      int(m),
      int(n),
      int(k),
      MMA{}};
  auto queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  const int threads = size(MMA{});
  const auto device = queue.get_device();
  const int sms = device.get_info<sycl::ext::intel::info::device::gpu_slices>() *
                  device.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  const int groups = std::min(int(tiles), sms * (512 / threads));
  namespace ex = sycl::ext::oneapi::experimental;
  namespace intel = sycl::ext::intel::experimental;
  ex::properties properties{ex::sub_group_size<16>, intel::grf_size<256>};
  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<std::uint32_t, 1> scratch(sycl::range<1>(128 * 4), handler);
    handler.parallel_for<Bf16DenseRankPartialsXe20Name>(
        sycl::nd_range<3>(sycl::range<3>(1, groups, threads), sycl::range<3>(1, 1, threads)),
        properties,
        [=](sycl::nd_item<3> item) {
          Kernel{}(params, item, scratch.template get_multi_ptr<sycl::access::decorated::no>().get());
        });
  });
  auto keys = partial.data_ptr<std::uint32_t>();
  auto ids = output.data_ptr<int64_t>();
  queue.parallel_for<Bf16DenseRankArgmaxXe20Name>(
      sycl::nd_range<1>(sycl::range<1>(m * 256), sycl::range<1>(256)), [=](sycl::nd_item<1> item) {
        const int row = item.get_group_linear_id(), tid = item.get_local_linear_id();
        std::uint64_t best = 0;
        for (int tile = tid; tile < tiles_n; tile += 256) {
          const auto local_key = keys[int64_t(row) * tiles_n + tile];
          const auto id = std::uint32_t(tile * 128 + 127 - (local_key & 127));
          const auto key = (std::uint64_t(local_key >> 7) << 32) | std::uint64_t(0xffffffffu - id);
          best = sycl::max(best, key);
        }
        best = sycl::reduce_over_group(item.get_group(), best, sycl::maximum<std::uint64_t>());
        if (tid == 0) ids[row] = 0xffffffffu - std::uint32_t(best);
      });
  return output;
}
