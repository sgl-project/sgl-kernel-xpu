#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "cutlass/kernel_hardware_info.h"
#include "kernels/gemm/xe20/bf16_dense_gelu_tanh_mul_lut.hpp"
#include "sgl_kernel_export.h"

namespace {
template <class Tile>
class Bf16DenseGeluTanhMulLutXe20Name;
}  // namespace

// Availability of the optimized specialization, independent of device/dtype
// eligibility. Callers can query once before compiling or capturing a forward.
SGL_KERNEL_EXPORT bool bf16_dense_gelu_tanh_mul_lut_xe20_prefill_supported(int64_t m, int64_t n, int64_t k) {
  return m == 4096 && n == 24576 && k == 1536;
}

SGL_KERNEL_EXPORT torch::Tensor
bf16_dense_gelu_tanh_mul_lut_xe20(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& table) {
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "bf16_dense_gelu_tanh_mul_lut_xe20 expects rank-two tensors");
  TORCH_CHECK(
      input.is_xpu() && weight.device() == input.device(), "bf16_dense_gelu_tanh_mul_lut_xe20 expects one XPU device");
  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
      "bf16_dense_gelu_tanh_mul_lut_xe20 expects BF16 tensors");
  TORCH_CHECK(
      input.is_contiguous() && weight.is_contiguous(), "bf16_dense_gelu_tanh_mul_lut_xe20 expects contiguous tensors");
  TORCH_CHECK(input.size(1) == weight.size(1), "bf16_dense_gelu_tanh_mul_lut_xe20 K mismatch");
  TORCH_CHECK(
      table.device() == input.device() && table.scalar_type() == at::kBFloat16,
      "bf16_dense_gelu_tanh_mul_lut_xe20 expects a BF16 table on the "
      "input device");
  TORCH_CHECK(
      table.dim() == 1 && table.numel() == 65536 && table.is_contiguous(),
      "bf16_dense_gelu_tanh_mul_lut_xe20 expects a contiguous "
      "65536-entry table");
  const int64_t m = input.size(0), n = weight.size(0), k = input.size(1);
  TORCH_CHECK(
      k > 0 && k % 32 == 0 && n % 64 == 0,
      "bf16_dense_gelu_tanh_mul_lut_xe20 requires positive K divisible "
      "by 32 and N divisible by 64");
  // TLA 2D descriptors encode byte width/pitch and row height in 24 bits.
  // B0/B1 each have I=N/2 rows; D has I BF16 columns, so its byte width is N.
  constexpr int64_t max_2d = 0xFFFFFF;
  TORCH_CHECK(
      m <= max_2d && k <= max_2d / 2 && n <= max_2d,
      "bf16_dense_gelu_tanh_mul_lut_xe20 dimensions exceed the 24-bit "
      "2D descriptor range");
  constexpr int64_t max_int = std::numeric_limits<int>::max();
  TORCH_CHECK(
      m <= max_int - 127 && n <= max_int - 63 && k <= max_int / 2,
      "bf16_dense_gelu_tanh_mul_lut_xe20 dimensions exceed the "
      "supported integer range");
  TORCH_CHECK(
      input.numel() <= max_int && weight.numel() <= max_int && m * (n / 2) <= max_int,
      "bf16_dense_gelu_tanh_mul_lut_xe20 tensor offsets exceed int32");
  const int64_t tiles = ((m + 127) / 128) * ((n / 2 + 31) / 32);
  TORCH_CHECK(tiles <= max_int, "bf16_dense_gelu_tanh_mul_lut_xe20 tile count exceeds int32");
  c10::DeviceGuard guard(input.device());
  auto output = at::empty({m, n / 2}, input.options());
  if (m == 0 || n == 0) return output;

  // TLA 2D loads require 64-byte bases. Contiguous offset views can have a
  // misaligned base; clone only those operands on the current XPU stream.
  const auto aligned_input = reinterpret_cast<uintptr_t>(input.data_ptr()) % 64 == 0 ? input : input.clone();
  const auto aligned_weight = reinterpret_cast<uintptr_t>(weight.data_ptr()) % 64 == 0 ? weight : weight.clone();

  auto launch = [&]<bool Prefill>() {
    using namespace cute;
    using Element = cutlass::bfloat16_t;
    using Tile = std::conditional_t<Prefill, Shape<_128, _128, _32>, Shape<_128, _32, _32>>;
    using SG = std::conditional_t<
        Prefill,
        Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>,
        Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>>;
    auto dummy =
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), make_layout(make_shape(1, 1), make_stride(1, _1{})));
    auto dummy_bias =
        make_tensor(make_gmem_ptr(static_cast<float*>(nullptr)), make_layout(make_shape(1), make_stride(_1{})));
    using Tensor = decltype(dummy);
    using Bias = decltype(dummy_bias);
    using MMA = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, Element>>, Layout<Tile>, SG>::TiledMMA;
    using Base =
        MoE::MoEGEMM<Tile, SG, Tensor, Tensor, Tensor, Bias, MMA, MoE::ActivationType::GELU, true, false, Element>;
    using Kernel = sgl::gemm::Bf16DenseGeluTanhMulLut<Base, MMA, Element, Prefill>;
    typename Kernel::Params params{
        static_cast<const Element*>(aligned_input.data_ptr()),
        static_cast<const Element*>(aligned_weight.data_ptr()),
        static_cast<Element*>(output.data_ptr()),
        static_cast<const Element*>(table.data_ptr()),
        int(m),
        int(n),
        int(k),
        int(k),
        MMA{}};
    auto queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
    const int threads = size(MMA{});
    const auto device = queue.get_device();
    const int sms = device.get_info<sycl::ext::intel::info::device::gpu_slices>() *
                    device.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
    const int64_t selected_tiles = Prefill ? ((m + 127) / 128) * ((n / 2 + 127) / 128) : tiles;
    const int groups = std::min(int(selected_tiles), sms * (512 / threads));
    namespace ex = sycl::ext::oneapi::experimental;
    namespace intel = sycl::ext::intel::experimental;
    ex::properties properties{ex::sub_group_size<16>, intel::grf_size<256>};
    sycl::range<3> local(1, 1, threads);
    sycl::range<3> global(1, groups, threads);
    queue.submit([&](sycl::handler& handler) {
      handler.parallel_for<Bf16DenseGeluTanhMulLutXe20Name<Tile>>(
          sycl::nd_range<3>(global, local), properties, [=](sycl::nd_item<3> item) { Kernel{}(params, item); });
    });
  };
  if (bf16_dense_gelu_tanh_mul_lut_xe20_prefill_supported(m, n, k)) {
    launch.template operator()<true>();
  } else {
    launch.template operator()<false>();
  }
  return output;
}
