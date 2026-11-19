#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <limits>

#include "cutlass/kernel_hardware_info.h"
#include "kernels/gemm/xe20/bf16_dense_gemm.hpp"
#include "sgl_kernel_export.h"

namespace {
template <class Tile, bool Pipeline, bool MFast>
class Bf16DenseGemmXe20Name;
}  // namespace

SGL_KERNEL_EXPORT torch::Tensor bf16_dense_gemm_xe20(const torch::Tensor& input, const torch::Tensor& weight) {
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "bf16_dense_gemm_xe20 expects rank-two tensors");
  TORCH_CHECK(input.is_xpu() && weight.device() == input.device(), "bf16_dense_gemm_xe20 expects one XPU device");
  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
      "bf16_dense_gemm_xe20 expects BF16 tensors");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "bf16_dense_gemm_xe20 expects contiguous tensors");
  TORCH_CHECK(input.size(1) == weight.size(1), "bf16_dense_gemm_xe20 K mismatch");
  const int64_t m = input.size(0), n = weight.size(0), k = input.size(1);
  TORCH_CHECK(
      k > 0 && k % 32 == 0 && n % 64 == 0,
      "bf16_dense_gemm_xe20 requires positive K divisible by 32 and N divisible by 64");
  constexpr int64_t max_int = std::numeric_limits<int>::max();
  TORCH_CHECK(
      m <= max_int - 127 && n <= max_int - 63 && k <= max_int / 2,
      "bf16_dense_gemm_xe20 dimensions exceed the supported integer range");
  const int64_t tiles = ((m + 127) / 128) * ((n + 63) / 64);
  TORCH_CHECK(tiles <= max_int, "bf16_dense_gemm_xe20 tile count exceeds int32");
  c10::DeviceGuard guard(input.device());
  auto output = at::empty({m, n}, input.options());
  if (m == 0 || n == 0) return output;

  auto launch = [&]<bool Pipeline, bool MFast = false>() {
    using namespace cute;
    using Element = cutlass::bfloat16_t;
    using Tile = Shape<_128, _64, _32>;
    using SG = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
    auto dummy =
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), make_layout(make_shape(1, 1), make_stride(1, _1{})));
    auto dummy_bias =
        make_tensor(make_gmem_ptr(static_cast<float*>(nullptr)), make_layout(make_shape(1), make_stride(_1{})));
    using Tensor = decltype(dummy);
    using Bias = decltype(dummy_bias);
    using MMA = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, Element>>, Layout<Tile>, SG>::TiledMMA;
    using Base = MoE::MoEGEMM<
        Tile,
        SG,
        Tensor,
        Tensor,
        Tensor,
        Bias,
        MMA,
        MoE::ActivationType::GELU,
        false,
        false,
        Element,
        Element,
        Element,
        Element,
        void,
        Pipeline>;
    using Kernel = sgl::gemm::Bf16DenseStatic<Base, MMA, Element, MFast>;
    typename Kernel::Params params{
        static_cast<const Element*>(input.data_ptr()),
        static_cast<const Element*>(weight.data_ptr()),
        static_cast<Element*>(output.data_ptr()),
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
    const int groups = std::min(int(tiles), sms * (512 / threads));
    namespace ex = sycl::ext::oneapi::experimental;
    namespace intel = sycl::ext::intel::experimental;
    ex::properties properties{ex::sub_group_size<16>, intel::grf_size<256>};
    sycl::range<3> local(1, 1, threads);
    sycl::range<3> global(1, groups, threads);
    queue.submit([&](sycl::handler& handler) {
      handler.parallel_for<Bf16DenseGemmXe20Name<Tile, Pipeline, MFast>>(
          sycl::nd_range<3>(global, local), properties, [=](sycl::nd_item<3> item) { Kernel{}(params, item); });
    });
  };
  // Register prefetch benefits selected larger problems; retain the single-slot
  // mainloop for small batches and other geometries.
  if (n == 1536 && m == 512 && (k == 12288 || k == 6144)) {
    launch.template operator()<true, true>();
  } else if (
      m == 256 && ((n == 1536 && (k == 12288 || k == 6144 || k == 2048 || k == 4096)) || (n == 2560 && k == 1536))) {
    launch.template operator()<true>();
  } else {
    launch.template operator()<false>();
  }
  return output;
}
