#undef SYCL_INTEL_TARGET
#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "Utils.h"
#include "kernels/gemm/xe20/fp8_w8a16_gemm.hpp"
#include "sgl_kernel_export.h"

using namespace cute;

namespace {

template <typename, typename, int>
class Fp8W8A16DenseXe20Name;

template <typename Tile, typename SGLayout, int KSlices>
void launch_fp8_w8a16_dense(
    sycl::queue& q,
    const torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale) {
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementD = cutlass::bfloat16_t;
  auto make_dummy_tensor = [&](auto val, auto stride) {
    return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
  };
  using TensorA = decltype(make_dummy_tensor(ElementA{}, Stride<int, _1>{}));
  using TensorB = decltype(make_dummy_tensor(ElementB{}, Stride<int, _1>{}));
  using TensorD = decltype(make_dummy_tensor(ElementD{}, Stride<int, _1>{}));
  using MMA = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, ElementA>>, Layout<Tile>, SGLayout>::TiledMMA;
  using Kernel = fp8_w8a16_xe20::DenseW8A16Kernel<MMA, TensorA, TensorB, TensorD, KSlices>;

  const int M = static_cast<int>(input.size(0));
  const int K = static_cast<int>(input.size(1));
  const int N = static_cast<int>(weight.size(0));
  typename Kernel::Params params{
      reinterpret_cast<const ElementA*>(input.data_ptr()),
      reinterpret_cast<const ElementB*>(weight.data_ptr()),
      weight_scale.data_ptr<float>(),
      reinterpret_cast<ElementD*>(output.data_ptr()),
      M,
      N,
      K,
      static_cast<int>(input.stride(0)),
      static_cast<int>(weight.stride(0)),
      static_cast<int>(output.stride(0)),
      static_cast<int>(weight_scale.stride(0))};

  const int num_wg = ceil_div(M, Kernel::BLK_M) * ceil_div(N, Kernel::BLK_N);
  sycl::range<3> local(1, 1, Kernel::WGThreads);
  sycl::range<3> global(1, 1, static_cast<size_t>(num_wg) * Kernel::WGThreads);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<16>, intelex::grf_size<256>};
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> slm(sycl::range<1>(Kernel::SlmFloats), h);
    h.parallel_for<Fp8W8A16DenseXe20Name<Tile, SGLayout, KSlices>>(
        sycl::nd_range<3>(global, local), kernel_props, [=](sycl::nd_item<3> item) {
          float* slm_ptr = static_cast<float*>(slm.template get_multi_ptr<sycl::access::decorated::no>().get());
          Kernel{}(params, item, slm_ptr);
        });
  });
}

using Tile_8_64_32 = Shape<_8, _64, _32>;
using Tile_16_64_32 = Shape<_16, _64, _32>;
using Tile_32_64_32 = Shape<_32, _64, _32>;
using Tile_64_64_32 = Shape<_64, _64, _32>;
using Tile_128_128_16 = Shape<_128, _128, _16>;
using SG_1_4_1 = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using SG_4_2_1 = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;

// Split K only when each slice gets 8-16 scale groups (1-2K elements); measured on B70 at Qwen3-32B
// TP4 decode shapes, other splits lost to the unsplit kernel.
int pick_k_slices(int k_groups) {
  for (int s : {4, 2}) {
    if (k_groups % s == 0 && k_groups / s >= 8 && k_groups / s <= 16) return s;
  }
  return 1;
}

}  // namespace

#define FP8_W8A16_DENSE_KSLICES(TILE)                                                    \
  do {                                                                                   \
    if (k_slices == 4) {                                                                 \
      launch_fp8_w8a16_dense<TILE, SG_1_4_1, 4>(q, output, input, weight, weight_scale); \
    } else if (k_slices == 2) {                                                          \
      launch_fp8_w8a16_dense<TILE, SG_1_4_1, 2>(q, output, input, weight, weight_scale); \
    } else {                                                                             \
      launch_fp8_w8a16_dense<TILE, SG_1_4_1, 1>(q, output, input, weight, weight_scale); \
    }                                                                                    \
  } while (0)

SGL_KERNEL_EXPORT void fp8_w8a16_mm_xe20(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    int64_t config) {
  CHECK_DEVICE(output);
  CHECK_DEVICE(input);
  CHECK_DEVICE(weight);
  CHECK_DEVICE(weight_scale);
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float8_e4m3fn, "weight must be float8_e4m3fn");
  TORCH_CHECK(weight_scale.scalar_type() == at::kFloat, "weight_scale must be float32");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && output.dim() == 2 && weight_scale.dim() == 2);
  TORCH_CHECK(input.stride(1) == 1 && weight.stride(1) == 1 && output.stride(1) == 1 && weight_scale.stride(1) == 1);
  const int64_t M = input.size(0), K = input.size(1), N = weight.size(0);
  TORCH_CHECK(weight.size(1) == K, "weight K mismatch");
  TORCH_CHECK(output.size(0) == M && output.size(1) == N, "output must be [M, N]");
  TORCH_CHECK(K % 128 == 0 && N % 64 == 0, "K must be a multiple of 128 and N of 64");
  TORCH_CHECK(
      weight_scale.size(0) == (N + 127) / 128 && weight_scale.size(1) == K / 128,
      "weight_scale must be [N/128, K/128]");
  // 2D block loads need 64B-aligned bases and 16B-aligned row pitches.
  TORCH_CHECK(input.stride(0) * 2 % 16 == 0 && output.stride(0) * 2 % 16 == 0 && weight.stride(0) % 16 == 0);
  if (M == 0) return;

  auto q = at::xpu::getCurrentXPUStream().queue();
  const int k_groups = static_cast<int>(K / 128);
  int tile = static_cast<int>(config % 10);
  int k_slices = static_cast<int>(config / 10);
  if (config < 0) {
    tile = M <= 8 ? 0 : (M <= 16 ? 1 : (M <= 32 ? 2 : (M <= 128 ? 4 : 3)));
    k_slices = tile == 3 ? 1 : pick_k_slices(k_groups);
  }
  TORCH_CHECK(k_slices == 1 || k_slices == 2 || k_slices == 4, "k_slices must be 1, 2 or 4");
  TORCH_CHECK(k_groups % k_slices == 0, "K / 128 must be divisible by k_slices");
  switch (tile) {
    case 0:
      FP8_W8A16_DENSE_KSLICES(Tile_8_64_32);
      break;
    case 1:
      FP8_W8A16_DENSE_KSLICES(Tile_16_64_32);
      break;
    case 2:
      FP8_W8A16_DENSE_KSLICES(Tile_32_64_32);
      break;
    case 3:
      TORCH_CHECK(k_slices == 1, "the 128x128 tile does not split K");
      launch_fp8_w8a16_dense<Tile_128_128_16, SG_4_2_1, 1>(q, output, input, weight, weight_scale);
      break;
    case 4:
      FP8_W8A16_DENSE_KSLICES(Tile_64_64_32);
      break;
    default:
      TORCH_CHECK(false, "unknown tile config");
  }
}

#undef FP8_W8A16_DENSE_KSLICES
#undef SYCL_INTEL_TARGET
