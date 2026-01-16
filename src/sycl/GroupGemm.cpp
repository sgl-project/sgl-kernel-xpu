#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>

#include "Utils.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/device_memory.h"
#include "kernels/moe/moe_kernel.hpp"

using namespace cute;
using namespace MoE;

using ElementAccumulator = float;  // <- data type of accumulator

template <typename, typename, typename, typename, typename, typename, bool, bool>
class GemmCuteName;

template <typename Tile, typename SGLayout, bool FuseSiLU, bool WithBias>
void MoEGEMMLauncher(
    sycl::queue q,
    const void* activations,
    const void* weights,
    const void* scales,
    const void* bias,
    void* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* num_rows_per_expert_device,
    const int num_experts,
    int* workspace) {
  using Element = cutlass::bfloat16_t;

  auto make_dummy_tensor = [&](auto val, auto stride) {
    return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
  };
  auto make_dummy_bias = [&](auto val) {
    return make_tensor(make_gmem_ptr(&val), make_layout(Shape<int>{}, Stride<_1>{}));
  };
  using StrideA = Stride<int, _1>;
  using StrideB = Stride<int, _1>;
  using StrideD = Stride<int, _1>;
  using StrideBias = Stride<_1>;
  using TensorA = decltype(make_dummy_tensor(Element{}, StrideA{}));
  using TensorB = decltype(make_dummy_tensor(Element{}, StrideB{}));
  using TensorD = decltype(make_dummy_tensor(Element{}, StrideD{}));
  using TensorBias = decltype(make_dummy_bias(Element{}));

  using ElementA_non_CV = cutlass::platform::remove_cv_t<Element>;
  using MMA =
      typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, ElementA_non_CV>>, Layout<Tile>, SGLayout>::TiledMMA;
  auto mma = MMA{};

  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(mma);

  static constexpr int MaxThreadsPerSM = 512;

  TORCH_CHECK(
      MaxThreadsPerSM % MaxThreadsPerWorkgroup == 0, "MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup")

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>, intelex::grf_size<256>};

  using Kernel = MoE::MoEGEMM<Tile, SGLayout, TensorA, TensorB, TensorD, TensorBias, MMA, FuseSiLU, WithBias, Element>;
  typename Kernel::Params params{
      static_cast<const Element*>(activations),
      static_cast<const Element*>(weights),
      static_cast<const Element*>(bias),
      static_cast<Element*>(outputs),
      num_rows_per_expert_device,
      gemm_n,
      gemm_k,
      num_experts,
      workspace,
      mma,
  };

  auto stream = at::xpu::getCurrentXPUStream();
  auto Q = stream.queue();
  auto event = Q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), h);
    h.parallel_for<GemmCuteName<Tile, SGLayout, TensorA, TensorB, TensorD, Element, FuseSiLU, WithBias>>(
        sycl::nd_range<3>(global * local, local), kernel_props, [=](sycl::nd_item<3> item) {
          int32_t* slm_mem =
              static_cast<int32_t*>(local_mem.template get_multi_ptr<sycl::access::decorated::no>().get());
          Kernel{}(params, item, slm_mem);
        });
  });
}

#define LAUNCH_MOE(...)                       \
  MoEGEMMLauncher<__VA_ARGS__>(               \
      queue,                                  \
      activations.data_ptr(),                 \
      weights.data_ptr(),                     \
      nullptr,                                \
      bias_ptr,                               \
      output.data_ptr(),                      \
      gemm_n,                                 \
      gemm_k,                                 \
      total_rows_for_experts.data_ptr<int>(), \
      n_experts,                              \
      static_cast<int*>(atomic_buffer.data_ptr()))

#define DISPATCH_MOE_BY_CONDITION(condition, ...) \
  do {                                            \
    if (condition) {                              \
      LAUNCH_MOE(__VA_ARGS__, true);              \
    } else {                                      \
      LAUNCH_MOE(__VA_ARGS__, false);             \
    }                                             \
  } while (0)

#define DISPATCH_MOE_BY_FUSE_ACT_BIAS(FuseSilu, WithBias, ...) \
  do {                                                         \
    if (FuseSilu) {                                            \
      if (WithBias) {                                          \
        LAUNCH_MOE(__VA_ARGS__, true, true);                   \
      } else {                                                 \
        LAUNCH_MOE(__VA_ARGS__, true, false);                  \
      }                                                        \
    } else {                                                   \
      if (WithBias) {                                          \
        LAUNCH_MOE(__VA_ARGS__, false, true);                  \
      } else {                                                 \
        LAUNCH_MOE(__VA_ARGS__, false, false);                 \
      }                                                        \
    }                                                          \
  } while (0)

void moe_grouped_mm_nt(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& weights,
    const std::optional<at::Tensor>& bias,
    const torch::Tensor& total_rows_for_experts,
    const int64_t n_experts,
    bool fuse_silu) {
  int total_m = activations.sizes()[0];
  int gemm_k = activations.sizes()[1];
  auto weights_shape = weights.sizes().vec();
  int gemm_n = weights.sizes()[1];
  int avg_m = total_m / n_experts;

  TORCH_CHECK(weights_shape.size() == 3, "weights must be 3D");
  TORCH_CHECK(weights_shape[0] == n_experts, "weights must have n_experts as the first dimension");
  TORCH_CHECK(weights_shape[1] == gemm_n, "weights must be gemm_n * gemm_k");
  TORCH_CHECK(
      weights_shape[0] == total_rows_for_experts.size(0),
      "rows_for_experts must have the same size as the first dimension of weights");
  TORCH_CHECK(output.sizes()[0] == total_m, "output must have the same number of rows as activations");
  if (fuse_silu) {
    TORCH_CHECK(output.sizes()[1] == gemm_n / 2, "output must have half the number of columns as activations");
  } else {
    TORCH_CHECK(output.sizes()[1] == gemm_n, "output must have the same number of columns as activations");
  }
  TORCH_CHECK(n_experts % 8 == 0, "n_experts must be a multiple of 8 for the current implementation");
  TORCH_CHECK(
      activations.scalar_type() == weights.scalar_type(), "activations and weights must have the same data type");
  TORCH_CHECK(
      activations.scalar_type() == at::ScalarType::BFloat16,
      "Only bfloat16 are supported in moe_grouped_mm_nt currently");
  if (bias.has_value()) {
    TORCH_CHECK(bias->size(0) == n_experts && bias->size(1) == gemm_n, "bias shape mismatch with weight");
  }

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  at::Tensor atomic_buffer = at::empty({static_cast<long>(1)}, activations.options().dtype(at::kInt));
  bool with_bias = bias.has_value();
  void* bias_ptr = with_bias ? bias->data_ptr() : nullptr;

  if (avg_m <= 8) {
    DISPATCH_MOE_BY_FUSE_ACT_BIAS(
        fuse_silu, with_bias, Shape<_8, _64, _32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>);
  } else if (avg_m <= 16) {
    DISPATCH_MOE_BY_FUSE_ACT_BIAS(
        fuse_silu, with_bias, Shape<_16, _64, _32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>);
  } else if (avg_m <= 32) {
    DISPATCH_MOE_BY_FUSE_ACT_BIAS(
        fuse_silu, with_bias, Shape<_32, _64, _32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>);
  } else if (avg_m <= 128) {
    if (fuse_silu) {
      DISPATCH_MOE_BY_CONDITION(with_bias, Shape<_128, _32, _32>, Layout<Shape<_2, _8, _1>, Stride<_8, _1, _0>>, true);
    } else {
      DISPATCH_MOE_BY_CONDITION(with_bias, Shape<_128, _64, _32>, Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>, false);
    }
  } else {
    if (fuse_silu) {
      DISPATCH_MOE_BY_CONDITION(with_bias, Shape<_256, _32, _32>, Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>, true);
    } else {
      DISPATCH_MOE_BY_CONDITION(
          with_bias, Shape<_256, _128, _32>, Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>, false);
    }
  }
}
