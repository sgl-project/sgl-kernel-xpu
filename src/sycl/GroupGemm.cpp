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
#include "kernels/moe/moe_tile_scheduler.hpp"

using namespace cute;
using namespace MoE;

using ElementAccumulator = float;  // <- data type of accumulator

template <typename, typename, typename, typename, typename, typename>
class GemmCuteName;

template <typename Tile, typename SGLayout>
void MoEGEMMLauncher(
    sycl::queue q,
    const void* activations,
    const void* weights,
    const void* scales,
    void* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* num_rows_per_expert_device,
    const int num_experts) {
  using Element = cutlass::bfloat16_t;
  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  // For example, in a framework, you could query device ID.
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info{0, sm_count};
  auto dummy_problem_shape = cute::Shape<int, int, int>{1, gemm_k, gemm_n};
  // The GroupedGEMM API requires creation of  a vector of ProblemShape objects
  // for each GEMM problem, which is used in the GroupedGEMM tile-scheduler. If
  // there are 32 groups, then a vector of 32 `ProblemShape` objects is created.
  // Since these would not be known at compile time for a framework, they would
  // have to be created at run-time instead. However, for MoEGEMM, I just
  // provide one dummy shape, and then the custom code in tile scheduler can
  // derive the shape of each GEMM problem.
  auto dummy_group_problem_shape =
      cutlass::gemm::GroupProblemShape<Shape<int, int, int>>{1, &dummy_problem_shape, nullptr};

  using ClusterShape = Shape<_1, _1, _1>;
  auto scheduler_params = PersistentTileSchedulerXeMoE<ProblemShape>::to_underlying_arguments(
      dummy_group_problem_shape,
      Tile{},
      ClusterShape{},
      hw_info,
      PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{1, RasterOrderOptions::AlongN});
  auto group_distribution = PersistentTileSchedulerXeMoE<ProblemShape>::get_grid_shape(
      scheduler_params,
      dummy_group_problem_shape,
      Tile{},
      ClusterShape{},
      hw_info,
      PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{1, RasterOrderOptions::AlongN});

  auto make_dummy_tensor = [&](auto val, auto stride) {
    return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
  };
  using StrideA = Stride<int, _1>;
  using StrideB = Stride<int, _1>;
  using StrideD = Stride<int, _1>;
  using TensorA = decltype(make_dummy_tensor(Element{}, StrideA{}));
  using TensorB = decltype(make_dummy_tensor(Element{}, StrideB{}));
  using TensorD = decltype(make_dummy_tensor(Element{}, StrideD{}));

  using Kernel = MoE::MoEGEMM<Tile, SGLayout, TensorA, TensorB, TensorD, Element>;
  typename Kernel::Params params{
      static_cast<const Element*>(activations),
      static_cast<const Element*>(weights),
      static_cast<Element*>(outputs),
      num_rows_per_expert_device,
      gemm_n,
      gemm_k,
      num_experts,
      scheduler_params,
  };

  auto MaxThreadsPerWorkgroup = Kernel::SGPerWG::value * intel::sg_size;
  dim3 local_range{static_cast<unsigned int>(MaxThreadsPerWorkgroup), 1, 1};

  sycl::range<3> local = {local_range.x, local_range.y, local_range.z};
  sycl::range<3> groups = {group_distribution.x, group_distribution.y, group_distribution.z};
  sycl::range<3> global = {local[0] * groups[0], local[1] * groups[1], local[2] * groups[2]};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<16>, intelex::grf_size<256>};
  sycl::queue Q = compat::get_default_queue();
  auto event = Q.submit([&](sycl::handler& h) {
    h.parallel_for<GemmCuteName<Tile, SGLayout, TensorA, TensorB, TensorD, Element>>(
        sycl::nd_range<3>(global, local), kernel_props, [=](sycl::nd_item<3> item) { Kernel{}(params, item); });
  });
}

#define LAUNCH_MOE(...)                       \
  MoEGEMMLauncher<__VA_ARGS__>(               \
      queue,                                  \
      activations.data_ptr(),                 \
      weights.data_ptr(),                     \
      nullptr,                                \
      output.data_ptr(),                      \
      gemm_n,                                 \
      gemm_k,                                 \
      total_rows_for_experts.data_ptr<int>(), \
      n_experts)

void moe_grouped_mm_nt(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& weights,
    const torch::Tensor& total_rows_for_experts,
    const int64_t n_experts) {
  int total_m = activations.sizes()[0];
  int gemm_k = activations.sizes()[1];
  auto weights_shape = weights.sizes().vec();
  int gemm_n = weights.sizes()[1];

  TORCH_CHECK(weights_shape.size() == 3, "weights must be 3D");
  TORCH_CHECK(weights_shape[0] == n_experts, "weights must have n_experts as the first dimension");
  TORCH_CHECK(weights_shape[1] == gemm_n, "weights must be gemm_n * gemm_k");
  TORCH_CHECK(
      weights_shape[0] == total_rows_for_experts.size(0),
      "rows_for_experts must have the same size as the first dimension of weights");
  TORCH_CHECK(output.sizes()[0] == total_m, "output must have the same number of rows as activations");
  TORCH_CHECK(output.sizes()[1] == gemm_n, "output must have the same number of columns as activations");
  TORCH_CHECK(n_experts % 8 == 0, "n_experts must be a multiple of 8 for the current implementation");
  TORCH_CHECK(
      activations.scalar_type() == weights.scalar_type(), "activations and weights must have the same data type");
  TORCH_CHECK(
      activations.scalar_type() == at::ScalarType::BFloat16,
      "Only bfloat16 are supported in moe_grouped_mm_nt currently");

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  if (total_m <= 8) {
    LAUNCH_MOE(Shape<_8, _64, _32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>);
  } else if (total_m <= 16) {
    LAUNCH_MOE(Shape<_16, _64, _32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>);
  } else if (total_m <= 32) {
    LAUNCH_MOE(Shape<_32, _64, _32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>);
  } else {
    LAUNCH_MOE(Shape<_256, _256, _32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>);
  }
}
