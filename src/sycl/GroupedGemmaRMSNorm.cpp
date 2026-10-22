#include <ATen/MemoryOverlap.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

constexpr int VEC = 8;  // 16-byte accesses for 2-byte dtypes
constexpr int ALIGN_BYTES = 16;
constexpr int MAX_WG_SIZE = 256;
constexpr int SG_SIZE = 16;
constexpr int MAX_ITERS = 8;                                           // largest ITERS instantiated in dispatch_iters
constexpr int64_t MAX_GRID_DIM = std::numeric_limits<int32_t>::max();  // SYCL id queries are 32-bit

template <typename T>
struct alignas(sizeof(T) * VEC) Vec8 {
  T v[VEC];
};

// out = x * rsqrt(mean(x^2) + eps) * (1 + w), with the mean taken per group_size slice of the last dim.
// One work-group handles one (row, group) of a 2-D grid; the group index selects the weight slice.
template <typename scalar_t, int ITERS>
struct GroupedGemmaRMSNormKernel {
  const scalar_t* __restrict__ input;
  const scalar_t* __restrict__ weight;
  scalar_t* __restrict__ output;
  int64_t group_size;
  int64_t num_groups;
  float eps;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<2> item) const {
    using vec_t = Vec8<scalar_t>;
    const int64_t group = item.get_group(1);
    const int64_t wg = static_cast<int64_t>(item.get_group(0)) * num_groups + group;
    const int64_t lid = item.get_local_id(1);
    const int64_t wg_size = item.get_local_range(1);
    const int64_t num_vecs = group_size / VEC;

    const vec_t* x = reinterpret_cast<const vec_t*>(input + wg * group_size);
    vec_t* y = reinterpret_cast<vec_t*>(output + wg * group_size);
    const vec_t* w = reinterpret_cast<const vec_t*>(weight + group * group_size);

    // Weight loads are issued before the reduction so their latency overlaps it.
    vec_t reg[ITERS];
    vec_t wreg[ITERS];
    float sum_sq = 0.f;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int64_t idx = it * wg_size + lid;
      if (idx < num_vecs) {
        reg[it] = x[idx];
        wreg[it] = w[idx];
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          const float f = static_cast<float>(reg[it].v[i]);
          sum_sq += f * f;
        }
      }
    }
    sum_sq = sycl::reduce_over_group(item.get_group(), sum_sq, sycl::plus<float>());
    const float rstd = sycl::rsqrt(sum_sq / static_cast<float>(group_size) + eps);

#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int64_t idx = it * wg_size + lid;
      if (idx < num_vecs) {
        const vec_t& wv = wreg[it];
        vec_t out;
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          out.v[i] =
              static_cast<scalar_t>(static_cast<float>(reg[it].v[i]) * rstd * (1.0f + static_cast<float>(wv.v[i])));
        }
        y[idx] = out;
      }
    }
  }
};

template <typename scalar_t, int ITERS>
void launch(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    int64_t rows,
    int64_t group_size,
    int64_t num_groups,
    double eps) {
  const int64_t num_vecs = group_size / VEC;
  const int64_t per_thread = (num_vecs + ITERS - 1) / ITERS;
  const int64_t wg_size = std::min<int64_t>(MAX_WG_SIZE, (per_thread + SG_SIZE - 1) / SG_SIZE * SG_SIZE);

  GroupedGemmaRMSNormKernel<scalar_t, ITERS> kernel{
      reinterpret_cast<const scalar_t*>(input.data_ptr()),
      reinterpret_cast<const scalar_t*>(weight.data_ptr()),
      reinterpret_cast<scalar_t*>(output.data_ptr()),
      group_size,
      num_groups,
      static_cast<float>(eps)};
  auto q = c10::xpu::getCurrentXPUStream().queue();
  sycl_kernel_submit(sycl::range<2>(rows, num_groups * wg_size), sycl::range<2>(1, wg_size), q, kernel);
}

template <typename scalar_t>
void dispatch_iters(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    int64_t rows,
    int64_t group_size,
    int64_t num_groups,
    double eps) {
  const int64_t iters = (group_size / VEC + MAX_WG_SIZE - 1) / MAX_WG_SIZE;
  TORCH_CHECK(
      iters <= MAX_ITERS,
      "grouped_gemma_rmsnorm: group_size ",
      group_size,
      " is too large (max ",
      MAX_ITERS * VEC * MAX_WG_SIZE,
      ")");
  if (iters <= 1) {
    launch<scalar_t, 1>(input, weight, output, rows, group_size, num_groups, eps);
  } else if (iters <= 2) {
    launch<scalar_t, 2>(input, weight, output, rows, group_size, num_groups, eps);
  } else if (iters <= 4) {
    launch<scalar_t, 4>(input, weight, output, rows, group_size, num_groups, eps);
  } else {
    launch<scalar_t, MAX_ITERS>(input, weight, output, rows, group_size, num_groups, eps);
  }
}

}  // namespace

namespace at::native::xpu {

// Gemma RMSNorm with the variance computed per group_size slice of the last dim. weight has the full hidden size.
SGL_KERNEL_EXPORT void grouped_gemma_rmsnorm(
    torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps, int64_t group_size) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(output);
  TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dim");
  TORCH_CHECK(input.sizes() == output.sizes(), "output must have the same shape as input");
  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
      "grouped_gemma_rmsnorm supports bfloat16 and float16, got ",
      input.scalar_type());
  TORCH_CHECK(output.scalar_type() == input.scalar_type(), "output dtype must match input");
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "weight dtype must match input");
  TORCH_CHECK(group_size > 0 && group_size % VEC == 0, "group_size must be a positive multiple of ", VEC);

  const int64_t hidden = input.size(-1);
  TORCH_CHECK(hidden % group_size == 0, "hidden size ", hidden, " must be divisible by group_size ", group_size);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == hidden, "weight must be 1-D with hidden_size elements");
  TORCH_CHECK(
      weight.device() == input.device() && output.device() == input.device(),
      "input, weight and output must be on the same device");
  for (const auto* t : {&input, &weight, &output}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % ALIGN_BYTES == 0, "tensors must be 16-byte aligned");
  }

  if (input.numel() == 0) {
    return;
  }
  // Exact aliasing (out=input) is safe: each work-group reads its slice before writing it.
  at::assert_no_partial_overlap(output, input);
  at::assert_no_overlap(output, weight);
  const int64_t rows = input.numel() / hidden;
  const int64_t num_groups = hidden / group_size;
  TORCH_CHECK(
      rows <= MAX_GRID_DIM && num_groups * MAX_WG_SIZE <= MAX_GRID_DIM,
      "grouped_gemma_rmsnorm: input is too large (rows ",
      rows,
      ", groups per row ",
      num_groups,
      ")");

  c10::DeviceGuard guard(input.device());
  SYCL_DISPATCH_ONLY_FLOATING16_TYPES(
      at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "grouped_gemma_rmsnorm", [&]() {
        dispatch_iters<scalar_t>(input, weight, output, rows, group_size, num_groups, eps);
      });
}

}  // namespace at::native::xpu
