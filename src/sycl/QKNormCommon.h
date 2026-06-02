#pragma once

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <c10/xpu/XPUStream.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "MemoryAccess.h"
#include "Norm.h"
#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu::qknorm {

template <typename scalar_t>
struct AccType {
  using type = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
};

template <typename scalar_t>
using acc_type_t = typename AccType<scalar_t>::type;

struct CTAConfig {
  int workgroup_size;
  int iters;
  int vec_size;
};

struct WarpConfig {
  int subgroups_per_wg;
  int workgroup_size;
};

template <typename scalar_t, typename weight_t>
struct Layout {
  scalar_t* q;
  scalar_t* k;
  const weight_t* q_weight;
  const weight_t* k_weight;
  int64_t q_token_stride;
  int64_t k_token_stride;
  int64_t q_head_stride;
  int64_t k_head_stride;
  int64_t num_tokens;
  int64_t num_q_heads;
  int64_t num_k_heads;
  int64_t head_dim;
};

struct NoPostOp {
  template <typename accscalar_t, int num_tiles, int vec_size, int head_dim>
  void apply(accscalar_t (&)[num_tiles][vec_size], sycl::nd_item<1>, int64_t, int64_t, bool, int64_t, int64_t) const {}
};

template <typename scalar_t, typename weight_t>
Layout<scalar_t, weight_t>
make_separated_layout(torch::Tensor& q, torch::Tensor& k, torch::Tensor& q_weight, torch::Tensor& k_weight) {
  return Layout<scalar_t, weight_t>{
      static_cast<scalar_t*>(q.data_ptr()),
      static_cast<scalar_t*>(k.data_ptr()),
      static_cast<const weight_t*>(q_weight.data_ptr()),
      static_cast<const weight_t*>(k_weight.data_ptr()),
      q.stride(0),
      k.stride(0),
      q.stride(1),
      k.stride(1),
      q.size(0),
      q.size(1),
      k.size(1),
      q.size(2)};
}

template <typename scalar_t, typename weight_t>
Layout<scalar_t, weight_t> make_packed_qkv_layout_from_ptr(
    scalar_t* qkv_ptr,
    const weight_t* q_weight_ptr,
    const weight_t* k_weight_ptr,
    int64_t num_tokens,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim) {
  const int64_t token_stride = (num_heads_q + num_heads_k + num_heads_v) * head_dim;
  scalar_t* q_base = qkv_ptr;
  scalar_t* k_base = q_base + num_heads_q * head_dim;
  return Layout<scalar_t, weight_t>{
      q_base,
      k_base,
      q_weight_ptr,
      k_weight_ptr,
      token_stride,
      token_stride,
      head_dim,
      head_dim,
      num_tokens,
      num_heads_q,
      num_heads_k,
      head_dim};
}

template <typename scalar_t, typename weight_t>
Layout<scalar_t, weight_t> make_packed_qkv_layout(
    torch::Tensor& qkv,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim) {
  return make_packed_qkv_layout_from_ptr<scalar_t, weight_t>(
      static_cast<scalar_t*>(qkv.data_ptr()),
      static_cast<const weight_t*>(q_weight.data_ptr()),
      static_cast<const weight_t*>(k_weight.data_ptr()),
      qkv.size(0),
      num_heads_q,
      num_heads_k,
      num_heads_v,
      head_dim);
}

template <typename scalar_t>
constexpr int max_vec_size() {
  constexpr int float4_size = sizeof(float) * 4;
  return float4_size / sizeof(scalar_t);
}

inline int max_workgroup_size() {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int max_wg_size = static_cast<int>(dpcppMaxWorkGroupSize(dev_id));
  if constexpr (NUM_REDUCE_STAGES == 16) {
    max_wg_size = std::min(max_wg_size, 512);
  }
  return max_wg_size;
}

inline int cap_workgroup_size(int max_wg_size) {
  if constexpr (NUM_REDUCE_STAGES == 16) {
    max_wg_size = std::min(max_wg_size, 512);
  }
  return std::max(max_wg_size, NUM_REDUCE_STAGES);
}

template <typename scalar_t, typename weight_t>
int vec_size(const Layout<scalar_t, weight_t>& layout, int max_vec_size_value, int64_t tile_lanes) {
  int result = get_min_vec_size(max_vec_size_value, layout.q, layout.k, layout.q_weight, layout.k_weight);
  while (result > 1 && (layout.head_dim % (tile_lanes * result) != 0 || layout.q_token_stride % result != 0 ||
                        layout.k_token_stride % result != 0 || layout.q_head_stride % result != 0 ||
                        layout.k_head_stride % result != 0)) {
    result = result >> 1;
  }
  return result;
}

inline WarpConfig warp_config(int64_t num_works, int max_wg_size) {
  max_wg_size = cap_workgroup_size(max_wg_size);
  const int max_subgroups_per_wg = std::max(1, max_wg_size / NUM_REDUCE_STAGES);
  const int subgroups_per_wg = std::max(1, static_cast<int>(std::min<int64_t>(num_works, max_subgroups_per_wg)));
  return {subgroups_per_wg, subgroups_per_wg * NUM_REDUCE_STAGES};
}

inline WarpConfig warp_config(int64_t num_works) {
  return warp_config(num_works, max_workgroup_size());
}

inline int cta_workgroup_size(int head_dim, int vec_size_value) {
  const int max_wg_size = max_workgroup_size();
  const int plane_vecs = (head_dim + vec_size_value - 1) / vec_size_value;
  int workgroup_size = (plane_vecs + NUM_REDUCE_STAGES - 1) / NUM_REDUCE_STAGES * NUM_REDUCE_STAGES;
  workgroup_size = std::min(workgroup_size, max_wg_size);
  return std::max(workgroup_size, NUM_REDUCE_STAGES);
}

template <typename scalar_t, typename weight_t>
CTAConfig
cta_config(const Layout<scalar_t, weight_t>& layout, int64_t num_works, int64_t head_dim, const char* op_name) {
  TORCH_CHECK(num_works <= std::numeric_limits<int>::max(), op_name, ": num_works is too large");
  TORCH_CHECK(head_dim <= std::numeric_limits<int>::max(), op_name, ": head_dim is too large");

  const int head_dim_int = static_cast<int>(head_dim);
  const int max_vec_size_value = max_vec_size<scalar_t>();
  constexpr int64_t cta_tile_lanes = 1;
  const int vec_size_value = vec_size<scalar_t, weight_t>(layout, max_vec_size_value, cta_tile_lanes);
  const int workgroup_size = cta_workgroup_size(head_dim_int, vec_size_value);
  const int iters = (head_dim_int + workgroup_size * vec_size_value - 1) / (workgroup_size * vec_size_value);
  return {workgroup_size, iters, vec_size_value};
}

template <typename scalar_t, typename weight_t, int head_dim, int vec_size_value, typename PostOp>
struct WarpKernel {
  using accscalar_t = acc_type_t<scalar_t>;
  using vec_t = aligned_vector_loop<scalar_t, vec_size_value>;
  using weight_vec_t = aligned_vector_loop<weight_t, vec_size_value>;
  static_assert(head_dim % (NUM_REDUCE_STAGES * vec_size_value) == 0);
  static constexpr int num_tiles = head_dim / (NUM_REDUCE_STAGES * vec_size_value);

  Layout<scalar_t, weight_t> layout;
  accscalar_t eps;
  PostOp post_op;

  [[sycl::reqd_sub_group_size(NUM_REDUCE_STAGES)]] void operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const int64_t local_id = item.get_local_id(0);
    const int64_t subgroup_id = local_id / NUM_REDUCE_STAGES;
    const int64_t lane_id = local_id % NUM_REDUCE_STAGES;
    const int64_t subgroups_per_wg = item.get_local_range(0) / NUM_REDUCE_STAGES;
    const int64_t worker_id = item.get_group(0) * subgroups_per_wg + subgroup_id;
    const int64_t num_workers = item.get_group_range(0) * subgroups_per_wg;
    const int64_t total_qk_heads = layout.num_q_heads + layout.num_k_heads;
    const int64_t num_works = layout.num_tokens * total_qk_heads;

    for (int64_t work_id = worker_id; work_id < num_works; work_id += num_workers) {
      const int64_t token_id = work_id / total_qk_heads;
      const int64_t head_id = work_id % total_qk_heads;
      const bool is_q = head_id < layout.num_q_heads;
      const int64_t local_head_id = is_q ? head_id : head_id - layout.num_q_heads;

      scalar_t* input = is_q ? layout.q + token_id * layout.q_token_stride + local_head_id * layout.q_head_stride
                             : layout.k + token_id * layout.k_token_stride + local_head_id * layout.k_head_stride;
      const weight_t* weight = is_q ? layout.q_weight : layout.k_weight;

      accscalar_t sum = 0;
      vec_t input_tiles[num_tiles];
#pragma unroll
      for (int tile = 0; tile < num_tiles; ++tile) {
        const int64_t dim = (static_cast<int64_t>(tile) * NUM_REDUCE_STAGES + lane_id) * vec_size_value;
        const vec_t values = *(reinterpret_cast<const vec_t*>(input + dim));
        input_tiles[tile] = values;
#pragma unroll
        for (int v = 0; v < vec_size_value; ++v) {
          const accscalar_t value = static_cast<accscalar_t>(values[v]);
          sum += value * value;
        }
      }

      sum = sycl::reduce_over_group(sg, sum, sycl::plus<accscalar_t>());
      const accscalar_t rstd = Numerics<accscalar_t>::rsqrt(sum / static_cast<accscalar_t>(head_dim) + eps);

      accscalar_t output_tiles[num_tiles][vec_size_value];
#pragma unroll
      for (int tile = 0; tile < num_tiles; ++tile) {
        const int64_t dim = (static_cast<int64_t>(tile) * NUM_REDUCE_STAGES + lane_id) * vec_size_value;
        const vec_t values = input_tiles[tile];
        const weight_vec_t gamma = *(reinterpret_cast<const weight_vec_t*>(weight + dim));
#pragma unroll
        for (int v = 0; v < vec_size_value; ++v) {
          output_tiles[tile][v] = static_cast<accscalar_t>(values[v]) * rstd * static_cast<accscalar_t>(gamma[v]);
        }
      }

      post_op.template apply<accscalar_t, num_tiles, vec_size_value, head_dim>(
          output_tiles, item, token_id, local_head_id, is_q, lane_id, work_id);

#pragma unroll
      for (int tile = 0; tile < num_tiles; ++tile) {
        const int64_t dim = (static_cast<int64_t>(tile) * NUM_REDUCE_STAGES + lane_id) * vec_size_value;
        vec_t output;
#pragma unroll
        for (int v = 0; v < vec_size_value; ++v) {
          output[v] = static_cast<scalar_t>(output_tiles[tile][v]);
        }
        *(reinterpret_cast<vec_t*>(input + dim)) = output;
      }
    }
  }
};

template <typename scalar_t, typename weight_t, int ITERS, int vec_size_value>
struct CTARegisterKernel {
  using accscalar_t = acc_type_t<scalar_t>;
  using vec_t = aligned_vector_loop<scalar_t, vec_size_value>;
  using weight_vec_t = aligned_vector_loop<weight_t, vec_size_value>;
  static_assert(ITERS > 0);

  Layout<scalar_t, weight_t> layout;
  accscalar_t eps;

  [[sycl::reqd_sub_group_size(NUM_REDUCE_STAGES)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t workgroup_id = item.get_group(0);
    const int64_t num_workgroups = item.get_group_range(0);
    const int64_t local_id = item.get_local_id(0);
    const int64_t workgroup_size = item.get_local_range(0);
    const int64_t total_qk_heads = layout.num_q_heads + layout.num_k_heads;
    const int64_t num_works = layout.num_tokens * total_qk_heads;

    for (int64_t work_id = workgroup_id; work_id < num_works; work_id += num_workgroups) {
      const int64_t token_id = work_id / total_qk_heads;
      const int64_t head_id = work_id % total_qk_heads;
      const bool is_q = head_id < layout.num_q_heads;
      const int64_t local_head_id = is_q ? head_id : head_id - layout.num_q_heads;

      scalar_t* input = is_q ? layout.q + token_id * layout.q_token_stride + local_head_id * layout.q_head_stride
                             : layout.k + token_id * layout.k_token_stride + local_head_id * layout.k_head_stride;
      const weight_t* weight = is_q ? layout.q_weight : layout.k_weight;

      accscalar_t sum = 0;
      vec_t input_tiles[ITERS];
#pragma unroll
      for (int tile = 0; tile < ITERS; ++tile) {
        const int64_t dim = (static_cast<int64_t>(tile) * workgroup_size + local_id) * vec_size_value;
        if (dim < layout.head_dim) {
          const vec_t values = *(reinterpret_cast<const vec_t*>(input + dim));
          input_tiles[tile] = values;
#pragma unroll
          for (int v = 0; v < vec_size_value; ++v) {
            const accscalar_t value = static_cast<accscalar_t>(values[v]);
            sum += value * value;
          }
        }
      }

      sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<accscalar_t>());
      const accscalar_t rstd = Numerics<accscalar_t>::rsqrt(sum / static_cast<accscalar_t>(layout.head_dim) + eps);

#pragma unroll
      for (int tile = 0; tile < ITERS; ++tile) {
        const int64_t dim = (static_cast<int64_t>(tile) * workgroup_size + local_id) * vec_size_value;
        if (dim < layout.head_dim) {
          const vec_t values = input_tiles[tile];
          const weight_vec_t gamma = *(reinterpret_cast<const weight_vec_t*>(weight + dim));
          vec_t output;
#pragma unroll
          for (int v = 0; v < vec_size_value; ++v) {
            output[v] =
                static_cast<scalar_t>(static_cast<accscalar_t>(values[v]) * rstd * static_cast<accscalar_t>(gamma[v]));
          }
          *(reinterpret_cast<vec_t*>(input + dim)) = output;
        }
      }
    }
  }
};

template <typename scalar_t, typename weight_t, int vec_size_value>
struct CTATwoPassKernel {
  using accscalar_t = acc_type_t<scalar_t>;
  using vec_t = aligned_vector_loop<scalar_t, vec_size_value>;
  using weight_vec_t = aligned_vector_loop<weight_t, vec_size_value>;

  Layout<scalar_t, weight_t> layout;
  accscalar_t eps;

  [[sycl::reqd_sub_group_size(NUM_REDUCE_STAGES)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t workgroup_id = item.get_group(0);
    const int64_t num_workgroups = item.get_group_range(0);
    const int64_t local_id = item.get_local_id(0);
    const int64_t workgroup_size = item.get_local_range(0);
    const int64_t total_qk_heads = layout.num_q_heads + layout.num_k_heads;
    const int64_t num_works = layout.num_tokens * total_qk_heads;

    for (int64_t work_id = workgroup_id; work_id < num_works; work_id += num_workgroups) {
      const int64_t token_id = work_id / total_qk_heads;
      const int64_t head_id = work_id % total_qk_heads;
      const bool is_q = head_id < layout.num_q_heads;
      const int64_t local_head_id = is_q ? head_id : head_id - layout.num_q_heads;

      scalar_t* input = is_q ? layout.q + token_id * layout.q_token_stride + local_head_id * layout.q_head_stride
                             : layout.k + token_id * layout.k_token_stride + local_head_id * layout.k_head_stride;
      const weight_t* weight = is_q ? layout.q_weight : layout.k_weight;

      accscalar_t sum = 0;
      for (int64_t dim = local_id * vec_size_value; dim < layout.head_dim; dim += workgroup_size * vec_size_value) {
        const vec_t values = *(reinterpret_cast<const vec_t*>(input + dim));
#pragma unroll
        for (int v = 0; v < vec_size_value; ++v) {
          const accscalar_t value = static_cast<accscalar_t>(values[v]);
          sum += value * value;
        }
      }

      sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<accscalar_t>());
      const accscalar_t rstd = Numerics<accscalar_t>::rsqrt(sum / static_cast<accscalar_t>(layout.head_dim) + eps);

      for (int64_t dim = local_id * vec_size_value; dim < layout.head_dim; dim += workgroup_size * vec_size_value) {
        const vec_t values = *(reinterpret_cast<const vec_t*>(input + dim));
        const weight_vec_t gamma = *(reinterpret_cast<const weight_vec_t*>(weight + dim));
        vec_t output;
#pragma unroll
        for (int v = 0; v < vec_size_value; ++v) {
          output[v] =
              static_cast<scalar_t>(static_cast<accscalar_t>(values[v]) * rstd * static_cast<accscalar_t>(gamma[v]));
        }
        *(reinterpret_cast<vec_t*>(input + dim)) = output;
      }
    }
  }
};

template <typename scalar_t, typename weight_t, int head_dim, int vec_size_value, typename PostOp>
void launch_warp_kernel(const Layout<scalar_t, weight_t>& layout, acc_type_t<scalar_t> eps, const PostOp& post_op) {
  const int64_t num_works = layout.num_tokens * (layout.num_q_heads + layout.num_k_heads);
  const WarpConfig config = warp_config(num_works);
  const int subgroups_per_wg = config.subgroups_per_wg;
  const int workgroup_size = config.workgroup_size;

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  const int dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const int total_resource = std::max(
      NUM_REDUCE_STAGES,
      static_cast<int>(dpcppMaxWorkItemsPerTile(dev_id) / dpcppMaxSubGroupSize(dev_id) * NUM_REDUCE_STAGES));
  const int max_resident_wgs = std::max(1, total_resource / workgroup_size);
  const int64_t needed_wgs = (num_works + subgroups_per_wg - 1) / subgroups_per_wg;
  const int num_wgs = std::max(1, static_cast<int>(std::min<int64_t>(needed_wgs, max_resident_wgs)));

  using kernel_t = WarpKernel<scalar_t, weight_t, head_dim, vec_size_value, PostOp>;
  kernel_t kernel{layout, eps, post_op};

  sycl_kernel_submit(
      sycl::range<1>(static_cast<size_t>(num_wgs * workgroup_size)),
      sycl::range<1>(static_cast<size_t>(workgroup_size)),
      queue,
      kernel);
}

template <typename scalar_t, typename weight_t, int head_dim, int vec_size_value, typename PostOp>
void launch_warp_kernel_if_supported(
    const Layout<scalar_t, weight_t>& layout, acc_type_t<scalar_t> eps, const PostOp& post_op, const char* op_name) {
  if constexpr (head_dim % (NUM_REDUCE_STAGES * vec_size_value) == 0) {
    launch_warp_kernel<scalar_t, weight_t, head_dim, vec_size_value, PostOp>(layout, eps, post_op);
  } else {
    TORCH_CHECK(false, op_name, ": unsupported warp vec_size for head_dim");
  }
}

template <typename scalar_t, typename weight_t, int head_dim, typename PostOp>
void dispatch_warp_vec_size(
    const Layout<scalar_t, weight_t>& layout, acc_type_t<scalar_t> eps, const PostOp& post_op, const char* op_name) {
  const int max_vec_size_value = max_vec_size<scalar_t>();
  constexpr int64_t subgroup_tile_lanes = NUM_REDUCE_STAGES;
  const int vec_size_value = vec_size<scalar_t, weight_t>(layout, max_vec_size_value, subgroup_tile_lanes);

#define QKNORM_WARP_VEC_CASE(VEC_SIZE) \
  launch_warp_kernel_if_supported<scalar_t, weight_t, head_dim, VEC_SIZE, PostOp>(layout, eps, post_op, op_name)

  switch (vec_size_value) {
    case 16:
      QKNORM_WARP_VEC_CASE(16);
      break;
    case 8:
      QKNORM_WARP_VEC_CASE(8);
      break;
    case 4:
      QKNORM_WARP_VEC_CASE(4);
      break;
    case 2:
      QKNORM_WARP_VEC_CASE(2);
      break;
    default:
      QKNORM_WARP_VEC_CASE(1);
      break;
  }

#undef QKNORM_WARP_VEC_CASE
}

template <typename scalar_t, typename weight_t, int ITERS, int vec_size_value>
void launch_cta_register_kernel(
    const Layout<scalar_t, weight_t>& layout, acc_type_t<scalar_t> eps, const CTAConfig& config) {
  const int64_t num_works = layout.num_tokens * (layout.num_q_heads + layout.num_k_heads);
  const int workgroup_size = config.workgroup_size;

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  const int dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const int max_resident_wgs = std::max(1, static_cast<int>(dpcppMaxWorkItemsPerTile(dev_id)) / workgroup_size);
  const int num_wgs = std::max(1, std::min(static_cast<int>(num_works), max_resident_wgs));

  CTARegisterKernel<scalar_t, weight_t, ITERS, vec_size_value> kernel{layout, eps};

  sycl_kernel_submit(
      sycl::range<1>(static_cast<size_t>(num_wgs * workgroup_size)),
      sycl::range<1>(static_cast<size_t>(workgroup_size)),
      queue,
      kernel);
}

template <typename scalar_t, typename weight_t, int vec_size_value>
void launch_cta_two_pass_kernel(
    const Layout<scalar_t, weight_t>& layout, acc_type_t<scalar_t> eps, const CTAConfig& config) {
  const int64_t num_works = layout.num_tokens * (layout.num_q_heads + layout.num_k_heads);
  const int workgroup_size = config.workgroup_size;

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  const int dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const int max_resident_wgs = std::max(1, static_cast<int>(dpcppMaxWorkItemsPerTile(dev_id)) / workgroup_size);
  const int num_wgs = std::max(1, std::min(static_cast<int>(num_works), max_resident_wgs));

  CTATwoPassKernel<scalar_t, weight_t, vec_size_value> kernel{layout, eps};

  sycl_kernel_submit(
      sycl::range<1>(static_cast<size_t>(num_wgs * workgroup_size)),
      sycl::range<1>(static_cast<size_t>(workgroup_size)),
      queue,
      kernel);
}

template <typename scalar_t, typename weight_t, int vec_size_value>
void dispatch_cta_iters(const CTAConfig& config, const Layout<scalar_t, weight_t>& layout, acc_type_t<scalar_t> eps) {
#define QKNORM_CTA_REGISTER_CASE(ITERS) \
  launch_cta_register_kernel<scalar_t, weight_t, ITERS, vec_size_value>(layout, eps, config)

  switch (config.iters) {
    case 1:
      QKNORM_CTA_REGISTER_CASE(1);
      break;
    case 2:
      QKNORM_CTA_REGISTER_CASE(2);
      break;
    case 3:
      QKNORM_CTA_REGISTER_CASE(3);
      break;
    case 4:
      QKNORM_CTA_REGISTER_CASE(4);
      break;
    case 5:
      QKNORM_CTA_REGISTER_CASE(5);
      break;
    case 6:
      QKNORM_CTA_REGISTER_CASE(6);
      break;
    case 7:
      QKNORM_CTA_REGISTER_CASE(7);
      break;
    case 8:
      QKNORM_CTA_REGISTER_CASE(8);
      break;
    default:
      launch_cta_two_pass_kernel<scalar_t, weight_t, vec_size_value>(layout, eps, config);
      break;
  }

#undef QKNORM_CTA_REGISTER_CASE
}

template <typename scalar_t, typename weight_t>
void dispatch_cta_vec_size(
    const CTAConfig& config, const Layout<scalar_t, weight_t>& layout, acc_type_t<scalar_t> eps) {
#define QKNORM_CTA_VEC_CASE(VEC_SIZE) dispatch_cta_iters<scalar_t, weight_t, VEC_SIZE>(config, layout, eps)

  switch (config.vec_size) {
    case 16:
      QKNORM_CTA_VEC_CASE(16);
      break;
    case 8:
      QKNORM_CTA_VEC_CASE(8);
      break;
    case 4:
      QKNORM_CTA_VEC_CASE(4);
      break;
    case 2:
      QKNORM_CTA_VEC_CASE(2);
      break;
    default:
      QKNORM_CTA_VEC_CASE(1);
      break;
  }

#undef QKNORM_CTA_VEC_CASE
}

template <typename scalar_t, typename weight_t, typename PostOp>
void dispatch_head_dim(
    const Layout<scalar_t, weight_t>& layout,
    acc_type_t<scalar_t> eps,
    const PostOp& post_op,
    const char* op_name,
    bool allow_cta) {
  TORCH_CHECK(layout.head_dim > 0, op_name, ": head_dim must be positive");
  switch (layout.head_dim) {
    case 64:
      dispatch_warp_vec_size<scalar_t, weight_t, 64, PostOp>(layout, eps, post_op, op_name);
      break;
    case 128:
      dispatch_warp_vec_size<scalar_t, weight_t, 128, PostOp>(layout, eps, post_op, op_name);
      break;
    case 256:
      dispatch_warp_vec_size<scalar_t, weight_t, 256, PostOp>(layout, eps, post_op, op_name);
      break;
    default: {
      TORCH_CHECK(allow_cta, op_name, ": unsupported head dimension: ", layout.head_dim);
      const int64_t num_works = layout.num_tokens * (layout.num_q_heads + layout.num_k_heads);
      const CTAConfig config = cta_config<scalar_t, weight_t>(layout, num_works, layout.head_dim, op_name);
      dispatch_cta_vec_size<scalar_t, weight_t>(config, layout, eps);
      break;
    }
  }
}

}  // namespace at::native::xpu::qknorm
