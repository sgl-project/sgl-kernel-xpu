#include <ATen/ATen.h>
#include <ATen/core/Array.h>

#include "MemoryAccess.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"

namespace at::native::xpu{

constexpr int NUM_REDUCE_STAGES = 16;

static constexpr auto dpcpp_local_fence = sycl::access::fence_space::local_space;
static constexpr auto dpcpp_global_fence = sycl::access::fence_space::global_space;
static constexpr auto dpcpp_global_and_local_fence = sycl::access::fence_space::global_and_local;

inline std::pair<int64_t, int64_t> _check_layer_norm_inputs(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */) {
  const int normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();

  if (input_ndim < normalized_ndim || !input_shape.slice(input_ndim - normalized_ndim).equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

  const int axis = input_ndim - normalized_ndim;
  const int64_t M = c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  const int64_t N = c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend());

  return std::make_pair(M, N);
}

template <typename accscalar_t, typename reduce_op, typename nd_item_id, typename local_shared>
static inline void norm_group_reduce(
    nd_item_id item_id,
    int sub_group_num,
    accscalar_t& mean,
    accscalar_t& rstd,
    const local_shared& local_mean,
    const local_shared& local_rstd,
    reduce_op bin_op) {
  auto sg = item_id.get_sub_group();
#pragma unroll
  for (int i = 1; i < NUM_REDUCE_STAGES; i <<= 1) {
    mean = bin_op(mean, static_cast<accscalar_t>(sycl::shift_group_left(sg, mean, i)));
    rstd = bin_op(rstd, static_cast<accscalar_t>(sycl::shift_group_left(sg, rstd, i)));
  }
  if (sub_group_num == 1) {
    mean = sycl::group_broadcast(sg, mean, 0);
    rstd = sycl::group_broadcast(sg, rstd, 0);
    return;
  }
  uint32_t sg_local_id = sg.get_local_linear_id();
  uint32_t sg_id = sg.get_group_linear_id();

  int idx = sg_id;
  if (sg_local_id == 0) {
    local_mean[sg_id] = mean;
    local_rstd[sg_id] = rstd;
  }
  item_id.barrier(dpcpp_local_fence);

  if (idx == 0) {
    mean = 0;
    rstd = 0;
    if (sg_local_id < sub_group_num) {
      mean = accscalar_t(local_mean[sg_local_id]);
      rstd = accscalar_t(local_rstd[sg_local_id]);
    }
    for (int i = sg_local_id + NUM_REDUCE_STAGES; i < sub_group_num; i += NUM_REDUCE_STAGES) {
      mean = bin_op(mean, static_cast<accscalar_t>(local_mean[i]));
      rstd = bin_op(rstd, static_cast<accscalar_t>(local_rstd[i]));
    }
#pragma unroll
    for (int i = 1; i < NUM_REDUCE_STAGES; i <<= 1) {
      mean = bin_op(mean, static_cast<accscalar_t>(sycl::shift_group_left(sg, mean, i)));
      rstd = bin_op(rstd, static_cast<accscalar_t>(sycl::shift_group_left(sg, rstd, i)));
      if (i >= ((sub_group_num + 1) >> 1)) break;
    }

    if (sg_local_id == 0) {
      local_mean[0] = mean;
      local_rstd[0] = rstd;
    }
  }
  item_id.barrier(dpcpp_local_fence);
  mean = local_mean[0];
  rstd = local_rstd[0];
}

class NormConfig {
 public:
  NormConfig(int Batch, int Plane, int problem_dim, int element_size_bytes)
      : Batch(Batch), Plane(Plane), problem_dim(problem_dim), element_size_bytes(element_size_bytes) {
    semaphores_ptr = nullptr;
    scratchpad_ptr = nullptr;
    sub_group_num_global = 1;

    get_max_vec_size();
    if (problem_dim == 1) {
      get_workgroup_size();
      WGPlane = (Plane + workgroup_num_foreach - 1) / workgroup_num_foreach;
    } else {
      get_workgroup_size_row();
    }
  }

  int Batch;
  int Plane;
  int WGPlane;
  int problem_dim;
  int element_size_bytes;
  int max_vec_size;

  int block_row;
  int workgroup_num;
  int workgroup_num_foreach;
  int workgroup_size;
  int sub_group_num;

  int* semaphores_ptr;
  void* scratchpad_ptr;
  int sub_group_num_global;

  template <typename scalar_t>
  void init_global_reduce(const Tensor& X, Tensor& semaphores, Tensor& scratchpad) {
    if (workgroup_num_foreach > 1) {
      int semaphores_size = workgroup_num;
      semaphores = at::zeros(semaphores_size, X.options().dtype(kInt));
      const auto kAccType = (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16) ? kFloat : X.scalar_type();
      int scratchpad_size = 2 * Batch * workgroup_num_foreach * sizeof(acc_type<scalar_t>);
      scratchpad = at::zeros(scratchpad_size, X.options().dtype(kAccType));
      semaphores_ptr = semaphores.data_ptr<int>();
      scratchpad_ptr = scratchpad.data_ptr();
      sub_group_num_global = (workgroup_num_foreach + NUM_REDUCE_STAGES - 1) / NUM_REDUCE_STAGES;
    }
  }

  void get_max_vec_size() {
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int total_resource = dpcppMaxWorkItemsPerTile(dev_id);

    constexpr int float4_size = sizeof(float) * 4;
    max_vec_size = float4_size / element_size_bytes;
    while ((max_vec_size >> 1) * total_resource >= (Batch * Plane) && (max_vec_size >> 1) >= 1) {
      max_vec_size = max_vec_size >> 1;
    }
  }

  // get resource size for Reduce problem [Batch, Plane]
  // the reduce is performed on Plane dimension
  void get_workgroup_size() {
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int max_workgroup_size = dpcppMaxWorkGroupSize(dev_id);
    if constexpr (NUM_REDUCE_STAGES == 16) {
      // WA for BMG. The actual max work group size on BMG is 512 (64 HW thread
      // * 16 SIMD per SS), which conflicts with 1024 returned from SYCL
      // runtime.
      max_workgroup_size = std::min(max_workgroup_size, 512);
    }
    int total_resource = dpcppMaxWorkItemsPerTile(dev_id);
    workgroup_num = total_resource / max_workgroup_size;
    int max_workgroup_num_foreach = 1;
    workgroup_size = max_workgroup_size;

    // To keep high occupancy, we should activate at least workgroup_num number
    // of WG if Batch is larger than workgroup_num, use only one WG to process
    // Plane elements if Batch is smaller than workgroup_num, use
    // workgroup_num_foreach to process Plan elements
    while (workgroup_num > Batch) {
      workgroup_num = workgroup_num >> 1;
      max_workgroup_num_foreach = max_workgroup_num_foreach << 1;
    }
    workgroup_num_foreach = (Plane + workgroup_size * max_vec_size - 1) / (workgroup_size * max_vec_size);
    workgroup_num_foreach = std::min(workgroup_num_foreach, max_workgroup_num_foreach);
    // Reduce will waste the EU resource, then
    // minimize the workgroup_size and maximize the workgroup_num
    while (workgroup_num << 1 <= Batch && (workgroup_size >> 1) >= NUM_REDUCE_STAGES) {
      workgroup_num = workgroup_num << 1;
      workgroup_size = workgroup_size >> 1;
    }

    // Workgroup_num should larger or equal to Batch
    workgroup_num = std::max(workgroup_num, int(Batch));
    // At least one subgroup for reduce
    sub_group_num = (workgroup_size + NUM_REDUCE_STAGES - 1) / NUM_REDUCE_STAGES;
  }

  void get_workgroup_size_row() {
    // enlarge the occupancy, compute the least workgroup_num
    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int max_workgroup_size = dpcppMaxWorkGroupSize(dev_id);
    int total_resource = dpcppMaxWorkItemsPerTile(dev_id);
    workgroup_num = total_resource / max_workgroup_size;

    int max_block_row = max_workgroup_size / NUM_REDUCE_STAGES;
    block_row = 1;
    while ((block_row << 2) <= Batch && (block_row << 1) <= max_block_row) {
      block_row = block_row << 1;
    }
    workgroup_size = max_workgroup_size / block_row;

    // maximize the vec_size
    size_t problem_size = Plane;
    constexpr int float4_size = sizeof(float) * 4;
    max_vec_size = float4_size / element_size_bytes;
    while ((max_vec_size >> 1) * workgroup_num * workgroup_size >= Plane && (max_vec_size >> 1) >= 1) {
      max_vec_size = max_vec_size >> 1;
    }

    // maximize the workgroup_size, and minimize the block_row
    while ((workgroup_size >> 1) * workgroup_num * max_vec_size > Plane && (workgroup_size >> 1) >= NUM_REDUCE_STAGES) {
      workgroup_size = workgroup_size >> 1;
    }
    while ((workgroup_size << 1) * workgroup_num * max_vec_size <= Plane &&
           (workgroup_size << 1) <= max_workgroup_size) {
      workgroup_size = workgroup_size << 1;
    }
    block_row = max_workgroup_size / workgroup_size;

    workgroup_num = (Plane + workgroup_size * max_vec_size - 1) / (workgroup_size * max_vec_size);
  }
};

template <typename scalar_t, typename mean_t, typename weight_t, bool one_moment = false>
class NormForward {
 public:
  using accscalar_t = acc_type<scalar_t>;
  NormForward() = delete;
  NormForward(
      scalar_t* X_data,
      scalar_t* Y_data,
      mean_t* mean_data,
      mean_t* var_data,
      weight_t* gamma_data,
      weight_t* beta_data,
      accscalar_t eps)
      : X_data(X_data),
        Y_data(Y_data),
        mean_data(mean_data),
        var_data(var_data),
        gamma_data(gamma_data),
        beta_data(beta_data),
        eps(eps) {}

  int get_rowwise_reduce_vec_size(int Plane, int vec_size) {
    vec_size = std::min(
        vec_size, can_vectorize_up_to<scalar_t>(dpcppGetDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(X_data)));

    while (Plane % vec_size != 0) {
      vec_size = vec_size >> 1;
    }
    return vec_size;
  }

  int get_update_vec_size(int Plane, int vec_size) {
    vec_size = get_min_vec_size(vec_size, X_data, Y_data, gamma_data, beta_data);

    while (Plane % vec_size != 0) {
      vec_size = vec_size >> 1;
    }
    return vec_size;
  }

  int get_eltwise_update_vec_size(int vec_size) {
    vec_size = std::min(
        vec_size, can_vectorize_up_to<scalar_t>(dpcppGetDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(X_data)));
    vec_size = std::min(
        vec_size, can_vectorize_up_to<scalar_t>(dpcppGetDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(Y_data)));
    return vec_size;
  }

  template <int vec_size, typename vec_t, typename weight_vec_t, typename index_t, typename nd_item_id>
  void reduce_combine(nd_item_id item_id, const NormConfig& cfg, accscalar_t& sum1, accscalar_t& sum2) const {
    auto group_id = item_id.get_group(0);
    auto group_id_foreach = item_id.get_group(1);
    auto local_id = item_id.get_local_id(2);
    index_t group_offset = group_id * cfg.Plane;

    for (index_t j = local_id * vec_size; j < cfg.WGPlane; j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < cfg.Plane) {
        vec_t value = *(reinterpret_cast<vec_t*>(X_data + group_offset + plane_offset));
        for (int v = 0; v < vec_size; ++v) {
          sum1 += static_cast<accscalar_t>(value[v]);
          sum2 += static_cast<accscalar_t>(value[v]) * static_cast<accscalar_t>(value[v]);
        }
      }
    }
  }

  template <typename nd_item_id>
  void reduce_project(nd_item_id item_id, accscalar_t sum1, accscalar_t sum2, const NormConfig& cfg) const {
    auto group_id = item_id.get_group(0);
    accscalar_t scale = static_cast<accscalar_t>(cfg.Plane);
    sum2 = (sum2 - sum1 * sum1 / scale) / scale;
    sum1 = sum1 / scale;
    mean_data[group_id] = static_cast<mean_t>(sum1);
    var_data[group_id] =
        static_cast<mean_t>(Numerics<accscalar_t>::rsqrt(sum2 < 0 ? 0 : sum2 + static_cast<accscalar_t>(eps)));
  }

 public:
  scalar_t* X_data;
  scalar_t* Y_data;
  mean_t* mean_data;
  mean_t* var_data;
  weight_t* gamma_data;
  weight_t* beta_data;
  accscalar_t eps;
};

bool canUse32BitIndexMath(const at::Tensor& t, int64_t max_elem) {
  int64_t elements = t.numel();

  if (elements == 0) {
    return true;
  }

  if (elements >= max_elem) {
    return false;
  }

  int64_t offset = 0;
  int64_t linearId = elements - 1;

  for (int i = t.dim() - 1; i >= 0; --i) {
    int64_t curDimIndex = linearId % t.size(i);
    int64_t curDimOffset = curDimIndex * t.stride(i);
    offset += curDimOffset;
    linearId /= t.size(i);
  }

  if (offset >= max_elem) {
    return false;
  }

  return true;
}

}
