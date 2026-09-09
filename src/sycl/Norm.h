#include <ATen/ATen.h>
#include <ATen/core/Array.h>

#include <type_traits>

#include "MemoryAccess.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"

namespace at::native::xpu {

constexpr int NUM_REDUCE_STAGES = 16;

inline std::tuple<int64_t, int64_t> _check_layer_norm_inputs(
    const torch::Tensor& input,
    IntArrayRef normalized_shape,
    std::optional<torch::Tensor>& weight /* optional */,
    std::optional<torch::Tensor>& bias /* optional */) {
  CHECK_LAST_DIM_CONTIGUOUS(input);
  TORCH_CHECK(input.dim() == 2 || input.dim() == 3 || input.dim() == 4, "input must be a 2D, 3D, or 4D tensor");
#define TENSOR_CHECK(T)                          \
  if (T.has_value()) {                           \
    CHECK_LAST_DIM_CONTIGUOUS(T.value());        \
    auto device = input.device();                \
    CHECK_EQ(T.value().device(), device);        \
    CHECK_DIM(1, T.value());                     \
    CHECK_EQ(input.size(-1), T.value().size(0)); \
  }

  TENSOR_CHECK(weight)
  TENSOR_CHECK(bias)

  // Note: for 3D inputs, the leading dimensions do not need to be flattenable
  // into a single batch dimension.  The kernel indexes rows using both an
  // outer stride and an inner (head-like) stride when necessary, so sliced
  // views of a packed buffer (e.g. per-head slices of a QKV tensor) are
  // supported natively without requiring a contiguous copy.

  int64_t hidden_size = input.size(-1);
  int64_t batch_size = input.numel() / hidden_size;

  return std::make_tuple(batch_size, hidden_size);
}

class NormConfig {
 public:
  NormConfig(
      int Batch,
      int Plane,
      int problem_dim,
      int element_size_bytes,
      int input_batch_stride,
      int output_batch_stride,
      int input_inner0_size = 1,
      int input_inner0_stride = 0,
      int output_inner0_size = 1,
      int output_inner0_stride = 0,
      int input_inner1_size = 1,
      int input_inner1_stride = 0,
      int output_inner1_size = 1,
      int output_inner1_stride = 0)
      : Batch(Batch),
        Plane(Plane),
        problem_dim(problem_dim),
        element_size_bytes(element_size_bytes),
        input_batch_stride(input_batch_stride),
        output_batch_stride(output_batch_stride),
        input_inner0_size(input_inner0_size),
        input_inner0_stride(input_inner0_stride),
        output_inner0_size(output_inner0_size),
        output_inner0_stride(output_inner0_stride),
        input_inner1_size(input_inner1_size),
        input_inner1_stride(input_inner1_stride),
        output_inner1_size(output_inner1_size),
        output_inner1_stride(output_inner1_stride) {
    semaphores_ptr = nullptr;
    scratchpad_ptr = nullptr;
    sub_group_num_global = 1;
    update_vec_size = 1;

    get_max_vec_size();
    if (problem_dim == 1) {
      get_workgroup_size();
      WGPlane = (Plane + workgroup_num_foreach - 1) / workgroup_num_foreach;
    } else {
      get_workgroup_size_row();
    }
  }

  template <
      typename GetUpdateVecSizeFn,
      typename = std::enable_if_t<std::is_invocable_r_v<int, GetUpdateVecSizeFn, int, int>>>
  NormConfig(
      int Batch,
      int Plane,
      int problem_dim,
      int element_size_bytes,
      int input_batch_stride,
      int output_batch_stride,
      GetUpdateVecSizeFn get_update_vec_size,
      int input_inner0_size = 1,
      int input_inner0_stride = 0,
      int output_inner0_size = 1,
      int output_inner0_stride = 0,
      int input_inner1_size = 1,
      int input_inner1_stride = 0,
      int output_inner1_size = 1,
      int output_inner1_stride = 0)
      : NormConfig(
            Batch,
            Plane,
            problem_dim,
            element_size_bytes,
            input_batch_stride,
            output_batch_stride,
            input_inner0_size,
            input_inner0_stride,
            output_inner0_size,
            output_inner0_stride,
            input_inner1_size,
            input_inner1_stride,
            output_inner1_size,
            output_inner1_stride) {
    workgroup_num = Batch;
    workgroup_num_foreach = 1;
    WGPlane = Plane;
    get_workgroup_size_single_wg_per_row(get_update_vec_size);
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
  int update_vec_size;

  int input_batch_stride;
  int output_batch_stride;
  // Inner-stride support for non-flattenable 3D/4D tensors.  A tensor viewed
  // as (outer, inner0, inner1, plane) has its flattened row index r split as
  //   outer  = r / (inner0_size * inner1_size)
  //   inner0 = (r % (inner0_size * inner1_size)) / inner1_size
  //   inner1 = r % inner1_size
  // For 2D, flattenable 3D, or 4D tensors whose leading dim folds away,
  // inner1_size is 1 so the inner1 term collapses to zero (and inner0_size is
  // 1 too when fully flattenable).  inner1 is only non-trivial for 4D tensors
  // whose leading dim is independent of the rest (e.g. a batched QKV slice).
  int input_inner0_size;
  int input_inner0_stride;
  int output_inner0_size;
  int output_inner0_stride;
  int input_inner1_size;
  int input_inner1_stride;
  int output_inner1_size;
  int output_inner1_stride;
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

  int get_stride_aligned_vec_size(int vec_size) const {
    while (vec_size > 1 && (input_batch_stride % vec_size != 0 || output_batch_stride % vec_size != 0 ||
                            input_inner0_stride % vec_size != 0 || output_inner0_stride % vec_size != 0 ||
                            input_inner1_stride % vec_size != 0 || output_inner1_stride % vec_size != 0)) {
      vec_size = vec_size >> 1;
    }
    return vec_size;
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

  // Configure workgroup sizing where each workgroup processes one full row
  // (workgroup_num = Batch, workgroup_num_foreach = 1).
  // Strategy:
  //   - Small Batch: use max WG size for highest per-row parallelism (low iters).
  //   - Large Batch: reduce WG size to improve WG occupancy, subject
  //     to keeping iters within max_cached_iters to avoid register spills.
  //   - If reduce size is too large (iters > max_cached_iters at max WG), the
  //     caller should fallback to a non-cached (reload) implementation.
  //   - The vec_size policy is provided by callback to keep per-kernel
  //     alignment rules close to the corresponding forward functor.
  template <typename GetUpdateVecSizeFn>
  void get_workgroup_size_single_wg_per_row(GetUpdateVecSizeFn get_update_vec_size, int max_cached_iters = 8) {
    constexpr int float4_size = sizeof(float) * 4;
    max_vec_size = float4_size / element_size_bytes;
    update_vec_size = get_update_vec_size(WGPlane, max_vec_size);
    update_vec_size = get_stride_aligned_vec_size(update_vec_size);
    while (update_vec_size > 1 && (Plane / update_vec_size) < NUM_REDUCE_STAGES) {
      update_vec_size = update_vec_size >> 1;
    }

    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int max_wg_size = static_cast<int>(dpcppMaxWorkGroupSize(dev_id));
    if constexpr (NUM_REDUCE_STAGES == 16) {
      // See the note of max_workgroup_size in get_workgroup_size
      max_wg_size = std::min(max_wg_size, 512);
    }
    int total_resource = static_cast<int>(dpcppMaxWorkItemsPerTile(dev_id)) /
                         static_cast<int>(dpcppMaxSubGroupSize(dev_id)) * NUM_REDUCE_STAGES;

    // Start with the smallest WG that covers all vector lanes, capped at max.
    int plane_vecs = (Plane + update_vec_size - 1) / update_vec_size;
    workgroup_size = (plane_vecs + NUM_REDUCE_STAGES - 1) / NUM_REDUCE_STAGES * NUM_REDUCE_STAGES;
    workgroup_size = std::min(workgroup_size, max_wg_size);

    int iters = (WGPlane + workgroup_size * update_vec_size - 1) / (workgroup_size * update_vec_size);

    // Reduce WG size to improve occupancy: only when the device cannot concurrently
    // hold half the batch at the current WG size. Floor at 4 subgroups (64
    // work-items) to maintain adequate per-row reduction throughput — below
    // this each WG has too few SIMD threads to utilize XVEs effectively.
    while ((iters << 1) <= max_cached_iters && (total_resource / workgroup_size) < (Batch >> 1) &&
           (workgroup_size >> 1) >= NUM_REDUCE_STAGES * 4) {
      workgroup_size = workgroup_size >> 1;
      iters = (WGPlane + workgroup_size * update_vec_size - 1) / (workgroup_size * update_vec_size);
    }

    workgroup_size = std::max(workgroup_size, NUM_REDUCE_STAGES);
    sub_group_num = workgroup_size / NUM_REDUCE_STAGES;
  }
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

}  // namespace at::native::xpu
