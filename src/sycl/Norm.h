#include <ATen/ATen.h>
#include <ATen/core/Array.h>

#include <type_traits>

#include "MemoryAccess.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"

namespace at::native::xpu {

constexpr int NUM_REDUCE_STAGES = 16;
// Register-cached rows keep at most this many vectors per work-item; beyond it they spill.
constexpr int kMaxCachedIters = 8;

// Workgroup cap for a one-workgroup-per-row norm. A bf16 row of 8192 needs 1024 lanes, so the
// default cap forces two passes. A 1024-lane workgroup at sub-group size 16 needs 64 hardware
// threads, a whole Xe-core on Xe2, so one wide pass wins only while every row fits in a single wave;
// past that it adds a wave. Measured on Arc Pro B60 (20 Xe-cores): -26% device time at 1-20 rows.
constexpr int kDefaultRowWorkgroupCap = 512;
constexpr int kWideRowWorkgroupCap = 1024;

// How many kWideRowWorkgroupCap-lane workgroups the device holds at once; 20 on Arc Pro B60.
inline int wide_rows_per_wave(DeviceId dev_id) {
  const int64_t threads_per_core = dpcppGpuEuCountPerSubslice(dev_id) * dpcppGpuHWThreadsPerEU(dev_id);
  const int64_t threads_per_workgroup = kWideRowWorkgroupCap / NUM_REDUCE_STAGES;
  return static_cast<int>(dpcppGpuSubsliceCount(dev_id) * (threads_per_core / threads_per_workgroup));
}

inline int max_workgroup_for_row(int device_max_wg, int plane_vecs, int batch, int rows_per_wave) {
  const bool wide_removes_a_pass = plane_vecs > kDefaultRowWorkgroupCap;
  const bool fits_one_wave = batch <= rows_per_wave;
  return std::min(
      device_max_wg, (wide_removes_a_pass && fits_one_wave) ? kWideRowWorkgroupCap : kDefaultRowWorkgroupCap);
}

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

  // Launch config for contiguous inputs with one workgroup per row. Produces the same geometry as
  // get_workgroup_size_single_wg_per_row, without its per-call device-property queries.
  struct FastRowTag {};
  NormConfig(FastRowTag, int Batch_, int Plane_, int element_size_bytes_, int vec, int batch_stride)
      : Batch(Batch_),
        Plane(Plane_),
        WGPlane(Plane_),
        problem_dim(1),
        element_size_bytes(element_size_bytes_),
        max_vec_size(vec),
        block_row(0),
        workgroup_num(Batch_),
        workgroup_num_foreach(1),
        update_vec_size(vec),
        input_batch_stride(batch_stride),
        output_batch_stride(batch_stride),
        input_inner0_size(1),
        input_inner0_stride(0),
        output_inner0_size(1),
        output_inner0_stride(0),
        input_inner1_size(1),
        input_inner1_stride(0),
        output_inner1_size(1),
        output_inner1_stride(0),
        semaphores_ptr(nullptr),
        scratchpad_ptr(nullptr),
        sub_group_num_global(1) {
    // Cache the device properties per device; the cap depends on row width and batch, so it is per call.
    // Keyed on the current device because that is whose queue the kernel is submitted to.
    static thread_local DeviceId cached_device = -1;
    static thread_local int cached_device_max_wg = 0;
    static thread_local int cached_total_resource = 0;
    static thread_local int cached_wide_rows_per_wave = 0;
    const DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue();
    if (dev_id != cached_device) {
      cached_device = dev_id;
      cached_device_max_wg = static_cast<int>(dpcppMaxWorkGroupSize(dev_id));
      cached_wide_rows_per_wave = wide_rows_per_wave(dev_id);
      cached_total_resource = static_cast<int>(dpcppMaxWorkItemsPerTile(dev_id)) /
                              static_cast<int>(dpcppMaxSubGroupSize(dev_id)) * NUM_REDUCE_STAGES;
    }

    const int plane_vecs = (Plane_ + vec - 1) / vec;
    const int max_wg = max_workgroup_for_row(cached_device_max_wg, plane_vecs, Batch_, cached_wide_rows_per_wave);
    workgroup_size = (plane_vecs + NUM_REDUCE_STAGES - 1) / NUM_REDUCE_STAGES * NUM_REDUCE_STAGES;
    workgroup_size = std::min(workgroup_size, max_wg);

    workgroup_size =
        occupancy_limited_workgroup(workgroup_size, WGPlane, vec, Batch_, cached_total_resource, kMaxCachedIters);
    sub_group_num = workgroup_size / NUM_REDUCE_STAGES;
  }

  // Reduce WG size to improve occupancy: only when the device cannot concurrently
  // hold half the batch at the current WG size. Floor at 4 subgroups (64
  // work-items) to maintain adequate per-row reduction throughput — below
  // this each WG has too few SIMD threads to utilize XVEs effectively.
  static int occupancy_limited_workgroup(
      int workgroup_size, int wg_plane, int vec_size, int batch, int total_resource, int max_cached_iters) {
    int iters = (wg_plane + workgroup_size * vec_size - 1) / (workgroup_size * vec_size);
    while ((iters << 1) <= max_cached_iters && (total_resource / workgroup_size) < (batch >> 1) &&
           (workgroup_size >> 1) >= NUM_REDUCE_STAGES * 4) {
      workgroup_size = workgroup_size >> 1;
      iters = (wg_plane + workgroup_size * vec_size - 1) / (workgroup_size * vec_size);
    }
    return std::max(workgroup_size, NUM_REDUCE_STAGES);
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
  void
  get_workgroup_size_single_wg_per_row(GetUpdateVecSizeFn get_update_vec_size, int max_cached_iters = kMaxCachedIters) {
    constexpr int float4_size = sizeof(float) * 4;
    max_vec_size = float4_size / element_size_bytes;
    update_vec_size = get_update_vec_size(WGPlane, max_vec_size);
    update_vec_size = get_stride_aligned_vec_size(update_vec_size);
    while (update_vec_size > 1 && (Plane / update_vec_size) < NUM_REDUCE_STAGES) {
      update_vec_size = update_vec_size >> 1;
    }

    auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
    int total_resource = static_cast<int>(dpcppMaxWorkItemsPerTile(dev_id)) /
                         static_cast<int>(dpcppMaxSubGroupSize(dev_id)) * NUM_REDUCE_STAGES;

    // Start with the smallest WG that covers all vector lanes, capped at max. The cap depends on the
    // row width and the batch, so plane_vecs has to be known first -- see max_workgroup_for_row.
    int plane_vecs = (Plane + update_vec_size - 1) / update_vec_size;
    const int max_wg_size = max_workgroup_for_row(
        static_cast<int>(dpcppMaxWorkGroupSize(dev_id)), plane_vecs, Batch, wide_rows_per_wave(dev_id));
    workgroup_size = (plane_vecs + NUM_REDUCE_STAGES - 1) / NUM_REDUCE_STAGES * NUM_REDUCE_STAGES;
    workgroup_size = std::min(workgroup_size, max_wg_size);

    workgroup_size =
        occupancy_limited_workgroup(workgroup_size, WGPlane, update_vec_size, Batch, total_resource, max_cached_iters);
    sub_group_num = workgroup_size / NUM_REDUCE_STAGES;
  }
};

// Launch params for the contiguous fast path. Every inner size/stride is a compile-time constant,
// so compute_row_offset folds its integer div/mod (Xe has no hardware divide) to row * batch_stride.
struct NormFastParams {
  int Plane;
  int workgroup_size;
  int input_batch_stride;
  int output_batch_stride;

  static constexpr int input_inner0_size = 1;
  static constexpr int input_inner0_stride = 0;
  static constexpr int output_inner0_size = 1;
  static constexpr int output_inner0_stride = 0;
  static constexpr int input_inner1_size = 1;
  static constexpr int input_inner1_stride = 0;
  static constexpr int output_inner1_size = 1;
  static constexpr int output_inner1_stride = 0;
};
static_assert(sizeof(NormFastParams) == 4 * sizeof(int), "only the four live ints should be stored");

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
