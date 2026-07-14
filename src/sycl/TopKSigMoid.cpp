#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

namespace TopKSigmoidImpl {

template <typename T>
struct FusedTopkSigmoid {
  static constexpr int sub_group_size = 32;
  static constexpr int max_group_size = 1024;
  static constexpr int malloc_per_item = 8;
  static constexpr float kNegInfinity = -std::numeric_limits<float>::infinity();

  FusedTopkSigmoid(
      float* topk_weights,
      int* topk_ids,
      const T* gating_output,
      const float* correction_bias,
      const bool renormalize,
      const float routed_scaling_factor,
      const int num_fused_shared_experts,
      const int tokens,
      const int experts,
      const int top_k)
      : topk_weights(topk_weights),
        topk_ids(topk_ids),
        gating_output(gating_output),
        correction_bias(correction_bias),
        renormalize(renormalize),
        routed_scaling_factor(routed_scaling_factor),
        num_fused_shared_experts(num_fused_shared_experts),
        tokens(tokens),
        experts(experts),
        top_k(top_k) {}

  static inline sycl::nd_range<3> get_nd_range(const int tokens, const int experts) {
    int calc_per_item = div_up(experts, sub_group_size);
    int group_size = div_up(experts, calc_per_item);
    group_size = group_size < sub_group_size ? sub_group_size : group_size;
    group_size = group_size < max_group_size ? group_size : max_group_size;
    int sub_groups_per_group = div_up(group_size, sub_group_size);
    group_size = sub_groups_per_group * sub_group_size;
    int global_size = div_up(tokens, sub_groups_per_group);
    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(1, 1, global_size);
    return sycl::nd_range<3>(global * local, local);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]]
  void operator()(sycl::nd_item<3> item) const {
    int group_id = item.get_group_linear_id();
    int local_range = item.get_local_range(2);
    int sub_groups_per_group = local_range / sub_group_size;
    int calc_per_item = (experts + sub_group_size - 1) / sub_group_size;

    sycl::sub_group sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();

    int tid = group_id * sub_groups_per_group + sg_id;
    if (tid >= tokens) return;

    // Keep ranking scores in float so bf16/fp16 inputs do not perturb top-k order.
    float local_elems[malloc_per_item];
    int local_idx[malloc_per_item];

    int start_offset = sg_local_id * calc_per_item;
    int local_num = calc_per_item;
    if (start_offset + local_num >= experts) local_num = experts - start_offset;
    if (local_num < 0) local_num = 0;

    for (int e = 0; e < calc_per_item; ++e) {
      local_elems[e] = kNegInfinity;
      local_idx[e] = -1;
    }

    for (int e = 0; e < local_num; ++e) {
      int expert_id = start_offset + e;
      float logit = static_cast<float>(gating_output[tid * experts + expert_id]);

      // sigmoid activation
      float sig = 1.0f / (1.0f + sycl::native::exp(-logit));

      local_idx[e] = expert_id;

      // add bias for ranking purposes only
      if (correction_bias != nullptr) sig += correction_bias[expert_id];

      local_elems[e] = sig;
    }

    // Top-K selection
    float topk_weights_local[malloc_per_item];
    int topk_ids_local[malloc_per_item];

    const int routed_topk = top_k - num_fused_shared_experts;
    float row_sum_for_renormalize = 0.0f;

    for (int k = 0; k < routed_topk; ++k) {
      float k_max = kNegInfinity;
      int k_max_idx = -1;
      int remove_ix = -1;

      for (int e = 0; e < calc_per_item; ++e) {
        float my_val = local_elems[e];
        int my_idx = local_idx[e];

        // butterfly argmax across sub-group — same as softmax
        for (int offset = sub_group_size / 2; offset > 0; offset /= 2) {
          float other_val = sycl::permute_group_by_xor(sg, my_val, offset);
          int other_idx = sycl::permute_group_by_xor(sg, my_idx, offset);
          if (other_val > my_val || (other_val == my_val && other_idx < my_idx)) {
            my_val = other_val;
            my_idx = other_idx;
          }
        }

        if (my_val > k_max || (my_val == k_max && my_idx < k_max_idx)) {
          k_max = my_val;
          k_max_idx = my_idx;
          remove_ix = (k_max_idx == local_idx[e]) ? e : -1;
        }
      }
      topk_weights_local[k] = k_max;
      topk_ids_local[k] = k_max_idx;

      // owner lane clears winning slot to prevent re-selection
      if (remove_ix != -1) {
        local_elems[remove_ix] = kNegInfinity;
        local_idx[remove_ix] = -1;
      }
    }

    // Reuse the selected ranking scores. When bias participates in ranking,
    // subtract it back out so outputs stay as raw sigmoid weights.
    if (correction_bias != nullptr) {
      for (int i = 0; i < routed_topk; ++i) {
        int id = topk_ids_local[i];
        topk_weights_local[i] -= correction_bias[id];
      }
    }

    for (int i = 0; i < routed_topk; ++i) {
      row_sum_for_renormalize += topk_weights_local[i];
    }

    // Write output
    if (sg_local_id == 0) {
      int offset = tid * top_k;
      for (int i = 0; i < routed_topk; ++i) {
        topk_weights[offset + i] = topk_weights_local[i];
        topk_ids[offset + i] = (topk_ids_local[i] >= 0) ? topk_ids_local[i] : 0;
      }

      if (num_fused_shared_experts > 0) {
        const int shared_idx = offset + routed_topk;
        topk_ids[shared_idx] = experts;
        if (renormalize) {
          topk_weights[shared_idx] = 1.0f;
        } else {
          topk_weights[shared_idx] = row_sum_for_renormalize / routed_scaling_factor;
        }
      }

      if (renormalize) {
        const float scale = routed_scaling_factor / (row_sum_for_renormalize + 1e-20f);
        for (int i = 0; i < routed_topk; ++i) {
          topk_weights[offset + i] *= scale;
        }
      }
    }
  }

  float* topk_weights;
  int* topk_ids;
  const T* gating_output;
  const float* correction_bias;
  const bool renormalize;
  const float routed_scaling_factor;
  const int num_fused_shared_experts;
  const int tokens;
  const int experts;
  const int top_k;
};

/// Aligned array type
template <
    typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray {
  T data[N];
};

static constexpr int sub_group_size = 32;
static constexpr int WARP_SIZE = sub_group_size;
static constexpr float NEG_INIFINITY = -std::numeric_limits<float>::infinity();
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*
  This gating kernel is adapted from
  https://github.com/sgl-project/sglang/blob/v0.5.12/sgl-kernel/csrc/moe/moe_topk_sigmoid_kernels.cu

  A Top-K gating sigmoid written to exploit when the number of experts in the MoE layers
  are a small power of 2. This allows us to cleanly share the rows among the threads in
  a single warp and eliminate communication between warps (so no need to use shared mem).

  It fuses the sigmoid, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small power of 2.
  2) This implementation assumes k is small, but will work for any k.
*/

template <typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
struct TopkGatingSigmoid {
  [[sycl::reqd_sub_group_size(sub_group_size)]] void operator()(sycl::nd_item<2> item) const {
    // We begin by enforcing compile time assertions and setting up compile time constants.
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    // Restrictions based on previous section.
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
    // This, each block processes a chunk of rows. We start by computing the start row for each block.
    auto sg = item.get_sub_group();
    auto local_id_x = item.get_local_id(1);
    auto local_id_y = item.get_local_id(0);
    auto group_id_x = item.get_group(1);

    const int cta_base_row = group_id_x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per warp.
    const int warp_base_row = cta_base_row + local_id_y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = local_id_x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows) {
      return;
    }

    const bool row_is_active = finished ? !finished[thread_row] : true;

    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.
    const T* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the first column to start loads.
    const int thread_group_idx = local_id_x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param. In theory,
    // this can support all powers of 2 up to 16.
    // NOTE(woosuk): The original implementation uses CUTLASS aligned array here.
    // We defined our own aligned array and use it here to avoid the dependency on CUTLASS.
    using AccessType = AlignedArray<T, ELTS_PER_LDG>;

    // Finally, we pull in the data from global mem
    T row_chunk_temp[VPT];
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk_temp);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    // Note(Byron): interleaved loads to achieve better memory coalescing
    // | thread[0] | thread[1] | thread[2] | thread[3] | thread[0] | thread[1] | thread[2] | thread[3] | ...
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    float row_chunk[VPT];
#pragma unroll
    // Note(Byron): upcast logits to float32
    for (int ii = 0; ii < VPT; ++ii) {
      float val = static_cast<float>(row_chunk_temp[ii]);
      val = 1.0f / (1.0f + sycl::native::exp(-val));
      if (correction_bias != nullptr) {
        /*
        LDG is interleaved
        |thread0 LDG| |thread1 LDG| |thread0 LDG| |thread1 LDG|
        |--------- group0 --------| |----------group1 --------|
                                      ^ local2
        */
        const int group_id = ii / ELTS_PER_LDG;
        const int local_id = ii % ELTS_PER_LDG;
        const int expert_idx = first_elt_read_by_thread + group_id * THREADS_PER_ROW * ELTS_PER_LDG + local_id;
        val = val + correction_bias[expert_idx];
      }
      row_chunk[ii] = val;
    }

    // Now, row_chunk contains the sigmoid of the row chunk. Now, I want to find the topk elements in each row, along
    // with the max index.
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    float row_sum_for_renormalize = 0;

    const int topk = k + num_fused_shared_experts;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      // First, each thread does the local argmax
      float max_val = row_chunk[0];
      int expert = start_col;
#pragma unroll
      for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          float val = row_chunk[ldg * ELTS_PER_LDG + ii];

          // No check on the experts here since columns with the smallest index are processed first and only
          // updated if > (not >=)
          if (val > max_val) {
            max_val = val;
            expert = col + ii;
          }
        }
      }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
// This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
// then blank out their max with -inf and the warp can run more iterations...
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max = sycl::permute_group_by_xor(sg, max_val, mask);
        int other_expert = sycl::permute_group_by_xor(sg, expert, mask);

        // We want lower indices to "win" in every thread so we break ties this way
        if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
          max_val = other_max;
          expert = other_expert;
        }
      }

      // Write the max for this k iteration to global memory.
      if (thread_group_idx == 0) {
        // Add a guard to ignore experts not included by this node
        const bool node_uses_expert = expert >= start_expert && expert < end_expert;
        const bool should_process_row = row_is_active && node_uses_expert;

        // The lead thread from each sub-group will write out the final results to global memory. (This will be a
        // single) thread per row of the input/output matrices.
        const int idx = k * thread_row + k_idx;
        if (correction_bias != nullptr) {
          max_val -= correction_bias[expert];
        }
        output[idx] = max_val;
        indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
        row_sum_for_renormalize += max_val;
      }

      // Finally, we clear the value in the thread with the current max if there is another iteration to run.
      if (k_idx + 1 < k) {
        const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
        const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

        // Only the thread in the group which produced the max will reset the "winning" value to -inf.
        if (thread_group_idx == thread_to_clear_in_group) {
          const int offset_for_expert = expert % ELTS_PER_LDG;
          // Safe to set to any negative value since row_chunk values must be between 0 and 1.
          row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
        }
      }
    }

    if (num_fused_shared_experts > 0 && thread_group_idx == 0) {
      const int last_idx = topk * thread_row + k;
      if (renormalize) {
        output[last_idx] = 1.0f;
      } else {
        output[last_idx] = row_sum_for_renormalize / routed_scaling_factor;
      }
      indices[last_idx] = NUM_EXPERTS;
    }

    // Fuse renormalization of topk_weights into this kernel
    if (renormalize && thread_group_idx == 0) {
      float row_sum_for_renormalize_inv = routed_scaling_factor / (row_sum_for_renormalize + 1e-20f);
#pragma unroll
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        const int idx = topk * thread_row + k_idx;
        output[idx] = output[idx] * row_sum_for_renormalize_inv;
      }
    }
  }

  const T* input;
  const bool* finished;
  float* output;
  const int num_rows;
  int* indices;
  const float* correction_bias;
  const int k;
  const int start_expert;
  const int end_expert;
  const bool renormalize;
  const float routed_scaling_factor;
  const int num_fused_shared_experts;
};

namespace detail {
// Constructs some constants needed to partition the work across threads at compile time.
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");
  static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
}  // namespace detail

template <typename T, int NUM_EXPERTS, int WARPS_PER_TB>
void launch_topk_gating_sigmoid(
    const T* input,
    const bool* finished,
    float* output,
    int* indices,
    const float* correction_bias,
    const int num_rows,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize,
    const float routed_scaling_factor,
    const int num_fused_shared_experts,
    sycl::queue& queue) {
  static constexpr std::size_t MAX_BYTES_PER_LDG = 16;

  static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(T) * NUM_EXPERTS);
  using Constants = detail::TopkConstants<T, NUM_EXPERTS, BYTES_PER_LDG>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  sycl::range<2> grid(1, num_blocks);
  sycl::range<2> block(WARPS_PER_TB, WARP_SIZE);

  using Kernel = TopkGatingSigmoid<T, VPT, NUM_EXPERTS, WARPS_PER_TB, BYTES_PER_LDG>;

  Kernel task{
      input,
      finished,
      output,
      num_rows,
      indices,
      correction_bias,
      k,
      start_expert,
      end_expert,
      renormalize,
      routed_scaling_factor,
      num_fused_shared_experts};

  sycl_kernel_submit(grid * block, block, queue, task);
}

template <typename T>
void launch_fused_topk_sigmoid(
    sycl::queue& queue,
    const T* gating_output,
    float* topk_weights,
    int* topk_indices,
    const float* correction_bias,
    const bool renormalize,
    const float routed_scaling_factor,
    const int num_fused_shared_experts,
    const int top_k,
    const int num_tokens,
    const int num_experts) {
  using Kernel = FusedTopkSigmoid<T>;
  auto range = Kernel::get_nd_range(num_tokens, num_experts);
  Kernel task(
      topk_weights,
      topk_indices,
      gating_output,
      correction_bias,
      renormalize,
      routed_scaling_factor,
      num_fused_shared_experts,
      num_tokens,
      num_experts,
      top_k);
  sycl_kernel_submit(range.get_global_range(), range.get_local_range(), queue, task);
}

template <typename T>
void fused_topk_sigmoid(
    const T* gating_output,
    float* topk_weights,
    int* topk_indices,
    const float* correction_bias,
    const bool renormalize,
    const float routed_scaling_factor,
    const int num_fused_shared_experts,
    const int num_tokens,
    const int num_experts,
    const int topk) {
  auto queue = at::xpu::getCurrentXPUStream().queue();
  constexpr int WARPS_PER_TB = 4;

#define LAUNCH_GATING_SIGMOID(TYPE, NUM_EXPERTS, WARPS_PER_TB) \
  launch_topk_gating_sigmoid<TYPE, NUM_EXPERTS, WARPS_PER_TB>( \
      gating_output,                                           \
      nullptr,                                                 \
      topk_weights,                                            \
      topk_indices,                                            \
      correction_bias,                                         \
      num_tokens,                                              \
        topk - num_fused_shared_experts,                         \
      0,                                                       \
      num_experts,                                             \
      renormalize,                                             \
        routed_scaling_factor,                                   \
        num_fused_shared_experts,                                \
      queue);

  switch (num_experts) {
    case 1:
      LAUNCH_GATING_SIGMOID(T, 1, WARPS_PER_TB);
      break;
    case 2:
      LAUNCH_GATING_SIGMOID(T, 2, WARPS_PER_TB);
      break;
    case 4:
      LAUNCH_GATING_SIGMOID(T, 4, WARPS_PER_TB);
      break;
    case 8:
      LAUNCH_GATING_SIGMOID(T, 8, WARPS_PER_TB);
      break;
    case 16:
      LAUNCH_GATING_SIGMOID(T, 16, WARPS_PER_TB);
      break;
    case 32:
      LAUNCH_GATING_SIGMOID(T, 32, WARPS_PER_TB);
      break;
    case 64:
      LAUNCH_GATING_SIGMOID(T, 64, WARPS_PER_TB);
      break;
    case 128:
      LAUNCH_GATING_SIGMOID(T, 128, WARPS_PER_TB);
      break;
    case 256:
      LAUNCH_GATING_SIGMOID(T, 256, WARPS_PER_TB);
      break;
    default:
      launch_fused_topk_sigmoid(
          queue,
          gating_output,
          topk_weights,
          topk_indices,
          correction_bias,
          renormalize,
          routed_scaling_factor,
          num_fused_shared_experts,
          topk,
          num_tokens,
          num_experts);
  }
#undef LAUNCH_GATING_SIGMOID
}
};  // namespace TopKSigmoidImpl

/**
 * @brief Perform topk after sigmoid on gating_output.
 * @param topk_weights The topk_weights tensor of shape [n_tokens, n_topk].
 * @param topk_indices The topk_indices tensor of shape [n_tokens, n_topk].
 * @param gating_output The gating output tensor of shape [n_tokens, n_experts].
 * @param renormalize The renormalize bool whether the topk_weights needs to be renormalized.
 * @param correction_bias The correction_bias tensor of shape [num_experts].
 * @return void.
 */
void topk_sigmoid(
    at::Tensor& topk_weights,
    at::Tensor& topk_indices,
    at::Tensor& gating_output,
    bool renormalize,
  const c10::optional<at::Tensor>& correction_bias,
  double routed_scaling_factor,
  int64_t num_fused_shared_experts) {
  auto shape = gating_output.sizes().vec();
  TORCH_CHECK(shape.size() == 2, "gating_output must be 2D");
  int64_t n_tokens = shape[0];
  int64_t n_experts = shape[1];
  TORCH_CHECK(n_experts <= 256, "n_experts only support up to 256, got ", n_experts);
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "topk_weights should be Float");
  TORCH_CHECK(topk_indices.scalar_type() == at::kInt, "topk_indices should be Int");

  int64_t n_topk = topk_weights.size(1);
  // The max topk value is 8, which is constrained by 'malloc_per_item'.
  auto max_topk = n_experts < 8 ? n_experts : 8;
  TORCH_CHECK(0 < n_topk && n_topk <= max_topk, "topk must be less than or equal to num_experts and 8");
  TORCH_CHECK(num_fused_shared_experts <= 1, "num_fused_shared_experts must be <= 1");

  const float* bias_ptr = nullptr;
  if (correction_bias.has_value()) {
    const auto& bias = correction_bias.value();
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == n_experts, "correction_bias must be 1D [num_experts]");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "correction_bias must be float32");
    bias_ptr = bias.data_ptr<float>();
  }

  // Cover Float in addition to the reduced float types (Half, BFloat16): some
  // models (e.g. Nemotron-3-Nano MoE) emit fp32 router gating logits. The kernel
  // already upcasts each element to float internally, so the fp32 path is a
  // no-op cast and adds no numerical change.
  DISPATCH_FLOAT_TYPES(gating_output.scalar_type(), "fused_topk_sigmoid_kernel", [&] {
    TopKSigmoidImpl::fused_topk_sigmoid<scalar_t>(
        gating_output.data_ptr<scalar_t>(),
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        bias_ptr,
        renormalize,
        static_cast<float>(routed_scaling_factor),
        static_cast<int>(num_fused_shared_experts),
        n_tokens,
        n_experts,
        n_topk);
  });
}
}  // namespace at::native::xpu
