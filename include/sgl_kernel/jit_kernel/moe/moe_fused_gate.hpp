/**
 * MoE Fused Gate SYCL JIT Kernel for SGLang XPU
 *
 * Hierarchical 2-layer top-k expert selection (DeepSeek-V3 grouped-topk style):
 *   1. sigmoid(input) + bias                       -> per-expert score
 *   2. select top `topk_group` expert groups using the top-2 score sum per group
 *   3. select `topk` experts within the surviving groups
 *   4. optionally append fused shared experts and renormalize / rescale weights
 *
 * Input  : [num_rows, num_experts]        (gate logits, native dtype)
 * Bias   : [num_experts]                   (native dtype)
 * Output : [num_rows, topk]                (float32 weights)
 * Indices: [num_rows, topk]                (int32 expert ids)
 *
 * This is a self-contained JIT port of the AOT *dynamic* kernel in
 * src/sycl/MoE_fused_gate.cpp (moe_fused_gate_kernel_dynamic /
 * KernelParamsDynamic). VPT / NUM_EXPERTS / NUM_EXPERT_GROUPS are runtime
 * values, so a single compiled .so per dtype serves any config satisfying
 *   - num_experts is a power of 2
 *   - num_experts % num_expert_group == 0  (=> num_expert_group is a power of 2)
 *   - VPT = num_experts / num_expert_group <= MAX_VPT (32)
 *
 * Kept numerically identical to the AOT kernel:
 *   - one logical sub-group (size = num_expert_group) processes one row
 *   - ROWS_PER_WG rows per work-group, sub-group size pinned to MAX_VPT (32)
 *   - argmax reductions via ::sycl::permute_group_by_xor within the sub-group
 *   - sigmoid accumulated in fp32
 */

#pragma once

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl_kernel {

// Grid constants (must match AOT MoE_fused_gate.cpp).
static constexpr int kMoeGateRowsPerWg = 8;  // rows per work-group
static constexpr int kMoeGateMaxVpt = 32;    // pinned sub-group size / max values-per-thread

template <typename T>
constexpr T moe_gate_type_max() {
  return std::numeric_limits<T>::max();
}

// ---------------------------------------------------------------------------
// Kernel functor. Runtime-parameterized port of moe_fused_gate_impl using the
// KernelParamsDynamic path from the AOT source (VPT / num_experts /
// num_expert_groups are runtime members, not compile-time constants).
// ---------------------------------------------------------------------------
template <typename T>
class MoeFusedGateKernel {
 public:
  static constexpr T TYPE_MAX_V = moe_gate_type_max<T>();

  MoeFusedGateKernel(
      const T* input,
      const T* bias,
      float* output,
      int32_t* indices,
      int32_t num_rows,
      int32_t vpt,
      int32_t num_experts,
      int32_t num_expert_groups,
      int32_t topk_group,
      int32_t topk,
      int32_t num_fused_shared_experts,
      float routed_scaling_factor,
      bool apply_routed_scaling_factor_on_output)
      : input_(input),
        bias_(bias),
        output_(output),
        indices_(indices),
        num_rows_(num_rows),
        vpt_(vpt),
        num_experts_(num_experts),
        num_expert_groups_(num_expert_groups),
        topk_group_(topk_group),
        topk_(topk),
        num_fused_shared_experts_(num_fused_shared_experts),
        routed_scaling_factor_(routed_scaling_factor),
        apply_routed_scaling_factor_on_output_(apply_routed_scaling_factor_on_output) {}

  [[sycl::reqd_sub_group_size(kMoeGateMaxVpt)]] void operator()(::sycl::nd_item<3> item) const {
    if (item.get_global_linear_id() / kMoeGateMaxVpt >= static_cast<size_t>(num_rows_)) return;

    const int vpt = vpt_;
    const int num_experts = num_experts_;
    const int num_expert_groups = num_expert_groups_;

    const int32_t topk_excluding_share_expert_fusion = topk_ - num_fused_shared_experts_;

    auto sg = item.get_sub_group();

    const int block_id = item.get_global_id(0);
    const int row_local_id = item.get_local_id(1);
    const int thread_id = item.get_local_id(2);

    // Row / chunk offsets into the [num_rows, num_experts] input.
    const int64_t token_row_offset = block_id * kMoeGateRowsPerWg * num_experts + row_local_id * num_experts;
    const int64_t token_row_chunk_offset = token_row_offset + thread_id * vpt;

    const T* thread_row_ptr = input_ + token_row_chunk_offset;
    const T* bias_ptr = bias_ + thread_id * vpt;

    T row_chunk[kMoeGateMaxVpt];
    T bias_chunk[kMoeGateMaxVpt];
    if (sg.get_local_id()[0] < static_cast<size_t>(num_expert_groups)) {
      for (int i = 0; i < vpt; ++i) {
        row_chunk[i] = thread_row_ptr[i];
        bias_chunk[i] = bias_ptr[i];
      }

      ////////////////////// Sigmoid //////////////////////
      for (int i = 0; i < vpt; ++i) {
        float x = static_cast<float>(-row_chunk[i]);
        row_chunk[i] = static_cast<T>(1.0f / (1.0f + ::sycl::exp(x)));
      }

      ////////////////////// Add Bias //////////////////////
      for (int i = 0; i < vpt; ++i) {
        bias_chunk[i] = row_chunk[i] + bias_chunk[i];
      }
    }

    ////////////////////// Exclude Groups //////////////////////
    for (int k_idx = 0; k_idx < num_expert_groups - topk_group_; ++k_idx) {
      int expert = thread_id * vpt;
      // local top-2
      T max_val = -TYPE_MAX_V;
      T max_val_second = -TYPE_MAX_V;
      if (sg.get_local_id()[0] < static_cast<size_t>(num_expert_groups)) {
        for (int i = 0; i < vpt; ++i) {
          float val = bias_chunk[i];
          if (val > max_val) {
            max_val_second = max_val;
            max_val = val;
          } else if (val > max_val_second) {
            max_val_second = val;
          }
        }
      }

      // Group weight = sum of top-2 sigmoid weights in the group.
      T max_sum = max_val + max_val_second;

      const uint32_t lane = sg.get_local_id()[0];  // 0..31
      const uint32_t logical_lane = lane & (num_expert_groups - 1);
      const uint32_t group_base = lane & ~(num_expert_groups - 1);

      for (int mask = num_expert_groups / 2; mask > 0; mask >>= 1) {
        const uint32_t target_logical = logical_lane ^ mask;
        const uint32_t target_lane = group_base + target_logical;
        const uint32_t xor_mask = lane ^ target_lane;

        T other_max_sum = ::sycl::permute_group_by_xor(sg, max_sum, xor_mask);
        int other_expert = ::sycl::permute_group_by_xor(sg, expert, xor_mask);

        // higher indices win
        if ((max_sum > other_max_sum) || ((other_max_sum == max_sum) && other_expert > expert)) {
          max_sum = other_max_sum;
          expert = other_expert;
        }
      }

      // exclude the winning group's experts from further consideration
      {
        const int thread_to_clear_in_group = expert / vpt;
        const int thread_group_idx = item.get_global_linear_id() % num_expert_groups;
        if (thread_group_idx == thread_to_clear_in_group) {
          for (int i = 0; i < vpt; ++i) {
            bias_chunk[i] = TYPE_MAX_V;
          }
        }
      }
    }

    ////////////////////// Topk //////////////////////
    float output_sum = 0.0f;
    const int first_elt_read_by_thread = thread_id * vpt;
    for (int k_idx = 0; k_idx < topk_excluding_share_expert_fusion; ++k_idx) {
      T max_val = -TYPE_MAX_V;
      int expert = 0;
      if (sg.get_local_id()[0] < static_cast<size_t>(num_expert_groups)) {
        max_val = bias_chunk[0];
        expert = first_elt_read_by_thread;
        if (max_val != TYPE_MAX_V) {
          for (int i = 1; i < vpt; ++i) {
            T val = bias_chunk[i];
            if (val > max_val) {
              max_val = val;
              expert = first_elt_read_by_thread + i;
            }
          }
        } else {
          max_val = -TYPE_MAX_V;
        }
      }

      const uint32_t lane = sg.get_local_id()[0];
      const uint32_t logical_lane = lane & (num_expert_groups - 1);
      const uint32_t group_base = lane & ~(num_expert_groups - 1);

      // argmax reduce over the logical sub-group
      for (int mask = num_expert_groups / 2; mask > 0; mask >>= 1) {
        const uint32_t target_logical = logical_lane ^ mask;
        const uint32_t target_lane = group_base + target_logical;
        const uint32_t xor_mask = lane ^ target_lane;

        T other_max = ::sycl::permute_group_by_xor(sg, max_val, xor_mask);
        int other_expert = ::sycl::permute_group_by_xor(sg, expert, xor_mask);

        // lower indices win on ties
        if ((other_max > max_val) || ((other_max == max_val) && other_expert < expert)) {
          max_val = other_max;
          expert = other_expert;
        }
      }

      // Store the selected top-k index/weight and clear it for the next round.
      if (sg.get_local_id()[0] < static_cast<size_t>(num_expert_groups)) {
        const int thread_to_clear_in_group = expert / vpt;
        const int thread_row = item.get_global_linear_id() / kMoeGateMaxVpt;
        const int idx = topk_ * thread_row + k_idx;
        const int thread_group_idx = item.get_global_linear_id() % num_expert_groups;
        if (thread_group_idx == thread_to_clear_in_group) {
          const int expert_to_clear_in_thread = expert % vpt;
          bias_chunk[expert_to_clear_in_thread] = -TYPE_MAX_V;
          output_[idx] = static_cast<float>(row_chunk[expert_to_clear_in_thread]);
          indices_[idx] = static_cast<int32_t>(expert);
        }

        if (thread_group_idx == 0) {
          output_sum += output_[idx];
        }
      }
    }

    if (sg.get_local_id()[0] < static_cast<size_t>(num_expert_groups)) {
      const int thread_group_idx = item.get_global_linear_id() % num_expert_groups;
      if (thread_group_idx == 0 && num_fused_shared_experts_ > 0) {
        const int thread_row = item.get_global_linear_id() / kMoeGateMaxVpt;
        int32_t last_idx = topk_ * thread_row + topk_excluding_share_expert_fusion;
        int32_t expert_offset = 0;
        indices_[last_idx] = static_cast<int32_t>(num_experts + expert_offset);
        output_[last_idx] = output_sum / routed_scaling_factor_;

        if (num_fused_shared_experts_ > 1) {
          for (int i = 1; i < num_fused_shared_experts_; ++i) {
            ++last_idx;
            ++expert_offset;
            indices_[last_idx] = static_cast<int32_t>(num_experts + expert_offset);
            output_[last_idx] = output_sum / routed_scaling_factor_;
          }
        }
      }

      ////////////////////// Rescale Output //////////////////////
      if (thread_group_idx == 0) {
        const int thread_row = item.get_global_linear_id() / kMoeGateMaxVpt;
        for (int i = 0; i < topk_; ++i) {
          const int64_t idx = topk_ * thread_row + i;
          output_[idx] = output_[idx] / output_sum;
          if (apply_routed_scaling_factor_on_output_) {
            output_[idx] *= routed_scaling_factor_;
          }
        }
      }
    }
  }

 private:
  const T* input_;
  const T* bias_;
  float* output_;
  int32_t* indices_;
  int32_t num_rows_;
  int32_t vpt_;
  int32_t num_experts_;
  int32_t num_expert_groups_;
  int32_t topk_group_;
  int32_t topk_;
  int32_t num_fused_shared_experts_;
  float routed_scaling_factor_;
  bool apply_routed_scaling_factor_on_output_;
};

// ---------------------------------------------------------------------------
// Host launcher (mirrors moe_fused_gate_kernel_dynamic in the AOT source).
// ---------------------------------------------------------------------------
template <typename T>
void moe_fused_gate_launcher(
    ::sycl::queue& queue,
    const void* input,
    const void* bias,
    void* output,
    void* indices,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  if (num_rows <= 0) return;

  // VPT = num_experts / num_expert_group (e.g. DeepSeek-V3: 256 / 8 = 32).
  const int32_t vpt = static_cast<int32_t>(num_experts / num_expert_group);

  const T* input_ptr = static_cast<const T*>(input);
  const T* bias_ptr = static_cast<const T*>(bias);
  float* output_ptr = static_cast<float*>(output);
  int32_t* indices_ptr = static_cast<int32_t*>(indices);

  const uint32_t num_blocks = (num_rows + kMoeGateRowsPerWg - 1) / kMoeGateRowsPerWg;
  const ::sycl::range<3> global_range{num_blocks, kMoeGateRowsPerWg, kMoeGateMaxVpt};
  const ::sycl::range<3> local_range{1, kMoeGateRowsPerWg, kMoeGateMaxVpt};

  MoeFusedGateKernel<T> task(
      input_ptr,
      bias_ptr,
      output_ptr,
      indices_ptr,
      static_cast<int32_t>(num_rows),
      vpt,
      static_cast<int32_t>(num_experts),
      static_cast<int32_t>(num_expert_group),
      static_cast<int32_t>(topk_group),
      static_cast<int32_t>(topk),
      static_cast<int32_t>(num_fused_shared_experts),
      routed_scaling_factor,
      apply_routed_scaling_factor_on_output);

  queue.submit([&](::sycl::handler& cgh) { cgh.parallel_for(::sycl::nd_range<3>(global_range, local_range), task); });
  // NOTE: no .wait() -- PyTorch owns stream synchronization.
}

// ---------------------------------------------------------------------------
// C API for the Python ctypes wrapper. One symbol per dtype; num_experts and
// num_expert_group are passed at runtime (dynamic kernel, no config allowlist).
// ---------------------------------------------------------------------------
#define DEFINE_MOE_GATE_FORWARD(DTYPE_SUFFIX, DTYPE)       \
  extern "C" void moe_fused_gate_forward_##DTYPE_SUFFIX(   \
      void* queue_ptr,                                     \
      const void* input,                                   \
      const void* bias,                                    \
      void* output,                                        \
      void* indices,                                       \
      int64_t num_rows,                                    \
      int64_t num_experts,                                 \
      int64_t num_expert_group,                            \
      int64_t topk_group,                                  \
      int64_t topk,                                        \
      int64_t num_fused_shared_experts,                    \
      float routed_scaling_factor,                         \
      int32_t apply_routed_scaling_factor_on_output) {     \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr); \
    moe_fused_gate_launcher<DTYPE>(                        \
        queue,                                             \
        input,                                             \
        bias,                                              \
        output,                                            \
        indices,                                           \
        num_rows,                                          \
        num_experts,                                       \
        num_expert_group,                                  \
        topk_group,                                        \
        topk,                                              \
        num_fused_shared_experts,                          \
        routed_scaling_factor,                             \
        apply_routed_scaling_factor_on_output != 0);       \
  }

// Compile only the requested dtype variant when SGL_MOE_GATE_DTYPE_* is set,
// otherwise emit all dtypes (for testing / pre-compilation).
#if defined(SGL_MOE_GATE_DTYPE_fp32)
DEFINE_MOE_GATE_FORWARD(fp32, float)
#elif defined(SGL_MOE_GATE_DTYPE_fp16)
DEFINE_MOE_GATE_FORWARD(fp16, ::sycl::half)
#elif defined(SGL_MOE_GATE_DTYPE_bf16)
DEFINE_MOE_GATE_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16)
#else
DEFINE_MOE_GATE_FORWARD(fp32, float)
DEFINE_MOE_GATE_FORWARD(fp16, ::sycl::half)
DEFINE_MOE_GATE_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16)
#endif

#undef DEFINE_MOE_GATE_FORWARD

}  // namespace sycl_kernel
}  // namespace sgl
