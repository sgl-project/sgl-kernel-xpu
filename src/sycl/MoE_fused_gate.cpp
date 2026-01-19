#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"

static constexpr int ROWS_PER_WG = 8;  // maximum CTA per work group
static constexpr int MAX_VPT = 32;     // maximum VPT we support, > params.VPT = num_expert / num_expert_group

template <typename T>
constexpr T max_value() {
  return std::numeric_limits<T>::max();
}

template <typename T, typename Params>
struct moe_fused_gate_impl {
  static constexpr T TYPE_MAX_V = max_value<T>();
  moe_fused_gate_impl(
      const T* input,
      const T* bias,
      float* output,
      int32_t* indices,
      int32_t num_rows,
      int32_t topk_group,
      int32_t topk,
      int32_t num_fused_shared_experts,
      double routed_scaling_factor,
      bool apply_routed_scaling_factor_on_output,
      Params params)
      : input_(input),
        bias_(bias),
        output_(output),
        indices_(indices),
        num_rows_(num_rows),
        topk_group_(topk_group),
        topk_(topk),
        num_fused_shared_experts_(num_fused_shared_experts),
        routed_scaling_factor_(routed_scaling_factor),
        apply_routed_scaling_factor_on_output_(apply_routed_scaling_factor_on_output),
        params_(params) {}

  [[sycl::reqd_sub_group_size(MAX_VPT)]]
  void operator()(sycl::nd_item<3> item) const {
    if (item.get_global_linear_id() / MAX_VPT >= num_rows_) return;

    // Calculate topk_excluding_share_expert_fusion from topk
    int32_t topk_excluding_share_expert_fusion = topk_ - num_fused_shared_experts_;
    // int32_t topk_excluding_share_expert_fusion = topk_;

    auto sg = item.get_sub_group();

    int block_id = item.get_global_id(0);
    int row_local_id = item.get_local_id(1);
    int thread_id = item.get_local_id(2);

    // pepare row offset and chunk offset
    int64_t token_row_offset = block_id * ROWS_PER_WG * params_.NUM_EXPERTS + row_local_id * params_.NUM_EXPERTS;
    int64_t token_row_chunk_offset = token_row_offset + thread_id * params_.VPT;

    auto* thread_row_ptr = input_ + token_row_chunk_offset;
    auto* bias_ptr = bias_ + thread_id * params_.VPT;

    T row_chunk[MAX_VPT];
    T bias_chunk[MAX_VPT];
    if (sg.get_local_id()[0] < params_.NUM_EXPERT_GROUPS) {
#pragma unroll
      for (int i = 0; i < params_.VPT; ++i) {
        row_chunk[i] = thread_row_ptr[i];
        bias_chunk[i] = bias_ptr[i];
      }

////////////////////// Sigmoid //////////////////////
#pragma unroll
      for (int i = 0; i < params_.VPT; ++i) {
        float x = static_cast<float>(-row_chunk[i]);
        row_chunk[i] = static_cast<T>(1.0f / (1.0f + sycl::exp(x)));
      }

////////////////////// Add Bias //////////////////////
#pragma unroll
      for (int i = 0; i < params_.VPT; ++i) {
        bias_chunk[i] = row_chunk[i] + bias_chunk[i];
      }
    }
    ////////////////////// Exclude Groups //////////////////////
    for (int k_idx = 0; k_idx < params_.NUM_EXPERT_GROUPS - topk_group_;
         ++k_idx) {  // QQ NOTE Here params.THREADS_PER_ROW = num_expert_group
      int expert = thread_id * params_.VPT;
      // local argmax
      T max_val = -TYPE_MAX_V;
      T max_val_second = -TYPE_MAX_V;
      if (sg.get_local_id()[0] < params_.NUM_EXPERT_GROUPS) {
#pragma unroll
        for (int i = 0; i < params_.VPT; ++i) {
          float val = bias_chunk[i];
          if (val > max_val) {
            max_val_second = max_val;
            max_val = val;
          } else if (val > max_val_second) {
            max_val_second = val;
          }
        }
      }

      // QQ NOTE: currently fixed to pick top2 sigmoid weight value in each expert group and sum them as the group
      // weight to select expert groups
      T max_sum = max_val + max_val_second;

      uint32_t lane = sg.get_local_id()[0];  // 0..15

      // Logical subgroup of size 8
      uint32_t logical_lane = lane & (params_.NUM_EXPERT_GROUPS - 1);  // lane % 8
      uint32_t group_base = lane & ~(params_.NUM_EXPERT_GROUPS - 1);   // (lane / 8) * 8

      // sug-group shuffle to find higher indices
      for (int mask = params_.NUM_EXPERT_GROUPS / 2; mask > 0; mask >>= 1) {
        uint32_t target_logical = logical_lane ^ mask;
        uint32_t target_lane = group_base + target_logical;

        // Convert absolute lane → xor distance
        uint32_t xor_mask = lane ^ target_lane;

        T other_max_sum = sycl::permute_group_by_xor(sg, max_sum, xor_mask);
        int other_expert = sycl::permute_group_by_xor(sg, expert, xor_mask);

        // higher indices win
        if ((max_sum > other_max_sum) || ((other_max_sum == max_sum) && other_expert > expert)) {
          max_sum = other_max_sum;
          expert = other_expert;
        }
      }

      // exclude topk expert group from experts
      {
        int const thread_to_clear_in_group = expert / params_.VPT;
        int thread_group_idx = item.get_global_linear_id() % params_.NUM_EXPERT_GROUPS;

        if (thread_group_idx == thread_to_clear_in_group) {
#pragma unroll
          for (int i = 0; i < params_.VPT; ++i) {
            bias_chunk[i] = TYPE_MAX_V;
          }
        }
      }
    }

    ////////////////////// Topk //////////////////////
    float output_sum = 0.0f;
    int first_elt_read_by_thread = thread_id * params_.VPT;
    for (int k_idx = 0; k_idx < topk_excluding_share_expert_fusion; ++k_idx) {
      T max_val = -TYPE_MAX_V;
      int expert = 0;
      if (sg.get_local_id()[0] < params_.NUM_EXPERT_GROUPS) {
        // local argmax
        max_val = bias_chunk[0];
        expert = first_elt_read_by_thread;
        if ((max_val != TYPE_MAX_V)) {
#pragma unroll
          for (int i = 1; i < params_.VPT; ++i) {
            T val = bias_chunk[i];
            if ((val > max_val)) {
              max_val = val;
              expert = first_elt_read_by_thread + i;
            }
          }
        } else {
          max_val = -TYPE_MAX_V;
        }
      }

      uint32_t lane = sg.get_local_id()[0];  // 0..15

      // Logical subgroup of size 8
      uint32_t logical_lane = lane & (params_.NUM_EXPERT_GROUPS - 1);  // lane % 8
      uint32_t group_base = lane & ~(params_.NUM_EXPERT_GROUPS - 1);   // (lane / 8) * 8

// argmax reduce
#pragma unroll
      for (int mask = params_.NUM_EXPERT_GROUPS / 2; mask > 0; mask >>= 1) {
        uint32_t target_logical = logical_lane ^ mask;
        uint32_t target_lane = group_base + target_logical;

        // Convert absolute lane → xor distance
        uint32_t xor_mask = lane ^ target_lane;

        T other_max = sycl::permute_group_by_xor(sg, max_val, xor_mask);
        int other_expert = sycl::permute_group_by_xor(sg, expert, xor_mask);

        // lower indices to win
        if ((other_max > max_val) || ((other_max == max_val) && other_expert < expert)) {
          max_val = other_max;
          expert = other_expert;
        }
      }

      // Select Topk indicis and weight
      if (sg.get_local_id()[0] < params_.NUM_EXPERT_GROUPS) {
        // div thr0, thr1 .. thr7 etc..
        int const thread_to_clear_in_group = expert / params_.VPT;
        // Skip Logical groups
        int thread_row = item.get_global_linear_id() / MAX_VPT;
        int idx = topk_ * thread_row + k_idx;
        ;
        int thread_group_idx = item.get_global_linear_id() % params_.NUM_EXPERT_GROUPS;  // liner % 8 t0, t1... t7
        if (thread_group_idx == thread_to_clear_in_group) {
          int expert_to_clear_in_thread = expert % params_.VPT;
          // clear the max value in the thread
          bias_chunk[expert_to_clear_in_thread] = -TYPE_MAX_V;
          // store output
          output_[idx] = static_cast<float>(row_chunk[expert_to_clear_in_thread]);
          indices_[idx] = static_cast<int32_t>(expert);
        }

        // accumulate sum for all elements
        if (thread_group_idx == 0) {
          output_sum += output_[idx];
        }
      }
    }

    if (sg.get_local_id()[0] < params_.NUM_EXPERT_GROUPS) {
      int thread_group_idx = item.get_global_linear_id() % params_.NUM_EXPERT_GROUPS;
      if (thread_group_idx == 0 && num_fused_shared_experts_ > 0) {
        // IMP Skip Logical groups
        int thread_row = item.get_global_linear_id() / MAX_VPT;
        int32_t last_idx = topk_ * thread_row + topk_excluding_share_expert_fusion;
        int32_t expert_offset = 0;
        indices_[last_idx] = static_cast<int32_t>(params_.NUM_EXPERTS + expert_offset);

        // Set the weight to the sum of all weights divided by routed_scaling_factor
        output_[last_idx] = output_sum / routed_scaling_factor_;

        if (num_fused_shared_experts_ > 1) {
          for (int i = 1; i < num_fused_shared_experts_; ++i) {
            ++last_idx;
            ++expert_offset;
            indices_[last_idx] = static_cast<int32_t>(params_.NUM_EXPERTS + expert_offset);
            // Set the weight to the sum of all weights divided by routed_scaling_factor
            output_[last_idx] = output_sum / routed_scaling_factor_;
          }
        }
      }

      ////////////////////// Rescale Output //////////////////////
      if (thread_group_idx == 0) {
        // IMP Skip Logical groups
        int thread_row = item.get_global_linear_id() / MAX_VPT;
#pragma unroll
        for (int i = 0; i < topk_; ++i) {
          int64_t const idx = topk_ * thread_row + i;
          output_[idx] = output_[idx] / output_sum;
          if (apply_routed_scaling_factor_on_output_) {
            output_[idx] *= routed_scaling_factor_;
          }
        }
      }
    }
  }

  const T* input_;
  const T* bias_;
  float* output_;
  int32_t* indices_;
  int32_t num_rows_;
  int32_t topk_group_;
  int32_t topk_;
  int32_t num_fused_shared_experts_;
  float routed_scaling_factor_;
  bool apply_routed_scaling_factor_on_output_;
  Params params_;
};

//------------------------------------------------------------------------------
// Templated Kernel Version (using compile-time constants)
//------------------------------------------------------------------------------
template <int VPT_, int NUM_EXPERTS_, int THREADS_PER_ROW_>
struct KernelParams {
  static constexpr int VPT = VPT_;
  static constexpr int NUM_EXPERTS = NUM_EXPERTS_;
  static constexpr int NUM_EXPERT_GROUPS = THREADS_PER_ROW_;
};

template <typename T, int VPT, int NUM_EXPERTS, int THREADS_PER_ROW>
void moe_fused_gate_kernel(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output,
    torch::Tensor& indices,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  auto input_ptr = reinterpret_cast<T*>(input.data_ptr());
  auto bias_ptr = reinterpret_cast<T*>(bias.data_ptr());
  auto output_ptr = reinterpret_cast<float*>(output.data_ptr());
  auto indices_ptr = reinterpret_cast<int32_t*>(indices.data_ptr());

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  uint32_t num_blocks = (num_rows + ROWS_PER_WG - 1) / ROWS_PER_WG;
  sycl::range<3> global_range{num_blocks, ROWS_PER_WG, MAX_VPT};
  sycl::range<3> local_range{1, ROWS_PER_WG, MAX_VPT};

  KernelParams<VPT, NUM_EXPERTS, THREADS_PER_ROW> params;
  using Kernel = moe_fused_gate_impl<T, decltype(params)>;

  Kernel task(
      input_ptr,
      bias_ptr,
      output_ptr,
      indices_ptr,
      num_rows,
      topk_group,
      topk,
      num_fused_shared_experts,
      routed_scaling_factor,
      apply_routed_scaling_factor_on_output,
      params);
  ;

  sycl_kernel_submit(global_range, local_range, queue, task);
}

// Macro to compute compile-time constants and launch the kernel.
#define LAUNCH_MOE_GATE_CONFIG(T, EXPERTS, EXPERT_GROUP)      \
  do {                                                        \
    constexpr int VPT = (EXPERTS) / (EXPERT_GROUP);           \
    moe_fused_gate_kernel<T, VPT, (EXPERTS), (EXPERT_GROUP)>( \
        input,                                                \
        bias,                                                 \
        output,                                               \
        indices,                                              \
        num_rows,                                             \
        topk_group,                                           \
        topk,                                                 \
        num_fused_shared_experts,                             \
        routed_scaling_factor,                                \
        apply_routed_scaling_factor_on_output);               \
    dispatched = true;                                        \
  } while (0)

//------------------------------------------------------------------------------
// Dynamic Kernel Version (parameters computed at runtime)
//------------------------------------------------------------------------------
struct KernelParamsDynamic {
  int VPT;
  int NUM_EXPERTS;
  int NUM_EXPERT_GROUPS;
};

template <typename T>
void moe_fused_gate_kernel_dynamic(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output,
    torch::Tensor& indices,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  auto input_ptr = reinterpret_cast<T*>(input.data_ptr());
  auto bias_ptr = reinterpret_cast<T*>(bias.data_ptr());
  auto output_ptr = reinterpret_cast<float*>(output.data_ptr());
  auto indices_ptr = reinterpret_cast<int32_t*>(indices.data_ptr());
  int32_t num_experts = input.size(1);

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  uint32_t num_blocks = (num_rows + ROWS_PER_WG - 1) / ROWS_PER_WG;
  sycl::range<3> global_range{num_blocks, ROWS_PER_WG, MAX_VPT};
  sycl::range<3> local_range{1, ROWS_PER_WG, MAX_VPT};

  KernelParamsDynamic params;
  params.NUM_EXPERTS = num_experts;       // e.g, for deepseek v3, this is 256
  params.VPT = num_experts / topk_group;  // e.g., for deepseek v3, this is 256 / 8 = 32
  params.NUM_EXPERT_GROUPS = topk_group;  // fixed as num_expert_group, e.g., for deepseek v3, this is 8

  using Kernel = moe_fused_gate_impl<T, decltype(params)>;

  Kernel task(
      input_ptr,
      bias_ptr,
      output_ptr,
      indices_ptr,
      num_rows,
      topk_group,
      topk,
      num_fused_shared_experts,
      routed_scaling_factor,
      apply_routed_scaling_factor_on_output,
      params);

  sycl_kernel_submit(global_range, local_range, queue, task);
}

//------------------------------------------------------------------------------
// Host Launcher Function
//------------------------------------------------------------------------------
std::vector<at::Tensor> moe_fused_gate(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  TORCH_CHECK(input.dtype() == bias.dtype(), "input and bias should have the same dtype");

  int64_t num_rows = input.size(0);
  int32_t num_experts = input.size(1);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());

  auto output = torch::empty({num_rows, topk}, options);
  auto indices = torch::empty({num_rows, topk}, options.dtype(torch::kInt32));

  // Check 1: Ensure that num_experts is a power of 2.
  TORCH_CHECK((num_experts & (num_experts - 1)) == 0, "num_experts must be a power of 2, but got ", num_experts);

  // Check 2: Ensure that num_experts is divisible by num_expert_group. (this also means num_expert_group is power of 2)
  TORCH_CHECK(
      num_experts % num_expert_group == 0,
      "num_experts must be divisible by num_expert_group, but got ",
      num_experts,
      " / ",
      num_expert_group);

  int computed_vpt = num_experts / num_expert_group;

  // Check 3: Ensure that num_experts/num_expert_group does not exceed MAX_VPT=32. Maximum VPT indicate max value per
  // threads we can process.
  TORCH_CHECK(
      computed_vpt <= MAX_VPT,
      "Per group experts: num_experts / num_expert_group = (",
      computed_vpt,
      ") exceeds the maximum supported (",
      MAX_VPT,
      ")");

  int64_t num_blocks = (num_rows + ROWS_PER_WG - 1) / ROWS_PER_WG;
  bool dispatched = false;

  SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "moe_sum_reduce_impl", [&]() {
        switch (num_experts) {
          case 512:
            if (num_expert_group == 16) {
              LAUNCH_MOE_GATE_CONFIG(scalar_t, 512, 16);
            }
            break;
          case 256:
            if (num_expert_group == 8) {
              LAUNCH_MOE_GATE_CONFIG(scalar_t, 256, 8);
            } else if (num_expert_group == 16) {
              LAUNCH_MOE_GATE_CONFIG(scalar_t, 256, 16);
            }
            break;
          case 128:
            if (num_expert_group == 4) {
              LAUNCH_MOE_GATE_CONFIG(scalar_t, 128, 4);
            } else if (num_expert_group == 8) {
              LAUNCH_MOE_GATE_CONFIG(scalar_t, 128, 8);
            }
            break;
          default:
            break;
        }
        if (!dispatched) {
          // Fallback to the dynamic kernel if none of the supported combinations match.
          // currently only support num_experts / num_expert_group <= 32 for dynamic kernels
          moe_fused_gate_kernel_dynamic<scalar_t>(
              input,
              bias,
              output,
              indices,
              num_rows,
              topk_group,
              topk,
              num_fused_shared_experts,
              routed_scaling_factor,
              apply_routed_scaling_factor_on_output);
        }
      });
  return {output, indices};
}
