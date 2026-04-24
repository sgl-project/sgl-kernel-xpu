#include <ATen/ATen.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

constexpr int block_size = 128;

template <typename T>
struct compute_problem_sizes_sycl_K_T {
  compute_problem_sizes_sycl_K_T(
      const T* topk_ids,
      T* problem_sizes1,
      T* problem_sizes2,
      T* atomic_buffer,
      const uint32_t num_experts,
      const uint32_t topk_length,
      const uint32_t n,
      const uint32_t k,
      const uint32_t max_tokens_per_expert)
      : topk_ids_(topk_ids),
        problem_sizes1_(problem_sizes1),
        problem_sizes2_(problem_sizes2),
        atomic_buffer_(atomic_buffer),
        num_experts_(num_experts),
        topk_length_(topk_length),
        n_(n),
        k_(k),
        max_tokens_per_expert_(max_tokens_per_expert) {}

  void operator()(sycl::nd_item<1> item) const {
    int thread_id = item.get_local_linear_id();
    int expert_id = item.get_group(0);

    // thread_id < topk_length_ is always true for all launched threads because
    // the WG size is min(max_wg_size, topk_length) (see compute_problem_sizes_sycl_impl).
    T occurrences = 0;
    for (int i = thread_id; i < topk_length_; i += max_tokens_per_expert_) {
      occurrences += (topk_ids_[i] == expert_id);
    }

    // Use work-group reduction instead of an atomic accumulation so that the
    // caller does not need to pre-zero the atomic_buffer (expert_offsets) array.
    T final_occurrences = sycl::reduce_over_group(item.get_group(), occurrences, sycl::plus<T>());

    if (thread_id == 0) {
      // Write per-expert token count so compute_expert_offsets can read it.
      atomic_buffer_[expert_id] = final_occurrences;
      problem_sizes1_[expert_id * 3] = final_occurrences;
      problem_sizes1_[expert_id * 3 + 1] = static_cast<int32_t>(2 * n_);
      problem_sizes1_[expert_id * 3 + 2] = static_cast<int32_t>(k_);
      problem_sizes2_[expert_id * 3] = final_occurrences;
      problem_sizes2_[expert_id * 3 + 1] = static_cast<int32_t>(k_);
      problem_sizes2_[expert_id * 3 + 2] = static_cast<int32_t>(n_);
    }
  }

  const T* topk_ids_;
  T* problem_sizes1_;
  T* problem_sizes2_;
  T* atomic_buffer_;
  const uint32_t num_experts_;
  const uint32_t topk_length_;
  const uint32_t n_;
  const uint32_t k_;
  const uint32_t max_tokens_per_expert_;
};

template <typename T>
void compute_problem_sizes_sycl_impl(
    const torch::Tensor& topk_ids,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& expert_offsets,
    const uint32_t num_experts,
    const uint32_t n,
    const uint32_t k) {
  const T* topk_ptr = static_cast<const T*>(topk_ids.data_ptr());
  T* problem_sizes1_ptr = static_cast<T*>(problem_sizes1.data_ptr());
  T* problem_sizes2_ptr = static_cast<T*>(problem_sizes2.data_ptr());
  T* atomic_buffer = static_cast<T*>(expert_offsets.data_ptr());

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using Kernel = compute_problem_sizes_sycl_K_T<T>;

  const uint32_t topk_length = topk_ids.numel();
  auto dev_id = topk_ids.device().index();
  uint32_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  uint32_t max_tokens_per_expert = static_cast<uint32_t>(sycl::min(max_wg_size, topk_length));

  sycl::range<1> global_range{num_experts * max_tokens_per_expert};
  sycl::range<1> local_range{max_tokens_per_expert};

  Kernel task(
      topk_ptr,
      problem_sizes1_ptr,
      problem_sizes2_ptr,
      atomic_buffer,
      num_experts,
      topk_length,
      n,
      k,
      max_tokens_per_expert);

  sycl_kernel_submit(global_range, local_range, queue, task);

  return;
}

template <typename T, int BLOCK_SIZE>
struct compute_expert_blockscale_offsets_sycl_K_T {
  compute_expert_blockscale_offsets_sycl_K_T(
      const T* problem_sizes1, T* expert_offsets, T* blockscale_offsets, T* atomic_buffer, const T num_experts)
      : problem_sizes1_(problem_sizes1),
        expert_offsets_(expert_offsets),
        blockscale_offsets_(blockscale_offsets),
        atomic_buffer_(atomic_buffer),
        num_experts_(num_experts) {}

  void operator()(sycl::nd_item<1> item) const {
    T tot_offset = 0;
    T tot_rounded_offset = 0;
    expert_offsets_[0] = 0;
    blockscale_offsets_[0] = 0;
    for (int i = 0; i < num_experts_; ++i) {
      atomic_buffer_[i] = tot_offset;
      T num_tokens = problem_sizes1_[i * 3];
      T rounded_num_tokens = div_up(num_tokens, static_cast<T>(BLOCK_SIZE)) * BLOCK_SIZE;  // align to block_size
      tot_offset += num_tokens;
      tot_rounded_offset += rounded_num_tokens;
      expert_offsets_[i + 1] = tot_offset;
      blockscale_offsets_[i + 1] = tot_rounded_offset;
    }
  }

  const T* problem_sizes1_;
  T* expert_offsets_;
  T* blockscale_offsets_;
  T* atomic_buffer_;
  const uint32_t num_experts_;
};

template <typename T>
void compute_expert_blockscale_offsets_sycl_impl(
    torch::Tensor& problem_sizes1,
    torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets,
    torch::Tensor& atomic_buffer,
    const uint32_t num_experts) {
  const T* problem_sizes1_ptr = static_cast<const T*>(problem_sizes1.data_ptr());
  T* expert_offsets_ptr = static_cast<T*>(expert_offsets.data_ptr());
  T* blockscale_offsets_ptr = static_cast<T*>(blockscale_offsets.data_ptr());
  T* atomic_buffer_ptr = static_cast<T*>(atomic_buffer.data_ptr());

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using Kernel = compute_expert_blockscale_offsets_sycl_K_T<T, block_size>;

  Kernel task(problem_sizes1_ptr, expert_offsets_ptr, blockscale_offsets_ptr, atomic_buffer_ptr, num_experts);

  sycl_kernel_submit(1, 1, queue, task);
  return;
}

template <typename T>
struct compute_expert_offsets_sycl_k_T {
  compute_expert_offsets_sycl_k_T(T* expert_offsets, T* atomic_buffer, const uint32_t num_experts)
      : expert_offsets_(expert_offsets), atomic_buffer_(atomic_buffer), num_experts_(num_experts) {}

  void operator()(sycl::nd_item<1> it) const {
    int lid = it.get_local_id(0);
    T x = (lid < num_experts_) ? expert_offsets_[lid] : 0;
    T scanned = exclusive_scan_over_group(it.get_group(), x, sycl::plus<T>());
    if (lid < num_experts_) atomic_buffer_[lid] = scanned;
  }

  T* expert_offsets_;
  T* atomic_buffer_;
  const uint32_t num_experts_;
};

template <typename T>
void compute_expert_offsets_sycl_impl(
    torch::Tensor& expert_offsets, torch::Tensor& atomic_buffer, const uint32_t num_experts) {
  T* expert_offsets_ptr = static_cast<T*>(expert_offsets.data_ptr());
  T* atomic_buffer_ptr = static_cast<T*>(atomic_buffer.data_ptr());

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using Kernel = compute_expert_offsets_sycl_k_T<T>;

  Kernel task(expert_offsets_ptr, atomic_buffer_ptr, num_experts);

  sycl_kernel_submit(num_experts, num_experts, queue, task);
  return;
}

template <typename T>
struct compute_arg_sorts_sycl_K_T {
  compute_arg_sorts_sycl_K_T(
      const T* topk_ids,
      T* input_permutation,
      T* output_permutation,
      T* atomic_buffer,
      const int32_t topk_length,
      const int32_t topk)
      : topk_ids_(topk_ids),
        input_permutation_(input_permutation),
        output_permutation_(output_permutation),
        atomic_buffer_(atomic_buffer),
        topk_length_(topk_length),
        topk_(topk) {}

  // One thread per token-expert pair. Device-scope atomic on per-expert counter
  // (atomic_buffer[e] pre-loaded with expert start offsets by compute_expert_offsets).
  // O(topk_length) total work vs the previous O(num_experts * topk_length) scan.
  void operator()(sycl::nd_item<1> item) const {
    int i = item.get_global_id(0);
    if (i >= topk_length_) return;

    T expert = topk_ids_[i];

    sycl::atomic_ref<
        T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        counter(atomic_buffer_[expert]);

    T pos = counter.fetch_add(1);
    input_permutation_[pos] = i / topk_;
    output_permutation_[i] = pos;
  }

  const T* topk_ids_;
  T* input_permutation_;
  T* output_permutation_;
  T* atomic_buffer_;
  const uint32_t topk_length_;
  const uint32_t topk_;
};

template <typename T>
void compute_arg_sorts_sycl_impl(
    const torch::Tensor& topk_ids,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    torch::Tensor& atomic_buffer,
    const uint32_t num_experts) {
  const T* topk_ids_ptr = static_cast<const T*>(topk_ids.data_ptr());
  T* input_permutation_ptr = static_cast<T*>(input_permutation.data_ptr());
  T* output_permutation_ptr = static_cast<T*>(output_permutation.data_ptr());
  T* atomic_buffer_ptr = static_cast<T*>(atomic_buffer.data_ptr());

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using Kernel = compute_arg_sorts_sycl_K_T<T>;

  const uint32_t topk_length = topk_ids.numel();
  const int32_t topk = topk_ids.size(1);
  auto dev_id = topk_ids.device().index();
  uint32_t wg_size = static_cast<uint32_t>(std::min((uint32_t)dpcppMaxWorkGroupSize(dev_id), topk_length));
  uint32_t num_wgs = (topk_length + wg_size - 1) / wg_size;

  sycl::range<1> global_range{num_wgs * wg_size};
  sycl::range<1> local_range{wg_size};

  Kernel task(topk_ids_ptr, input_permutation_ptr, output_permutation_ptr, atomic_buffer_ptr, topk_length, topk);

  sycl_kernel_submit(global_range, local_range, queue, task);
  return;
}

void prepare_moe_input(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    const std::optional<torch::Tensor>& blockscale_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k) {
  TORCH_CHECK(topk_ids.scalar_type() == problem_sizes1.scalar_type(), "problem_sizes1 must have same type as topk_ids");
  TORCH_CHECK(topk_ids.scalar_type() == expert_offsets.scalar_type(), "expert_offsets must have same type as topk_ids");
  TORCH_CHECK(topk_ids.scalar_type() == problem_sizes2.scalar_type(), "problem_sizes2 must have same type as topk_ids");
  TORCH_CHECK(
      topk_ids.scalar_type() == input_permutation.scalar_type(), "input_permutation must have same type as topk_ids");
  TORCH_CHECK(
      topk_ids.scalar_type() == output_permutation.scalar_type(), "output_permutation must have same type as topk_ids");

  AT_DISPATCH_INDEX_TYPES(topk_ids.scalar_type(), "prepare_moe_input", [&] {
    using index_t = index_t;

    auto options_type = torch::TensorOptions().dtype(topk_ids.dtype()).device(topk_ids.device());
    torch::Tensor atomic_buffer = torch::empty(num_experts + 1, options_type);

    compute_problem_sizes_sycl_impl<index_t>(
        topk_ids, problem_sizes1, problem_sizes2, expert_offsets, num_experts, n, k);

    if (blockscale_offsets.has_value()) {
      compute_expert_blockscale_offsets_sycl_impl<index_t>(
          problem_sizes1, expert_offsets, blockscale_offsets.value(), atomic_buffer, num_experts);
    } else {
      compute_expert_offsets_sycl_impl<index_t>(expert_offsets, atomic_buffer, num_experts);
    }

    compute_arg_sorts_sycl_impl<index_t>(topk_ids, input_permutation, output_permutation, atomic_buffer, num_experts);
  });
  return;
}

// Scatter kernel: 1 WG per source token, reads token once, scatters to topk destinations.
// Equivalent to IPEX MoEScatter but uses precomputed src2dst_map (c_map / output_permutation).
template <typename T>
struct ScatterTokensToExperts {
  static constexpr int WGSize = 256;
  static constexpr int ElemsPerItem = sizeof(float) * 4 / sizeof(T);  // 4 for bf16/fp16, 16 for fp8
  static constexpr int Stride = WGSize * ElemsPerItem;
  static constexpr int MAX_TOPK = 16;

  // Use uint8_t storage for FP8 types since SYCL doesn't natively support FP8 vectors
  using storage_t = std::conditional_t<std::is_same_v<T, c10::Float8_e4m3fn>, uint8_t, T>;

  ScatterTokensToExperts(
      const T* input, T* output, const int32_t* src2dst_map, const int32_t topk, const int32_t hidden_dim)
      : input_(input), output_(output), src2dst_map_(src2dst_map), topk_(topk), hidden_dim_(hidden_dim) {}

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    int token_id = item.get_group(0);
    int local_id = item.get_local_linear_id();

    // Load topk destination row indices for this token (loop-invariant)
    int dst_rows[MAX_TOPK];
    for (int k = 0; k < topk_ && k < MAX_TOPK; ++k) {
      dst_rows[k] = src2dst_map_[token_id * topk_ + k];
    }

    // Source base pointer (sequential, coalesced reads)
    const T* src_base = input_ + token_id * hidden_dim_ + local_id * ElemsPerItem;

    const int loop_count = (hidden_dim_ + Stride - 1) / Stride;
    for (int loop = 0; loop < loop_count; ++loop) {
      if (loop * Stride + local_id * ElemsPerItem < hidden_dim_) {
        using vec_t = sycl::vec<storage_t, ElemsPerItem>;
        vec_t data = *(reinterpret_cast<const vec_t*>(reinterpret_cast<const storage_t*>(src_base + loop * Stride)));
        for (int k = 0; k < topk_ && k < MAX_TOPK; ++k) {
          T* dst = output_ + dst_rows[k] * hidden_dim_ + local_id * ElemsPerItem + loop * Stride;
          *(reinterpret_cast<vec_t*>(reinterpret_cast<storage_t*>(dst))) = data;
        }
      }
    }
  }

  const T* input_;
  T* output_;
  const int32_t* src2dst_map_;
  const int32_t topk_;
  const int32_t hidden_dim_;
};

template <typename T>
void scatter_tokens_to_experts_impl(
    const torch::Tensor& input_tensor, const torch::Tensor& src2dst_map, torch::Tensor& output_tensor) {
  auto input = reinterpret_cast<T*>(input_tensor.data_ptr());
  auto src2dst = reinterpret_cast<const int32_t*>(src2dst_map.data_ptr());
  auto output = reinterpret_cast<T*>(output_tensor.data_ptr());

  uint32_t num_tokens = input_tensor.size(0);
  uint32_t num_dest_rows = output_tensor.size(0);
  uint32_t hidden_dim = input_tensor.size(1);
  int32_t topk = static_cast<int32_t>(num_dest_rows / num_tokens);

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using Kernel = ScatterTokensToExperts<T>;
  sycl::range<1> global_range{num_tokens * Kernel::WGSize};
  sycl::range<1> local_range{Kernel::WGSize};

  Kernel task(input, output, src2dst, topk, hidden_dim);
  sycl_kernel_submit(global_range, local_range, queue, task);
}

void scatter_tokens_to_experts(
    const torch::Tensor& input_tensor, const torch::Tensor& src2dst_map, torch::Tensor& output_tensor) {
  TORCH_CHECK(
      input_tensor.scalar_type() == output_tensor.scalar_type(),
      "Input and output tensors must have the same data type");

  // Handle FP8 type separately
  if (input_tensor.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    scatter_tokens_to_experts_impl<c10::Float8_e4m3fn>(input_tensor, src2dst_map, output_tensor);
    return;
  }

  SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      input_tensor.scalar_type(),
      "scatter_tokens_to_experts_impl",
      [&]() { scatter_tokens_to_experts_impl<scalar_t>(input_tensor, src2dst_map, output_tensor); });
}

template <typename T, typename T1, bool APPLY_ROUTED_SCALING>
struct ApplyShuffleMulSum {
  static constexpr int WGSize = 256;
  static constexpr int ElemsPerItem = sizeof(float) * 4 / sizeof(T);  // 4 for bf16/fp16
  static constexpr int Stride = WGSize * ElemsPerItem;
  static constexpr int MAX_TOPK = 16;

  ApplyShuffleMulSum(
      const T* input,
      T* output,
      const int32_t* dst2src_map,
      const T1* factors,
      const int32_t topk,
      const int32_t hidden_dim,
      float routed_scaling_factor)
      : input_(input),
        output_(output),
        dst2src_map_(dst2src_map),
        factors_(factors),
        topk_(topk),
        hidden_dim_(hidden_dim),
        routed_scaling_factor_(routed_scaling_factor) {}

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    int out_tkn_id = item.get_group(0);
    int local_id = item.get_local_linear_id();

    // Preload src row indices and weights (loop-invariant over hidden dim)
    int src_indices[MAX_TOPK];
    float weights[MAX_TOPK];
    for (int k = 0; k < topk_ && k < MAX_TOPK; ++k) {
      src_indices[k] = static_cast<int>(dst2src_map_[out_tkn_id * topk_ + k]);
      weights[k] = (factors_ != nullptr) ? static_cast<float>(factors_[out_tkn_id * topk_ + k]) : 0.0f;
    }

    T* dst_base = output_ + out_tkn_id * hidden_dim_ + local_id * ElemsPerItem;
    const int loop_count = (hidden_dim_ + Stride - 1) / Stride;

    for (int loop = 0; loop < loop_count; ++loop) {
      if (loop * Stride + local_id * ElemsPerItem < hidden_dim_) {
        // Float accumulator for better precision and perf
        sycl::vec<float, ElemsPerItem> acc;
        for (int j = 0; j < ElemsPerItem; ++j)
          acc[j] = 0.0f;

        for (int k = 0; k < topk_ && k < MAX_TOPK; ++k) {
          const T* src = input_ + src_indices[k] * hidden_dim_ + local_id * ElemsPerItem + loop * Stride;
          using vec_t = sycl::vec<T, ElemsPerItem>;
          vec_t reg = *(reinterpret_cast<const vec_t*>(src));
          for (int j = 0; j < ElemsPerItem; ++j) {
            if constexpr (APPLY_ROUTED_SCALING) {
              acc[j] += static_cast<float>(reg[j]) * weights[k] * routed_scaling_factor_;
            } else {
              acc[j] += static_cast<float>(reg[j]) * weights[k];
            }
          }
        }

        using vec_t = sycl::vec<T, ElemsPerItem>;
        vec_t store;
        for (int j = 0; j < ElemsPerItem; ++j) {
          store[j] = static_cast<T>(acc[j]);
        }
        *(reinterpret_cast<vec_t*>(dst_base + loop * Stride)) = store;
      }
    }
  }
  const T* input_;
  T* output_;
  const int32_t* dst2src_map_;
  const T1* factors_;
  const int32_t topk_;
  const int32_t hidden_dim_;
  float routed_scaling_factor_;
};

template <typename T, typename T1, bool APPLY_ROUTED_SCALING>
void apply_shuffle_mul_sum_impl(
    const T* input,
    T* output,
    const int32_t* dst2src_map,
    const T1* factors,
    const uint32_t out_tkns,
    const uint32_t out_hidden_dims,
    const int topk,
    float routed_scaling_factor) {
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using Kernel = ApplyShuffleMulSum<T, T1, APPLY_ROUTED_SCALING>;

  sycl::range<1> global_range{out_tkns * Kernel::WGSize};
  sycl::range<1> local_range{Kernel::WGSize};

  Kernel task(input, output, dst2src_map, factors, topk, out_hidden_dims, routed_scaling_factor);

  sycl_kernel_submit(global_range, local_range, queue, task);
  return;
}

void apply_shuffle_mul_sum(
    const torch::Tensor& input,
    torch::Tensor& output,
    const torch::Tensor& permutation,
    double routed_scaling_factor,
    const std::optional<torch::Tensor>& factors) {
  int m = output.size(0);
  int topk = int(permutation.size(0) / m);
  bool use_routed_scaling = routed_scaling_factor != 1.0f;

  SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "apply_shuffle_mul_sum", [&]() {
        using input_t = scalar_t;
        if (factors.has_value()) {
          SYCL_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::BFloat16, at::ScalarType::Half, factors.value().scalar_type(), "factors dispatch", [&]() {
                using factors_t = scalar_t;
                if (use_routed_scaling) {
                  apply_shuffle_mul_sum_impl<input_t, factors_t, true>(
                      reinterpret_cast<input_t*>(input.data_ptr()),
                      reinterpret_cast<input_t*>(output.data_ptr()),
                      reinterpret_cast<int32_t*>(permutation.data_ptr()),
                      reinterpret_cast<factors_t*>(factors->data_ptr()),
                      output.size(0),
                      output.size(1),
                      topk,
                      routed_scaling_factor);
                } else {
                  apply_shuffle_mul_sum_impl<input_t, factors_t, false>(
                      reinterpret_cast<input_t*>(input.data_ptr()),
                      reinterpret_cast<input_t*>(output.data_ptr()),
                      reinterpret_cast<int32_t*>(permutation.data_ptr()),
                      reinterpret_cast<factors_t*>(factors->data_ptr()),
                      output.size(0),
                      output.size(1),
                      topk,
                      routed_scaling_factor);
                }
              });
        } else {
          if (use_routed_scaling) {
            apply_shuffle_mul_sum_impl<input_t, input_t, true>(
                reinterpret_cast<input_t*>(input.data_ptr()),
                reinterpret_cast<input_t*>(output.data_ptr()),
                reinterpret_cast<int32_t*>(permutation.data_ptr()),
                nullptr,
                output.size(0),
                output.size(1),
                topk,
                routed_scaling_factor);
          } else {
            apply_shuffle_mul_sum_impl<input_t, input_t, false>(
                reinterpret_cast<input_t*>(input.data_ptr()),
                reinterpret_cast<input_t*>(output.data_ptr()),
                reinterpret_cast<int32_t*>(permutation.data_ptr()),
                nullptr,
                output.size(0),
                output.size(1),
                topk,
                routed_scaling_factor);
          }
        }
      });
  return;
}
