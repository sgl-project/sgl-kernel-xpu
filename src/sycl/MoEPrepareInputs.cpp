#include <ATen/ATen.h>
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
    if (thread_id < topk_length_) {
      int expert_id = item.get_group(0);

      T occurrences = 0;
      for (int i = thread_id; i < topk_length_; i += max_tokens_per_expert_) {
        occurrences += (topk_ids_[i] == expert_id);
      }

      sycl::atomic_ref<
          T,
          sycl::memory_order::relaxed,
          sycl::memory_scope::work_group,
          sycl::access::address_space::generic_space>
          atomic_counter(atomic_buffer_[expert_id]);

      atomic_counter.fetch_add(occurrences);

      item.barrier(sycl::access::fence_space::local_space);

      if (thread_id == 0) {
        T final_occurrences = atomic_buffer_[expert_id];
        problem_sizes1_[expert_id * 3] = final_occurrences;
        problem_sizes1_[expert_id * 3 + 1] = static_cast<int32_t>(2 * n_);
        problem_sizes1_[expert_id * 3 + 2] = static_cast<int32_t>(k_);
        problem_sizes2_[expert_id * 3] = final_occurrences;
        problem_sizes2_[expert_id * 3 + 1] = static_cast<int32_t>(k_);
        problem_sizes2_[expert_id * 3 + 2] = static_cast<int32_t>(n_);
      }
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
      const int32_t topk,
      const int32_t num_experts,
      const int32_t max_tokens_per_expert)
      : topk_ids_(topk_ids),
        input_permutation_(input_permutation),
        output_permutation_(output_permutation),
        atomic_buffer_(atomic_buffer),
        topk_length_(topk_length),
        topk_(topk),
        num_experts_(num_experts),
        max_tokens_per_expert_(max_tokens_per_expert) {}

  void operator()(sycl::nd_item<1> item) const {
    int expert_id = item.get_group(0);

    sycl::atomic_ref<
        T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::work_group,
        sycl::access::address_space::generic_space>
        atomic_counter(atomic_buffer_[expert_id]);

    for (int32_t i = item.get_local_id(0); i < topk_length_; i += max_tokens_per_expert_) {
      if (topk_ids_[i] == expert_id) {
        T start = atomic_counter.fetch_add(1);
        input_permutation_[start] = i / topk_;
        output_permutation_[i] = start;
      }
    }
  }

  const T* topk_ids_;
  T* input_permutation_;
  T* output_permutation_;
  T* atomic_buffer_;
  const uint32_t topk_length_;
  const uint32_t topk_;
  const uint32_t num_experts_;
  const uint32_t max_tokens_per_expert_;
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
  uint32_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  uint32_t max_tokens_per_expert = static_cast<uint32_t>(sycl::min(max_wg_size, topk_length));

  sycl::range<1> global_range{num_experts * max_tokens_per_expert};
  sycl::range<1> local_range{max_tokens_per_expert};

  Kernel task(
      topk_ids_ptr,
      input_permutation_ptr,
      output_permutation_ptr,
      atomic_buffer_ptr,
      topk_length,
      topk,
      num_experts,
      max_tokens_per_expert);

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
    torch::Tensor atomic_buffer = torch::zeros(num_experts + 1, options_type);

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

template <typename T, uint32_t N>
struct alignas(16) VecArray {
  T data[N];
};

template <typename T, uint32_t N>
inline void vec_cast_to_float(const VecArray<T, N>& src, float (&dst)[N]) {
#pragma unroll
  for (uint32_t i = 0; i < N; ++i)
    dst[i] = static_cast<float>(src.data[i]);
}

template <typename T, uint32_t N>
inline void vec_cast_from_float(const float (&src)[N], VecArray<T, N>& dst) {
#pragma unroll
  for (uint32_t i = 0; i < N; ++i)
    dst.data[i] = static_cast<T>(src[i]);
}

template <typename T>
struct ShuffleRowsKernel {
  static constexpr uint32_t ELEM_PER_THREAD = 16u / sizeof(T);  // 128-bit
  using DataElem = VecArray<T, ELEM_PER_THREAD>;

  const T* input;
  const int32_t* dst2src_map;
  T* output;
  int64_t num_cols;
  int64_t num_elems_in_col;  // == num_cols / ELEM_PER_THREAD

  ShuffleRowsKernel(const T* input_, const int32_t* dst2src_map_, T* output_, int64_t num_cols_)
      : input(input_),
        dst2src_map(dst2src_map_),
        output(output_),
        num_cols(num_cols_),
        num_elems_in_col(num_cols_ / static_cast<int64_t>(ELEM_PER_THREAD)) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t dest_row_idx = static_cast<int64_t>(item.get_group(0));
    const int64_t source_row_idx = static_cast<int64_t>(dst2src_map[dest_row_idx]);

    const auto* source_row_ptr = reinterpret_cast<const DataElem*>(input + source_row_idx * num_cols);
    auto* dest_row_ptr = reinterpret_cast<DataElem*>(output + dest_row_idx * num_cols);

    const int64_t start_offset = static_cast<int64_t>(item.get_local_id(0));
    const int64_t stride = static_cast<int64_t>(item.get_local_range(0));

    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      dest_row_ptr[elem_index] = source_row_ptr[elem_index];
    }
  }
};

template <typename T>
void shuffle_rows_kernel_impl(
    const torch::Tensor& input_tensor, const torch::Tensor& dst2src_map, torch::Tensor& output_tensor) {
  auto input = reinterpret_cast<T*>(input_tensor.data_ptr());
  auto dst2srcmap = reinterpret_cast<const int32_t*>(dst2src_map.data_ptr());
  auto output = reinterpret_cast<T*>(output_tensor.data_ptr());
  uint32_t num_dest_rows = output_tensor.size(0);
  uint32_t num_cols = input_tensor.size(1);

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  auto dev_id = input_tensor.device().index();
  uint32_t max_wg_size = static_cast<uint32_t>(dpcppMaxWorkGroupSize(dev_id));
  uint32_t threads_per_block = std::min(max_wg_size, 256u);

  using Kernel = ShuffleRowsKernel<T>;
  sycl::range<1> global_range{static_cast<size_t>(num_dest_rows) * threads_per_block};
  sycl::range<1> local_range{threads_per_block};

  Kernel task(input, dst2srcmap, output, num_cols);

  sycl_kernel_submit(global_range, local_range, queue, task);
}

void shuffle_rows(const torch::Tensor& input_tensor, const torch::Tensor& dst2src_map, torch::Tensor& output_tensor) {
  TORCH_CHECK(
      input_tensor.scalar_type() == output_tensor.scalar_type(),
      "Input and output tensors must have the same data type");
  TORCH_CHECK(dst2src_map.scalar_type() == at::kInt, "dst2src_map must have dtype int32");
  TORCH_CHECK(
      input_tensor.is_contiguous() && output_tensor.is_contiguous() && dst2src_map.is_contiguous(),
      "input, output, and dst2src_map must all be contiguous");
  TORCH_CHECK(
      input_tensor.size(1) % (16 / input_tensor.element_size()) == 0,
      "num_cols must be divisible by 16/sizeof(dtype) for aligned vectorized shuffle");
  SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, input_tensor.scalar_type(), "shuffle_rows_kernel_impl", [&]() {
        shuffle_rows_kernel_impl<scalar_t>(input_tensor, dst2src_map, output_tensor);
      });
  return;
}

template <typename T, typename T1, uint32_t VecSize>
struct ApplyShuffleMulSum {
  using LoadVec = VecArray<T, VecSize>;

  ApplyShuffleMulSum(
      const T* input, T* output, const int32_t* permutation, const T1* factors, int m, int topk, int row_stride)
      : input_(input),
        output_(output),
        permutation_(permutation),
        factors_(factors),
        m_(m),
        topk_(topk),
        row_stride_(row_stride) {}

  void operator()(sycl::nd_item<1> item) const {
    const int i = static_cast<int>(item.get_group(0));  // token index
    const int thread_idx = static_cast<int>(item.get_local_id(0));
    const int stride = static_cast<int>(item.get_local_range(0));

    if (i >= m_) return;

    const int vec_count = row_stride_ / static_cast<int>(VecSize);

    for (int d_vec_idx = thread_idx; d_vec_idx < vec_count; d_vec_idx += stride) {
      const int d = d_vec_idx * static_cast<int>(VecSize);

      float sum[VecSize] = {};

      for (int j = 0; j < topk_; ++j) {
        const int token_major_idx = i * topk_ + j;
        const int src_row = permutation_[token_major_idx];

        LoadVec raw;
        raw = *reinterpret_cast<const LoadVec*>(input_ + src_row * row_stride_ + d);

        float val[VecSize];
        vec_cast_to_float(raw, val);

        const float factor = (factors_ != nullptr) ? static_cast<float>(factors_[token_major_idx]) : 1.0f;

#pragma unroll
        for (uint32_t k = 0; k < VecSize; ++k)
          sum[k] += factor * val[k];
      }

      LoadVec out_vec;
      vec_cast_from_float(sum, out_vec);
      *reinterpret_cast<LoadVec*>(output_ + i * row_stride_ + d) = out_vec;
    }

    // tail processing
    const int remainder_start = vec_count * static_cast<int>(VecSize);
    for (int d = remainder_start + thread_idx; d < row_stride_; d += stride) {
      float sum_val = 0.0f;
      for (int j = 0; j < topk_; ++j) {
        const int token_major_idx = i * topk_ + j;
        const int src_row = permutation_[token_major_idx];

        const float val = static_cast<float>(input_[src_row * row_stride_ + d]);
        const float factor = (factors_ != nullptr) ? static_cast<float>(factors_[token_major_idx]) : 1.0f;

        sum_val += factor * val;
      }
      output_[i * row_stride_ + d] = static_cast<T>(sum_val);
    }
  }

  const T* input_;              // [m * topk, row_stride]
  T* output_;                   // [m, row_stride]
  const int32_t* permutation_;  // [m * topk]
  const T1* factors_;           // [m * topk], nullable
  int m_;
  int topk_;
  int row_stride_;
};

template <typename T, typename T1>
void apply_shuffle_mul_sum_impl(
    const torch::Tensor& input_tensor,
    torch::Tensor& output_tensor,
    const torch::Tensor& permutation_tensor,
    const std::optional<torch::Tensor>& factors_tensor) {
  auto input = reinterpret_cast<T*>(input_tensor.data_ptr());
  auto permutation = reinterpret_cast<const int32_t*>(permutation_tensor.data_ptr());
  auto output = reinterpret_cast<T*>(output_tensor.data_ptr());
  T1* factors = nullptr;
  if (factors_tensor.has_value()) factors = reinterpret_cast<T1*>(factors_tensor->data_ptr());

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  auto dev_id = input_tensor.device().index();
  uint32_t max_wg_size = static_cast<uint32_t>(dpcppMaxWorkGroupSize(dev_id));

  constexpr uint32_t VecSize = 16u / sizeof(T);  // 128-bit

  int m = output_tensor.size(0);
  int topk = int(permutation_tensor.size(0) / m);
  int row_stride = output_tensor.size(1);
  const uint32_t vec_count = static_cast<uint32_t>(row_stride) / VecSize;
  const uint32_t local_size = std::max(1u, sycl::min(max_wg_size, vec_count));

  sycl::range<1> global_range{static_cast<size_t>(m) * local_size};
  sycl::range<1> local_range{local_size};

  ApplyShuffleMulSum<T, T1, VecSize> task(input, output, permutation, factors, m, topk, row_stride);

  sycl_kernel_submit(global_range, local_range, queue, task);
}

void apply_shuffle_mul_sum(
    const torch::Tensor& input,
    torch::Tensor& output,
    const torch::Tensor& permutation,
    const std::optional<torch::Tensor>& factors) {
  TORCH_CHECK(input.dim() == 2, "input_tensor must be 2D [m * topk, row_stride]");
  TORCH_CHECK(output.dim() == 2, "output_tensor must be 2D [m, row_stride]");
  TORCH_CHECK(permutation.dim() == 1, "permutation must be 1D [m * topk]");

  // Validate dtypes to match the assumptions in apply_shuffle_mul_sum_impl.
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "input and output must have the same dtype");
  TORCH_CHECK(permutation.scalar_type() == at::kInt, "permutation must have dtype int32");

  // Validate contiguity / dense row-major layout assumptions.
  TORCH_CHECK(
      input.stride(1) == 1 && output.stride(1) == 1,
      "input and output must be contiguous in the last dimension (stride(1) == 1)");
  TORCH_CHECK(permutation.is_contiguous(), "permutation tensor must be contiguous");
  if (factors.has_value()) {
    TORCH_CHECK(factors->is_contiguous(), "factors tensor must be contiguous when provided");
  }

  // Validate shape relationships: input is [m * topk, row_stride],
  // output is [m, row_stride], permutation is [m * topk].
  TORCH_CHECK(input.size(1) == output.size(1), "input and output must have the same row_stride (size(1))");
  TORCH_CHECK(input.size(0) == permutation.numel(), "input.size(0) must equal permutation.numel() (m * topk)");

  SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "apply_shuffle_mul_sum", [&]() {
        using input_t = scalar_t;
        if (factors.has_value()) {
          SYCL_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::BFloat16, at::ScalarType::Half, factors->scalar_type(), "factors_dtype", [&]() {
                using factor_t = scalar_t;
                apply_shuffle_mul_sum_impl<input_t, factor_t>(input, output, permutation, factors);
              });
        } else {
          apply_shuffle_mul_sum_impl<input_t, float>(input, output, permutation, factors);
        }
      });
  return;
}
