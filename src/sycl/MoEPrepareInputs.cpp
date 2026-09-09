#include <ATen/ATen.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <limits>
#include <sycl/sycl.hpp>
#include <unordered_map>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

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

// Kernel 1: Per-WG SLM histogram → device atomic flush → per-element global offset
// SLM is dynamically sized at runtime based on num_experts (no compile-time limit)
template <typename T>
struct compute_arg_sorts_count_sycl_K_T : public __SYCL_KER_CONFIG_CONVENTION__ {
  compute_arg_sorts_count_sycl_K_T(
      const T* topk_ids,
      T* local_offsets,
      T* global_counts,
      const uint32_t topk_length,
      const uint32_t num_experts,
      const uint32_t wg_size)
      : topk_ids_(topk_ids),
        local_offsets_(local_offsets),
        global_counts_(global_counts),
        topk_length_(topk_length),
        num_experts_(num_experts),
        wg_size_(wg_size) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    // Dynamic SLM allocation based on actual num_experts at runtime
    slm_ = sycl::local_accessor<T, 1>(num_experts_, cgh);
  }

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    int gid = item.get_global_id(0);
    int lid = item.get_local_id(0);

    // ===== Phase 1: Zero SLM histogram =====
    for (int e = lid; e < (int)num_experts_; e += wg_size_)
      slm_[e] = static_cast<T>(0);
    sycl::group_barrier(item.get_group());

    // ===== Phase 2: Accumulate into SLM histogram, store per-element local offset =====
    if (gid < (int)topk_length_) {
      T expert = topk_ids_[gid];
      sycl::atomic_ref<
          T,
          sycl::memory_order::relaxed,
          sycl::memory_scope::work_group,
          sycl::access::address_space::local_space>
          local_cnt(slm_[expert]);
      T local_old = local_cnt.fetch_add(static_cast<T>(1));
      // Store this element's offset within its WG's expert bucket
      local_offsets_[gid] = local_old;
    }
    sycl::group_barrier(item.get_group());

    // ===== Phase 3: Flush SLM histogram to global counts =====
    // Store this WG's base offset back into SLM for Phase 4
    for (int e = lid; e < (int)num_experts_; e += wg_size_) {
      T count = slm_[e];
      if (count > static_cast<T>(0)) {
        sycl::atomic_ref<
            T,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>
            g_cnt(global_counts_[e]);
        T wg_start = g_cnt.fetch_add(count);
        slm_[e] = wg_start;
      } else {
        slm_[e] = static_cast<T>(0);
      }
    }
    sycl::group_barrier(item.get_group());

    // ===== Phase 4: Fix per-element offset to be global =====
    // local_offsets[gid] += this WG's base for that expert
    if (gid < (int)topk_length_) {
      T expert = topk_ids_[gid];
      local_offsets_[gid] += slm_[expert];
    }
  }

  const T* topk_ids_;
  T* local_offsets_;
  T* global_counts_;
  const uint32_t topk_length_;
  const uint32_t num_experts_;
  const uint32_t wg_size_;
  mutable sycl::local_accessor<T, 1> slm_;
};

// Kernel 2: Simple scatter using precomputed per-element offsets
// No SLM, no atomics — pure gather/scatter
template <typename T>
struct compute_arg_sorts_scatter_sycl_K_T {
  compute_arg_sorts_scatter_sycl_K_T(
      const T* topk_ids,
      const T* local_offsets,
      const T* expert_base_offsets,
      T* input_permutation,
      T* output_permutation,
      const uint32_t topk_length,
      const uint32_t topk)
      : topk_ids_(topk_ids),
        local_offsets_(local_offsets),
        expert_base_offsets_(expert_base_offsets),
        input_permutation_(input_permutation),
        output_permutation_(output_permutation),
        topk_length_(topk_length),
        topk_(topk) {}

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    int gid = item.get_global_id(0);

    if (gid < (int)topk_length_) {
      T expert = topk_ids_[gid];
      // Final position = expert's base (from prefix-sum) + this element's global offset
      T pos = expert_base_offsets_[expert] + local_offsets_[gid];

      input_permutation_[pos] = static_cast<T>(gid / topk_);
      output_permutation_[gid] = pos;
    }
  }

  const T* topk_ids_;
  const T* local_offsets_;
  const T* expert_base_offsets_;
  T* input_permutation_;
  T* output_permutation_;
  const uint32_t topk_length_;
  const uint32_t topk_;
};

template <typename T>
void compute_arg_sorts_sycl_impl(
    const torch::Tensor& topk_ids,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    torch::Tensor& atomic_buffer,  // holds expert prefix-sum offsets from compute_expert_offsets
    const uint32_t num_experts) {
  // Guard dynamic SLM allocation based on the histogram element size.
  // Each expert uses sizeof(T) bytes in the SLM histogram.
  constexpr uint32_t MAX_EXPERTS_SLM = 1024;
  constexpr uint32_t SLM_BYTES_PER_WG = 64 * 1024;  // Xe2 (B60) SLM per work-group
  const uint32_t max_experts_slm = SLM_BYTES_PER_WG / static_cast<uint32_t>(sizeof(T));
  TORCH_CHECK(
      num_experts <= max_experts_slm,
      "compute_arg_sorts: num_experts (",
      num_experts,
      ") exceeds SLM capacity (",
      max_experts_slm,
      " experts for sizeof(T)=",
      sizeof(T),
      ")");

  const T* topk_ids_ptr = static_cast<const T*>(topk_ids.data_ptr());
  T* input_permutation_ptr = static_cast<T*>(input_permutation.data_ptr());
  T* output_permutation_ptr = static_cast<T*>(output_permutation.data_ptr());

  const uint32_t topk_length = topk_ids.numel();
  const uint32_t topk = static_cast<uint32_t>(topk_ids.size(1));
  auto dev_id = topk_ids.device().index();

  uint32_t wg_size = static_cast<uint32_t>(std::min((uint32_t)dpcppMaxWorkGroupSize(dev_id), topk_length));
  uint32_t num_wgs = (topk_length + wg_size - 1) / wg_size;

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  sycl::range<1> global_range{(size_t)num_wgs * wg_size};
  sycl::range<1> local_range{wg_size};

// Note: this value is tuned for BMG. For decode step with less tokens, the old kernel is better. We need to tune
// this value for other hardwares in the future.
#define THRESHOLD 768
  if (topk_length < THRESHOLD) {
    T* atomic_buffer_ptr = static_cast<T*>(atomic_buffer.data_ptr());
    using Kernel = compute_arg_sorts_sycl_K_T<T>;
    Kernel task(topk_ids_ptr, input_permutation_ptr, output_permutation_ptr, atomic_buffer_ptr, topk_length, topk);
    sycl_kernel_submit(global_range, local_range, queue, task);

  } else {
    T* expert_base_offsets_ptr = static_cast<T*>(atomic_buffer.data_ptr());
    auto options = torch::TensorOptions().dtype(topk_ids.dtype()).device(topk_ids.device());

    // Per-element global offset buffer
    torch::Tensor local_offsets = torch::empty({(int64_t)topk_length}, options);
    // Device-wide running count per expert (must start at 0)
    torch::Tensor global_counts = torch::zeros({(int64_t)num_experts}, options);

    T* local_offsets_ptr = static_cast<T*>(local_offsets.data_ptr());
    T* global_counts_ptr = static_cast<T*>(global_counts.data_ptr());

    // Kernel 1: SLM histogram per WG → device atomic flush → per-element global offset
    // sycl_ker_config_convention allocates SLM dynamically based on num_experts
    using K1 = compute_arg_sorts_count_sycl_K_T<T>;
    K1 k1(topk_ids_ptr, local_offsets_ptr, global_counts_ptr, topk_length, num_experts, wg_size);
    sycl_kernel_submit(global_range, local_range, queue, k1);

    // Kernel 2: Simple scatter — no atomics, no SLM
    // Implicit in-order queue ordering guarantees kernel 1 is complete
    using K2 = compute_arg_sorts_scatter_sycl_K_T<T>;
    K2 k2(
        topk_ids_ptr,
        local_offsets_ptr,
        expert_base_offsets_ptr,
        input_permutation_ptr,
        output_permutation_ptr,
        topk_length,
        topk);
    sycl_kernel_submit(global_range, local_range, queue, k2);
  }
#undef THRESHOLD
}

SGL_KERNEL_EXPORT void prepare_moe_input(
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

template <typename IndexType, typename ScalarT>
struct PrepareMoeInputSmall : public __SYCL_KER_CONFIG_CONVENTION__ {
  // TODO: Add benchmarked WG/max-route specializations when this path is
  // enabled beyond BMG; select among static variants using device capabilities.
  static constexpr int WGSize = 256;
  static constexpr int MaxRoutes = 64;
  static constexpr int ElementsPerVector = 8;
  static constexpr int RequiredSubGroupSize = 16;

  static_assert(WGSize % RequiredSubGroupSize == 0);
  static_assert(MaxRoutes <= WGSize);
  static_assert(ElementsPerVector * sizeof(ScalarT) == 16);

  PrepareMoeInputSmall(
      const ScalarT* input,
      const IndexType* topk_ids,
      int32_t* expert_counts,
      int32_t* output_permutation,
      ScalarT* output,
      int32_t num_experts,
      int32_t input_rows,
      int32_t topk,
      int32_t hidden_dim)
      : input_(input),
        topk_ids_(topk_ids),
        expert_counts_(expert_counts),
        output_permutation_(output_permutation),
        output_(output),
        num_experts_(num_experts),
        input_rows_(input_rows),
        topk_(topk),
        hidden_dim_(hidden_dim) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    route_positions_ = sycl::local_accessor<int32_t, 1>(MaxRoutes, cgh);
    local_counts_ = sycl::local_accessor<int32_t, 1>(num_experts_, cgh);
  }

  [[sycl::reqd_sub_group_size(RequiredSubGroupSize)]] void operator()(sycl::nd_item<1> item) const {
    int local_id = item.get_local_linear_id();
    for (int expert = local_id; expert < num_experts_; expert += WGSize) {
      expert_counts_[expert] = 0;
      local_counts_[expert] = 0;
    }
    sycl::group_barrier(item.get_group());

    if (input_rows_ == 1) {
      if (local_id == 0) {
        int32_t order[16];
        for (int rank = 0; rank < topk_; ++rank) {
          order[rank] = rank;
        }
        for (int rank = 1; rank < topk_; ++rank) {
          int32_t current = order[rank];
          int insert_at = rank;
          while (insert_at > 0 && topk_ids_[order[insert_at - 1]] > topk_ids_[current]) {
            order[insert_at] = order[insert_at - 1];
            --insert_at;
          }
          order[insert_at] = current;
        }
        for (int destination = 0; destination < topk_; ++destination) {
          int32_t route = order[destination];
          ++expert_counts_[static_cast<int32_t>(topk_ids_[route])];
          route_positions_[route] = destination;
          output_permutation_[route] = destination;
        }
      }
      sycl::group_barrier(item.get_group());

      using Vector = sycl::vec<uint16_t, ElementsPerVector>;
      auto input_vectors = reinterpret_cast<const Vector*>(input_);
      auto output_vectors = reinterpret_cast<Vector*>(output_);
      int vector_count = hidden_dim_ % ElementsPerVector == 0 ? hidden_dim_ / ElementsPerVector : 0;
      for (int vector_id = local_id; vector_id < vector_count; vector_id += WGSize) {
        Vector value = input_vectors[vector_id];
        for (int route = 0; route < topk_; ++route) {
          output_vectors[route_positions_[route] * vector_count + vector_id] = value;
        }
      }
      for (int column = vector_count * ElementsPerVector + local_id; column < hidden_dim_; column += WGSize) {
        ScalarT value = input_[column];
        for (int route = 0; route < topk_; ++route) {
          output_[route_positions_[route] * hidden_dim_ + column] = value;
        }
      }
      return;
    }

    int route_count = topk_ * input_rows_;
    if (local_id < route_count) {
      int32_t expert = static_cast<int32_t>(topk_ids_[local_id]);
      sycl::atomic_ref<
          int32_t,
          sycl::memory_order::relaxed,
          sycl::memory_scope::work_group,
          sycl::access::address_space::local_space>
          count(local_counts_[expert]);
      count.fetch_add(1);
    }
    sycl::group_barrier(item.get_group());

    if (local_id == 0) {
      int32_t offset = 0;
      for (int expert = 0; expert < num_experts_; ++expert) {
        int32_t count = local_counts_[expert];
        expert_counts_[expert] = count;
        local_counts_[expert] = offset;
        offset += count;
      }
    }
    sycl::group_barrier(item.get_group());

    if (local_id < route_count) {
      int32_t expert = static_cast<int32_t>(topk_ids_[local_id]);
      int32_t destination = local_counts_[expert];
      for (int route = 0; route < local_id; ++route) {
        destination += static_cast<int32_t>(topk_ids_[route]) == expert;
      }
      route_positions_[local_id] = destination;
      output_permutation_[local_id] = destination;
    }
    sycl::group_barrier(item.get_group());

    using Vector = sycl::vec<uint16_t, ElementsPerVector>;
    auto input_vectors = reinterpret_cast<const Vector*>(input_);
    auto output_vectors = reinterpret_cast<Vector*>(output_);
    int vector_count = hidden_dim_ % ElementsPerVector == 0 ? hidden_dim_ / ElementsPerVector : 0;
    int vector_tasks = route_count * vector_count;
    for (int task = local_id; task < vector_tasks; task += WGSize) {
      int route = task / vector_count;
      int vector_id = task % vector_count;
      int source_row = route / topk_;
      output_vectors[route_positions_[route] * vector_count + vector_id] =
          input_vectors[source_row * vector_count + vector_id];
    }
    int tail_start = vector_count * ElementsPerVector;
    int tail_size = hidden_dim_ - tail_start;
    int tail_tasks = route_count * tail_size;
    for (int task = local_id; task < tail_tasks; task += WGSize) {
      int route = task / tail_size;
      int column = tail_start + task % tail_size;
      int source_row = route / topk_;
      output_[route_positions_[route] * hidden_dim_ + column] = input_[source_row * hidden_dim_ + column];
    }
  }

  const ScalarT* input_;
  const IndexType* topk_ids_;
  int32_t* expert_counts_;
  int32_t* output_permutation_;
  ScalarT* output_;
  int32_t num_experts_;
  int32_t input_rows_;
  int32_t topk_;
  int32_t hidden_dim_;
  mutable sycl::local_accessor<int32_t, 1> route_positions_;
  mutable sycl::local_accessor<int32_t, 1> local_counts_;
};

template <typename Kernel>
size_t prepare_moe_input_small_local_memory_capacity(at::DeviceIndex device_index) {
  static thread_local std::unordered_map<at::DeviceIndex, size_t> capacity_by_device;
  auto cached = capacity_by_device.find(device_index);
  if (cached != capacity_by_device.end()) {
    return cached->second;
  }

  auto* properties = at::xpu::getDeviceProperties(device_index);
  TORCH_CHECK(
      std::find(properties->sub_group_sizes.begin(), properties->sub_group_sizes.end(), Kernel::RequiredSubGroupSize) !=
          properties->sub_group_sizes.end(),
      "prepare_moe_input_small requires subgroup size ",
      Kernel::RequiredSubGroupSize);
  TORCH_CHECK(
      dpcppMaxWorkGroupSize<Kernel>(device_index) >= Kernel::WGSize,
      "prepare_moe_input_small requires work-group size ",
      Kernel::WGSize);
  return capacity_by_device.emplace(device_index, properties->local_mem_size).first->second;
}

SGL_KERNEL_EXPORT void prepare_moe_input_small(
    const torch::Tensor& input,
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_counts,
    torch::Tensor& output_permutation,
    torch::Tensor& output) {
  TORCH_CHECK(
      input.is_xpu() && input.dim() == 2 && input.is_contiguous(), "input must be contiguous XPU [rows, hidden_dim]");
  TORCH_CHECK(
      topk_ids.is_xpu() && topk_ids.dim() == 2 && topk_ids.is_contiguous() && topk_ids.size(0) == input.size(0),
      "topk_ids must be contiguous XPU [rows, topk] with rows matching input");
  TORCH_CHECK(topk_ids.size(1) > 0 && topk_ids.size(1) <= 16, "topk must be in [1, 16]");
  TORCH_CHECK(
      topk_ids.numel() <= (PrepareMoeInputSmall<int32_t, c10::BFloat16>::MaxRoutes), "routed rows must be <= 64");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
  TORCH_CHECK(
      expert_counts.is_xpu() && expert_counts.dim() == 1 && expert_counts.numel() > 0 && expert_counts.is_contiguous(),
      "expert_counts must be a non-empty contiguous XPU vector");
  TORCH_CHECK(
      output_permutation.is_xpu() && output_permutation.dim() == 1 && output_permutation.is_contiguous(),
      "output_permutation must be a contiguous XPU vector");
  TORCH_CHECK(
      output.is_xpu() && output.dim() == 2 && output.is_contiguous(),
      "output must be contiguous XPU [routes, hidden_dim]");
  TORCH_CHECK(
      input.device() == topk_ids.device() && input.device() == expert_counts.device() &&
          input.device() == output_permutation.device() && input.device() == output.device(),
      "all tensors must be on the same device");
  TORCH_CHECK(expert_counts.scalar_type() == at::kInt, "expert_counts must be int32");
  TORCH_CHECK(output_permutation.scalar_type() == at::kInt, "output_permutation must be int32");
  TORCH_CHECK(output.scalar_type() == input.scalar_type(), "output dtype must match input");
  TORCH_CHECK(output.size(0) == topk_ids.numel() && output.size(1) == input.size(1), "output shape mismatch");
  TORCH_CHECK(output_permutation.numel() == topk_ids.numel(), "output_permutation shape mismatch");
  TORCH_CHECK(
      expert_counts.numel() <= std::numeric_limits<int32_t>::max() &&
          input.size(0) <= std::numeric_limits<int32_t>::max() && input.size(1) <= std::numeric_limits<int32_t>::max(),
      "expert count and input dimensions must fit in int32");

  auto queue = at::xpu::getCurrentXPUStream().queue();
  AT_DISPATCH_INDEX_TYPES(topk_ids.scalar_type(), "prepare_moe_input_small", [&] {
    using Kernel = PrepareMoeInputSmall<index_t, c10::BFloat16>;
    const size_t local_memory_capacity = prepare_moe_input_small_local_memory_capacity<Kernel>(input.device().index());
    const size_t required_local_memory =
        (static_cast<size_t>(Kernel::MaxRoutes) + static_cast<size_t>(expert_counts.numel())) * sizeof(int32_t);
    TORCH_CHECK(
        required_local_memory <= local_memory_capacity,
        "prepare_moe_input_small requires ",
        required_local_memory,
        " bytes of local memory");
    Kernel task(
        input.const_data_ptr<c10::BFloat16>(),
        topk_ids.const_data_ptr<index_t>(),
        expert_counts.mutable_data_ptr<int32_t>(),
        output_permutation.mutable_data_ptr<int32_t>(),
        output.mutable_data_ptr<c10::BFloat16>(),
        static_cast<int32_t>(expert_counts.numel()),
        static_cast<int32_t>(input.size(0)),
        static_cast<int32_t>(topk_ids.size(1)),
        static_cast<int32_t>(input.size(1)));
    sycl_kernel_submit(Kernel::WGSize, Kernel::WGSize, queue, task);
  });
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

SGL_KERNEL_EXPORT void scatter_tokens_to_experts(
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
      weights[k] = (factors_ != nullptr) ? static_cast<float>(factors_[out_tkn_id * topk_ + k]) : 1.0f;
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

SGL_KERNEL_EXPORT void apply_shuffle_mul_sum(
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
