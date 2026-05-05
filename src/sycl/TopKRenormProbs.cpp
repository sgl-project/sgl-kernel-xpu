#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdlib>
#include <optional>
#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"

//----------------- set element type options --------------------//

template <typename T>
struct ToSyclElementType {
  using type = T;
};

template <>
struct ToSyclElementType<at::Half> {
  using type = sycl::half;
};

template <>
struct ToSyclElementType<at::BFloat16> {
  using type = sycl::ext::oneapi::bfloat16;
};

//----------------- radix traits for bit-pattern transformation --------------------//

template <typename T>
struct RadixTopKTraits;

template <>
struct RadixTopKTraits<float> {
  using DType = float;
  using OrderedType = uint32_t;
  static constexpr uint32_t kRadixBits = 8;
  static constexpr uint32_t kRadix = 1u << kRadixBits;
  static constexpr uint32_t kNumRounds = 4;

  static inline OrderedType to_ordered(float val) {
    uint32_t bits = sycl::bit_cast<uint32_t>(val);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  }

  static inline float from_ordered(OrderedType ordered) {
    uint32_t bits = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
    return sycl::bit_cast<float>(bits);
  }
};

template <>
struct RadixTopKTraits<sycl::half> {
  using DType = sycl::half;
  using OrderedType = uint16_t;
  static constexpr uint32_t kRadixBits = 8;
  static constexpr uint32_t kRadix = 256;
  static constexpr uint32_t kNumRounds = 2;

  static inline OrderedType to_ordered(sycl::half val) {
    uint16_t bits = sycl::bit_cast<uint16_t>(val);
    return (bits & 0x8000u) ? ~bits : (bits ^ 0x8000u);
  }

  static inline sycl::half from_ordered(OrderedType ordered) {
    uint16_t bits = (ordered & 0x8000u) ? (ordered ^ 0x8000u) : ~ordered;
    return sycl::bit_cast<sycl::half>(bits);
  }
};

template <>
struct RadixTopKTraits<sycl::ext::oneapi::bfloat16> {
  using DType = sycl::ext::oneapi::bfloat16;
  using OrderedType = uint16_t;
  static constexpr uint32_t kRadixBits = 8;
  static constexpr uint32_t kRadix = 256;
  static constexpr uint32_t kNumRounds = 2;

  static inline OrderedType to_ordered(sycl::ext::oneapi::bfloat16 val) {
    uint16_t bits = sycl::bit_cast<uint16_t>(val);
    return (bits & 0x8000u) ? ~bits : (bits ^ 0x8000u);
  }

  static inline sycl::ext::oneapi::bfloat16 from_ordered(OrderedType ordered) {
    uint16_t bits = (ordered & 0x8000u) ? (ordered ^ 0x8000u) : ~ordered;
    return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(bits);
  }
};

//----------------- single-cta kernel implementation --------------------//

template <typename DType>
struct TopKRenormProbsSingleCTA : public __SYCL_KER_CONFIG_CONVENTION__ {
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;
  static constexpr uint32_t kOrderedBits = sizeof(OrderedType) * 8;
  static constexpr uint32_t kWgSize = 1024;
  static constexpr uint32_t kRadix = Traits::kRadix;
  static constexpr uint32_t kNumRounds = Traits::kNumRounds;
  static constexpr uint32_t kNumSubGroups = kWgSize / 32;

  const DType* probs;
  DType* renorm_probs;
  const int64_t* maybe_top_k_arr;
  int top_k_val;
  int batch_size;
  int vocab_size;

  sycl::local_accessor<uint32_t, 1> subgroup_hist_;
  sycl::local_accessor<uint32_t, 1> scalars_;
  sycl::local_accessor<float, 1> block_sum_;

  void sycl_ker_config_convention(sycl::handler& cgh) {
    subgroup_hist_ = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(kNumSubGroups * kRadix), cgh);
    scalars_ = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(2), cgh);
    block_sum_ = sycl::local_accessor<float, 1>(sycl::range<1>(1), cgh);
  }

  TopKRenormProbsSingleCTA(
      const DType* probs,
      DType* renorm_probs,
      const int64_t* maybe_top_k_arr,
      int top_k_val,
      int batch_size,
      int vocab_size)
      : probs(probs),
        renorm_probs(renorm_probs),
        maybe_top_k_arr(maybe_top_k_arr),
        top_k_val(top_k_val),
        batch_size(batch_size),
        vocab_size(vocab_size) {}

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t row_idx = item.get_group(0);
    if (row_idx >= static_cast<uint32_t>(batch_size)) return;

    const uint32_t tid = item.get_local_id(0);
    const uint32_t vocab_u32 = static_cast<uint32_t>(vocab_size);

    uint32_t k = maybe_top_k_arr ? static_cast<uint32_t>(maybe_top_k_arr[row_idx]) : static_cast<uint32_t>(top_k_val);
    if (k > vocab_u32) k = vocab_u32;

    const size_t row_offset = static_cast<size_t>(row_idx) * static_cast<size_t>(vocab_u32);
    sycl::sub_group sg = item.get_sub_group();
    const uint32_t sg_id = sg.get_group_id()[0];
    const uint32_t sg_lid = sg.get_local_id()[0];

    if (tid == 0) {
      scalars_[0] = 0u;
      scalars_[1] = k;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (uint32_t round = 0u; round < kNumRounds; ++round) {
      const uint32_t prefix = scalars_[0];
      const uint32_t remaining_k = scalars_[1];
      const uint32_t shift = kOrderedBits - (round + 1u) * Traits::kRadixBits;

      const OrderedType prefix_mask =
          (round == 0) ? OrderedType(0)
                       : static_cast<OrderedType>(~OrderedType(0) << (kOrderedBits - round * Traits::kRadixBits));

      for (uint32_t i = tid; i < kNumSubGroups * kRadix; i += kWgSize) {
        subgroup_hist_[i] = 0u;
      }
      item.barrier(sycl::access::fence_space::local_space);

      const uint32_t hist_base = sg_id * kRadix;

      for (uint32_t col = tid; col < vocab_u32; col += kWgSize) {
        const OrderedType ordered = Traits::to_ordered(probs[row_offset + col]);
        if ((ordered & prefix_mask) == static_cast<OrderedType>(prefix)) {
          const uint32_t bucket = (ordered >> shift) & 0xFFu;
          sycl::atomic_ref<
              uint32_t,
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::local_space>(subgroup_hist_[hist_base + bucket])
              .fetch_add(1u);
        }
      }
      item.barrier(sycl::access::fence_space::local_space);

      for (uint32_t bin = tid; bin < kRadix; bin += kWgSize) {
        uint32_t total = 0u;
        for (uint32_t s = 0; s < kNumSubGroups; ++s) {
          total += subgroup_hist_[s * kRadix + bin];
        }
        subgroup_hist_[bin] = total;
      }
      item.barrier(sycl::access::fence_space::local_space);

      if (tid == 0) {
        uint32_t bucket = 0u;
        uint32_t suffix = 0u;
        for (int i = kRadix - 1; i >= 0; --i) {
          suffix += subgroup_hist_[i];
          if (suffix >= remaining_k) {
            bucket = static_cast<uint32_t>(i);
            break;
          }
        }
        const uint32_t count_above = suffix - subgroup_hist_[bucket];
        scalars_[0] = prefix | (bucket << shift);
        scalars_[1] = remaining_k - count_above;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    const float pivot = static_cast<float>(Traits::from_ordered(static_cast<OrderedType>(scalars_[0])));

    if (tid == 0) block_sum_[0] = 0.0f;
    item.barrier(sycl::access::fence_space::local_space);

    constexpr uint32_t kVecSize = 4;
    using vec_in = vec_t<DType, kVecSize>;
    const uint32_t num_vec_elems = vocab_u32 / kVecSize;
    const uint32_t vec_tail_start = num_vec_elems * kVecSize;

    float thread_sum = 0.0f;
    for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
      vec_in v;
      v.load(0, sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(
                    probs + row_offset + i * kVecSize));
      #pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        float val = static_cast<float>(v[j]);
        thread_sum += sycl::select(0.0f, val, val >= pivot);
      }
    }
    for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
      const float val = static_cast<float>(probs[row_offset + col]);
      thread_sum += sycl::select(0.0f, val, val >= pivot);
    }

    float warp_sum = sycl::reduce_over_group(sg, thread_sum, sycl::plus<float>());
    if (sg_lid == 0) {
      sycl::atomic_ref<
          float,
          sycl::memory_order::relaxed,
          sycl::memory_scope::work_group,
          sycl::access::address_space::local_space>(block_sum_[0])
          .fetch_add(warp_sum);
    }
    item.barrier(sycl::access::fence_space::local_space);

    const float inv_sum = (block_sum_[0] > 0.0f) ? (1.0f / block_sum_[0]) : 0.0f;

    using vec_out = vec_t<DType, kVecSize>;
    for (uint32_t i = tid; i < num_vec_elems; i += kWgSize) {
      vec_in v;
      v.load(0, sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(
                    probs + row_offset + i * kVecSize));
      vec_out out;
      #pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        float val = static_cast<float>(v[j]);
        out[j] = static_cast<DType>(sycl::select(0.0f, val * inv_sum, val >= pivot));
      }
      out.store(0, sycl::multi_ptr<DType, sycl::access::address_space::global_space>(
                       renorm_probs + row_offset + i * kVecSize));
    }
    for (uint32_t col = vec_tail_start + tid; col < vocab_u32; col += kWgSize) {
      const float val = static_cast<float>(probs[row_offset + col]);
      renorm_probs[row_offset + col] = static_cast<DType>(sycl::select(0.0f, val * inv_sum, val >= pivot));
    }
  }
};

template <typename TensorDType>
void launch_single_cta_kernel(
    at::Tensor probs,
    at::Tensor renorm_probs,
    const int64_t* maybe_top_k_ptr,
    int top_k_val,
    int batch_size,
    int vocab_size,
    sycl::queue& queue) {
  using KernelDType = typename ToSyclElementType<TensorDType>::type;

  const KernelDType* probs_ptr = reinterpret_cast<const KernelDType*>(probs.data_ptr<TensorDType>());
  auto* renorm_probs_ptr = reinterpret_cast<KernelDType*>(renorm_probs.data_ptr<TensorDType>());

  const int local_size = 1024;
  const int global_size = batch_size * local_size;

  auto kernel = TopKRenormProbsSingleCTA<KernelDType>(
      probs_ptr, renorm_probs_ptr, maybe_top_k_ptr, top_k_val, batch_size, vocab_size);

  sycl_kernel_submit(global_size, local_size, queue, kernel);
}

void top_k_renorm_probs(
    const at::Tensor& probs, at::Tensor& renorm_probs, const std::optional<at::Tensor>& maybe_top_k_arr, int64_t top_k_val) {
  CHECK_INPUT(probs);
  CHECK_INPUT(renorm_probs);
  TORCH_CHECK(probs.dim() == 2, "probs must be a 2D tensor [batch_size, vocab_size]");
  TORCH_CHECK(renorm_probs.dim() == 2, "renorm_probs must be a 2D tensor [batch_size, vocab_size]");
  TORCH_CHECK(probs.sizes() == renorm_probs.sizes(), "Input tensors must have the same shape");
  TORCH_CHECK(probs.scalar_type() == renorm_probs.scalar_type(), "Input tensors must have the same dtype");
  TORCH_CHECK(
      probs.scalar_type() == torch::kFloat32 || probs.scalar_type() == torch::kHalf ||
          probs.scalar_type() == torch::kBFloat16,
      "Input tensors must be float32, float16, or bfloat16");

  if (maybe_top_k_arr.has_value()) {
    CHECK_INPUT((*maybe_top_k_arr));
    TORCH_CHECK(maybe_top_k_arr->dim() == 1, "maybe_top_k_arr must be a 1D tensor [batch_size]");
    TORCH_CHECK(maybe_top_k_arr->size(0) == probs.size(0), "maybe_top_k_arr size must match batch_size");
    TORCH_CHECK(maybe_top_k_arr->scalar_type() == torch::kInt64, "maybe_top_k_arr must be int64");
  } else {
    TORCH_CHECK(top_k_val > 0, "top_k_val must be positive");
  }

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  int batch_size = probs.size(0);
  int vocab_size = probs.size(1);

  const int64_t* maybe_top_k_ptr = maybe_top_k_arr.has_value() ? maybe_top_k_arr->data_ptr<int64_t>() : nullptr;

  auto dtype = probs.scalar_type();

  if (dtype == torch::kFloat32) {
    launch_single_cta_kernel<float>(
        probs, renorm_probs, maybe_top_k_ptr, static_cast<int>(top_k_val), batch_size, vocab_size, queue);
  } else if (dtype == torch::kHalf) {
    launch_single_cta_kernel<at::Half>(
        probs, renorm_probs, maybe_top_k_ptr, static_cast<int>(top_k_val), batch_size, vocab_size, queue);
  } else if (dtype == torch::kBFloat16) {
    launch_single_cta_kernel<at::BFloat16>(
        probs, renorm_probs, maybe_top_k_ptr, static_cast<int>(top_k_val), batch_size, vocab_size, queue);
  }
}
