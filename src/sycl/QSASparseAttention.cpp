#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>

#include <cmath>
#include <cstdint>
#include <cute/tensor.hpp>
#include <optional>
#include <sycl/sycl.hpp>

#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;
using Bf16Dpas = cute::XE_DPAS_TT<1, float, cute::bfloat16_t>;
using DpasTraits = cute::MMA_Traits<Bf16Dpas>;
constexpr int kHeadDim = 256;
constexpr int kTileSize = 16;
constexpr int kShortSplitSize = 64;
constexpr int kLongSplitSize = 128;

template <int SplitSize>
struct QSASplitKernel {
  const bf16* q;
  const bf16* k;
  const bf16* v;
  const int32_t* slots;
  bf16* output;
  float* partial;
  float* stats;
  int rows;
  int heads;
  int kv_heads;
  int topk;
  int splits;
  float scale;

  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<1> item) const {
    const int group = item.get_group(0);
    const int split = group % splits;
    const int head = (group / splits) % heads;
    const int row = group / (splits * heads);
    const int kv_head = head / (heads / kv_heads);
    const int dim = item.get_local_id(0);
    const int token = dim / kTileSize;
    const int lane = dim % kTileSize;
    auto wg = item.get_group();
    float* scores = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kTileSize]>(wg);
    float* weights = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kTileSize]>(wg);
    int32_t* selected = *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[kTileSize]>(wg);
    float* state = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[3]>(wg);
    float* partial_scores = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kTileSize * kTileSize]>(wg);

    float numerator = 0.0f;
    if (dim == 0) {
      state[0] = -INFINITY;
      state[1] = 0.0f;
    }

    const int end = sycl::min(topk, (split + 1) * SplitSize);
    for (int start = split * SplitSize; start < end; start += kTileSize) {
      const int column = start + token;
      const int32_t slot = column < topk ? slots[row * topk + column] : -1;
      if (lane == 0) selected[token] = slot;
      // Publish selected slots before other subgroups gather K and V.
      item.barrier(sycl::access::fence_space::local_space);
      const int d = token * kTileSize;
      const int query_dim = DpasTraits::ALayout{}(lane, 0);
      const bf16 query_value = q[(row * heads + head) * kHeadDim + d + query_dim];
      Bf16Dpas::AVector query;
      __builtin_memcpy(&query, &query_value, sizeof(query_value));
      uint16_t key_values[kTileSize];
      for (int j = 0; j < kTileSize; ++j) {
        // The DPAS B fragment interleaves reduction dimensions and token columns.
        const int coordinate = DpasTraits::BLayout{}(lane, j);
        const int token_idx = coordinate % kTileSize;
        const int key_dim = coordinate / kTileSize;
        const int32_t key_slot = selected[token_idx];
        const int64_t offset = (static_cast<int64_t>(key_slot) * kv_heads + kv_head) * kHeadDim;
        key_values[j] = key_slot >= 0 ? sycl::bit_cast<uint16_t>(k[offset + d + key_dim]) : 0;
      }
      Bf16Dpas::BVector key;
      __builtin_memcpy(&key, key_values, sizeof(key_values));
      Bf16Dpas::DVector partial_score = 0.0f;
      Bf16Dpas::fma(partial_score, query, key, partial_score);
      partial_scores[token * kTileSize + DpasTraits::CLayout{}(lane, 0)] = partial_score[0];
      // Finish writing all subgroup QK partials before summing them into scores.
      item.barrier(sycl::access::fence_space::local_space);
      if (dim < kTileSize) {
        float score = 0.0f;
        for (int i = 0; i < kTileSize; ++i)
          score += partial_scores[i * kTileSize + dim];
        scores[dim] = selected[dim] >= 0 ? score * scale : -INFINITY;
      }
      // Publish every tile score before lane zero computes the softmax.
      item.barrier(sycl::access::fence_space::local_space);

      if (dim == 0) {
        float tile_max = -INFINITY;
        for (int t = 0; t < kTileSize; ++t)
          tile_max = sycl::fmax(tile_max, scores[t]);
        state[2] = 1.0f;
        if (tile_max != -INFINITY) {
          const float next_max = sycl::fmax(state[0], tile_max);
          state[2] = state[0] == -INFINITY ? 0.0f : sycl::exp(state[0] - next_max);
          float tile_sum = 0.0f;
          for (int t = 0; t < kTileSize; ++t) {
            weights[t] = scores[t] == -INFINITY ? 0.0f : sycl::exp(scores[t] - next_max);
            tile_sum += weights[t];
          }
          state[0] = next_max;
          state[1] = state[1] * state[2] + tile_sum;
        } else {
          for (int t = 0; t < kTileSize; ++t)
            weights[t] = 0.0f;
        }
      }
      // Publish weights and the rescaling factor before all lanes update P*V.
      item.barrier(sycl::access::fence_space::local_space);
      numerator *= state[2];
      const int prob_idx = DpasTraits::ALayout{}(lane, 0);
      const bf16 probability = static_cast<bf16>(weights[prob_idx]);
      Bf16Dpas::AVector prob_fragment;
      __builtin_memcpy(&prob_fragment, &probability, sizeof(probability));
      uint16_t value_fragments[kTileSize];
      for (int j = 0; j < kTileSize; ++j) {
        const int coordinate = DpasTraits::BLayout{}(lane, j);
        const int token_idx = coordinate / kTileSize;
        const int value_dim = token * kTileSize + coordinate % kTileSize;
        const int32_t value_slot = selected[token_idx];
        const int64_t offset = (static_cast<int64_t>(value_slot) * kv_heads + kv_head) * kHeadDim;
        value_fragments[j] = value_slot >= 0 ? sycl::bit_cast<uint16_t>(v[offset + value_dim]) : 0;
      }
      Bf16Dpas::BVector values;
      __builtin_memcpy(&values, value_fragments, sizeof(value_fragments));
      Bf16Dpas::DVector weighted_value = 0.0f;
      Bf16Dpas::fma(weighted_value, prob_fragment, values, weighted_value);
      numerator += weighted_value[0];
      // Finish reading shared slots and weights before the next tile overwrites them.
      if (start + kTileSize < end) item.barrier(sycl::access::fence_space::local_space);
    }

    if (splits == 1) {
      output[(row * heads + head) * kHeadDim + dim] = static_cast<bf16>(state[1] > 0.0f ? numerator / state[1] : 0.0f);
    } else {
      const int64_t index = static_cast<int64_t>(split) * rows * heads + row * heads + head;
      partial[index * kHeadDim + dim] = numerator;
      if (dim == 0) {
        stats[index * 2] = state[0];
        stats[index * 2 + 1] = state[1];
      }
    }
  }
};

struct QSAMergeKernel {
  const float* partial;
  const float* stats;
  bf16* output;
  int heads;
  int rows;
  int splits;

  void operator()(sycl::nd_item<1> item) const {
    const int group = item.get_group(0);
    const int dim = item.get_local_id(0);
    float maximum = -INFINITY;
    for (int split = 0; split < splits; ++split) {
      const int64_t index = static_cast<int64_t>(split) * rows * heads + group;
      maximum = sycl::fmax(maximum, stats[index * 2]);
    }
    float sum = 0.0f;
    float numerator = 0.0f;
    if (maximum != -INFINITY) {
      for (int split = 0; split < splits; ++split) {
        const int64_t index = static_cast<int64_t>(split) * rows * heads + group;
        if (stats[index * 2 + 1] > 0.0f) {
          const float weight = sycl::exp(stats[index * 2] - maximum);
          sum += weight * stats[index * 2 + 1];
          numerator += weight * partial[index * kHeadDim + dim];
        }
      }
    }
    output[group * kHeadDim + dim] = static_cast<bf16>(sum > 0.0f ? numerator / sum : 0.0f);
  }
};

}  // namespace

SGL_KERNEL_EXPORT at::Tensor qsa_sparse_attention(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& token_slots,
    std::optional<double> softmax_scale) {
  CHECK_INPUT(q);
  CHECK_INPUT(k_cache);
  CHECK_INPUT(v_cache);
  CHECK_INPUT(token_slots);
  TORCH_CHECK(
      q.device() == k_cache.device() && q.device() == v_cache.device() && q.device() == token_slots.device(),
      "QSA tensors must be on the same XPU device");
  TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 3 && v_cache.dim() == 3, "QSA Q/K/V must be rank-3");
  TORCH_CHECK(
      token_slots.dim() == 2 && token_slots.size(0) == q.size(0),
      "QSA token_slots must have shape [query_rows, selected_tokens]");
  TORCH_CHECK(
      q.scalar_type() == at::kBFloat16 && k_cache.scalar_type() == at::kBFloat16 &&
          v_cache.scalar_type() == at::kBFloat16,
      "QSA only supports BF16 Q/K/V");
  TORCH_CHECK(token_slots.scalar_type() == at::kInt, "QSA token_slots must be int32");
  TORCH_CHECK(
      q.size(2) == kHeadDim && k_cache.size(2) == kHeadDim && v_cache.size(2) == kHeadDim,
      "QSA head dimension must be 256");
  TORCH_CHECK(
      q.size(1) > 0 && k_cache.size(1) > 0 && q.size(1) % k_cache.size(1) == 0 && k_cache.sizes() == v_cache.sizes(),
      "QSA requires matching K/V caches and query heads divisible by KV heads");
  const double scale = softmax_scale.value_or(0.0) == 0.0 ? 1.0 / std::sqrt(kHeadDim) : *softmax_scale;
  TORCH_CHECK(std::isfinite(scale), "QSA softmax scale must be finite");

  c10::DeviceGuard guard(q.device());
  auto out = at::empty_like(q);
  const int rows = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  const int kv_heads = static_cast<int>(k_cache.size(1));
  const int topk = static_cast<int>(token_slots.size(1));
  if (rows == 0) return out;
  if (topk == 0) return at::zeros_like(q);
  const int split_size = topk <= 512 ? kShortSplitSize : kLongSplitSize;
  const int splits = (topk + split_size - 1) / split_size;
  at::Tensor partial, stats;
  if (splits > 1) {
    partial = at::empty({splits, rows, heads, kHeadDim}, q.options().dtype(at::kFloat));
    stats = at::empty({splits, rows, heads, 2}, q.options().dtype(at::kFloat));
  }

  auto& queue = at::xpu::getCurrentXPUStream().queue();
  auto launch_split = [&]<int SplitSize>() {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(static_cast<size_t>(rows) * heads * splits * kHeadDim, kHeadDim),
          QSASplitKernel<SplitSize>{
              reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(k_cache.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(v_cache.data_ptr<at::BFloat16>()),
              token_slots.data_ptr<int32_t>(),
              reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
              splits > 1 ? partial.data_ptr<float>() : nullptr,
              splits > 1 ? stats.data_ptr<float>() : nullptr,
              rows,
              heads,
              kv_heads,
              topk,
              splits,
              static_cast<float>(scale)});
    });
  };
  if (topk <= 512) {
    launch_split.template operator()<kShortSplitSize>();
  } else {
    launch_split.template operator()<kLongSplitSize>();
  }
  if (splits > 1) {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(static_cast<size_t>(rows) * heads * kHeadDim, kHeadDim),
          QSAMergeKernel{
              partial.data_ptr<float>(),
              stats.data_ptr<float>(),
              reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
              heads,
              rows,
              splits});
    });
  }
  return out;
}
