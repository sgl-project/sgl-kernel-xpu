#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>

#include <cmath>
#include <cstdint>
#include <cute/tensor.hpp>
#include <limits>
#include <optional>
#include <sycl/sycl.hpp>

#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;
using Bf16Dpas = cute::XE_DPAS_TT<1, float, cute::bfloat16_t>;
using DpasTraits = cute::MMA_Traits<Bf16Dpas>;
// Qwen3.8-Flash-Next only: 256-wide heads require exactly 16 subgroups of 16 lanes in this mapping.
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
  int64_t rows;
  int64_t heads;
  int64_t kv_heads;
  int64_t cache_tokens;
  int64_t topk;
  int64_t splits;
  float scale;

  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<1> item) const {
    const int64_t group = item.get_group(0);
    const int64_t split = group % splits;
    const int64_t head = (group / splits) % heads;
    const int64_t row = group / (splits * heads);
    const int64_t kv_head = head / (heads / kv_heads);
    const int dim = item.get_local_id(0);
    const int dim_tile = dim / kTileSize;
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

    const int64_t end = split * SplitSize + sycl::min(static_cast<int64_t>(SplitSize), topk - split * SplitSize);
    for (int64_t start = split * SplitSize; start < end;) {
      const bool valid_column = dim_tile < topk - start;
      const int64_t column = valid_column ? start + dim_tile : 0;
      const int32_t slot = valid_column ? slots[row * topk + column] : -1;
      // Mask invalid cache slots before scoring or gathering K and V.
      if (lane == 0) selected[dim_tile] = slot >= 0 && static_cast<int64_t>(slot) < cache_tokens ? slot : -1;
      // Publish selected slots before other subgroups gather K and V.
      item.barrier(sycl::access::fence_space::local_space);
      const int d = dim_tile * kTileSize;
      const int query_dim = DpasTraits::ALayout{}(lane, 0);
      const cute::bfloat16_t query_value =
          sycl::bit_cast<cute::bfloat16_t>(q[(row * heads + head) * kHeadDim + d + query_dim]);
      cute::bfloat16_t key_values[kTileSize];
      for (int j = 0; j < kTileSize; ++j) {
        // The DPAS B fragment interleaves reduction dimensions and token columns.
        const int coordinate = DpasTraits::BLayout{}(lane, j);
        const int token_idx = coordinate % kTileSize;
        const int key_dim = coordinate / kTileSize;
        const int32_t key_slot = selected[token_idx];
        const int64_t offset = (static_cast<int64_t>(key_slot) * kv_heads + kv_head) * kHeadDim;
        key_values[j] = key_slot >= 0 ? sycl::bit_cast<cute::bfloat16_t>(k[offset + d + key_dim]) : cute::bfloat16_t{};
      }
      float partial_score[1] = {};
      auto query = cute::make_tensor(cute::make_rmem_ptr(&query_value), cute::make_layout(cute::Shape<cute::Int<1>>{}));
      auto key =
          cute::make_tensor(cute::make_rmem_ptr(key_values), cute::make_layout(cute::Shape<cute::Int<kTileSize>>{}));
      auto score =
          cute::make_tensor(cute::make_rmem_ptr(partial_score), cute::make_layout(cute::Shape<cute::Int<1>>{}));
      cute::gemm(cute::MMA_Atom<Bf16Dpas>{}, query, key, score);
      partial_scores[dim_tile * kTileSize + DpasTraits::CLayout{}(lane, 0)] = partial_score[0];
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
      const cute::bfloat16_t probability = sycl::bit_cast<cute::bfloat16_t>(static_cast<bf16>(weights[prob_idx]));
      cute::bfloat16_t value_fragments[kTileSize];
      for (int j = 0; j < kTileSize; ++j) {
        const int coordinate = DpasTraits::BLayout{}(lane, j);
        const int token_idx = coordinate / kTileSize;
        const int value_dim = dim_tile * kTileSize + coordinate % kTileSize;
        const int32_t value_slot = selected[token_idx];
        const int64_t offset = (static_cast<int64_t>(value_slot) * kv_heads + kv_head) * kHeadDim;
        value_fragments[j] =
            value_slot >= 0 ? sycl::bit_cast<cute::bfloat16_t>(v[offset + value_dim]) : cute::bfloat16_t{};
      }
      float weighted_value[1] = {};
      auto prob = cute::make_tensor(cute::make_rmem_ptr(&probability), cute::make_layout(cute::Shape<cute::Int<1>>{}));
      auto values = cute::make_tensor(
          cute::make_rmem_ptr(value_fragments), cute::make_layout(cute::Shape<cute::Int<kTileSize>>{}));
      auto value =
          cute::make_tensor(cute::make_rmem_ptr(weighted_value), cute::make_layout(cute::Shape<cute::Int<1>>{}));
      cute::gemm(cute::MMA_Atom<Bf16Dpas>{}, prob, values, value);
      numerator += weighted_value[0];
      // Finish reading shared slots and weights before the next tile overwrites them.
      if (end - start <= kTileSize) break;
      item.barrier(sycl::access::fence_space::local_space);
      start += kTileSize;
    }

    if (splits == 1) {
      output[(row * heads + head) * kHeadDim + dim] = static_cast<bf16>(state[1] > 0.0f ? numerator / state[1] : 0.0f);
    } else {
      const int64_t index = split * rows * heads + row * heads + head;
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
  int64_t heads;
  int64_t rows;
  int64_t splits;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t group = item.get_group(0);
    const int dim = item.get_local_id(0);
    float maximum = -INFINITY;
    for (int64_t split = 0; split < splits; ++split) {
      const int64_t index = split * rows * heads + group;
      maximum = sycl::fmax(maximum, stats[index * 2]);
    }
    float sum = 0.0f;
    float numerator = 0.0f;
    if (maximum != -INFINITY) {
      for (int64_t split = 0; split < splits; ++split) {
        const int64_t index = split * rows * heads + group;
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
  const int64_t rows = q.size(0);
  const int64_t heads = q.size(1);
  const int64_t kv_heads = k_cache.size(1);
  const int64_t topk = token_slots.size(1);
  if (rows == 0) return out;
  if (topk == 0) return at::zeros_like(q);
  const int split_size = topk <= 512 ? kShortSplitSize : kLongSplitSize;
  const int64_t splits = (topk - 1) / split_size + 1;
  TORCH_CHECK(
      rows <= std::numeric_limits<int64_t>::max() / topk &&
          rows <= std::numeric_limits<int64_t>::max() / kHeadDim / heads / splits,
      "QSA slot or partial-buffer offsets exceed int64");
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
              k_cache.size(0),
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
