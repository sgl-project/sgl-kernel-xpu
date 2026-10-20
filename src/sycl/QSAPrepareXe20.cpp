#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "cutlass/float8.h"
#include "sgl_kernel_export.h"

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;
using fp8 = cutlass::float_e4m3_t;
constexpr int kSubgroup = 16;
constexpr int kThreads = 128;
constexpr int kTasksPerGroup = kThreads / kSubgroup;

template <typename T>
struct TypeTag {
  using type = T;
};

inline float rounded(float x) {
  return static_cast<float>(bf16(x));
}

template <typename Output>
Output encode(bf16 x) {
  if constexpr (std::is_same_v<Output, bf16>)
    return x;
  else
    return fp8(float(x));
}

// ROPE: Rotate each dimension pair: (a, b) -> (a cos - b sin, b cos + a sin).
// Read the selected MRoPE axis and cast the result to the index storage dtype.
struct Rope {
  const float* cache;
  const int32_t* axes;
  int rotary_dim;
  int capacity;
  bool neox;

  template <typename Output>
  void store(Output* output, const int64_t* positions, int pair, float first, float second) const {
    const int half = rotary_dim / 2;
    const int a = neox ? pair : 2 * pair;
    const int b = neox ? pair + half : a + 1;
    const int axis = axes[pair];
    const int64_t pos = axis >= 0 && axis < 3 ? positions[axis] : -1;
    if (pos < 0 || pos >= capacity) {
      output[a] = Output(0.0f);
      output[b] = Output(0.0f);
      return;
    }
    const float* row = cache + pos * rotary_dim;
    const float c = rounded(row[pair]);
    const float s = rounded(row[half + pair]);
    output[a] = encode<Output>(bf16(rounded(first * c) - rounded(second * s)));
    output[b] = encode<Output>(bf16(rounded(second * c) + rounded(first * s)));
  }
};

// Per token and head, apply Gemma RMSNorm and RoPE to index Q.
// Also save raw BF16 K and RoPE positions once per token for group compression.
template <typename Output>
struct QPrep {
  const bf16* qk;
  const int64_t* positions;
  int64_t axis_stride;
  int axes_count;
  const bf16* weight;
  const int64_t* locs;
  bf16* state;
  int64_t* rope_positions;
  Output* out;
  Rope rope;
  int tokens, heads, padded, dim, slots;
  float eps;

  [[sycl::reqd_sub_group_size(kSubgroup)]]
  void operator()(sycl::nd_item<1> item) const {
    const int task = item.get_group(0) * kTasksPerGroup + item.get_local_id(0) / kSubgroup;
    const int token = task / padded;
    const int head = task % padded;
    const int lane = item.get_local_id(0) % kSubgroup;
    if (token >= tokens) return;
    const auto sg = item.get_sub_group();
    int64_t pos[3];
    for (int a = 0; a < 3; ++a)
      pos[a] = positions[(a < axes_count ? a : 0) * axis_stride + token];
    const int64_t loc = locs[token];
    if (head == 0 && loc >= 0 && loc < slots) {
      for (int d = lane; d < dim; d += kSubgroup)
        state[loc * dim + d] = qk[(static_cast<int64_t>(token) * (heads + 1) + heads) * dim + d];
      if (lane < 3) rope_positions[loc * 3 + lane] = pos[lane];
    }

    Output* dst = out + (static_cast<int64_t>(token) * padded + head) * dim;
    if (head >= heads) {
      for (int d = lane; d < dim; d += kSubgroup)
        dst[d] = Output(0.0f);
      return;
    }
    const bf16* src = qk + (static_cast<int64_t>(token) * (heads + 1) + head) * dim;
    float square = 0.0f;
    for (int d = lane * 8; d < dim; d += kSubgroup * 8) {
      for (int i = 0; i < 8 && d + i < dim; ++i) {
        const float x = src[d + i];
        square += x * x;
      }
    }
    square = sycl::reduce_over_group(sg, square, sycl::plus<float>());
    const float scale = sycl::rsqrt(square / dim + eps);
    for (int pair = lane; pair < rope.rotary_dim / 2; pair += kSubgroup) {
      const int a = rope.neox ? pair : 2 * pair;
      const int b = rope.neox ? pair + rope.rotary_dim / 2 : a + 1;
      const float first = rounded(float(src[a]) * scale * (1.0f + float(weight[a])));
      const float second = rounded(float(src[b]) * scale * (1.0f + float(weight[b])));
      rope.store(dst, pos, pair, first, second);
    }
    for (int d = rope.rotary_dim + lane; d < dim; d += kSubgroup)
      dst[d] = encode<Output>(bf16(float(src[d]) * scale * (1.0f + float(weight[d]))));
  }
};

// Average raw keys over a completed group, then apply Gemma RMSNorm and RoPE.
// Store one compressed index K at the group's first token position.
template <typename Output>
struct KCompress {
  const bf16* state;
  const int32_t* groups;
  const int64_t* rope_positions;
  const bf16* weight;
  const int32_t* writes;
  Output* out;
  Rope rope;
  int count, ratio, dim, slots, out_slots;
  float eps;

  float pooled(int64_t group, int d) const {
    float sum = 0.0f;
    for (int r = 0; r < ratio; ++r)
      sum += float(state[static_cast<int64_t>(groups[group * ratio + r]) * dim + d]);
    return rounded(sum / ratio);
  }

  [[sycl::reqd_sub_group_size(kSubgroup)]]
  void operator()(sycl::nd_item<1> item) const {
    const int64_t group = item.get_group(0) * kTasksPerGroup + item.get_local_id(0) / kSubgroup;
    const int lane = item.get_local_id(0) % kSubgroup;
    if (group >= count || writes[group] < 0 || writes[group] >= out_slots) return;
    for (int r = 0; r < ratio; ++r)
      if (groups[group * ratio + r] < 0 || groups[group * ratio + r] >= slots) return;
    const auto sg = item.get_sub_group();
    float square = 0.0f;
    for (int d = lane * 8; d < dim; d += kSubgroup * 8) {
      for (int i = 0; i < 8 && d + i < dim; ++i) {
        const float x = pooled(group, d + i);
        square += x * x;
      }
    }
    square = sycl::reduce_over_group(sg, square, sycl::plus<float>());
    const float scale = sycl::rsqrt(square / dim + eps);
    const int64_t* pos = rope_positions + static_cast<int64_t>(groups[group * ratio]) * 3;
    Output* dst = out + static_cast<int64_t>(writes[group]) * dim;
    for (int pair = lane; pair < rope.rotary_dim / 2; pair += kSubgroup) {
      const int a = rope.neox ? pair : 2 * pair;
      const int b = rope.neox ? pair + rope.rotary_dim / 2 : a + 1;
      const float first = rounded(pooled(group, a) * scale * (1.0f + float(weight[a])));
      const float second = rounded(pooled(group, b) * scale * (1.0f + float(weight[b])));
      rope.store(dst, pos, pair, first, second);
    }
    for (int d = rope.rotary_dim + lane; d < dim; d += kSubgroup)
      dst[d] = encode<Output>(bf16(pooled(group, d) * scale * (1.0f + float(weight[d]))));
  }
};

void check_tensor(const at::Tensor& tensor, const at::Tensor& anchor, at::ScalarType dtype, const char* name) {
  CHECK_INPUT(tensor);
  TORCH_CHECK(tensor.device() == anchor.device() && tensor.scalar_type() == dtype, name, " device or dtype mismatch");
}

void check_common(
    const at::Tensor& anchor,
    const at::Tensor& cache,
    const at::Tensor& axis,
    const at::Tensor& weight,
    int64_t dim,
    int64_t rotary,
    double eps) {
  CHECK_INPUT(anchor);
  TORCH_CHECK(anchor.scalar_type() == at::kBFloat16, "QSA prep requires BF16");
  TORCH_CHECK(dim == 64 || dim == 128 || dim == 256, "QSA prep requires head_dim 64, 128 or 256");
  TORCH_CHECK(rotary > 0 && rotary <= dim && rotary % 2 == 0, "QSA prep invalid rotary_dim");
  TORCH_CHECK(std::isfinite(eps) && eps >= 0, "QSA prep invalid eps");
  check_tensor(cache, anchor, at::kFloat, "cos_sin_cache");
  check_tensor(axis, anchor, at::kInt, "axis_map");
  check_tensor(weight, anchor, at::kBFloat16, "weight");
  TORCH_CHECK(
      cache.dim() == 2 && cache.size(1) == rotary && cache.size(0) > 0 && axis.dim() == 1 &&
          axis.size(0) == rotary / 2 && weight.dim() == 1 && weight.size(0) == dim,
      "QSA prep invalid cos_sin_cache, axis_map or weight shape");
}

Rope rope_args(const at::Tensor& cache, const at::Tensor& axis, int64_t rotary, bool neox) {
  return {
      cache.data_ptr<float>(),
      axis.data_ptr<int32_t>(),
      static_cast<int>(rotary),
      static_cast<int>(cache.size(0)),
      neox};
}

}  // namespace

SGL_KERNEL_EXPORT at::Tensor qsa_index_q_norm_rope_store(
    const at::Tensor& qk,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& axis_map,
    const at::Tensor& weight,
    const at::Tensor& cache_loc,
    at::Tensor key_state_buffer,
    at::Tensor rope_position_buffer,
    int64_t num_q_heads,
    int64_t rotary_dim,
    double eps,
    bool is_neox_style,
    std::optional<int64_t> q_heads_padded,
    std::optional<at::ScalarType> out_dtype) {
  TORCH_CHECK(qk.dim() == 2 && weight.dim() == 1, "QSA query prep requires qk [tokens, (heads+1)*dim]");
  const auto dtype = out_dtype.value_or(at::kBFloat16);
  TORCH_CHECK(
      dtype == at::kBFloat16 || dtype == at::kFloat8_e4m3fn,
      "Xe2 QSA preparation supports BF16 or float8_e4m3fn output only");
  const int64_t dim = weight.size(0), tokens = qk.size(0);
  const int64_t padded = q_heads_padded.value_or(num_q_heads);
  check_common(qk, cos_sin_cache, axis_map, weight, dim, rotary_dim, eps);
  const at::Tensor* inputs[] = {&positions, &cache_loc, &key_state_buffer, &rope_position_buffer};
  for (const at::Tensor* t : inputs)
    TORCH_CHECK(t->device() == qk.device(), "QSA query prep tensors must be on the same XPU");
  check_tensor(cache_loc, qk, at::kLong, "cache_loc");
  check_tensor(key_state_buffer, qk, at::kBFloat16, "key_state_buffer");
  check_tensor(rope_position_buffer, qk, at::kLong, "rope_position_buffer");
  TORCH_CHECK(
      positions.scalar_type() == at::kLong && positions.device() == qk.device() &&
          (positions.dim() == 1 || positions.dim() == 2) &&
          (positions.dim() == 1 || positions.size(0) == 1 || positions.size(0) == 3) && positions.size(-1) == tokens &&
          positions.stride(-1) == 1,
      "QSA query prep requires int64 positions [tokens] or [1|3, tokens] with contiguous token axis");
  TORCH_CHECK(
      num_q_heads > 0 && padded >= num_q_heads && padded <= std::numeric_limits<int>::max() &&
          tokens <= std::numeric_limits<int>::max() && qk.size(1) == (num_q_heads + 1) * dim &&
          cache_loc.numel() == tokens && key_state_buffer.dim() == 2 && key_state_buffer.size(1) == dim &&
          rope_position_buffer.dim() == 2 && rope_position_buffer.size(0) == key_state_buffer.size(0) &&
          rope_position_buffer.size(1) == 3 && key_state_buffer.size(0) <= std::numeric_limits<int>::max(),
      "QSA query prep invalid head count or buffer shapes");
  c10::DeviceGuard guard(qk.device());
  auto out = at::empty({tokens, padded, dim}, qk.options().dtype(dtype));
  if (!tokens) return out;
  const size_t tasks = static_cast<size_t>(tokens) * padded;
  auto launch = [&](auto output_type) {
    using Output = typename decltype(output_type)::type;
    sycl_kernel_submit(
        static_cast<int64_t>((tasks + kTasksPerGroup - 1) / kTasksPerGroup) * kThreads,
        static_cast<int64_t>(kThreads),
        at::xpu::getCurrentXPUStream().queue(),
        QPrep<Output>{
            reinterpret_cast<const bf16*>(qk.data_ptr<at::BFloat16>()),
            positions.data_ptr<int64_t>(),
            positions.dim() == 1 ? 0 : positions.stride(0),
            static_cast<int>(positions.dim() == 1 ? 1 : positions.size(0)),
            reinterpret_cast<const bf16*>(weight.data_ptr<at::BFloat16>()),
            cache_loc.data_ptr<int64_t>(),
            reinterpret_cast<bf16*>(key_state_buffer.data_ptr<at::BFloat16>()),
            rope_position_buffer.data_ptr<int64_t>(),
            reinterpret_cast<Output*>(out.data_ptr()),
            rope_args(cos_sin_cache, axis_map, rotary_dim, is_neox_style),
            static_cast<int>(tokens),
            static_cast<int>(num_q_heads),
            static_cast<int>(padded),
            static_cast<int>(dim),
            static_cast<int>(key_state_buffer.size(0)),
            static_cast<float>(eps)});
  };
  if (dtype == at::kBFloat16)
    launch(TypeTag<bf16>{});
  else
    launch(TypeTag<fp8>{});
  return out;
}

SGL_KERNEL_EXPORT void qsa_index_k_compress_store(
    const at::Tensor& key_state_buffer,
    const at::Tensor& group_locs,
    const at::Tensor& rope_position_buffer,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& axis_map,
    const at::Tensor& weight,
    const at::Tensor& write_locs,
    at::Tensor compressed_k_buffer,
    int64_t compress_ratio,
    int64_t rotary_dim,
    double eps,
    bool is_neox_style) {
  TORCH_CHECK(key_state_buffer.dim() == 2 && weight.dim() == 1, "QSA compression requires state [slots, dim]");
  const int64_t dim = weight.size(0);
  check_common(key_state_buffer, cos_sin_cache, axis_map, weight, dim, rotary_dim, eps);
  check_tensor(group_locs, key_state_buffer, at::kInt, "group_locs");
  check_tensor(rope_position_buffer, key_state_buffer, at::kLong, "rope_position_buffer");
  check_tensor(write_locs, key_state_buffer, at::kInt, "write_locs");
  TORCH_CHECK(
      compressed_k_buffer.device() == key_state_buffer.device() &&
          (compressed_k_buffer.scalar_type() == at::kBFloat16 ||
           compressed_k_buffer.scalar_type() == at::kFloat8_e4m3fn),
      "compressed_k_buffer must be XPU BF16 or float8_e4m3fn");
  CHECK_INPUT(compressed_k_buffer);
  TORCH_CHECK(
      compress_ratio > 0 && compress_ratio <= 16 && group_locs.dim() == 2 && group_locs.size(1) == compress_ratio &&
          write_locs.dim() == 1 && write_locs.size(0) == group_locs.size(0) && key_state_buffer.size(1) == dim &&
          rope_position_buffer.dim() == 2 && rope_position_buffer.size(0) == key_state_buffer.size(0) &&
          rope_position_buffer.size(1) == 3 && compressed_k_buffer.dim() == 2 && compressed_k_buffer.size(1) == dim &&
          group_locs.size(0) <= std::numeric_limits<int>::max() &&
          key_state_buffer.size(0) <= std::numeric_limits<int>::max() &&
          compressed_k_buffer.size(0) <= std::numeric_limits<int>::max(),
      "QSA compression invalid ratio or buffer shapes");
  if (!group_locs.size(0)) return;
  c10::DeviceGuard guard(key_state_buffer.device());
  const size_t groups = group_locs.size(0);
  auto launch = [&](auto output_type) {
    using Output = typename decltype(output_type)::type;
    sycl_kernel_submit(
        static_cast<int64_t>((groups + kTasksPerGroup - 1) / kTasksPerGroup) * kThreads,
        static_cast<int64_t>(kThreads),
        at::xpu::getCurrentXPUStream().queue(),
        KCompress<Output>{
            reinterpret_cast<const bf16*>(key_state_buffer.data_ptr<at::BFloat16>()),
            group_locs.data_ptr<int32_t>(),
            rope_position_buffer.data_ptr<int64_t>(),
            reinterpret_cast<const bf16*>(weight.data_ptr<at::BFloat16>()),
            write_locs.data_ptr<int32_t>(),
            reinterpret_cast<Output*>(compressed_k_buffer.data_ptr()),
            rope_args(cos_sin_cache, axis_map, rotary_dim, is_neox_style),
            static_cast<int>(groups),
            static_cast<int>(compress_ratio),
            static_cast<int>(dim),
            static_cast<int>(key_state_buffer.size(0)),
            static_cast<int>(compressed_k_buffer.size(0)),
            static_cast<float>(eps)});
  };
  if (compressed_k_buffer.scalar_type() == at::kBFloat16)
    launch(TypeTag<bf16>{});
  else
    launch(TypeTag<fp8>{});
}
