// Qwen3.8-Flash-Next HyperConnection elementwise ops (bf16), ported from DeepKloX csrc/hc.
//   qwen4_hc_silu:     out = silu(x / hc)
//   qwen4_hc_gate_mix: out[t, d] = mean_s(sigmoid(gate[t, s, d]) * x[t, s, d])
//   qwen4_hc_combine_norm: qwen4_hc_combine followed by the grouped Gemma RMSNorm of each stream
//   qwen4_hc_combine:  out[t, s, d] = residual[t, s, d] + block[t, d] * 2 * sigmoid(injection[t, s] / hc)
// The linear projections around them stay as GEMMs. Everything is accumulated in fp32 and rounded to bf16 once.
#include <ATen/MemoryOverlap.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

// Elementwise kernels are written once for a vector width V: 8 bf16 values (one 16-byte access) when the shape and
// pointers allow it, otherwise 1.
constexpr int VEC_SIZE = 8;
template <int V>
using bf16xV = vec_t<bf16, V>;
using bf16x8 = bf16xV<VEC_SIZE>;
static_assert(sizeof(bf16x8) == 16, "bf16x8 must be one 16-byte access");

constexpr int SG_SIZE = 16;
constexpr int FLAT_WG = 64;
constexpr int64_t MAX_ID = std::numeric_limits<int32_t>::max();  // SYCL id queries are 32-bit
// combine_norm calls with fewer (token, stream) rows use the decode configuration.
constexpr int64_t COMBINE_NORM_FEW_ROWS = 256;

inline float sigmoid(float v) {
  return 1.0f / (1.0f + sycl::exp(-v));
}

template <int N>
inline void to_float(const bf16xV<N>& input, float (&output)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    output[i] = static_cast<float>(input[i]);
  }
}

template <int N>
inline bf16xV<N> to_bf16(const float (&input)[N]) {
  bf16xV<N> output;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    output[i] = static_cast<bf16>(input[i]);
  }
  return output;
}

template <int V>
struct SiluKernel {
  const bf16xV<V>* input;
  bf16xV<V>* output;
  uint32_t items;
  float divisor;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    const uint32_t i = item.get_global_id(0);
    if (i >= items) return;
    float v[V];
    to_float<V>(input[i], v);
#pragma unroll
    for (int k = 0; k < V; ++k) {
      const float s = v[k] / divisor;
      v[k] = s * sigmoid(s);
    }
    output[i] = to_bf16<V>(v);
  }
};

// Indices are 32-bit (numel <= MAX_ID is checked); 64-bit division is slow on the GPU.
// One work-item per output vector; `vecs_per_row` is the hidden size in vectors.
template <int V>
struct GateMixKernel {
  const bf16xV<V>* input;
  const bf16xV<V>* gate;
  bf16xV<V>* output;
  uint32_t items;
  uint32_t vecs_per_row;
  uint32_t hc;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    const uint32_t i = item.get_global_id(0);
    if (i >= items) return;
    const uint32_t token = i / vecs_per_row;
    const uint32_t base = token * hc * vecs_per_row + (i - token * vecs_per_row);
    float acc[V] = {};
    const auto accumulate = [&](const bf16xV<V>& x, const bf16xV<V>& g) {
      float xv[V], gv[V];
      to_float<V>(x, xv);
      to_float<V>(g, gv);
#pragma unroll
      for (int k = 0; k < V; ++k) {
        acc[k] += sigmoid(gv[k]) * xv[k];
      }
    };
    if (hc == 4) {
      // Issue all stream loads before the math, which matters for decode latency.
      bf16xV<V> xs[4], gs[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        xs[s] = input[base + s * vecs_per_row];
        gs[s] = gate[base + s * vecs_per_row];
      }
#pragma unroll
      for (int s = 0; s < 4; ++s)
        accumulate(xs[s], gs[s]);
    } else {
      for (uint32_t s = 0; s < hc; ++s)
        accumulate(input[base + s * vecs_per_row], gate[base + s * vecs_per_row]);
    }
#pragma unroll
    for (int k = 0; k < V; ++k) {
      acc[k] /= static_cast<float>(hc);
    }
    output[i] = to_bf16<V>(acc);
  }
};

// One work-item per vector of `residual`; the block row is re-read per stream from cache.
template <int V>
struct CombineKernel {
  const bf16xV<V>* residual;
  const bf16xV<V>* block;
  const bf16* injection;
  bf16xV<V>* output;
  uint32_t items;
  uint32_t vecs_per_row;
  uint32_t hc;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    const uint32_t i = item.get_global_id(0);
    if (i >= items) return;
    const uint32_t row = i / vecs_per_row;  // (token, stream)
    const uint32_t token = row / hc;
    const float w = injection == nullptr ? 1.0f : 2.0f * sigmoid(static_cast<float>(injection[row]) / hc);
    float rv[V], bv[V];
    to_float<V>(residual[i], rv);
    to_float<V>(block[token * vecs_per_row + (i - row * vecs_per_row)], bv);
#pragma unroll
    for (int k = 0; k < V; ++k) {
      rv[k] += bv[k] * w;
    }
    output[i] = to_bf16<V>(rv);
  }
};

template <typename Kernel>
void launch_flat(sycl::queue& q, int64_t items, const Kernel& kernel) {
  const int64_t groups = div_up(items, static_cast<int64_t>(FLAT_WG));
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * FLAT_WG), sycl::range<1>(FLAT_WG)), kernel);
  });
}

// Calls f(std::integral_constant<int, V>) with V = 8 when `vectorizable`, else 1.
template <typename F>
void dispatch_vec(bool vectorizable, F&& f) {
  if (vectorizable) {
    f(std::integral_constant<int, VEC_SIZE>{});
  } else {
    f(std::integral_constant<int, 1>{});
  }
}

// `inner` is the contiguous extent (in elements) every vector must tile; all tensors must be 16-byte aligned.
bool can_vectorize(int64_t inner, std::initializer_list<const at::Tensor*> tensors) {
  if (inner % VEC_SIZE != 0) return false;
  for (const at::Tensor* t : tensors) {
    if (reinterpret_cast<uintptr_t>(t->data_ptr()) % sizeof(bf16x8) != 0) return false;
  }
  return true;
}

void check_bf16_xpu(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_xpu(), name, " must be on XPU");
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bfloat16");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// An output may alias an input exactly (in place) but must not partially overlap it.
void check_no_partial_overlap(const at::Tensor& output, std::initializer_list<const at::Tensor*> inputs) {
  for (const at::Tensor* t : inputs)
    at::assert_no_partial_overlap(output, *t);
}

void check_no_overlap(const at::Tensor& output, std::initializer_list<const at::Tensor*> inputs) {
  for (const at::Tensor* t : inputs)
    at::assert_no_overlap(output, *t);
}

template <typename T>
const T* as(const at::Tensor& t) {
  return reinterpret_cast<const T*>(t.data_ptr());
}

template <typename T>
T* as_mut(at::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

// Returns the optional [tokens, hc] injection pointer after validating it.
const bf16* injection_ptr(const std::optional<at::Tensor>& injection, int64_t tokens, int64_t hc) {
  if (!injection.has_value()) return nullptr;
  check_bf16_xpu(*injection, "injection");
  TORCH_CHECK(
      injection->dim() == 2 && injection->size(0) == tokens && injection->size(1) == hc,
      "injection must be [tokens, hc_count]");
  return as<bf16>(*injection);
}

// combined = bf16(residual + block * w); normalized = bf16(combined * rsqrt(mean(combined^2) + eps) * (1 + weight)),
// both per (token, stream) row, i.e. the combine followed by the next grouped Gemma RMSNorm.
struct CombineNormArgs {
  const bf16x8* residual;
  const bf16x8* block;
  const bf16* injection;
  const bf16x8* weight;
  bf16x8* combined;
  bf16x8* normalized;
  int64_t rows;
  int hc;
  int vecs_per_row;
  bool shared_weight;
  float eps;
};

// The kernel repeats these fields instead of embedding CombineNormArgs: a nested struct argument is ~25% slower.
// Each work-group owns ROWS_PER_GROUP rows of SPLIT sub-groups each. Lane l of part p handles vectors
// c * SPLIT * SG + p * SG + l, so every load and store is a contiguous 16 * SG byte stripe. The whole row stays in
// registers between the sum-of-squares reduction and the normalization.
template <int VECS_PER_THREAD, int SPLIT, int ROWS_PER_GROUP>
struct CombineNormVecKernel {
  static constexpr int GROUP_SIZE = SG_SIZE * SPLIT * ROWS_PER_GROUP;
  const bf16x8* residual;
  const bf16x8* block;
  const bf16* injection;
  const bf16x8* weight;
  bf16x8* combined;
  bf16x8* normalized;
  int64_t rows;
  int hc;
  int vecs_per_row;
  bool shared_weight;
  float eps;
  sycl::local_accessor<float, 1> partials;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    const auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int sg_id = sg.get_group_linear_id();
    const int local_row = sg_id / SPLIT;
    const int part = sg_id - local_row * SPLIT;
    const int64_t row = static_cast<int64_t>(item.get_group(0)) * ROWS_PER_GROUP + local_row;
    const bool active = row < rows;
    const int64_t safe_row = active ? row : 0;
    const int64_t token = safe_row / hc;
    const int stream = static_cast<int>(safe_row - token * hc);
    const int first = part * SG_SIZE + lane;
    constexpr int STRIDE = SG_SIZE * SPLIT;

    const bf16x8* residual_row = residual + safe_row * vecs_per_row;
    const bf16x8* block_row = block + token * vecs_per_row;
    bf16x8* combined_row = combined + safe_row * vecs_per_row;
    const float w = injection == nullptr ? 1.0f : 2.0f * sigmoid(static_cast<float>(injection[safe_row]) / hc);

    bf16x8 values[VECS_PER_THREAD];
    float sum_sq = 0.0f;
#pragma unroll
    for (int c = 0; c < VECS_PER_THREAD; ++c) {
      const int index = c * STRIDE + first;
      if (active && index < vecs_per_row) {
        float rv[VEC_SIZE], bv[VEC_SIZE];
        to_float<VEC_SIZE>(residual_row[index], rv);
        to_float<VEC_SIZE>(block_row[index], bv);
#pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k) {
          rv[k] += bv[k] * w;
        }
        values[c] = to_bf16<VEC_SIZE>(rv);
        combined_row[index] = values[c];
        float mv[VEC_SIZE];
        to_float<VEC_SIZE>(values[c], mv);
#pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k) {
          sum_sq += mv[k] * mv[k];
        }
      }
    }

    sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
    if constexpr (SPLIT > 1) {
      if (lane == 0) {
        partials[sg_id] = sum_sq;
      }
      sycl::group_barrier(item.get_group());
      sum_sq = 0.0f;
#pragma unroll
      for (int other = 0; other < SPLIT; ++other) {
        sum_sq += partials[local_row * SPLIT + other];
      }
    }
    if (!active) return;
    const float scale = sycl::rsqrt(sum_sq / static_cast<float>(vecs_per_row * VEC_SIZE) + eps);

    const bf16x8* weight_row = weight + (shared_weight ? 0 : stream * vecs_per_row);
    bf16x8* normalized_row = normalized + safe_row * vecs_per_row;
#pragma unroll
    for (int c = 0; c < VECS_PER_THREAD; ++c) {
      const int index = c * STRIDE + first;
      if (index < vecs_per_row) {
        float v[VEC_SIZE], wv[VEC_SIZE], out[VEC_SIZE];
        to_float<VEC_SIZE>(values[c], v);
        to_float<VEC_SIZE>(weight_row[index], wv);
#pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k) {
          out[k] = v[k] * scale * (1.0f + wv[k]);
        }
        normalized_row[index] = to_bf16<VEC_SIZE>(out);
      }
    }
  }
};

// One work-group per (token, stream) row for shapes the vector kernel does not cover.
struct CombineNormScalarKernel {
  const bf16* residual;
  const bf16* block;
  const bf16* injection;
  const bf16* weight;
  bf16* combined;
  bf16* normalized;
  int64_t hidden;
  int64_t hc;
  bool shared_weight;
  float eps;
  sycl::local_accessor<float, 1> scratch;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t row = item.get_group(0);
    const int64_t token = row / hc;
    const int64_t stream = row % hc;
    const int64_t base = row * hidden;
    const float w =
        injection == nullptr ? 1.0f : 2.0f * sigmoid(static_cast<float>(injection[row]) / static_cast<float>(hc));
    float sum_sq = 0.0f;
    for (int64_t d = item.get_local_id(0); d < hidden; d += item.get_local_range(0)) {
      const bf16 v =
          static_cast<bf16>(static_cast<float>(residual[base + d]) + static_cast<float>(block[token * hidden + d]) * w);
      combined[base + d] = v;
      const float f = static_cast<float>(v);
      sum_sq += f * f;
    }
    sum_sq = sycl::reduce_over_group(item.get_group(), sum_sq, sycl::plus<float>());
    if (item.get_local_id(0) == 0) {
      scratch[0] = sycl::rsqrt(sum_sq / static_cast<float>(hidden) + eps);
    }
    sycl::group_barrier(item.get_group());
    const int64_t weight_base = shared_weight ? 0 : stream * hidden;
    for (int64_t d = item.get_local_id(0); d < hidden; d += item.get_local_range(0)) {
      normalized[base + d] = static_cast<bf16>(
          static_cast<float>(combined[base + d]) * scratch[0] * (1.0f + static_cast<float>(weight[weight_base + d])));
    }
  }
};

// Returns false when a row does not fit the register tile.
template <int VECS_PER_THREAD, int SPLIT, int ROWS_PER_GROUP>
bool launch_combine_norm_vec(sycl::queue& q, const CombineNormArgs& args) {
  using Kernel = CombineNormVecKernel<VECS_PER_THREAD, SPLIT, ROWS_PER_GROUP>;
  if (args.vecs_per_row > VECS_PER_THREAD * SG_SIZE * SPLIT) return false;
  const int64_t groups = div_up(args.rows, static_cast<int64_t>(ROWS_PER_GROUP));
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partials(SPLIT * ROWS_PER_GROUP, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * Kernel::GROUP_SIZE), sycl::range<1>(Kernel::GROUP_SIZE)),
        Kernel{
            args.residual,
            args.block,
            args.injection,
            args.weight,
            args.combined,
            args.normalized,
            args.rows,
            args.hc,
            args.vecs_per_row,
            args.shared_weight,
            args.eps,
            partials});
  });
  return true;
}

}  // namespace

namespace at::native::xpu {

SGL_KERNEL_EXPORT void qwen4_hc_silu(torch::Tensor& output, const torch::Tensor& input, int64_t hc_count) {
  check_bf16_xpu(input, "input");
  check_bf16_xpu(output, "output");
  TORCH_CHECK(output.sizes() == input.sizes(), "output must have the same shape as input");
  TORCH_CHECK(hc_count > 0, "hc_count must be positive");
  check_no_partial_overlap(output, {&input});
  const int64_t n = input.numel();
  if (n == 0) return;
  TORCH_CHECK(n <= MAX_ID, "qwen4_hc_silu: input too large: ", input.sizes());

  c10::DeviceGuard guard(input.device());
  auto q = c10::xpu::getCurrentXPUStream().queue();
  dispatch_vec(can_vectorize(n, {&input, &output}), [&](auto vec) {
    constexpr int V = decltype(vec)::value;
    const uint32_t items = n / V;
    launch_flat(
        q, items, SiluKernel<V>{as<bf16xV<V>>(input), as_mut<bf16xV<V>>(output), items, static_cast<float>(hc_count)});
  });
}

SGL_KERNEL_EXPORT void
qwen4_hc_gate_mix(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& gate, int64_t hc_count) {
  check_bf16_xpu(input, "input");
  check_bf16_xpu(gate, "gate");
  check_bf16_xpu(output, "output");
  TORCH_CHECK(input.dim() == 2 && gate.sizes() == input.sizes(), "input and gate must be [tokens, HC * hidden]");
  TORCH_CHECK(hc_count > 0 && input.size(1) % hc_count == 0, "HC * hidden must be divisible by hc_count");
  const int64_t rows = input.size(0), hidden = input.size(1) / hc_count;
  TORCH_CHECK(
      output.dim() == 2 && output.size(0) == rows && output.size(1) == hidden, "output must be [tokens, hidden]");
  check_no_overlap(output, {&input, &gate});
  if (rows == 0 || hidden == 0) return;
  TORCH_CHECK(input.numel() <= MAX_ID, "qwen4_hc_gate_mix: input too large: ", input.sizes());

  c10::DeviceGuard guard(input.device());
  auto q = c10::xpu::getCurrentXPUStream().queue();
  dispatch_vec(can_vectorize(hidden, {&input, &gate, &output}), [&](auto vec) {
    constexpr int V = decltype(vec)::value;
    const uint32_t vecs_per_row = hidden / V;
    const uint32_t items = rows * vecs_per_row;
    launch_flat(
        q,
        items,
        GateMixKernel<V>{
            as<bf16xV<V>>(input),
            as<bf16xV<V>>(gate),
            as_mut<bf16xV<V>>(output),
            items,
            vecs_per_row,
            static_cast<uint32_t>(hc_count)});
  });
}

SGL_KERNEL_EXPORT void qwen4_hc_combine(
    torch::Tensor& output,
    const torch::Tensor& residual,
    const torch::Tensor& block_output,
    const std::optional<torch::Tensor>& injection,
    int64_t hc_count) {
  check_bf16_xpu(residual, "residual");
  check_bf16_xpu(block_output, "block_output");
  check_bf16_xpu(output, "output");
  TORCH_CHECK(residual.dim() == 2 && block_output.dim() == 2, "residual and block_output must be 2D");
  TORCH_CHECK(hc_count > 0 && residual.size(1) % hc_count == 0, "HC * hidden must be divisible by hc_count");
  const int64_t rows = residual.size(0), hidden = residual.size(1) / hc_count;
  TORCH_CHECK(block_output.size(0) == rows && block_output.size(1) == hidden, "block_output must be [tokens, hidden]");
  TORCH_CHECK(output.sizes() == residual.sizes(), "output must have the same shape as residual");
  const bf16* injection_data = injection_ptr(injection, rows, hc_count);
  check_no_partial_overlap(output, {&residual});
  check_no_overlap(output, {&block_output});
  if (injection_data != nullptr) check_no_overlap(output, {&*injection});
  if (rows == 0 || hidden == 0) return;
  TORCH_CHECK(residual.numel() <= MAX_ID, "qwen4_hc_combine: input too large: ", residual.sizes());

  c10::DeviceGuard guard(residual.device());
  auto q = c10::xpu::getCurrentXPUStream().queue();
  dispatch_vec(can_vectorize(hidden, {&residual, &block_output, &output}), [&](auto vec) {
    constexpr int V = decltype(vec)::value;
    const uint32_t vecs_per_row = hidden / V;
    const uint32_t items = residual.numel() / V;
    launch_flat(
        q,
        items,
        CombineKernel<V>{
            as<bf16xV<V>>(residual),
            as<bf16xV<V>>(block_output),
            injection_data,
            as_mut<bf16xV<V>>(output),
            items,
            vecs_per_row,
            static_cast<uint32_t>(hc_count)});
  });
}

SGL_KERNEL_EXPORT void qwen4_hc_combine_norm(
    torch::Tensor& combined,
    torch::Tensor& normalized,
    const torch::Tensor& residual,
    const torch::Tensor& block_output,
    const std::optional<torch::Tensor>& injection,
    const torch::Tensor& weight,
    double eps,
    int64_t hc_count) {
  check_bf16_xpu(residual, "residual");
  check_bf16_xpu(block_output, "block_output");
  check_bf16_xpu(weight, "weight");
  check_bf16_xpu(combined, "combined");
  check_bf16_xpu(normalized, "normalized");
  TORCH_CHECK(residual.dim() == 2 && block_output.dim() == 2, "residual and block_output must be 2D");
  TORCH_CHECK(hc_count > 0 && residual.size(1) % hc_count == 0, "HC * hidden must be divisible by hc_count");
  const int64_t tokens = residual.size(0), hidden = residual.size(1) / hc_count;
  TORCH_CHECK(
      block_output.size(0) == tokens && block_output.size(1) == hidden, "block_output must be [tokens, hidden]");
  TORCH_CHECK(
      weight.numel() == hidden || weight.numel() == residual.size(1),
      "weight must have hidden or HC * hidden elements");
  TORCH_CHECK(
      combined.sizes() == residual.sizes() && normalized.sizes() == residual.sizes(),
      "combined and normalized must have the same shape as residual");
  const bf16* injection_data = injection_ptr(injection, tokens, hc_count);
  check_no_partial_overlap(combined, {&residual});
  check_no_overlap(combined, {&block_output, &weight});
  check_no_overlap(normalized, {&residual, &block_output, &combined, &weight});
  if (injection_data != nullptr) {
    check_no_overlap(combined, {&*injection});
    check_no_overlap(normalized, {&*injection});
  }
  if (tokens == 0 || hidden == 0) return;
  const int64_t rows = tokens * hc_count;
  TORCH_CHECK(rows <= MAX_ID, "qwen4_hc_combine_norm: input too large: ", residual.sizes());

  c10::DeviceGuard guard(residual.device());
  auto q = c10::xpu::getCurrentXPUStream().queue();
  const bool shared_weight = weight.numel() == hidden;
  const float eps_f = static_cast<float>(eps);
  if (can_vectorize(hidden, {&residual, &block_output, &weight, &combined, &normalized})) {
    const CombineNormArgs args{
        as<bf16x8>(residual),
        as<bf16x8>(block_output),
        injection_data,
        as<bf16x8>(weight),
        as_mut<bf16x8>(combined),
        as_mut<bf16x8>(normalized),
        rows,
        static_cast<int>(hc_count),
        static_cast<int>(hidden / VEC_SIZE),
        shared_weight,
        eps_f};
    // Few rows (decode): 20 sub-groups per row, one vector per lane, for latency. Many rows (prefill): 4 sub-groups
    // per row with 5 vectors per lane in registers and 4 rows per work-group.
    if (rows < COMBINE_NORM_FEW_ROWS ? launch_combine_norm_vec<1, 20, 1>(q, args)
                                     : launch_combine_norm_vec<5, 4, 4>(q, args)) {
      return;
    }
  }
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> scratch(1, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(rows * FLAT_WG), sycl::range<1>(FLAT_WG)),
        CombineNormScalarKernel{
            as<bf16>(residual),
            as<bf16>(block_output),
            injection_data,
            as<bf16>(weight),
            as_mut<bf16>(combined),
            as_mut<bf16>(normalized),
            hidden,
            hc_count,
            shared_weight,
            eps_f,
            scratch});
  });
}

}  // namespace at::native::xpu
