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
// combine_norm tile selection by (token, stream) row count, tuned on BMG to avoid a partial last wave.
constexpr int64_t COMBINE_NORM_MID_ROWS_MIN = 128;
constexpr int64_t COMBINE_NORM_MID_ROWS_MAX = 256;
constexpr int64_t COMBINE_NORM_MANY_ROWS = 8192;
// silu and gate_mix calls with at most this many outputs are latency-bound and use 2-element vectors: twice as many
// work-items, each with a shorter dependent chain of exp/reciprocal, spread over more EU threads.
constexpr int64_t NARROW_MAX_OUTPUTS = 48 * 1024;
constexpr int NARROW_VEC = 2;

// Rounded fp32 division is slow; kernels use native math and multiply by host-side
// reciprocals instead. The results are rounded to bf16.
inline float sigmoid(float v) {
  return sycl::native::recip(1.0f + sycl::native::exp(-v));
}

// The weight of the block output for one (token, stream) row. HAS_INJECTION is a compile-time choice where a runtime
// branch around the injection load would delay other loads.
template <bool HAS_INJECTION>
inline float combine_weight(const bf16* injection, uint32_t row, float inv_hc) {
  if constexpr (HAS_INJECTION) {
    return 2.0f * sigmoid(static_cast<float>(injection[row]) * inv_hc);
  } else {
    return 1.0f;
  }
}

inline float combine_weight_dynamic(const bf16* injection, uint32_t row, float inv_hc) {
  return injection == nullptr ? combine_weight<false>(injection, row, inv_hc)
                              : combine_weight<true>(injection, row, inv_hc);
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

// bf16(residual + block * w), accumulated in fp32.
template <int N>
inline bf16xV<N> combine_values(const bf16xV<N>& residual, const bf16xV<N>& block, float w) {
  float rv[N], bv[N];
  to_float<N>(residual, rv);
  to_float<N>(block, bv);
#pragma unroll
  for (int k = 0; k < N; ++k) {
    rv[k] += bv[k] * w;
  }
  return to_bf16<N>(rv);
}

template <int V>
struct SiluKernel {
  const bf16xV<V>* input;
  bf16xV<V>* output;
  uint32_t items;
  float inv_hc;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    const uint32_t i = item.get_global_id(0);
    if (i >= items) return;
    float v[V];
    to_float<V>(input[i], v);
#pragma unroll
    for (int k = 0; k < V; ++k) {
      const float s = v[k] * inv_hc;
      v[k] = s * sigmoid(s);
    }
    output[i] = to_bf16<V>(v);
  }
};

// Indices are 32-bit (numel <= MAX_ID is checked). The row kernels below use a 2D range (row, vector in row) so that
// no work-item needs an integer division to locate its row.
// One work-item per output vector; `vecs_per_row` is the hidden size in vectors.
template <int V>
struct GateMixKernel {
  const bf16xV<V>* input;
  const bf16xV<V>* gate;
  bf16xV<V>* output;
  uint32_t vecs_per_row;
  uint32_t hc;
  float inv_hc;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<2> item) const {
    const uint32_t token = item.get_global_id(0);
    const uint32_t c = item.get_global_id(1);
    if (c >= vecs_per_row) return;
    const uint32_t base = token * hc * vecs_per_row + c;
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
      acc[k] *= inv_hc;
    }
    output[token * vecs_per_row + c] = to_bf16<V>(acc);
  }
};

// One work-item per vector of `residual`; the block row is re-read per stream from cache. HAS_INJECTION is a template
// parameter because a runtime branch around the injection load delays the other loads.
template <int V, bool HAS_INJECTION>
struct CombineKernel {
  const bf16xV<V>* residual;
  const bf16xV<V>* block;
  const bf16* injection;
  bf16xV<V>* output;
  uint32_t vecs_per_row;
  uint32_t hc;
  float inv_hc;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<2> item) const {
    const uint32_t row = item.get_global_id(0);  // (token, stream)
    const uint32_t c = item.get_global_id(1);
    if (c >= vecs_per_row) return;
    const uint32_t token = row / hc;  // uniform across the work-group
    const uint32_t i = row * vecs_per_row + c;
    const float w = combine_weight<HAS_INJECTION>(injection, row, inv_hc);
    output[i] = combine_values<V>(residual[i], block[token * vecs_per_row + c], w);
  }
};

template <typename Kernel>
void launch_flat(sycl::queue& q, int64_t items, const Kernel& kernel) {
  const int64_t groups = div_up(items, static_cast<int64_t>(FLAT_WG));
  sycl_kernel_submit(groups * FLAT_WG, static_cast<int64_t>(FLAT_WG), q, kernel);
}

// Launches `rows` x `cols` work-items as a 2D range of 1 x FLAT_WG work-groups.
template <typename Kernel>
void launch_rows(sycl::queue& q, int64_t rows, int64_t cols, const Kernel& kernel) {
  const int64_t groups = div_up(cols, static_cast<int64_t>(FLAT_WG));
  TORCH_CHECK(rows * groups * FLAT_WG <= MAX_ID, "qwen4_hc: too many rows for the padded 2D range: ", rows);
  sycl_kernel_submit(sycl::range<2>(rows, groups * FLAT_WG), sycl::range<2>(1, FLAT_WG), q, kernel);
}

// Calls f(std::integral_constant<int, V>) for a runtime V in {8, 4, 2, 1}.
template <typename F>
void dispatch_vec(int v, F&& f) {
  switch (v) {
    case 8:
      return f(std::integral_constant<int, 8>{});
    case 4:
      return f(std::integral_constant<int, 4>{});
    case 2:
      return f(std::integral_constant<int, 2>{});
    default:
      return f(std::integral_constant<int, 1>{});
  }
}

// Widest V <= max_v such that `inner` elements (the contiguous extent every vector must tile) and all pointers are
// V-aligned.
int pick_vec(int64_t inner, std::initializer_list<const at::Tensor*> tensors, int max_v = VEC_SIZE) {
  for (int v = max_v; v > 1; v /= 2) {
    bool ok = inner % v == 0;
    for (const at::Tensor* t : tensors) {
      ok = ok && reinterpret_cast<uintptr_t>(t->data_ptr()) % (v * sizeof(bf16)) == 0;
    }
    if (ok) return v;
  }
  return 1;
}

bool can_vectorize(int64_t inner, std::initializer_list<const at::Tensor*> tensors) {
  return pick_vec(inner, tensors) == VEC_SIZE;
}

void check_bf16_xpu(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_xpu(), name, " must be on XPU");
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bfloat16");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// An output may alias an input exactly (in place) but must not partially overlap it.
void check_no_partial_overlap(const at::Tensor& output, const at::Tensor& input) {
  at::assert_no_partial_overlap(output, input);
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
  uint32_t rows;
  uint32_t hc;
  uint32_t vecs_per_row;
  bool shared_weight;
  float eps;
  float inv_hc;
  float inv_hidden;
};

// One work-group per (token, stream) row, made of SPLIT sub-groups. Lane l of sub-group p handles vectors
// c * SPLIT * SG + p * SG + l (c < VECS_PER_THREAD), so every load and store is a contiguous 16 * SG byte stripe.
// The whole row stays in registers between the sum-of-squares reduction and the normalization; the per-sub-group
// partial sums are combined through local memory when SPLIT > 1.
// All loads are issued up front with clamped indices and only the stores are predicated on the index being inside
// the row, so the memory round-trips of all chunks overlap instead of being serialized by per-chunk branches.
template <int VECS_PER_THREAD, int SPLIT>
struct CombineNormVecKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  // The kernel repeats these fields instead of embedding CombineNormArgs: a nested struct argument is ~25% slower.
  static constexpr int GROUP_SIZE = SG_SIZE * SPLIT;
  const bf16x8* residual;
  const bf16x8* block;
  const bf16* injection;
  const bf16x8* weight;
  bf16x8* combined;
  bf16x8* normalized;
  uint32_t hc;
  uint32_t vecs_per_row;
  bool shared_weight;
  float eps;
  float inv_hc;
  float inv_hidden;
  sycl::local_accessor<float, 1> partials;

  void sycl_ker_config_convention(sycl::handler& cgh) {
    partials = sycl::local_accessor<float, 1>(SPLIT, cgh);
  }

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    const auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int sg_id = sg.get_group_linear_id();
    const uint32_t row = item.get_group(0);
    const uint32_t token = row / hc;
    const uint32_t stream = row - token * hc;
    const int first = sg_id * SG_SIZE + lane;
    const int last = static_cast<int>(vecs_per_row) - 1;
    constexpr int STRIDE = SG_SIZE * SPLIT;

    const bf16x8* residual_row = residual + row * vecs_per_row;
    const bf16x8* block_row = block + token * vecs_per_row;
    const bf16x8* weight_row = weight + (shared_weight ? 0 : stream * vecs_per_row);
    bf16x8 values[VECS_PER_THREAD], blocks[VECS_PER_THREAD], weights[VECS_PER_THREAD];
#pragma unroll
    for (int c = 0; c < VECS_PER_THREAD; ++c) {
      const int index = sycl::min(c * STRIDE + first, last);
      values[c] = residual_row[index];
      blocks[c] = block_row[index];
      weights[c] = weight_row[index];
    }
    const float w = combine_weight_dynamic(injection, row, inv_hc);

    // `combined` may alias `residual`; each lane reads its own vectors before writing them.
    bf16x8* combined_row = combined + row * vecs_per_row;
    float sum_sq = 0.0f;
#pragma unroll
    for (int c = 0; c < VECS_PER_THREAD; ++c) {
      const int index = c * STRIDE + first;
      if (index > last) continue;
      values[c] = combine_values<VEC_SIZE>(values[c], blocks[c], w);
      combined_row[index] = values[c];
      float mv[VEC_SIZE];
      to_float<VEC_SIZE>(values[c], mv);
#pragma unroll
      for (int k = 0; k < VEC_SIZE; ++k) {
        sum_sq += mv[k] * mv[k];
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
        sum_sq += partials[other];
      }
    }
    const float scale = sycl::rsqrt(sum_sq * inv_hidden + eps);

    bf16x8* normalized_row = normalized + row * vecs_per_row;
#pragma unroll
    for (int c = 0; c < VECS_PER_THREAD; ++c) {
      const int index = c * STRIDE + first;
      if (index > last) continue;
      float v[VEC_SIZE], wv[VEC_SIZE], out[VEC_SIZE];
      to_float<VEC_SIZE>(values[c], v);
      to_float<VEC_SIZE>(weights[c], wv);
#pragma unroll
      for (int k = 0; k < VEC_SIZE; ++k) {
        out[k] = v[k] * scale * (1.0f + wv[k]);
      }
      normalized_row[index] = to_bf16<VEC_SIZE>(out);
    }
  }
};

// One work-group per (token, stream) row for shapes the vector kernel does not cover.
struct CombineNormScalarKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
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
  float inv_hc;
  float inv_hidden;
  sycl::local_accessor<float, 1> scratch;

  void sycl_ker_config_convention(sycl::handler& cgh) {
    scratch = sycl::local_accessor<float, 1>(1, cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int64_t row = item.get_group(0);
    const int64_t token = row / hc;
    const int64_t stream = row % hc;
    const int64_t base = row * hidden;
    const float w = combine_weight_dynamic(injection, row, inv_hc);
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
      scratch[0] = sycl::rsqrt(sum_sq * inv_hidden + eps);
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
template <int VECS_PER_THREAD, int SPLIT>
bool launch_combine_norm_vec(sycl::queue& q, const CombineNormArgs& args) {
  using Kernel = CombineNormVecKernel<VECS_PER_THREAD, SPLIT>;
  if (args.vecs_per_row > VECS_PER_THREAD * SG_SIZE * SPLIT) return false;
  sycl_kernel_submit(
      static_cast<int64_t>(args.rows) * Kernel::GROUP_SIZE,
      static_cast<int64_t>(Kernel::GROUP_SIZE),
      q,
      Kernel{
          {},
          args.residual,
          args.block,
          args.injection,
          args.weight,
          args.combined,
          args.normalized,
          args.hc,
          args.vecs_per_row,
          args.shared_weight,
          args.eps,
          args.inv_hc,
          args.inv_hidden,
          {}});
  return true;
}

}  // namespace

namespace at::native::xpu {

SGL_KERNEL_EXPORT void qwen4_hc_silu(torch::Tensor& output, const torch::Tensor& input, int64_t hc_count) {
  check_bf16_xpu(input, "input");
  check_bf16_xpu(output, "output");
  TORCH_CHECK(output.sizes() == input.sizes(), "output must have the same shape as input");
  TORCH_CHECK(hc_count > 0, "hc_count must be positive");
  check_no_partial_overlap(output, input);
  const int64_t n = input.numel();
  if (n == 0) return;
  TORCH_CHECK(n <= MAX_ID, "qwen4_hc_silu: input too large: ", input.sizes());

  c10::DeviceGuard guard(input.device());
  auto q = c10::xpu::getCurrentXPUStream().queue();
  const float inv_hc = 1.0f / static_cast<float>(hc_count);
  const int max_v = n <= NARROW_MAX_OUTPUTS ? NARROW_VEC : VEC_SIZE;
  dispatch_vec(pick_vec(n, {&input, &output}, max_v), [&](auto vec) {
    constexpr int V = decltype(vec)::value;
    const uint32_t items = n / V;
    launch_flat(q, items, SiluKernel<V>{as<bf16xV<V>>(input), as_mut<bf16xV<V>>(output), items, inv_hc});
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
  const int max_v = rows * hidden <= NARROW_MAX_OUTPUTS ? NARROW_VEC : VEC_SIZE;
  dispatch_vec(pick_vec(hidden, {&input, &gate, &output}, max_v), [&](auto vec) {
    constexpr int V = decltype(vec)::value;
    const uint32_t vecs_per_row = hidden / V;
    launch_rows(
        q,
        rows,
        vecs_per_row,
        GateMixKernel<V>{
            as<bf16xV<V>>(input),
            as<bf16xV<V>>(gate),
            as_mut<bf16xV<V>>(output),
            vecs_per_row,
            static_cast<uint32_t>(hc_count),
            1.0f / static_cast<float>(hc_count)});
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
  check_no_partial_overlap(output, residual);
  check_no_overlap(output, {&block_output});
  if (injection_data != nullptr) check_no_overlap(output, {&*injection});
  if (rows == 0 || hidden == 0) return;
  TORCH_CHECK(residual.numel() <= MAX_ID, "qwen4_hc_combine: input too large: ", residual.sizes());

  c10::DeviceGuard guard(residual.device());
  auto q = c10::xpu::getCurrentXPUStream().queue();
  dispatch_vec(pick_vec(hidden, {&residual, &block_output, &output}), [&](auto vec) {
    constexpr int V = decltype(vec)::value;
    const uint32_t vecs_per_row = hidden / V;
    const auto launch = [&](auto has_injection) {
      launch_rows(
          q,
          rows * hc_count,
          vecs_per_row,
          CombineKernel<V, decltype(has_injection)::value>{
              as<bf16xV<V>>(residual),
              as<bf16xV<V>>(block_output),
              injection_data,
              as_mut<bf16xV<V>>(output),
              vecs_per_row,
              static_cast<uint32_t>(hc_count),
              1.0f / static_cast<float>(hc_count)});
    };
    if (injection_data != nullptr) {
      launch(std::true_type{});
    } else {
      launch(std::false_type{});
    }
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
  check_no_partial_overlap(combined, residual);
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
  // Both kernels must compute the same combine weight as qwen4_hc_combine.
  const float inv_hc = 1.0f / static_cast<float>(hc_count);
  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  if (can_vectorize(hidden, {&residual, &block_output, &weight, &combined, &normalized})) {
    const CombineNormArgs args{
        as<bf16x8>(residual),
        as<bf16x8>(block_output),
        injection_data,
        as<bf16x8>(weight),
        as_mut<bf16x8>(combined),
        as_mut<bf16x8>(normalized),
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(hc_count),
        static_cast<uint32_t>(hidden / VEC_SIZE),
        shared_weight,
        eps_f,
        inv_hc,
        inv_hidden};
    // One row per work-group. By default 10 sub-groups per row with 2 vectors per lane; 4 sub-groups with 5 vectors
    // in a middle range and 20 sub-groups with 1 vector for many rows, where those fill the GPU better.
    bool launched;
    if (rows >= COMBINE_NORM_MID_ROWS_MIN && rows <= COMBINE_NORM_MID_ROWS_MAX) {
      launched = launch_combine_norm_vec<5, 4>(q, args);
    } else if (rows >= COMBINE_NORM_MANY_ROWS) {
      launched = launch_combine_norm_vec<1, 20>(q, args);
    } else {
      launched = launch_combine_norm_vec<2, 10>(q, args);
    }
    if (launched) return;
  }
  sycl_kernel_submit(
      rows * FLAT_WG,
      static_cast<int64_t>(FLAT_WG),
      q,
      CombineNormScalarKernel{
          {},
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
          inv_hc,
          inv_hidden,
          {}});
}

}  // namespace at::native::xpu
