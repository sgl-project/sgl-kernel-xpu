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

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;
using Chunk = sycl::vec<uint32_t, 4>;  // 8 bf16 values, one 16-byte access

constexpr int CHUNK_VALUES = 8;
constexpr int SG_SIZE = 16;
constexpr int SCALAR_WG = 256;
constexpr int64_t MAX_ID = std::numeric_limits<int32_t>::max();  // SYCL id queries are 32-bit
// silu inputs below this many values use the scalar kernel, which has lower latency at decode sizes.
constexpr int64_t SILU_VEC_MIN_VALUES = 1 << 16;
// combine calls with fewer tokens use one chunk per work-item.
constexpr int64_t COMBINE_FEW_TOKENS = 64;
// combine_norm calls with fewer (token, stream) rows use the decode configuration.
constexpr int64_t COMBINE_NORM_FEW_ROWS = 256;

inline float sigmoid(float v) {
  return 1.0f / (1.0f + sycl::exp(-v));
}

inline bool aligned16(const void* p) {
  return reinterpret_cast<uintptr_t>(p) % sizeof(Chunk) == 0;
}

inline void unpack(const Chunk& c, float (&v)[CHUNK_VALUES]) {
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    v[2 * w] = sycl::bit_cast<float>(c[w] << 16);
    v[2 * w + 1] = sycl::bit_cast<float>(c[w] & 0xffff0000u);
  }
}

inline uint32_t bf16_bits(float v) {
  return sycl::bit_cast<uint16_t>(static_cast<bf16>(v));
}

inline Chunk pack(const float (&v)[CHUNK_VALUES]) {
  Chunk c;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    c[w] = bf16_bits(v[2 * w]) | (bf16_bits(v[2 * w + 1]) << 16);
  }
  return c;
}

// Vectorized kernels use a 2D range: dim 0 is a row, dim 1 walks the hidden chunks, `ITEMS` chunks per work-item
// spaced one global row apart so each sub-group access stays contiguous.
struct SiluVecKernel {
  const Chunk* input;
  Chunk* output;
  int64_t items;
  float divisor;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t i = item.get_global_id(0);
    if (i >= items) return;
    float v[CHUNK_VALUES];
    unpack(input[i], v);
#pragma unroll
    for (int k = 0; k < CHUNK_VALUES; ++k) {
      const float s = v[k] / divisor;
      v[k] = s * sigmoid(s);
    }
    output[i] = pack(v);
  }
};

struct SiluScalarKernel {
  const bf16* input;
  bf16* output;
  int64_t elements;
  float divisor;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t i = item.get_global_id(0);
    if (i >= elements) return;
    const float s = static_cast<float>(input[i]) / divisor;
    output[i] = static_cast<bf16>(s * sigmoid(s));
  }
};

template <int HC, int ITEMS>
struct GateMixVecKernel {
  const Chunk* input;
  const Chunk* gate;
  Chunk* output;
  int hidden_chunks;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<2> item) const {
    const int64_t token = item.get_global_id(0);
    const int first = item.get_global_id(1);
    const int step = item.get_global_range(1);
    const int64_t base = token * HC * hidden_chunks;
    Chunk xs[ITEMS][HC];
    Chunk gs[ITEMS][HC];
#pragma unroll
    for (int p = 0; p < ITEMS; ++p) {
      const int c = first + p * step;
      if (c < hidden_chunks) {
#pragma unroll
        for (int s = 0; s < HC; ++s) {
          xs[p][s] = input[base + s * hidden_chunks + c];
          gs[p][s] = gate[base + s * hidden_chunks + c];
        }
      }
    }
#pragma unroll
    for (int p = 0; p < ITEMS; ++p) {
      const int c = first + p * step;
      if (c < hidden_chunks) {
        float acc[CHUNK_VALUES] = {};
#pragma unroll
        for (int s = 0; s < HC; ++s) {
          float xv[CHUNK_VALUES], gv[CHUNK_VALUES];
          unpack(xs[p][s], xv);
          unpack(gs[p][s], gv);
#pragma unroll
          for (int k = 0; k < CHUNK_VALUES; ++k) {
            acc[k] += sigmoid(gv[k]) * xv[k];
          }
        }
#pragma unroll
        for (int k = 0; k < CHUNK_VALUES; ++k) {
          acc[k] /= static_cast<float>(HC);
        }
        output[token * hidden_chunks + c] = pack(acc);
      }
    }
  }
};

struct GateMixScalarKernel {
  const bf16* input;
  const bf16* gate;
  bf16* output;
  int64_t elements;
  int64_t hidden;
  int64_t hc;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t i = item.get_global_id(0);
    if (i >= elements) return;
    const int64_t row_base = (i / hidden) * hidden * hc;
    const int64_t d = i % hidden;
    float acc = 0.0f;
    for (int64_t s = 0; s < hc; ++s) {
      const int64_t o = row_base + s * hidden + d;
      acc += sigmoid(static_cast<float>(gate[o])) * static_cast<float>(input[o]);
    }
    output[i] = static_cast<bf16>(acc / static_cast<float>(hc));
  }
};

// Dim 0 walks the [tokens * HC] rows of `residual`, so each sub-group streams one contiguous row stripe; the block
// row is re-read per stream from cache.
template <int HC, int ITEMS>
struct CombineVecKernel {
  const Chunk* residual;
  const Chunk* block;
  const bf16* injection;
  Chunk* output;
  int hidden_chunks;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<2> item) const {
    const int64_t row = item.get_global_id(0);
    const int64_t token = row / HC;
    const int first = item.get_global_id(1);
    const int step = item.get_global_range(1);
    Chunk rs[ITEMS];
    Chunk bs[ITEMS];
#pragma unroll
    for (int p = 0; p < ITEMS; ++p) {
      const int c = first + p * step;
      if (c < hidden_chunks) {
        rs[p] = residual[row * hidden_chunks + c];
        bs[p] = block[token * hidden_chunks + c];
      }
    }
    const float w = injection == nullptr ? 1.0f : 2.0f * sigmoid(static_cast<float>(injection[row]) / HC);
#pragma unroll
    for (int p = 0; p < ITEMS; ++p) {
      const int c = first + p * step;
      if (c < hidden_chunks) {
        float rv[CHUNK_VALUES], bv[CHUNK_VALUES];
        unpack(rs[p], rv);
        unpack(bs[p], bv);
#pragma unroll
        for (int k = 0; k < CHUNK_VALUES; ++k) {
          rv[k] += bv[k] * w;
        }
        output[row * hidden_chunks + c] = pack(rv);
      }
    }
  }
};

struct CombineScalarKernel {
  const bf16* residual;
  const bf16* block;
  const bf16* injection;
  bf16* output;
  int64_t elements;
  int64_t hidden;
  int64_t hc;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t i = item.get_global_id(0);
    if (i >= elements) return;
    const int64_t row_width = hidden * hc;
    const int64_t token = i / row_width;
    const int64_t stream = (i % row_width) / hidden;
    const int64_t d = i % hidden;
    const float w = injection == nullptr
                        ? 1.0f
                        : 2.0f * sigmoid(static_cast<float>(injection[token * hc + stream]) / static_cast<float>(hc));
    output[i] = static_cast<bf16>(static_cast<float>(residual[i]) + static_cast<float>(block[token * hidden + d]) * w);
  }
};

template <typename Kernel>
void launch_scalar(sycl::queue& q, int64_t elements, const Kernel& kernel) {
  const int64_t groups = (elements + SCALAR_WG - 1) / SCALAR_WG;
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * SCALAR_WG), sycl::range<1>(SCALAR_WG)), kernel);
  });
}

template <typename Kernel>
void launch_rows(sycl::queue& q, int64_t rows, int hidden_chunks, int items, int group, const Kernel& kernel) {
  const int64_t per_row = (hidden_chunks + items - 1) / items;
  const int64_t columns = (per_row + group - 1) / group * group;
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(rows, columns), sycl::range<2>(1, group)), kernel);
  });
}

void check_bf16_xpu(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_xpu(), name, " must be on XPU");
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bfloat16");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

const bf16* bf16_ptr(const at::Tensor& t) {
  return reinterpret_cast<const bf16*>(t.data_ptr());
}
bf16* bf16_ptr_mut(at::Tensor& t) {
  return reinterpret_cast<bf16*>(t.data_ptr());
}
const Chunk* chunk_ptr(const at::Tensor& t) {
  return reinterpret_cast<const Chunk*>(t.data_ptr());
}
Chunk* chunk_ptr_mut(at::Tensor& t) {
  return reinterpret_cast<Chunk*>(t.data_ptr());
}

// combined = bf16(residual + block * w); normalized = bf16(combined * rsqrt(mean(combined^2) + eps) * (1 + weight)),
// both per (token, stream) row, i.e. the combine followed by the next grouped Gemma RMSNorm.
//
// Each work-group owns ROWS_PER_GROUP rows of SPLIT sub-groups each. Lane l of part p handles chunks
// c * SPLIT * SG + p * SG + l, so every load and store is a contiguous 16 * SG byte stripe. The whole row stays in
// registers between the sum-of-squares reduction and the normalization.
template <int CHUNKS, int SPLIT, int ROWS_PER_GROUP>
struct CombineNormVecKernel {
  static constexpr int GROUP_SIZE = SG_SIZE * SPLIT * ROWS_PER_GROUP;
  const Chunk* residual;
  const Chunk* block;
  const bf16* injection;
  const Chunk* weight;
  Chunk* combined;
  Chunk* normalized;
  int64_t rows;
  int hc;
  int row_chunks;
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

    const Chunk* residual_row = residual + safe_row * row_chunks;
    const Chunk* block_row = block + token * row_chunks;
    Chunk* combined_row = combined + safe_row * row_chunks;
    const float w = injection == nullptr ? 1.0f : 2.0f * sigmoid(static_cast<float>(injection[safe_row]) / hc);

    Chunk values[CHUNKS];
    float sum_sq = 0.0f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const int index = c * STRIDE + first;
      if (active && index < row_chunks) {
        float rv[CHUNK_VALUES], bv[CHUNK_VALUES];
        unpack(residual_row[index], rv);
        unpack(block_row[index], bv);
#pragma unroll
        for (int k = 0; k < CHUNK_VALUES; ++k) {
          rv[k] += bv[k] * w;
        }
        values[c] = pack(rv);
        combined_row[index] = values[c];
        float mv[CHUNK_VALUES];
        unpack(values[c], mv);
#pragma unroll
        for (int k = 0; k < CHUNK_VALUES; ++k) {
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
    const float scale = sycl::rsqrt(sum_sq / static_cast<float>(row_chunks * CHUNK_VALUES) + eps);

    const Chunk* weight_row = weight + (shared_weight ? 0 : stream * row_chunks);
    Chunk* normalized_row = normalized + safe_row * row_chunks;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const int index = c * STRIDE + first;
      if (index < row_chunks) {
        float v[CHUNK_VALUES], wv[CHUNK_VALUES], out[CHUNK_VALUES];
        unpack(values[c], v);
        unpack(weight_row[index], wv);
#pragma unroll
        for (int k = 0; k < CHUNK_VALUES; ++k) {
          out[k] = v[k] * scale * (1.0f + wv[k]);
        }
        normalized_row[index] = pack(out);
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
template <int CHUNKS, int SPLIT, int ROWS_PER_GROUP>
bool launch_combine_norm_vec(
    sycl::queue& q,
    const at::Tensor& residual,
    const at::Tensor& block,
    const bf16* injection,
    const at::Tensor& weight,
    at::Tensor& combined,
    at::Tensor& normalized,
    int64_t rows,
    int64_t hc,
    int64_t hidden,
    bool shared_weight,
    float eps) {
  using Kernel = CombineNormVecKernel<CHUNKS, SPLIT, ROWS_PER_GROUP>;
  const int64_t row_chunks = hidden / CHUNK_VALUES;
  if (row_chunks > static_cast<int64_t>(CHUNKS) * SG_SIZE * SPLIT) return false;
  const int64_t groups = (rows + ROWS_PER_GROUP - 1) / ROWS_PER_GROUP;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partials(SPLIT * ROWS_PER_GROUP, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * Kernel::GROUP_SIZE), sycl::range<1>(Kernel::GROUP_SIZE)),
        Kernel{
            chunk_ptr(residual),
            chunk_ptr(block),
            injection,
            chunk_ptr(weight),
            chunk_ptr_mut(combined),
            chunk_ptr_mut(normalized),
            rows,
            static_cast<int>(hc),
            static_cast<int>(row_chunks),
            shared_weight,
            eps,
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
  TORCH_CHECK(output.device() == input.device(), "output and input must be on the same device");
  TORCH_CHECK(hc_count > 0, "hc_count must be positive");
  at::assert_no_partial_overlap(output, input);
  const int64_t n = input.numel();
  if (n == 0) return;
  TORCH_CHECK(n <= MAX_ID, "qwen4_hc_silu: input too large: ", input.sizes());

  c10::DeviceGuard guard(input.device());
  auto q = c10::xpu::getCurrentXPUStream().queue();
  const float divisor = static_cast<float>(hc_count);
  if (n >= SILU_VEC_MIN_VALUES && n % CHUNK_VALUES == 0 && aligned16(input.data_ptr()) &&
      aligned16(output.data_ptr())) {
    const int64_t items = n / CHUNK_VALUES;
    launch_scalar(q, items, SiluVecKernel{chunk_ptr(input), chunk_ptr_mut(output), items, divisor});
  } else {
    launch_scalar(q, n, SiluScalarKernel{bf16_ptr(input), bf16_ptr_mut(output), n, divisor});
  }
}

SGL_KERNEL_EXPORT void
qwen4_hc_gate_mix(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& gate, int64_t hc_count) {
  check_bf16_xpu(input, "input");
  check_bf16_xpu(gate, "gate");
  check_bf16_xpu(output, "output");
  TORCH_CHECK(input.dim() == 2 && gate.sizes() == input.sizes(), "input and gate must be [tokens, HC * hidden]");
  TORCH_CHECK(hc_count > 0 && input.size(1) % hc_count == 0, "HC * hidden must be divisible by hc_count");
  TORCH_CHECK(
      gate.device() == input.device() && output.device() == input.device(),
      "input, gate and output must be on the same device");
  const int64_t rows = input.size(0), hidden = input.size(1) / hc_count;
  TORCH_CHECK(
      output.dim() == 2 && output.size(0) == rows && output.size(1) == hidden, "output must be [tokens, hidden]");
  at::assert_no_overlap(output, input);
  at::assert_no_overlap(output, gate);
  if (rows == 0 || hidden == 0) return;
  TORCH_CHECK(rows <= MAX_ID && input.numel() <= MAX_ID, "qwen4_hc_gate_mix: input too large: ", input.sizes());

  c10::DeviceGuard guard(input.device());
  auto q = c10::xpu::getCurrentXPUStream().queue();
  if (hc_count == 4 && hidden % CHUNK_VALUES == 0 && aligned16(input.data_ptr()) && aligned16(gate.data_ptr()) &&
      aligned16(output.data_ptr())) {
    const int hidden_chunks = static_cast<int>(hidden / CHUNK_VALUES);
    launch_rows(
        q,
        rows,
        hidden_chunks,
        1,
        64,
        GateMixVecKernel<4, 1>{chunk_ptr(input), chunk_ptr(gate), chunk_ptr_mut(output), hidden_chunks});
  } else {
    launch_scalar(
        q,
        output.numel(),
        GateMixScalarKernel{bf16_ptr(input), bf16_ptr(gate), bf16_ptr_mut(output), output.numel(), hidden, hc_count});
  }
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
  TORCH_CHECK(
      block_output.device() == residual.device() && output.device() == residual.device(),
      "residual, block_output and output must be on the same device");
  const int64_t rows = residual.size(0), hidden = residual.size(1) / hc_count;
  TORCH_CHECK(block_output.size(0) == rows && block_output.size(1) == hidden, "block_output must be [tokens, hidden]");
  TORCH_CHECK(output.sizes() == residual.sizes(), "output must have the same shape as residual");
  const bf16* injection_ptr = nullptr;
  if (injection.has_value()) {
    check_bf16_xpu(*injection, "injection");
    TORCH_CHECK(
        injection->device() == residual.device() && injection->dim() == 2 && injection->size(0) == rows &&
            injection->size(1) == hc_count,
        "injection must be [tokens, hc_count] on the same device");
    at::assert_no_overlap(output, *injection);
    injection_ptr = bf16_ptr(*injection);
  }
  at::assert_no_partial_overlap(output, residual);
  at::assert_no_overlap(output, block_output);
  if (rows == 0 || hidden == 0) return;
  TORCH_CHECK(
      rows * hc_count <= MAX_ID && residual.numel() <= MAX_ID, "qwen4_hc_combine: input too large: ", residual.sizes());

  c10::DeviceGuard guard(residual.device());
  auto q = c10::xpu::getCurrentXPUStream().queue();
  if (hc_count == 4 && hidden % CHUNK_VALUES == 0 && aligned16(residual.data_ptr()) &&
      aligned16(block_output.data_ptr()) && aligned16(output.data_ptr())) {
    const int hidden_chunks = static_cast<int>(hidden / CHUNK_VALUES);
    if (rows < COMBINE_FEW_TOKENS) {
      launch_rows(
          q,
          rows * hc_count,
          hidden_chunks,
          1,
          128,
          CombineVecKernel<4, 1>{
              chunk_ptr(residual), chunk_ptr(block_output), injection_ptr, chunk_ptr_mut(output), hidden_chunks});
    } else {
      launch_rows(
          q,
          rows * hc_count,
          hidden_chunks,
          2,
          64,
          CombineVecKernel<4, 2>{
              chunk_ptr(residual), chunk_ptr(block_output), injection_ptr, chunk_ptr_mut(output), hidden_chunks});
    }
  } else {
    launch_scalar(
        q,
        residual.numel(),
        CombineScalarKernel{
            bf16_ptr(residual),
            bf16_ptr(block_output),
            injection_ptr,
            bf16_ptr_mut(output),
            residual.numel(),
            hidden,
            hc_count});
  }
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
      block_output.device() == residual.device() && weight.device() == residual.device() &&
          combined.device() == residual.device() && normalized.device() == residual.device(),
      "all tensors must be on the same device");
  TORCH_CHECK(
      block_output.size(0) == tokens && block_output.size(1) == hidden, "block_output must be [tokens, hidden]");
  TORCH_CHECK(
      weight.numel() == hidden || weight.numel() == residual.size(1),
      "weight must have hidden or HC * hidden elements");
  TORCH_CHECK(
      combined.sizes() == residual.sizes() && normalized.sizes() == residual.sizes(),
      "combined and normalized must have the same shape as residual");
  const bf16* injection_ptr = nullptr;
  if (injection.has_value()) {
    check_bf16_xpu(*injection, "injection");
    TORCH_CHECK(
        injection->device() == residual.device() && injection->dim() == 2 && injection->size(0) == tokens &&
            injection->size(1) == hc_count,
        "injection must be [tokens, hc_count] on the same device");
    at::assert_no_overlap(combined, *injection);
    at::assert_no_overlap(normalized, *injection);
    injection_ptr = bf16_ptr(*injection);
  }
  at::assert_no_partial_overlap(combined, residual);
  at::assert_no_overlap(combined, block_output);
  at::assert_no_overlap(normalized, residual);
  at::assert_no_overlap(normalized, block_output);
  at::assert_no_overlap(normalized, combined);
  at::assert_no_overlap(combined, weight);
  at::assert_no_overlap(normalized, weight);
  if (tokens == 0 || hidden == 0) return;
  const int64_t rows = tokens * hc_count;
  TORCH_CHECK(rows <= MAX_ID, "qwen4_hc_combine_norm: input too large: ", residual.sizes());

  c10::DeviceGuard guard(residual.device());
  auto q = c10::xpu::getCurrentXPUStream().queue();
  const bool shared_weight = weight.numel() == hidden;
  const float eps_f = static_cast<float>(eps);
  if (hidden % CHUNK_VALUES == 0 && aligned16(residual.data_ptr()) && aligned16(block_output.data_ptr()) &&
      aligned16(weight.data_ptr()) && aligned16(combined.data_ptr()) && aligned16(normalized.data_ptr())) {
    // Few rows (decode): 20 sub-groups per row, one chunk per lane, for latency. Many rows (prefill): 4 sub-groups
    // per row with 5 chunks per lane in registers and 4 rows per work-group.
    const bool launched = rows < COMBINE_NORM_FEW_ROWS ? launch_combine_norm_vec<1, 20, 1>(
                                                             q,
                                                             residual,
                                                             block_output,
                                                             injection_ptr,
                                                             weight,
                                                             combined,
                                                             normalized,
                                                             rows,
                                                             hc_count,
                                                             hidden,
                                                             shared_weight,
                                                             eps_f)
                                                       : launch_combine_norm_vec<5, 4, 4>(
                                                             q,
                                                             residual,
                                                             block_output,
                                                             injection_ptr,
                                                             weight,
                                                             combined,
                                                             normalized,
                                                             rows,
                                                             hc_count,
                                                             hidden,
                                                             shared_weight,
                                                             eps_f);
    if (launched) return;
  }
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> scratch(1, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(rows * SCALAR_WG), sycl::range<1>(SCALAR_WG)),
        CombineNormScalarKernel{
            bf16_ptr(residual),
            bf16_ptr(block_output),
            injection_ptr,
            bf16_ptr(weight),
            bf16_ptr_mut(combined),
            bf16_ptr_mut(normalized),
            hidden,
            hc_count,
            shared_weight,
            eps_f,
            scratch});
  });
}

}  // namespace at::native::xpu
