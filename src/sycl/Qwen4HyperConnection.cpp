// Qwen3.8-Flash-Next HyperConnection elementwise ops (bf16), ported from DeepKloX csrc/hc.
//   qwen4_hc_silu:     out = silu(x / hc)
//   qwen4_hc_gate_mix: out[t, d] = mean_s(sigmoid(gate[t, s, d]) * x[t, s, d])
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

}  // namespace

namespace at::native::xpu {

SGL_KERNEL_EXPORT void qwen4_hc_silu(torch::Tensor& output, const torch::Tensor& input, int64_t hc_count) {
  check_bf16_xpu(input, "input");
  check_bf16_xpu(output, "output");
  TORCH_CHECK(output.sizes() == input.sizes(), "output must have the same shape as input");
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

}  // namespace at::native::xpu
