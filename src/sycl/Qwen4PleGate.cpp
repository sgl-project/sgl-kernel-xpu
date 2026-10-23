#include <ATen/MemoryOverlap.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int VEC = 8;  // 16-byte accesses for bf16
constexpr int ALIGN_BYTES = 16;
constexpr int MAX_WG_SIZE = 256;
constexpr int SG_SIZE = 16;
constexpr int VECS_PER_THREAD = 2;
constexpr float GATE_FLOOR = 1.0e-6f;
constexpr int64_t MAX_GRID_DIM = std::numeric_limits<int32_t>::max();  // SYCL id queries are 32-bit

template <typename T>
struct alignas(sizeof(T) * VEC) Vec8 {
  T v[VEC];
};
using bf16x8 = Vec8<bf16>;

inline float round_bf16(float f) {
  return static_cast<float>(static_cast<bf16>(f));
}

// sigmoid(sign(g) * sqrt(max(|g|, floor))) with a bf16 rounding after each step, mirroring the eager bf16 ops.
// NaN is propagated like eager; sycl::fmax would otherwise replace it with the floor.
inline float gate_activation(float gate) {
  if (gate != gate) {
    return gate;
  }
  const float magnitude = round_bf16(sycl::fmax(sycl::fabs(gate), GATE_FLOOR));
  const float root = round_bf16(sycl::sqrt(magnitude));
  const float sign = gate > 0.f ? 1.f : (gate < 0.f ? -1.f : 0.f);
  const float transformed = round_bf16(root * sign);
  return round_bf16(1.f / (1.f + sycl::exp(-transformed)));
}

// output[t, s, :] = gate_activation(gate[t, s]) * value[t, :]; one work-group per (token, stream).
struct GateValueKernel {
  const bf16* gate;
  const bf16* value;
  bf16* output;
  int64_t hc_count;
  int64_t hidden;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<2> item) const {
    const int64_t row = item.get_group(0);
    const int64_t token = row / hc_count;
    const float activated = gate_activation(static_cast<float>(gate[row]));

    const bf16x8* v = reinterpret_cast<const bf16x8*>(value + token * hidden);
    bf16x8* y = reinterpret_cast<bf16x8*>(output + row * hidden);
    const int64_t num_vecs = hidden / VEC;
    for (int64_t idx = item.get_local_id(1); idx < num_vecs; idx += item.get_local_range(1)) {
      const bf16x8 vv = v[idx];
      bf16x8 out;
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        out.v[i] = static_cast<bf16>(activated * static_cast<float>(vv.v[i]));
      }
      y[idx] = out;
    }
  }
};

// gate = bf16(bf16(sum(bf16(key * query))) * hidden^-0.5); output = gate_activation(gate) * value.
struct GateReduceKernel {
  const bf16* key;
  const bf16* query;
  const bf16* value;
  bf16* output;
  int64_t hc_count;
  int64_t hidden;
  float scale;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<2> item) const {
    const int64_t row = item.get_group(0);
    const int64_t token = row / hc_count;
    const int64_t lid = item.get_local_id(1);
    const int64_t wg_size = item.get_local_range(1);
    const int64_t num_vecs = hidden / VEC;

    const bf16x8* k = reinterpret_cast<const bf16x8*>(key + row * hidden);
    const bf16x8* q = reinterpret_cast<const bf16x8*>(query + row * hidden);
    float dot = 0.f;
    for (int64_t idx = lid; idx < num_vecs; idx += wg_size) {
      const bf16x8 kv = k[idx];
      const bf16x8 qv = q[idx];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        dot += round_bf16(static_cast<float>(kv.v[i]) * static_cast<float>(qv.v[i]));
      }
    }
    dot = sycl::reduce_over_group(item.get_group(), dot, sycl::plus<float>());
    const float activated = gate_activation(round_bf16(round_bf16(dot) * scale));

    const bf16x8* v = reinterpret_cast<const bf16x8*>(value + token * hidden);
    bf16x8* y = reinterpret_cast<bf16x8*>(output + row * hidden);
    for (int64_t idx = lid; idx < num_vecs; idx += wg_size) {
      const bf16x8 vv = v[idx];
      bf16x8 out;
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        out.v[i] = static_cast<bf16>(activated * static_cast<float>(vv.v[i]));
      }
      y[idx] = out;
    }
  }
};

int64_t pick_wg_size(int64_t hidden) {
  const int64_t per_thread = (hidden / VEC + VECS_PER_THREAD - 1) / VECS_PER_THREAD;
  return std::min<int64_t>(MAX_WG_SIZE, (per_thread + SG_SIZE - 1) / SG_SIZE * SG_SIZE);
}

template <typename Kernel>
void launch(const Kernel& kernel, int64_t rows, int64_t hidden) {
  const int64_t wg_size = pick_wg_size(hidden);
  auto q = c10::xpu::getCurrentXPUStream().queue();
  sycl_kernel_submit(sycl::range<2>(rows, wg_size), sycl::range<2>(1, wg_size), q, kernel);
}

void check_bf16_xpu(const torch::Tensor& t, const char* name) {
  CHECK_INPUT(t);
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bfloat16, got ", t.scalar_type());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % ALIGN_BYTES == 0, name, " must be 16-byte aligned");
}

}  // namespace

namespace at::native::xpu {

// output[N, HC, H] = gate_activation(gate[N, HC, 1]) * value[N, H]
SGL_KERNEL_EXPORT void qwen4_gate_value(torch::Tensor& output, torch::Tensor& gate, torch::Tensor& value) {
  check_bf16_xpu(output, "output");
  check_bf16_xpu(gate, "gate");
  check_bf16_xpu(value, "value");
  TORCH_CHECK(gate.dim() == 3 && gate.size(2) == 1, "gate must have shape [N, HC, 1]");
  TORCH_CHECK(value.dim() == 2 && value.size(0) == gate.size(0), "value must have shape [N, H]");
  const int64_t tokens = gate.size(0), hc_count = gate.size(1), hidden = value.size(1);
  TORCH_CHECK(hidden > 0 && hidden % VEC == 0, "hidden size must be a positive multiple of ", VEC);
  TORCH_CHECK(
      output.dim() == 3 && output.size(0) == tokens && output.size(1) == hc_count && output.size(2) == hidden,
      "output must have shape [N, HC, H]");
  TORCH_CHECK(
      gate.device() == value.device() && output.device() == value.device(),
      "gate, value and output must be on the same device");
  at::assert_no_overlap(output, gate);
  at::assert_no_overlap(output, value);
  const int64_t rows = tokens * hc_count;
  if (rows == 0) {
    return;
  }
  TORCH_CHECK(rows <= MAX_GRID_DIM, "qwen4_gate_value: too many rows: ", rows);

  c10::DeviceGuard guard(value.device());
  launch(
      GateValueKernel{
          reinterpret_cast<const bf16*>(gate.data_ptr()),
          reinterpret_cast<const bf16*>(value.data_ptr()),
          reinterpret_cast<bf16*>(output.data_ptr()),
          hc_count,
          hidden},
      rows,
      hidden);
}

// output[N, HC, H] = gate_activation(sum(key * query, -1) / sqrt(H)) * value[N, H]
SGL_KERNEL_EXPORT void
qwen4_gate_reduce(torch::Tensor& output, torch::Tensor& key, torch::Tensor& query, torch::Tensor& value) {
  check_bf16_xpu(output, "output");
  check_bf16_xpu(key, "key");
  check_bf16_xpu(query, "query");
  check_bf16_xpu(value, "value");
  TORCH_CHECK(key.dim() == 3, "key must have shape [N, HC, H]");
  TORCH_CHECK(query.sizes() == key.sizes() && output.sizes() == key.sizes(), "query and output must match key's shape");
  const int64_t tokens = key.size(0), hc_count = key.size(1), hidden = key.size(2);
  TORCH_CHECK(hidden > 0 && hidden % VEC == 0, "hidden size must be a positive multiple of ", VEC);
  TORCH_CHECK(value.dim() == 2 && value.size(0) == tokens && value.size(1) == hidden, "value must have shape [N, H]");
  TORCH_CHECK(
      query.device() == key.device() && value.device() == key.device() && output.device() == key.device(),
      "key, query, value and output must be on the same device");
  at::assert_no_partial_overlap(output, key);
  at::assert_no_partial_overlap(output, query);
  at::assert_no_overlap(output, value);
  const int64_t rows = tokens * hc_count;
  if (rows == 0) {
    return;
  }
  TORCH_CHECK(rows <= MAX_GRID_DIM, "qwen4_gate_reduce: too many rows: ", rows);

  c10::DeviceGuard guard(key.device());
  launch(
      GateReduceKernel{
          reinterpret_cast<const bf16*>(key.data_ptr()),
          reinterpret_cast<const bf16*>(query.data_ptr()),
          reinterpret_cast<const bf16*>(value.data_ptr()),
          reinterpret_cast<bf16*>(output.data_ptr()),
          hc_count,
          hidden,
          static_cast<float>(1.0 / std::sqrt(static_cast<double>(hidden)))},
      rows,
      hidden);
}

}  // namespace at::native::xpu
