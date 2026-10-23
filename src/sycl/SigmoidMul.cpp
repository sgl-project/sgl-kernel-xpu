#include <ATen/MemoryOverlap.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

constexpr int VEC_BYTES = 16;
constexpr int WG_SIZE = 128;
constexpr int SG_SIZE = 16;
constexpr int64_t MAX_ROWS = std::numeric_limits<int32_t>::max();  // SYCL id queries are 32-bit
constexpr int64_t MAX_HIDDEN = std::numeric_limits<int32_t>::max() / 2;

// out[r, c] = input[r, c] * T(sigmoid(gate[r, c / head_dim, c % head_dim])). The sigmoid is rounded to T
// before the multiply like eager `input * torch.sigmoid(gate)`; native exp/recip still round to the same
// bf16/fp16 sigmoid for every input value, so the result is bit-exact with eager.
template <typename T, int V>
struct SigmoidMulKernel {
  using Vec = aligned_vector_loop<T, V>;
  const T* input;
  const T* gate;
  T* output;
  uint32_t vecs_per_row;
  uint32_t head_dim;
  int64_t gate_stride_row;
  int64_t gate_stride_head;

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<2> item) const {
    const uint32_t v = item.get_global_id(1);
    if (v >= vecs_per_row) {
      return;
    }
    const int64_t row = item.get_group(0);
    const uint32_t col = v * V;
    const uint32_t head = col / head_dim;
    const int64_t g_off = row * gate_stride_row + head * gate_stride_head + (col - head * head_dim);
    const int64_t x_off = row * vecs_per_row * V + col;
    const Vec x = *reinterpret_cast<const Vec*>(input + x_off);
    const Vec g = *reinterpret_cast<const Vec*>(gate + g_off);
    Vec y;
#pragma unroll
    for (int i = 0; i < V; ++i) {
      const float s =
          static_cast<float>(static_cast<T>(sycl::native::recip(1.f + sycl::native::exp(-static_cast<float>(g[i])))));
      y[i] = static_cast<T>(static_cast<float>(x[i]) * s);
    }
    *reinterpret_cast<Vec*>(output + x_off) = y;
  }
};

template <typename T, int V>
void launch(
    const torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& gate,
    int64_t rows,
    int64_t hidden,
    int64_t head_dim,
    int64_t gate_stride_row,
    int64_t gate_stride_head) {
  const int64_t vecs_per_row = hidden / V;
  const int64_t wg_size = std::min<int64_t>(WG_SIZE, (vecs_per_row + SG_SIZE - 1) / SG_SIZE * SG_SIZE);
  const int64_t col_groups = (vecs_per_row + wg_size - 1) / wg_size;
  auto q = c10::xpu::getCurrentXPUStream().queue();
  sycl_kernel_submit(
      sycl::range<2>(rows, col_groups * wg_size),
      sycl::range<2>(1, wg_size),
      q,
      SigmoidMulKernel<T, V>{
          reinterpret_cast<const T*>(input.data_ptr()),
          reinterpret_cast<const T*>(gate.data_ptr()),
          reinterpret_cast<T*>(output.data_ptr()),
          static_cast<uint32_t>(vecs_per_row),
          static_cast<uint32_t>(head_dim),
          gate_stride_row,
          gate_stride_head});
}

template <typename T>
void dispatch_vec(
    const torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& gate,
    int64_t rows,
    int64_t hidden,
    int64_t head_dim,
    int64_t gate_stride_row,
    int64_t gate_stride_head) {
  constexpr int V = VEC_BYTES / sizeof(T);
  auto aligned = [](const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % VEC_BYTES == 0; };
  if (head_dim % V == 0 && gate_stride_row % V == 0 && gate_stride_head % V == 0 && aligned(output) && aligned(input) &&
      aligned(gate)) {
    launch<T, V>(output, input, gate, rows, hidden, head_dim, gate_stride_row, gate_stride_head);
  } else {
    launch<T, 1>(output, input, gate, rows, hidden, head_dim, gate_stride_row, gate_stride_head);
  }
}

}  // namespace

namespace at::native::xpu {

// output[N, H] = input[N, H] * sigmoid(gate), gate is [N, H] or [N, heads, head_dim] with heads * head_dim == H.
// The gate may be strided (e.g. a chunk of a q/gate projection); its last dim must be contiguous.
SGL_KERNEL_EXPORT void fused_sigmoid_mul(torch::Tensor& output, torch::Tensor& input, torch::Tensor& gate) {
  CHECK_INPUT(output);
  CHECK_INPUT(input);
  CHECK_DEVICE(gate);
  TORCH_CHECK(input.dim() == 2, "input must have shape [N, H]");
  TORCH_CHECK(output.sizes() == input.sizes(), "output must match input's shape");
  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
      "input must be bfloat16 or float16, got ",
      input.scalar_type());
  TORCH_CHECK(
      output.scalar_type() == input.scalar_type() && gate.scalar_type() == input.scalar_type(),
      "input, gate and output must have the same dtype");
  TORCH_CHECK(
      gate.device() == input.device() && output.device() == input.device(),
      "input, gate and output must be on the same device");
  const int64_t rows = input.size(0), hidden = input.size(1);
  TORCH_CHECK(gate.size(0) == rows, "gate must have the same number of rows as input");
  TORCH_CHECK(gate.stride(-1) == 1, "gate must be contiguous in the last dim");
  int64_t head_dim, gate_stride_head;
  if (gate.dim() == 2) {
    TORCH_CHECK(gate.size(1) == hidden, "gate must have shape [N, H]");
    head_dim = hidden;
    gate_stride_head = 0;
  } else {
    TORCH_CHECK(gate.dim() == 3 && gate.size(1) * gate.size(2) == hidden, "gate must have shape [N, heads, head_dim]");
    head_dim = gate.size(2);
    gate_stride_head = gate.stride(1);
  }
  at::assert_no_partial_overlap(output, input);
  TORCH_CHECK(
      !output.is_alias_of(gate) || at::get_overlap_status(output, gate) != at::MemOverlapStatus::TooHard,
      "output and gate must not share storage when overlap cannot be determined");
  at::assert_no_overlap(output, gate);
  if (rows == 0 || hidden == 0) {
    return;
  }
  TORCH_CHECK(rows <= MAX_ROWS && hidden <= MAX_HIDDEN, "fused_sigmoid_mul: input too large: ", input.sizes());

  c10::DeviceGuard guard(input.device());
  if (input.scalar_type() == at::kBFloat16) {
    dispatch_vec<sycl::ext::oneapi::bfloat16>(
        output, input, gate, rows, hidden, head_dim, gate.stride(0), gate_stride_head);
  } else {
    dispatch_vec<sycl::half>(output, input, gate, rows, hidden, head_dim, gate.stride(0), gate_stride_head);
  }
}

}  // namespace at::native::xpu
