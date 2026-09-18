// residual_output = residual + x * gate
// out = layer_norm(residual_output)[* weight + bias] * (1 + scale) + shift
// Ports sglang/python/sglang/kernels/ops/diffusion/norm/scale_residual_norm_scale_shift_triton.py

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>
#include <tuple>
#include <type_traits>
#include <vector>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace at::native::xpu {

namespace {

// FLOATING_TYPES_AND2 binds scalar_t to the SYCL types but WEIGHT_TYPES binds weight_t
// to the c10 ones; map them so T and P convert identically inside the kernel.
template <typename T>
using srnss_sycl_t = std::conditional_t<
    std::is_same_v<T, at::Half>,
    ::sycl::half,
    std::conditional_t<std::is_same_v<T, at::BFloat16>, ::sycl::ext::oneapi::bfloat16, T>>;

// Global access is vectorized by reinterpreting each row as aligned_vector_loop, the
// form RMSNorm.cpp uses; select_vec_size() on the host guarantees the alignment.
// sycl::vec::load/store with a global multi_ptr (as in per_tensor_quant_fp8.cpp) was
// measured 3.3x slower here for bf16 -- 119 vs 393 GB/s at S=32760 H=5120 -- because
// sycl::vec<bfloat16, N> scalarizes instead of emitting a 16-byte block load.
constexpr int kSrnssSubGroupSize = 16;
constexpr int kSrnssThreadsPerBlock = 256;

// Mirrors MAX_FUSED_HIDDEN in the Triton reference; also bounds row_vals below.
constexpr int kSrnssMaxHidden = 8192;

template <typename T, typename P, int kVecSize>
class ScaleResidualNormScaleShiftKernel {
 public:
  using Vec = aligned_vector_loop<T, kVecSize>;
  using PVec = aligned_vector_loop<P, kVecSize>;

  static constexpr int kMaxVecsPerThread = kSrnssMaxHidden / (kSrnssThreadsPerBlock * kVecSize);

  ScaleResidualNormScaleShiftKernel(
      T* residual_out,
      T* out,
      const T* residual,
      const T* x,
      const P* gate,
      const P* weight,
      const P* bias,
      const P* scale,
      const P* shift,
      uint32_t num_rows,
      uint32_t hidden,
      uint32_t gate_row_div,
      uint32_t gate_row_stride,
      bool scale_is_vec,
      bool shift_is_vec,
      float eps)
      : residual_out_(residual_out),
        out_(out),
        residual_(residual),
        x_(x),
        gate_(gate),
        weight_(weight),
        bias_(bias),
        scale_(scale),
        shift_(shift),
        num_rows_(num_rows),
        hidden_(hidden),
        gate_row_div_(gate_row_div),
        gate_row_stride_(gate_row_stride),
        scale_is_vec_(scale_is_vec),
        shift_is_vec_(shift_is_vec),
        eps_(eps) {}

  [[sycl::reqd_sub_group_size(kSrnssSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    const uint32_t row = static_cast<uint32_t>(item.get_group(0));
    if (row >= num_rows_) return;

    const uint32_t tid = static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t num_vecs = hidden_ / kVecSize;
    const size_t row_base = static_cast<size_t>(row) * hidden_;

    const Vec* x_vec = reinterpret_cast<const Vec*>(x_ + row_base);
    const Vec* residual_vec = reinterpret_cast<const Vec*>(residual_ + row_base);
    Vec* residual_out_vec = reinterpret_cast<Vec*>(residual_out_ + row_base);
    Vec* out_vec = reinterpret_cast<Vec*>(out_ + row_base);

    // gate_row_stride == 0 broadcasts a 3-D gate across every row.
    const PVec* gate_vec =
        gate_ == nullptr
            ? nullptr
            : reinterpret_cast<const PVec*>(gate_ + static_cast<size_t>(row / gate_row_div_) * gate_row_stride_);
    const PVec* weight_vec = weight_ == nullptr ? nullptr : reinterpret_cast<const PVec*>(weight_);
    const PVec* bias_vec = bias_ == nullptr ? nullptr : reinterpret_cast<const PVec*>(bias_);
    const PVec* scale_vec = reinterpret_cast<const PVec*>(scale_);
    const PVec* shift_vec = reinterpret_cast<const PVec*>(shift_);

    float row_vals[kMaxVecsPerThread][kVecSize];
    float thread_sum = 0.0f;

#pragma unroll
    for (int j = 0; j < kMaxVecsPerThread; ++j) {
      const uint32_t i = tid + static_cast<uint32_t>(j) * kSrnssThreadsPerBlock;
      if (i < num_vecs) {
        const Vec xv = x_vec[i];
        const Vec rv = residual_vec[i];
        PVec gv;
        if (gate_vec != nullptr) gv = gate_vec[i];
        Vec res_out;
#pragma unroll
        for (int v = 0; v < kVecSize; ++v) {
          float scaled = static_cast<float>(xv[v]);
          if (gate_vec != nullptr) scaled *= static_cast<float>(gv[v]);
          const float val = static_cast<float>(rv[v]) + scaled;
          row_vals[j][v] = val;
          thread_sum += val;
          res_out[v] = static_cast<T>(val);
        }
        residual_out_vec[i] = res_out;
      }
    }

    const float mean =
        ::sycl::reduce_over_group(item.get_group(), thread_sum, ::sycl::plus<float>()) / static_cast<float>(hidden_);

    float thread_var = 0.0f;
#pragma unroll
    for (int j = 0; j < kMaxVecsPerThread; ++j) {
      const uint32_t i = tid + static_cast<uint32_t>(j) * kSrnssThreadsPerBlock;
      if (i < num_vecs) {
#pragma unroll
        for (int v = 0; v < kVecSize; ++v) {
          const float centered = row_vals[j][v] - mean;
          row_vals[j][v] = centered;
          thread_var += centered * centered;
        }
      }
    }

    const float var =
        ::sycl::reduce_over_group(item.get_group(), thread_var, ::sycl::plus<float>()) / static_cast<float>(hidden_);
    // 1/sqrt, not rsqrt: matches the Triton reference.
    const float inv_std = 1.0f / ::sycl::sqrt(var + eps_);

    const float scalar_scale = scale_is_vec_ ? 0.0f : static_cast<float>(scale_[0]);
    const float scalar_shift = shift_is_vec_ ? 0.0f : static_cast<float>(shift_[0]);

#pragma unroll
    for (int j = 0; j < kMaxVecsPerThread; ++j) {
      const uint32_t i = tid + static_cast<uint32_t>(j) * kSrnssThreadsPerBlock;
      if (i < num_vecs) {
        PVec wv, bv, scv, shv;
        if (weight_vec != nullptr) wv = weight_vec[i];
        if (bias_vec != nullptr) bv = bias_vec[i];
        if (scale_is_vec_) scv = scale_vec[i];
        if (shift_is_vec_) shv = shift_vec[i];

        Vec ov;
#pragma unroll
        for (int v = 0; v < kVecSize; ++v) {
          float normed = row_vals[j][v] * inv_std;
          if (weight_vec != nullptr) normed *= static_cast<float>(wv[v]);
          if (bias_vec != nullptr) normed += static_cast<float>(bv[v]);
          const float sc = scale_is_vec_ ? static_cast<float>(scv[v]) : scalar_scale;
          const float sh = shift_is_vec_ ? static_cast<float>(shv[v]) : scalar_shift;
          ov[v] = static_cast<T>(normed * (1.0f + sc) + sh);
        }
        out_vec[i] = ov;
      }
    }
  }

 private:
  T* residual_out_;
  T* out_;
  const T* residual_;
  const T* x_;
  const P* gate_;
  const P* weight_;
  const P* bias_;
  const P* scale_;
  const P* shift_;
  uint32_t num_rows_;
  uint32_t hidden_;
  uint32_t gate_row_div_;
  uint32_t gate_row_stride_;
  bool scale_is_vec_;
  bool shift_is_vec_;
  float eps_;
};

struct VectorOperand {
  const void* data;
  int64_t element_size;
};

int32_t select_vec_size(int64_t hidden, const std::vector<VectorOperand>& operands) {
  int32_t vec_size = (hidden % 8 == 0) ? 8 : (hidden % 4 == 0) ? 4 : (hidden % 2 == 0) ? 2 : 1;
  while (vec_size > 1) {
    bool aligned = true;
    for (const auto& operand : operands) {
      const int64_t alignment = vec_size * operand.element_size;
      if (reinterpret_cast<uintptr_t>(operand.data) % static_cast<uintptr_t>(alignment) != 0) {
        aligned = false;
        break;
      }
    }
    if (aligned) break;
    vec_size /= 2;
  }
  return vec_size;
}

struct LaunchArgs {
  void* residual_out;
  void* out;
  const void* residual;
  const void* x;
  const void* gate;
  const void* weight;
  const void* bias;
  const void* scale;
  const void* shift;
  int64_t num_rows;
  int64_t hidden;
  int64_t gate_row_div;
  int64_t gate_row_stride;
  int32_t scale_is_vec;
  int32_t shift_is_vec;
  int32_t vec_size;
  float eps;
};

template <typename T, typename P>
void launch(::sycl::queue& queue, const LaunchArgs& a) {
  if (a.num_rows <= 0 || a.hidden <= 0) return;

#define SRNSS_SUBMIT(VEC_SIZE)                                 \
  sycl_kernel_submit(                                          \
      static_cast<size_t>(a.num_rows) * kSrnssThreadsPerBlock, \
      kSrnssThreadsPerBlock,                                   \
      queue,                                                   \
      ScaleResidualNormScaleShiftKernel<T, P, VEC_SIZE>(       \
          static_cast<T*>(a.residual_out),                     \
          static_cast<T*>(a.out),                              \
          static_cast<const T*>(a.residual),                   \
          static_cast<const T*>(a.x),                          \
          static_cast<const P*>(a.gate),                       \
          static_cast<const P*>(a.weight),                     \
          static_cast<const P*>(a.bias),                       \
          static_cast<const P*>(a.scale),                      \
          static_cast<const P*>(a.shift),                      \
          static_cast<uint32_t>(a.num_rows),                   \
          static_cast<uint32_t>(a.hidden),                     \
          static_cast<uint32_t>(a.gate_row_div),               \
          static_cast<uint32_t>(a.gate_row_stride),            \
          a.scale_is_vec != 0,                                 \
          a.shift_is_vec != 0,                                 \
          a.eps))

  switch (a.vec_size) {
    case 8:
      SRNSS_SUBMIT(8);
      break;
    case 4:
      SRNSS_SUBMIT(4);
      break;
    case 2:
      SRNSS_SUBMIT(2);
      break;
    default:
      SRNSS_SUBMIT(1);
      break;
  }
#undef SRNSS_SUBMIT
}

void check_param(const torch::Tensor& t, const torch::Tensor& x, at::ScalarType param_dtype, const char* name) {
  CHECK_INPUT(t);
  TORCH_CHECK(t.device() == x.device(), "fused_scale_residual_norm_scale_shift: ", name, " must be on x's device");
  TORCH_CHECK(
      t.scalar_type() == param_dtype,
      "fused_scale_residual_norm_scale_shift: gate/weight/bias/scale/shift must share one dtype, but ",
      name,
      " is ",
      t.scalar_type(),
      " and another is ",
      param_dtype);
}

}  // namespace

SGL_KERNEL_EXPORT std::tuple<torch::Tensor, torch::Tensor> fused_scale_residual_norm_scale_shift(
    torch::Tensor residual,
    torch::Tensor x,
    std::optional<torch::Tensor> gate,
    std::optional<torch::Tensor> weight,
    std::optional<torch::Tensor> bias,
    torch::Tensor scale,
    torch::Tensor shift,
    double eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(residual);
  TORCH_CHECK(
      x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half ||
          x.scalar_type() == at::ScalarType::BFloat16,
      "fused_scale_residual_norm_scale_shift: x must be float32, float16 or bfloat16, got ",
      x.scalar_type());
  TORCH_CHECK(
      x.dim() == 3 && x.size(0) == 1,
      "fused_scale_residual_norm_scale_shift: x must be [1, seq_len, hidden], got ",
      x.sizes());
  TORCH_CHECK(
      residual.sizes() == x.sizes() && residual.scalar_type() == x.scalar_type() && residual.device() == x.device(),
      "fused_scale_residual_norm_scale_shift: residual must match x in shape, dtype and device");

  const int64_t seq_len = x.size(1);
  const int64_t hidden = x.size(2);
  TORCH_CHECK(
      hidden <= kSrnssMaxHidden,
      "fused_scale_residual_norm_scale_shift: hidden must be <= ",
      kSrnssMaxHidden,
      ", got ",
      hidden);
  const at::ScalarType param_dtype = scale.scalar_type();
  check_param(scale, x, param_dtype, "scale");
  check_param(shift, x, param_dtype, "shift");
  TORCH_CHECK(
      scale.numel() == 1 || scale.numel() == hidden,
      "fused_scale_residual_norm_scale_shift: scale must have 1 or hidden elements, got ",
      scale.numel());
  TORCH_CHECK(
      shift.numel() == 1 || shift.numel() == hidden,
      "fused_scale_residual_norm_scale_shift: shift must have 1 or hidden elements, got ",
      shift.numel());

  int64_t gate_row_div = 1;
  int64_t gate_row_stride = 0;
  const void* gate_ptr = nullptr;
  if (gate.has_value()) {
    const torch::Tensor& g = gate.value();
    check_param(g, x, param_dtype, "gate");
    TORCH_CHECK(
        (g.dim() == 3 || g.dim() == 4) && g.size(0) == 1 && g.size(-1) == hidden,
        "fused_scale_residual_norm_scale_shift: gate must be [1, 1, hidden] or [1, num_frames, 1, hidden], got ",
        g.sizes());
    if (g.dim() == 3) {
      TORCH_CHECK(g.size(1) == 1, "fused_scale_residual_norm_scale_shift: a 3-D gate must be [1, 1, hidden]");
    } else {
      TORCH_CHECK(
          g.size(2) == 1 && seq_len % g.size(1) == 0,
          "fused_scale_residual_norm_scale_shift: a 4-D gate must be [1, num_frames, 1, hidden] with num_frames "
          "dividing seq_len");
      gate_row_div = seq_len / g.size(1);
      gate_row_stride = hidden;
    }
    gate_ptr = g.data_ptr();
  }

  const void* weight_ptr = nullptr;
  const void* bias_ptr = nullptr;
  if (weight.has_value()) {
    check_param(weight.value(), x, param_dtype, "weight");
    TORCH_CHECK(
        weight.value().numel() == hidden,
        "fused_scale_residual_norm_scale_shift: weight must have hidden elements, got ",
        weight.value().numel());
    weight_ptr = weight.value().data_ptr();
  }
  if (bias.has_value()) {
    check_param(bias.value(), x, param_dtype, "bias");
    TORCH_CHECK(
        bias.value().numel() == hidden,
        "fused_scale_residual_norm_scale_shift: bias must have hidden elements, got ",
        bias.value().numel());
    bias_ptr = bias.value().data_ptr();
  }

  torch::Tensor out = at::empty_like(x);
  torch::Tensor residual_output = at::empty_like(x);

  const int32_t scale_is_vec = scale.numel() == hidden ? 1 : 0;
  const int32_t shift_is_vec = shift.numel() == hidden ? 1 : 0;

  const int64_t act_size = x.element_size();
  const int64_t param_size = scale.element_size();
  std::vector<VectorOperand> vector_operands = {
      {out.data_ptr(), act_size},
      {residual_output.data_ptr(), act_size},
      {residual.data_ptr(), act_size},
      {x.data_ptr(), act_size},
  };
  if (gate_ptr != nullptr) vector_operands.push_back({gate_ptr, param_size});
  if (weight_ptr != nullptr) vector_operands.push_back({weight_ptr, param_size});
  if (bias_ptr != nullptr) vector_operands.push_back({bias_ptr, param_size});
  if (scale_is_vec != 0) vector_operands.push_back({scale.data_ptr(), param_size});
  if (shift_is_vec != 0) vector_operands.push_back({shift.data_ptr(), param_size});
  const LaunchArgs args = {
      residual_output.data_ptr(),
      out.data_ptr(),
      residual.data_ptr(),
      x.data_ptr(),
      gate_ptr,
      weight_ptr,
      bias_ptr,
      scale.data_ptr(),
      shift.data_ptr(),
      seq_len,
      hidden,
      gate_row_div,
      gate_row_stride,
      scale_is_vec,
      shift_is_vec,
      select_vec_size(hidden, vector_operands),
      static_cast<float>(eps)};

  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();
  SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, x.scalar_type(), "fused_scale_residual_norm_scale_shift", [&]() {
        SYCL_DISPATCH_WEIGHT_TYPES(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            param_dtype,
            "fused_scale_residual_norm_scale_shift",
            [&]() { launch<scalar_t, srnss_sycl_t<weight_t>>(queue, args); });
      });

  return {out, residual_output};
}

}  // namespace at::native::xpu
