#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

using bf16_t = sycl::ext::oneapi::bfloat16;

template <typename index_t, int N>
struct IndexedScaleShiftBf16Kernel {
  using vec_t = aligned_vector_loop<bf16_t, N>;

  bf16_t* x;
  const bf16_t* shift;
  const bf16_t* scale;
  const index_t* indices;
  int64_t rows;
  int64_t hidden_size;
  int64_t stride_x_row;
  int64_t stride_shift_row;
  int64_t stride_scale_row;
  int64_t stride_indices;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t row = item.get_group(0);
    const int64_t offset = item.get_local_linear_id();
    const int64_t step = item.get_local_range(0);
    const int64_t index = static_cast<int64_t>(indices[row * stride_indices]);
    const int64_t bound = hidden_size / N;
    const int64_t x_row_base = row * stride_x_row / N;
    const int64_t shift_row_base = index * stride_shift_row / N;
    const int64_t scale_row_base = index * stride_scale_row / N;

    auto* x_vec = reinterpret_cast<vec_t*>(x);
    const auto* shift_vec = reinterpret_cast<const vec_t*>(shift);
    const auto* scale_vec = reinterpret_cast<const vec_t*>(scale);

    for (int64_t i = offset; i < bound; i += step) {
      vec_t x_val = x_vec[x_row_base + i];
      const vec_t shift_val = shift_vec[shift_row_base + i];
      const vec_t scale_val = scale_vec[scale_row_base + i];
#pragma unroll
      for (int v = 0; v < N; ++v) {
        const float x_value = static_cast<float>(x_val[v]);
        const float shift_value = static_cast<float>(shift_val[v]);
        const float scale_value = static_cast<float>(scale_val[v]);
        const float one_plus_scale = static_cast<float>(static_cast<bf16_t>(1.0f + scale_value));
        const float scaled = static_cast<float>(static_cast<bf16_t>(x_value * one_plus_scale));
        x_val[v] = static_cast<bf16_t>(scaled + shift_value);
      }
      x_vec[x_row_base + i] = x_val;
    }
  }
};

template <typename index_t, int N>
struct IndexedGateBf16Kernel {
  using vec_t = aligned_vector_loop<bf16_t, N>;

  bf16_t* x;
  const bf16_t* gate;
  const bf16_t* other;
  const index_t* indices;
  int64_t rows;
  int64_t hidden_size;
  int64_t stride_x_row;
  int64_t stride_gate_row;
  int64_t stride_other_row;
  int64_t stride_indices;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t row = item.get_group(0);
    const int64_t offset = item.get_local_linear_id();
    const int64_t step = item.get_local_range(0);
    const int64_t index = static_cast<int64_t>(indices[row * stride_indices]);
    const int64_t bound = hidden_size / N;
    const int64_t x_row_base = row * stride_x_row / N;
    const int64_t gate_row_base = index * stride_gate_row / N;
    const int64_t other_row_base = row * stride_other_row / N;

    auto* x_vec = reinterpret_cast<vec_t*>(x);
    const auto* gate_vec = reinterpret_cast<const vec_t*>(gate);
    const auto* other_vec = reinterpret_cast<const vec_t*>(other);

    for (int64_t i = offset; i < bound; i += step) {
      vec_t x_val = x_vec[x_row_base + i];
      const vec_t gate_val = gate_vec[gate_row_base + i];
      const vec_t other_val = other_vec[other_row_base + i];
#pragma unroll
      for (int v = 0; v < N; ++v) {
        const float x_value = static_cast<float>(x_val[v]);
        const float gate_value = static_cast<float>(gate_val[v]);
        const float other_value = static_cast<float>(other_val[v]);
        const float gated = static_cast<float>(static_cast<bf16_t>(gate_value * other_value));
        x_val[v] = static_cast<bf16_t>(x_value + gated);
      }
      x_vec[x_row_base + i] = x_val;
    }
  }
};

static int64_t indexed_modulation_workgroup_size(int64_t hidden_size) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const int64_t max_wg = dpcppMaxWorkGroupSize(dev_id);
  const int64_t wg_cap = std::min(hidden_size, max_wg);
  int64_t wg_size = 1;
  while ((wg_size << 1) <= wg_cap) {
    wg_size <<= 1;
  }
  return wg_size;
}

// Largest N (lanes of bf16_t packed into one aligned_vector_loop load/store) such that
// hidden_size and every indexed row stride divide evenly by N; falls back to 1 (scalar).
// Memory-bandwidth-bound, so always prefer the widest N regardless of wg_size/occupancy.
static int select_indexed_modulation_vec_size(int64_t hidden_size, std::initializer_list<int64_t> row_strides) {
  int vec_size = 16 / sizeof(bf16_t);
  while (vec_size > 1) {
    bool ok = hidden_size % vec_size == 0;
    for (int64_t stride : row_strides) {
      ok = ok && (stride % vec_size == 0);
    }
    if (ok) break;
    vec_size >>= 1;
  }
  return vec_size;
}

template <typename Fn>
static void dispatch_index_type(const at::Tensor& indices, const char* kernel_name, Fn&& fn) {
  if (indices.scalar_type() == at::ScalarType::Long) {
    fn(int64_t{});
  } else if (indices.scalar_type() == at::ScalarType::Int) {
    fn(int32_t{});
  } else {
    TORCH_CHECK(false, kernel_name, ": indices must be int32 or int64, got ", indices.scalar_type());
  }
}

static void check_indexed_modulation_inputs(
    const char* kernel_name, const torch::Tensor& x, const torch::Tensor& indexed, const torch::Tensor& indices) {
  CHECK_DEVICE(x);
  CHECK_DEVICE(indexed);
  CHECK_DEVICE(indices);
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, kernel_name, ": x must be bf16");
  TORCH_CHECK(indexed.scalar_type() == at::ScalarType::BFloat16, kernel_name, ": indexed tensor must be bf16");
  TORCH_CHECK(x.dim() == 2, kernel_name, ": x must be 2D [rows, hidden]");
  TORCH_CHECK(indexed.dim() == 2, kernel_name, ": indexed tensor must be 2D [num_indices, hidden]");
  TORCH_CHECK(indices.dim() == 1, kernel_name, ": indices must be 1D [rows]");
  TORCH_CHECK(indices.size(0) == x.size(0), kernel_name, ": indices rows must match x rows");
  TORCH_CHECK(indexed.size(1) == x.size(1), kernel_name, ": hidden sizes must match");
  TORCH_CHECK(x.stride(1) == 1, kernel_name, ": x must be contiguous in the last dimension");
  TORCH_CHECK(indexed.stride(1) == 1, kernel_name, ": indexed tensor must be contiguous in the last dimension");
  TORCH_CHECK(indices.stride(0) == 1, kernel_name, ": indices must be contiguous");
}

SGL_KERNEL_EXPORT void
indexed_scale_shift_bf16_(torch::Tensor& x, torch::Tensor& shift, torch::Tensor& scale, torch::Tensor& indices) {
  check_indexed_modulation_inputs("indexed_scale_shift_bf16_", x, shift, indices);
  check_indexed_modulation_inputs("indexed_scale_shift_bf16_", x, scale, indices);
  TORCH_CHECK(shift.size(0) == scale.size(0), "indexed_scale_shift_bf16_: shift and scale rows must match");

  const int64_t rows = x.size(0);
  if (rows == 0) return;

  auto queue = at::xpu::getCurrentXPUStream().queue();
  const int64_t hidden_size = x.size(1);
  const int64_t wg_size = indexed_modulation_workgroup_size(hidden_size);
  const int vec_size = select_indexed_modulation_vec_size(hidden_size, {x.stride(0), shift.stride(0), scale.stride(0)});

  dispatch_index_type(indices, "indexed_scale_shift_bf16_", [&](auto index_tag) {
    using index_t = decltype(index_tag);
#define VEC_LAUNCH(N)                                                    \
  case N: {                                                              \
    IndexedScaleShiftBf16Kernel<index_t, N> kernel{                      \
        reinterpret_cast<bf16_t*>(x.data_ptr<at::BFloat16>()),           \
        reinterpret_cast<const bf16_t*>(shift.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const bf16_t*>(scale.data_ptr<at::BFloat16>()), \
        indices.data_ptr<index_t>(),                                     \
        rows,                                                            \
        hidden_size,                                                     \
        x.stride(0),                                                     \
        shift.stride(0),                                                 \
        scale.stride(0),                                                 \
        indices.stride(0)};                                              \
    sycl_kernel_submit(rows* wg_size, wg_size, queue, kernel);           \
    break;                                                               \
  }
    switch (vec_size) {
      VEC_LAUNCH(1);
      VEC_LAUNCH(2);
      VEC_LAUNCH(4);
      VEC_LAUNCH(8);
      default:
        TORCH_CHECK(false, "indexed_scale_shift_bf16_: unsupported vector size: ", vec_size);
    }
#undef VEC_LAUNCH
  });
}

SGL_KERNEL_EXPORT void
indexed_gate_bf16_(torch::Tensor& x, torch::Tensor& gate, torch::Tensor& other, torch::Tensor& indices) {
  check_indexed_modulation_inputs("indexed_gate_bf16_", x, gate, indices);
  CHECK_DEVICE(other);
  TORCH_CHECK(other.scalar_type() == at::ScalarType::BFloat16, "indexed_gate_bf16_: other must be bf16");
  TORCH_CHECK(other.dim() == 2, "indexed_gate_bf16_: other must be 2D [rows, hidden]");
  TORCH_CHECK(other.size(0) == x.size(0), "indexed_gate_bf16_: other rows must match x rows");
  TORCH_CHECK(other.size(1) == x.size(1), "indexed_gate_bf16_: hidden sizes must match");
  TORCH_CHECK(other.stride(1) == 1, "indexed_gate_bf16_: other must be contiguous in the last dimension");

  const int64_t rows = x.size(0);
  if (rows == 0) return;

  auto queue = at::xpu::getCurrentXPUStream().queue();
  const int64_t hidden_size = x.size(1);
  const int64_t wg_size = indexed_modulation_workgroup_size(hidden_size);
  const int vec_size = select_indexed_modulation_vec_size(hidden_size, {x.stride(0), gate.stride(0), other.stride(0)});

  dispatch_index_type(indices, "indexed_gate_bf16_", [&](auto index_tag) {
    using index_t = decltype(index_tag);
#define VEC_LAUNCH(N)                                                    \
  case N: {                                                              \
    IndexedGateBf16Kernel<index_t, N> kernel{                            \
        reinterpret_cast<bf16_t*>(x.data_ptr<at::BFloat16>()),           \
        reinterpret_cast<const bf16_t*>(gate.data_ptr<at::BFloat16>()),  \
        reinterpret_cast<const bf16_t*>(other.data_ptr<at::BFloat16>()), \
        indices.data_ptr<index_t>(),                                     \
        rows,                                                            \
        hidden_size,                                                     \
        x.stride(0),                                                     \
        gate.stride(0),                                                  \
        other.stride(0),                                                 \
        indices.stride(0)};                                              \
    sycl_kernel_submit(rows* wg_size, wg_size, queue, kernel);           \
    break;                                                               \
  }
    switch (vec_size) {
      VEC_LAUNCH(1);
      VEC_LAUNCH(2);
      VEC_LAUNCH(4);
      VEC_LAUNCH(8);
      default:
        TORCH_CHECK(false, "indexed_gate_bf16_: unsupported vector size: ", vec_size);
    }
#undef VEC_LAUNCH
  });
}
