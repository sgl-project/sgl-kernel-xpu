#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

using bf16_t = sycl::ext::oneapi::bfloat16;

template <typename index_t>
struct IndexedScaleShiftBf16Kernel {
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
    const int64_t column = item.get_local_linear_id();
    const int64_t step = item.get_local_range(0);
    const int64_t index = static_cast<int64_t>(indices[row * stride_indices]);

    for (int64_t offset = column; offset < hidden_size; offset += step) {
      const float x_value = static_cast<float>(x[row * stride_x_row + offset]);
      const float shift_value = static_cast<float>(shift[index * stride_shift_row + offset]);
      const float scale_value = static_cast<float>(scale[index * stride_scale_row + offset]);
      const float one_plus_scale = static_cast<float>(static_cast<bf16_t>(1.0f + scale_value));
      const float scaled = static_cast<float>(static_cast<bf16_t>(x_value * one_plus_scale));
      x[row * stride_x_row + offset] = static_cast<bf16_t>(scaled + shift_value);
    }
  }
};

template <typename index_t>
struct IndexedGateBf16Kernel {
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
    const int64_t column = item.get_local_linear_id();
    const int64_t step = item.get_local_range(0);
    const int64_t index = static_cast<int64_t>(indices[row * stride_indices]);

    for (int64_t offset = column; offset < hidden_size; offset += step) {
      const float x_value = static_cast<float>(x[row * stride_x_row + offset]);
      const float gate_value = static_cast<float>(gate[index * stride_gate_row + offset]);
      const float other_value = static_cast<float>(other[row * stride_other_row + offset]);
      const float gated = static_cast<float>(static_cast<bf16_t>(gate_value * other_value));
      x[row * stride_x_row + offset] = static_cast<bf16_t>(x_value + gated);
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

template <typename KernelT>
static void launch_indexed_modulation_kernel(sycl::queue& queue, int64_t rows, int64_t wg_size, KernelT kernel) {
  sycl_kernel_submit(rows * wg_size, wg_size, queue, kernel);
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
    const char* kernel_name,
    const torch::Tensor& x,
    const torch::Tensor& indexed,
    const torch::Tensor& indices) {
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

SGL_KERNEL_EXPORT void indexed_scale_shift_bf16_(
    torch::Tensor& x, torch::Tensor& shift, torch::Tensor& scale, torch::Tensor& indices) {
  check_indexed_modulation_inputs("indexed_scale_shift_bf16_", x, shift, indices);
  check_indexed_modulation_inputs("indexed_scale_shift_bf16_", x, scale, indices);
  TORCH_CHECK(shift.size(0) == scale.size(0), "indexed_scale_shift_bf16_: shift and scale rows must match");

  const int64_t rows = x.size(0);
  if (rows == 0) return;

  auto queue = at::xpu::getCurrentXPUStream().queue();
  const int64_t hidden_size = x.size(1);
  const int64_t wg_size = indexed_modulation_workgroup_size(hidden_size);

  dispatch_index_type(indices, "indexed_scale_shift_bf16_", [&](auto index_tag) {
    using index_t = decltype(index_tag);
    IndexedScaleShiftBf16Kernel<index_t> kernel{
        reinterpret_cast<bf16_t*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16_t*>(shift.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16_t*>(scale.data_ptr<at::BFloat16>()),
        indices.data_ptr<index_t>(),
        rows,
        hidden_size,
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0)};
    launch_indexed_modulation_kernel(queue, rows, wg_size, kernel);
  });
}

SGL_KERNEL_EXPORT void indexed_gate_bf16_(
    torch::Tensor& x, torch::Tensor& gate, torch::Tensor& other, torch::Tensor& indices) {
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

  dispatch_index_type(indices, "indexed_gate_bf16_", [&](auto index_tag) {
    using index_t = decltype(index_tag);
    IndexedGateBf16Kernel<index_t> kernel{
        reinterpret_cast<bf16_t*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16_t*>(gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16_t*>(other.data_ptr<at::BFloat16>()),
        indices.data_ptr<index_t>(),
        rows,
        hidden_size,
        x.stride(0),
        gate.stride(0),
        other.stride(0),
        indices.stride(0)};
    launch_indexed_modulation_kernel(queue, rows, wg_size, kernel);
  });
}