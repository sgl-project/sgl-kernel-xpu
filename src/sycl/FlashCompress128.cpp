#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "Compress.h"
#include "Utils.h"

namespace at::native::xpu {

constexpr int64_t kTileDim = 64;

namespace FlashCompress128Impl {

template <typename buffer_t, typename input_t, typename out_t>
struct FlashCompress128DecodeKernel {
  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t gid = static_cast<uint32_t>(item.get_group(0));
    const uint32_t lid = static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t bid = gid / num_split_;
    const int64_t split_offset = static_cast<int64_t>(gid % num_split_) * kTileDim;

    if (bid >= batch_size_) {
      return;
    }

    const DecodePlan plan = plan_d_[bid];
    if (plan.seq_len == 0 || plan.write_loc < 0) {
      return;
    }

    const int64_t write_loc = static_cast<int64_t>(plan.write_loc);
    buffer_t* kv_dst = kv_buffer_ + write_loc * elem_size_;
    const input_t* kv_src = kv_input_ + static_cast<int64_t>(bid) * elem_size_;

    const int64_t h = split_offset + static_cast<int64_t>(lid);
    if (h < head_dim_) {
      kv_dst[h] = static_cast<buffer_t>(kv_src[h]);
      kv_dst[head_dim_ + h] = static_cast<buffer_t>(kv_src[head_dim_ + h]);
    }

    // Compress only when a 128-token chunk is completed.
    if ((plan.write_loc % 128) != 127) {
      return;
    }

    item.barrier(sycl::access::fence_space::global_and_local);

    // Last split may be partial when head_dim is not divisible by kTileDim.
    if (h >= head_dim_) {
      return;
    }

    const int64_t page = static_cast<int64_t>(plan.read_page_1);
    const buffer_t* kv_page = kv_buffer_ + page * page_elem_size_;
    out_t* out_row = kv_output_ + static_cast<int64_t>(bid) * head_dim_;

    // Numerically stable softmax: first pass computes max(logits).
    float max_score = -std::numeric_limits<float>::infinity();
    for (int64_t t = 0; t < 128; ++t) {
      const int64_t row_off = t * elem_size_;
      const float score =
          static_cast<float>(kv_page[row_off + head_dim_ + h]) + static_cast<float>(ape_[t * head_dim_ + h]);
      max_score = sycl::fmax(max_score, score);
    }

    // Second pass computes exp-sum and weighted value sum in fp32.
    float exp_sum = 0.0f;
    float weighted_sum = 0.0f;
    for (int64_t t = 0; t < 128; ++t) {
      const int64_t row_off = t * elem_size_;
      const float score =
          static_cast<float>(kv_page[row_off + head_dim_ + h]) + static_cast<float>(ape_[t * head_dim_ + h]);
      const float w = sycl::exp(score - max_score);
      exp_sum += w;
      weighted_sum += static_cast<float>(kv_page[row_off + h]) * w;
    }

    out_row[h] = static_cast<out_t>(weighted_sum / exp_sum);
  }

  buffer_t* kv_buffer_;
  const input_t* kv_input_;
  out_t* kv_output_;
  const input_t* ape_;
  const DecodePlan* plan_d_;
  uint32_t batch_size_;
  int64_t head_dim_;
  int64_t elem_size_;
  int64_t page_elem_size_;
  uint32_t num_split_;
};

}  // namespace FlashCompress128Impl

void flash_compress128_decode(
    torch::Tensor kv_buffer, torch::Tensor kv_input, torch::Tensor kv_output, torch::Tensor ape, torch::Tensor plan_d) {
  TORCH_CHECK(
      kv_buffer.is_xpu() && kv_buffer.dim() == 3 && kv_buffer.is_contiguous(),
      "kv_buffer must be a contiguous 3D XPU tensor");
  TORCH_CHECK(
      kv_input.is_xpu() && kv_input.dim() == 2 && kv_input.is_contiguous(),
      "kv_input must be a contiguous 2D XPU tensor");
  TORCH_CHECK(
      kv_output.is_xpu() && kv_output.dim() == 2 && kv_output.is_contiguous(),
      "kv_output must be a contiguous 2D XPU tensor");
  TORCH_CHECK(ape.is_xpu() && ape.dim() == 2 && ape.is_contiguous(), "ape must be a contiguous 2D XPU tensor");
  TORCH_CHECK(
      plan_d.is_xpu() && plan_d.dim() == 2 && plan_d.dtype() == torch::kUInt8 && plan_d.is_contiguous(),
      "plan_d must be a contiguous 2D XPU uint8 tensor");
  TORCH_CHECK(kv_input.dtype() == kv_output.dtype(), "kv_input and kv_output must have the same dtype");
  TORCH_CHECK(kv_input.dtype() == ape.dtype(), "kv_input and ape must have same dtype");

  const int64_t batch_size = kv_input.size(0);
  const int64_t elem_size = kv_input.size(1);
  const int64_t head_dim = kv_output.size(1);
  TORCH_CHECK(elem_size == 2 * head_dim, "kv_input last dim must be 2 * head_dim");
  TORCH_CHECK(
      kv_buffer.size(1) == 128 && kv_buffer.size(2) == elem_size,
      "kv_buffer shape must be [num_pages, 128, 2*head_dim]");
  TORCH_CHECK(kv_output.size(0) == batch_size, "kv_output batch must match kv_input batch");
  TORCH_CHECK(ape.size(0) == 128 && ape.size(1) == head_dim, "ape shape must be [128, head_dim]");
  TORCH_CHECK(
      plan_d.size(0) == batch_size && plan_d.size(1) == static_cast<int64_t>(sizeof(DecodePlan)),
      "plan_d shape must be [B, 16]");

  if (batch_size == 0) {
    return;
  }

  const int64_t page_elem_size = 128 * elem_size;
  const uint32_t num_split = static_cast<uint32_t>((head_dim + kTileDim - 1) / kTileDim);
  auto queue = c10::xpu::getCurrentXPUStream().queue();

  SYCL_DISPATCH_FLOATING_TYPES(at::kHalf, at::kBFloat16, kv_input.scalar_type(), "FlashCompress128Decode", [&]() {
    using input_t = scalar_t;
    using output_t = scalar_t;
    SYCL_DISPATCH_WEIGHT_TYPES(at::kHalf, at::kBFloat16, kv_buffer.scalar_type(), "FlashCompress128Decode", [&]() {
      queue.submit([&](sycl::handler& cgh) {
        FlashCompress128Impl::FlashCompress128DecodeKernel<weight_t, input_t, output_t> kernel{
            kv_buffer.data_ptr<weight_t>(),
            kv_input.data_ptr<input_t>(),
            kv_output.data_ptr<output_t>(),
            ape.data_ptr<input_t>(),
            reinterpret_cast<const DecodePlan*>(plan_d.data_ptr<uint8_t>()),
            static_cast<uint32_t>(batch_size),
            head_dim,
            elem_size,
            page_elem_size,
            num_split};
        constexpr uint32_t kLocalSize = 64;
        const uint32_t global_size = static_cast<uint32_t>(batch_size) * num_split * kLocalSize;
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kLocalSize)), kernel);
      });
    });
  });
}

}  // namespace at::native::xpu
