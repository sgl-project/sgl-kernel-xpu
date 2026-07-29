#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "Compress.h"
#include "Utils.h"

namespace at::native::xpu {

constexpr int64_t kTileDim = 64;  // kTileElements (4) * kWarpThreads (16)
constexpr int64_t kTileElements = 4;

namespace FlashCompress4Impl {

template <typename buffer_t, typename input_t, typename out_t>
struct FlashCompress4DecodeKernel {
  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t gid = static_cast<uint32_t>(item.get_group(0));
    const uint32_t lid = static_cast<uint32_t>(item.get_local_id(0));

    // Compute global warp id: group_id*4 + local_warp_id (since local_size=64 = 4 warps)
    const uint32_t global_wid = gid * 4U + (lid / 16U);
    const uint32_t bid = global_wid / num_split_;
    const uint32_t split_id = global_wid % num_split_;
    const int64_t split_offset = static_cast<int64_t>(split_id) * kTileDim;
    const uint32_t tid_in_warp = lid % 16U;

    if (bid >= batch_size_) {
      return;
    }

    const DecodePlan plan = plan_d_[bid];
    if (plan.seq_len == 0 || plan.write_loc < 0) {
      return;
    }

    // Write current token to buffer
    const int64_t write_loc = static_cast<int64_t>(plan.write_loc);
    buffer_t* kv_dst = kv_buffer_ + write_loc * elem_size_;
    const input_t* kv_src = kv_input_ + static_cast<int64_t>(bid) * elem_size_;

    // Each thread processes 4 consecutive head_dim positions
    for (int32_t i = 0; i < kTileElements; ++i) {
      const int64_t h = split_offset + i * 16 + tid_in_warp;
      if (h < head_dim_) {
        // Write all 4 * head_dim elements (kv + score pairs for 4 positions)
        for (int64_t j = 0; j < 4; ++j) {
          kv_dst[j * head_dim_ + h] = static_cast<buffer_t>(kv_src[j * head_dim_ + h]);
        }
      }
    }

    item.barrier(sycl::access::fence_space::global_and_local);

    // Compress only when a 4-token chunk is completed (every 4 tokens).
    if (plan.seq_len % 4 != 0) {
      return;
    }

    // Buffer row layout per token: | kv_overlap(head_dim) | kv(head_dim) | score_overlap(head_dim) | score(head_dim) |
    // row stride = elem_size_ = 4 * head_dim_
    // Overlap positions (t=0..3): from kv_buf_0, kv_overlap at [row+0], score_overlap at [row+2*head_dim]
    // Fresh positions  (t=4..7): from kv_buf_1, kv at [row+head_dim], score at [row+3*head_dim]
    const bool need_overlap = plan.seq_len > 4;
    const buffer_t* kv_buf_0 = kv_buffer_ + static_cast<int64_t>(plan.read_page_0) * page_elem_size_;
    const buffer_t* kv_buf_1 = kv_buffer_ + static_cast<int64_t>(plan.read_page_1) * page_elem_size_;
    out_t* out_row = kv_output_ + static_cast<int64_t>(bid) * head_dim_;

    for (int32_t i = 0; i < kTileElements; ++i) {
      const int64_t h = split_offset + static_cast<int64_t>(i) * 16 + tid_in_warp;
      if (h >= head_dim_) {
        continue;
      }

      // First pass: max score
      float max_score = -std::numeric_limits<float>::infinity();
      for (int64_t t = 0; t < 8; ++t) {
        const int64_t row_off = (t % 4) * elem_size_;
        float score_val;
        if (t < 4) {
          score_val = need_overlap ? static_cast<float>(kv_buf_0[row_off + 2 * head_dim_ + h])
                                   : -std::numeric_limits<float>::infinity();
        } else {
          score_val = static_cast<float>(kv_buf_1[row_off + 3 * head_dim_ + h]);
        }
        max_score = sycl::fmax(max_score, score_val + static_cast<float>(ape_[t * head_dim_ + h]));
      }

      // Second pass: weighted sum
      float exp_sum = 0.0f;
      float weighted_sum = 0.0f;
      for (int64_t t = 0; t < 8; ++t) {
        const int64_t row_off = (t % 4) * elem_size_;
        float kv_val, score_val;
        if (t < 4) {
          kv_val = need_overlap ? static_cast<float>(kv_buf_0[row_off + h]) : 0.0f;
          score_val = need_overlap ? static_cast<float>(kv_buf_0[row_off + 2 * head_dim_ + h])
                                   : -std::numeric_limits<float>::infinity();
        } else {
          kv_val = static_cast<float>(kv_buf_1[row_off + head_dim_ + h]);
          score_val = static_cast<float>(kv_buf_1[row_off + 3 * head_dim_ + h]);
        }
        const float w = sycl::exp(score_val + static_cast<float>(ape_[t * head_dim_ + h]) - max_score);
        exp_sum += w;
        weighted_sum += kv_val * w;
      }

      out_row[h] = static_cast<out_t>(weighted_sum / exp_sum);
    }
  }

  buffer_t* kv_buffer_;
  const input_t* kv_input_;
  out_t* kv_output_;
  const input_t* ape_;
  const DecodePlan* plan_d_;
  uint32_t batch_size_;
  int64_t head_dim_;
  int64_t elem_size_;       // 4 * head_dim
  int64_t page_elem_size_;  // 4 * 4 * head_dim = 16 * head_dim
  uint32_t num_split_;
};

}  // namespace FlashCompress4Impl

void flash_compress4_decode(
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
  TORCH_CHECK(elem_size == 4 * head_dim, "kv_input last dim must be 4 * head_dim");
  TORCH_CHECK(
      kv_buffer.size(1) == 4 && kv_buffer.size(2) == elem_size, "kv_buffer shape must be [num_pages, 4, 4*head_dim]");
  TORCH_CHECK(kv_output.size(0) == batch_size, "kv_output batch must match kv_input batch");
  TORCH_CHECK(ape.size(0) == 8 && ape.size(1) == head_dim, "ape shape must be [8, head_dim]");
  TORCH_CHECK(
      plan_d.size(0) == batch_size && plan_d.size(1) == static_cast<int64_t>(sizeof(DecodePlan)),
      "plan_d shape must be [B, 16]");

  if (batch_size == 0) {
    return;
  }

  const int64_t page_elem_size = 4 * elem_size;  // 4 positions * 4 * head_dim
  const uint32_t num_split = static_cast<uint32_t>((head_dim + kTileDim - 1) / kTileDim);
  auto queue = c10::xpu::getCurrentXPUStream().queue();

  SYCL_DISPATCH_FLOATING_TYPES(at::kHalf, at::kBFloat16, kv_input.scalar_type(), "FlashCompress4Decode", [&]() {
    using input_t = scalar_t;
    using output_t = scalar_t;
    SYCL_DISPATCH_WEIGHT_TYPES(at::kHalf, at::kBFloat16, kv_buffer.scalar_type(), "FlashCompress4Decode", [&]() {
      queue.submit([&](sycl::handler& cgh) {
        FlashCompress4Impl::FlashCompress4DecodeKernel<weight_t, input_t, output_t> kernel{
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
        constexpr uint32_t kLocalSize = 64;  // 4 sub_groups, each handling 1 warp's work
        const uint32_t global_size = static_cast<uint32_t>(batch_size) * num_split * 16;
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kLocalSize)), kernel);
      });
    });
  });
}

}  // namespace at::native::xpu
