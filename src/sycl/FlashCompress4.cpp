#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "Compress.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace at::native::xpu {

constexpr int64_t kTileDim = 64;  // kTileElements (4) * kWarpThreads (16)
constexpr int64_t kTileElements = 4;
constexpr uint32_t kWarpThreads = 16;

namespace FlashCompress4Impl {

template <typename buffer_t, typename input_t, typename out_t>
struct FlashCompress4DecodeKernel {
  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t global_tid = static_cast<uint32_t>(item.get_global_id(0));
    const uint32_t local_tid = static_cast<uint32_t>(item.get_local_id(0));

    const uint32_t global_wid = global_tid / kWarpThreads;  // warp id
    const uint32_t global_bid = global_wid / num_split_;    // batch id
    const uint32_t global_sid = global_wid % num_split_;    // split id
    const int64_t split_offset = static_cast<int64_t>(global_sid) * kTileDim;
    const uint32_t lane_id = local_tid % kWarpThreads;

    if (global_bid >= batch_size_) {
      return;
    }

    const DecodePlan plan = plan_d_[global_bid];
    if (plan.seq_len == 0 || plan.write_loc < 0) {
      return;
    }

    // Decode path: write the current token to page buffer first.
    const int64_t write_loc = static_cast<int64_t>(plan.write_loc);
    buffer_t* kv_dst = kv_buffer_ + write_loc * elem_size_;
    const input_t* kv_src = kv_input_ + static_cast<int64_t>(global_bid) * elem_size_;

    // One warp handles one split; each lane processes strided head_dim elements.
    for (int32_t i = 0; i < kTileElements; ++i) {
      const int64_t h = split_offset + static_cast<int64_t>(i) * kWarpThreads + lane_id;
      for (int64_t j = 0; j < 4; ++j) {
        kv_dst[j * head_dim_ + h] = static_cast<buffer_t>(kv_src[j * head_dim_ + h]);
      }
    }

    // Compress only when we close a 4-token chunk.
    if (plan.seq_len % 4 != 0) {
      return;
    }

    // Buffer row layout:
    // [kv_overlap | kv | score_overlap | score], row stride = elem_size_.
    // t in [0, 3] reads overlap terms from kv_buf_0; t in [4, 7] reads fresh terms from kv_buf_1.
    const bool need_overlap = plan.seq_len > 4;
    const buffer_t* kv_buf_0 = kv_buffer_ + static_cast<int64_t>(plan.read_page_0) * page_elem_size_;
    const buffer_t* kv_buf_1 = kv_buffer_ + static_cast<int64_t>(plan.read_page_1) * page_elem_size_;
    out_t* kv_out = kv_output_ + static_cast<int64_t>(global_bid) * head_dim_;

    for (int32_t i = 0; i < kTileElements; ++i) {
      const int64_t h = split_offset + static_cast<int64_t>(i) * kWarpThreads + lane_id;

      // Load all 8 tokens into registers (overlap [0,3] from kv_buf_0, fresh [4,7] from kv_buf_1).
      float kv_reg[8];
      float score_reg[8];
      for (int32_t t = 0; t < 4; ++t) {
        const int64_t row_off = static_cast<int64_t>(t) * elem_size_;
        kv_reg[t] = need_overlap ? static_cast<float>(kv_buf_0[row_off + h]) : 0.0f;
        score_reg[t] = (need_overlap ? static_cast<float>(kv_buf_0[row_off + 2 * head_dim_ + h])
                                     : -std::numeric_limits<float>::infinity()) +
                       static_cast<float>(ape_[static_cast<int64_t>(t) * head_dim_ + h]);
      }
      for (int32_t t = 0; t < 4; ++t) {
        const int64_t row_off = static_cast<int64_t>(t) * elem_size_;
        kv_reg[t + 4] = static_cast<float>(kv_buf_1[row_off + head_dim_ + h]);
        score_reg[t + 4] = static_cast<float>(kv_buf_1[row_off + 3 * head_dim_ + h]) +
                           static_cast<float>(ape_[static_cast<int64_t>(t + 4) * head_dim_ + h]);
      }

      // Pass 1: max score.
      float max_score = score_reg[0];
      for (int32_t t = 1; t < 8; ++t) {
        max_score = sycl::fmax(max_score, score_reg[t]);
      }

      // Pass 2: weighted kv reduction.
      float exp_sum = 0.0f;
      float weighted_sum = 0.0f;
      for (int32_t t = 0; t < 8; ++t) {
        const float w = sycl::exp(score_reg[t] - max_score);
        exp_sum += w;
        weighted_sum += kv_reg[t] * w;
      }

      kv_out[h] = static_cast<out_t>(weighted_sum / exp_sum);
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

SGL_KERNEL_EXPORT void flash_compress4_decode(
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
  TORCH_CHECK(head_dim % kTileDim == 0, "flash_compress4_decode requires head_dim divisible by ", kTileDim);
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

  const int64_t page_elem_size = 4 * elem_size;
  const uint32_t num_split = static_cast<uint32_t>(head_dim / kTileDim);
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
        constexpr uint32_t kWarpsPerGroup = 4;
        constexpr uint32_t kLocalSize = kWarpsPerGroup * kWarpThreads;
        const uint32_t num_warps = static_cast<uint32_t>(batch_size) * num_split;
        const uint32_t num_groups = (num_warps + kWarpsPerGroup - 1) / kWarpsPerGroup;
        const uint32_t global_size = num_groups * kLocalSize;
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kLocalSize)), kernel);
      });
    });
  });
}

}  // namespace at::native::xpu
