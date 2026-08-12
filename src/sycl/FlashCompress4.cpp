#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "Compress.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace at::native::xpu {

constexpr int64_t kTileElements = 4;
constexpr uint32_t kSubGroupSize = 16;
constexpr int64_t kTileDim = kTileElements * static_cast<int64_t>(kSubGroupSize);  // 64
constexpr uint32_t kSubGroupsPerBlock = 4;
constexpr uint32_t kBlockSize = kSubGroupsPerBlock * kSubGroupSize;  // 64

namespace FlashCompress4Impl {

template <typename buffer_t, typename input_t>
inline void c4_write_token_strided(
    buffer_t* kv_dst,
    const input_t* kv_src,
    const int64_t split_offset,
    const uint32_t lane_id,
    const int64_t row_stride) {
  const int64_t lane_base = split_offset + static_cast<int64_t>(lane_id) * kTileElements;

  if constexpr (std::is_same_v<buffer_t, input_t> && at::is_reduced_floating_point_v<input_t>) {
    // Fast path: copy 4 contiguous elements (8 bytes) per row via two 32-bit moves.
    for (int64_t i = 0; i < kTileElements; ++i) {
      const int64_t row_off = lane_base + i * row_stride;
      const uint32_t* src32 = reinterpret_cast<const uint32_t*>(kv_src + row_off);
      uint32_t* dst32 = reinterpret_cast<uint32_t*>(kv_dst + row_off);
      dst32[0] = src32[0];
      dst32[1] = src32[1];
    }
  } else {
    // Mixed dtype fallback: preserve conversion semantics.
    for (int64_t i = 0; i < kTileElements; ++i) {
      const int64_t row_off = lane_base + i * row_stride;
      for (int64_t x = 0; x < kTileElements; ++x) {
        kv_dst[row_off + x] = static_cast<buffer_t>(kv_src[row_off + x]);
      }
    }
  }
}

template <typename buffer_t, typename input_t, typename out_t>
inline void c4_forward(
    const buffer_t* kv_buf_0,
    const buffer_t* kv_buf_1,
    const input_t* kv_src,
    out_t* kv_out,
    const input_t* ape,
    const bool need_overlap,
    const int32_t buffer_len,
    const int64_t split_offset,
    const uint32_t lane_id,
    const int64_t head_dim,
    const int64_t elem_size) {
  for (int32_t i = 0; i < kTileElements; ++i) {
    const int64_t h = split_offset + static_cast<int64_t>(i) * kSubGroupSize + lane_id;

    float kv_reg[8];
    float score_reg[8];

    for (int32_t t = 0; t < 8; ++t) {
      const int64_t row_off = static_cast<int64_t>(t % 4) * elem_size;
      float kv_val;
      float score_val;

      if (t < 4) {
        if (need_overlap && t < buffer_len) {
          kv_val = static_cast<float>(kv_buf_0[row_off + h]);
          score_val = static_cast<float>(kv_buf_0[row_off + 2 * head_dim + h]);
        } else if (need_overlap) {
          const int64_t offset = (static_cast<int64_t>(t) - 7) * elem_size;
          kv_val = static_cast<float>(kv_src[offset + h]);
          score_val = static_cast<float>(kv_src[offset + 2 * head_dim + h]);
        } else {
          kv_val = 0.0f;
          score_val = -std::numeric_limits<float>::infinity();
        }
      } else {
        if (t < buffer_len) {
          kv_val = static_cast<float>(kv_buf_1[row_off + head_dim + h]);
          score_val = static_cast<float>(kv_buf_1[row_off + 3 * head_dim + h]);
        } else {
          const int32_t j = t - 4;
          const int64_t offset = (static_cast<int64_t>(j) - 3) * elem_size;
          kv_val = static_cast<float>(kv_src[offset + head_dim + h]);
          score_val = static_cast<float>(kv_src[offset + 3 * head_dim + h]);
        }
      }

      kv_reg[t] = kv_val;
      score_reg[t] = score_val + static_cast<float>(ape[static_cast<int64_t>(t) * head_dim + h]);
    }

    float max_score = score_reg[0];
    for (int32_t t = 1; t < 8; ++t) {
      max_score = sycl::fmax(max_score, score_reg[t]);
    }

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

template <typename buffer_t, typename input_t, typename out_t>
struct FlashCompress4DecodeKernel {
  [[sycl::reqd_sub_group_size(kSubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t gid = static_cast<uint32_t>(item.get_group(0));
    const uint32_t lid = static_cast<uint32_t>(item.get_local_id(0));

    const uint32_t global_sg_id = gid * kSubGroupsPerBlock + (lid / kSubGroupSize);
    const uint32_t global_bid = global_sg_id / num_split_;  // batch id
    const uint32_t global_sid = global_sg_id % num_split_;  // split id
    const int64_t split_offset = static_cast<int64_t>(global_sid) * kTileDim;
    const uint32_t lane_id = lid % kSubGroupSize;

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

    // One sub-group handles one split; each lane copies a contiguous 4-element chunk per row.
    c4_write_token_strided<buffer_t, input_t>(kv_dst, kv_src, split_offset, lane_id, head_dim_);

    // Compress only when we close a 4-token chunk.
    if (plan.seq_len % 4 != 0) {
      return;
    }

    // Buffer row layout:
    // [kv_overlap | kv | score_overlap | score], row stride = elem_size_.
    // t in [0, 3] reads overlap terms from kv_buf_0; t in [4, 7] reads fresh terms from kv_buf_1.
    const bool need_overlap = plan.seq_len > 4;
    const buffer_t* kv_buf_0 = kv_buffer_;
    if (need_overlap) {
      if (plan.read_page_0 < 0) {
        return;
      }
      kv_buf_0 = kv_buffer_ + static_cast<int64_t>(plan.read_page_0) * page_elem_size_;
    }
    if (plan.read_page_1 < 0) {
      return;
    }
    const buffer_t* kv_buf_1 = kv_buffer_ + static_cast<int64_t>(plan.read_page_1) * page_elem_size_;
    out_t* kv_out = kv_output_ + static_cast<int64_t>(global_bid) * head_dim_;

    c4_forward<buffer_t, input_t, out_t>(
        kv_buf_0, kv_buf_1, kv_src, kv_out, ape_, need_overlap, 8, split_offset, lane_id, head_dim_, elem_size_);
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

template <typename buffer_t, typename input_t, typename out_t>
struct FlashCompress4CompressKernel {
  [[sycl::reqd_sub_group_size(kSubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t gid = static_cast<uint32_t>(item.get_group(0));
    const uint32_t lid = static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t global_sg_id = gid * kSubGroupsPerBlock + (lid / kSubGroupSize);
    const uint32_t pid = global_sg_id / num_split_;
    const uint32_t split_id = global_sg_id % num_split_;
    const int64_t split_offset = static_cast<int64_t>(split_id) * kTileDim;
    const uint32_t lane_id = lid % kSubGroupSize;

    if (pid >= num_compress_) {
      return;
    }

    const CompressPlan plan = plan_c_[pid];
    if (plan.is_invalid()) {
      return;
    }

    const bool need_overlap = plan.seq_len > 4;
    const int32_t buffer_len = static_cast<int32_t>(plan.buffer_len);
    const buffer_t* kv_buf_0 = kv_buffer_;
    const buffer_t* kv_buf_1 = kv_buffer_;
    if (need_overlap && buffer_len > 0) {
      kv_buf_0 = kv_buffer_ + static_cast<int64_t>(plan.read_page_0) * page_elem_size_;
    }
    if (buffer_len > 4) {
      kv_buf_1 = kv_buffer_ + static_cast<int64_t>(plan.read_page_1) * page_elem_size_;
    }
    // kv_src points to position ragged_id in the ragged input
    const input_t* kv_src = kv_input_ + static_cast<int64_t>(plan.ragged_id) * elem_size_;
    out_t* out_row = kv_output_ + static_cast<int64_t>(pid) * head_dim_;

    c4_forward<buffer_t, input_t, out_t>(
        kv_buf_0,
        kv_buf_1,
        kv_src,
        out_row,
        ape_,
        need_overlap,
        buffer_len,
        split_offset,
        lane_id,
        head_dim_,
        elem_size_);
  }

  buffer_t* kv_buffer_;
  const input_t* kv_input_;
  out_t* kv_output_;
  const input_t* ape_;
  const CompressPlan* plan_c_;
  uint32_t num_compress_;
  int64_t head_dim_;
  int64_t elem_size_;       // 4 * head_dim
  int64_t page_elem_size_;  // 16 * head_dim
  uint32_t num_split_;
};

template <typename buffer_t, typename input_t>
struct FlashCompress4WriteKernel {
  [[sycl::reqd_sub_group_size(kSubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t gid = static_cast<uint32_t>(item.get_group(0));
    const uint32_t lid = static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t global_sg_id = gid * kSubGroupsPerBlock + (lid / kSubGroupSize);
    const uint32_t pid = global_sg_id / num_split_;
    const uint32_t split_id = global_sg_id % num_split_;
    // Split along head_dim columns (kTileDim=64) and copy all 4 rows with stride=head_dim.
    const int64_t head_dim = elem_size_ / 4;
    const int64_t split_offset = static_cast<int64_t>(split_id) * kTileDim;
    const uint32_t lane_id = lid % kSubGroupSize;

    if (pid >= num_write_) {
      return;
    }

    const WritePlan plan = plan_w_[pid];
    if (plan.is_invalid()) {
      return;
    }

    const input_t* kv_src = kv_input_ + static_cast<int64_t>(plan.ragged_id) * elem_size_;
    buffer_t* kv_dst = kv_buffer_ + static_cast<int64_t>(plan.write_loc) * elem_size_;

    c4_write_token_strided<buffer_t, input_t>(kv_dst, kv_src, split_offset, lane_id, head_dim);
  }

  buffer_t* kv_buffer_;
  const input_t* kv_input_;
  const WritePlan* plan_w_;
  uint32_t num_write_;
  int64_t elem_size_;  // 4 * head_dim
  uint32_t num_split_;
};

}  // namespace FlashCompress4Impl

SGL_KERNEL_EXPORT void flash_compress4_decode(
    torch::Tensor kv_buffer, torch::Tensor kv_input, torch::Tensor kv_output, torch::Tensor ape, torch::Tensor plan_d) {
  CHECK_INPUT(kv_buffer);
  CHECK_DIM(3, kv_buffer);
  CHECK_INPUT(kv_input);
  CHECK_DIM(2, kv_input);
  CHECK_INPUT(kv_output);
  CHECK_DIM(2, kv_output);
  CHECK_INPUT(ape);
  CHECK_DIM(2, ape);
  CHECK_INPUT(plan_d);
  CHECK_DIM(2, plan_d);
  CHECK_EQ(plan_d.dtype(), torch::kUInt8);
  CHECK_EQ(kv_input.dtype(), kv_output.dtype());
  CHECK_EQ(kv_input.dtype(), ape.dtype());

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
        const uint32_t num_sgs = static_cast<uint32_t>(batch_size) * num_split;
        const uint32_t num_blocks = (num_sgs + kSubGroupsPerBlock - 1) / kSubGroupsPerBlock;
        const uint32_t global_size = num_blocks * kBlockSize;
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kBlockSize)), kernel);
      });
    });
  });
}

SGL_KERNEL_EXPORT void flash_compress4_prefill(
    torch::Tensor kv_buffer,
    torch::Tensor kv_input,
    torch::Tensor kv_output,
    torch::Tensor ape,
    torch::Tensor plan_c,
    torch::Tensor plan_w) {
  CHECK_INPUT(kv_buffer);
  CHECK_DIM(3, kv_buffer);
  CHECK_INPUT(kv_input);
  CHECK_DIM(2, kv_input);
  CHECK_INPUT(kv_output);
  CHECK_DIM(2, kv_output);
  CHECK_INPUT(ape);
  CHECK_DIM(2, ape);
  CHECK_INPUT(plan_c);
  CHECK_DIM(2, plan_c);
  CHECK_EQ(plan_c.dtype(), torch::kUInt8);
  CHECK_INPUT(plan_w);
  CHECK_DIM(2, plan_w);
  CHECK_EQ(plan_w.dtype(), torch::kUInt8);
  CHECK_EQ(kv_input.dtype(), kv_output.dtype());
  CHECK_EQ(kv_input.dtype(), ape.dtype());

  const int64_t num_compress = plan_c.size(0);
  const int64_t num_write = plan_w.size(0);
  const int64_t num_q_tokens = kv_input.size(0);
  const int64_t elem_size = kv_input.size(1);
  const int64_t head_dim = kv_output.size(1);
  TORCH_CHECK(head_dim % kTileDim == 0, "flash_compress4_prefill requires head_dim divisible by ", kTileDim);
  TORCH_CHECK(elem_size == 4 * head_dim, "kv_input last dim must be 4 * head_dim");
  TORCH_CHECK(
      kv_buffer.size(1) == 4 && kv_buffer.size(2) == elem_size, "kv_buffer shape must be [num_pages, 4, 4*head_dim]");
  TORCH_CHECK(ape.size(0) == 8 && ape.size(1) == head_dim, "ape shape must be [8, head_dim]");
  TORCH_CHECK(
      plan_c.size(1) == static_cast<int64_t>(sizeof(CompressPlan)), "plan_c row size must be sizeof(CompressPlan)");
  TORCH_CHECK(plan_w.size(1) == static_cast<int64_t>(sizeof(WritePlan)), "plan_w row size must be sizeof(WritePlan)");
  TORCH_CHECK(kv_output.size(0) == num_compress, "kv_output rows must match num compress plans");
  TORCH_CHECK(num_q_tokens >= num_write, "invalid prefill plan: num_q < num_w");

  if (num_compress == 0 && num_write == 0) {
    return;
  }

  const int64_t page_elem_size = 4 * elem_size;
  const uint32_t num_split = static_cast<uint32_t>(head_dim / kTileDim);
  auto queue = c10::xpu::getCurrentXPUStream().queue();

  SYCL_DISPATCH_FLOATING_TYPES(at::kHalf, at::kBFloat16, kv_input.scalar_type(), "FlashCompress4Prefill", [&]() {
    using input_t = scalar_t;
    using output_t = scalar_t;
    SYCL_DISPATCH_WEIGHT_TYPES(at::kHalf, at::kBFloat16, kv_buffer.scalar_type(), "FlashCompress4Prefill", [&]() {
      if (num_compress > 0) {
        queue.submit([&](sycl::handler& cgh) {
          FlashCompress4Impl::FlashCompress4CompressKernel<weight_t, input_t, output_t> kernel{
              kv_buffer.data_ptr<weight_t>(),
              kv_input.data_ptr<input_t>(),
              kv_output.data_ptr<output_t>(),
              ape.data_ptr<input_t>(),
              reinterpret_cast<const CompressPlan*>(plan_c.data_ptr<uint8_t>()),
              static_cast<uint32_t>(num_compress),
              head_dim,
              elem_size,
              page_elem_size,
              num_split};
          const uint32_t num_sgs = static_cast<uint32_t>(num_compress) * num_split;
          const uint32_t num_blocks = (num_sgs + kSubGroupsPerBlock - 1) / kSubGroupsPerBlock;
          const uint32_t global_size = num_blocks * kBlockSize;
          cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kBlockSize)), kernel);
        });
      }
      if (num_write > 0) {
        // Write path uses the same split count as head_dim partitioning.
        const uint32_t num_split_write = num_split;
        queue.submit([&](sycl::handler& cgh) {
          FlashCompress4Impl::FlashCompress4WriteKernel<weight_t, input_t> kernel{
              kv_buffer.data_ptr<weight_t>(),
              kv_input.data_ptr<input_t>(),
              reinterpret_cast<const WritePlan*>(plan_w.data_ptr<uint8_t>()),
              static_cast<uint32_t>(num_write),
              elem_size,
              num_split_write};
          const uint32_t num_sgs = static_cast<uint32_t>(num_write) * num_split_write;
          const uint32_t num_blocks = (num_sgs + kSubGroupsPerBlock - 1) / kSubGroupsPerBlock;
          const uint32_t global_size = num_blocks * kBlockSize;
          cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kBlockSize)), kernel);
        });
      }
    });
  });
}

}  // namespace at::native::xpu
