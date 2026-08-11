#include <ATen/ATen.h>
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
constexpr int64_t kTokenGroups = 8;
constexpr int64_t kTokensPerGroup = 128 / kTokenGroups;                          // 16
constexpr uint32_t kBlockSize = static_cast<uint32_t>(kTileDim * kTokenGroups);  // 512
constexpr uint32_t kWriteBlockSize = 128;
constexpr uint32_t kSubGroupsPerWriteBlock = kWriteBlockSize / kSubGroupSize;  // 8

namespace FlashCompress128Impl {

template <typename buffer_t, typename input_t>
inline void c128_write_token_strided(
    buffer_t* kv_dst,
    const input_t* kv_src,
    const int64_t split_offset,
    const uint32_t lane_id,
    const int64_t row_stride) {
  const int64_t lane_base = split_offset + static_cast<int64_t>(lane_id) * kTileElements;

  if constexpr (std::is_same_v<buffer_t, input_t> && sizeof(input_t) == 2) {
    // Fast path: copy 4 contiguous elements (8 bytes) per row via two 32-bit moves.
    const uint32_t* src32_row0 = reinterpret_cast<const uint32_t*>(kv_src + lane_base);
    const uint32_t* src32_row1 = reinterpret_cast<const uint32_t*>(kv_src + row_stride + lane_base);
    uint32_t* dst32_row0 = reinterpret_cast<uint32_t*>(kv_dst + lane_base);
    uint32_t* dst32_row1 = reinterpret_cast<uint32_t*>(kv_dst + row_stride + lane_base);
    dst32_row0[0] = src32_row0[0];
    dst32_row0[1] = src32_row0[1];
    dst32_row1[0] = src32_row1[0];
    dst32_row1[1] = src32_row1[1];
  } else {
    // Mixed dtype fallback: preserve conversion semantics.
    for (int64_t x = 0; x < kTileElements; ++x) {
      kv_dst[lane_base + x] = static_cast<buffer_t>(kv_src[lane_base + x]);
      kv_dst[row_stride + lane_base + x] = static_cast<buffer_t>(kv_src[row_stride + lane_base + x]);
    }
  }
}

// Load + online softmax + weighted reduction shared by decode and prefill compress.
// For decode, pass buffer_len=128 so all tokens load from kv_buf.
template <typename buffer_t, typename input_t, typename out_t>
inline void c128_forward(
    const buffer_t* kv_buf,
    const input_t* kv_src,
    const input_t* ape,
    int64_t fresh_start,
    int64_t buffer_len,
    int64_t t_start,
    int64_t head_dim,
    int64_t elem_size,
    int64_t h,
    int64_t tg,
    int64_t h_lane,
    out_t* kv_out,
    sycl::nd_item<1> item,
    sycl::local_accessor<float, 1> shared) {
  float kv_reg[kTokensPerGroup];
  float score_reg[kTokensPerGroup];
  for (int64_t i = 0; i < kTokensPerGroup; ++i) {
    const int64_t t = t_start + i;
    if (t < buffer_len) {
      const buffer_t* row_ptr = kv_buf + t * elem_size;
      kv_reg[i] = static_cast<float>(row_ptr[h]);
      score_reg[i] = static_cast<float>(row_ptr[head_dim + h]) + static_cast<float>(ape[t * head_dim + h]);
    } else {
      const input_t* row_ptr = kv_src + (fresh_start + (t - buffer_len)) * elem_size;
      kv_reg[i] = static_cast<float>(row_ptr[h]);
      score_reg[i] = static_cast<float>(row_ptr[head_dim + h]) + static_cast<float>(ape[t * head_dim + h]);
    }
  }

  float local_max = score_reg[0];
  for (int64_t i = 1; i < kTokensPerGroup; ++i)
    local_max = sycl::fmax(local_max, score_reg[i]);

  float local_exp_sum = 0.0f;
  float local_weighted_sum = 0.0f;
  for (int64_t i = 0; i < kTokensPerGroup; ++i) {
    const float exp_score = sycl::exp(score_reg[i] - local_max);
    local_exp_sum += exp_score;
    local_weighted_sum += kv_reg[i] * exp_score;
  }

  // Shared memory layout (3 sections, each [kTokenGroups][kTileDim]).
  const size_t kSmemSection = static_cast<size_t>(kTokenGroups * kTileDim);
  const size_t smem_idx = static_cast<size_t>(tg * kTileDim + h_lane);
  shared[smem_idx] = local_max;
  item.barrier(sycl::access::fence_space::local_space);

  float global_max = local_max;
  for (int64_t g = 0; g < kTokenGroups; ++g)
    global_max = sycl::fmax(global_max, shared[static_cast<size_t>(g * kTileDim) + static_cast<size_t>(h_lane)]);

  const float rescale = sycl::exp(local_max - global_max);
  shared[kSmemSection + smem_idx] = local_exp_sum * rescale;
  shared[2 * kSmemSection + smem_idx] = local_weighted_sum * rescale;
  item.barrier(sycl::access::fence_space::local_space);

  if (tg == 0) {
    float exp_sum = 0.0f;
    float weighted_sum = 0.0f;
    for (int64_t g = 0; g < kTokenGroups; ++g) {
      exp_sum += shared[kSmemSection + static_cast<size_t>(g * kTileDim) + static_cast<size_t>(h_lane)];
      weighted_sum += shared[2 * kSmemSection + static_cast<size_t>(g * kTileDim) + static_cast<size_t>(h_lane)];
    }
    kv_out[h] = static_cast<out_t>(weighted_sum / exp_sum);
  }
}

template <typename buffer_t, typename input_t, typename out_t>
struct FlashCompress128DecodeKernel {
  [[sycl::reqd_sub_group_size(kSubGroupSize)]]
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

    // lid decomposition: tg = token group (0..7), h_lane = column within split (0..63)
    const int64_t h_lane = static_cast<int64_t>(lid % static_cast<uint32_t>(kTileDim));
    const int64_t tg = static_cast<int64_t>(lid / static_cast<uint32_t>(kTileDim));
    const int64_t h = split_offset + h_lane;
    const uint32_t sg_id = lid / kSubGroupSize;
    const uint32_t lane_id = lid % kSubGroupSize;

    // only the first subgroup performs decode writeback.
    if (sg_id == 0) {
      buffer_t* kv_dst = kv_buffer_ + static_cast<int64_t>(plan.write_loc) * elem_size_;
      const input_t* kv_src = kv_input_ + static_cast<int64_t>(bid) * elem_size_;
      c128_write_token_strided<buffer_t, input_t>(kv_dst, kv_src, split_offset, lane_id, head_dim_);
    }

    // Compress only when a 128-token chunk is completed.
    if (plan.seq_len % 128 != 0) {
      return;
    }

    item.barrier(sycl::access::fence_space::global_and_local);

    // buffer_len=128: all tokens come from kv_buf, kv_src/fresh_start unused.
    c128_forward(
        kv_buffer_ + static_cast<int64_t>(plan.read_page_1) * page_elem_size_,
        kv_input_,
        ape_,
        0,
        128,
        tg * kTokensPerGroup,
        head_dim_,
        elem_size_,
        h,
        tg,
        h_lane,
        kv_output_ + static_cast<int64_t>(bid) * head_dim_,
        item,
        shared_);
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
  sycl::local_accessor<float, 1> shared_;
};

template <typename buffer_t, typename input_t, typename out_t>
struct FlashCompress128PrefillCompressKernel {
  [[sycl::reqd_sub_group_size(kSubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t gid = static_cast<uint32_t>(item.get_group(0));
    const uint32_t lid = static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t pid = gid / num_split_;
    const int64_t split_offset = static_cast<int64_t>(gid % num_split_) * kTileDim;

    if (pid >= num_compress_) {
      return;
    }

    const CompressPlan plan = plan_c_[pid];
    if (plan.is_invalid()) {
      return;
    }

    // lid decomposition: tg = token group (0..7), h_lane = column within split (0..63)
    const int64_t h_lane = static_cast<int64_t>(lid % static_cast<uint32_t>(kTileDim));
    const int64_t tg = static_cast<int64_t>(lid / static_cast<uint32_t>(kTileDim));
    const int64_t h = split_offset + h_lane;
    const int64_t buffer_len = static_cast<int64_t>(plan.buffer_len);
    const int64_t fresh_start = static_cast<int64_t>(plan.ragged_id) - 127 + buffer_len;

    c128_forward(
        kv_buffer_ + static_cast<int64_t>(plan.read_page_1) * page_elem_size_,
        kv_input_,
        ape_,
        fresh_start,
        buffer_len,
        tg * kTokensPerGroup,
        head_dim_,
        elem_size_,
        h,
        tg,
        h_lane,
        kv_output_ + static_cast<int64_t>(pid) * head_dim_,
        item,
        shared_);
  }

  buffer_t* kv_buffer_;
  const input_t* kv_input_;
  out_t* kv_output_;
  const input_t* ape_;
  const CompressPlan* plan_c_;
  uint32_t num_compress_;
  int64_t head_dim_;
  int64_t elem_size_;
  int64_t page_elem_size_;
  uint32_t num_split_;
  sycl::local_accessor<float, 1> shared_;
};

template <typename buffer_t, typename input_t>
struct FlashCompress128PrefillWriteKernel {
  [[sycl::reqd_sub_group_size(kSubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t gid = static_cast<uint32_t>(item.get_group(0));
    const uint32_t lid = static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t global_tid = gid * kWriteBlockSize + lid;
    const uint32_t global_sg_id = global_tid / kSubGroupSize;
    const uint32_t pid = global_sg_id / num_split_;
    // Contiguous flatten split: split [head_dim * 2] into num_split tiles of size (kTileDim * 2).
    const int64_t split_offset = static_cast<int64_t>(global_sg_id % num_split_) * (kTileDim * 2);

    if (pid >= num_write_) {
      return;
    }

    const WritePlan plan = plan_w_[pid];
    if (plan.is_invalid()) {
      return;
    }

    buffer_t* kv_dst = kv_buffer_ + static_cast<int64_t>(plan.write_loc) * elem_size_;
    const input_t* kv_src = kv_input_ + static_cast<int64_t>(plan.ragged_id) * elem_size_;
    const uint32_t lane_id = lid % kSubGroupSize;
    c128_write_token_strided<buffer_t, input_t>(kv_dst, kv_src, split_offset, lane_id, kTileDim);
  }

  buffer_t* kv_buffer_;
  const input_t* kv_input_;
  const WritePlan* plan_w_;
  uint32_t num_write_;
  int64_t elem_size_;
  uint32_t num_split_;
};

}  // namespace FlashCompress128Impl

SGL_KERNEL_EXPORT void flash_compress128_decode(
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
  TORCH_CHECK(head_dim % kTileDim == 0, "flash_compress128_decode requires head_dim divisible by ", kTileDim);
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
  const uint32_t num_split = static_cast<uint32_t>(head_dim / kTileDim);
  auto queue = c10::xpu::getCurrentXPUStream().queue();

  SYCL_DISPATCH_FLOATING_TYPES(at::kHalf, at::kBFloat16, kv_input.scalar_type(), "FlashCompress128Decode", [&]() {
    using input_t = scalar_t;
    using output_t = scalar_t;
    SYCL_DISPATCH_WEIGHT_TYPES(at::kHalf, at::kBFloat16, kv_buffer.scalar_type(), "FlashCompress128Decode", [&]() {
      queue.submit([&](sycl::handler& cgh) {
        // Shared memory: 3 sections × [kTokenGroups × kTileDim] floats.
        const size_t kSmemElems = static_cast<size_t>(3 * kTokenGroups * kTileDim);
        sycl::local_accessor<float, 1> shared(sycl::range<1>(kSmemElems), cgh);
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
            num_split,
            shared};
        const uint32_t global_size = static_cast<uint32_t>(batch_size) * num_split * kBlockSize;
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kBlockSize)), kernel);
      });
    });
  });
}

SGL_KERNEL_EXPORT void flash_compress128_prefill(
    torch::Tensor kv_buffer,
    torch::Tensor kv_input,
    torch::Tensor kv_output,
    torch::Tensor ape,
    torch::Tensor plan_c,
    torch::Tensor plan_w) {
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
      plan_c.is_xpu() && plan_c.dim() == 2 && plan_c.dtype() == torch::kUInt8 && plan_c.is_contiguous(),
      "plan_c must be a contiguous 2D XPU uint8 tensor");
  TORCH_CHECK(
      plan_w.is_xpu() && plan_w.dim() == 2 && plan_w.dtype() == torch::kUInt8 && plan_w.is_contiguous(),
      "plan_w must be a contiguous 2D XPU uint8 tensor");
  TORCH_CHECK(kv_input.dtype() == kv_output.dtype(), "kv_input and kv_output must have the same dtype");
  TORCH_CHECK(kv_input.dtype() == ape.dtype(), "kv_input and ape must have same dtype");

  const int64_t num_q_tokens = kv_input.size(0);
  const int64_t elem_size = kv_input.size(1);
  const int64_t head_dim = kv_output.size(1);
  const int64_t num_compress = kv_output.size(0);
  const int64_t num_write = plan_w.size(0);
  TORCH_CHECK(head_dim % kTileDim == 0, "flash_compress128_prefill requires head_dim divisible by ", kTileDim);
  TORCH_CHECK(elem_size == 2 * head_dim, "kv_input last dim must be 2 * head_dim");
  TORCH_CHECK(
      kv_buffer.size(1) == 128 && kv_buffer.size(2) == elem_size,
      "kv_buffer shape must be [num_pages, 128, 2*head_dim]");
  TORCH_CHECK(ape.size(0) == 128 && ape.size(1) == head_dim, "ape shape must be [128, head_dim]");
  TORCH_CHECK(
      plan_c.size(0) == num_compress && plan_c.size(1) == static_cast<int64_t>(sizeof(CompressPlan)),
      "plan_c shape must be [C, 16]");
  TORCH_CHECK(plan_w.size(1) == static_cast<int64_t>(sizeof(WritePlan)), "plan_w shape must be [W, 8]");
  TORCH_CHECK(num_q_tokens >= num_write, "invalid prefill plan: num_q < num_w");

  if (num_compress == 0 && num_write == 0) {
    return;
  }

  const int64_t page_elem_size = 128 * elem_size;
  const uint32_t num_split = static_cast<uint32_t>(head_dim / kTileDim);
  auto queue = c10::xpu::getCurrentXPUStream().queue();

  SYCL_DISPATCH_FLOATING_TYPES(at::kHalf, at::kBFloat16, kv_input.scalar_type(), "FlashCompress128Prefill", [&]() {
    using input_t = scalar_t;
    using output_t = scalar_t;
    SYCL_DISPATCH_WEIGHT_TYPES(at::kHalf, at::kBFloat16, kv_buffer.scalar_type(), "FlashCompress128Prefill", [&]() {
      if (num_compress > 0) {
        queue.submit([&](sycl::handler& cgh) {
          // Shared memory: 3 sections × [kTokenGroups × kTileDim] floats.
          const size_t kSmemElems = static_cast<size_t>(3 * kTokenGroups * kTileDim);
          sycl::local_accessor<float, 1> shared(sycl::range<1>(kSmemElems), cgh);
          FlashCompress128Impl::FlashCompress128PrefillCompressKernel<weight_t, input_t, output_t> kernel{
              kv_buffer.data_ptr<weight_t>(),
              kv_input.data_ptr<input_t>(),
              kv_output.data_ptr<output_t>(),
              ape.data_ptr<input_t>(),
              reinterpret_cast<const CompressPlan*>(plan_c.data_ptr<uint8_t>()),
              static_cast<uint32_t>(num_compress),
              head_dim,
              elem_size,
              page_elem_size,
              num_split,
              shared};
          const uint32_t global_size = static_cast<uint32_t>(num_compress) * num_split * kBlockSize;
          cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kBlockSize)), kernel);
        });
      }

      if (num_write > 0) {
        queue.submit([&](sycl::handler& cgh) {
          FlashCompress128Impl::FlashCompress128PrefillWriteKernel<weight_t, input_t> kernel{
              kv_buffer.data_ptr<weight_t>(),
              kv_input.data_ptr<input_t>(),
              reinterpret_cast<const WritePlan*>(plan_w.data_ptr<uint8_t>()),
              static_cast<uint32_t>(num_write),
              elem_size,
              num_split};
          const uint32_t total_sgs = static_cast<uint32_t>(num_write) * num_split;
          const uint32_t num_w_blocks = (total_sgs + kSubGroupsPerWriteBlock - 1) / kSubGroupsPerWriteBlock;
          const uint32_t global_size = num_w_blocks * kWriteBlockSize;
          cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kWriteBlockSize)), kernel);
        });
      }
    });
  });
}

}  // namespace at::native::xpu
