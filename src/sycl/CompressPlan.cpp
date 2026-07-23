#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <limits>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

struct alignas(16) DecodePlan {
  uint32_t seq_len;
  int32_t write_loc;
  int32_t read_page_0;
  int32_t read_page_1;
};
static_assert(sizeof(DecodePlan) == 16, "DecodePlan must be 16 bytes");

namespace CompressPlanImpl {

// Kernel for plan_compress_decode
struct CompressDecodeKernel {
  void operator()(sycl::nd_item<1> item) const {
    uint32_t idx = item.get_global_id(0);
    if (idx >= batch_size) return;

    int64_t rid = rid_ptr[idx];
    int32_t seq_len = static_cast<int32_t>(seq_ptr[idx]);
    int32_t position_1 = seq_len - 1;
    int32_t position_0 = sycl::max(position_1 - compress_ratio, 0);

    const auto compute_loc = [&](int32_t swa_loc) {
      int32_t swa_page = swa_loc / swa_page_size;
      int32_t ring_offset = swa_loc % ring_size;
      return swa_page * ring_size + ring_offset;
    };
    const auto compute_c128_loc = [&](int64_t rid_, int32_t position) {
      return static_cast<int32_t>(rid_ * ring_size + position % ring_size);
    };

    int32_t write_loc;
    int32_t read_page_0;
    int32_t read_page_1;
    if (compress_ratio == 128) {
      write_loc = compute_c128_loc(rid, position_1);
      read_page_0 = compute_c128_loc(rid, position_0) / 128;
      read_page_1 = write_loc / 128;
    } else {
      const int32_t* mapping = r2t_ptr + rid * stride_r2t;

      // Look up full token indices
      const auto raw_loc_0 = mapping[position_0];
      const auto raw_loc_1 = mapping[position_1];

      // Convert to swa indices
      const auto state_loc_0 = f2s_ptr[raw_loc_0];
      const auto state_loc_1 = f2s_ptr[raw_loc_1];
      // Compute ring buffer locations
      write_loc = static_cast<int32_t>(compute_loc(state_loc_1));
      read_page_0 = static_cast<int32_t>(compute_loc(state_loc_0) / compress_ratio);
      read_page_1 = static_cast<int32_t>(write_loc / compress_ratio);
    }

    // Pack into output: [seq_len, write_loc, read_page_0, read_page_1]
    plan_d[idx] = DecodePlan{
        static_cast<uint32_t>(seq_len),
        write_loc,
        read_page_0,
        read_page_1,
    };
  }

  const int64_t* rid_ptr;
  const int64_t* seq_ptr;
  const int32_t* r2t_ptr;
  const int64_t* f2s_ptr;
  DecodePlan* plan_d;
  int32_t swa_page_size;
  int32_t ring_size;
  int32_t compress_ratio;
  int64_t stride_r2t;
  uint32_t batch_size;
};

// Kernel for plan_compress_prefill stage 0
struct CompressPrefillStage0Kernel {
  void operator()(sycl::nd_item<1> item) const {
    if (item.get_global_id(0) != 0) {
      return;
    }

    const bool is_overlap = (compress_ratio_ == 4);
    const int32_t window_size = compress_ratio_ * (is_overlap ? 2 : 1);

    uint32_t counter = 0;
    uint32_t counter_c = 0;
    uint32_t counter_w = 0;

    for (uint32_t batch_id = 0; batch_id < batch_size_; ++batch_id) {
      const int32_t seq_len = static_cast<int32_t>(seq_lens_ptr_[batch_id]);
      const int32_t extend_len = static_cast<int32_t>(extend_lens_ptr_[batch_id]);
      const int32_t prefix_len = seq_len - extend_len;
      const int32_t last_c_pos = (seq_len / compress_ratio_) * compress_ratio_;
      const int32_t first_w_pos = sycl::min(last_c_pos - (is_overlap ? compress_ratio_ : 0), seq_len - mtp_pad_);

      for (int32_t j = 0; j < extend_len; ++j) {
        const int32_t position = prefix_len + j;
        const uint32_t ragged_id = counter + static_cast<uint32_t>(j);

        if (((position + 1) % compress_ratio_) == 0) {
          const int32_t buffer_len = window_size - sycl::min(j + 1, window_size);
          int32_t* plan = plan_c_i32_ + static_cast<int32_t>(counter_c) * 4;
          plan[0] = position + 1;
          plan[1] = ((buffer_len & 0xFFFF) << 16) | (static_cast<int32_t>(ragged_id) & 0xFFFF);
          plan[2] = -1;
          plan[3] = static_cast<int32_t>(batch_id);
          ++counter_c;
        }

        bool do_write = position >= first_w_pos;
        if (!do_write && is_overlap) {
          do_write = (position % swa_page_size_) >= (swa_page_size_ - compress_ratio_);
        }
        if (do_write) {
          int32_t* plan = plan_w_i32_ + static_cast<int32_t>(counter_w) * 2;
          plan[0] = ((static_cast<int32_t>(batch_id) & 0xFFFF) << 16) | (static_cast<int32_t>(ragged_id) & 0xFFFF);
          plan[1] = position + 1;
          ++counter_w;
        }
      }
      counter += static_cast<uint32_t>(extend_len);
    }

    for (uint32_t k = counter_c; k < num_q_tokens_; ++k) {
      int32_t* plan = plan_c_i32_ + static_cast<int32_t>(k) * 4;
      plan[0] = -1;
      plan[1] = 0;
      plan[2] = -1;
      plan[3] = -1;
    }
    for (uint32_t k = counter_w; k < num_q_tokens_; ++k) {
      int32_t* plan = plan_w_i32_ + static_cast<int32_t>(k) * 2;
      plan[0] = -1;
      plan[1] = -1;
    }
  }

  int32_t* plan_c_i32_;
  int32_t* plan_w_i32_;
  const int64_t* seq_lens_ptr_;
  const int64_t* extend_lens_ptr_;
  uint32_t batch_size_;
  uint32_t num_q_tokens_;
  int32_t compress_ratio_;
  int32_t swa_page_size_;
  int32_t mtp_pad_;
};

// Kernel for plan_compress_prefill stage 1
struct CompressPrefillStage1Kernel {
  void operator()(sycl::nd_item<1> item) const {
    uint32_t idx = static_cast<uint32_t>(item.get_global_id(0));
    if (idx >= num_work_) return;

    int32_t* plan_c = plan_c_i32_ + idx * 4;
    if (plan_c[0] != -1) {
      int32_t batch_id = plan_c[3];
      int32_t position_1 = plan_c[0] - 1;
      int32_t position_0 = sycl::max(position_1 - compress_ratio_, 0);
      int32_t buffer_len = (plan_c[1] >> 16) & 0xFFFF;
      bool has_buffer = buffer_len > 0;

      int64_t rid = rid_ptr_[batch_id];

      int64_t read_page_1;
      int64_t read_page_0;
      if (compress_ratio_ == 128) {
        read_page_1 = rid * ring_size_div_ratio_ + ((static_cast<int64_t>(position_1) % ring_size_) / compress_ratio_);
        read_page_0 = rid * ring_size_div_ratio_ + ((static_cast<int64_t>(position_0) % ring_size_) / compress_ratio_);
      } else {
        const int32_t* mapping = r2t_ptr_ + rid * stride_r2t_;
        int64_t raw_loc_1 = static_cast<int64_t>(mapping[position_1]);
        int64_t raw_loc_0 = static_cast<int64_t>(mapping[position_0]);
        int64_t state_loc_1 = f2s_ptr_[raw_loc_1];
        int64_t state_loc_0 = f2s_ptr_[raw_loc_0];
        int64_t write_loc = (state_loc_1 / swa_page_size_) * ring_size_ + (state_loc_1 % ring_size_);
        int64_t read_loc_0 = (state_loc_0 / swa_page_size_) * ring_size_ + (state_loc_0 % ring_size_);
        read_page_1 = write_loc / static_cast<int64_t>(compress_ratio_);
        read_page_0 = read_loc_0 / static_cast<int64_t>(compress_ratio_);
      }

      plan_c[2] = has_buffer ? static_cast<int32_t>(read_page_0) : -1;
      plan_c[3] = has_buffer ? static_cast<int32_t>(read_page_1) : batch_id;
    }

    int32_t* plan_w = plan_w_i32_ + idx * 2;
    if (plan_w[0] != -1) {
      int32_t packed = plan_w[0];
      int32_t batch_id = (packed >> 16) & 0xFFFF;
      int32_t position = plan_w[1] - 1;
      int64_t rid = rid_ptr_[batch_id];

      int64_t write_loc;
      if (compress_ratio_ == 128) {
        write_loc = rid * ring_size_ + (static_cast<int64_t>(position) % ring_size_);
      } else {
        const int32_t* mapping = r2t_ptr_ + rid * stride_r2t_;
        int64_t raw_loc = static_cast<int64_t>(mapping[position]);
        int64_t state_loc = f2s_ptr_[raw_loc];
        write_loc = (state_loc / swa_page_size_) * ring_size_ + (state_loc % ring_size_);
      }

      plan_w[0] = packed & 0xFFFF;
      plan_w[1] = static_cast<int32_t>(write_loc);
    }
  }

  int32_t* plan_c_i32_;
  int32_t* plan_w_i32_;
  const int64_t* rid_ptr_;
  const int32_t* r2t_ptr_;
  const int64_t* f2s_ptr_;
  int32_t swa_page_size_;
  int32_t ring_size_;
  int32_t ring_size_div_ratio_;
  int32_t compress_ratio_;
  int64_t stride_r2t_;
  uint32_t num_work_;
};

}  // namespace CompressPlanImpl

constexpr int64_t kPlanCBytes = sizeof(int32_t) * 4;
constexpr int64_t kPlanWBytes = sizeof(int32_t) * 2;
constexpr int64_t kPlanDBytes = sizeof(int32_t) * 4;

// SYCL helper to launch kernels
inline sycl::nd_range<1> get_1d_range(int32_t size) {
  constexpr int32_t local_size = 256;
  return sycl::nd_range<1>(
      sycl::range<1>((size + local_size - 1) / local_size * local_size), sycl::range<1>(local_size));
}

// XPU wrapper for plan_compress_decode
torch::Tensor plan_compress_decode(
    torch::Tensor req_pool_indices,
    torch::Tensor req_to_token,
    torch::Tensor full_to_state,
    torch::Tensor seq_lens,
    int64_t compress_ratio,
    int64_t swa_page_size,
    int64_t ring_size) {
  TORCH_CHECK(
      req_pool_indices.is_xpu() && req_pool_indices.dtype() == torch::kInt64,
      "req_pool_indices must be an int64 XPU tensor");
  TORCH_CHECK(
      req_to_token.is_xpu() && req_to_token.dtype() == torch::kInt32, "req_to_token must be an int32 XPU tensor");
  TORCH_CHECK(
      full_to_state.is_xpu() && full_to_state.dtype() == torch::kInt64, "full_to_state must be an int64 XPU tensor");
  TORCH_CHECK(seq_lens.is_xpu() && seq_lens.dtype() == torch::kInt64, "seq_lens must be an int64 XPU tensor");
  TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be a 2D tensor");
  TORCH_CHECK(req_to_token.stride(1) == 1, "req_to_token must be contiguous in the last dim");
  TORCH_CHECK(compress_ratio > 0, "compress_ratio must be > 0");

  uint32_t batch_size = static_cast<uint32_t>(seq_lens.numel());
  if (batch_size == 0) {
    return torch::empty({0, static_cast<int64_t>(sizeof(DecodePlan))}, req_pool_indices.options().dtype(torch::kUInt8));
  }

  auto output = torch::empty(
      {static_cast<int64_t>(batch_size), static_cast<int64_t>(sizeof(DecodePlan))},
      req_pool_indices.options().dtype(torch::kUInt8));

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  queue.submit([&](sycl::handler& cgh) {
    CompressPlanImpl::CompressDecodeKernel kernel{
        req_pool_indices.data_ptr<int64_t>(),
        seq_lens.data_ptr<int64_t>(),
        req_to_token.data_ptr<int32_t>(),
        full_to_state.data_ptr<int64_t>(),
        reinterpret_cast<DecodePlan*>(output.data_ptr<uint8_t>()),
        static_cast<int32_t>(swa_page_size),
        static_cast<int32_t>(ring_size),
        static_cast<int32_t>(compress_ratio),
        req_to_token.size(1),
        batch_size};
    cgh.parallel_for(get_1d_range(batch_size), kernel);
  });

  return output;
}

std::tuple<torch::Tensor, torch::Tensor> plan_compress_prefill(
    torch::Tensor req_pool_indices,
    torch::Tensor req_to_token,
    torch::Tensor full_to_state,
    torch::Tensor seq_lens,
    torch::Tensor extend_lens,
    torch::Tensor pin_buffer,
    int64_t num_q_tokens,
    int64_t compress_ratio,
    int64_t swa_page_size,
    int64_t ring_size,
    bool use_cuda_graph) {
  (void)pin_buffer;

  TORCH_CHECK(
      req_pool_indices.is_xpu() && req_pool_indices.dtype() == torch::kInt64 && req_pool_indices.dim() == 1,
      "req_pool_indices must be a 1D int64 XPU tensor");
  TORCH_CHECK(
      req_to_token.is_xpu() && req_to_token.dtype() == torch::kInt32 && req_to_token.dim() == 2,
      "req_to_token must be a 2D int32 XPU tensor");
  TORCH_CHECK(
      full_to_state.is_xpu() && full_to_state.dtype() == torch::kInt64, "full_to_state must be an int64 XPU tensor");
  TORCH_CHECK(
      (seq_lens.is_xpu() || seq_lens.device().is_cpu()) && seq_lens.dtype() == torch::kInt64 && seq_lens.dim() == 1,
      "seq_lens must be a 1D int64 tensor on XPU or CPU");
  TORCH_CHECK(
      (extend_lens.is_xpu() || extend_lens.device().is_cpu()) && extend_lens.dtype() == torch::kInt64 &&
          extend_lens.dim() == 1,
      "extend_lens must be a 1D int64 tensor on XPU or CPU");
  TORCH_CHECK(seq_lens.numel() == extend_lens.numel(), "seq_lens and extend_lens must have the same length");
  TORCH_CHECK(req_pool_indices.numel() == seq_lens.numel(), "req_pool_indices and seq_lens must have the same length");
  TORCH_CHECK(!use_cuda_graph, "plan_compress_prefill only supports use_cuda_graph=False on XPU");

  auto seq_lens_xpu = seq_lens;
  if (!seq_lens_xpu.is_xpu()) {
    seq_lens_xpu = seq_lens.to(req_pool_indices.options().dtype(torch::kInt64));
  }
  auto extend_lens_xpu = extend_lens;
  if (!extend_lens_xpu.is_xpu()) {
    extend_lens_xpu = extend_lens.to(req_pool_indices.options().dtype(torch::kInt64));
  }

  const uint32_t num_q_tokens_u32 = static_cast<uint32_t>(num_q_tokens);
  const int32_t compress_ratio_i32 = static_cast<int32_t>(compress_ratio);
  const int32_t swa_page_size_i32 = static_cast<int32_t>(swa_page_size);
  const int32_t ring_size_i32 = static_cast<int32_t>(ring_size);

  const uint32_t batch_size = static_cast<uint32_t>(seq_lens_xpu.numel());
  constexpr uint32_t kMaxTokens = static_cast<uint32_t>(std::numeric_limits<uint16_t>::max());
  constexpr uint32_t kMaxPrefillBatchSize = 1024;
  TORCH_CHECK(compress_ratio_i32 == 4 || compress_ratio_i32 == 128, "compress_ratio must be 4 or 128");
  TORCH_CHECK(
      num_q_tokens_u32 >= batch_size && num_q_tokens_u32 <= kMaxTokens, "num_q_tokens must be in [batch_size, 65535]");
  TORCH_CHECK(
      swa_page_size_i32 % ring_size_i32 == 0 && ring_size_i32 % compress_ratio_i32 == 0,
      "swa_page_size must be divisible by ring_size and ring_size must be divisible by compress_ratio");
  TORCH_CHECK(batch_size <= kMaxPrefillBatchSize, "XPU plan only supports batch size up to 1024");

  auto options_u8 = req_pool_indices.options().dtype(torch::kUInt8);
  if (num_q_tokens_u32 == 0) {
    return {
        torch::empty({0, kPlanCBytes}, options_u8),
        torch::empty({0, kPlanWBytes}, options_u8),
    };
  }

  auto plan_c = torch::empty({num_q_tokens_u32, kPlanCBytes}, options_u8);
  auto plan_w = torch::empty({num_q_tokens_u32, kPlanWBytes}, options_u8);
  constexpr int32_t kMaxMTPDraftTokens = 4;
  const int32_t mtp_pad = std::min(ring_size_i32 - compress_ratio_i32, kMaxMTPDraftTokens);

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  queue.submit([&](sycl::handler& cgh) {
    CompressPlanImpl::CompressPrefillStage0Kernel kernel{
        reinterpret_cast<int32_t*>(plan_c.data_ptr<uint8_t>()),
        reinterpret_cast<int32_t*>(plan_w.data_ptr<uint8_t>()),
        seq_lens_xpu.data_ptr<int64_t>(),
        extend_lens_xpu.data_ptr<int64_t>(),
        batch_size,
        num_q_tokens_u32,
        compress_ratio_i32,
        swa_page_size_i32,
        mtp_pad};
    cgh.parallel_for(get_1d_range(1), kernel);
  });

  int64_t stride_r2t = req_to_token.stride(0);
  if (num_q_tokens_u32 > 0) {
    queue.submit([&](sycl::handler& cgh) {
      CompressPlanImpl::CompressPrefillStage1Kernel kernel{
          reinterpret_cast<int32_t*>(plan_c.data_ptr<uint8_t>()),
          reinterpret_cast<int32_t*>(plan_w.data_ptr<uint8_t>()),
          req_pool_indices.data_ptr<int64_t>(),
          req_to_token.data_ptr<int32_t>(),
          full_to_state.data_ptr<int64_t>(),
          swa_page_size_i32,
          ring_size_i32,
          ring_size_i32 / compress_ratio_i32,
          compress_ratio_i32,
          stride_r2t,
          num_q_tokens_u32};
      cgh.parallel_for(get_1d_range(static_cast<int32_t>(num_q_tokens_u32)), kernel);
    });
  }

  return {plan_c, plan_w};
}

}  // namespace at::native::xpu
