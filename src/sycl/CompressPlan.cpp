#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <limits>
#include <sycl/sycl.hpp>
#include <utility>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

struct alignas(16) DecodePlan {
  uint32_t seq_len;
  int32_t write_loc;
  int32_t read_page_0;
  int32_t read_page_1;
};

struct alignas(16) CompressPlan {
  uint32_t seq_len;
  uint16_t ragged_id;
  uint16_t buffer_len;
  int32_t read_page_0;
  int32_t read_page_1;

  static constexpr CompressPlan invalid() {
    return CompressPlan{-1u, 0, 0, -1, -1};
  }

  constexpr bool is_invalid() const {
    return seq_len == -1u;
  }
};

struct alignas(8) WritePlan {
  uint32_t ragged_id;
  int32_t write_loc;

  static constexpr WritePlan invalid() {
    return WritePlan{-1u, -1};
  }

  constexpr bool is_invalid() const {
    return ragged_id == -1u;
  }
};

inline WritePlan pack_w(uint32_t ragged_id, uint32_t batch_id, int32_t seq_len) {
  return {static_cast<uint32_t>(ragged_id | (batch_id << 16)), seq_len};
}

inline std::pair<uint16_t, uint16_t> unpack_w(WritePlan plan) {
  return {static_cast<uint16_t>(plan.ragged_id), static_cast<uint16_t>(plan.ragged_id >> 16)};
}

static_assert(sizeof(DecodePlan) == 16, "DecodePlan must be 16 bytes");
static_assert(sizeof(CompressPlan) == 16, "CompressPlan must be 16 bytes");
static_assert(sizeof(WritePlan) == 8, "WritePlan must be 8 bytes");
constexpr uint32_t kStage0BlockSize = 256;
constexpr uint32_t kStage0SubGroupSize = 32;
constexpr uint32_t kStage0NumSubGroups = kStage0BlockSize / kStage0SubGroupSize;
static_assert(kStage0BlockSize % kStage0SubGroupSize == 0, "kStage0BlockSize must be divisible by subgroup size");

using Stage0LocalAtomic = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::work_group,
    sycl::access::address_space::local_space>;

inline uint32_t wg_reserve_u32(sycl::nd_item<1>& item, uint32_t& counter, uint32_t want) {
  auto g = item.get_group();
  const uint32_t my_off = sycl::exclusive_scan_over_group(g, want, sycl::plus<uint32_t>());
  const uint32_t wg_total = sycl::reduce_over_group(g, want, sycl::plus<uint32_t>());
  uint32_t wg_base = 0;
  if (item.get_local_id(0) == 0 && wg_total > 0) {
    wg_base = Stage0LocalAtomic(counter).fetch_add(wg_total);
  }
  wg_base = sycl::group_broadcast(g, wg_base, 0);
  return wg_base + my_off;
}

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
  [[sycl::reqd_sub_group_size(kStage0SubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    const auto tx = static_cast<uint32_t>(item.get_local_id(0));
    const auto block_size = static_cast<uint32_t>(item.get_local_range(0));
    auto sg = item.get_sub_group();
    const auto lane_id = tx % kStage0SubGroupSize;
    const auto sg_id = tx / kStage0SubGroupSize;
    const bool is_overlap = (compress_ratio_ == 4);
    const int32_t window_size = compress_ratio_ * (is_overlap ? 2 : 1);

    // === Stage A: load per-batch fields, init shared scratch ===
    if (tx == 0) {
      counter_c_local_[0] = 0;
      counter_w_local_[0] = 0;
    }

    if (tx < kStage0NumSubGroups) {
      warp_max_[tx] = 0;
      warp_min_[tx] = 0xFFFFFFFFu;
    }

    uint32_t local_max_extend = 0;
    uint32_t local_min_extend = 0xFFFFFFFFu;
    for (uint32_t batch_id = tx; batch_id < batch_size_; batch_id += block_size) {
      const int32_t seq_len = static_cast<int32_t>(seq_lens_ptr_[batch_id]);
      const int32_t extend_len = static_cast<int32_t>(extend_lens_ptr_[batch_id]);
      s_seq_len_[batch_id] = seq_len;
      s_prefix_len_[batch_id] = seq_len - extend_len;
      const uint32_t extend_len_u32 = static_cast<uint32_t>(extend_len);
      local_max_extend = sycl::max(local_max_extend, extend_len_u32);
      local_min_extend = sycl::min(local_min_extend, extend_len_u32);
    }

    // === Stage B: min/max(extend_len) for MTP-uniform detection ===
    // Level 1: reduce inside each subgroup.
    uint32_t sg_max = sycl::reduce_over_group(sg, local_max_extend, sycl::maximum<uint32_t>());
    uint32_t sg_min = sycl::reduce_over_group(sg, local_min_extend, sycl::minimum<uint32_t>());
    if (lane_id == 0) {
      warp_max_[sg_id] = sg_max;
      warp_min_[sg_id] = sg_min;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Level 2: subgroup 0 reduces subgroup outputs to block-wide min/max.
    if (sg_id == 0) {
      uint32_t v_max = (lane_id < kStage0NumSubGroups) ? warp_max_[lane_id] : 0u;
      uint32_t v_min = (lane_id < kStage0NumSubGroups) ? warp_min_[lane_id] : 0xFFFFFFFFu;
      uint32_t block_max = sycl::reduce_over_group(sg, v_max, sycl::maximum<uint32_t>());
      uint32_t block_min = sycl::reduce_over_group(sg, v_min, sycl::minimum<uint32_t>());
      if (lane_id == 0) {
        s_max_extend_[0] = block_max;
        s_min_extend_[0] = block_min;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    // MTP-uniform means each request shares the same small extend length E.
    const bool is_mtp_extend =
        (s_min_extend_[0] == s_max_extend_[0]) && (s_max_extend_[0] > 0) && (s_max_extend_[0] <= 32);

    // === Stage C: emit valid plans ===
    if (is_mtp_extend) {
      // Path 1 (token-driven): map k -> (batch_id, j) via E.
      const uint32_t e = s_max_extend_[0];
      // num_q_tokens_ is padded capacity; cap work at real tokens for safety.
      const uint32_t num_real_q = batch_size_ * e;
      for (uint32_t k_base = 0; k_base < num_real_q; k_base += block_size) {
        const uint32_t k = k_base + tx;
        uint32_t batch_id = 0;
        uint32_t ragged_id = 0;
        int32_t position = 0;
        int32_t buffer_len = 0;
        uint32_t do_compress_flag = 0;
        uint32_t do_write_flag = 0;

        if (k < num_real_q) {
          batch_id = k / e;
          const uint32_t j = k % e;
          const int32_t prefix_len = s_prefix_len_[batch_id];
          const int32_t seq_len = s_seq_len_[batch_id];
          position = prefix_len + static_cast<int32_t>(j);
          ragged_id = k;

          if (((position + 1) % compress_ratio_) == 0) {
            do_compress_flag = 1u;
            buffer_len = window_size - sycl::min(static_cast<int32_t>(j) + 1, window_size);
          }

          const int32_t last_c_pos = (seq_len / compress_ratio_) * compress_ratio_;
          const int32_t first_w_pos = sycl::min(last_c_pos - (is_overlap ? compress_ratio_ : 0), seq_len - mtp_pad_);
          bool do_write = position >= first_w_pos;
          if (!do_write && is_overlap) {
            do_write = (position % swa_page_size_) >= (swa_page_size_ - compress_ratio_);
          }
          do_write_flag = do_write ? 1u : 0u;
        }

        const uint32_t c_out_idx = wg_reserve_u32(item, counter_c_local_[0], do_compress_flag);
        if (k < num_real_q && do_compress_flag != 0u) {
          const uint32_t out_idx = c_out_idx;
          plan_c_[out_idx] = CompressPlan{
              .seq_len = static_cast<uint32_t>(position + 1),
              .ragged_id = static_cast<uint16_t>(ragged_id),
              .buffer_len = static_cast<uint16_t>(buffer_len),
              .read_page_0 = -1,
              .read_page_1 = static_cast<int32_t>(batch_id),
          };
        }
        item.barrier(sycl::access::fence_space::local_space);

        const uint32_t w_out_idx = wg_reserve_u32(item, counter_w_local_[0], do_write_flag);
        if (k < num_real_q && do_write_flag != 0u) {
          plan_w_[w_out_idx] = pack_w(ragged_id, batch_id, position + 1);
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
    } else {
      // Path 2: general prefill (long extend_len). Iterate batches in an outer loop;
      // the whole block sweeps each batch's tokens in parallel.
      uint32_t base_e = 0;
      for (uint32_t batch_id = 0; batch_id < batch_size_; ++batch_id) {
        const int32_t prefix_len = s_prefix_len_[batch_id];
        const int32_t seq_len = s_seq_len_[batch_id];
        const int32_t extend_len = seq_len - prefix_len;
        const int32_t last_c_pos = (seq_len / compress_ratio_) * compress_ratio_;
        const int32_t first_w_pos = sycl::min(last_c_pos - (is_overlap ? compress_ratio_ : 0), seq_len - mtp_pad_);

        // Chunk by block_size to avoid cross-chunk interleaving when extend_len is large.
        for (int32_t j_base = 0; j_base < extend_len; j_base += static_cast<int32_t>(block_size)) {
          const int32_t j = j_base + static_cast<int32_t>(tx);
          int32_t buffer_len = 0;
          uint32_t do_compress_flag = 0;
          uint32_t do_write_flag = 0;
          if (j < extend_len) {
            const int32_t position = prefix_len + j;
            const uint32_t ragged_id = base_e + static_cast<uint32_t>(j);

            if (((position + 1) % compress_ratio_) == 0) {
              do_compress_flag = 1u;
              buffer_len = window_size - sycl::min(j + 1, window_size);
            }

            bool do_write = position >= first_w_pos;
            if (!do_write && is_overlap) {
              do_write = (position % swa_page_size_) >= (swa_page_size_ - compress_ratio_);
            }
            do_write_flag = do_write ? 1u : 0u;
          }

          const uint32_t c_out_idx = wg_reserve_u32(item, counter_c_local_[0], do_compress_flag);
          if (j < extend_len && do_compress_flag != 0u) {
            const int32_t position = prefix_len + j;
            const uint32_t ragged_id = base_e + static_cast<uint32_t>(j);
            plan_c_[c_out_idx] = CompressPlan{
                .seq_len = static_cast<uint32_t>(position + 1),
                .ragged_id = static_cast<uint16_t>(ragged_id),
                .buffer_len = static_cast<uint16_t>(buffer_len),
                .read_page_0 = -1,
                .read_page_1 = static_cast<int32_t>(batch_id),
            };
          }
          item.barrier(sycl::access::fence_space::local_space);

          const uint32_t w_out_idx = wg_reserve_u32(item, counter_w_local_[0], do_write_flag);
          if (j < extend_len && do_write_flag != 0u) {
            const uint32_t ragged_id = base_e + static_cast<uint32_t>(j);
            const int32_t position = prefix_len + j;
            plan_w_[w_out_idx] = pack_w(ragged_id, batch_id, position + 1);
          }

          item.barrier(sycl::access::fence_space::local_space);
        }

        base_e += static_cast<uint32_t>(extend_len);
        item.barrier(sycl::access::fence_space::local_space);
      }
    }

    // === Stage D: pad [counter_c, num_q) / [counter_w, num_q) with invalid ===
    item.barrier(sycl::access::fence_space::local_space);
    const uint32_t total_c = counter_c_local_[0];
    const uint32_t total_w = counter_w_local_[0];
    for (uint32_t k = total_c + tx; k < num_q_tokens_; k += block_size) {
      plan_c_[k] = CompressPlan::invalid();
    }
    for (uint32_t k = total_w + tx; k < num_q_tokens_; k += block_size) {
      plan_w_[k] = WritePlan::invalid();
    }
  }

  CompressPlan* plan_c_;
  WritePlan* plan_w_;
  const int64_t* seq_lens_ptr_;
  const int64_t* extend_lens_ptr_;
  uint32_t batch_size_;
  uint32_t num_q_tokens_;
  int32_t compress_ratio_;
  int32_t swa_page_size_;
  int32_t mtp_pad_;
  sycl::local_accessor<uint32_t, 1> counter_c_local_;
  sycl::local_accessor<uint32_t, 1> counter_w_local_;
  sycl::local_accessor<int32_t, 1> s_seq_len_;
  sycl::local_accessor<int32_t, 1> s_prefix_len_;
  sycl::local_accessor<uint32_t, 1> warp_max_;
  sycl::local_accessor<uint32_t, 1> warp_min_;
  sycl::local_accessor<uint32_t, 1> s_max_extend_;
  sycl::local_accessor<uint32_t, 1> s_min_extend_;
};

// Kernel for plan_compress_prefill stage 1
struct CompressPrefillStage1Kernel {
  void operator()(sycl::nd_item<1> item) const {
    const auto idx = static_cast<uint32_t>(item.get_global_id(0));
    if (idx >= num_work_) return;

    const auto compute_loc = [&](int32_t swa_loc) {
      const auto swa_page = swa_loc / swa_page_size_;
      const auto ring_offset = swa_loc % ring_size_;
      return swa_page * ring_size_ + ring_offset;
    };
    const auto compute_c128_loc = [&](int64_t rid, int32_t position) {
      return static_cast<int32_t>(rid * ring_size_ + position % ring_size_);
    };

    auto plan_c = idx < num_c_ ? plan_c_[idx] : CompressPlan::invalid();
    if (!plan_c.is_invalid()) {
      if (plan_c.buffer_len > 0) {
        const auto batch_id = plan_c.read_page_1;
        const auto rid = rid_ptr_[batch_id];
        const auto position_1 = static_cast<int32_t>(plan_c.seq_len - 1);
        const auto position_0 = sycl::max(position_1 - compress_ratio_, 0);

        if (compress_ratio_ == 128) {
          plan_c.read_page_0 = compute_c128_loc(rid, position_0) / 128;
          plan_c.read_page_1 = compute_c128_loc(rid, position_1) / 128;
        } else {
          const auto mapping = r2t_ptr_ + rid * stride_r2t_;
          const auto raw_loc_1 = mapping[position_1];
          const auto raw_loc_0 = mapping[position_0];
          const auto state_loc_1 = f2s_ptr_[raw_loc_1];
          const auto state_loc_0 = f2s_ptr_[raw_loc_0];
          plan_c.read_page_0 = compute_loc(state_loc_0) / compress_ratio_;
          plan_c.read_page_1 = compute_loc(state_loc_1) / compress_ratio_;
        }
        plan_c_[idx] = plan_c;
      }
    } else if (idx < num_c_padded_) {
      plan_c_[idx] = CompressPlan::invalid();
    }

    auto plan_w = idx < num_w_ ? plan_w_[idx] : WritePlan::invalid();
    if (!plan_w.is_invalid()) {
      const auto [ragged_id, batch_id] = unpack_w(plan_w);
      const auto rid = rid_ptr_[batch_id];
      const auto position = static_cast<int32_t>(plan_w.write_loc - 1);

      if (compress_ratio_ == 128) {
        plan_w.write_loc = compute_c128_loc(rid, position);
      } else {
        const auto mapping = r2t_ptr_ + rid * stride_r2t_;
        const auto raw_loc = mapping[position];
        plan_w.write_loc = compute_loc(f2s_ptr_[raw_loc]);
      }

      plan_w.ragged_id = ragged_id;
      plan_w_[idx] = plan_w;
    } else if (idx < num_w_padded_) {
      plan_w_[idx] = WritePlan::invalid();
    }
  }

  CompressPlan* plan_c_;
  WritePlan* plan_w_;
  const int64_t* rid_ptr_;
  const int32_t* r2t_ptr_;
  const int64_t* f2s_ptr_;
  int32_t swa_page_size_;
  int32_t ring_size_;
  int32_t compress_ratio_;
  int64_t stride_r2t_;
  uint32_t num_c_;
  uint32_t num_w_;
  uint32_t num_c_padded_;
  uint32_t num_w_padded_;
  uint32_t num_work_;
};

}  // namespace CompressPlanImpl

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
        torch::empty({0, static_cast<int64_t>(sizeof(CompressPlan))}, options_u8),
        torch::empty({0, static_cast<int64_t>(sizeof(WritePlan))}, options_u8),
    };
  }

  auto plan_c = torch::empty({num_q_tokens_u32, static_cast<int64_t>(sizeof(CompressPlan))}, options_u8);
  auto plan_w = torch::empty({num_q_tokens_u32, static_cast<int64_t>(sizeof(WritePlan))}, options_u8);
  constexpr int32_t kMaxMTPDraftTokens = 4;
  const int32_t mtp_pad = std::min(ring_size_i32 - compress_ratio_i32, kMaxMTPDraftTokens);

  auto queue = c10::xpu::getCurrentXPUStream().queue();
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint32_t, 1> counter_c_local(sycl::range<1>(1), cgh);
    sycl::local_accessor<uint32_t, 1> counter_w_local(sycl::range<1>(1), cgh);
    sycl::local_accessor<int32_t, 1> s_seq_len(sycl::range<1>(kMaxPrefillBatchSize), cgh);
    sycl::local_accessor<int32_t, 1> s_prefix_len(sycl::range<1>(kMaxPrefillBatchSize), cgh);
    sycl::local_accessor<uint32_t, 1> warp_max(sycl::range<1>(kStage0NumSubGroups), cgh);
    sycl::local_accessor<uint32_t, 1> warp_min(sycl::range<1>(kStage0NumSubGroups), cgh);
    sycl::local_accessor<uint32_t, 1> s_max_extend(sycl::range<1>(1), cgh);
    sycl::local_accessor<uint32_t, 1> s_min_extend(sycl::range<1>(1), cgh);

    CompressPlanImpl::CompressPrefillStage0Kernel kernel{
        reinterpret_cast<CompressPlan*>(plan_c.data_ptr<uint8_t>()),
        reinterpret_cast<WritePlan*>(plan_w.data_ptr<uint8_t>()),
        seq_lens_xpu.data_ptr<int64_t>(),
        extend_lens_xpu.data_ptr<int64_t>(),
        batch_size,
        num_q_tokens_u32,
        compress_ratio_i32,
        swa_page_size_i32,
        mtp_pad,
        counter_c_local,
        counter_w_local,
        s_seq_len,
        s_prefix_len,
        warp_max,
        warp_min,
        s_max_extend,
        s_min_extend};
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(kStage0BlockSize), sycl::range<1>(kStage0BlockSize)), kernel);
  });

  int64_t stride_r2t = req_to_token.stride(0);
  const uint32_t num_c = num_q_tokens_u32;
  const uint32_t num_w = num_q_tokens_u32;
  const uint32_t num_c_padded = num_q_tokens_u32;
  const uint32_t num_w_padded = num_q_tokens_u32;
  const uint32_t num_work = std::max(num_c_padded, num_w_padded);
  if (num_q_tokens_u32 > 0) {
    queue.submit([&](sycl::handler& cgh) {
      CompressPlanImpl::CompressPrefillStage1Kernel kernel{
          reinterpret_cast<CompressPlan*>(plan_c.data_ptr<uint8_t>()),
          reinterpret_cast<WritePlan*>(plan_w.data_ptr<uint8_t>()),
          req_pool_indices.data_ptr<int64_t>(),
          req_to_token.data_ptr<int32_t>(),
          full_to_state.data_ptr<int64_t>(),
          swa_page_size_i32,
          ring_size_i32,
          compress_ratio_i32,
          stride_r2t,
          num_c,
          num_w,
          num_c_padded,
          num_w_padded,
          num_work};
      cgh.parallel_for(get_1d_range(static_cast<int32_t>(num_work)), kernel);
    });
  }

  return {plan_c, plan_w};
}

}  // namespace at::native::xpu
