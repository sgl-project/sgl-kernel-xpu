#include <ATen/ATen.h>

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

#include "Utils.h"
#include "comm/General.h"
#include "sgl_kernel_export.h"

namespace sgl {
namespace sycl_kernel {

static constexpr int kWarpSize = 32;
static constexpr int kCTASize = 512;
static constexpr int kNumWarps = kCTASize / kWarpSize;

static constexpr int kMaxTopK = 32;
static constexpr int kMaxNumBlocks = 4096;

static constexpr int kRadixBits = 8;
// kCTASize must stay > kRadixSize; the radix passes init the counters from thread kRadixSize.
static constexpr int kRadixSize = 1 << kRadixBits;
static constexpr int kSmallThreshold = 8 * kNumWarps;
// Must stay <= 32: topk_radix_regM tracks per-thread liveness in a uint32_t bitmask.
static constexpr int kItersRegM = kMaxNumBlocks / kCTASize;

static constexpr float kNegInf = -std::numeric_limits<float>::infinity();

static constexpr int32_t kInvalidBlockId = -1;

struct TopKSmem {
  uint32_t warp_sum[kNumWarps];
  alignas(64) uint32_t counter;
  alignas(64) uint32_t counter_final;
  alignas(64) uint32_t threshold_bin;
  uint32_t equal_count;
  uint32_t above_count;
  alignas(64) uint32_t histogram[2][kRadixSize];
  float small_scores[kSmallThreshold];
};

struct PageTableSmem {
  TopKSmem base;
  alignas(64) int32_t s_topk[kMaxTopK];
  alignas(64) int32_t s_sorted[kMaxTopK];
  alignas(64) int32_t s_eff_kv;
};

// Order-preserving float -> uint32 map so the radix passes can bucket on raw bits.
inline uint32_t score_to_key(float x) {
  uint32_t b = ::sycl::bit_cast<uint32_t>(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

inline float clip_nan(float x) {
  return ::sycl::isnan(x) ? kNegInf : x;
}

inline uint32_t atomic_add_local_u32(uint32_t* addr, uint32_t v) {
  ::sycl::atomic_ref<
      uint32_t,
      ::sycl::memory_order::relaxed,
      ::sycl::memory_scope::work_group,
      ::sycl::access::address_space::local_space>
      ref(*addr);
  return ref.fetch_add(v);
}

inline int32_t atomic_add_local_i32(int32_t* addr, int32_t v) {
  ::sycl::atomic_ref<
      int32_t,
      ::sycl::memory_order::relaxed,
      ::sycl::memory_scope::work_group,
      ::sycl::access::address_space::local_space>
      ref(*addr);
  return ref.fetch_add(v);
}

inline void find_threshold(
    const ::sycl::nd_item<1>& item,
    const ::sycl::sub_group& sg,
    TopKSmem* smem,
    uint32_t* histogram,
    int tx,
    int warp_id,
    int lane_id,
    uint32_t total_active,
    uint32_t topk_remain) {
  uint32_t hist_val = 0;
  uint32_t warp_inc = 0;
  if (tx < kRadixSize) {
    hist_val = histogram[tx];
    warp_inc = ::sycl::inclusive_scan_over_group(sg, hist_val, ::sycl::plus<uint32_t>());
    if (lane_id == kWarpSize - 1) {
      smem->warp_sum[warp_id] = warp_inc;
    }
  }
  item.barrier(::sycl::access::fence_space::local_space);

  if (tx < kRadixSize) {
    const uint32_t masked = (lane_id < warp_id) ? smem->warp_sum[lane_id] : 0u;
    const uint32_t inter = ::sycl::reduce_over_group(sg, masked, ::sycl::plus<uint32_t>());
    const uint32_t prefix = inter + warp_inc;
    const uint32_t above = total_active - prefix;
    if (above < topk_remain && above + hist_val >= topk_remain) {
      smem->threshold_bin = tx;
      smem->above_count = above;
      smem->equal_count = hist_val;
    }
  }
  item.barrier(::sycl::access::fence_space::local_space);
}

inline void topk_small(
    const ::sycl::nd_item<1>& item,
    const ::sycl::sub_group& sg,
    const float* scores,
    uint32_t num_blocks,
    int32_t* topk_out,
    uint32_t topk,
    TopKSmem* smem,
    int tx,
    int warp_id,
    int lane_id) {
  if (tx < static_cast<int>(num_blocks)) {
    smem->small_scores[tx] = clip_nan(scores[tx]);
  }

  item.barrier(::sycl::access::fence_space::local_space);

  constexpr int kNumCandidates = kSmallThreshold / kNumWarps;
  constexpr int kNumTargets = kSmallThreshold / kWarpSize;
  float candidates[kNumCandidates];
  float target[kNumTargets];

#pragma unroll
  for (int i = 0; i < kNumTargets; ++i) {
    const int idx = lane_id + i * kWarpSize;
    target[i] = (idx < static_cast<int>(num_blocks)) ? smem->small_scores[idx] : kNegInf;
  }
#pragma unroll
  for (int i = 0; i < kNumCandidates; ++i) {
    const int idx = warp_id + i * kNumWarps;
    candidates[i] = (idx < static_cast<int>(num_blocks)) ? smem->small_scores[idx] : kNegInf;
  }

#pragma unroll
  for (int i = 0; i < kNumCandidates; ++i) {
    const int idx = warp_id + i * kNumWarps;
    if (idx >= static_cast<int>(num_blocks)) break;
    uint32_t rank = 0;
#pragma unroll
    for (int j = 0; j < kNumTargets; ++j) {
      const int delta = lane_id + j * kWarpSize - idx;
      const bool outranks = (target[j] > candidates[i]) || ((target[j] == candidates[i]) && (delta < 0));
      rank += outranks ? 1u : 0u;
    }
    rank = ::sycl::reduce_over_group(sg, rank, ::sycl::plus<uint32_t>());
    if (rank < topk) {
      topk_out[rank] = idx;
    }
  }
}

inline void topk_radix_reg1(
    const ::sycl::nd_item<1>& item,
    const ::sycl::sub_group& sg,
    const float* scores,
    uint32_t num_blocks,
    int32_t* topk_out,
    uint32_t topk,
    TopKSmem* smem,
    int tx,
    int warp_id,
    int lane_id) {
  bool active = tx < static_cast<int>(num_blocks);
  const float value = active ? clip_nan(scores[tx]) : kNegInf;
  const uint32_t key = score_to_key(value);
  uint32_t topk_remain = topk;
  uint32_t write_pos = topk;  // sentinel: not selected

  if (tx < kRadixSize) smem->histogram[0][tx] = 0;
  if (tx == kRadixSize) {
    smem->counter = 0;
    smem->counter_final = 0;
  }
  item.barrier(::sycl::access::fence_space::local_space);

  uint32_t total_active = num_blocks;

  // Left unrolled by hand: mixing break with #pragma unroll can defeat the
  // unroller on some SYCL toolchains.
  for (int round = 0; round < 4; ++round) {
    const uint32_t shift = 24u - static_cast<uint32_t>(round) * 8u;
    const uint32_t bin = (key >> shift) & 0xFFu;
    const int hist_idx = round & 1;
    uint32_t* histogram = smem->histogram[hist_idx];

    if (active) {
      atomic_add_local_u32(&histogram[bin], 1u);
    }
    if (round < 3 && tx < kRadixSize) {
      smem->histogram[hist_idx ^ 1][tx] = 0;
    }
    item.barrier(::sycl::access::fence_space::local_space);

    find_threshold(item, sg, smem, histogram, tx, warp_id, lane_id, total_active, topk_remain);

    const uint32_t threshold_bin = smem->threshold_bin;
    const uint32_t above_count = smem->above_count;
    const uint32_t equal_count = smem->equal_count;

    if (round < 3) total_active = equal_count;
    topk_remain -= above_count;

    if (active) {
      if (bin > threshold_bin) {
        write_pos = atomic_add_local_u32(&smem->counter, 1u);
        active = false;
      } else if (bin < threshold_bin) {
        active = false;
      } else if (round == 3) {
        write_pos = topk - topk_remain + atomic_add_local_u32(&smem->counter_final, 1u);
      }
    }

    if (round == 3 || topk_remain == 0) break;
  }

  if (write_pos < topk) {
    topk_out[write_pos] = tx;
  }
}

inline void topk_radix_regM(
    const ::sycl::nd_item<1>& item,
    const ::sycl::sub_group& sg,
    const float* scores,
    uint32_t num_blocks,
    int32_t* topk_out,
    uint32_t topk,
    TopKSmem* smem,
    int tx,
    int warp_id,
    int lane_id) {
  uint32_t key[kItersRegM];
  uint32_t active = 0u;
#pragma unroll
  for (int i = 0; i < kItersRegM; ++i) {
    const int idx = i * kCTASize + tx;
    if (idx < static_cast<int>(num_blocks)) {
      key[i] = score_to_key(clip_nan(scores[idx]));
      active |= 1u << i;
    }
  }

  if (tx < kRadixSize) smem->histogram[0][tx] = 0;
  if (tx == kRadixSize) {
    smem->counter = 0;
    smem->counter_final = 0;
  }
  item.barrier(::sycl::access::fence_space::local_space);

  uint32_t topk_remain = topk;
  uint32_t total_active = num_blocks;

  for (int round = 0; round < 4; ++round) {
    const uint32_t shift = 24u - static_cast<uint32_t>(round) * 8u;
    const int hb = round & 1;

#pragma unroll
    for (int i = 0; i < kItersRegM; ++i) {
      if (active & (1u << i)) {
        atomic_add_local_u32(&smem->histogram[hb][(key[i] >> shift) & 0xFFu], 1u);
      }
    }
    if (round < 3 && tx < kRadixSize) {
      smem->histogram[hb ^ 1][tx] = 0;
    }
    item.barrier(::sycl::access::fence_space::local_space);

    find_threshold(item, sg, smem, smem->histogram[hb], tx, warp_id, lane_id, total_active, topk_remain);
    const uint32_t threshold_bin = smem->threshold_bin;
    const uint32_t above_count = smem->above_count;
    const uint32_t equal_count = smem->equal_count;

    if (round < 3) total_active = equal_count;
    topk_remain -= above_count;

#pragma unroll
    for (int i = 0; i < kItersRegM; ++i) {
      if (active & (1u << i)) {
        const uint32_t bin = (key[i] >> shift) & 0xFFu;
        if (bin > threshold_bin) {
          const uint32_t pos = atomic_add_local_u32(&smem->counter, 1u);
          topk_out[pos] = i * kCTASize + tx;
          active &= ~(1u << i);
        } else if (bin < threshold_bin) {
          active &= ~(1u << i);
        } else if (round == 3) {
          const uint32_t pos = topk - topk_remain + atomic_add_local_u32(&smem->counter_final, 1u);
          if (pos < topk) topk_out[pos] = i * kCTASize + tx;
        }
      }
    }

    if (round == 3 || topk_remain == 0) break;
  }
}

// The radix regimes take write positions from an atomic counter, so among equal scores
// the selected set is reproducible but its order is not. Only the small regime is stable.
inline void topk_forward(
    const ::sycl::nd_item<1>& item,
    const ::sycl::sub_group& sg,
    const float* scores,
    uint32_t num_blocks,
    int32_t* topk_out,
    uint32_t topk,
    TopKSmem* smem,
    int tx,
    int warp_id,
    int lane_id) {
  if (num_blocks <= static_cast<uint32_t>(kSmallThreshold)) {
    topk_small(item, sg, scores, num_blocks, topk_out, topk, smem, tx, warp_id, lane_id);
  } else if (num_blocks <= static_cast<uint32_t>(kCTASize)) {
    topk_radix_reg1(item, sg, scores, num_blocks, topk_out, topk, smem, tx, warp_id, lane_id);
  } else {
    topk_radix_regM(item, sg, scores, num_blocks, topk_out, topk, smem, tx, warp_id, lane_id);
  }
}

template <typename SeqLenT>
class MinimaxDecodeTopKBlockKernel {
 public:
  MinimaxDecodeTopKBlockKernel(
      const float* score,
      const SeqLenT* seq_lens,
      int32_t* topk_idx,
      int32_t batch,
      int32_t num_heads,
      int32_t max_seqblock,
      int32_t block_size,
      int32_t topk,
      ::sycl::local_accessor<TopKSmem, 1> smem)
      : score_(score),
        seq_lens_(seq_lens),
        topk_idx_(topk_idx),
        batch_(batch),
        num_heads_(num_heads),
        max_seqblock_(max_seqblock),
        block_size_(block_size),
        topk_(topk),
        smem_(smem) {}

  [[sycl::reqd_sub_group_size(kWarpSize)]] void operator()(::sycl::nd_item<1> item) const {
    const int group = static_cast<int>(item.get_group(0));
    const int b = group / num_heads_;
    const int h = group - b * num_heads_;

    const ::sycl::sub_group sg = item.get_sub_group();
    const int tx = static_cast<int>(item.get_local_id(0));
    const int warp_id = static_cast<int>(sg.get_group_linear_id());
    const int lane_id = static_cast<int>(sg.get_local_linear_id());

    const int64_t seq_len = static_cast<int64_t>(seq_lens_[b]);
    const int num_blocks_raw = static_cast<int>((seq_len + block_size_ - 1) / block_size_);
    const int num_blocks = num_blocks_raw < max_seqblock_ ? num_blocks_raw : max_seqblock_;

    int32_t* out = topk_idx_ + (static_cast<int64_t>(h) * batch_ + b) * topk_;

    if (num_blocks <= topk_) {
      for (int i = tx; i < topk_; i += kCTASize) {
        out[i] = (i < num_blocks) ? static_cast<int32_t>(i) : kInvalidBlockId;
      }
      return;
    }

    for (int i = tx; i < topk_; i += kCTASize) {
      out[i] = kInvalidBlockId;
    }
    item.barrier();

    TopKSmem* smem_typed = &smem_[0];

    const float* row = score_ + (static_cast<int64_t>(h) * batch_ + b) * static_cast<int64_t>(max_seqblock_);
    topk_forward(
        item,
        sg,
        row,
        static_cast<uint32_t>(num_blocks),
        out,
        static_cast<uint32_t>(topk_),
        smem_typed,
        tx,
        warp_id,
        lane_id);
  }

 private:
  const float* score_;
  const SeqLenT* seq_lens_;
  int32_t* topk_idx_;
  int32_t batch_;
  int32_t num_heads_;
  int32_t max_seqblock_;
  int32_t block_size_;
  int32_t topk_;
  ::sycl::local_accessor<TopKSmem, 1> smem_;
};

template <typename SeqLenT>
void minimax_decode_topk_launcher(
    ::sycl::queue& queue,
    const void* score,
    const void* seq_lens,
    void* topk_idx,
    int32_t batch,
    int32_t num_heads,
    int32_t max_seqblock,
    int32_t block_size,
    int32_t topk) {
  if (batch == 0 || num_heads == 0) return;

  const size_t num_groups = static_cast<size_t>(batch) * num_heads;
  queue.submit([&](::sycl::handler& cgh) {
    ::sycl::local_accessor<TopKSmem, 1> smem(::sycl::range<1>(1), cgh);
    cgh.parallel_for(
        ::sycl::nd_range<1>(::sycl::range<1>(num_groups * kCTASize), ::sycl::range<1>(kCTASize)),
        MinimaxDecodeTopKBlockKernel<SeqLenT>(
            static_cast<const float*>(score),
            static_cast<const SeqLenT*>(seq_lens),
            static_cast<int32_t*>(topk_idx),
            batch,
            num_heads,
            max_seqblock,
            block_size,
            topk,
            smem));
  });
  // Do NOT .wait(): this runs on the current XPU stream and PyTorch owns synchronization.
}

template <typename SeqLenT>
class MinimaxDecodeTopKPageTableKernel {
 public:
  MinimaxDecodeTopKPageTableKernel(
      const float* score,
      const SeqLenT* seq_lens,
      const int32_t* req_to_token,
      const int64_t* slot_ids,
      int32_t* page_table,
      int32_t* seq_lens_out,
      int32_t batch,
      int32_t num_heads,
      int32_t max_seqblock,
      int32_t block_size,
      int32_t topk,
      int32_t page_size,
      int32_t r2t_stride,
      int32_t max_kv_len,
      int32_t max_reqs,
      int32_t max_sparse_pages,
      ::sycl::local_accessor<PageTableSmem, 1> smem)
      : score_(score),
        seq_lens_(seq_lens),
        req_to_token_(req_to_token),
        slot_ids_(slot_ids),
        page_table_(page_table),
        seq_lens_out_(seq_lens_out),
        batch_(batch),
        num_heads_(num_heads),
        max_seqblock_(max_seqblock),
        block_size_(block_size),
        topk_(topk),
        page_size_(page_size),
        r2t_stride_(r2t_stride),
        max_kv_len_(max_kv_len),
        max_reqs_(max_reqs),
        max_sparse_pages_(max_sparse_pages),
        smem_(smem) {}

  [[sycl::reqd_sub_group_size(kWarpSize)]] void operator()(::sycl::nd_item<1> item) const {
    const int group = static_cast<int>(item.get_group(0));
    const int b = group / num_heads_;
    const int h = group - b * num_heads_;

    const ::sycl::sub_group sg = item.get_sub_group();
    const int tx = static_cast<int>(item.get_local_id(0));
    const int warp_id = static_cast<int>(sg.get_group_linear_id());
    const int lane_id = static_cast<int>(sg.get_local_linear_id());

    const int64_t seq_len = static_cast<int64_t>(seq_lens_[b]);
    const int num_blocks_raw = static_cast<int>((seq_len + block_size_ - 1) / block_size_);
    const int num_blocks = num_blocks_raw < max_seqblock_ ? num_blocks_raw : max_seqblock_;
    const int ppb = block_size_ / page_size_;

    const int64_t out_row = static_cast<int64_t>(b) * num_heads_ + h;
    int32_t* pt_row = page_table_ + out_row * max_sparse_pages_;
    // slot_ids may fall outside [0, max_reqs); % keeps the operand's sign, hence the fixup.
    const int64_t slot = static_cast<int64_t>(slot_ids_[b]) % max_reqs_;
    const int64_t r2t_base = (slot < 0 ? slot + max_reqs_ : slot) * r2t_stride_;

    if (num_blocks <= topk_) {
      if (tx == 0) {
        seq_lens_out_[out_row] = static_cast<int32_t>(seq_len);
      }
      const int total = num_blocks * ppb;
      for (int e = tx; e < total; e += kCTASize) {
        const int slot = e / ppb;
        const int pp = e % ppb;
        int tok = slot * block_size_ + pp * page_size_;
        if (tok >= max_kv_len_) tok = max_kv_len_ - 1;
        pt_row[e] = req_to_token_[r2t_base + tok] / page_size_ * num_heads_ + h;
      }
      return;
    }

    PageTableSmem* smem_typed = &smem_[0];

    const int k_eff = topk_;
    const float* row = score_ + (static_cast<int64_t>(h) * batch_ + b) * static_cast<int64_t>(max_seqblock_);

    if (tx < k_eff) {
      smem_typed->s_topk[tx] = 0;
    }
    if (tx == 0) {
      smem_typed->s_eff_kv = 0;
    }
    item.barrier(::sycl::access::fence_space::local_space);

    topk_forward(
        item,
        sg,
        row,
        static_cast<uint32_t>(num_blocks),
        smem_typed->s_topk,
        static_cast<uint32_t>(topk_),
        &smem_typed->base,
        tx,
        warp_id,
        lane_id);
    item.barrier(::sycl::access::fence_space::local_space);

    // Rank-by-compare is quadratic, but k_eff <= kMaxTopK (32) beats a sort network here.
    for (int slot = tx; slot < k_eff; slot += kCTASize) {
      const int32_t v = smem_typed->s_topk[slot];
      int rank = 0;
      for (int j = 0; j < k_eff; ++j) {
        if (smem_typed->s_topk[j] < v) ++rank;
      }
      smem_typed->s_sorted[rank] = v;
      const int rem = static_cast<int>(seq_len - static_cast<int64_t>(v) * block_size_);
      const int contrib = rem < block_size_ ? rem : block_size_;
      atomic_add_local_i32(&smem_typed->s_eff_kv, contrib);
    }
    item.barrier(::sycl::access::fence_space::local_space);

    if (tx == 0) {
      seq_lens_out_[out_row] = smem_typed->s_eff_kv;
    }

    const int total = k_eff * ppb;
    for (int e = tx; e < total; e += kCTASize) {
      const int slot = e / ppb;
      const int pp = e % ppb;
      int tok = smem_typed->s_sorted[slot] * block_size_ + pp * page_size_;
      if (tok >= max_kv_len_) tok = max_kv_len_ - 1;
      pt_row[e] = req_to_token_[r2t_base + tok] / page_size_ * num_heads_ + h;
    }
  }

 private:
  const float* score_;
  const SeqLenT* seq_lens_;
  const int32_t* req_to_token_;
  const int64_t* slot_ids_;
  int32_t* page_table_;
  int32_t* seq_lens_out_;
  int32_t batch_;
  int32_t num_heads_;
  int32_t max_seqblock_;
  int32_t block_size_;
  int32_t topk_;
  int32_t page_size_;
  int32_t r2t_stride_;
  int32_t max_kv_len_;
  int32_t max_reqs_;
  int32_t max_sparse_pages_;
  ::sycl::local_accessor<PageTableSmem, 1> smem_;
};

template <typename SeqLenT>
void minimax_decode_topk_page_table_launcher(
    ::sycl::queue& queue,
    const void* score,
    const void* seq_lens,
    const void* req_to_token,
    const void* slot_ids,
    void* page_table,
    void* seq_lens_out,
    int32_t batch,
    int32_t num_heads,
    int32_t max_seqblock,
    int32_t block_size,
    int32_t topk,
    int32_t page_size,
    int32_t r2t_stride,
    int32_t max_kv_len,
    int32_t max_reqs,
    int32_t max_sparse_pages) {
  if (batch == 0 || num_heads == 0 || max_reqs == 0) return;

  const size_t num_groups = static_cast<size_t>(batch) * num_heads;
  queue.submit([&](::sycl::handler& cgh) {
    ::sycl::local_accessor<PageTableSmem, 1> smem(::sycl::range<1>(1), cgh);
    cgh.parallel_for(
        ::sycl::nd_range<1>(::sycl::range<1>(num_groups * kCTASize), ::sycl::range<1>(kCTASize)),
        MinimaxDecodeTopKPageTableKernel<SeqLenT>(
            static_cast<const float*>(score),
            static_cast<const SeqLenT*>(seq_lens),
            static_cast<const int32_t*>(req_to_token),
            static_cast<const int64_t*>(slot_ids),
            static_cast<int32_t*>(page_table),
            static_cast<int32_t*>(seq_lens_out),
            batch,
            num_heads,
            max_seqblock,
            block_size,
            topk,
            page_size,
            r2t_stride,
            max_kv_len,
            max_reqs,
            max_sparse_pages,
            smem));
  });
  // Do NOT .wait(): this runs on the current XPU stream and PyTorch owns synchronization.
}

}  // namespace sycl_kernel
}  // namespace sgl

namespace mmtopk = sgl::sycl_kernel;

namespace {

// Both widths are instantiated because callers pass whichever their scheduler produced.
enum class SeqLenKind { kI32, kI64 };

SeqLenKind seq_len_kind(const at::Tensor& seq_lens) {
  TORCH_CHECK(
      seq_lens.scalar_type() == at::kInt || seq_lens.scalar_type() == at::kLong,
      "seq_lens must be int32 or int64, got ",
      seq_lens.scalar_type());
  return seq_lens.scalar_type() == at::kInt ? SeqLenKind::kI32 : SeqLenKind::kI64;
}

void check_score_and_seq_lens(const at::Tensor& score, const at::Tensor& seq_lens, int64_t block_size, int64_t topk) {
  TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be float32, got ", score.scalar_type());
  TORCH_CHECK(score.dim() == 3, "score must be 3-D, got ", score.dim(), "-D");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1-D, got ", seq_lens.dim(), "-D");
  TORCH_CHECK(score.device().is_xpu(), "score must be on XPU, got ", score.device());
  TORCH_CHECK(
      seq_lens.device() == score.device(),
      "score and seq_lens must be on the same device, got ",
      score.device(),
      " vs ",
      seq_lens.device());

  TORCH_CHECK(block_size >= 1, "block_size must be >= 1, got ", block_size);
  // topk < 1 would enter the radix path with topk_remain == 0, leaving
  // threshold_bin uninitialized in find_threshold.
  TORCH_CHECK(topk >= 1, "topk must be >= 1, got ", topk);
  TORCH_CHECK(topk <= mmtopk::kMaxTopK, "topk (", topk, ") exceeds kMaxTopK (", mmtopk::kMaxTopK, ")");

  const int64_t batch = score.size(1);
  const int64_t max_seqblock = score.size(2);
  TORCH_CHECK(seq_lens.numel() == batch, "seq_lens length (", seq_lens.numel(), ") must match batch (", batch, ")");
  TORCH_CHECK(
      max_seqblock <= mmtopk::kMaxNumBlocks,
      "max_seqblock (",
      max_seqblock,
      ") exceeds kMaxNumBlocks (",
      mmtopk::kMaxNumBlocks,
      "); increase kMaxNumBlocks above if needed");
}

}  // namespace

SGL_KERNEL_EXPORT void minimax_decode_topk(
    const at::Tensor& score, const at::Tensor& seq_lens, const at::Tensor& out, int64_t block_size, int64_t topk) {
  check_score_and_seq_lens(score, seq_lens, block_size, topk);

  const int64_t num_heads = score.size(0);
  const int64_t batch = score.size(1);
  const int64_t max_seqblock = score.size(2);

  TORCH_CHECK(out.scalar_type() == at::kInt, "out must be int32, got ", out.scalar_type());
  TORCH_CHECK(
      out.dim() == 3 && out.size(0) == num_heads && out.size(1) == batch && out.size(2) == topk,
      "out shape must be (",
      num_heads,
      ", ",
      batch,
      ", ",
      topk,
      "), got ",
      out.sizes());
  TORCH_CHECK(out.device() == score.device(), "out device (", out.device(), ") must match score device");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

  const at::Tensor score_c = score.contiguous();
  const at::Tensor seq_lens_c = seq_lens.contiguous();

  auto& queue = dpcppGetCurrentQueue();
  const auto b = static_cast<int32_t>(batch);
  const auto h = static_cast<int32_t>(num_heads);
  const auto s = static_cast<int32_t>(max_seqblock);
  const auto bs = static_cast<int32_t>(block_size);
  const auto k = static_cast<int32_t>(topk);

  switch (seq_len_kind(seq_lens_c)) {
    case SeqLenKind::kI32:
      mmtopk::minimax_decode_topk_launcher<int32_t>(
          queue, score_c.const_data_ptr(), seq_lens_c.const_data_ptr(), out.data_ptr(), b, h, s, bs, k);
      break;
    case SeqLenKind::kI64:
      mmtopk::minimax_decode_topk_launcher<int64_t>(
          queue, score_c.const_data_ptr(), seq_lens_c.const_data_ptr(), out.data_ptr(), b, h, s, bs, k);
      break;
  }
}

SGL_KERNEL_EXPORT std::tuple<at::Tensor, at::Tensor> minimax_decode_topk_page_table(
    const at::Tensor& score,
    const at::Tensor& seq_lens,
    const at::Tensor& req_to_token,
    const at::Tensor& slot_ids,
    int64_t block_size,
    int64_t topk,
    int64_t page_size) {
  check_score_and_seq_lens(score, seq_lens, block_size, topk);

  TORCH_CHECK(req_to_token.scalar_type() == at::kInt, "req_to_token must be int32, got ", req_to_token.scalar_type());
  TORCH_CHECK(slot_ids.scalar_type() == at::kLong, "slot_ids must be int64, got ", slot_ids.scalar_type());
  TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be 2-D, got ", req_to_token.dim(), "-D");
  TORCH_CHECK(slot_ids.dim() == 1, "slot_ids must be 1-D, got ", slot_ids.dim(), "-D");
  TORCH_CHECK(req_to_token.device() == score.device(), "score and req_to_token must be on the same device");
  TORCH_CHECK(slot_ids.device() == score.device(), "score and slot_ids must be on the same device");

  TORCH_CHECK(page_size >= 1, "page_size must be >= 1, got ", page_size);
  TORCH_CHECK(
      block_size % page_size == 0, "block_size (", block_size, ") must be a multiple of page_size (", page_size, ")");

  const int64_t num_heads = score.size(0);
  const int64_t batch = score.size(1);
  const int64_t max_seqblock = score.size(2);
  TORCH_CHECK(slot_ids.numel() == batch, "slot_ids length (", slot_ids.numel(), ") must match batch (", batch, ")");

  // The kernel addresses req_to_token flat as r2t_base + tok, so only the inner stride
  // must be 1; row-pitched slices of a larger pool are common, so don't demand contiguity.
  TORCH_CHECK(
      req_to_token.stride(1) == 1, "req_to_token must have unit inner stride, got strides ", req_to_token.strides());

  const at::Tensor score_c = score.contiguous();
  const at::Tensor seq_lens_c = seq_lens.contiguous();
  const at::Tensor slot_ids_c = slot_ids.contiguous();

  const int64_t ppb = block_size / page_size;
  const int64_t max_sparse_pages = topk * ppb;
  const int64_t max_reqs = req_to_token.size(0);
  const int64_t max_kv_len = req_to_token.size(1);
  const int64_t r2t_stride = req_to_token.stride(0);

  auto options = score.options().dtype(at::kInt);
  at::Tensor page_table = at::empty({batch * num_heads, max_sparse_pages}, options);
  at::Tensor real_seq_lens = at::empty({batch * num_heads}, options);

  auto& queue = dpcppGetCurrentQueue();
  const auto b = static_cast<int32_t>(batch);
  const auto h = static_cast<int32_t>(num_heads);
  const auto s = static_cast<int32_t>(max_seqblock);
  const auto bs = static_cast<int32_t>(block_size);
  const auto k = static_cast<int32_t>(topk);
  const auto ps = static_cast<int32_t>(page_size);
  const auto stride = static_cast<int32_t>(r2t_stride);
  const auto kv_len = static_cast<int32_t>(max_kv_len);
  const auto reqs = static_cast<int32_t>(max_reqs);
  const auto pages = static_cast<int32_t>(max_sparse_pages);

  switch (seq_len_kind(seq_lens_c)) {
    case SeqLenKind::kI32:
      mmtopk::minimax_decode_topk_page_table_launcher<int32_t>(
          queue,
          score_c.const_data_ptr(),
          seq_lens_c.const_data_ptr(),
          req_to_token.const_data_ptr(),
          slot_ids_c.const_data_ptr(),
          page_table.data_ptr(),
          real_seq_lens.data_ptr(),
          b,
          h,
          s,
          bs,
          k,
          ps,
          stride,
          kv_len,
          reqs,
          pages);
      break;
    case SeqLenKind::kI64:
      mmtopk::minimax_decode_topk_page_table_launcher<int64_t>(
          queue,
          score_c.const_data_ptr(),
          seq_lens_c.const_data_ptr(),
          req_to_token.const_data_ptr(),
          slot_ids_c.const_data_ptr(),
          page_table.data_ptr(),
          real_seq_lens.data_ptr(),
          b,
          h,
          s,
          bs,
          k,
          ps,
          stride,
          kv_len,
          reqs,
          pages);
      break;
  }
  return {page_table, real_seq_lens};
}
