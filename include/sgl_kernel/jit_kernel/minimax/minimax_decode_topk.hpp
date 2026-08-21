/**
 * MiniMax decode block top-k SYCL kernel for Intel XPU.
 *
 * Ports the CUDA implementation at
 * ``sglang/python/sglang/kernels/jit/csrc/minimax/minimax_decode_topk.cuh``
 * (used by the MiniMax M3 sparse decode indexer). Given a per-(head, batch)
 * row of block scores ``score[H, B, S]`` (fp32), select the ``topk`` highest-
 * scoring block ids. Two output modes:
 *
 *   1. ``MinimaxDecodeTopKBlockKernel`` -> block-id output
 *      ``topk_idx[H, B, T]`` (front-packed, ``-1`` padded). Used as a
 *      drop-in for the Triton 2-stage top-k fallback.
 *
 *   2. ``MinimaxDecodeTopKPageTableKernel`` -> paged page-table output
 *      consumed by the dense backend (trtllm_mha / fa3). Emits the paged
 *      addresses for each selected block plus the effective KV length.
 *      Head-encoded page indices for DP attention (num_heads > 1).
 *
 * Both kernels share the same 3-regime top-k selection (``topk_forward``):
 *   * ``num_blocks <= kSmallThreshold`` : O(n^2) rank-by-compare, no radix.
 *   * ``num_blocks <= kCTASize``        : 4-pass 8-bit radix, one element per
 *                                        thread in a register.
 *   * ``num_blocks <= kMaxNumBlocks``   : 4-pass 8-bit radix, ``kIters``
 *                                        elements per thread in registers,
 *                                        with a uint32_t liveness bitmask.
 *
 * Differences from the CUDA reference:
 *   * PDL (Programmatic Dependent Launch, Hopper-only) has no XPU equivalent
 *     and is dropped.
 *   * Tie-breaking among exactly-equal scores is unspecified in the two radix
 *     regimes: write positions come from an atomic counter, so the order among
 *     equal keys depends on arrival. The small regime does reproduce CUDA's
 *     "lower block id wins". The *set* of selected ids is deterministic either
 *     way; only the order within it (and which of several equal-scoring blocks
 *     lands in the last slot) can differ.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl_kernel {

// ============================================================================
// Compile-time constants (mirror CUDA TopKTrait)
// ============================================================================

// Sub-group / work-group layout: 32-wide sub-groups, 512-thread work-groups.
// Pinned so the 4-pass radix's (256-bucket histogram = 8 sub-groups * 32 lanes)
// and the register-M regime's (kIters = kMaxNumBlocks / kCTASize = 8) both
// match the CUDA warp layout the algorithm was written against.
static constexpr int kWarpSize = 32;
static constexpr int kCTASize = 512;
static constexpr int kNumWarps = kCTASize / kWarpSize;  // 16

// Top-k bounds.
static constexpr int kMaxTopK = 32;         // both kernels
static constexpr int kMaxNumBlocks = 4096;  // register-M regime cap

// Radix / small-regime constants.
static constexpr int kRadixBits = 8;
static constexpr int kRadixSize = 1 << kRadixBits;           // 256
static constexpr int kSmallThreshold = 8 * kNumWarps;        // 128
static constexpr int kItersRegM = kMaxNumBlocks / kCTASize;  // 8, fits in uint32_t bitmask

static constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// Sentinel matching the CUDA output contract:
//   topk_out[k_eff..topk) = -1 (block-id kernel only)
static constexpr int32_t kInvalidBlockId = -1;

// ============================================================================
// Shared-memory layouts
// ============================================================================

// Mirrors ``TopKTrait::Smem`` in the CUDA source. Allocated directly as a
// ``local_accessor<TopKSmem, 1>`` so the struct's own alignment is guaranteed
// (``atomic_ref`` on the counters needs it). ``alignas(64)`` on the counters
// keeps them on separate cache lines (Xe2 has 64-byte L1 lines; CUDA used 128,
// which was NVIDIA-specific).
struct TopKSmem {
  uint32_t warp_sum[kNumWarps];  // 64 B (16 warps)
  alignas(64) uint32_t counter;
  alignas(64) uint32_t counter_final;
  alignas(64) uint32_t threshold_bin;
  uint32_t equal_count;
  uint32_t above_count;
  alignas(64) uint32_t histogram[2][kRadixSize];  // 2 KB (double-buffered)
  float small_scores[kSmallThreshold];            // 512 B (small regime only)
};

// Extra scratch for the page-table kernel: the trait's shared-memory output
// buffer + ascending-sorted block ids + accumulated effective KV length.
struct PageTableSmem {
  TopKSmem base;
  alignas(64) int32_t s_topk[kMaxTopK];    // trait writes here
  alignas(64) int32_t s_sorted[kMaxTopK];  // ascending-sorted for page emit
  alignas(64) int32_t s_eff_kv;
};

// ============================================================================
// Device helpers (inline, header-only)
// ============================================================================

// Bit-reinterpret float -> radix-sortable uint32_t. Flips the sign bit so a
// higher float compares greater as a uint32_t; also flips the whole word for
// negatives (twos-complement of sign-magnitude fp32).
inline uint32_t score_to_key(float x) {
  uint32_t b = ::sycl::bit_cast<uint32_t>(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Replace NaN with -inf so NaN scores never win the top-k. Matches CUDA
// ``clip_nan``.
inline float clip_nan(float x) {
  return ::sycl::isnan(x) ? kNegInf : x;
}

// Local-memory atomic add returning the previous value. Matches CUDA
// ``atomicAdd(&smem_counter, 1)`` semantics on shared memory.
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

// Find the radix bin holding the ``topk_remain``-th largest of ``total_active``
// elements currently counted in ``histogram``. Writes ``threshold_bin``,
// ``above_count``, ``equal_count`` to ``smem`` for the scatter phase.
//
// Layout: 256 histogram buckets are covered by the first 8 sub-groups
// (kRadixSize / kWarpSize = 8), each doing a warp-inclusive prefix over 32
// buckets. Cross-warp reduction sums the per-warp totals for buckets < warp_id
// to get the strictly-above count.
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
    // Warp-inclusive prefix sum: replaces CUDA's manual __shfl_up_sync loop.
    warp_inc = ::sycl::inclusive_scan_over_group(sg, hist_val, ::sycl::plus<uint32_t>());
    if (lane_id == kWarpSize - 1) {
      smem->warp_sum[warp_id] = warp_inc;
    }
  }
  item.barrier(::sycl::access::fence_space::local_space);

  if (tx < kRadixSize) {
    // Sum warp totals for warps strictly below this one (bins 0..warp_id-1).
    // reduce_over_group across the 32 lanes gives the same cross-warp scan
    // the CUDA source used via warp::reduce_sum + masking.
    const uint32_t masked = (lane_id < warp_id) ? smem->warp_sum[lane_id] : 0u;
    const uint32_t inter = ::sycl::reduce_over_group(sg, masked, ::sycl::plus<uint32_t>());
    const uint32_t prefix = inter + warp_inc;      // count in bins [0, tx]
    const uint32_t above = total_active - prefix;  // count in bins > tx
    if (above < topk_remain && above + hist_val >= topk_remain) {
      smem->threshold_bin = tx;
      smem->above_count = above;
      smem->equal_count = hist_val;
    }
  }
  item.barrier(::sycl::access::fence_space::local_space);
}

// ---------------------------------------------------------------------------
// Small regime: O(n^2) rank-by-compare
// ---------------------------------------------------------------------------
// Each candidate's rank = number of scores that outrank it; if rank < topk,
// write to topk_out[rank]. Only reachable when num_blocks <= kSmallThreshold.
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
  // Barrier: all threads must see the small_scores writes before the rank-by-
  // compare loop below reads them. Matches CUDA __syncthreads() at
  // minimax_decode_topk.cuh:112.
  item.barrier(::sycl::access::fence_space::local_space);

  constexpr int kNumCandidates = kSmallThreshold / kNumWarps;  // 128/16 = 8
  constexpr int kNumTargets = kSmallThreshold / kWarpSize;     // 128/32 = 4
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
      // Tie-break: lower block id wins (matches CUDA is_greater lambda).
      const bool outranks = (target[j] > candidates[i]) || ((target[j] == candidates[i]) && (delta < 0));
      rank += outranks ? 1u : 0u;
    }
    // Sum per-lane partial ranks across the warp.
    rank = ::sycl::reduce_over_group(sg, rank, ::sycl::plus<uint32_t>());
    if (rank < topk) {
      topk_out[rank] = idx;
    }
  }
}

// ---------------------------------------------------------------------------
// Register-1 regime: 4-pass 8-bit radix, one element per thread
// ---------------------------------------------------------------------------
// One block-score per thread held in a register; passes atomically update the
// SMEM histogram, then scatter above/below/equal per the threshold bin.
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

  // 4 rounds, MSB -> LSB in 8-bit chunks. Manually unrolled: mixing break with
  // #pragma unroll can defeat the unroller on some SYCL toolchains.
  for (int round = 0; round < 4; ++round) {
    const uint32_t shift = 24u - static_cast<uint32_t>(round) * 8u;
    const uint32_t bin = (key >> shift) & 0xFFu;
    const int hist_idx = round & 1;
    uint32_t* histogram = smem->histogram[hist_idx];

    if (active) {
      atomic_add_local_u32(&histogram[bin], 1u);
    }
    // Zero the OTHER buffer so it's ready for the next round; the current one
    // is being written by the atomics above.
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

    // Scatter: above -> write now; equal at final round -> fill trailing tail;
    // below or equal at non-final round -> drop or stay live.
    if (active) {
      if (bin > threshold_bin) {
        write_pos = atomic_add_local_u32(&smem->counter, 1u);
        active = false;
      } else if (bin < threshold_bin) {
        active = false;
      } else if (round == 3) {
        write_pos = topk - topk_remain + atomic_add_local_u32(&smem->counter_final, 1u);
      }
      // bin == threshold_bin && round < 3: stay active for the next pass.
    }

    if (round == 3 || topk_remain == 0) break;
  }

  if (write_pos < topk) {
    topk_out[write_pos] = tx;
  }
}

// ---------------------------------------------------------------------------
// Register-M regime: 4-pass 8-bit radix, kItersRegM elements per thread
// ---------------------------------------------------------------------------
// Each thread caches up to kItersRegM keys in registers (row read from global
// exactly once). Liveness is a uint32_t bitmask (bit i = slot i still live);
// selection is an in-loop scatter, no per-element position array in SMEM.
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
        // bin == threshold_bin && round < 3: slot stays live for the next pass.
      }
    }

    if (round == 3 || topk_remain == 0) break;
  }
}

// ---------------------------------------------------------------------------
// Top-k dispatcher: pick the right regime based on num_blocks
// ---------------------------------------------------------------------------
// Writes ``topk_out[0..k_eff)`` = selected block ids (front-packed, unordered);
// caller is responsible for the ``k_eff..topk`` padding (either -1 for the
// block-id kernel or leaving it uninitialized for the page-table kernel which
// only reads k_eff slots).
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
    // topk_small issues its own barrier between the small_scores load and the
    // rank-by-compare loop. Rank-by-compare writes go directly to global
    // topk_out; no trailing barrier needed here (callers that need one after
    // topk_forward issue it themselves -- e.g. the page-table kernel does so
    // before reading s_topk for the ascending sort).
    topk_small(item, sg, scores, num_blocks, topk_out, topk, smem, tx, warp_id, lane_id);
  } else if (num_blocks <= static_cast<uint32_t>(kCTASize)) {
    topk_radix_reg1(item, sg, scores, num_blocks, topk_out, topk, smem, tx, warp_id, lane_id);
  } else {
    topk_radix_regM(item, sg, scores, num_blocks, topk_out, topk, smem, tx, warp_id, lane_id);
  }
}

// ============================================================================
// Kernel 1: block-id output
// ============================================================================
// One work-group (kCTASize threads) per (batch, head) row. Grid layout:
//   group_id = b * num_heads + h  (b-major, matches head-major score access)
//
// Trivial case ``num_blocks <= topk``: identity block ids, ``-1`` padded.
// Otherwise ``topk_forward`` selects the top-k; padding is written explicitly
// because the trait leaves ``[k_eff, topk)`` untouched.
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
    // Recover (b, h) from the flattened group id. b-major so consecutive
    // groups share seq_lens[b], matching the head-major score row indexing.
    const int b = group / num_heads_;
    const int h = group - b * num_heads_;

    const ::sycl::sub_group sg = item.get_sub_group();
    const int tx = static_cast<int>(item.get_local_id(0));
    const int warp_id = static_cast<int>(sg.get_group_linear_id());
    const int lane_id = static_cast<int>(sg.get_local_linear_id());

    const int64_t seq_len = static_cast<int64_t>(seq_lens_[b]);
    const int num_blocks_raw = static_cast<int>((seq_len + block_size_ - 1) / block_size_);
    // Never scan past the materialized score columns (cuda-graph static shape
    // can be larger than the live seq_len).
    const int num_blocks = num_blocks_raw < max_seqblock_ ? num_blocks_raw : max_seqblock_;

    int32_t* out = topk_idx_ + (static_cast<int64_t>(h) * batch_ + b) * topk_;

    if (num_blocks <= topk_) {
      // Trivial: identity block ids, -1 padded.
      for (int i = tx; i < topk_; i += kCTASize) {
        out[i] = (i < num_blocks) ? static_cast<int32_t>(i) : kInvalidBlockId;
      }
      return;
    }

    // Pad the [k_eff, topk) tail. k_eff = topk here (num_blocks > topk), so
    // there is nothing to pad -- but the trait only writes k_eff slots for
    // pathological num_blocks == topk + 1 with the equal-scan tail landing.
    // Initialize the whole row to -1 first so any un-written slot stays -1.
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

// ============================================================================
// Host launcher: block-id kernel
// ============================================================================

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
  // NOTE: do NOT call .wait() -- the kernel runs on the current XPU stream and
  // synchronization is handled on the PyTorch side (SKILL.md pitfall #2).
}

// ============================================================================
// Kernel 2: page-table output (fused top-k + page-table transform)
// ============================================================================
// After top-k selection, sort selected block ids ascending (so the final
// partial block's pages land last), accumulate the effective KV length, then
// emit ``k_eff * ppb`` pages via ``req_to_token``. Head-encoded page indices
// for DP attention (page = base_page * num_heads + h).
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
    // Wrap the slot into range like the Triton reference does: out-of-range or
    // negative slot_ids would otherwise index req_to_token out of bounds.
    const int64_t slot = static_cast<int64_t>(slot_ids_[b]) % max_reqs_;
    const int64_t r2t_base = (slot < 0 ? slot + max_reqs_ : slot) * r2t_stride_;

    if (num_blocks <= topk_) {
      // Trivial: every block selected in ascending order, all tokens valid.
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

    // Non-trivial: run top-k -> ascending sort -> effective KV -> page emit.
    PageTableSmem* smem_typed = &smem_[0];

    const int k_eff = topk_;
    const float* row = score_ + (static_cast<int64_t>(h) * batch_ + b) * static_cast<int64_t>(max_seqblock_);

    // Zero the s_topk slots that the trait may not touch (e.g. the equal-scan
    // tail underfills). Since num_blocks > topk here, k_eff == topk, so the
    // trait writes all topk slots -- but keep the init defensive.
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

    // Ascending sort by rank-by-compare (k_eff <= kMaxTopK = 32). Each slot
    // computes its rank = #other-slots with smaller block id, then writes to
    // s_sorted[rank]. Same pass accumulates the effective KV length:
    // each selected block contributes min(block_size, seq_len - v*block_size)
    // valid tokens (only the final block can be partial).
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

    // Parallel page emit: one thread per output page.
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

// ============================================================================
// Host launcher: page-table kernel
// ============================================================================

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
  // NOTE: do NOT call .wait() -- see block-id launcher comment above.
}

// ============================================================================
// C API for Python (ctypes) binding
// ============================================================================
//
// Two exported symbols per kernel, one per SeqLenT variant (i32/i64). Two-level
// macro so a caller can #define the suffix as a token before invoking.

#define _DEFINE_MINIMAX_DECODE_TOPK(SUFFIX, T)                                               \
  extern "C" void minimax_decode_topk_##SUFFIX(                                              \
      void* queue_ptr,                                                                       \
      const void* score,                                                                     \
      const void* seq_lens,                                                                  \
      void* topk_idx,                                                                        \
      int32_t batch,                                                                         \
      int32_t num_heads,                                                                     \
      int32_t max_seqblock,                                                                  \
      int32_t block_size,                                                                    \
      int32_t topk) {                                                                        \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);                                   \
    minimax_decode_topk_launcher<T>(                                                         \
        queue, score, seq_lens, topk_idx, batch, num_heads, max_seqblock, block_size, topk); \
  }
#define DEFINE_MINIMAX_DECODE_TOPK(SUFFIX, T) _DEFINE_MINIMAX_DECODE_TOPK(SUFFIX, T)

#define _DEFINE_MINIMAX_DECODE_TOPK_PAGE_TABLE(SUFFIX, T)  \
  extern "C" void minimax_decode_topk_page_table_##SUFFIX( \
      void* queue_ptr,                                     \
      const void* score,                                   \
      const void* seq_lens,                                \
      const void* req_to_token,                            \
      const void* slot_ids,                                \
      void* page_table,                                    \
      void* seq_lens_out,                                  \
      int32_t batch,                                       \
      int32_t num_heads,                                   \
      int32_t max_seqblock,                                \
      int32_t block_size,                                  \
      int32_t topk,                                        \
      int32_t page_size,                                   \
      int32_t r2t_stride,                                  \
      int32_t max_kv_len,                                  \
      int32_t max_reqs,                                    \
      int32_t max_sparse_pages) {                          \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr); \
    minimax_decode_topk_page_table_launcher<T>(            \
        queue,                                             \
        score,                                             \
        seq_lens,                                          \
        req_to_token,                                      \
        slot_ids,                                          \
        page_table,                                        \
        seq_lens_out,                                      \
        batch,                                             \
        num_heads,                                         \
        max_seqblock,                                      \
        block_size,                                        \
        topk,                                              \
        page_size,                                         \
        r2t_stride,                                        \
        max_kv_len,                                        \
        max_reqs,                                          \
        max_sparse_pages);                                 \
  }
#define DEFINE_MINIMAX_DECODE_TOPK_PAGE_TABLE(SUFFIX, T) _DEFINE_MINIMAX_DECODE_TOPK_PAGE_TABLE(SUFFIX, T)

DEFINE_MINIMAX_DECODE_TOPK(i32, int32_t)
DEFINE_MINIMAX_DECODE_TOPK(i64, int64_t)
DEFINE_MINIMAX_DECODE_TOPK_PAGE_TABLE(i32, int32_t)
DEFINE_MINIMAX_DECODE_TOPK_PAGE_TABLE(i64, int64_t)

#undef DEFINE_MINIMAX_DECODE_TOPK
#undef _DEFINE_MINIMAX_DECODE_TOPK
#undef DEFINE_MINIMAX_DECODE_TOPK_PAGE_TABLE
#undef _DEFINE_MINIMAX_DECODE_TOPK_PAGE_TABLE

}  // namespace sycl_kernel
}  // namespace sgl
