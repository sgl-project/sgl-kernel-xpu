/**
 * HiSparse: load_cache_to_device_buffer SYCL kernel (Intel XPU).
 *
 * Ports load_cache_to_device_buffer_kernel<...> from the CUDA source
 * (sglang jit_kernel/csrc/hisparse.cuh). One work-group processes one request:
 * it hashes the request's top-k token positions, classifies the current hot
 * device buffer slots into hits / evictables, streams the missing tokens in
 * from the host cache into evicted slots, and rewrites the per-request LRU
 * order (evictables at the front, hits at the back).
 *
 * CUDA -> SYCL mapping:
 *   - warp (32 lanes)          -> sub-group pinned to kWarpSize (32)
 *   - __ballot + popc(&before) -> exclusive_scan_over_group (local prefix)
 *   - popc(mask)               -> reduce_over_group (sub-group total)
 *   - __shfl_up / __shfl       -> inclusive_scan_over_group / group_broadcast
 *   - atomicCAS (shared)       -> atomic_ref<..., local_space>
 *   - extern __shared__        -> local_accessor<char, 1>
 *   - __syncthreads()          -> item.barrier(local_space)
 *
 * The sub-group is pinned to 32 lanes so the slot<->lane mapping
 * (slot_idx = chunk * 32 + lane) and the shared-memory layout match the CUDA
 * kernel bit-for-bit, giving identical eviction ordering and outputs.
 */

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include "c4_layout.hpp"

namespace sgl {
namespace sycl_kernel {
namespace hisparse {

// Fixed logical warp width (matches the CUDA kernel this ports).
static constexpr int kWarpSize = 32;

static constexpr int32_t kTokenHit = static_cast<int32_t>(0xFFFFFFFF);  // -1 sentinel "already resident"
static constexpr int32_t kHashEmpty = -1;

// Knuth multiplicative hash for the open-addressing table of size hash_size.
inline int hash_slot(int32_t key, int hash_size) {
  return static_cast<int>((static_cast<uint32_t>(key) * 2654435761u) % static_cast<uint32_t>(hash_size));
}

// Cooperative linear (non-paged) item copy across a sub-group. Used by the
// generic (non-DSv4) miss-copy path where device + host are both linear with
// stride == item_size_bytes.
inline void transfer_item_linear(int lane_id, int sg_size, const void* src, void* dst, int64_t item_size_bytes) {
  const int64_t nwords = item_size_bytes / 8;
  const int64_t* s = static_cast<const int64_t*>(src);
  int64_t* d = static_cast<int64_t*>(dst);
  for (int64_t j = lane_id; j < nwords; j += sg_size) {
    d[j] = s[j];
  }
  const int64_t tail_start = nwords * 8;
  const char* sc = static_cast<const char*>(src) + tail_start;
  char* dc = static_cast<char*>(dst) + tail_start;
  for (int64_t j = lane_id; j < item_size_bytes - tail_start; j += sg_size) {
    dc[j] = sc[j];
  }
}

// Shared-memory size calculation (mirrors the CUDA SmemLayout).
// Layout: int32_t region (4-byte aligned) followed by int16_t region.
template <int NUM_TOP_K, int HOT_BUFFER_SIZE>
struct SmemLayout {
  static constexpr int HASH_SIZE = NUM_TOP_K * 2;
  static constexpr int NUM_BUFFER_CHUNKS = (HOT_BUFFER_SIZE + kWarpSize - 1) / kWarpSize;
  // int32_t region: top_k_tokens + chunk_offset + evict_chunk_offset + hash_keys + {total_hits, newest_hit}
  static constexpr int TOTAL_INT32 = NUM_TOP_K + (NUM_BUFFER_CHUNKS + 1) + (NUM_BUFFER_CHUNKS + 1) + HASH_SIZE + 2;
  // int16_t region: lru_slots_out + hash_vals
  static constexpr int TOTAL_INT16 = HOT_BUFFER_SIZE + HASH_SIZE;
  static constexpr size_t BYTES = TOTAL_INT32 * sizeof(int32_t) + TOTAL_INT16 * sizeof(int16_t);
};

// Local (shared) memory atomic CAS returning the previous value, matching
// CUDA atomicCAS(addr, compare, val) semantics.
inline int32_t atomic_cas_local(int32_t* addr, int32_t compare, int32_t val) {
  ::sycl::atomic_ref<
      int32_t,
      ::sycl::memory_order::relaxed,
      ::sycl::memory_scope::work_group,
      ::sycl::access::address_space::local_space>
      ref(*addr);
  int32_t expected = compare;
  ref.compare_exchange_strong(expected, val);
  // On success `expected` is unchanged (== compare); on failure it holds the
  // current value. Either way this equals CUDA atomicCAS's return value.
  return expected;
}

// Single-sub-group inclusive prefix scan over a shared array window
// [offset, offset+kWarpSize), threading a running accumulator. Mirrors the CUDA
// warp_inclusive_scan (which used __shfl_up_sync / __shfl_sync).
inline int warp_inclusive_scan(
    const ::sycl::sub_group& sg, int32_t* s_data, int lane_id, int sg_size, int offset, int count, int accumulator) {
  const int idx = lane_id + offset;
  int val = (idx < count) ? s_data[idx] : 0;
  val = ::sycl::inclusive_scan_over_group(sg, val, ::sycl::plus<int>());
  val += accumulator;
  if (idx < count) {
    s_data[idx] = val;
  }
  accumulator = ::sycl::group_broadcast(sg, val, sg_size - 1);
  return accumulator;
}

template <
    int BLOCK_SIZE,
    int NUM_TOP_K,
    int HOT_BUFFER_SIZE,
    bool IsMLA,
    bool IsDsv4Layout,
    typename SeqLensT,
    typename ReqPoolIndicesT>
class LoadCacheToDeviceBufferKernel {
 public:
  static_assert(!IsDsv4Layout || IsMLA, "DSv4 page-padded layout is K-only (MLA).");

  using Layout = SmemLayout<NUM_TOP_K, HOT_BUFFER_SIZE>;
  static constexpr int NUM_WARPS = BLOCK_SIZE / kWarpSize;
  static constexpr int NUM_TOKEN_CHUNKS = (NUM_TOP_K + kWarpSize - 1) / kWarpSize;
  static constexpr int NUM_BUFFER_CHUNKS = Layout::NUM_BUFFER_CHUNKS;
  static constexpr int HASH_SIZE = Layout::HASH_SIZE;

  LoadCacheToDeviceBufferKernel(
      const int32_t* top_k_tokens,
      int32_t* device_buffer_tokens,
      const int64_t* host_cache_locs,
      const int32_t* device_buffer_locs,
      const void* host_cache_k,
      const void* host_cache_v,
      void* device_buffer_k,
      void* device_buffer_v,
      int32_t* top_k_device_locs,
      const ReqPoolIndicesT* req_pool_indices,
      const SeqLensT* seq_lens,
      int16_t* lru_slots,
      const int32_t* num_real_reqs,
      int64_t buffer_stride_0,
      int64_t host_stride,
      int64_t lru_slot_stride_0,
      int64_t top_k_tokens_stride,
      int64_t top_k_device_locs_stride,
      int64_t page_size,
      int64_t item_size_bytes,
      ::sycl::local_accessor<char, 1> smem)
      : top_k_tokens_(top_k_tokens),
        device_buffer_tokens_(device_buffer_tokens),
        host_cache_locs_(host_cache_locs),
        device_buffer_locs_(device_buffer_locs),
        host_cache_k_(host_cache_k),
        host_cache_v_(host_cache_v),
        device_buffer_k_(device_buffer_k),
        device_buffer_v_(device_buffer_v),
        top_k_device_locs_(top_k_device_locs),
        req_pool_indices_(req_pool_indices),
        seq_lens_(seq_lens),
        lru_slots_(lru_slots),
        num_real_reqs_(num_real_reqs),
        buffer_stride_0_(buffer_stride_0),
        host_stride_(host_stride),
        lru_slot_stride_0_(lru_slot_stride_0),
        top_k_tokens_stride_(top_k_tokens_stride),
        top_k_device_locs_stride_(top_k_device_locs_stride),
        page_size_(page_size),
        item_size_bytes_(item_size_bytes),
        smem_(smem) {}

  [[sycl::reqd_sub_group_size(kWarpSize)]] void operator()(::sycl::nd_item<1> item) const {
    const int bid = static_cast<int>(item.get_group(0));
    // Early exit for padded blocks (CUDA graph pads batch to a captured size).
    if (bid >= num_real_reqs_[0]) return;

    const ::sycl::sub_group sg = item.get_sub_group();
    const int tid = static_cast<int>(item.get_local_id(0));
    const int warp_id = static_cast<int>(sg.get_group_linear_id());
    const int lane_id = static_cast<int>(sg.get_local_linear_id());
    const int sg_size = static_cast<int>(sg.get_max_local_range()[0]);

    const int64_t rid = static_cast<int64_t>(req_pool_indices_[bid]);
    const int64_t seq_len = static_cast<int64_t>(seq_lens_[bid]);

    // Per-request base offsets.
    const int32_t* req_top_k_tokens = top_k_tokens_ + bid * top_k_tokens_stride_;
    int32_t* req_top_k_device_locs = top_k_device_locs_ + bid * top_k_device_locs_stride_;

    const int64_t buffer_offset = rid * buffer_stride_0_;
    int32_t* req_device_buffer_tokens = device_buffer_tokens_ + buffer_offset;
    const int32_t* req_device_buffer_locs = device_buffer_locs_ + buffer_offset;
    const int64_t* req_host_cache_locs = host_cache_locs_ + rid * host_stride_;
    int16_t* req_lru_slots = lru_slots_ + rid * lru_slot_stride_0_;

    // Fast path: short sequences have all tokens resident in device-buffer order.
    if (seq_len <= HOT_BUFFER_SIZE) {
      const int count = (seq_len < NUM_TOP_K) ? static_cast<int>(seq_len) : NUM_TOP_K;
      for (int i = tid; i < count; i += BLOCK_SIZE) {
        int32_t token_pos = req_top_k_tokens[i];
        if (token_pos >= 0) {
          req_top_k_device_locs[i] = req_device_buffer_locs[token_pos];
        }
      }
      return;
    }

    // Carve up the shared-memory scratch: int32 region first, then int16.
    // SYCL local memory is allocated max-aligned, so the int32 reinterpret is safe.
    char* smem_raw = &smem_[0];
    int32_t* smem_i32 = reinterpret_cast<int32_t*>(smem_raw);
    int32_t* s_top_k_tokens = smem_i32;                                        // NUM_TOP_K
    int32_t* s_chunk_offset = s_top_k_tokens + NUM_TOP_K;                       // NUM_BUFFER_CHUNKS + 1
    int32_t* s_evict_chunk_offset = s_chunk_offset + (NUM_BUFFER_CHUNKS + 1);   // NUM_BUFFER_CHUNKS + 1
    int32_t* s_hash_keys = s_evict_chunk_offset + (NUM_BUFFER_CHUNKS + 1);      // HASH_SIZE
    int32_t* s_total_hits_ptr = s_hash_keys + HASH_SIZE;                        // 1
    int32_t* s_newest_hit_ptr = s_hash_keys + HASH_SIZE + 1;                    // 1

    int16_t* smem_i16 = reinterpret_cast<int16_t*>(smem_i32 + Layout::TOTAL_INT32);
    int16_t* s_lru_slots_out = smem_i16;                                       // HOT_BUFFER_SIZE
    int16_t* s_hash_vals = s_lru_slots_out + HOT_BUFFER_SIZE;                   // HASH_SIZE

    // Initialize counters, hash table, and prefix-sum offsets.
    if (tid == 0) {
      *s_total_hits_ptr = 0;
      *s_newest_hit_ptr = 0;
    }
    for (int i = tid; i < HASH_SIZE; i += BLOCK_SIZE) {
      s_hash_keys[i] = kHashEmpty;
    }
    for (int i = tid; i < NUM_BUFFER_CHUNKS + 1; i += BLOCK_SIZE) {
      s_chunk_offset[i] = 0;
      s_evict_chunk_offset[i] = 0;
    }
    item.barrier(::sycl::access::fence_space::local_space);

    const int newest_slot = HOT_BUFFER_SIZE;
    const int32_t newest_token = static_cast<int32_t>(seq_len - 1);

    // Insert top-k token positions into the shared-memory hash table.
    for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
      int32_t token_idx = req_top_k_tokens[i];
      if (token_idx == newest_token) {
        // The latest token lives at newest_slot (first slot of the extra page),
        // excluded from LRU tracking. Bind and mark it as a hit.
        s_top_k_tokens[i] = kTokenHit;
        req_top_k_device_locs[i] = req_device_buffer_locs[newest_slot];
        *s_newest_hit_ptr = 1;
      } else {
        int slot = hash_slot(token_idx, HASH_SIZE);
        while (true) {
          int32_t old = atomic_cas_local(&s_hash_keys[slot], kHashEmpty, token_idx);
          if (old == kHashEmpty || old == token_idx) {
            s_hash_vals[slot] = static_cast<int16_t>(i);
            break;
          }
          slot = (slot + 1) % HASH_SIZE;
        }
        s_top_k_tokens[i] = token_idx;
      }
    }
    item.barrier(::sycl::access::fence_space::local_space);

    // Pass over hot-buffer slots: classify hits vs evictables and compact them.
    constexpr int ITERATIONS_PER_WARP_BUFFER = (NUM_BUFFER_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
    int total_hit_count = 0;
    int total_evict_count = 0;
    for (int iter = 0; iter < ITERATIONS_PER_WARP_BUFFER; iter++) {
      const int chunk_idx = warp_id + iter * NUM_WARPS;
      const bool has_valid_chunk = chunk_idx < NUM_BUFFER_CHUNKS;

      const int slot_idx = chunk_idx * kWarpSize + lane_id;
      const bool has_valid_slot = has_valid_chunk && (slot_idx < HOT_BUFFER_SIZE);
      const int16_t buf_slot = has_valid_slot ? req_lru_slots[slot_idx] : static_cast<int16_t>(-1);
      int32_t my_buffer_token = (buf_slot >= 0) ? req_device_buffer_tokens[buf_slot] : -1;
      int my_found_top_k_idx = -1;
      if (my_buffer_token >= 0) {
        int h = hash_slot(my_buffer_token, HASH_SIZE);
        while (true) {
          int32_t k = s_hash_keys[h];
          if (k == my_buffer_token) {
            my_found_top_k_idx = static_cast<int32_t>(s_hash_vals[h]);
            break;
          }
          if (k == kHashEmpty) break;
          h = (h + 1) % HASH_SIZE;
        }
      }
      const bool is_hit = my_found_top_k_idx >= 0;
      const bool is_evictable = has_valid_slot && !is_hit;

      // Record hits: bind the top-k index to this resident slot's device loc.
      if (is_hit) {
        s_top_k_tokens[my_found_top_k_idx] = kTokenHit;
        req_top_k_device_locs[my_found_top_k_idx] = req_device_buffer_locs[buf_slot];
      }

      int local_hit_offset = 0;
      int local_evict_offset = 0;
      if (has_valid_chunk) {
        local_hit_offset = ::sycl::exclusive_scan_over_group(sg, is_hit ? 1 : 0, ::sycl::plus<int>());
        local_evict_offset = ::sycl::exclusive_scan_over_group(sg, is_evictable ? 1 : 0, ::sycl::plus<int>());
        const int warp_hits = ::sycl::reduce_over_group(sg, is_hit ? 1 : 0, ::sycl::plus<int>());
        const int warp_evicts = ::sycl::reduce_over_group(sg, is_evictable ? 1 : 0, ::sycl::plus<int>());
        if (lane_id == 0) {
          s_chunk_offset[chunk_idx + 1] = warp_hits;
          s_evict_chunk_offset[chunk_idx + 1] = warp_evicts;
        }
      }
      item.barrier(::sycl::access::fence_space::local_space);

      if (warp_id == 0) {
        total_hit_count =
            warp_inclusive_scan(sg, s_chunk_offset, lane_id, sg_size, chunk_idx + 1, NUM_BUFFER_CHUNKS + 1, total_hit_count);
        total_evict_count = warp_inclusive_scan(
            sg, s_evict_chunk_offset, lane_id, sg_size, chunk_idx + 1, NUM_BUFFER_CHUNKS + 1, total_evict_count);
        if (tid == 0) {
          *s_total_hits_ptr = total_hit_count;
        }
      }
      item.barrier(::sycl::access::fence_space::local_space);

      // Hits grow forward from index 0.
      if (is_hit) {
        int hit_offset = s_chunk_offset[chunk_idx] + local_hit_offset;
        s_lru_slots_out[hit_offset] = buf_slot;
      }
      // Evictables grow backward from HOT_BUFFER_SIZE - 1.
      if (is_evictable) {
        int evict_offset = s_evict_chunk_offset[chunk_idx] + local_evict_offset;
        s_lru_slots_out[HOT_BUFFER_SIZE - 1 - evict_offset] = buf_slot;
      }
    }
    item.barrier(::sycl::access::fence_space::local_space);

    // Reset offsets for the miss-counting phase (NUM_TOKEN_CHUNKS + 1 entries).
    for (int i = tid; i < NUM_TOKEN_CHUNKS + 1; i += BLOCK_SIZE) {
      s_chunk_offset[i] = 0;
    }
    item.barrier(::sycl::access::fence_space::local_space);

    // Pass over top-k tokens: identify misses and assign them evictable slots.
    int total_misses = 0;
    constexpr int ITERATIONS_PER_WARP_TOKEN = (NUM_TOKEN_CHUNKS + NUM_WARPS - 1) / NUM_WARPS;
    for (int iter = 0; iter < ITERATIONS_PER_WARP_TOKEN; iter++) {
      const int chunk_idx = warp_id + iter * NUM_WARPS;
      const bool has_valid_chunk = chunk_idx < NUM_TOKEN_CHUNKS;

      const int chunk_token_start = chunk_idx * kWarpSize;
      const int my_token_idx = chunk_token_start + lane_id;
      const bool has_valid_token = has_valid_chunk && (my_token_idx < NUM_TOP_K);

      int32_t my_token = 0;
      bool is_miss = false;
      int local_miss_offset = 0;

      if (has_valid_token) {
        is_miss = s_top_k_tokens[my_token_idx] != kTokenHit;
        if (is_miss) {
          my_token = s_top_k_tokens[my_token_idx];
        }
      }

      if (has_valid_chunk) {
        local_miss_offset = ::sycl::exclusive_scan_over_group(sg, is_miss ? 1 : 0, ::sycl::plus<int>());
        const int warp_miss_count = ::sycl::reduce_over_group(sg, is_miss ? 1 : 0, ::sycl::plus<int>());
        if (lane_id == 0) {
          s_chunk_offset[chunk_idx + 1] = warp_miss_count;
        }
      }
      item.barrier(::sycl::access::fence_space::local_space);

      if (warp_id == 0) {
        total_misses =
            warp_inclusive_scan(sg, s_chunk_offset, lane_id, sg_size, chunk_idx + 1, NUM_TOKEN_CHUNKS + 1, total_misses);
      }
      item.barrier(::sycl::access::fence_space::local_space);

      if (is_miss) {
        int miss_offset = s_chunk_offset[chunk_idx] + local_miss_offset;
        int16_t evict_slot = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - miss_offset];
        // Reuse s_top_k_tokens as miss scratch: miss_offset < my_token_idx always
        // holds (hits are skipped), so compacted writes never overrun pending reads.
        s_top_k_tokens[miss_offset] = my_token;
        req_top_k_device_locs[my_token_idx] = req_device_buffer_locs[evict_slot];
        req_device_buffer_tokens[evict_slot] = my_token;
      }
    }
    item.barrier(::sycl::access::fence_space::local_space);

    total_misses = NUM_TOP_K - *s_total_hits_ptr - *s_newest_hit_ptr;
    // Rewrite LRU order: misses then remaining evictables at the front (LRU),
    // hits at the back (MRU).
    {
      const int total_evictable = HOT_BUFFER_SIZE - *s_total_hits_ptr;
      for (int i = tid; i < HOT_BUFFER_SIZE; i += BLOCK_SIZE) {
        if (i < total_misses) {
          req_lru_slots[total_evictable - total_misses + i] = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - i];
        } else if (i < total_evictable) {
          req_lru_slots[i - total_misses] = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - i];
        } else {
          req_lru_slots[i] = s_lru_slots_out[i - total_evictable];
        }
      }
    }

    // Each sub-group copies one miss directly from host cache to device buffer.
    for (int miss_idx = warp_id; miss_idx < total_misses; miss_idx += NUM_WARPS) {
      const int32_t miss_token = s_top_k_tokens[miss_idx];
      const int16_t evict_slot = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - miss_idx];

      const int64_t src_loc = req_host_cache_locs[miss_token];
      const int64_t dst_loc = static_cast<int64_t>(req_device_buffer_locs[evict_slot]);

      if constexpr (IsDsv4Layout) {
        // Page-padded C4 device layout + page-padded host layout, K-only.
        transfer_item(
            lane_id,
            sg_size,
            device_buffer_k_,
            const_cast<void*>(host_cache_k_),
            static_cast<int32_t>(dst_loc),
            static_cast<int32_t>(src_loc));
      } else {
        // Generic path: device + host both linear, stride == item_size_bytes.
        const char* src_k = static_cast<const char*>(host_cache_k_) + src_loc * item_size_bytes_;
        char* dst_k = static_cast<char*>(device_buffer_k_) + dst_loc * item_size_bytes_;
        transfer_item_linear(lane_id, sg_size, src_k, dst_k, item_size_bytes_);

        if constexpr (!IsMLA) {
          const char* src_v = static_cast<const char*>(host_cache_v_) + src_loc * item_size_bytes_;
          char* dst_v = static_cast<char*>(device_buffer_v_) + dst_loc * item_size_bytes_;
          transfer_item_linear(lane_id, sg_size, src_v, dst_v, item_size_bytes_);
        }
      }
    }
  }

 private:
  const int32_t* top_k_tokens_;
  int32_t* device_buffer_tokens_;
  const int64_t* host_cache_locs_;
  const int32_t* device_buffer_locs_;
  const void* host_cache_k_;
  const void* host_cache_v_;
  void* device_buffer_k_;
  void* device_buffer_v_;
  int32_t* top_k_device_locs_;
  const ReqPoolIndicesT* req_pool_indices_;
  const SeqLensT* seq_lens_;
  int16_t* lru_slots_;
  const int32_t* num_real_reqs_;
  int64_t buffer_stride_0_;
  int64_t host_stride_;
  int64_t lru_slot_stride_0_;
  int64_t top_k_tokens_stride_;
  int64_t top_k_device_locs_stride_;
  int64_t page_size_;
  int64_t item_size_bytes_;
  ::sycl::local_accessor<char, 1> smem_;
};

template <
    int BLOCK_SIZE,
    int NUM_TOP_K,
    int HOT_BUFFER_SIZE,
    bool IsMLA,
    bool IsDsv4Layout,
    typename SeqLensT,
    typename ReqPoolIndicesT>
void load_cache_to_device_buffer_launcher(
    ::sycl::queue& queue,
    const void* top_k_tokens,
    void* device_buffer_tokens,
    const void* host_cache_locs,
    const void* device_buffer_locs,
    const void* host_cache_k,
    const void* host_cache_v,
    void* device_buffer_k,
    void* device_buffer_v,
    void* top_k_device_locs,
    const void* req_pool_indices,
    const void* seq_lens,
    void* lru_slots,
    const void* num_real_reqs,
    int64_t batch_size,
    int64_t buffer_stride_0,
    int64_t host_stride,
    int64_t lru_slot_stride_0,
    int64_t top_k_tokens_stride,
    int64_t top_k_device_locs_stride,
    int64_t page_size,
    int64_t item_size_bytes) {
  if (batch_size == 0) {
    return;
  }
  using Kernel = LoadCacheToDeviceBufferKernel<
      BLOCK_SIZE,
      NUM_TOP_K,
      HOT_BUFFER_SIZE,
      IsMLA,
      IsDsv4Layout,
      SeqLensT,
      ReqPoolIndicesT>;
  constexpr size_t smem_bytes = SmemLayout<NUM_TOP_K, HOT_BUFFER_SIZE>::BYTES;

  queue.submit([&](::sycl::handler& cgh) {
    ::sycl::local_accessor<char, 1> smem(::sycl::range<1>(smem_bytes), cgh);
    cgh.parallel_for(
        ::sycl::nd_range<1>(
            ::sycl::range<1>(static_cast<size_t>(batch_size) * BLOCK_SIZE), ::sycl::range<1>(BLOCK_SIZE)),
        Kernel(
            static_cast<const int32_t*>(top_k_tokens),
            static_cast<int32_t*>(device_buffer_tokens),
            static_cast<const int64_t*>(host_cache_locs),
            static_cast<const int32_t*>(device_buffer_locs),
            host_cache_k,
            host_cache_v,
            device_buffer_k,
            device_buffer_v,
            static_cast<int32_t*>(top_k_device_locs),
            static_cast<const ReqPoolIndicesT*>(req_pool_indices),
            static_cast<const SeqLensT*>(seq_lens),
            static_cast<int16_t*>(lru_slots),
            static_cast<const int32_t*>(num_real_reqs),
            buffer_stride_0,
            host_stride,
            lru_slot_stride_0,
            top_k_tokens_stride,
            top_k_device_locs_stride,
            page_size,
            item_size_bytes,
            smem));
  });
}

// ============================================================================
// C API for Python (ctypes) binding
// ============================================================================
//
// The compile-time template config (block size, top-k, hot-buffer size, MLA /
// DSv4 flags) is fixed per module via -D macros, mirroring the CUDA JIT that
// bakes the same values into template arguments. The seq_lens / req_pool_indices
// dtype combination (i32/i64) is selected at call time by picking the matching
// exported symbol.

#define _DEFINE_LOAD_CACHE(SEQ_SUFFIX, SEQ_T, RPI_SUFFIX, RPI_T)                                                    \
  extern "C" void load_cache_to_device_buffer_##SEQ_SUFFIX##_##RPI_SUFFIX(                                           \
      void* queue_ptr,                                                                                               \
      const void* top_k_tokens,                                                                                      \
      void* device_buffer_tokens,                                                                                    \
      const void* host_cache_locs,                                                                                   \
      const void* device_buffer_locs,                                                                                \
      const void* host_cache_k,                                                                                      \
      const void* host_cache_v,                                                                                      \
      void* device_buffer_k,                                                                                         \
      void* device_buffer_v,                                                                                         \
      void* top_k_device_locs,                                                                                       \
      const void* req_pool_indices,                                                                                  \
      const void* seq_lens,                                                                                          \
      void* lru_slots,                                                                                               \
      const void* num_real_reqs,                                                                                     \
      int64_t batch_size,                                                                                            \
      int64_t buffer_stride_0,                                                                                       \
      int64_t host_stride,                                                                                           \
      int64_t lru_slot_stride_0,                                                                                     \
      int64_t top_k_tokens_stride,                                                                                   \
      int64_t top_k_device_locs_stride,                                                                              \
      int64_t page_size,                                                                                             \
      int64_t item_size_bytes) {                                                                                     \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);                                                           \
    load_cache_to_device_buffer_launcher<                                                                           \
        SGL_HISPARSE_BLOCK_SIZE,                                                                                     \
        SGL_HISPARSE_NUM_TOP_K,                                                                                      \
        SGL_HISPARSE_HOT_BUFFER_SIZE,                                                                                \
        (SGL_HISPARSE_IS_MLA != 0),                                                                                  \
        (SGL_HISPARSE_IS_DSV4 != 0),                                                                                 \
        SEQ_T,                                                                                                       \
        RPI_T>(                                                                                                      \
        queue,                                                                                                       \
        top_k_tokens,                                                                                                \
        device_buffer_tokens,                                                                                        \
        host_cache_locs,                                                                                             \
        device_buffer_locs,                                                                                          \
        host_cache_k,                                                                                                \
        host_cache_v,                                                                                                \
        device_buffer_k,                                                                                             \
        device_buffer_v,                                                                                             \
        top_k_device_locs,                                                                                           \
        req_pool_indices,                                                                                            \
        seq_lens,                                                                                                    \
        lru_slots,                                                                                                   \
        num_real_reqs,                                                                                               \
        batch_size,                                                                                                  \
        buffer_stride_0,                                                                                             \
        host_stride,                                                                                                 \
        lru_slot_stride_0,                                                                                           \
        top_k_tokens_stride,                                                                                         \
        top_k_device_locs_stride,                                                                                    \
        page_size,                                                                                                   \
        item_size_bytes);                                                                                            \
  }
#define DEFINE_LOAD_CACHE(SEQ_SUFFIX, SEQ_T, RPI_SUFFIX, RPI_T) _DEFINE_LOAD_CACHE(SEQ_SUFFIX, SEQ_T, RPI_SUFFIX, RPI_T)

#if defined(SGL_HISPARSE_BLOCK_SIZE) && defined(SGL_HISPARSE_NUM_TOP_K) && defined(SGL_HISPARSE_HOT_BUFFER_SIZE) && \
    defined(SGL_HISPARSE_IS_MLA) && defined(SGL_HISPARSE_IS_DSV4)
DEFINE_LOAD_CACHE(i64, int64_t, i64, int64_t)
DEFINE_LOAD_CACHE(i64, int64_t, i32, int32_t)
DEFINE_LOAD_CACHE(i32, int32_t, i64, int64_t)
DEFINE_LOAD_CACHE(i32, int32_t, i32, int32_t)
#endif

#undef DEFINE_LOAD_CACHE
#undef _DEFINE_LOAD_CACHE

}  // namespace hisparse
}  // namespace sycl_kernel
}  // namespace sgl
