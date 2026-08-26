/**
 * HiSparse: load_cache_to_device_buffer SYCL kernel (Intel XPU).
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "c4_layout.hpp"

namespace sgl {
namespace sycl_kernel {
namespace hisparse {

static constexpr int32_t kTokenHit = static_cast<int32_t>(0xFFFFFFFF);  // -1 sentinel "already resident"
static constexpr int32_t kHashEmpty = -1;

// Knuth multiplicative hash into an open-addressing table of size hash_size.
// hash_mask is hash_size-1 for power-of-two sizes (else 0), avoiding a modulo.
inline int hash_slot(int32_t key, int hash_size, int hash_mask) {
  const uint32_t h = static_cast<uint32_t>(key) * 2654435761u;
  return hash_mask != 0 ? static_cast<int>(h & static_cast<uint32_t>(hash_mask))
                        : static_cast<int>(h % static_cast<uint32_t>(hash_size));
}

// Linear-probe step: (slot + 1) % hash_size without the division.
inline int hash_probe_next(int slot, int hash_size) {
  const int next = slot + 1;
  return next == hash_size ? 0 : next;
}

// Cooperative linear (non-paged) item copy across a sub-group, for the generic
// miss path where device + host are both linear with stride item_size_bytes.
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

// Local-memory atomic CAS returning the previous value.
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
  // current value.
  return expected;
}

// Single-sub-group inclusive prefix scan over the local-memory window
// [offset, offset+sg_size), threading a running accumulator.
inline int sub_group_inclusive_scan(
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

// Local-memory layout: an int32_t region followed by an int16_t region starting
// at int32 slot total_int32, so it inherits 4-byte alignment. num_top_k and
// hot_buffer_size are runtime values, so the host computes the layout and hands
// the offsets to the kernel.
struct SmemLayout {
  int hash_size;
  int hash_mask;  // hash_size-1 if a power of two, else 0
  int num_buffer_chunks;
  int num_token_chunks;
  int total_int32;
  int total_int16;
  int total_int32_slots;  // allocation size of the int32 local_accessor

  static SmemLayout make(int num_top_k, int hot_buffer_size) {
    SmemLayout l{};
    l.hash_size = num_top_k * 2;
    l.hash_mask = (l.hash_size & (l.hash_size - 1)) == 0 ? l.hash_size - 1 : 0;
    l.num_buffer_chunks = (hot_buffer_size + kSubGroupSize - 1) / kSubGroupSize;
    l.num_token_chunks = (num_top_k + kSubGroupSize - 1) / kSubGroupSize;
    // int32 region: top_k_tokens + chunk_offset + evict_chunk_offset + hash_keys
    //               + {total_hits, newest_hit}
    l.total_int32 = num_top_k + (l.num_buffer_chunks + 1) + (l.num_buffer_chunks + 1) + l.hash_size + 2;
    // int16 region: lru_slots_out + hash_vals
    l.total_int16 = hot_buffer_size + l.hash_size;
    l.total_int32_slots =
        l.total_int32 + static_cast<int>((l.total_int16 * sizeof(int16_t) + sizeof(int32_t) - 1) / sizeof(int32_t));
    return l;
  }

  size_t bytes() const {
    return static_cast<size_t>(total_int32_slots) * sizeof(int32_t);
  }
};

// seq_lens and req_pool_indices are int32 or int64 depending on the caller. Each
// is read once per work-group, so a runtime dtype branch beats instantiating the
// kernel for both index types.
inline int64_t load_index(const void* base, bool is_i64, int bid) {
  return is_i64 ? static_cast<const int64_t*>(base)[bid] : static_cast<int64_t>(static_cast<const int32_t*>(base)[bid]);
}

// IsMLA / IsDsv4Layout stay compile-time because they select the inner copy loop.
// Aggregate-initialized so the launcher can name the fields: the list is long
// enough that positional arguments are easy to transpose silently.
template <bool IsMLA, bool IsDsv4Layout>
struct LoadCacheToDeviceBufferKernel {
  static_assert(!IsDsv4Layout || IsMLA, "DSv4 page-padded layout is K-only (MLA).");

  const int32_t* top_k_tokens_;
  int32_t* device_buffer_tokens_;
  const int64_t* host_cache_locs_;
  const int32_t* device_buffer_locs_;
  const void* host_cache_k_;
  const void* host_cache_v_;
  void* device_buffer_k_;
  void* device_buffer_v_;
  int32_t* top_k_device_locs_;
  const void* req_pool_indices_;
  const void* seq_lens_;
  int16_t* lru_slots_;
  const int32_t* num_real_reqs_;
  bool req_pool_indices_is_i64_;
  bool seq_lens_is_i64_;
  int64_t buffer_stride_0_;
  int64_t host_stride_;
  int64_t lru_slot_stride_0_;
  int64_t top_k_tokens_stride_;
  int64_t top_k_device_locs_stride_;
  int64_t item_size_bytes_;
  // Runtime shape / layout, precomputed on the host (see SmemLayout).
  int block_size_;
  int num_sub_groups_;
  int num_top_k_;
  int hot_buffer_size_;
  int hash_size_;
  int hash_mask_;
  int num_buffer_chunks_;
  int num_token_chunks_;
  int iters_per_sg_buffer_;
  int iters_per_sg_token_;
  int total_int32_;
  ::sycl::local_accessor<int32_t, 1> smem_;

  [[sycl::reqd_sub_group_size(kSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    const int bid = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    int32_t* req_top_k_device_locs = top_k_device_locs_ + bid * top_k_device_locs_stride_;

    // A graph-captured batch is padded to the captured size. Keep padded output
    // rows invalid without a separate fill kernel.
    if (bid >= num_real_reqs_[0]) {
      for (int i = tid; i < num_top_k_; i += block_size_) {
        req_top_k_device_locs[i] = -1;
      }
      return;
    }

    const ::sycl::sub_group sg = item.get_sub_group();
    const int sg_id = static_cast<int>(sg.get_group_linear_id());
    const int lane_id = static_cast<int>(sg.get_local_linear_id());
    const int sg_size = static_cast<int>(sg.get_max_local_range()[0]);

    const int64_t rid = load_index(req_pool_indices_, req_pool_indices_is_i64_, bid);
    const int64_t seq_len = load_index(seq_lens_, seq_lens_is_i64_, bid);

    // Per-request base offsets.
    const int32_t* req_top_k_tokens = top_k_tokens_ + bid * top_k_tokens_stride_;

    const int64_t buffer_offset = rid * buffer_stride_0_;
    int32_t* req_device_buffer_tokens = device_buffer_tokens_ + buffer_offset;
    const int32_t* req_device_buffer_locs = device_buffer_locs_ + buffer_offset;
    const int64_t* req_host_cache_locs = host_cache_locs_ + rid * host_stride_;
    int16_t* req_lru_slots = lru_slots_ + rid * lru_slot_stride_0_;

    // Fast path: short sequences have all tokens resident in device-buffer order.
    if (seq_len <= hot_buffer_size_) {
      const int count = (seq_len < num_top_k_) ? static_cast<int>(seq_len) : num_top_k_;
      for (int i = tid; i < num_top_k_; i += block_size_) {
        int32_t device_loc = -1;
        if (i < count) {
          const int32_t token_pos = req_top_k_tokens[i];
          if (token_pos >= 0) {
            device_loc = req_device_buffer_locs[token_pos];
          }
        }
        req_top_k_device_locs[i] = device_loc;
      }
      return;
    }

    // Scratch is one int32_t accessor; the int16 region starts at slot
    // total_int32_, so both regions stay 4-byte aligned.
    int32_t* smem_i32 = &smem_[0];
    int32_t* s_top_k_tokens = smem_i32;                                         // num_top_k
    int32_t* s_chunk_offset = s_top_k_tokens + num_top_k_;                      // num_buffer_chunks + 1
    int32_t* s_evict_chunk_offset = s_chunk_offset + (num_buffer_chunks_ + 1);  // num_buffer_chunks + 1
    int32_t* s_hash_keys = s_evict_chunk_offset + (num_buffer_chunks_ + 1);     // hash_size
    int32_t* s_total_hits_ptr = s_hash_keys + hash_size_;                       // 1
    int32_t* s_newest_hit_ptr = s_hash_keys + hash_size_ + 1;                   // 1

    int16_t* smem_i16 = reinterpret_cast<int16_t*>(smem_i32 + total_int32_);
    int16_t* s_lru_slots_out = smem_i16;                        // hot_buffer_size
    int16_t* s_hash_vals = s_lru_slots_out + hot_buffer_size_;  // hash_size

    // Initialize counters, hash table, and prefix-sum offsets.
    if (tid == 0) {
      *s_total_hits_ptr = 0;
      *s_newest_hit_ptr = 0;
    }
    for (int i = tid; i < hash_size_; i += block_size_) {
      s_hash_keys[i] = kHashEmpty;
    }
    for (int i = tid; i < num_buffer_chunks_ + 1; i += block_size_) {
      s_chunk_offset[i] = 0;
      s_evict_chunk_offset[i] = 0;
    }
    item.barrier(::sycl::access::fence_space::local_space);

    const int newest_slot = hot_buffer_size_;
    const int32_t newest_token = static_cast<int32_t>(seq_len - 1);

    // Insert top-k token positions into the local-memory hash table.
    for (int i = tid; i < num_top_k_; i += block_size_) {
      int32_t token_idx = req_top_k_tokens[i];
      if (token_idx == newest_token) {
        // The latest token lives at newest_slot, outside LRU tracking: bind it
        // directly and mark it a hit.
        s_top_k_tokens[i] = kTokenHit;
        req_top_k_device_locs[i] = req_device_buffer_locs[newest_slot];
        *s_newest_hit_ptr = 1;
      } else {
        int slot = hash_slot(token_idx, hash_size_, hash_mask_);
        while (true) {
          int32_t old = atomic_cas_local(&s_hash_keys[slot], kHashEmpty, token_idx);
          if (old == kHashEmpty || old == token_idx) {
            s_hash_vals[slot] = static_cast<int16_t>(i);
            break;
          }
          slot = hash_probe_next(slot, hash_size_);
        }
        s_top_k_tokens[i] = token_idx;
      }
    }
    item.barrier(::sycl::access::fence_space::local_space);

    // Pass over hot-buffer slots: classify hits vs evictables and compact them.
    int total_hit_count = 0;
    int total_evict_count = 0;
    for (int iter = 0; iter < iters_per_sg_buffer_; iter++) {
      const int chunk_idx = sg_id + iter * num_sub_groups_;
      const bool has_valid_chunk = chunk_idx < num_buffer_chunks_;

      const int slot_idx = chunk_idx * kSubGroupSize + lane_id;
      const bool has_valid_slot = has_valid_chunk && (slot_idx < hot_buffer_size_);
      const int16_t buf_slot = has_valid_slot ? req_lru_slots[slot_idx] : static_cast<int16_t>(-1);
      int32_t my_buffer_token = (buf_slot >= 0) ? req_device_buffer_tokens[buf_slot] : -1;
      int my_found_top_k_idx = -1;
      if (my_buffer_token >= 0) {
        int h = hash_slot(my_buffer_token, hash_size_, hash_mask_);
        while (true) {
          int32_t k = s_hash_keys[h];
          if (k == my_buffer_token) {
            my_found_top_k_idx = static_cast<int32_t>(s_hash_vals[h]);
            break;
          }
          if (k == kHashEmpty) break;
          h = hash_probe_next(h, hash_size_);
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
        const int sg_hits = ::sycl::reduce_over_group(sg, is_hit ? 1 : 0, ::sycl::plus<int>());
        const int sg_evicts = ::sycl::reduce_over_group(sg, is_evictable ? 1 : 0, ::sycl::plus<int>());
        if (lane_id == 0) {
          s_chunk_offset[chunk_idx + 1] = sg_hits;
          s_evict_chunk_offset[chunk_idx + 1] = sg_evicts;
        }
      }
      item.barrier(::sycl::access::fence_space::local_space);

      if (sg_id == 0) {
        // Bound the scan window to num_sub_groups lanes: only that many entries
        // were written this iteration, and letting the remaining lanes join with
        // the wide count would fold stale values from earlier iterations into the
        // accumulator and write it into slots future iterations read.
        const int scan_count = ::std::min(chunk_idx + 1 + num_sub_groups_, num_buffer_chunks_ + 1);
        total_hit_count =
            sub_group_inclusive_scan(sg, s_chunk_offset, lane_id, sg_size, chunk_idx + 1, scan_count, total_hit_count);
        total_evict_count = sub_group_inclusive_scan(
            sg, s_evict_chunk_offset, lane_id, sg_size, chunk_idx + 1, scan_count, total_evict_count);
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
      // Evictables grow backward from hot_buffer_size - 1.
      if (is_evictable) {
        int evict_offset = s_evict_chunk_offset[chunk_idx] + local_evict_offset;
        s_lru_slots_out[hot_buffer_size_ - 1 - evict_offset] = buf_slot;
      }
    }
    item.barrier(::sycl::access::fence_space::local_space);

    // Reset offsets for the miss-counting phase (num_token_chunks + 1 entries).
    for (int i = tid; i < num_token_chunks_ + 1; i += block_size_) {
      s_chunk_offset[i] = 0;
    }
    item.barrier(::sycl::access::fence_space::local_space);

    // Pass over top-k tokens: identify misses and assign them evictable slots.
    int total_misses = 0;
    for (int iter = 0; iter < iters_per_sg_token_; iter++) {
      const int chunk_idx = sg_id + iter * num_sub_groups_;
      const bool has_valid_chunk = chunk_idx < num_token_chunks_;

      const int chunk_token_start = chunk_idx * kSubGroupSize;
      const int my_token_idx = chunk_token_start + lane_id;
      const bool has_valid_token = has_valid_chunk && (my_token_idx < num_top_k_);

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
        const int sg_miss_count = ::sycl::reduce_over_group(sg, is_miss ? 1 : 0, ::sycl::plus<int>());
        if (lane_id == 0) {
          s_chunk_offset[chunk_idx + 1] = sg_miss_count;
        }
      }
      item.barrier(::sycl::access::fence_space::local_space);

      if (sg_id == 0) {
        // Same bounded window as the buffer pass above.
        const int scan_count = ::std::min(chunk_idx + 1 + num_sub_groups_, num_token_chunks_ + 1);
        total_misses =
            sub_group_inclusive_scan(sg, s_chunk_offset, lane_id, sg_size, chunk_idx + 1, scan_count, total_misses);
      }
      item.barrier(::sycl::access::fence_space::local_space);

      if (is_miss) {
        int miss_offset = s_chunk_offset[chunk_idx] + local_miss_offset;
        int16_t evict_slot = s_lru_slots_out[hot_buffer_size_ - 1 - miss_offset];
        // Reuse s_top_k_tokens as miss scratch: miss_offset < my_token_idx always
        // holds (hits are skipped), so compacted writes never overrun pending reads.
        s_top_k_tokens[miss_offset] = my_token;
        req_top_k_device_locs[my_token_idx] = req_device_buffer_locs[evict_slot];
        req_device_buffer_tokens[evict_slot] = my_token;
      }
    }
    item.barrier(::sycl::access::fence_space::local_space);

    total_misses = num_top_k_ - *s_total_hits_ptr - *s_newest_hit_ptr;
    // Rewrite LRU order: misses then remaining evictables at the front (LRU),
    // hits at the back (MRU).
    {
      const int total_evictable = hot_buffer_size_ - *s_total_hits_ptr;
      for (int i = tid; i < hot_buffer_size_; i += block_size_) {
        if (i < total_misses) {
          req_lru_slots[total_evictable - total_misses + i] = s_lru_slots_out[hot_buffer_size_ - 1 - i];
        } else if (i < total_evictable) {
          req_lru_slots[i - total_misses] = s_lru_slots_out[hot_buffer_size_ - 1 - i];
        } else {
          req_lru_slots[i] = s_lru_slots_out[i - total_evictable];
        }
      }
    }

    // Each sub-group copies one miss directly from host cache to device buffer.
    for (int miss_idx = sg_id; miss_idx < total_misses; miss_idx += num_sub_groups_) {
      const int32_t miss_token = s_top_k_tokens[miss_idx];
      const int16_t evict_slot = s_lru_slots_out[hot_buffer_size_ - 1 - miss_idx];

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
};

}  // namespace hisparse
}  // namespace sycl_kernel
}  // namespace sgl
