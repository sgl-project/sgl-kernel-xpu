/**
 * HiSparse C4 paged-cache layout helpers (SYCL / Intel XPU).
 *
 * Ports device::hisparse::{get_pointer_paged, transfer_item} from the CUDA
 * kernel (sglang jit_kernel/include/sgl_kernel/deepseek_v4/kvcacheio.cuh).
 *
 * Paged C4 cache layout (per page of kPageSize tokens):
 *   VALUE 0, VALUE 1, ..., VALUE 63,     (kValueBytes each)
 *   SCALE 0, SCALE 1, ..., SCALE 63,     (kScaleBytes each)
 *   [padding to align the page to a 576-byte boundary]
 *
 * FlashMLA requires each page to be aligned to 576 bytes.
 */

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl_kernel {
namespace hisparse {

// C4 paged layout constants (must match kvcacheio.cuh exactly).
inline constexpr int64_t kPageSize = 64;
inline constexpr int64_t kPageBits = 6;  // log2(kPageSize)
inline constexpr int64_t kValueBytes = 576;
inline constexpr int64_t kScaleBytes = 8;
inline constexpr int64_t kItemBytes = kValueBytes + kScaleBytes;

// div_ceil(kItemBytes * kPageSize, 576) * 576 -> page byte stride.
inline constexpr int64_t kPageBytes = ((kItemBytes * kPageSize + 576 - 1) / 576) * 576;
inline constexpr int64_t kScaleOffset = kValueBytes * kPageSize;

// int64-word counts for the strided copy loops.
inline constexpr int kValueWords = static_cast<int>(kValueBytes / 8);  // 72
inline constexpr int kScaleWords = static_cast<int>(kScaleBytes / 8);  // 1

static_assert(kValueBytes % 8 == 0, "kValueBytes must be a multiple of 8");
static_assert(kScaleBytes % 8 == 0, "kScaleBytes must be a multiple of 8");
static_assert((int64_t(1) << kPageBits) == kPageSize, "kPageBits must equal log2(kPageSize)");

struct PointerInfo {
  int64_t* value_ptr;
  int64_t* scale_ptr;
};

// Resolve the value/scale int64 pointers for a single token slot in a paged
// C4 cache. `index` is the logical token slot; the layout is page-padded.
inline PointerInfo get_pointer_paged(void* cache, int32_t index) {
  const int32_t page_num = index >> kPageBits;
  const int32_t page_offset = index & (kPageSize - 1);
  char* base = static_cast<char*>(cache) + static_cast<int64_t>(page_num) * kPageBytes;
  char* value_ptr = base + static_cast<int64_t>(page_offset) * kValueBytes;
  char* scale_ptr = base + kScaleOffset + static_cast<int64_t>(page_offset) * kScaleBytes;
  return {reinterpret_cast<int64_t*>(value_ptr), reinterpret_cast<int64_t*>(scale_ptr)};
}

// Copy one C4 item (value + scale) between page-padded caches, cooperatively
// across a sub-group. `lane_id`/`sg_size` are the sub-group local id and width;
// the strided loops make this correct for any Intel SIMD width (8/16/32).
inline void transfer_item(int lane_id, int sg_size, void* dst_cache, void* src_cache, int32_t dst_index, int32_t src_index) {
  const PointerInfo dst = get_pointer_paged(dst_cache, dst_index);
  const PointerInfo src = get_pointer_paged(src_cache, src_index);

  for (int j = lane_id; j < kValueWords; j += sg_size) {
    dst.value_ptr[j] = src.value_ptr[j];
  }
  for (int j = lane_id; j < kScaleWords; j += sg_size) {
    dst.scale_ptr[j] = src.scale_ptr[j];
  }
}

}  // namespace hisparse
}  // namespace sycl_kernel
}  // namespace sgl
