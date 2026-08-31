/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// KV-cache scatter/gather transfer kernels for XPU (SYCL port of sgl-kernel
// csrc/kvcacheio/transfer.cu).
//
// Layout conventions:
//   lf  = layer-first  [num_layers, num_tokens, item_size]
//   pf  = page-first   [num_tokens, num_layers, item_size]
//         (per-token addressing: base + token * layout_dim + layer * item_size,
//          where layout_dim = num_layers * item_size)
//   ph  = page-head    [num_pages, head_num, page_size, num_layers, head_dim]
//
// pf/ph pools live in pinned host memory (CPU-side eviction/disaggregation
// buffers); the device kernels address them directly over the fabric.  Both
// lf↔pf and lf↔ph layout conversions have on-device kernels here, matching the
// CUDA sgl-kernel API.  The pf_lf/lf_pf variants mirror CUDA transfer_kernel_impl
// with get_global_offset_pf; a Python copy_ fallback (transfer_kv_*_direct_*)
// remains available for the page_first_direct layout.

#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "Utils.h"
#include "comm/General.h"
#include "sgl_kernel_export.h"

// Sub-group size on Xe2/BMG.  WARP_SIZE in the CUDA source = 32;
// Xe2 sub-group size = 16.  All work-items within one sub-group cooperate
// to copy one token's KV entry.
//
// Cache hints: CUDA uses ld.global.nc (L1 bypass) + st.global.cg (L2 write).
// The XPU equivalent is ESIMD block_load/block_store with cache_hint::streaming.
// Not implemented here because: (a) ESIMD requires a different kernel model
// incompatible with sycl::nd_item; (b) at our working-set sizes (>8 MB) we
// already achieve 50–95% of DRAM peak without them.
static constexpr int64_t XPU_SG_SIZE = 16;

// ---------------------------------------------------------------------------
// Offset helpers – used only by TransferKVPageHeadKernel.
// The standard lf/lf_tbl addressing is inlined directly in TransferKVKernel.
// ---------------------------------------------------------------------------

// Layer-first with pointer table, per-head variant (for lf→ph source path).
static inline const char* get_offset_per_head_lf_tbl(
    const char* /*unused*/,
    const uintptr_t* tbl,
    int64_t layer_id,
    int64_t /*layout_dim*/,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t /*page_size*/) {
  return reinterpret_cast<const char*>(tbl[layer_id]) + page_id * item_size_bytes +
         item_size_bytes / head_num * head_id;
}

// Layer-first contiguous, per-head variant (for ph→lf destination path).
// layer_id is unused because callers pre-slice dst to the correct layer base
// (e.g. dst_k_pool_kernel_ptrs[layer_idx]), so layer_dim is always 0 here.
// This matches the CUDA get_global_offset_per_head_lf with layer_dim=0.
static inline char* get_offset_per_head_lf_mut(
    char* base,
    const uintptr_t* /*unused*/,
    int64_t layer_id,
    int64_t layer_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t /*page_size*/) {
  return base + layer_id * layer_dim + page_id * item_size_bytes + item_size_bytes / head_num * head_id;
}

// Page-head layout: [num_pages, head_num, page_size, num_layers, head_dim]
// page_id here is a flat token index.
static inline char* get_offset_ph_mut(
    char* base,
    const uintptr_t* /*unused*/,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t page_size) {
  const int64_t head_dim_bytes = item_size_bytes / head_num;
  return base + (page_id / page_size) * page_size * page_dim +  // page bucket
         (page_dim / head_num) * head_id * page_size +          // head dim
         (page_id % page_size) * page_dim / head_num +          // slot within page
         layer_id * head_dim_bytes;                             // layer slot
}

// Page-head layout, const variant (for ph→lf source path).
static inline const char* get_offset_ph(
    const char* base,
    const uintptr_t* /*unused*/,
    int64_t layer_id,
    int64_t page_dim,
    int64_t page_id,
    int64_t item_size_bytes,
    int64_t head_id,
    int64_t head_num,
    int64_t page_size) {
  const int64_t head_dim_bytes = item_size_bytes / head_num;
  return base + (page_id / page_size) * page_size * page_dim + (page_dim / head_num) * head_id * page_size +
         (page_id % page_size) * page_dim / head_num + layer_id * head_dim_bytes;
}

// ---------------------------------------------------------------------------
// Kernel functors
// ---------------------------------------------------------------------------

// Standard transfer kernel (lf↔lf and lf_tbl↔lf_tbl).
// One sub-group (XPU_SG_SIZE work-items) handles one token per iteration.
// IsMLA=true: only K is transferred (no separate V tensor).
template <bool IsMLA>
struct TransferKVKernel {
  void operator()(sycl::nd_item<1> item) const {
    const int64_t lane = static_cast<int64_t>(item.get_local_id(0)) % XPU_SG_SIZE;
    const int64_t sg_id =
        static_cast<int64_t>(item.get_local_id(0)) / XPU_SG_SIZE +
        static_cast<int64_t>(item.get_group(0)) * (static_cast<int64_t>(item.get_local_range(0)) / XPU_SG_SIZE);

    for (int64_t i = 0; i < items_per_sg_; ++i) {
      const int64_t item_id = sg_id * items_per_sg_ + i;
      if (item_id >= num_items_) break;

      const int64_t src_page = src_indices_[item_id];
      const int64_t dst_page = dst_indices_[item_id];

      for (int64_t layer = start_layer_; layer < start_layer_ + num_layers_; ++layer) {
        const char* src_k_ptr = src_k_base_ == nullptr
                                    ? reinterpret_cast<const char*>(src_k_tbl_[layer]) + src_page * item_size_
                                    : src_k_base_ + src_page * item_size_;
        char* dst_k_ptr = dst_k_base_ == nullptr ? reinterpret_cast<char*>(dst_k_tbl_[layer]) + dst_page * item_size_
                                                 : dst_k_base_ + dst_page * item_size_;

        const int64_t chunks = item_size_ / static_cast<int64_t>(sizeof(uint64_t));
        const auto* src64 = reinterpret_cast<const uint64_t*>(src_k_ptr);
        auto* dst64 = reinterpret_cast<uint64_t*>(dst_k_ptr);

        if constexpr (!IsMLA) {
          // Interleave K and V loads/stores so both streams are in-flight
          // simultaneously, hiding load latency across the two independent
          // address streams.
          const char* src_v_ptr = src_v_base_ == nullptr
                                      ? reinterpret_cast<const char*>(src_v_tbl_[layer]) + src_page * item_size_
                                      : src_v_base_ + src_page * item_size_;
          char* dst_v_ptr = dst_v_base_ == nullptr ? reinterpret_cast<char*>(dst_v_tbl_[layer]) + dst_page * item_size_
                                                   : dst_v_base_ + dst_page * item_size_;

          const auto* sv64 = reinterpret_cast<const uint64_t*>(src_v_ptr);
          auto* dv64 = reinterpret_cast<uint64_t*>(dst_v_ptr);
          for (int64_t j = lane; j < chunks; j += XPU_SG_SIZE) {
            const uint64_t k_val = src64[j];
            const uint64_t v_val = sv64[j];
            dst64[j] = k_val;
            dv64[j] = v_val;
          }
        } else {
          for (int64_t j = lane; j < chunks; j += XPU_SG_SIZE) {
            dst64[j] = src64[j];
          }
        }
      }
    }
  }

  // K source: either a flat base pointer (single-layer lf) or a layer table.
  const char* src_k_base_;
  char* dst_k_base_;
  const char* src_v_base_;
  char* dst_v_base_;
  const uintptr_t* src_k_tbl_;
  const uintptr_t* dst_k_tbl_;
  const uintptr_t* src_v_tbl_;
  const uintptr_t* dst_v_tbl_;
  const int64_t* src_indices_;
  const int64_t* dst_indices_;
  int64_t start_layer_;
  int64_t num_layers_;
  int64_t num_items_;
  int64_t items_per_sg_;
  int64_t item_size_;
};

// Page-first transfer kernel (pf↔lf).
// pf layout:  [num_pages, num_layers, page_size, item_size], addressed as
//   base + page_id * layout_dim + layer_id * item_size
// lf side is the flat/tabled layer-first addressing (base + page_id * item_size,
// or a per-layer pointer table).  Direction is fixed at compile time:
//   IsPfToLf=true : src is pf, dst is lf   (transfer_kv_*_pf_lf)
//   IsPfToLf=false: src is lf, dst is pf   (transfer_kv_*_lf_pf)
// IsMLA=true: only K is transferred (no separate V tensor).
// Mirrors CUDA transfer_kernel_impl with get_global_offset_pf on the pf side.
template <bool IsMLA, bool IsPfToLf>
struct TransferKVPageFirstKernel {
  // pf side: base + page * layout_dim + layer * item_size.
  static inline const char*
  pf_offset(const char* base, int64_t layer, int64_t page, int64_t item_size, int64_t layout_dim) {
    return base + page * layout_dim + layer * item_size;
  }
  // lf side: flat base (base + page * item_size) or per-layer table.
  static inline const char*
  lf_offset(const char* base, const uintptr_t* tbl, int64_t layer, int64_t page, int64_t item_size) {
    return base == nullptr ? reinterpret_cast<const char*>(tbl[layer]) + page * item_size : base + page * item_size;
  }

  void operator()(sycl::nd_item<1> item) const {
    const int64_t lane = static_cast<int64_t>(item.get_local_id(0)) % XPU_SG_SIZE;
    const int64_t sg_id =
        static_cast<int64_t>(item.get_local_id(0)) / XPU_SG_SIZE +
        static_cast<int64_t>(item.get_group(0)) * (static_cast<int64_t>(item.get_local_range(0)) / XPU_SG_SIZE);

    const int64_t chunks = item_size_ / static_cast<int64_t>(sizeof(uint64_t));

    for (int64_t i = 0; i < items_per_sg_; ++i) {
      const int64_t item_id = sg_id * items_per_sg_ + i;
      if (item_id >= num_items_) break;

      const int64_t src_page = src_indices_[item_id];
      const int64_t dst_page = dst_indices_[item_id];

      for (int64_t layer = start_layer_; layer < start_layer_ + num_layers_; ++layer) {
        const char* src_k_ptr;
        char* dst_k_ptr;
        if constexpr (IsPfToLf) {
          src_k_ptr = pf_offset(src_k_base_, layer, src_page, item_size_, src_layout_dim_);
          dst_k_ptr = const_cast<char*>(lf_offset(dst_k_base_, dst_k_tbl_, layer, dst_page, item_size_));
        } else {
          src_k_ptr = lf_offset(src_k_base_, src_k_tbl_, layer, src_page, item_size_);
          dst_k_ptr = const_cast<char*>(pf_offset(dst_k_base_, layer, dst_page, item_size_, dst_layout_dim_));
        }

        const auto* src64 = reinterpret_cast<const uint64_t*>(src_k_ptr);
        auto* dst64 = reinterpret_cast<uint64_t*>(dst_k_ptr);

        if constexpr (!IsMLA) {
          const char* src_v_ptr;
          char* dst_v_ptr;
          if constexpr (IsPfToLf) {
            src_v_ptr = pf_offset(src_v_base_, layer, src_page, item_size_, src_layout_dim_);
            dst_v_ptr = const_cast<char*>(lf_offset(dst_v_base_, dst_v_tbl_, layer, dst_page, item_size_));
          } else {
            src_v_ptr = lf_offset(src_v_base_, src_v_tbl_, layer, src_page, item_size_);
            dst_v_ptr = const_cast<char*>(pf_offset(dst_v_base_, layer, dst_page, item_size_, dst_layout_dim_));
          }
          const auto* sv64 = reinterpret_cast<const uint64_t*>(src_v_ptr);
          auto* dv64 = reinterpret_cast<uint64_t*>(dst_v_ptr);
          for (int64_t j = lane; j < chunks; j += XPU_SG_SIZE) {
            const uint64_t k_val = src64[j];
            const uint64_t v_val = sv64[j];
            dst64[j] = k_val;
            dv64[j] = v_val;
          }
        } else {
          for (int64_t j = lane; j < chunks; j += XPU_SG_SIZE) {
            dst64[j] = src64[j];
          }
        }
      }
    }
  }

  // pf side uses a flat base pointer; lf side may use a base or a layer table.
  const char* src_k_base_;
  char* dst_k_base_;
  const char* src_v_base_;
  char* dst_v_base_;
  const uintptr_t* src_k_tbl_;
  const uintptr_t* dst_k_tbl_;
  const uintptr_t* src_v_tbl_;
  const uintptr_t* dst_v_tbl_;
  const int64_t* src_indices_;
  const int64_t* dst_indices_;
  int64_t start_layer_;
  int64_t num_layers_;
  int64_t num_items_;
  int64_t items_per_sg_;
  int64_t item_size_;
  int64_t src_layout_dim_;
  int64_t dst_layout_dim_;
};

// Page-head transfer kernel: loops over heads because each head's data is
// non-contiguous in the page-head layout.
// Direction is fixed to lf→ph (IsLfToPh=true) or ph→lf (false).
template <bool IsLfToPh>
struct TransferKVPageHeadKernel {
  void operator()(sycl::nd_item<1> item) const {
    const int64_t lane = static_cast<int64_t>(item.get_local_id(0)) % XPU_SG_SIZE;
    const int64_t sg_id =
        static_cast<int64_t>(item.get_local_id(0)) / XPU_SG_SIZE +
        static_cast<int64_t>(item.get_group(0)) * (static_cast<int64_t>(item.get_local_range(0)) / XPU_SG_SIZE);

    const int64_t head_dim_bytes = item_size_ / head_num_;

    for (int64_t i = 0; i < items_per_sg_; ++i) {
      const int64_t item_id = sg_id * items_per_sg_ + i;
      if (item_id >= num_items_) break;

      const int64_t src_page = src_indices_[item_id];
      const int64_t dst_page = dst_indices_[item_id];

      for (int64_t layer = start_layer_; layer < start_layer_ + num_layers_; ++layer) {
        for (int64_t head = 0; head < head_num_; ++head) {
          const char* sk;
          char* dk;
          const char* sv;
          char* dv;

          if constexpr (IsLfToPh) {
            // src: lf_tbl per-head  dst: ph
            sk = get_offset_per_head_lf_tbl(
                nullptr, src_k_tbl_, layer, 0, src_page, item_size_, head, head_num_, page_size_);
            dk = get_offset_ph_mut(
                dst_k_base_, nullptr, layer, dst_layout_dim_, dst_page, item_size_, head, head_num_, page_size_);
            sv = get_offset_per_head_lf_tbl(
                nullptr, src_v_tbl_, layer, 0, src_page, item_size_, head, head_num_, page_size_);
            dv = get_offset_ph_mut(
                dst_v_base_, nullptr, layer, dst_layout_dim_, dst_page, item_size_, head, head_num_, page_size_);
          } else {
            // src: ph  dst: lf per-head
            sk = get_offset_ph(
                src_k_base_, nullptr, layer, src_layout_dim_, src_page, item_size_, head, head_num_, page_size_);
            dk = get_offset_per_head_lf_mut(
                dst_k_base_, nullptr, layer, 0, dst_page, item_size_, head, head_num_, page_size_);
            sv = get_offset_ph(
                src_v_base_, nullptr, layer, src_layout_dim_, src_page, item_size_, head, head_num_, page_size_);
            dv = get_offset_per_head_lf_mut(
                dst_v_base_, nullptr, layer, 0, dst_page, item_size_, head, head_num_, page_size_);
          }

          const int64_t chunks = head_dim_bytes / static_cast<int64_t>(sizeof(uint64_t));
          const auto* sk64 = reinterpret_cast<const uint64_t*>(sk);
          auto* dk64 = reinterpret_cast<uint64_t*>(dk);
          const auto* sv64 = reinterpret_cast<const uint64_t*>(sv);
          auto* dv64 = reinterpret_cast<uint64_t*>(dv);
          for (int64_t j = lane; j < chunks; j += XPU_SG_SIZE) {
            dk64[j] = sk64[j];
            dv64[j] = sv64[j];
          }
        }
      }
    }
  }

  const char* src_k_base_;
  char* dst_k_base_;
  const char* src_v_base_;
  char* dst_v_base_;
  const uintptr_t* src_k_tbl_;
  const uintptr_t* dst_k_tbl_;
  const uintptr_t* src_v_tbl_;
  const uintptr_t* dst_v_tbl_;
  const int64_t* src_indices_;
  const int64_t* dst_indices_;
  int64_t start_layer_;
  int64_t num_layers_;
  int64_t num_items_;
  int64_t items_per_sg_;
  int64_t item_size_;
  int64_t src_layout_dim_;
  int64_t dst_layout_dim_;
  int64_t page_size_;
  int64_t head_num_;
};

// ---------------------------------------------------------------------------
// Launcher helpers
// ---------------------------------------------------------------------------

static int64_t div_up(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// Validate a per-layer pointer table: it must hold exactly num_layers entries
// and be a UInt64 tensor (each element is a raw data_ptr() value).  Reading it
// as a uintptr_t table with a mismatched length or dtype would dereference
// invalid addresses in the kernel.
static void check_layer_ptr_table(const at::Tensor& tbl, int64_t num_layers, const char* name) {
  TORCH_CHECK(tbl.scalar_type() == at::kUInt64, name, " must be a uint64 pointer table");
  TORCH_CHECK(tbl.numel() == num_layers, name, " must have num_layers entries");
}

// Launch the standard (non-page-head) transfer kernel.
// Uses TransferKVKernel<IsMLA>.
template <bool IsMLA>
static void launch_transfer_kv(
    const void* src_k,
    void* dst_k,
    const void* src_v,
    void* dst_v,
    const uintptr_t* src_k_tbl,
    const uintptr_t* dst_k_tbl,
    const uintptr_t* src_v_tbl,
    const uintptr_t* dst_v_tbl,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t start_layer,
    int64_t num_layers,
    int64_t item_size,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  TORCH_CHECK(item_size % 8 == 0, "item_size must be divisible by 8");
  TORCH_CHECK(sgs_per_wg > 0, "sgs_per_wg must be positive");
  TORCH_CHECK(src_indices.scalar_type() == at::kLong, "src_indices must be int64");
  TORCH_CHECK(dst_indices.scalar_type() == at::kLong, "dst_indices must be int64");
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "index count mismatch");

  const int64_t num_items = src_indices.numel();
  if (num_items == 0) return;  // nothing to transfer; avoids div-by-zero below

  // block_quota fixes the total sub-group pool (block_quota * sgs_per_wg
  // sub-groups); num_wgs then scales with the token count up to that pool.
  // A fixed pool of 16*32=512 sub-groups fills all 20 Xe-cores on B580 across
  // the whole token-count range (measured: 5-8x faster than an N-derived quota
  // at N<=1024, which starves the GPU to 1-2 work-groups).  block_quota<=0
  // falls back to this default.
  const int64_t effective_bq = block_quota > 0 ? block_quota : 16;
  const int64_t total_sgs = effective_bq * sgs_per_wg;
  const int64_t items_per_sg = div_up(num_items, total_sgs);
  const int64_t num_wgs = div_up(num_items, items_per_sg * sgs_per_wg);
  const int64_t wg_size = sgs_per_wg * XPU_SG_SIZE;

  TransferKVKernel<IsMLA> kernel{
      .src_k_base_ = static_cast<const char*>(src_k),
      .dst_k_base_ = static_cast<char*>(dst_k),
      .src_v_base_ = static_cast<const char*>(src_v),
      .dst_v_base_ = static_cast<char*>(dst_v),
      .src_k_tbl_ = src_k_tbl,
      .dst_k_tbl_ = dst_k_tbl,
      .src_v_tbl_ = src_v_tbl,
      .dst_v_tbl_ = dst_v_tbl,
      .src_indices_ = src_indices.data_ptr<int64_t>(),
      .dst_indices_ = dst_indices.data_ptr<int64_t>(),
      .start_layer_ = start_layer,
      .num_layers_ = num_layers,
      .num_items_ = num_items,
      .items_per_sg_ = items_per_sg,
      .item_size_ = item_size,
  };

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<decltype(kernel)>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(num_wgs * wg_size)), sycl::range<1>(static_cast<size_t>(wg_size))),
        kernel);
  };
  dpcppGetCurrentQueue().submit(cgf);
}

// Launch the page-first transfer kernel.  Uses TransferKVPageFirstKernel.
// The pf side is always a flat base pointer; the lf side may be a flat base
// (single layer) or a per-layer pointer table (all layers).
template <bool IsMLA, bool IsPfToLf>
static void launch_transfer_kv_page_first(
    const void* src_k,
    void* dst_k,
    const void* src_v,
    void* dst_v,
    const uintptr_t* src_k_tbl,
    const uintptr_t* dst_k_tbl,
    const uintptr_t* src_v_tbl,
    const uintptr_t* dst_v_tbl,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t start_layer,
    int64_t num_layers,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  TORCH_CHECK(item_size % 8 == 0, "item_size must be divisible by 8");
  TORCH_CHECK(sgs_per_wg > 0, "sgs_per_wg must be positive");
  TORCH_CHECK(src_indices.scalar_type() == at::kLong, "src_indices must be int64");
  TORCH_CHECK(dst_indices.scalar_type() == at::kLong, "dst_indices must be int64");
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "index count mismatch");

  const int64_t num_items = src_indices.numel();
  if (num_items == 0) return;  // nothing to transfer; avoids div-by-zero below

  const int64_t effective_bq = block_quota > 0 ? block_quota : 16;
  const int64_t total_sgs = effective_bq * sgs_per_wg;
  const int64_t items_per_sg = div_up(num_items, total_sgs);
  const int64_t num_wgs = div_up(num_items, items_per_sg * sgs_per_wg);
  const int64_t wg_size = sgs_per_wg * XPU_SG_SIZE;

  TransferKVPageFirstKernel<IsMLA, IsPfToLf> kernel{
      .src_k_base_ = static_cast<const char*>(src_k),
      .dst_k_base_ = static_cast<char*>(dst_k),
      .src_v_base_ = static_cast<const char*>(src_v),
      .dst_v_base_ = static_cast<char*>(dst_v),
      .src_k_tbl_ = src_k_tbl,
      .dst_k_tbl_ = dst_k_tbl,
      .src_v_tbl_ = src_v_tbl,
      .dst_v_tbl_ = dst_v_tbl,
      .src_indices_ = src_indices.data_ptr<int64_t>(),
      .dst_indices_ = dst_indices.data_ptr<int64_t>(),
      .start_layer_ = start_layer,
      .num_layers_ = num_layers,
      .num_items_ = num_items,
      .items_per_sg_ = items_per_sg,
      .item_size_ = item_size,
      .src_layout_dim_ = src_layout_dim,
      .dst_layout_dim_ = dst_layout_dim,
  };

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<decltype(kernel)>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(num_wgs * wg_size)), sycl::range<1>(static_cast<size_t>(wg_size))),
        kernel);
  };
  dpcppGetCurrentQueue().submit(cgf);
}

template <bool IsLfToPh>
static void launch_transfer_kv_page_head(
    const void* src_k,
    void* dst_k,
    const void* src_v,
    void* dst_v,
    const uintptr_t* src_k_tbl,
    const uintptr_t* dst_k_tbl,
    const uintptr_t* src_v_tbl,
    const uintptr_t* dst_v_tbl,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t num_layers,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t dst_layout_dim,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t sgs_per_wg,
    int64_t start_layer = 0) {
  TORCH_CHECK(item_size % 8 == 0, "item_size must be divisible by 8");
  TORCH_CHECK(head_num > 0, "head_num must be positive");
  TORCH_CHECK(item_size % head_num == 0, "item_size must be divisible by head_num");
  TORCH_CHECK(sgs_per_wg > 0, "sgs_per_wg must be positive");
  TORCH_CHECK(src_indices.scalar_type() == at::kLong, "src_indices must be int64");
  TORCH_CHECK(dst_indices.scalar_type() == at::kLong, "dst_indices must be int64");
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "index count mismatch");

  const int64_t num_items = src_indices.numel();
  if (num_items == 0) return;  // nothing to transfer; avoids div-by-zero below

  const int64_t effective_bq = block_quota > 0 ? block_quota : 16;
  const int64_t total_sgs = effective_bq * sgs_per_wg;
  const int64_t items_per_sg = div_up(num_items, total_sgs);
  const int64_t num_wgs = div_up(num_items, items_per_sg * sgs_per_wg);
  const int64_t wg_size = sgs_per_wg * XPU_SG_SIZE;

  TransferKVPageHeadKernel<IsLfToPh> kernel{
      .src_k_base_ = static_cast<const char*>(src_k),
      .dst_k_base_ = static_cast<char*>(dst_k),
      .src_v_base_ = static_cast<const char*>(src_v),
      .dst_v_base_ = static_cast<char*>(dst_v),
      .src_k_tbl_ = src_k_tbl,
      .dst_k_tbl_ = dst_k_tbl,
      .src_v_tbl_ = src_v_tbl,
      .dst_v_tbl_ = dst_v_tbl,
      .src_indices_ = src_indices.data_ptr<int64_t>(),
      .dst_indices_ = dst_indices.data_ptr<int64_t>(),
      .start_layer_ = start_layer,
      .num_layers_ = num_layers,
      .num_items_ = num_items,
      .items_per_sg_ = items_per_sg,
      .item_size_ = item_size,
      .src_layout_dim_ = src_layout_dim,
      .dst_layout_dim_ = dst_layout_dim,
      .page_size_ = page_size,
      .head_num_ = head_num,
  };

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<decltype(kernel)>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(num_wgs * wg_size)), sycl::range<1>(static_cast<size_t>(wg_size))),
        kernel);
  };
  dpcppGetCurrentQueue().submit(cgf);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Single-layer, lf→lf, K+V.
SGL_KERNEL_EXPORT void transfer_kv_per_layer(
    const at::Tensor& src_k,
    at::Tensor& dst_k,
    const at::Tensor& src_v,
    at::Tensor& dst_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  launch_transfer_kv<false>(
      src_k.data_ptr(),
      dst_k.data_ptr(),
      src_v.data_ptr(),
      dst_v.data_ptr(),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      src_indices,
      dst_indices,
      0,
      1,
      item_size,
      block_quota,
      sgs_per_wg);
}

// Single-layer, lf→lf, K only (MLA).
SGL_KERNEL_EXPORT void transfer_kv_per_layer_mla(
    const at::Tensor& src,
    at::Tensor& dst,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  launch_transfer_kv<true>(
      src.data_ptr(),
      dst.data_ptr(),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      src_indices,
      dst_indices,
      0,
      1,
      item_size,
      block_quota,
      sgs_per_wg);
}

// All-layers, lf_tbl→lf_tbl, K+V.
SGL_KERNEL_EXPORT void transfer_kv_all_layer(
    const at::Tensor& src_k_layers,
    const at::Tensor& dst_k_layers,
    const at::Tensor& src_v_layers,
    const at::Tensor& dst_v_layers,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  check_layer_ptr_table(src_k_layers, num_layers, "src_k_layers");
  check_layer_ptr_table(dst_k_layers, num_layers, "dst_k_layers");
  check_layer_ptr_table(src_v_layers, num_layers, "src_v_layers");
  check_layer_ptr_table(dst_v_layers, num_layers, "dst_v_layers");
  launch_transfer_kv<false>(
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      src_k_layers.data_ptr<uintptr_t>(),
      dst_k_layers.data_ptr<uintptr_t>(),
      src_v_layers.data_ptr<uintptr_t>(),
      dst_v_layers.data_ptr<uintptr_t>(),
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      block_quota,
      sgs_per_wg);
}

// All-layers, lf_tbl→lf_tbl, K only (MLA).
SGL_KERNEL_EXPORT void transfer_kv_all_layer_mla(
    const at::Tensor& src_layers,
    const at::Tensor& dst_layers,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  check_layer_ptr_table(src_layers, num_layers, "src_layers");
  check_layer_ptr_table(dst_layers, num_layers, "dst_layers");
  launch_transfer_kv<true>(
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      src_layers.data_ptr<uintptr_t>(),
      dst_layers.data_ptr<uintptr_t>(),
      nullptr,
      nullptr,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      block_quota,
      sgs_per_wg);
}

// All-layers, lf_tbl→ph (page-head destination).
SGL_KERNEL_EXPORT void transfer_kv_all_layer_lf_ph(
    const at::Tensor& src_k_layers,
    at::Tensor& dst_k,
    const at::Tensor& src_v_layers,
    at::Tensor& dst_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  check_layer_ptr_table(src_k_layers, num_layers, "src_k_layers");
  check_layer_ptr_table(src_v_layers, num_layers, "src_v_layers");
  launch_transfer_kv_page_head<true>(
      nullptr,
      dst_k.data_ptr(),
      nullptr,
      dst_v.data_ptr(),
      src_k_layers.data_ptr<uintptr_t>(),
      nullptr,
      src_v_layers.data_ptr<uintptr_t>(),
      nullptr,
      src_indices,
      dst_indices,
      num_layers,
      item_size,
      0,
      dst_layout_dim,
      page_size,
      head_num,
      block_quota,
      sgs_per_wg);
}

// Single-layer, ph→lf.
// layer_id: which layer slot to read from the page-head source layout.
// dst_k/dst_v already point to the correct layer's contiguous buffer.
SGL_KERNEL_EXPORT void transfer_kv_per_layer_ph_lf(
    const at::Tensor& src_k,
    at::Tensor& dst_k,
    const at::Tensor& src_v,
    at::Tensor& dst_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  // We launch with start_layer=layer_id, num_layers=1 so the ph offset
  // function reads the correct layer slot from the source.
  launch_transfer_kv_page_head<false>(
      src_k.data_ptr(),
      dst_k.data_ptr(),
      src_v.data_ptr(),
      dst_v.data_ptr(),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      src_indices,
      dst_indices,
      1,
      item_size,
      src_layout_dim,
      0,
      page_size,
      head_num,
      block_quota,
      sgs_per_wg,
      layer_id);
}

// Single-layer, pf→lf, K+V.
// src_k/src_v are the page-first pool base pointers; layer_id selects the layer
// slot within each page.  dst_k/dst_v are the contiguous per-layer buffers.
SGL_KERNEL_EXPORT void transfer_kv_per_layer_pf_lf(
    const at::Tensor& src_k,
    at::Tensor& dst_k,
    const at::Tensor& src_v,
    at::Tensor& dst_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  launch_transfer_kv_page_first<false, true>(
      src_k.data_ptr(),
      dst_k.data_ptr(),
      src_v.data_ptr(),
      dst_v.data_ptr(),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      src_indices,
      dst_indices,
      layer_id,
      1,
      item_size,
      src_layout_dim,
      0,
      block_quota,
      sgs_per_wg);
}

// All-layers, lf_tbl→pf, K+V.
SGL_KERNEL_EXPORT void transfer_kv_all_layer_lf_pf(
    const at::Tensor& src_k_layers,
    at::Tensor& dst_k,
    const at::Tensor& src_v_layers,
    at::Tensor& dst_v,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  check_layer_ptr_table(src_k_layers, num_layers, "src_k_layers");
  check_layer_ptr_table(src_v_layers, num_layers, "src_v_layers");
  launch_transfer_kv_page_first<false, false>(
      nullptr,
      dst_k.data_ptr(),
      nullptr,
      dst_v.data_ptr(),
      src_k_layers.data_ptr<uintptr_t>(),
      nullptr,
      src_v_layers.data_ptr<uintptr_t>(),
      nullptr,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      dst_layout_dim,
      block_quota,
      sgs_per_wg);
}

// Single-layer, pf→lf, K only (MLA).
SGL_KERNEL_EXPORT void transfer_kv_per_layer_mla_pf_lf(
    const at::Tensor& src,
    at::Tensor& dst,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  launch_transfer_kv_page_first<true, true>(
      src.data_ptr(),
      dst.data_ptr(),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      src_indices,
      dst_indices,
      layer_id,
      1,
      item_size,
      src_layout_dim,
      0,
      block_quota,
      sgs_per_wg);
}

// All-layers, lf_tbl→pf, K only (MLA).
SGL_KERNEL_EXPORT void transfer_kv_all_layer_mla_lf_pf(
    const at::Tensor& src_layers,
    at::Tensor& dst,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  check_layer_ptr_table(src_layers, num_layers, "src_layers");
  launch_transfer_kv_page_first<true, false>(
      nullptr,
      dst.data_ptr(),
      nullptr,
      nullptr,
      src_layers.data_ptr<uintptr_t>(),
      nullptr,
      nullptr,
      nullptr,
      src_indices,
      dst_indices,
      0,
      num_layers,
      item_size,
      0,
      dst_layout_dim,
      block_quota,
      sgs_per_wg);
}

// ===========================================================================
// Mamba HiCache transfer (SYCL port of sgl-kernel csrc/kvcacheio/transfer_mamba.cuh,
// i.e. TransferMambaKernel::run_pf_lf / ::run_lf_pf)
//
//   load  (pf→lf): dst[dst_idx[i]]        ← src[src_idx[i], layer_id]
//   backup(lf→pf): dst[dst_idx[i], layer] ← src[layer, src_idx[i]]  (all layers)
//
// Layouts (all contiguous):
//   page_first  : [pool, num_layers, item]  per-page stride = num_layers * item
//   layer_first : [num_layers, pool, item]  per-layer stride = pool * item
//   single-layer: [pool, item]              per-page stride = item
//
// Unlike the KV kernels above (which hard-require item_size % 8 == 0), the copy
// unit is chosen at runtime from pointer and stride alignment, so odd Mamba
// state sizes stay correct.
//
// backup takes the contiguous layer_first tensor plus its per-layer byte stride
// where CUDA takes a uint64 array of per-layer pointers: dereferencing a
// host-built pointer array is not portable across SYCL runtimes.
// ===========================================================================

static constexpr int64_t kMambaWgSize = 256;
// Split an item across work-groups only while each still has enough to amortize
// its launch: at least this many units per work-item.
static constexpr int64_t kMambaMinUnitsPerThread = 8;
// Measured on BMG (160 CUs): 1 leaves the large-item copies short of peak, 4 and
// 8 are within noise, so 4 fills the device while keeping chunks as large (and
// as streaming-friendly) as possible.
static constexpr int64_t kMambaGroupsPerCu = 4;

// Widest power-of-two copy unit (≤ 8B) dividing every address and byte stride,
// so every access stays naturally aligned.  OR-ing the quantities makes the
// lowest set bit the minimum over their individual lowest set bits, which is
// exactly the shared 2-power.
static inline int64_t mamba_copy_width(std::initializer_list<int64_t> byte_quantities) {
  int64_t combined = 0;
  for (int64_t q : byte_quantities)
    combined |= q;
  for (int64_t w : {8, 4, 2}) {
    if ((combined & (w - 1)) == 0) return w;
  }
  return 1;
}

// One work-group per copy unit (item for load, item × layer for backup) leaves
// most of the device idle when items are few and large -- 16 pages of 64Ki
// elements is 16 work-groups on a 160-CU GPU, measurably slower than a plain
// PyTorch indexed copy -- so each unit is split into `chunks` contiguous ranges
// until the grid fills the device.  With enough units to fill it (the common
// HiCache case) this returns chunks == 1.
struct MambaChunking {
  int64_t chunks;
  int64_t units_per_chunk;
};

static MambaChunking mamba_plan_chunks(int64_t num_copy_units, int64_t item_units) {
  const int64_t max_chunks = std::max<int64_t>(1, item_units / (kMambaWgSize * kMambaMinUnitsPerThread));
  const int64_t target_wgs = kMambaGroupsPerCu * dpcppMaxComputeUnitSize();
  const int64_t wanted = std::clamp<int64_t>(div_up(target_wgs, num_copy_units), 1, max_chunks);
  if (wanted == 1) return {1, item_units};

  // Round up to a whole work-group stride so every chunk boundary stays
  // cache-line aligned, then recompute the count from the rounded size so no
  // work-group is launched for an entirely empty range.
  const int64_t units_per_chunk = div_up(div_up(item_units, wanted), kMambaWgSize) * kMambaWgSize;
  return {div_up(item_units, units_per_chunk), units_per_chunk};
}

// Load: page_first → single-layer.  Grid = (num_items) × (chunks * kMambaWgSize).
// 2-D rather than flat-and-divided so the hardware supplies both indices:
// recovering (item, chunk) from a flat group id costs an int64 div+mod per
// work-item, which measurably slowed the small-item shapes.
template <typename CopyT>
struct TransferMambaLoadKernel {
  [[sycl::reqd_sub_group_size(XPU_SG_SIZE)]] void operator()(sycl::nd_item<2> item) const {
    const int64_t begin = static_cast<int64_t>(item.get_group(1)) * units_per_chunk_;
    if (begin >= item_units_) return;  // only reachable for a tail chunk
    const int64_t end = sycl::min(begin + units_per_chunk_, item_units_);

    const int64_t item_id = static_cast<int64_t>(item.get_group(0));
    const CopyT* src = src_ + src_indices_[item_id] * src_page_stride_ + layer_id_ * item_units_;
    CopyT* dst = dst_ + dst_indices_[item_id] * item_units_;

    const int64_t stride = static_cast<int64_t>(item.get_local_range(1));
    for (int64_t i = begin + static_cast<int64_t>(item.get_local_id(1)); i < end; i += stride) {
      dst[i] = src[i];
    }
  }

  const CopyT* src_;
  CopyT* dst_;
  const int64_t* src_indices_;
  const int64_t* dst_indices_;
  int64_t layer_id_;
  int64_t item_units_;
  int64_t src_page_stride_;
  int64_t units_per_chunk_;
};

// Backup: layer_first → page_first, all layers.
// Grid = (num_items) × (num_layers) × (chunks * kMambaWgSize), 3-D for the same
// reason the load grid is 2-D.
template <typename CopyT>
struct TransferMambaBackupKernel {
  [[sycl::reqd_sub_group_size(XPU_SG_SIZE)]] void operator()(sycl::nd_item<3> item) const {
    const int64_t begin = static_cast<int64_t>(item.get_group(2)) * units_per_chunk_;
    if (begin >= item_units_) return;  // only reachable for a tail chunk
    const int64_t end = sycl::min(begin + units_per_chunk_, item_units_);

    const int64_t item_id = static_cast<int64_t>(item.get_group(0));
    const int64_t layer = static_cast<int64_t>(item.get_group(1));
    const CopyT* src = src_ + layer * src_layer_stride_ + src_indices_[item_id] * item_units_;
    CopyT* dst = dst_ + dst_indices_[item_id] * dst_page_stride_ + layer * item_units_;

    const int64_t stride = static_cast<int64_t>(item.get_local_range(2));
    for (int64_t i = begin + static_cast<int64_t>(item.get_local_id(2)); i < end; i += stride) {
      dst[i] = src[i];
    }
  }

  const CopyT* src_;
  CopyT* dst_;
  const int64_t* src_indices_;
  const int64_t* dst_indices_;
  int64_t item_units_;
  int64_t src_layer_stride_;
  int64_t dst_page_stride_;
  int64_t units_per_chunk_;
};

template <typename CopyT>
static void launch_mamba_load(
    const void* src,
    void* dst,
    const int64_t* src_indices,
    const int64_t* dst_indices,
    int64_t layer_id,
    int64_t item_bytes,
    int64_t src_page_stride_bytes,
    int64_t num_items) {
  constexpr int64_t kUnit = static_cast<int64_t>(sizeof(CopyT));
  const int64_t item_units = item_bytes / kUnit;
  // Plain locals rather than a structured binding: the SYCL device pass compiles
  // as C++17, where capturing one in the command-group lambda is an extension.
  const MambaChunking plan = mamba_plan_chunks(num_items, item_units);
  const int64_t chunks = plan.chunks;

  TransferMambaLoadKernel<CopyT> kernel{
      .src_ = static_cast<const CopyT*>(src),
      .dst_ = static_cast<CopyT*>(dst),
      .src_indices_ = src_indices,
      .dst_indices_ = dst_indices,
      .layer_id_ = layer_id,
      .item_units_ = item_units,
      .src_page_stride_ = src_page_stride_bytes / kUnit,
      .units_per_chunk_ = plan.units_per_chunk,
  };

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<decltype(kernel)>(
        sycl::nd_range<2>(
            sycl::range<2>(static_cast<size_t>(num_items), static_cast<size_t>(chunks * kMambaWgSize)),
            sycl::range<2>(1, static_cast<size_t>(kMambaWgSize))),
        kernel);
  };
  dpcppGetCurrentQueue().submit(cgf);
}

template <typename CopyT>
static void launch_mamba_backup(
    const void* src,
    void* dst,
    const int64_t* src_indices,
    const int64_t* dst_indices,
    int64_t item_bytes,
    int64_t src_layer_stride_bytes,
    int64_t dst_page_stride_bytes,
    int64_t num_items,
    int64_t num_layers) {
  constexpr int64_t kUnit = static_cast<int64_t>(sizeof(CopyT));
  const int64_t item_units = item_bytes / kUnit;
  const MambaChunking plan = mamba_plan_chunks(num_items * num_layers, item_units);
  const int64_t chunks = plan.chunks;

  TransferMambaBackupKernel<CopyT> kernel{
      .src_ = static_cast<const CopyT*>(src),
      .dst_ = static_cast<CopyT*>(dst),
      .src_indices_ = src_indices,
      .dst_indices_ = dst_indices,
      .item_units_ = item_units,
      .src_layer_stride_ = src_layer_stride_bytes / kUnit,
      .dst_page_stride_ = dst_page_stride_bytes / kUnit,
      .units_per_chunk_ = plan.units_per_chunk,
  };

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<decltype(kernel)>(
        sycl::nd_range<3>(
            sycl::range<3>(
                static_cast<size_t>(num_items),
                static_cast<size_t>(num_layers),
                static_cast<size_t>(chunks * kMambaWgSize)),
            sycl::range<3>(1, 1, static_cast<size_t>(kMambaWgSize))),
        kernel);
  };
  dpcppGetCurrentQueue().submit(cgf);
}

// Dispatch on the runtime-chosen copy width.  The unit type carries no
// semantics -- this is a byte copy, not a per-dtype instantiation.
#define _MAMBA_DISPATCH_WIDTH(WIDTH, LAUNCH, ...) \
  switch (WIDTH) {                                \
    case 8:                                       \
      LAUNCH<uint64_t>(__VA_ARGS__);              \
      break;                                      \
    case 4:                                       \
      LAUNCH<uint32_t>(__VA_ARGS__);              \
      break;                                      \
    case 2:                                       \
      LAUNCH<uint16_t>(__VA_ARGS__);              \
      break;                                      \
    default:                                      \
      LAUNCH<uint8_t>(__VA_ARGS__);               \
      break;                                      \
  }

// Shared argument validation for both directions; returns the item count.
static int64_t check_mamba_transfer(
    const at::Tensor& src,
    const at::Tensor& dst,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t layout_dim) {
  TORCH_CHECK(src_indices.scalar_type() == at::kLong, "src_indices must be int64");
  TORCH_CHECK(dst_indices.scalar_type() == at::kLong, "dst_indices must be int64");
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "index count mismatch");
  TORCH_CHECK(src_indices.is_contiguous() && dst_indices.is_contiguous(), "indices must be contiguous");
  // Flat byte addressing assumes both pools are contiguous.
  TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src/dst must be contiguous");
  TORCH_CHECK(item_size > 0, "item_size must be positive");
  TORCH_CHECK(layout_dim >= item_size, "layout_dim must be at least item_size");
  TORCH_CHECK(layout_dim % item_size == 0, "layout_dim must be a whole number of items");
  return src_indices.numel();
}

// Load: page_first → single-layer, one layer slot.
// item_size and src_layout_dim are BYTES (per item, and per page in src).
SGL_KERNEL_EXPORT void transfer_kv_mamba_pf_lf(
    const at::Tensor& src,
    at::Tensor& dst,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim) {
  const int64_t num_items = check_mamba_transfer(src, dst, src_indices, dst_indices, item_size, src_layout_dim);
  if (num_items == 0) return;
  TORCH_CHECK(layer_id >= 0 && layer_id < src_layout_dim / item_size, "layer_id out of range for src_layout_dim");

  const auto src_addr = reinterpret_cast<int64_t>(src.const_data_ptr());
  const auto dst_addr = reinterpret_cast<int64_t>(dst.data_ptr());
  const int64_t width = mamba_copy_width({src_addr, dst_addr, item_size, src_layout_dim});

  _MAMBA_DISPATCH_WIDTH(
      width,
      launch_mamba_load,
      src.const_data_ptr(),
      dst.data_ptr(),
      src_indices.const_data_ptr<int64_t>(),
      dst_indices.const_data_ptr<int64_t>(),
      layer_id,
      item_size,
      src_layout_dim,
      num_items);
}

// Backup: layer_first → page_first, all layers.
// item_size and dst_layout_dim are BYTES (per item, and per page in dst).
SGL_KERNEL_EXPORT void transfer_kv_mamba_lf_pf(
    const at::Tensor& src_layers,
    at::Tensor& dst,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers) {
  const int64_t num_items = check_mamba_transfer(src_layers, dst, src_indices, dst_indices, item_size, dst_layout_dim);
  TORCH_CHECK(num_layers > 0, "num_layers must be positive");
  TORCH_CHECK(src_layers.dim() >= 1 && src_layers.size(0) == num_layers, "src_layers.size(0) must be num_layers");
  TORCH_CHECK(dst_layout_dim / item_size >= num_layers, "dst_layout_dim must hold num_layers items");
  if (num_items == 0) return;

  const int64_t src_layer_stride = src_layers.stride(0) * src_layers.element_size();
  const auto src_addr = reinterpret_cast<int64_t>(src_layers.const_data_ptr());
  const auto dst_addr = reinterpret_cast<int64_t>(dst.data_ptr());
  const int64_t width = mamba_copy_width({src_addr, dst_addr, item_size, dst_layout_dim, src_layer_stride});

  _MAMBA_DISPATCH_WIDTH(
      width,
      launch_mamba_backup,
      src_layers.const_data_ptr(),
      dst.data_ptr(),
      src_indices.const_data_ptr<int64_t>(),
      dst_indices.const_data_ptr<int64_t>(),
      item_size,
      src_layer_stride,
      dst_layout_dim,
      num_items,
      num_layers);
}

#undef _MAMBA_DISPATCH_WIDTH
