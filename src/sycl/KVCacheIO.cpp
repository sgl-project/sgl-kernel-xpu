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
//   lf  = layer-first  [num_layers, num_tokens, item_size]         on-device
//   pf  = page-first   [num_pages, num_layers, page_size, item_size] pinned host
//   ph  = page-head    [num_pages, head_num, page_size, num_layers, head_dim] pinned host
//
// pf and ph pools are always in pinned host memory — this is by design (they
// are CPU-side eviction/disaggregation buffers).  No on-device pf kernels are
// provided because there is no on-device pf use case in the current sglang
// KV-cache placement policy.

#include <ATen/ATen.h>
#include <cstdint>
#include <vector>

#include "Utils.h"
#include "comm/General.h"

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
  return base + layer_id * layer_dim + page_id * item_size_bytes +
         item_size_bytes / head_num * head_id;
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
  return base +
         (page_id / page_size) * page_size * page_dim +       // page bucket
         (page_dim / head_num) * head_id * page_size +         // head dim
         (page_id % page_size) * page_dim / head_num +         // slot within page
         layer_id * head_dim_bytes;                            // layer slot
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
  return base +
         (page_id / page_size) * page_size * page_dim +
         (page_dim / head_num) * head_id * page_size +
         (page_id % page_size) * page_dim / head_num +
         layer_id * head_dim_bytes;
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
    const int64_t sg_id = static_cast<int64_t>(item.get_local_id(0)) / XPU_SG_SIZE +
                          static_cast<int64_t>(item.get_group(0)) *
                              (static_cast<int64_t>(item.get_local_range(0)) / XPU_SG_SIZE);

    for (int64_t i = 0; i < items_per_sg_; ++i) {
      const int64_t item_id = sg_id * items_per_sg_ + i;
      if (item_id >= num_items_) break;

      const int64_t src_page = src_indices_[item_id];
      const int64_t dst_page = dst_indices_[item_id];

      for (int64_t layer = start_layer_; layer < start_layer_ + num_layers_; ++layer) {
        const char* src_k_ptr = src_k_base_ == nullptr
            ? reinterpret_cast<const char*>(src_k_tbl_[layer]) + src_page * item_size_
            : src_k_base_ + src_page * item_size_;
        char* dst_k_ptr = dst_k_base_ == nullptr
            ? reinterpret_cast<char*>(dst_k_tbl_[layer]) + dst_page * item_size_
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
          char* dst_v_ptr = dst_v_base_ == nullptr
              ? reinterpret_cast<char*>(dst_v_tbl_[layer]) + dst_page * item_size_
              : dst_v_base_ + dst_page * item_size_;

          const auto* sv64 = reinterpret_cast<const uint64_t*>(src_v_ptr);
          auto* dv64 = reinterpret_cast<uint64_t*>(dst_v_ptr);
          for (int64_t j = lane; j < chunks; j += XPU_SG_SIZE) {
            const uint64_t k_val = src64[j];
            const uint64_t v_val = sv64[j];
            dst64[j]  = k_val;
            dv64[j]   = v_val;
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

// Page-head transfer kernel: loops over heads because each head's data is
// non-contiguous in the page-head layout.
// Direction is fixed to lf→ph (IsLfToPh=true) or ph→lf (false).
template <bool IsLfToPh>
struct TransferKVPageHeadKernel {
  void operator()(sycl::nd_item<1> item) const {
    const int64_t lane = static_cast<int64_t>(item.get_local_id(0)) % XPU_SG_SIZE;
    const int64_t sg_id = static_cast<int64_t>(item.get_local_id(0)) / XPU_SG_SIZE +
                          static_cast<int64_t>(item.get_group(0)) *
                              (static_cast<int64_t>(item.get_local_range(0)) / XPU_SG_SIZE);

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
  TORCH_CHECK(src_indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "index count mismatch");

  const int64_t num_items = src_indices.numel();
  // Auto-scale block_quota so num_wgs lands in [16, 32]: enough to fill all
  // 20 Xe-cores on B580 without over-decomposing small workloads.
  // Caller-supplied block_quota=0 means "auto"; any non-zero value is used
  // directly for explicit override.
  const int64_t effective_bq = (block_quota > 0)
      ? block_quota
      : std::max(int64_t(1), div_up(num_items, sgs_per_wg * 16));
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
            sycl::range<1>(static_cast<size_t>(num_wgs * wg_size)),
            sycl::range<1>(static_cast<size_t>(wg_size))),
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
  TORCH_CHECK(item_size % head_num == 0, "item_size must be divisible by head_num");
  TORCH_CHECK(src_indices.numel() == dst_indices.numel(), "index count mismatch");

  const int64_t num_items = src_indices.numel();
  const int64_t effective_bq = (block_quota > 0)
      ? block_quota
      : std::max(int64_t(1), div_up(num_items, sgs_per_wg * 16));
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
            sycl::range<1>(static_cast<size_t>(num_wgs * wg_size)),
            sycl::range<1>(static_cast<size_t>(wg_size))),
        kernel);
  };
  dpcppGetCurrentQueue().submit(cgf);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Single-layer, lf→lf, K+V.
void transfer_kv_per_layer(
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
      src_k.data_ptr(), dst_k.data_ptr(),
      src_v.data_ptr(), dst_v.data_ptr(),
      nullptr, nullptr, nullptr, nullptr,
      src_indices, dst_indices,
      0, 1, item_size, block_quota, sgs_per_wg);
}

// Single-layer, lf→lf, K only (MLA).
void transfer_kv_per_layer_mla(
    const at::Tensor& src,
    at::Tensor& dst,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  launch_transfer_kv<true>(
      src.data_ptr(), dst.data_ptr(),
      nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr,
      src_indices, dst_indices,
      0, 1, item_size, block_quota, sgs_per_wg);
}

// All-layers, lf_tbl→lf_tbl, K+V.
void transfer_kv_all_layer(
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
  TORCH_CHECK(num_layers == src_k_layers.size(0), "num_layers mismatch");
  launch_transfer_kv<false>(
      nullptr, nullptr, nullptr, nullptr,
      src_k_layers.data_ptr<uintptr_t>(),
      dst_k_layers.data_ptr<uintptr_t>(),
      src_v_layers.data_ptr<uintptr_t>(),
      dst_v_layers.data_ptr<uintptr_t>(),
      src_indices, dst_indices,
      0, num_layers, item_size, block_quota, sgs_per_wg);
}

// All-layers, lf_tbl→lf_tbl, K only (MLA).
void transfer_kv_all_layer_mla(
    const at::Tensor& src_layers,
    const at::Tensor& dst_layers,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t sgs_per_wg) {
  TORCH_CHECK(num_layers == src_layers.size(0), "num_layers mismatch");
  launch_transfer_kv<true>(
      nullptr, nullptr, nullptr, nullptr,
      src_layers.data_ptr<uintptr_t>(),
      dst_layers.data_ptr<uintptr_t>(),
      nullptr, nullptr,
      src_indices, dst_indices,
      0, num_layers, item_size, block_quota, sgs_per_wg);
}

// All-layers, lf_tbl→ph (page-head destination).
void transfer_kv_all_layer_lf_ph(
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
  TORCH_CHECK(num_layers == src_k_layers.size(0), "num_layers mismatch");
  launch_transfer_kv_page_head<true>(
      nullptr, dst_k.data_ptr(),
      nullptr, dst_v.data_ptr(),
      src_k_layers.data_ptr<uintptr_t>(), nullptr,
      src_v_layers.data_ptr<uintptr_t>(), nullptr,
      src_indices, dst_indices,
      num_layers, item_size,
      0, dst_layout_dim,
      page_size, head_num,
      block_quota, sgs_per_wg);
}

// Single-layer, ph→lf.
// layer_id: which layer slot to read from the page-head source layout.
// dst_k/dst_v already point to the correct layer's contiguous buffer.
void transfer_kv_per_layer_ph_lf(
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
      src_k.data_ptr(), dst_k.data_ptr(),
      src_v.data_ptr(), dst_v.data_ptr(),
      nullptr, nullptr, nullptr, nullptr,
      src_indices, dst_indices,
      1, item_size,
      src_layout_dim, 0,
      page_size, head_num,
      block_quota, sgs_per_wg,
      layer_id);
}
