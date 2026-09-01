#include <ATen/ATen.h>

#include <cstdint>

#include "Utils.h"
#include "comm/General.h"
#include "sgl_kernel/hisparse/load_cache_to_device_buffer.hpp"
#include "sgl_kernel/hisparse/transfer_cache_dsv4_mla.hpp"
#include "sgl_kernel_export.h"

using namespace sgl::sycl_kernel::hisparse;

namespace {

// ---------------------------------------------------------------------------
// transfer_cache_dsv4_mla
// ---------------------------------------------------------------------------

template <int BLOCK_SIZE>
void launch_transfer_cache_dsv4_mla(
    void** src_caches,
    void** dst_caches,
    const int64_t* src_indices,
    const int64_t* dst_indices,
    uint32_t num_items,
    uint32_t num_layers) {
  constexpr int kNumSubGroups = BLOCK_SIZE / kSubGroupSize;
  const uint32_t num_groups = div_up(num_items, static_cast<uint32_t>(kNumSubGroups));
  const uint32_t total_sub_groups = num_groups * kNumSubGroups;

  TransferCacheDsv4MlaKernel<BLOCK_SIZE> kernel{
      .src_caches_ = src_caches,
      .dst_caches_ = dst_caches,
      .src_indices_ = src_indices,
      .dst_indices_ = dst_indices,
      .num_items_ = num_items,
      .num_layers_ = num_layers,
      .total_sub_groups_ = total_sub_groups,
  };

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<decltype(kernel)>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_groups) * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)),
        kernel);
  };
  dpcppGetCurrentQueue().submit(cgf);
}

// Validate a uint64 pointer table the same way KVCacheIO.cpp does.
void check_ptr_table(const at::Tensor& tbl, int64_t num_layers, const char* name) {
  TORCH_CHECK(tbl.scalar_type() == at::kUInt64, name, " must be a uint64 pointer table");
  TORCH_CHECK(tbl.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tbl.numel() == num_layers, name, " must have num_layers entries, got ", tbl.numel());
}

// ---------------------------------------------------------------------------
// load_cache_to_device_buffer
// ---------------------------------------------------------------------------

struct LoadCacheArgs {
  const int32_t* top_k_tokens;
  int32_t* device_buffer_tokens;
  const int64_t* host_cache_locs;
  const int32_t* device_buffer_locs;
  const void* host_cache_k;
  const void* host_cache_v;
  void* device_buffer_k;
  void* device_buffer_v;
  int32_t* top_k_device_locs;
  const void* req_pool_indices;
  const void* seq_lens;
  int16_t* lru_slots;
  const int32_t* num_real_reqs;
  bool req_pool_indices_is_i64;
  bool seq_lens_is_i64;
  int64_t buffer_stride_0;
  int64_t host_stride;
  int64_t lru_slot_stride_0;
  int64_t top_k_tokens_stride;
  int64_t top_k_device_locs_stride;
  int64_t item_size_bytes;
  int64_t batch_size;
  int block_size;
  int num_top_k;
  int hot_buffer_size;
};

template <bool IsMLA, bool IsDsv4Layout>
void launch_load_cache_to_device_buffer(const LoadCacheArgs& a, const SmemLayout& layout) {
  const int num_sub_groups = a.block_size / kSubGroupSize;

  auto cgf = DPCPP_Q_CGF(cgh) {
    sycl::local_accessor<int32_t, 1> smem(sycl::range<1>(static_cast<size_t>(layout.total_int32_slots)), cgh);
    LoadCacheToDeviceBufferKernel<IsMLA, IsDsv4Layout> kernel{
        .top_k_tokens_ = a.top_k_tokens,
        .device_buffer_tokens_ = a.device_buffer_tokens,
        .host_cache_locs_ = a.host_cache_locs,
        .device_buffer_locs_ = a.device_buffer_locs,
        .host_cache_k_ = a.host_cache_k,
        .host_cache_v_ = a.host_cache_v,
        .device_buffer_k_ = a.device_buffer_k,
        .device_buffer_v_ = a.device_buffer_v,
        .top_k_device_locs_ = a.top_k_device_locs,
        .req_pool_indices_ = a.req_pool_indices,
        .seq_lens_ = a.seq_lens,
        .lru_slots_ = a.lru_slots,
        .num_real_reqs_ = a.num_real_reqs,
        .req_pool_indices_is_i64_ = a.req_pool_indices_is_i64,
        .seq_lens_is_i64_ = a.seq_lens_is_i64,
        .buffer_stride_0_ = a.buffer_stride_0,
        .host_stride_ = a.host_stride,
        .lru_slot_stride_0_ = a.lru_slot_stride_0,
        .top_k_tokens_stride_ = a.top_k_tokens_stride,
        .top_k_device_locs_stride_ = a.top_k_device_locs_stride,
        .item_size_bytes_ = a.item_size_bytes,
        .block_size_ = a.block_size,
        .num_sub_groups_ = num_sub_groups,
        .num_top_k_ = a.num_top_k,
        .hot_buffer_size_ = a.hot_buffer_size,
        .hash_size_ = layout.hash_size,
        .hash_mask_ = layout.hash_mask,
        .num_buffer_chunks_ = layout.num_buffer_chunks,
        .num_token_chunks_ = layout.num_token_chunks,
        .iters_per_sg_buffer_ = div_up(layout.num_buffer_chunks, num_sub_groups),
        .iters_per_sg_token_ = div_up(layout.num_token_chunks, num_sub_groups),
        .total_int32_ = layout.total_int32,
        .smem_ = smem,
    };
    cgh.parallel_for<decltype(kernel)>(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(a.batch_size) * a.block_size),
            sycl::range<1>(static_cast<size_t>(a.block_size))),
        kernel);
  };
  dpcppGetCurrentQueue().submit(cgf);
}

void check_swap_in_tensor(const at::Tensor& t, at::ScalarType expected, const char* name) {
  TORCH_CHECK(t.scalar_type() == expected, name, " must be ", expected, ", got ", t.scalar_type());
  TORCH_CHECK(t.device().is_xpu(), name, " must be on XPU, got ", t.device());
  // The kernel is given only stride(0); every row must be contiguous.
  TORCH_CHECK(t.dim() < 2 || t.stride(-1) == 1, name, " must be row-contiguous (stride(-1) == 1)");
}

}  // namespace

SGL_KERNEL_EXPORT void transfer_cache_dsv4_mla(
    const at::Tensor& src_ptrs,
    const at::Tensor& dst_ptrs,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t block_size) {
  TORCH_CHECK(src_indices.scalar_type() == at::kLong, "src_indices must be int64");
  TORCH_CHECK(dst_indices.scalar_type() == at::kLong, "dst_indices must be int64");
  TORCH_CHECK(src_indices.is_contiguous(), "src_indices must be contiguous");
  TORCH_CHECK(dst_indices.is_contiguous(), "dst_indices must be contiguous");
  TORCH_CHECK(
      src_indices.numel() == dst_indices.numel(),
      "index count mismatch: ",
      src_indices.numel(),
      " vs ",
      dst_indices.numel());

  const int64_t num_layers = src_ptrs.numel();
  check_ptr_table(src_ptrs, num_layers, "src_ptrs");
  check_ptr_table(dst_ptrs, num_layers, "dst_ptrs");

  const int64_t num_items = src_indices.numel();
  if (num_items == 0 || num_layers == 0) return;  // nothing to transfer

  auto** src_caches = reinterpret_cast<void**>(src_ptrs.data_ptr<uint64_t>());
  auto** dst_caches = reinterpret_cast<void**>(dst_ptrs.data_ptr<uint64_t>());
  const auto* src_idx = src_indices.data_ptr<int64_t>();
  const auto* dst_idx = dst_indices.data_ptr<int64_t>();
  const auto items = static_cast<uint32_t>(num_items);
  const auto layers = static_cast<uint32_t>(num_layers);

  // block_size is a template parameter; 1024 is the default, the rest are
  // escape hatches (all three measure within noise on Xe2).
  switch (block_size) {
    case 256:
      launch_transfer_cache_dsv4_mla<256>(src_caches, dst_caches, src_idx, dst_idx, items, layers);
      break;
    case 512:
      launch_transfer_cache_dsv4_mla<512>(src_caches, dst_caches, src_idx, dst_idx, items, layers);
      break;
    case 1024:
      launch_transfer_cache_dsv4_mla<1024>(src_caches, dst_caches, src_idx, dst_idx, items, layers);
      break;
    default:
      TORCH_CHECK(false, "block_size must be one of 256, 512, 1024, got ", block_size);
  }
}

SGL_KERNEL_EXPORT void load_cache_to_device_buffer_mla(
    const at::Tensor& top_k_tokens,
    const at::Tensor& device_buffer_tokens,
    const at::Tensor& host_cache_locs,
    const at::Tensor& device_buffer_locs,
    const at::Tensor& host_cache,
    const at::Tensor& device_buffer,
    const at::Tensor& top_k_device_locs,
    const at::Tensor& req_pool_indices,
    const at::Tensor& seq_lens,
    const at::Tensor& lru_slots,
    const std::optional<at::Tensor>& num_real_reqs,
    int64_t item_size_bytes,
    int64_t num_top_k,
    int64_t hot_buffer_size,
    int64_t page_size,
    int64_t block_size,
    bool is_dsv4_layout) {
  TORCH_CHECK(num_top_k > 0, "num_top_k must be positive, got ", num_top_k);
  TORCH_CHECK(
      hot_buffer_size >= num_top_k, "hot_buffer_size (", hot_buffer_size, ") must be >= num_top_k (", num_top_k, ")");
  TORCH_CHECK(
      block_size > 0 && block_size % kSubGroupSize == 0,
      "block_size must be a positive multiple of ",
      kSubGroupSize,
      ", got ",
      block_size);
  TORCH_CHECK(item_size_bytes > 0, "item_size_bytes must be positive, got ", item_size_bytes);
  // int16_t slot indices are stored in the LRU array and the hash values.
  TORCH_CHECK(hot_buffer_size < 32767, "hot_buffer_size must fit in int16, got ", hot_buffer_size);

  check_swap_in_tensor(top_k_tokens, at::kInt, "top_k_tokens");
  check_swap_in_tensor(device_buffer_tokens, at::kInt, "device_buffer_tokens");
  check_swap_in_tensor(host_cache_locs, at::kLong, "host_cache_locs");
  check_swap_in_tensor(device_buffer_locs, at::kInt, "device_buffer_locs");
  check_swap_in_tensor(top_k_device_locs, at::kInt, "top_k_device_locs");
  check_swap_in_tensor(lru_slots, at::kShort, "lru_slots");
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::kInt || req_pool_indices.scalar_type() == at::kLong,
      "req_pool_indices must be int32 or int64, got ",
      req_pool_indices.scalar_type());
  TORCH_CHECK(
      seq_lens.scalar_type() == at::kInt || seq_lens.scalar_type() == at::kLong,
      "seq_lens must be int32 or int64, got ",
      seq_lens.scalar_type());
  TORCH_CHECK(req_pool_indices.is_contiguous(), "req_pool_indices must be contiguous");
  TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
  TORCH_CHECK(host_cache_locs.dim() >= 2, "host_cache_locs must be at least 2-D");

  TORCH_CHECK(
      device_buffer_tokens.stride(0) == device_buffer_locs.stride(0),
      "device_buffer_tokens and device_buffer_locs must share stride(0), got ",
      device_buffer_tokens.stride(0),
      " vs ",
      device_buffer_locs.stride(0));

  const int64_t batch_size = top_k_tokens.size(0);
  TORCH_CHECK(
      req_pool_indices.numel() >= batch_size && seq_lens.numel() >= batch_size,
      "req_pool_indices / seq_lens must cover the batch (",
      batch_size,
      "), got ",
      req_pool_indices.numel(),
      " / ",
      seq_lens.numel());
  if (batch_size == 0) return;

  const SmemLayout layout = SmemLayout::make(static_cast<int>(num_top_k), static_cast<int>(hot_buffer_size));
  const size_t local_mem_size = dpcppGetCurrentQueue().get_device().get_info<sycl::info::device::local_mem_size>();
  TORCH_CHECK(
      layout.bytes() <= local_mem_size,
      "hisparse swap-in needs ",
      layout.bytes(),
      " bytes of shared local memory for num_top_k=",
      num_top_k,
      " hot_buffer_size=",
      hot_buffer_size,
      ", but the device provides only ",
      local_mem_size);

  at::Tensor real_reqs =
      num_real_reqs.has_value() ? *num_real_reqs : at::full({1}, batch_size, top_k_tokens.options().dtype(at::kInt));
  TORCH_CHECK(real_reqs.scalar_type() == at::kInt, "num_real_reqs must be int32, got ", real_reqs.scalar_type());
  TORCH_CHECK(real_reqs.device().is_xpu(), "num_real_reqs must be on XPU, got ", real_reqs.device());

  LoadCacheArgs args{
      .top_k_tokens = top_k_tokens.data_ptr<int32_t>(),
      .device_buffer_tokens = device_buffer_tokens.data_ptr<int32_t>(),
      .host_cache_locs = host_cache_locs.data_ptr<int64_t>(),
      .device_buffer_locs = device_buffer_locs.data_ptr<int32_t>(),
      .host_cache_k = host_cache.data_ptr(),
      .host_cache_v = nullptr,  // MLA: K-only
      .device_buffer_k = device_buffer.data_ptr(),
      .device_buffer_v = nullptr,  // MLA: K-only
      .top_k_device_locs = top_k_device_locs.data_ptr<int32_t>(),
      .req_pool_indices = req_pool_indices.data_ptr(),
      .seq_lens = seq_lens.data_ptr(),
      .lru_slots = lru_slots.data_ptr<int16_t>(),
      .num_real_reqs = real_reqs.data_ptr<int32_t>(),
      .req_pool_indices_is_i64 = req_pool_indices.scalar_type() == at::kLong,
      .seq_lens_is_i64 = seq_lens.scalar_type() == at::kLong,
      .buffer_stride_0 = device_buffer_tokens.stride(0),
      .host_stride = host_cache_locs.size(1),
      .lru_slot_stride_0 = lru_slots.stride(0),
      .top_k_tokens_stride = top_k_tokens.stride(0),
      .top_k_device_locs_stride = top_k_device_locs.stride(0),
      .item_size_bytes = item_size_bytes,
      .batch_size = batch_size,
      .block_size = static_cast<int>(block_size),
      .num_top_k = static_cast<int>(num_top_k),
      .hot_buffer_size = static_cast<int>(hot_buffer_size),
  };
  (void)page_size;

  if (is_dsv4_layout) {
    launch_load_cache_to_device_buffer</*IsMLA=*/true, /*IsDsv4Layout=*/true>(args, layout);
  } else {
    launch_load_cache_to_device_buffer</*IsMLA=*/true, /*IsDsv4Layout=*/false>(args, layout);
  }
}
