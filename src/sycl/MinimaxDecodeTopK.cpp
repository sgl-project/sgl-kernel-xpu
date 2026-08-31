#include <ATen/ATen.h>

#include <cstdint>

#include "Utils.h"
#include "comm/General.h"
#include "sgl_kernel/minimax/minimax_decode_topk.hpp"
#include "sgl_kernel_export.h"

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
      "); increase kMaxNumBlocks in the header if needed");
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
