#include <ATen/ATen.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "QuantUtils.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kPackedDim = 64;
constexpr int64_t kGroupSize = 32;
constexpr int64_t kNumGroups = 4;
constexpr int64_t kCacheBytesPerToken = 68;

inline float scale_ue8m0(uint8_t ue8m0) {
  const uint32_t bits = static_cast<uint32_t>(ue8m0) << 23;
  return sycl::bit_cast<float>(bits);
}

inline float decode_fp4(uint8_t code) {
  const uint8_t magnitude = code & 0x7;
  float value = 0.0f;
  switch (magnitude) {
    case 1:
      value = 0.5f;
      break;
    case 2:
      value = 1.0f;
      break;
    case 3:
      value = 1.5f;
      break;
    case 4:
      value = 2.0f;
      break;
    case 5:
      value = 3.0f;
      break;
    case 6:
      value = 4.0f;
      break;
    case 7:
      value = 6.0f;
      break;
    default:
      break;
  }
  return (code & 0x8) ? -value : value;
}

struct Fp4PagedMqaLogitsKernel {
  const uint8_t* q_fp4;
  const uint8_t* q_scale;
  const uint8_t* kv_cache;
  const float* weights;
  const int32_t* seq_lens;
  const int32_t* block_tables;
  float* output;
  int batch;
  int heads;
  int page_size;
  int max_blocks;
  int max_seq_len;

  void operator()(sycl::nd_item<2> item) const {
    const int b = static_cast<int>(item.get_global_id(0));
    const int token = static_cast<int>(item.get_global_id(1));
    if (b >= batch || token >= max_seq_len || token >= seq_lens[b]) return;

    const int page_block = token / page_size;
    const int token_in_page = token % page_size;
    const int page_id = block_tables[b * max_blocks + page_block];
    // Page layout written by FusedNormRopeStore: packed nibbles for all tokens of
    // the page first, then the per-token UE8M0 group scales.
    const uint8_t* page_base = kv_cache + static_cast<int64_t>(page_id) * page_size * kCacheBytesPerToken;
    const uint8_t* k_payload = page_base + static_cast<int64_t>(token_in_page) * kPackedDim;
    const uint8_t* k_scales =
        page_base + static_cast<int64_t>(page_size) * kPackedDim + static_cast<int64_t>(token_in_page) * kNumGroups;
    const uint8_t* q_row = q_fp4 + static_cast<int64_t>(b) * heads * kPackedDim;
    const uint8_t* q_scale_row = q_scale + static_cast<int64_t>(b) * heads * kNumGroups;

    float score = 0.0f;
    for (int h = 0; h < heads; ++h) {
      const uint8_t* q_head = q_row + static_cast<int64_t>(h) * kPackedDim;
      const uint8_t* q_head_scale = q_scale_row + static_cast<int64_t>(h) * kNumGroups;
      float dot = 0.0f;
      for (int d = 0; d < kHeadDim; ++d) {
        const uint8_t q_code = (q_head[d / 2] >> ((d & 1) * 4)) & 0xF;
        const uint8_t k_code = (k_payload[d / 2] >> ((d & 1) * 4)) & 0xF;
        const float q_value = decode_fp4(q_code) * scale_ue8m0(q_head_scale[d / kGroupSize]);
        const float k_value = decode_fp4(k_code) * scale_ue8m0(k_scales[d / kGroupSize]);
        dot += q_value * k_value;
      }
      dot = sycl::fmax(dot, 0.0f);
      score += dot * weights[b * heads + h];
    }
    output[b * max_seq_len + token] = score;
  }
};

}  // namespace

SGL_KERNEL_EXPORT torch::Tensor fp4_paged_mqa_logits(
    const torch::Tensor& q_fp4,
    const torch::Tensor& q_scale,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& seq_lens,
    const torch::Tensor& block_tables,
    const std::optional<torch::Tensor>& schedule_metadata,
    int64_t max_seq_len,
    bool clean_logits) {
  (void)schedule_metadata;
  (void)clean_logits;
  CHECK_INPUT(q_fp4);
  CHECK_INPUT(q_scale);
  CHECK_INPUT(kv_cache);
  CHECK_INPUT(weights);
  CHECK_INPUT(seq_lens);
  CHECK_INPUT(block_tables);
  TORCH_CHECK(q_fp4.is_xpu() && q_scale.is_xpu() && kv_cache.is_xpu(), "FP4 tensors must be on XPU");
  TORCH_CHECK(q_fp4.dim() == 3 && q_scale.dim() == 3, "q_fp4/q_scale must be [B,H,D]");
  TORCH_CHECK(q_fp4.size(2) == kPackedDim && q_scale.size(2) == kNumGroups, "invalid FP4 Q shape");
  TORCH_CHECK(q_scale.size(0) == q_fp4.size(0) && q_scale.size(1) == q_fp4.size(1), "Q scale shape mismatch");
  TORCH_CHECK(
      kv_cache.dim() == 4 && kv_cache.size(2) == 1 && kv_cache.size(3) == kCacheBytesPerToken,
      "kv_cache must be [pages,page,1,68]");
  TORCH_CHECK(
      weights.dim() == 2 && weights.size(0) == q_fp4.size(0) && weights.size(1) == q_fp4.size(1),
      "weights must be [B,H]");
  TORCH_CHECK(
      seq_lens.scalar_type() == at::kInt && block_tables.scalar_type() == at::kInt,
      "seq_lens/block_tables must be int32");
  TORCH_CHECK(
      q_fp4.scalar_type() == at::kByte && q_scale.scalar_type() == at::kByte && kv_cache.scalar_type() == at::kByte,
      "FP4 buffers must be uint8");
  TORCH_CHECK(weights.scalar_type() == at::kFloat, "weights must be float32");
  TORCH_CHECK(block_tables.dim() == 2 && block_tables.size(0) == q_fp4.size(0), "block_tables shape mismatch");
  TORCH_CHECK(
      max_seq_len > 0 && block_tables.size(1) * kv_cache.size(1) >= max_seq_len,
      "block table does not cover max_seq_len");

  auto output = torch::zeros({q_fp4.size(0), max_seq_len}, q_fp4.options().dtype(torch::kFloat32));
  if (q_fp4.numel() == 0) return output;
  auto seq_flat = seq_lens.contiguous().view({-1});
  TORCH_CHECK(seq_flat.size(0) == q_fp4.size(0), "seq_lens must have B elements");
  auto queue = c10::xpu::getCurrentXPUStream().queue();
  Fp4PagedMqaLogitsKernel kernel{
      q_fp4.data_ptr<uint8_t>(),
      q_scale.data_ptr<uint8_t>(),
      kv_cache.data_ptr<uint8_t>(),
      weights.data_ptr<float>(),
      seq_flat.data_ptr<int32_t>(),
      block_tables.data_ptr<int32_t>(),
      output.data_ptr<float>(),
      static_cast<int>(q_fp4.size(0)),
      static_cast<int>(q_fp4.size(1)),
      static_cast<int>(kv_cache.size(1)),
      static_cast<int>(block_tables.size(1)),
      static_cast<int>(max_seq_len)};
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(q_fp4.size(0), max_seq_len), sycl::range<2>(1, 1)), kernel);
  });
  return output;
}
