#include <ATen/ATen.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

#include "Fp4Utils.h"
#include "QuantUtils.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

namespace fp4_utils = at::native::xpu::fp4;

// Bytes per packed 128-value FP4 row (two 4-bit values per byte).
constexpr int64_t kPackedDim = 64;
// Four groups of 32 values, each with one UE8M0 scale byte.
constexpr int64_t kNumGroups = 4;
// 64 payload bytes + 4 scale bytes per token; used for total page size,
// not as the per-token offset within a page.
constexpr int64_t kCacheBytesPerToken = 68;
constexpr int kSubgroupSize = 16;
constexpr int kWorkgroupSize = 128;
constexpr int kTokensPerWorkgroup = kWorkgroupSize / kSubgroupSize;

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

  [[sycl::reqd_sub_group_size(kSubgroupSize)]]
  void operator()(sycl::nd_item<2> item) const {
    const int b = static_cast<int>(item.get_group(0));
    const int lane = static_cast<int>(item.get_local_id(1)) % kSubgroupSize;
    const int token = static_cast<int>(item.get_group(1)) * kTokensPerWorkgroup + item.get_local_id(1) / kSubgroupSize;
    if (b >= batch || token >= max_seq_len) return;
    if (token >= seq_lens[b]) {
      if (lane == 0) output[b * max_seq_len + token] = 0.0f;
      return;
    }
    const auto subgroup = item.get_sub_group();

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

    // Each lane owns one packed byte per quantization group; reuse K across heads.
    float key_values[kNumGroups * 2];
#pragma unroll
    for (int g = 0; g < kNumGroups; ++g) {
      const uint8_t packed = k_payload[g * kSubgroupSize + lane];
      const float scale = fp4_utils::decode_ue8m0_scale(k_scales[g]);
      key_values[g * 2] = fp4_utils::decode_e2m1(packed & 0xF) * scale;
      key_values[g * 2 + 1] = fp4_utils::decode_e2m1(packed >> 4) * scale;
    }

    float score = 0.0f;
    for (int h = 0; h < heads; ++h) {
      const uint8_t* q_head = q_row + static_cast<int64_t>(h) * kPackedDim;
      const uint8_t* q_head_scale = q_scale_row + static_cast<int64_t>(h) * kNumGroups;
      float dot = 0.0f;
#pragma unroll
      for (int g = 0; g < kNumGroups; ++g) {
        const uint8_t packed = q_head[g * kSubgroupSize + lane];
        const float scale = fp4_utils::decode_ue8m0_scale(q_head_scale[g]);
        dot += (fp4_utils::decode_e2m1(packed & 0xF) * scale) * key_values[g * 2];
        dot += (fp4_utils::decode_e2m1(packed >> 4) * scale) * key_values[g * 2 + 1];
      }
      dot = sycl::reduce_over_group(subgroup, dot, sycl::plus<float>());
      dot = sycl::fmax(dot, 0.0f);
      score += dot * weights[b * heads + h];
    }
    if (lane == 0) output[b * max_seq_len + token] = score;
  }
};

}  // namespace

SGL_KERNEL_EXPORT torch::Tensor fp4_paged_mqa_logits(
    const torch::Tensor& q_fp4_input,
    const torch::Tensor& q_scale_input,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& seq_lens,
    const torch::Tensor& block_tables,
    const std::optional<torch::Tensor>& schedule_metadata,
    int64_t max_seq_len,
    bool clean_logits) {
  (void)schedule_metadata;
  (void)clean_logits;
  CHECK_INPUT(q_fp4_input);
  CHECK_INPUT(q_scale_input);
  CHECK_INPUT(kv_cache);
  CHECK_INPUT(weights);
  CHECK_INPUT(seq_lens);
  CHECK_INPUT(block_tables);

  auto q_fp4 = q_fp4_input;
  auto q_scale = q_scale_input;
  if (q_scale.scalar_type() != at::kByte) {
    TORCH_CHECK(q_scale.dim() > 0, "q_scale must have at least one dimension");
    const int64_t heads = q_scale.size(-1);
    TORCH_CHECK(heads > 0, "q_scale head dimension must be positive");
    auto q_scale_bytes = q_scale.view(at::kByte);
    TORCH_CHECK(
        q_scale_bytes.size(-1) % heads == 0, "q_scale byte representation must be divisible by its head dimension");
    std::vector<int64_t> q_scale_shape(q_scale.sizes().begin(), q_scale.sizes().end() - 1);
    q_scale_shape.push_back(heads);
    q_scale_shape.push_back(q_scale_bytes.size(-1) / heads);
    q_scale = q_scale_bytes.reshape(q_scale_shape);
  }
  if (q_fp4.dim() == 4) q_fp4 = q_fp4.squeeze(1);
  if (q_scale.dim() == 4) q_scale = q_scale.squeeze(1);

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

  auto output = torch::empty({q_fp4.size(0), max_seq_len}, q_fp4.options().dtype(torch::kFloat32));
  if (q_fp4.numel() == 0) return output.zero_();
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
  const int64_t token_groups = (max_seq_len + kTokensPerWorkgroup - 1) / kTokensPerWorkgroup;
  const sycl::range<2> global_range(q_fp4.size(0), token_groups * kWorkgroupSize);
  const sycl::range<2> local_range(1, kWorkgroupSize);
  sycl_kernel_submit(global_range, local_range, queue, kernel);
  return output;
}
