#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

// ============================================================
// EmbeddingLoRAAFwd Kernel
// ------------------------------------------------------------
// This kernel computes LoRA embedding A values using a 3D launch:
//   - group(0): segment index
//   - group(1): rank index
//   - group(2): token block within the segment
// and local_id(2): token lane inside the block.
//
// Per work-item flow:
//   1. Resolve (segment, rank_idx, token_idx) from launch geometry.
//   2. Read adapter id from weight_indices[segment] and clamp rank via lora_ranks.
//   3. Read segment length from seg_lens[segment] when provided, otherwise
//      derive it from seg_indptr.
//   4. Skip padded work-items (token_idx >= seg_len or rank_idx >= rank).
//   5. For valid tokens, write one output element from either base weights
//      or extra_embeddings (for token ids >= vocab_size).
//
// Notes:
//   - Negative token ids are skipped (no write).
//   - This kernel writes only active rank entries; output tail handling
//     (padding/initialization) is expected to be managed by caller policy.
//
// ============================================================

//----------------- set element type options --------------------//

template <typename T>
struct ToSyclElementType {
  using type = T;
};

template <>
struct ToSyclElementType<at::Half> {
  using type = sycl::half;
};

template <>
struct ToSyclElementType<at::BFloat16> {
  using type = sycl::ext::oneapi::bfloat16;
};

//----------------- Kernel definition --------------------//

template <typename DType>
struct EmbeddingLoRAAFwd : public __SYCL_KER_CONFIG_CONVENTION__ {
  const int64_t* input_ids;
  const DType* weights;
  const int* seg_indptr;
  const int* weight_indices;
  const int* lora_ranks;
  const DType* extra_embeddings;
  // seg_lens: optional per-segment token counts.
  // seg_lens[s] = seg_indptr[s+1] - seg_indptr[s] for each segment s (in general).
  const int* seg_lens;
  DType* output;

  int64_t vocab_size;
  int num_segments;
  int max_rank;
  int num_extra_tokens;
  int num_tokens;

  EmbeddingLoRAAFwd(
      const int64_t* input_ids,
      const DType* weights,
      const int* seg_indptr,
      const int* weight_indices,
      const int* lora_ranks,
      const DType* extra_embeddings,
      const int* seg_lens,
      DType* output,
      const int64_t vocab_size,
      const int num_segments,
      const int max_rank,
      const int num_extra_tokens,
      const int num_tokens)
      : input_ids(input_ids),
        weights(weights),
        seg_indptr(seg_indptr),
        weight_indices(weight_indices),
        lora_ranks(lora_ranks),
        extra_embeddings(extra_embeddings),
        seg_lens(seg_lens),
        output(output),
        vocab_size(vocab_size),
        num_segments(num_segments),
        max_rank(max_rank),
        num_extra_tokens(num_extra_tokens),
        num_tokens(num_tokens) {}

  static constexpr int BLOCK_TOK = 16;
  static constexpr int sub_group_size = 16;

  [[sycl::reqd_sub_group_size(sub_group_size)]]
  void operator()(sycl::nd_item<3> item) const {
    // group ids choose: which segment, which rank_idx in that segment, which token block
    const int seg = static_cast<int>(item.get_group(0));
    const int rank_idx = static_cast<int>(item.get_group(1));
    const int tok_block = static_cast<int>(item.get_group(2));

    // local thread id chooses: which token lane inside this token block
    const int lane = static_cast<int>(item.get_local_id(2));
    const int token_idx = tok_block * BLOCK_TOK + lane;

    if (seg >= num_segments) {
      return;
    }

    // Each segment maps directly to one LoRA adapter
    const int lora = weight_indices[seg];
    int rank = lora_ranks[lora];
    rank = sycl::clamp(rank, 0, max_rank);

    if (rank == 0) {
      return;
    }

    // Segment start and length
    const int seg_start = seg_indptr[seg];
    const int seg_len = (seg_lens != nullptr) ? seg_lens[seg] : (seg_indptr[seg + 1] - seg_indptr[seg]);

    // Some launched workgroups are padding because token_idx runs up to multiples of BLOCK_TOK
    if (token_idx >= seg_len) {
      return;
    }

    // Some local threads are padding because rank_idx runs up to max_rank
    if (rank_idx >= rank) {
      return;
    }

    const int token_pos = seg_start + token_idx;

    const int64_t tok = input_ids[token_pos];
    DType* out_row = output + static_cast<int64_t>(token_pos) * max_rank;

    // Negative token: Early return
    if (tok < 0) {
      return;
    }

    // Base vocab path
    if (tok < vocab_size) {
      const int64_t idx = (static_cast<int64_t>(lora) * max_rank + rank_idx) * vocab_size + tok;
      out_row[rank_idx] = weights[idx];
      return;
    }

    // Extra token path
    const int64_t e = tok - vocab_size;
    if (extra_embeddings != nullptr && e >= 0 && e < num_extra_tokens) {
      const int64_t idx = (static_cast<int64_t>(lora) * num_extra_tokens + e) * max_rank + rank_idx;
      out_row[rank_idx] = extra_embeddings[idx];
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) const {}
};

//----------------- Kernel launch --------------------//

template <typename TensorDType>
void launch_embedding_lora_a_fwd(
    const torch::Tensor& input_ids,
    const torch::Tensor& weights,
    const int64_t vocab_size,
    const torch::Tensor& seg_indptr,
    const torch::Tensor& weight_indices,
    const torch::Tensor& lora_ranks,
    const std::optional<torch::Tensor>& extra_embeddings,
    const std::optional<torch::Tensor>& seg_lens,
    torch::Tensor& output,
    int num_tokens,
    int max_rank,
    int num_segments,
    int num_extra_tokens,
    sycl::queue& queue) {
  using KernelDType = typename ToSyclElementType<TensorDType>::type;
  // max_len = maximum segment length of all segments
  int max_len = 0;
  if (seg_lens.has_value()) {
    max_len = seg_lens->max().item<int>();
  } else {
    auto seg_len_tensor = seg_indptr.slice(0, 1) - seg_indptr.slice(0, 0, seg_indptr.size(0) - 1);
    max_len = seg_len_tensor.max().item<int>();
  }

  constexpr int BLOCK_TOK = EmbeddingLoRAAFwd<KernelDType>::BLOCK_TOK;
  const int num_tok_blocks = (max_len + BLOCK_TOK - 1) / BLOCK_TOK;

  sycl::range<3> local(1, 1, BLOCK_TOK);
  sycl::range<3> global(
      static_cast<size_t>(num_segments),
      static_cast<size_t>(max_rank),
      static_cast<size_t>(num_tok_blocks * BLOCK_TOK));

  auto kernel = EmbeddingLoRAAFwd<KernelDType>(
      input_ids.data_ptr<int64_t>(),
      reinterpret_cast<const KernelDType*>(weights.data_ptr<TensorDType>()),
      seg_indptr.data_ptr<int>(),
      weight_indices.data_ptr<int>(),
      lora_ranks.data_ptr<int>(),
      extra_embeddings.has_value() ? reinterpret_cast<const KernelDType*>(extra_embeddings->data_ptr<TensorDType>())
                                   : nullptr,
      seg_lens.has_value() ? seg_lens->data_ptr<int>() : nullptr,
      reinterpret_cast<KernelDType*>(output.data_ptr<TensorDType>()),
      vocab_size,
      num_segments,
      max_rank,
      num_extra_tokens,
      num_tokens);
  sycl_kernel_submit(global, local, queue, kernel);
}

//----------------- Main API function --------------------//

void embedding_lora_a_fwd(
    torch::Tensor& output,           // [num_tokens, max_rank]
    const torch::Tensor& input_ids,  // [num_tokens,]
    const torch::Tensor& weights,    // [num_loras, max_rank, vocab_size]
    const int64_t vocab_size,
    const torch::Tensor& seg_indptr,                       // [num_segments + 1,]
    const torch::Tensor& weight_indices,                   // [num_segments,]
    const torch::Tensor& lora_ranks,                       // [num_loras,]
    const std::optional<torch::Tensor>& extra_embeddings,  // [num_loras, num_extra_tokens, max_rank]
    const std::optional<torch::Tensor>&
        seg_lens  // [num_segments,] optional; currently unused, reserved for future per-segment optimizations
) {
  CHECK_INPUT(input_ids);
  CHECK_INPUT(weights);
  CHECK_INPUT(seg_indptr);
  CHECK_INPUT(weight_indices);
  CHECK_INPUT(lora_ranks);
  CHECK_INPUT(output);
  if (extra_embeddings.has_value()) {
    CHECK_INPUT(extra_embeddings.value());
  }

  TORCH_CHECK(input_ids.dim() == 1, "input_ids must be a 1D tensor");
  TORCH_CHECK(weights.dim() == 3, "weights must be a 3D tensor");
  TORCH_CHECK(weights.size(2) == vocab_size, "weights' vocab_size dimension must match the provided vocab_size");
  TORCH_CHECK(seg_indptr.dim() == 1, "seg_indptr must be a 1D tensor");
  TORCH_CHECK(weight_indices.dim() == 1, "weight_indices must be a 1D tensor");
  TORCH_CHECK(lora_ranks.dim() == 1, "lora_ranks must be a 1D tensor");
  TORCH_CHECK(output.dim() == 2, "output must be a 2D tensor");

  const int64_t num_loras_i64 = weights.size(0);
  const int64_t max_rank_i64 = weights.size(1);
  const int64_t num_tokens_i64 = input_ids.size(0);

  TORCH_CHECK(lora_ranks.numel() == num_loras_i64, "lora_ranks.numel() must equal weights.size(0)");
  TORCH_CHECK(num_loras_i64 > 0, "weights.size(0) and lora_ranks.numel() must be greater than 0");
  TORCH_CHECK(
      num_tokens_i64 == 0 || seg_indptr.numel() >= 2, "seg_indptr must have at least 2 elements when num_tokens > 0");
  const int64_t num_segments_i64 = seg_indptr.numel() - 1;
  TORCH_CHECK(weight_indices.numel() == num_segments_i64, "weight_indices.numel() must equal seg_indptr.numel() - 1");
  if (num_segments_i64 > 0) {
    const int64_t min_weight_index = weight_indices.min().item<int64_t>();
    const int64_t max_weight_index = weight_indices.max().item<int64_t>();
    TORCH_CHECK(
        min_weight_index >= 0 && max_weight_index < num_loras_i64,
        "weight_indices values must be in [0, weights.size(0))");
  }
  if (num_tokens_i64 > 0) {
    TORCH_CHECK(seg_indptr[0].item<int64_t>() == 0, "seg_indptr[0] must be 0");
    TORCH_CHECK(
        seg_indptr[seg_indptr.numel() - 1].item<int64_t>() == num_tokens_i64, "seg_indptr[-1] must equal num_tokens");
  }
  if (seg_lens.has_value()) {
    CHECK_INPUT(seg_lens.value());
    TORCH_CHECK(seg_lens->dim() == 1, "seg_lens must be a 1D tensor");
    TORCH_CHECK(seg_lens->numel() == num_segments_i64, "seg_lens.numel() must equal num_segments");
  }

  if (extra_embeddings.has_value()) {
    TORCH_CHECK(extra_embeddings->dim() == 3, "extra_embeddings must be a 3D tensor");
    TORCH_CHECK(extra_embeddings->size(0) == num_loras_i64, "extra_embeddings.size(0) must equal weights.size(0)");
    TORCH_CHECK(
        extra_embeddings->size(2) == max_rank_i64, "extra_embeddings.size(2) must equal max_rank (weights.size(1))");
    TORCH_CHECK(
        extra_embeddings->scalar_type() == weights.scalar_type(), "extra_embeddings dtype must match weights dtype");
  }

  TORCH_CHECK(
      lora_ranks.min().item<int>() >= 0 && lora_ranks.max().item<int>() <= max_rank_i64,
      "All values in lora_ranks must be within the range [0, max_rank]");

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  int num_tokens = num_tokens_i64;
  int max_rank = max_rank_i64;
  int num_segments = num_segments_i64;
  int num_extra_tokens = extra_embeddings.has_value() ? extra_embeddings->size(1) : 0;

  if (num_tokens == 0) {
    return;
  }

  // Validate output tensor size and dtype
  TORCH_CHECK(
      output.size(0) == num_tokens && output.size(1) == max_rank,
      "Output tensor must have shape (num_tokens, max_rank)");
  TORCH_CHECK(output.scalar_type() == weights.scalar_type(), "Output tensor dtype must match weights dtype");

  // Dispatch kernel based on data type
  if (weights.scalar_type() == torch::kFloat32) {
    launch_embedding_lora_a_fwd<float>(
        input_ids,
        weights,
        vocab_size,
        seg_indptr,
        weight_indices,
        lora_ranks,
        extra_embeddings,
        seg_lens,
        output,
        num_tokens,
        max_rank,
        num_segments,
        num_extra_tokens,
        queue);
  } else if (weights.scalar_type() == torch::kHalf) {
    launch_embedding_lora_a_fwd<at::Half>(
        input_ids,
        weights,
        vocab_size,
        seg_indptr,
        weight_indices,
        lora_ranks,
        extra_embeddings,
        seg_lens,
        output,
        num_tokens,
        max_rank,
        num_segments,
        num_extra_tokens,
        queue);
  } else if (weights.scalar_type() == torch::kBFloat16) {
    launch_embedding_lora_a_fwd<at::BFloat16>(
        input_ids,
        weights,
        vocab_size,
        seg_indptr,
        weight_indices,
        lora_ranks,
        extra_embeddings,
        seg_lens,
        output,
        num_tokens,
        max_rank,
        num_segments,
        num_extra_tokens,
        queue);
  } else {
    TORCH_CHECK(false, "Unsupported data type for weights");
  }
}
