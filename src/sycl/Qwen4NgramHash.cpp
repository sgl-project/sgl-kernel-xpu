#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <limits>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

// Fixed Qwen3.8 PLE hash contract: 3-token contexts, bigram heads then trigram heads.
constexpr int64_t NGRAM_SIZE = 3;
constexpr int64_t HEADS_PER_NGRAM = 8;
constexpr int64_t NGRAM_HEADS = 2 * HEADS_PER_NGRAM;
constexpr int WG_SIZE = 256;
constexpr int SMALL_WG_SIZE = 32;
constexpr int64_t SMALL_TOKEN_LIMIT = 1024;
constexpr int TOKENS_PER_BLOCK = WG_SIZE / NGRAM_HEADS;
constexpr int MAX_BLOCKS_PER_WG = WG_SIZE / TOKENS_PER_BLOCK;  // one token window hashed per thread
constexpr int64_t TARGET_WGS = 80;  // enough resident work-groups to fill a B60 before batching blocks
constexpr int64_t MAX_GLOBAL_RANGE = std::numeric_limits<int32_t>::max();  // SYCL id queries are 32-bit

// Multiplication wraps modulo 2^64 like torch's int64 ops; signed overflow would be undefined behaviour.
inline int64_t wrap_mul(int64_t a, int64_t b) {
  return static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}

// Floored modulo by a positive divisor without a per-element 64-bit division (emulated on Xe), matching
// torch.remainder. inv = floor((2^64 - 1) / d) underestimates the quotient by at most one, and
// wrap_mod = 2^64 mod d folds negative dividends.
struct FixedDivisor {
  uint64_t d, inv, wrap_mod;

  static FixedDivisor make(uint64_t divisor) {
    const uint64_t inv = ~uint64_t{0} / divisor;
    const uint64_t max_mod = ~uint64_t{0} - inv * divisor;
    return {divisor, inv, max_mod + 1 == divisor ? 0 : max_mod + 1};
  }

  int64_t floor_mod(int64_t a) const {
    const uint64_t u = static_cast<uint64_t>(a);
    uint64_t r = u - sycl::mul_hi(u, inv) * d;
    if (r >= d) {
      r -= d;
    }
    if (a < 0) {  // a = u - 2^64
      r = r >= wrap_mod ? r - wrap_mod : r + d - wrap_mod;
    }
    return static_cast<int64_t>(r);
  }
};

// One 16-lane subgroup per token avoids SLM staging and a work-group barrier for small batches.
struct SmallNgramHashKernel {
  const int64_t *contexts, *multipliers, *vocab_sizes, *offsets;
  int64_t* output;
  int64_t tokens, eos_token_id;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = item.get_global_linear_id();
    const int64_t token = idx / NGRAM_HEADS;
    if (token >= tokens) {
      return;  // Uniform within each subgroup, including the partial work-group tail.
    }
    const int head = idx % NGRAM_HEADS;
    auto sg = item.get_sub_group();
    int64_t bigram = 0, trigram = 0;
    if (sg.get_local_linear_id() == 0) {
      const int64_t* ctx = contexts + token * NGRAM_SIZE;
      const int64_t previous_2 = (ctx[0] == eos_token_id || ctx[1] == eos_token_id) ? eos_token_id : ctx[0];
      bigram = wrap_mul(ctx[2], multipliers[0]) ^ wrap_mul(ctx[1], multipliers[1]);
      trigram = bigram ^ wrap_mul(previous_2, multipliers[2]);
    }
    bigram = sycl::group_broadcast(sg, bigram, 0);
    trigram = sycl::group_broadcast(sg, trigram, 0);
    const int64_t mix = head < HEADS_PER_NGRAM ? bigram : trigram;
    const FixedDivisor divisor = FixedDivisor::make(static_cast<uint64_t>(vocab_sizes[head]));
    output[idx] = divisor.floor_mod(mix) + offsets[head];
  }
};

// A work-group covers `blocks` blocks of TOKENS_PER_BLOCK tokens. Its first NGRAM_HEADS threads prepare the per-head
// divisors and each thread hashes at most one token's window into SLM, so the per-output work is a single modulo;
// then thread i writes output i of every block. Only the last position of each window is hashed; a token before an
// EOS (or two before, for the third position) is replaced by EOS so n-grams never cross a sequence boundary.
struct NgramHashKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  const int64_t* contexts;
  const int64_t* multipliers;
  const int64_t* vocab_sizes;
  const int64_t* offsets;
  int64_t* output;
  int64_t tokens;
  int64_t eos_token_id;
  int blocks;
  sycl::local_accessor<FixedDivisor, 1> divisors;
  sycl::local_accessor<int64_t, 1> mixes;  // [token][bigram, trigram]

  void sycl_ker_config_convention(sycl::handler& cgh) {
    divisors = sycl::local_accessor<FixedDivisor, 1>(NGRAM_HEADS, cgh);
    mixes = sycl::local_accessor<int64_t, 1>(MAX_BLOCKS_PER_WG * TOKENS_PER_BLOCK * 2, cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    const int local_id = item.get_local_linear_id();
    const int64_t first_token = item.get_group_linear_id() * blocks * TOKENS_PER_BLOCK;
    const int64_t wg_tokens = sycl::min<int64_t>(blocks * TOKENS_PER_BLOCK, tokens - first_token);
    // Build divisors and hash bigrams/trigrams into SLM to amortize the cost of the expensive 64-bit division.
    if (local_id < NGRAM_HEADS) {
      divisors[local_id] = FixedDivisor::make(static_cast<uint64_t>(vocab_sizes[local_id]));
    }
    if (local_id < wg_tokens) {
      const int64_t* ctx = contexts + (first_token + local_id) * NGRAM_SIZE;
      const int64_t token_0 = ctx[0], token_1 = ctx[1], token_2 = ctx[2];
      const int64_t previous_2 = (token_0 == eos_token_id || token_1 == eos_token_id) ? eos_token_id : token_0;
      const int64_t bigram = wrap_mul(token_2, multipliers[0]) ^ wrap_mul(token_1, multipliers[1]);
      mixes[local_id * 2] = bigram;
      mixes[local_id * 2 + 1] = bigram ^ wrap_mul(previous_2, multipliers[2]);
    }
    sycl::group_barrier(item.get_group());

    const int head = local_id % NGRAM_HEADS;
    const FixedDivisor divisor = divisors[head];
    const int64_t offset = offsets[head];
    const int is_trigram = head >= HEADS_PER_NGRAM;
    int64_t* out = output + first_token * NGRAM_HEADS;
    for (int64_t slot = local_id / NGRAM_HEADS; slot < wg_tokens; slot += TOKENS_PER_BLOCK) {
      out[slot * NGRAM_HEADS + head] = divisor.floor_mod(mixes[slot * 2 + is_trigram]) + offset;
    }
  }
};
static_assert(WG_SIZE % NGRAM_HEADS == 0 && NGRAM_HEADS <= WG_SIZE);

void check_int64_xpu(const torch::Tensor& t, const char* name) {
  CHECK_INPUT(t);
  TORCH_CHECK(t.scalar_type() == at::kLong, name, " must be int64, got ", t.scalar_type());
}

}  // namespace

namespace at::native::xpu {

// output[N, 16] = Qwen3.8 PLE n-gram ids for contexts[N, 3]. vocab_sizes must be positive.
SGL_KERNEL_EXPORT void qwen4_ngram_hash(
    torch::Tensor& output,
    torch::Tensor& contexts,
    torch::Tensor& multipliers,
    torch::Tensor& vocab_sizes,
    torch::Tensor& offsets,
    int64_t eos_token_id) {
  check_int64_xpu(output, "output");
  check_int64_xpu(contexts, "contexts");
  check_int64_xpu(multipliers, "multipliers");
  check_int64_xpu(vocab_sizes, "vocab_sizes");
  check_int64_xpu(offsets, "offsets");
  TORCH_CHECK(contexts.dim() == 2 && contexts.size(1) == NGRAM_SIZE, "contexts must have shape [N, ", NGRAM_SIZE, "]");
  TORCH_CHECK(multipliers.numel() == NGRAM_SIZE, "multipliers must have ", NGRAM_SIZE, " elements");
  TORCH_CHECK(vocab_sizes.numel() == NGRAM_HEADS, "vocab_sizes must have ", NGRAM_HEADS, " elements");
  TORCH_CHECK(offsets.numel() == NGRAM_HEADS, "offsets must have ", NGRAM_HEADS, " elements");
  const int64_t tokens = contexts.size(0);
  TORCH_CHECK(
      output.dim() == 2 && output.size(0) == tokens && output.size(1) == NGRAM_HEADS,
      "output must have shape [N, ",
      NGRAM_HEADS,
      "]");
  TORCH_CHECK(
      multipliers.device() == contexts.device() && vocab_sizes.device() == contexts.device() &&
          offsets.device() == contexts.device() && output.device() == contexts.device(),
      "all tensors must be on the same device");

  if (tokens == 0) {
    return;
  }
  // Batch more blocks per work-group only once the GPU is full, to amortize the divisor setup.
  const int64_t total_blocks = (tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
  const int blocks =
      static_cast<int>(std::clamp<int64_t>((total_blocks + TARGET_WGS - 1) / TARGET_WGS, 1, MAX_BLOCKS_PER_WG));
  const int64_t global_range = (total_blocks + blocks - 1) / blocks * WG_SIZE;
  TORCH_CHECK(global_range <= MAX_GLOBAL_RANGE, "qwen4_ngram_hash: too many tokens: ", tokens);

  c10::DeviceGuard guard(contexts.device());
  auto q = c10::xpu::getCurrentXPUStream().queue();
  if (tokens <= SMALL_TOKEN_LIMIT) {
    SmallNgramHashKernel kernel{
        contexts.data_ptr<int64_t>(),
        multipliers.data_ptr<int64_t>(),
        vocab_sizes.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        output.data_ptr<int64_t>(),
        tokens,
        eos_token_id};
    const int64_t small_range = (tokens * NGRAM_HEADS + SMALL_WG_SIZE - 1) / SMALL_WG_SIZE * SMALL_WG_SIZE;
    sycl_kernel_submit(small_range, static_cast<int64_t>(SMALL_WG_SIZE), q, kernel);
    return;
  }
  NgramHashKernel kernel{
      {},
      contexts.data_ptr<int64_t>(),
      multipliers.data_ptr<int64_t>(),
      vocab_sizes.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      output.data_ptr<int64_t>(),
      tokens,
      eos_token_id,
      blocks,
      {},
      {}};
  sycl_kernel_submit(global_range, static_cast<int64_t>(WG_SIZE), q, kernel);
}

}  // namespace at::native::xpu
