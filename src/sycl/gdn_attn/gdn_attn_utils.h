#pragma once

#include <torch/all.h>

#include <vector>

#include "../Utils.h"

namespace gdn {

static constexpr float l2norm_eps = 0.000001f;

static constexpr int chunk_size_xe2 = 64;
static constexpr int A_block_size_xe2 = 16;

// Minimum TOTAL token count across the pure-prefill batch the standalone
// `gated_delta_rule` op requires before it selects the chunk (parallel)
// kernel over the sequential (token-serial) kernel. The sequential kernel is
// token-serial across ALL sequences, so its cost scales with the total token
// count; the chunk path pays a fixed cost per call (padded-buffer allocation
// + per-sequence 64-token padding + scatter + per-chunk launches) that only
// amortizes once the total is large enough, below which the serial
// recurrence's lack of chunking overheads wins.
//
// Provenance: tuned from device-time crossover sweeps (see
// benchmark/bench_gdn_chunk_dispatch.py; single-sequence crossovers on the
// target hardware were on the order of a few hundred tokens for nk=16/nv=32
// and well under 512 for nk=4/nv=12 at TP4, versus the previous hardcoded
// 2048). 512 is a round value safely above the measured single-sequence
// crossovers -- below it the chunk path never wins; above it the chunk path
// wins for every measured batching shape (the previous 2048-only gate left
// multi-seq batches like 2x2048 on the slow serial kernel). The crossover is
// geometry-, device- AND batch-shape-dependent (many very short sequences
// inflate the padded buffer without doing useful work), so this is ONE
// global heuristic knob, not a tight optimum -- re-run the sweep per target
// before changing it.
static constexpr int64_t kChunkMinTokens = 512;

enum class ActMode {
  silu = 0,
  swish = 1,
};

// Byte alignment `gdn_attention`'s `make_ws_tensor` uses when carving scratch
// tensors out of a flat `workspace` buffer (see gdn_attn_interface_impl.hpp).
// Defined here (rather than as a local `constexpr` inside that function) so
// `gdn_workspace_bytes_needed` below can never silently drift out of sync
// with it.
static constexpr int64_t ws_align_bytes = 256;

inline int64_t ws_align_up(int64_t nbytes) {
  return div_up(nbytes, ws_align_bytes) * ws_align_bytes;
}

// One workspace scratch tensor's shape + dtype, as carved out of `workspace`
// by `gdn_attention`.
struct WsSection {
  std::vector<int64_t> shape;
  torch::ScalarType dtype;
};

// Single source of truth for the (q, k, v, b, a) workspace scratch-tensor
// shapes/dtypes that `gdn_attention` (gdn_attn_interface_impl.hpp) carves out
// of its `workspace` argument for a given call's shapes. The order returned
// here MUST exactly match the order q, k, v, b, a are carved in
// `gdn_attention`, since workspace byte offsets are assigned sequentially in
// that order. Used both by `gdn_attention` itself (to build the real scratch
// tensors) and by `gdn_workspace_bytes_needed` (to compute the exact byte
// total a caller should pre-allocate), so the two can never silently
// diverge -- unlike a hand-duplicated re-implementation of this layout math
// living separately in Python.
inline std::vector<WsSection> gdn_workspace_sections(
    int64_t num_prefills,
    int64_t non_spec_token,
    int64_t batch_size,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    int64_t tp_size,
    torch::ScalarType dtype) {
  int64_t nk = num_k_heads / tp_size;
  int64_t nv = num_v_heads / tp_size;
  if (num_prefills > 0) {
    // Prefill/chunk path: q, k, v (activation dtype) + b, a (always
    // float32), all padded up to a whole number of chunk_size_xe2-sized
    // chunks per sequence.
    int64_t padding_size = batch_size * (chunk_size_xe2 - 1);
    int64_t n = non_spec_token + padding_size;
    return {
        {{n, nk, head_k_dim}, dtype},
        {{n, nk, head_k_dim}, dtype},
        {{n, nv, head_v_dim}, dtype},
        {{nv, n}, torch::kFloat32},
        {{nv, n}, torch::kFloat32},
    };
  }
  // Decode (native) path: q, k, v, b, a all activation dtype, unpadded.
  return {
      {{non_spec_token, nk, head_k_dim}, dtype},
      {{non_spec_token, nk, head_k_dim}, dtype},
      {{non_spec_token, nv, head_v_dim}, dtype},
      {{non_spec_token, nv}, dtype},
      {{non_spec_token, nv}, dtype},
  };
}

// Exact number of bytes `gdn_attention` will carve out of its `workspace`
// argument (q + k + v + b + a, sequentially 256-byte aligned) for a call
// with the given shapes. Exposed to Python as the
// `gdn_attention_workspace_bytes_needed` op so callers can size/grow their
// cached workspace buffer without re-deriving this layout math themselves.
inline int64_t gdn_workspace_bytes_needed(
    int64_t num_prefills,
    int64_t num_decodes,
    int64_t non_spec_token,
    int64_t batch_size,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    int64_t tp_size,
    torch::ScalarType dtype) {
  if (non_spec_token <= 0 || (num_prefills + num_decodes) <= 0) return 0;
  auto sections = gdn_workspace_sections(
      num_prefills, non_spec_token, batch_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim, tp_size, dtype);
  int64_t cursor = 0;
  for (auto& sec : sections) {
    int64_t numel = 1;
    for (auto d : sec.shape)
      numel *= d;
    int64_t nbytes = numel * static_cast<int64_t>(c10::elementSize(sec.dtype));
    cursor = ws_align_up(cursor) + nbytes;
  }
  return cursor;
}

}  // namespace gdn
