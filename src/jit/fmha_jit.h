// Runtime-JIT dispatch layer for the FMHA forward kernels.
//
// Maps a runtime attention config (op kind, query-group size, head dim, page
// size, query dtype) to the matching *.cpp.in template, compiles it on first
// use via the generic JIT engine, and caches the resolved kernel entry in an
// O(1) front cache so the launch hot path is a single lookup + indirect call.
//
// This layer is torch-free: the caller (flash_attention.cpp) builds the POD
// `decode::Arguments` and passes its address as an opaque `const void*`; the
// JIT-compiled module casts it back (both sides share the runner header, so the
// struct layout is identical).
#pragma once

#include <string>

namespace sgl {
namespace fmha_jit {

// Decode-family kernel variants (each backed by its own *.cpp.in template).
enum class DecodeOp {
  kDecode,          // paged, 16-bit KV
  kSplitDecode,     // paged, 16-bit KV, split-KV
  kDecodeFp8,       // paged, fp8 KV
  kSplitDecodeFp8,  // paged, fp8 KV, split-KV
  kDecodeNoPage,    // non-paged (contiguous ragged) KV, 16-bit
};

// Launch a decode kernel. `is_fp16` selects the query dtype (false => bf16).
// `page_size` is ignored for kDecodeNoPage. Returns true on success; on failure
// returns false and, if `err` is non-null, fills it with a diagnostic.
bool decode_launch(
    DecodeOp op, int qg, int head_dim, int page_size, bool is_fp16, const void* params,
    std::string* err = nullptr);

// Force-compile (warm the cache) for a config without launching. Useful for
// startup pre-warm to hide first-use compile latency.
bool decode_prewarm(
    DecodeOp op, int qg, int head_dim, int page_size, bool is_fp16, std::string* err = nullptr);

// Prefill-family kernel variants (each backed by its own *.cpp.in template).
enum class PrefillOp {
  kPrefill,        // paged, 16-bit KV
  kPrefillFp8,     // paged, fp8 KV (bf16 query only)
  kPrefillNoPage,  // non-paged (contiguous ragged) KV, 16-bit
};

// Launch a prefill kernel. Prefill dispatches on head dim only; per-head-dim
// tile params are resolved internally (mirroring FMHAPrefillXe20.cmake).
bool prefill_launch(
    PrefillOp op, int head_dim, bool is_fp16, const void* params, std::string* err = nullptr);

bool prefill_prewarm(PrefillOp op, int head_dim, bool is_fp16, std::string* err = nullptr);

}  // namespace fmha_jit
}  // namespace sgl
