#pragma once

#include <cstdlib>
#include <vector>

#include "sycl/kernels/flash_attention_v2/xe_fmha_fwd_prefill_runner.hpp"

namespace chunkprefill {
struct Arguments : public prefill::Arguments {
  int num_kv_splits = 1;
};

inline int round_up_headdim(int head_size) {
  if (head_size <= 64) return 64;
  if (head_size <= 96) return 96;
  if (head_size <= 128) return 128;
  if (head_size <= 192) return 192;
  if (head_size <= 256) return 256;
  return 512;
}

template <int HEAD_DIM>
struct FmhaChunkPrefillDynamicRunner {
  void operator()(const Arguments& params) const;
};

}  // namespace chunkprefill
