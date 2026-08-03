// Host-only helper exposing `gdn::gdn_workspace_bytes_needed` (see
// gdn_attn/gdn_attn_utils.h) as a torch op, so Python callers can size a
// workspace buffer for `gdn_attention` using the exact same layout math the
// op itself uses internally, instead of re-implementing it.
#include <torch/all.h>

#include "gdn_attn/gdn_attn_utils.h"

int64_t gdn_attention_workspace_bytes_needed(
    const int64_t num_prefills,
    const int64_t num_decodes,
    const int64_t non_spec_token,
    const int64_t batch_size,
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    const int64_t tp_size,
    const torch::ScalarType dtype) {
  return gdn::gdn_workspace_bytes_needed(
      num_prefills,
      num_decodes,
      non_spec_token,
      batch_size,
      num_k_heads,
      num_v_heads,
      head_k_dim,
      head_v_dim,
      tp_size,
      dtype);
}
