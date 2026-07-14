#include "sparse_mla_prefill_fwd.hpp"

namespace FLASH_NAMESPACE {

// kernel instantiation for head_dim_qk = 576 with topk_length
template void launch_sparse_mla_prefill_fwd_kernel<576, true, false>(const XPUSparseAttnFwdParams& params);
template void launch_sparse_mla_prefill_fwd_kernel<576, true, true>(const XPUSparseAttnFwdParams& params);

}  // namespace FLASH_NAMESPACE
