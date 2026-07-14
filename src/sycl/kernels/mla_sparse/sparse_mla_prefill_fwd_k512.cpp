#include "sparse_mla_prefill_fwd.hpp"

namespace FLASH_NAMESPACE {

// kernel instantiation for head_dim_qk = 512
template void launch_sparse_mla_prefill_fwd_kernel<512, false, false>(const XPUSparseAttnFwdParams& params);
template void launch_sparse_mla_prefill_fwd_kernel<512, false, true>(const XPUSparseAttnFwdParams& params);

}  // namespace FLASH_NAMESPACE
