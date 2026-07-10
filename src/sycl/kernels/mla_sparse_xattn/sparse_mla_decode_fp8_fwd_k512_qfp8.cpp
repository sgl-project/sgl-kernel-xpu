#include "sparse_mla_decode_fp8_fwd.hpp"

namespace FLASH_NAMESPACE {

template void launch_sparse_mla_decode_fp8_fwd_kernel<512, false, true>(const XPUSparseDecodeAttnFwdParams& params);

} // namespace FLASH_NAMESPACE
