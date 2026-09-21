#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "Utils.h"
#include "kernels/lora/device/lora_permute_rows.hpp"
#include "sgl_kernel_export.h"

// ============================================================
// LoRA gather / scatter row-permute ops
// ------------------------------------------------------------
// Thin op wrappers over the shared row-permute kernel in
// kernels/lora/device/lora_permute_rows.hpp (reused by the fused
// chunked_sgmv_lora_shrink_forward entrypoint). See that header for the
// physical<->logical reordering contract:
//
//   gather : output[i, :]              = input[permutation[i], :]
//   scatter: output[permutation[i], :] = input[i, :]
//
// `permutation[logical] -> physical` must be a bijection over [0, num_rows).
// fp16 / bf16 / fp32 are all supported.
// ============================================================

namespace {

// Shared validation + dispatch for both gather and scatter. `input` and
// `output` must share shape and dtype; `permutation` must be a bijection over
// [0, num_rows) (only its range is checked here).
template <bool GATHER>
void permute_rows_impl(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& permutation) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(permutation);

  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  TORCH_CHECK(output.dim() == 2, "output must be a 2D tensor");
  TORCH_CHECK(permutation.dim() == 1, "permutation must be a 1D tensor");
  TORCH_CHECK(output.scalar_type() == input.scalar_type(), "output dtype must match input dtype");
  TORCH_CHECK(
      output.size(0) == input.size(0) && output.size(1) == input.size(1), "output must have the same shape as input");

  const int64_t num_rows = input.size(0);
  const int64_t width = input.size(1);
  TORCH_CHECK(permutation.numel() == num_rows, "permutation.numel() must equal input.size(0)");

  if (num_rows == 0 || width == 0) {
    return;
  }

  auto [min_p, max_p] = torch::aminmax(permutation);
  TORCH_CHECK(
      min_p.item<int64_t>() >= 0 && max_p.item<int64_t>() < num_rows,
      "permutation values must be in [0, input.size(0))");

  auto perm_i64 = permutation.scalar_type() == torch::kInt64 ? permutation : permutation.to(torch::kInt64);

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  lora_permute_rows_impl::permute_rows_dispatch<GATHER>(input, output, perm_i64, queue);
}

}  // namespace

//----------------- Main API functions --------------------//

// output[i, :] = input[permutation[i], :]
SGL_KERNEL_EXPORT void lora_gather_rows(
    torch::Tensor& output,            // [num_rows, width]
    const torch::Tensor& input,       // [num_rows, width]
    const torch::Tensor& permutation  // [num_rows,]  logical -> physical
) {
  permute_rows_impl</*GATHER=*/true>(output, input, permutation);
}

// output[permutation[i], :] = input[i, :]
SGL_KERNEL_EXPORT void lora_scatter_rows(
    torch::Tensor& output,            // [num_rows, width]
    const torch::Tensor& input,       // [num_rows, width]
    const torch::Tensor& permutation  // [num_rows,]  logical -> physical
) {
  permute_rows_impl</*GATHER=*/false>(output, input, permutation);
}
