#include <torch/all.h>
#include <torch/library.h>

// Registered from a side library so the image's sgl_kernel (FMHA, GDN, sampling)
// stays loaded. TORCH_LIBRARY_FRAGMENT adds these two ops to that namespace.
void gptq_entry_probe(torch::Tensor device_index);
torch::Tensor lm_head_gemv(torch::Tensor hidden, torch::Tensor weight);
torch::Tensor gptq_gemm(
    torch::Tensor a,
    torch::Tensor b_q_weight,
    torch::Tensor b_gptq_qzeros,
    torch::Tensor b_gptq_scales,
    torch::Tensor b_g_idx,
    bool use_shuffle,
    int64_t bit,
    int64_t zero_format);

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("gptq_entry_probe(Tensor device_index) -> ()");
  m.impl("gptq_entry_probe", torch::kXPU, &gptq_entry_probe);

  m.def(
      "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, Tensor b_gptq_scales, Tensor b_g_idx, bool use_shuffle, int bit, int zero_format=1) -> Tensor");
  m.impl("gptq_gemm", torch::kXPU, &gptq_gemm);

  m.def("lm_head_gemv(Tensor hidden, Tensor weight) -> Tensor");
  m.impl("lm_head_gemv", torch::kXPU, &lm_head_gemv);
}
