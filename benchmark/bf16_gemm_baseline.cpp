#include <ATen/ATen.h>
#include <dlfcn.h>
#include <torch/library.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>

static void baseline_gemm(at::Tensor& output, const at::Tensor& mat_a, const at::Tensor& mat_b, int64_t config) {
  using Function = void (*)(at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t);
  static Function function = [] {
    const char* path = std::getenv("SGL_BF16_BASELINE_KERNEL_LIBRARY");
    TORCH_CHECK(path, "SGL_BF16_BASELINE_KERNEL_LIBRARY is required");
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    TORCH_CHECK(handle, "Cannot load baseline GEMM: ", dlerror());
    void* symbol = dlsym(handle, "_Z16bf16_gemm_nt_xpuRN2at6TensorERKS0_S3_l");
    TORCH_CHECK(symbol, "Cannot resolve baseline GEMM: ", dlerror());
    Dl_info information{};
    TORCH_CHECK(dladdr(symbol, &information) && information.dli_fname, "Cannot identify baseline GEMM symbol image");
    TORCH_CHECK(
        std::filesystem::equivalent(path, information.dli_fname),
        "Baseline GEMM resolved to a different image: ",
        information.dli_fname);
    std::fprintf(stdout, "BASELINE_SYMBOL_IMAGE %s\n", information.dli_fname);
    std::fflush(stdout);
    return reinterpret_cast<Function>(symbol);
  }();
  function(output, mat_a, mat_b, config);
}

TORCH_LIBRARY(sgl_kernel_baseline, library) {
  library.def("bf16_gemm_nt(Tensor! output, Tensor mat_a, Tensor mat_b, int config=0) -> ()");
  library.impl("bf16_gemm_nt", c10::DispatchKey::XPU, &baseline_gemm);
}
