/* rms_norm_gated_sycl.cpp — plain-SYCL RMSNormGated for the GDN output norm.
 *
 * Replaces the triton layernorm_gated fallback (fla/layernorm_gated.py) which is
 * the LAST triton kernel on the Qwen3.5 GDN fast path (the ESIMD variant in
 * src/esimd/ is unbuildable — that whole bucket is disabled, src/CMakeLists.txt:47).
 * This lives in the gdn_attn bucket (standard SYCL, no ESIMD intrinsics, JIT/
 * PTL-safe), so it builds with the kernels that already work.
 *
 * For [rows, V]:  output[r] = (x[r] / rms(x[r])) * weight * silu(z[r])
 *   rms(x) = sqrt(mean(x^2) + eps),  silu(z) = z * sigmoid(z)
 * norm-before-gate, no bias, swish activation — the Qwen3.5 RMSNormGated config.
 *
 * Grid: one sub-group (sub_group_size lanes) per row; each lane strides over V.
 * Math in fp32 (matches triton/ref numerics), I/O fp16 or bf16.
 */
#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "vllm_xpu_utils.h"

namespace gdn {

template <typename T, int SG>
struct rms_norm_gated_kernel {
  const T* x;
  const T* z;
  const T* weight;
  T* out;
  int V;
  float eps;

  [[sycl::reqd_sub_group_size(SG)]] void operator()(
      sycl::nd_item<1> item) const {
    const int row = item.get_group(0);
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int base = row * V;

    // sum of squares of x over the row (fp32), reduced across the sub-group.
    float local_sq = 0.0f;
    for (int i = lane; i < V; i += SG) {
      float xv = static_cast<float>(x[base + i]);
      local_sq += xv * xv;
    }
    float sum_sq = sycl::reduce_over_group(sg, local_sq, sycl::plus<>());
    float inv_rms = sycl::rsqrt(sum_sq / static_cast<float>(V) + eps);

    // normed * weight * silu(z), stored back in T.
    for (int i = lane; i < V; i += SG) {
      float xv = static_cast<float>(x[base + i]);
      float wv = static_cast<float>(weight[i]);
      float zv = static_cast<float>(z[base + i]);
      float silu = zv / (1.0f + sycl::exp(-zv));
      out[base + i] = static_cast<T>(xv * inv_rms * wv * silu);
    }
  }
};

template <typename T>
static void launch_rms_norm_gated(
    sycl::queue& q,
    const T* x_ptr,
    const T* z_ptr,
    const T* w_ptr,
    T* out_ptr,
    int rows,
    int V,
    float eps) {
  constexpr int SG = 32;
  rms_norm_gated_kernel<T, SG> task{x_ptr, z_ptr, w_ptr, out_ptr, V, eps};
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>({static_cast<size_t>(rows) * SG}, {SG}), task);
  });
}

// output[rows,V] = rmsnorm(x) * weight * silu(z). x/z/weight/output all share
// dtype (fp16 or bf16), contiguous, V == hidden (head_v_dim, e.g. 128).
// Returns output.
at::Tensor rms_norm_gated(
    at::Tensor& output,         // [rows, V]
    const at::Tensor& x,        // [rows, V]
    const at::Tensor& z,        // [rows, V]
    const at::Tensor& weight,   // [V]
    double eps) {
  TORCH_CHECK(x.is_contiguous() && z.is_contiguous() && weight.is_contiguous() &&
                  output.is_contiguous(),
              "rms_norm_gated: x/z/weight/output must be contiguous");
  TORCH_CHECK(
      x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
      "rms_norm_gated: x must be fp16 or bf16");
  TORCH_CHECK(
      z.scalar_type() == x.scalar_type() &&
          weight.scalar_type() == x.scalar_type() &&
          output.scalar_type() == x.scalar_type(),
      "rms_norm_gated: x/z/weight/output must share dtype");
  const int rows = static_cast<int>(x.size(0));
  const int V = static_cast<int>(x.size(1));

  auto& q = vllm::xpu::vllmGetQueue();
  if (x.scalar_type() == at::kHalf) {
    using T = sycl::half;
    launch_rms_norm_gated<T>(
        q,
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<const T*>(z.data_ptr()),
        reinterpret_cast<const T*>(weight.data_ptr()),
        reinterpret_cast<T*>(output.data_ptr()),
        rows,
        V,
        static_cast<float>(eps));
  } else {
    using T = sycl::ext::oneapi::bfloat16;
    launch_rms_norm_gated<T>(
        q,
        reinterpret_cast<const T*>(x.data_ptr()),
        reinterpret_cast<const T*>(z.data_ptr()),
        reinterpret_cast<const T*>(weight.data_ptr()),
        reinterpret_cast<T*>(output.data_ptr()),
        rows,
        V,
        static_cast<float>(eps));
  }
  return output;
}

}  // namespace gdn
