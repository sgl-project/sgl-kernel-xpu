#pragma once

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>

#include "../../../Utils.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/numeric_types.h"
#include "sycl/kernels/gemm_sqrsum/collective/xe_gemm_sqrsum_mainloop.hpp"
#include "sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_runner.hpp"
#include "sycl/kernels/gemm_sqrsum/kernel/xe_gemm_sqrsum_kernel.hpp"

using namespace cute;

struct GemmSqrSumXe {
  using Element = cutlass::tfloat32_t;
  using ElementALoad = cutlass::bfloat16_t;

  using TileShape = Shape<Int<64>, Int<32>, Int<16>>;

  using MMAOperation = XE_DPAS_TT<8, float, Element>;
  using MmaAtom = MMA_Atom<MMAOperation>;

  using SubgroupLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;

  using TiledMma = typename TiledMMAHelper<MmaAtom, Layout<TileShape>, SubgroupLayout>::TiledMMA;

  // Pipeline Stages
  using DispatchPolicy = cutlass::gemm_sqrsum::XeDefault<3>;

  using TensorA = decltype(make_tensor(
      make_gmem_ptr<ElementALoad const>(nullptr),
      make_layout(make_shape(Int<64>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}))));
  using TensorB = decltype(make_tensor(
      make_gmem_ptr<Element const>(nullptr),
      make_layout(make_shape(Int<64>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}))));

  using CollectiveMainloop =
      cutlass::gemm_sqrsum::collective::XeGemmSqrSumMainloop<DispatchPolicy, TiledMma, TensorA, TensorB>;

  using Kernel = cutlass::gemm_sqrsum::kernel::GemmSqrSumKernel<CollectiveMainloop>;
};

inline void runGemmSqrSum(at::Tensor& C, at::Tensor& sqrsum, const at::Tensor& A, const at::Tensor& B) {
  using Kernel = typename GemmSqrSumXe::Kernel;
  using Runner = cutlass::gemm_sqrsum::device::GemmSqrSum<Kernel>;

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);

  int split_k = C.size(0);

  TORCH_CHECK(B.size(1) == K, "A.K (", K, ") must match B.K (", B.size(1), ") for GEMM");
  TORCH_CHECK(C.dim() == 3 && C.size(1) == M && C.size(2) == N, "C must be [n_splits, M, N]");
  TORCH_CHECK(
      sqrsum.dim() == 2 && sqrsum.size(0) == split_k && sqrsum.size(1) == M,
      "sqrsum must be [n_splits, M] matching C's leading dim");

  C.zero_();
  sqrsum.zero_();

  at::Tensor A_buf = A.scalar_type() == at::kBFloat16 ? A.contiguous() : A.to(at::kBFloat16);
  at::Tensor B_buf = B.scalar_type() == at::kFloat ? B.contiguous() : B.to(at::kFloat);

  typename Kernel::Arguments args;
  args.M = M;
  args.K = K;
  args.N = N;
  args.split_k = split_k;

  args.ptr_A = reinterpret_cast<GemmSqrSumXe::ElementALoad const*>(A_buf.data_ptr());
  args.stride_A_m = K;
  args.stride_A_k = 1;

  args.ptr_B = reinterpret_cast<GemmSqrSumXe::Element const*>(B_buf.data_ptr());
  args.stride_B_k = K;
  args.stride_B_n = 1;

  args.ptr_C = reinterpret_cast<typename Kernel::ElementC*>(C.data_ptr());
  args.stride_C_m = N;
  args.stride_C_n = 1;

  auto sqsc = torch::empty({split_k, M, N}, sqrsum.options().dtype(torch::kFloat32));
  args.ptr_sqrsum = sqsc.data_ptr<float>();
  args.ptr_sqrsum_scratch = sqsc.data_ptr<float>();
  args.stride_sqsc_m = N;
  args.stride_sqsc_n = 1;

  args.mainloop = typename GemmSqrSumXe::CollectiveMainloop::Arguments{};

  Runner runner;
  auto status = runner.run(args, nullptr, c10::xpu::getCurrentXPUStream().queue());

  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM+SqrSum kernel failed with status: ", int(status));

  sqrsum.copy_(sqsc.select(2, 0));
}
