#pragma once

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>

#include "../../../Utils.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/numeric_types.h"
#include "sycl/comm/common.h"
#include "sycl/kernels/gemm_sqrsum/collective/xe_gemm_sqrsum_mainloop.hpp"
#include "sycl/kernels/gemm_sqrsum/kernel/xe_gemm_sqrsum_kernel.hpp"

using namespace cute;

namespace cutlass::gemm_sqrsum::device {

template <class Kernel_>
class GemmSqrSum {
 public:
  using Kernel = Kernel_;
  using Arguments = typename Kernel::Arguments;
  using Params = typename Kernel::Params;

 private:
  Params params_;
  bool initialized_ = false;

 public:
  GemmSqrSum() = default;

  Params const& params() const {
    return params_;
  }

  static cutlass::Status can_implement(Arguments const& args) {
    return Kernel::can_implement(args) ? cutlass::Status::kSuccess : cutlass::Status::kErrorInvalidProblem;
  }

  static size_t get_workspace_size(Arguments const& args) {
    return Kernel::get_workspace_size(args);
  }

  cutlass::Status initialize(
      Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    params_ = Kernel::to_underlying_arguments(args, workspace);
    initialized_ = true;
    return cutlass::Status::kSuccess;
  }

  cutlass::Status update(Arguments const& args, void* workspace = nullptr) {
    return initialize(args, workspace);
  }

  static cutlass::Status run(Params& params, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    launch<Kernel, 256>(params);
    return cutlass::Status::kSuccess;
  }

  cutlass::Status
  run(Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    cutlass::Status status = initialize(args, workspace, queue);
    if (cutlass::Status::kSuccess == status) {
      status = run(params_, queue);
    }
    return status;
  }

  cutlass::Status operator()(
      Arguments const& args, void* workspace = nullptr, sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    return run(args, workspace, queue);
  }

  cutlass::Status run(sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    return run(params_, queue);
  }

  cutlass::Status operator()(sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue()) {
    return run(params_, queue);
  }
};

}  // namespace cutlass::gemm_sqrsum::device

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
