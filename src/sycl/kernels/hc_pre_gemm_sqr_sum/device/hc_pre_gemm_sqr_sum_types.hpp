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
#include "sycl/kernels/hc_pre_gemm_sqr_sum/collective/xe_hc_pre_gemm_sqr_sum_epilogue.hpp"
#include "sycl/kernels/hc_pre_gemm_sqr_sum/collective/xe_hc_pre_gemm_sqr_sum_mainloop.hpp"
#include "sycl/kernels/hc_pre_gemm_sqr_sum/kernel/xe_hc_pre_gemm_sqr_sum_kernel.hpp"

using namespace cute;

template <typename Element, typename Stride>
auto make_dummy_tensor_type(Element val, Stride stride) {
  return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<Stride>>(1), stride));
}

struct HcPreGemmSqrSumProblemShape {
  int M = 0;
  int N = 0;
  int K = 0;

  HcPreGemmSqrSumProblemShape() = default;
};

namespace cutlass::hc_pre_gemm_sqr_sum::device {

template <class Kernel_>
class HcPreGemmSqrSum {
 public:
  using Kernel = Kernel_;
  using Arguments = typename Kernel::Arguments;
  using Params = typename Kernel::Params;

 private:
  Params params_;
  bool initialized_ = false;

 public:
  HcPreGemmSqrSum() = default;

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

}  // namespace cutlass::hc_pre_gemm_sqr_sum::device

struct HcPreGemmSqrSumXe {
  using Element = cutlass::tfloat32_t;
  using ElementALoad = cutlass::bfloat16_t;

  using StrideA = Stride<int, _1>;
  using StrideB = Stride<int, _1>;
  using StrideC = Stride<int, _1>;
  using StrideSqsum = Stride<int, _1>;

  using TileShape = Shape<Int<64>, Int<32>, Int<16>>;

  using MMAOperation = XE_DPAS_TT<8, float, Element>;
  using MmaAtom = MMA_Atom<MMAOperation>;

  using SubgroupLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;

  using TiledMma = typename TiledMMAHelper<MmaAtom, Layout<TileShape>, SubgroupLayout>::TiledMMA;

  // Pipeline Stages
  using DispatchPolicy = cutlass::hc_pre_gemm_sqr_sum::XeDefault<3>;

  using TensorA = decltype(make_dummy_tensor_type(ElementALoad{}, StrideA{}));
  using TensorB = decltype(make_dummy_tensor_type(Element{}, StrideB{}));

  using CollectiveMainloop =
      cutlass::hc_pre_gemm_sqr_sum::collective::XeHcPreGemmSqrSumMainloop<DispatchPolicy, TiledMma, TensorA, TensorB>;

  using CollectiveEpilogue = cutlass::hc_pre_gemm_sqr_sum::collective::XeHcPreGemmSqrSumEpilogue<CollectiveMainloop>;

  using Kernel =
      cutlass::hc_pre_gemm_sqr_sum::kernel::XeHcPreGemmSqrSumKernel<HcPreGemmSqrSumProblemShape, CollectiveMainloop, CollectiveEpilogue>;
};

inline typename HcPreGemmSqrSumXe::Kernel::Arguments
args_from_options(at::Tensor& C, at::Tensor& sqr_sum, const at::Tensor& A, const at::Tensor& B) {
  using Kernel = typename HcPreGemmSqrSumXe::Kernel;
  using ElementALoad = typename HcPreGemmSqrSumXe::ElementALoad;
  using Element = typename HcPreGemmSqrSumXe::Element;

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);
  int split_k = C.size(0);

  HcPreGemmSqrSumProblemShape shape;
  shape.M = M;
  shape.N = N;
  shape.K = K;

  typename Kernel::KernelArguments kernel_args{};
  kernel_args.shape = shape;

  kernel_args.ptr_A = reinterpret_cast<ElementALoad const*>(A.data_ptr());
  kernel_args.dA = cute::make_stride(K, cute::_1{});

  kernel_args.ptr_B = reinterpret_cast<Element const*>(B.data_ptr());
  kernel_args.dB = cute::make_stride(K, cute::_1{});

  kernel_args.ptr_C = reinterpret_cast<typename Kernel::ElementC*>(C.data_ptr());
  kernel_args.dC = cute::make_stride(N, cute::_1{});

  kernel_args.ptr_sqr_sum = sqr_sum.data_ptr<float>();
  kernel_args.ptr_sqr_sum_scratch = sqr_sum.data_ptr<float>();
  kernel_args.dSqsum = cute::make_stride(1, cute::_1{});

  typename Kernel::Arguments args{};
  args.kernel = kernel_args;
  args.mainloop = typename HcPreGemmSqrSumXe::CollectiveMainloop::Arguments{};
  args.split_k = split_k;
  return args;
}

inline void runHcPreGemmSqrSum(at::Tensor& C, at::Tensor& sqr_sum, const at::Tensor& A, const at::Tensor& B) {
  using Runner = cutlass::hc_pre_gemm_sqr_sum::device::HcPreGemmSqrSum<typename HcPreGemmSqrSumXe::Kernel>;

  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(0);
  int split_k = C.size(0);

  TORCH_CHECK(B.size(1) == K, "A.K (", K, ") must match B.K (", B.size(1), ") for GEMM");
  TORCH_CHECK(C.dim() == 3 && C.size(1) == M && C.size(2) == N, "C must be [n_splits, M, N]");
  TORCH_CHECK(
      sqr_sum.dim() == 2 && sqr_sum.size(0) == split_k && sqr_sum.size(1) == M,
      "sqr_sum must be [n_splits, M] matching C's leading dim");

  auto args = args_from_options(C, sqr_sum, A, B);

  Runner runner;
  auto status = runner.run(args, nullptr, c10::xpu::getCurrentXPUStream().queue());

  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM+SqrSum kernel failed with status: ", int(status));
}
