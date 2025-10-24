#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>

#include "Utils.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/device_memory.h"
#include "kernels/moe/dispatch_policy.hpp"
#include "kernels/moe/xe_array_epilogue.hpp"
#include "kernels/moe/xe_array_mma.hpp"
#include "kernels/moe/xe_moe_gemm.hpp"

using namespace cute;

template <typename scalar_t>
struct MoERunner {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group
  template <typename Gemm>
  typename Gemm::Arguments args_from_options(
      const cutlass::KernelHardwareInfo& hw_info,
      const typename Gemm::ElementA* A_ptr,
      const typename Gemm::ElementB* B_ptr,
      typename Gemm::ElementC* D_ptr,
      const int gemm_N,
      const int gemm_K,
      const int* num_rows_per_expert_device,
      const int num_experts) {
    typename Gemm::Arguments arguments;
    decltype(arguments.fusion_args) fusion_args;

    fusion_args.alpha = 1;
    fusion_args.beta = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    // One alpha and beta per each group
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};

    using RasterOrderOptions =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;

    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        static_cast<const typename Gemm::ElementA**>((void*)A_ptr),
        static_cast<const typename Gemm::ElementB**>((void*)B_ptr),
        nullptr,  // static_cast<const ElementC**>((void*)D_ptr),
        static_cast<typename Gemm::ElementC**>((void*)D_ptr),
        fusion_args,
        hw_info,
        {1, RasterOrderOptions::AlongN},
        num_rows_per_expert_device,
        num_experts,
        gemm_N,
        gemm_K};

    return arguments;
  }

  void
  run(sycl::queue queue,
      const scalar_t* activations,
      const scalar_t* weights,
      scalar_t* outputs,
      const int gemm_n,
      const int gemm_k,
      const int* num_rows_per_expert_device,
      const int num_experts) {
    // The KernelHardwareInfo struct holds the number of EUs on the GPU with a
    // given device ID. This information is used by the underlying kernel.
    cutlass::KernelHardwareInfo hw_info;
    // Change device_id to another value if you are running on a machine with
    // multiple GPUs and wish to use a GPU other than that with device ID 0.
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
    using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

    // Workgroup-level tile
    using TileShape = Shape<_256, _256, _32>;

    using TiledMma =  // M=8,N=16,K=16, D=f32,A=bf16,B=bf16,C=f32
        typename TiledMMAHelper<
            MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
            Layout<TileShape>,
            Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

    constexpr int PipelineStages = 2;
    // Dispatch to grouped gemm algorithm
    using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16MoE<PipelineStages, cutlass::gemm::KernelXeMoEGEMM>;
    using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;

    // ScaledAcc needs to be supported in xe_builder.inl and xe_callbacks.cpp
    // This is a workaround
    using EpilogueOp = cutlass::epilogue::fusion::
        LinearCombination<float_t, float_t, float_t, float_t, cutlass::FloatRoundStyle::round_to_nearest>;
    using CopyOpG2R = XE_2D_U32x8x16_LD_N;
    using CopyOpR2G = XE_2D_U16x8x16_ST_N;

    using Stride = std::conditional_t<cute::is_tuple_v<std::remove_pointer_t<LayoutC>>, LayoutC, LayoutC*>;
    using FusionCallbacks = typename cutlass::epilogue::collective::detail::FusionOpInfo<
        EpilogueOp>::template FusionCallbacks<cutlass::epilogue::IntelXeXMX16Group, TileShape, TileShape, CopyOpG2R>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveEpilogue<
        cutlass::epilogue::IntelXeXMX16MoE,
        TileShape,
        scalar_t,
        Stride,
        scalar_t,
        Stride,
        FusionCallbacks,
        CopyOpG2R,
        void,
        void,
        CopyOpR2G,
        void,
        void>;

    // Mainloop
    using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
        GEMMDispatchPolicy,
        TileShape,
        scalar_t,
        cutlass::gemm::TagToStrideA_t<LayoutA*>,
        scalar_t,
        cutlass::gemm::TagToStrideB_t<LayoutB*>,
        TiledMma,
        GmemTiledCopyA,
        void,
        void,
        cute::identity,  // A
        GmemTiledCopyB,
        void,
        void,
        cute::identity  // B
        >;

    using GemmKernel = cutlass::gemm::kernel::
        GemmMoEUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::GroupScheduler>;

    using Gemm = cutlass::gemm::device::GemmMoEUniversalAdapter<GemmKernel>;

    Gemm gemm_op;
    auto arguments = args_from_options<Gemm>(
        hw_info, activations, weights, outputs, gemm_n, gemm_k, num_rows_per_expert_device, num_experts);
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    TORCH_CHECK(gemm_op.can_implement(arguments) == cutlass::Status::kSuccess, "GEMM configuration not supported.");

    TORCH_CHECK(
        gemm_op.initialize(arguments, workspace.get()) == cutlass::Status::kSuccess, "Failed to initialize GEMM.");

    // Run the GEMM
    TORCH_CHECK(gemm_op.run(&queue) == cutlass::Status::kSuccess, "Failed to run GEMM.");
  }
};

void moe_grouped_mm_nt(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& weights,
    const torch::Tensor& total_rows_for_experts,
    const int64_t n_experts) {
  int total_m = weights.sizes()[0];
  int gemm_k = activations.sizes()[1];
  auto weights_shape = weights.sizes().vec();
  int gemm_n = weights.sizes()[2];

  TORCH_CHECK(weights_shape.size() == 3, "weights must be 3D");
  TORCH_CHECK(weights_shape[0] == n_experts, "weights must have n_experts as the first dimension");
  TORCH_CHECK(weights_shape[1] == gemm_k, "weights must have the same size as matrix_a in the second dimension");
  TORCH_CHECK(
      weights_shape[0] == total_rows_for_experts.size(0),
      "rows_for_experts must have the same size as the first dimension of weights");
  TORCH_CHECK(output.sizes()[0] == total_m, "output must have the same number of rows as weights");
  TORCH_CHECK(output.sizes()[1] == gemm_n, "output must have the same number of columns as weights");
  TORCH_CHECK(n_experts % 8 == 0, "n_experts must be a multiple of 8 for the current implementation");
  TORCH_CHECK(
      activations.scalar_type() == weights.scalar_type(), "activations and weights must have the same data type");
  TORCH_CHECK(
      activations.scalar_type() == at::ScalarType::Half || activations.scalar_type() == at::ScalarType::BFloat16,
      "Only float16 and bfloat16 are supported in moe_grouped_mm_nt");

  DISPATCH_FLOAT_TYPES(activations.scalar_type(), "moe_grouped_mm_nt", [&] {
    auto stream = at::xpu::getCurrentXPUStream();
    auto queue = stream.queue();

    using Kernel = MoERunner<scalar_t>;
    Kernel kernel;
    kernel.run(
        queue,
        activations.data_ptr<scalar_t>(),
        weights.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        gemm_n,
        gemm_k,
        total_rows_for_experts.data_ptr<int>(),
        n_experts);
  });
}
