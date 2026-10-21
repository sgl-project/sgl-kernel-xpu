#pragma once
#include <cfloat>
#include <cute/tensor.hpp>
#include <random>

#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "moe_array_epilogue.hpp"
#include "moe_array_mma.hpp"
#include "moe_callbacks.hpp"
#include "moe_gemm_array_cooperative.hpp"
#include "moe_tile_scheduler.hpp"

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per
                                                                              // group

namespace gpu::cutlass_kernel {
namespace grouped_gemm {

// Build the shared MoE epilogue, persistent kernel, and device adapter around
// an already-constructed mainloop. Most policies use GenerateGemm below to
// construct a stock CUTLASS mainloop; fused formats such as NVFP4 reuse the
// same kernel template while supplying their own in-register conversion
// mainloop.
template <
    class CollectiveMainloop_,
    class ElementAccumulator,
    class ElementOutput,
    class LayoutC,
    class LayoutD,
    class TiledMma,
    class TileShape,
    class EpilogueDispatchPolicy,
    class EpilogueOp>
class GenerateGemmFromCollective_xe35 {
 public:
  using FusionCallBacks = cutlass::epilogue::fusion::
      FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,
      ElementAccumulator,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallBacks,
      void,
      void>;
  using CollectiveMainloop = CollectiveMainloop_;
  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::GroupScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

} }
