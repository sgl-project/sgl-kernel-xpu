// SPDX-License-Identifier: Apache-2.0
// Adapted from Omni omni_xpu_kernel/cute/cute_fmha_torch.cpp (Apache-2.0),
// source revision a69d2b903b9cdff9d285bc13e53d41338bd61da8.
// Dense D128 self-attention for diffusion workloads.
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/device_kernel.h"
#include "cute/util/compat.hpp"

#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"
using namespace cute;

namespace {

// Submit the dtype-specific kernel tag on the current PyTorch XPU queue.
template <typename Kernel, int GrfSize>
class SGLD_KERNEL_TAG {};

template <typename Kernel, int GrfSize = 256>
static void launch_on_torch_queue(typename Kernel::Params params) {
  static_assert(GrfSize == 128 || GrfSize == 256, "GRF size must be 128 or 256");

  compat::dim3 const block = Kernel::get_block_shape();
  compat::dim3 const grid = Kernel::get_grid_shape(params);
  int smem_size = Kernel::SharedStorageSize;

  const auto sycl_block = compat::dim3(block.x, block.y, block.z);
  const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  compat::experimental::launch_properties launch_props{
      syclex::work_group_scratch_size(smem_size),
  };
  compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<GrfSize>};
  compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};

  syclex::launch_config config(policy.get_range(), policy.get_launch_properties());
  auto cgf = [&](::sycl::handler& cgh) {
    auto KernelFunctor =
        compat::experimental::detail::build_kernel_functor<cutlass::device_kernel<Kernel>>(cgh, policy, params);
    syclex::detail::LaunchConfigAccess<sycl::nd_range<3>, decltype(policy.get_launch_properties())>
        ConfigAccess(config);
    cgh.parallel_for<SGLD_KERNEL_TAG<Kernel, GrfSize>>(
        ConfigAccess.getRange(), ConfigAccess.getProperties(), KernelFunctor);
  };
  auto stream = at::xpu::getCurrentXPUStream();
  auto q = stream.queue();
  q.submit(cgf);
}

static int checked_int(int64_t value, const char* label) {
  TORCH_CHECK(
      value >= 0 && value <= std::numeric_limits<int>::max(), label,
      " exceeds the CUTE int32 index range: ", value);
  return static_cast<int>(value);
}

// Dense D128 tile policy for BMG.
template <typename Element, int MmaK = 32>
struct D128TileKernel {
  static constexpr int QTile = 256, KvTile = 32, VTile = 32, HeadDim = 128;
  static constexpr int SubgroupLayoutQ = 16, PipelineStages = 2, GrfSize = 256;
  using ShapeQK = Shape<Int<QTile>, Int<KvTile>, Int<MmaK>>;
  using ShapePV = Shape<Int<QTile>, Int<VTile>, Int<KvTile>>;
  using ShapeOutput = Shape<Int<QTile>, Int<HeadDim>>;
  using SubgroupLayoutQK = Layout<Shape<Int<SubgroupLayoutQ>, _1, _1>>;

  using ElementQ = Element;
  using ElementK = Element;
  using ElementV = Element;
  using ElementO = Element;   // output dtype == input dtype (fp16/bf16)

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;

  static constexpr int SGTileQ =
      get<0>(shape_div(ShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, Element>;
  using SubgroupLayoutPV =
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

  using TiledMMAQK =
      typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<ShapeQK>, SubgroupLayoutQK>::TiledMMA;
  using TiledMMAPV =
      typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<ShapePV>, SubgroupLayoutPV>::TiledMMA;
  static constexpr int VTiles = get<1>(ShapeOutput{}) / get<1>(ShapePV{});

  using TensorQ = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideQ>>(1), StrideQ{})));
  using TensorK = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideK>>(1), StrideK{})));
  using TensorV = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideV>>(1), StrideV{})));
  using TensorO = decltype(make_tensor(make_gmem_ptr((Element*)nullptr),
                            make_layout(repeat<rank_v<StrideO>>(1), StrideO{})));
  using StrideScale = Stride<_1, int, int, int>;
  using TensorScale = decltype(make_tensor(make_gmem_ptr((float*)nullptr),
                              make_layout(repeat<4>(1), StrideScale{})));
  using TensorK_cache = TensorK;
  using TensorV_cache = TensorV;

  using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
  using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
      MainloopDispatchPolicy, false, false, false, false, false, false,
      TiledMMAQK, TiledMMAPV, VTiles,
      TensorQ, TensorK, TensorV, TensorScale, TensorScale, TensorScale, TensorK_cache, TensorV_cache,
      void, void, void, void, void>;

  using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
      CollectiveMainloop, ShapeOutput, TensorO, void>;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<false>;
  using Kernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
      ProblemShapeType, CollectiveMainloop, CollectiveEpilogue,
      cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>>;
};

template <typename Element, int MmaK = 32>
void run_d128_tile(
    const void* q_ptr, const void* k_ptr, const void* v_ptr, void* o_ptr,
    int B, int H, int Lq, int Lkv, int D, float scale) {
  using KT = D128TileKernel<Element, MmaK>;
  using K    = typename KT::Kernel;
  using PS   = typename KT::ProblemShapeType;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  PS shape;
  shape.batch = B;
  shape.num_heads_q = H;
  shape.num_heads_kv = H;
  shape.seq_len_qo = Lq;
  shape.seq_len_kv = Lkv;
  shape.seq_len_kv_cache = 0;
  shape.head_size_qk = D;
  shape.head_size_vo = D;

  // Logical CuTe modes: Q/K/O=(seq,dim,head,batch), V=(dim,seq,head,batch).
  const int HD = checked_int(static_cast<int64_t>(H) * D, "H*D");
  const int LqHD = checked_int(static_cast<int64_t>(Lq) * H * D, "Lq*H*D");
  const int LkvHD = checked_int(static_cast<int64_t>(Lkv) * H * D, "Lkv*H*D");
  typename KT::StrideQ stride_Q = cute::make_stride(HD, _1{}, D, LqHD);
  typename KT::StrideK stride_K = cute::make_stride(HD, _1{}, D, LkvHD);
  typename KT::StrideV stride_V = cute::make_stride(_1{}, HD, D, LkvHD);
  typename KT::StrideO stride_O = cute::make_stride(HD, _1{}, D, LqHD);

  typename K::Arguments arguments{
      {
          shape,
          static_cast<const Element*>(q_ptr), stride_Q,
          static_cast<const Element*>(k_ptr), stride_K,
          static_cast<const Element*>(v_ptr), stride_V,
          static_cast<Element*>(o_ptr),       stride_O,
          nullptr, {}, nullptr, {}, nullptr, {}, 1.f, 1.f, 1.f, 32,
          nullptr, stride_K,   // k_cache
          nullptr, stride_V,   // v_cache
      },
      {scale, nullptr, 0, nullptr},
      {},
      hw_info};

  size_t workspace_size = K::get_workspace_size(arguments);
  auto opts = at::TensorOptions().dtype(at::kByte).device(c10::Device(c10::kXPU, c10::xpu::current_device()));
  at::Tensor workspace = at::empty({(long)workspace_size}, opts);

  TORCH_CHECK(K::can_implement(arguments),
              "diffusion_sdp: can_implement failed (bad problem shape)");
  K::initialize_workspace(arguments, workspace.data_ptr());
  auto kernel_params = K::to_underlying_arguments(arguments, workspace.data_ptr());
  launch_on_torch_queue<K, KT::GrfSize>(kernel_params);
}

// Dense self-attention operator.
at::Tensor diffusion_sdp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "diffusion_sdp: expect [B,L,H,D]");
  // All three operands must be on XPU and share q's dtype — the kernel takes raw
  // data_ptr()s and reinterprets them as q's element type, so a CPU tensor or a
  // dtype mismatch would feed invalid pointers / misread data.
  TORCH_CHECK(q.device().is_xpu() && k.device().is_xpu() && v.device().is_xpu(),
              "diffusion_sdp: q, k, v must all be XPU tensors (got ",
              q.device(), ", ", k.device(), ", ", v.device(), ")");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
              "diffusion_sdp: q, k, v must share dtype (got ",
              q.scalar_type(), ", ", k.scalar_type(), ", ", v.scalar_type(), ")");
  // Public layout is [B, L, H, D]. Materialize strided views before launch;
  // the kernel and its output use the same contiguous layout.
  const int B = checked_int(q.size(0), "batch");
  const int Lq = checked_int(q.size(1), "query length");
  const int H = checked_int(q.size(2), "head count");
  const int D = checked_int(q.size(3), "head dimension");
  const int Lkv = checked_int(k.size(1), "key/value length");
  TORCH_CHECK(B == 1, "diffusion_sdp: only B==1 supported (got ", B, ")");
  TORCH_CHECK(D == 128, "diffusion_sdp: only head_dim==128 supported (got ", D, ")");
  TORCH_CHECK(Lq == Lkv,
              "diffusion_sdp: only self-attention with equal q/kv lengths is supported (got ",
              Lq, " and ", Lkv, ")");
  TORCH_CHECK(k.size(0) == B && v.size(0) == B, "diffusion_sdp: batch mismatch");
  TORCH_CHECK(k.size(2) == H && v.size(2) == H,
              "diffusion_sdp: q,k,v must share num_heads (got ", H, ",",
              k.size(2), ",", v.size(2), ")");
  TORCH_CHECK(k.size(3) == D && v.size(3) == D,
              "diffusion_sdp: q,k,v must share head_dim");
  TORCH_CHECK(v.size(1) == Lkv,
              "diffusion_sdp: k,v seq_len must match (got ", Lkv, ",",
              v.size(1), ")");

  TORCH_CHECK(k.device() == q.device() && v.device() == q.device(), "q/k/v must share device");
  TORCH_CHECK(Lq > 0 && H > 0, "sequence length and head count must be positive");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "only fp16/bf16 supported");
  c10::DeviceGuard guard(q.device());
  auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
  at::Tensor o = at::empty_like(qc);
  const float scale = 1.0f / std::sqrt((float)D);

#if SGLD_ATTENTION_BF16
#if SYCL_INTEL_TARGET == 20
  // Preserve the verified BF16 Z-Image MMA-K16 specialization.
  if (Lq == 4128 && H == 30) {
    run_d128_tile<cutlass::bfloat16_t, 16>(
        qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), o.data_ptr(), B, H, Lq,
        Lkv, D, scale);
    return o;
  }
#endif
  run_d128_tile<cutlass::bfloat16_t>(
      qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), o.data_ptr(), B, H, Lq,
      Lkv, D, scale);
#else
  run_d128_tile<cutlass::half_t>(
      qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), o.data_ptr(), B, H, Lq,
      Lkv, D, scale);
#endif
  return o;
}


}  // namespace
