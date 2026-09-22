#pragma once
#include <c10/util/Exception.h>  // TORCH_CHECK

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "cutlass/device_kernel.h"
namespace {

// dispatch bool
#define AT_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...) \
  [&] {                                          \
    if (BOOL_V) {                                \
      constexpr bool BOOL_NAME = true;           \
      return __VA_ARGS__();                      \
    } else {                                     \
      constexpr bool BOOL_NAME = false;          \
      return __VA_ARGS__();                      \
    }                                            \
  }()

// dispatch bool
#define AT_DISPATCH_BOOL_NO_RETURN(BOOL_V, BOOL_NAME, ...) \
  if (BOOL_V) {                                            \
    constexpr bool BOOL_NAME = true;                       \
    __VA_ARGS__;                                           \
  } else {                                                 \
    constexpr bool BOOL_NAME = false;                      \
    __VA_ARGS__;                                           \
  }

// Soft-cap (attn_logit_softcapping) is Gemma-2-only (head_dim 128/256), so the
// Softcap=true kernel variant is only instantiated for those head dims on the
// non-fp8 KV path; every other head dim and the fp8 path force Softcap=false and
// reject softcap>0. `fn` is invoked with a std::bool_constant softcap tag.
template <int HeadDim, bool IsFp8, typename Fn>
void dispatch_softcap(float softcap, Fn&& fn) {
  if constexpr (IsFp8) {
    TORCH_CHECK(softcap == 0.0f, "logit soft-cap is not supported with an fp8 KV cache");
    fn(std::false_type{});
  } else if constexpr (HeadDim == 128 || HeadDim == 256) {
    if (softcap != 0.0f) {
      fn(std::true_type{});
    } else {
      fn(std::false_type{});
    }
  } else {
    TORCH_CHECK(softcap == 0.0f, "logit soft-cap is only compiled for head_dim 128 and 256");
    fn(std::false_type{});
  }
}

template <typename Kernel>
class KernelCur {};

template <typename Kernel, int GrfSize = 256>
static void launch(typename Kernel::Params params) {
  static_assert(GrfSize == 128 || GrfSize == 256, "GRF size must be 128 or 256");

  compat::dim3 const block = Kernel::get_block_shape();
  compat::dim3 const grid = Kernel::get_grid_shape(params);

  // configure smem size and carveout
  int smem_size = Kernel::SharedStorageSize;

  const auto sycl_block = compat::dim3(block.x, block.y, block.z);
  const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

  using namespace compat::experimental;
  compat::experimental::launch_properties launch_props{
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
  };

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<GrfSize>};

  compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
  auto stream = at::xpu::getCurrentXPUStream();
  auto q = stream.queue();
  compat::experimental::launch<cutlass::device_kernel<Kernel>, KernelCur<Kernel>>(policy, q, params);
}

}  // namespace
