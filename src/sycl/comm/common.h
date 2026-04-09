#pragma once
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

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

  sycl::ext::oneapi::experimental::launch_config config(policy.get_range(), policy.get_launch_properties());
  auto cgf = [&](::sycl::handler& cgh) {
    auto KernelFunctor =
        compat::experimental::detail::build_kernel_functor<cutlass::device_kernel<Kernel>>(cgh, policy, params);
    sycl::ext::oneapi::experimental::detail::
        LaunchConfigAccess<sycl::nd_range<3>, decltype(policy.get_launch_properties())>
            ConfigAccess(config);
    cgh.parallel_for<KernelCur<Kernel>>(ConfigAccess.getRange(), ConfigAccess.getProperties(), KernelFunctor);
  };
  auto stream = at::xpu::getCurrentXPUStream();
  auto q = stream.queue();
  q.submit(cgf);
}

}  // namespace
