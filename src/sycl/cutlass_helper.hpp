#pragma once

#include <sycl/accessor.hpp>
#include "sycl/ext/oneapi/experimental/enqueue_functions.hpp"
#include "sycl/ext/oneapi/properties/properties.hpp"
#include <sycl/event.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/queue.hpp>
#include <sycl/range.hpp>
#include <sycl/reduction.hpp>

#include <syclcompat/defs.hpp>
#include <syclcompat/device.hpp>
#include <syclcompat/dims.hpp>
#include <syclcompat/traits.hpp>
#include <syclcompat/device.hpp>
#include <syclcompat/dims.hpp>
#include <syclcompat/launch_policy.hpp>

namespace sycl_exp = sycl::ext::oneapi::experimental;

template <auto F, typename Range, typename KProps, bool HasLocalMem,
          typename... Args>
struct KernelFunctor {
  KernelFunctor(KProps kernel_props, Args... args)
      : _kernel_properties{kernel_props},
        _argument_tuple(std::make_tuple(args...)) {}

  KernelFunctor(KProps kernel_props, sycl::local_accessor<char, 1> local_acc,
                Args... args)
      : _kernel_properties{kernel_props}, _local_acc{local_acc},
        _argument_tuple(std::make_tuple(args...)) {}

  auto get(sycl_exp::properties_tag) const { return _kernel_properties; }

  __syclcompat_inline__ void
  operator()(syclcompat::detail::range_to_item_t<Range>) const {
    if constexpr (HasLocalMem) {
      char *local_mem_ptr = static_cast<char *>(
          _local_acc.template get_multi_ptr<sycl::access::decorated::no>()
              .get());
      apply_helper(
          [lmem_ptr = local_mem_ptr](auto &&...args) {
            [[clang::always_inline]] F(args..., lmem_ptr);
          },
          _argument_tuple);
    } else {
      apply_helper([](auto &&...args) { [[clang::always_inline]] F(args...); },
                   _argument_tuple);
    }
  }

  KProps _kernel_properties;
  std::tuple<Args...> _argument_tuple;
  std::conditional_t<HasLocalMem, sycl::local_accessor<char, 1>, std::monostate>
      _local_acc; // monostate for empty type
};


template <auto F, typename LaunchPolicy, typename... Args>
sycl::event launch(LaunchPolicy launch_policy, sycl::queue q, Args... args) {
  static_assert(syclcompat::args_compatible<LaunchPolicy, F, Args...>,
                "Mismatch between device function signature and supplied "
                "arguments. Have you correctly handled local memory/char*?");

  sycl_exp::launch_config config(launch_policy.get_range(),
                                 launch_policy.get_launch_properties());

  return sycl_exp::submit_with_event(q, [&](sycl::handler &cgh) {
    auto KernelFunctor = build_kernel_functor<F>(cgh, launch_policy, args...);
    if constexpr (syclcompat::detail::is_range_v<
                      typename LaunchPolicy::RangeT>) {
      parallel_for(cgh, config, KernelFunctor);
    } else {
      static_assert(
          syclcompat::detail::is_nd_range_v<typename LaunchPolicy::RangeT>);
      nd_launch(cgh, config, KernelFunctor);
    }
  });
}
