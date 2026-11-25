// SPDX-License-Identifier: Apache-2.0
//
// Xe2 decode stages, the shape-free parts every persistent decode kernel uses:
// the types, the 16-bit dtype accesses, uncached loads / streaming stores, and
// the hand-off between the resident work-groups of a launch (a monotonic
// device counter per hand-off point: signal / wait_for). No model constants:
// the shapes are the model's (src/sycl/models/<model>/xe20/).
#pragma once
#include <cmath>
#include <cstdint>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/experimental/clock.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

namespace sgl_kernel::stage::xe20 {

// Launch geometry of a decode kernel, a compile-time type the model's shape
// traits derive from: NWG work-groups (one a Xe core, all resident) of WGS
// work-items in sub-groups of 16, at most MAXR rows (tokens) a launch.
template <int kNwg, int kWgs, int kMaxRows>
struct Launch {
  static constexpr int NWG = kNwg;
  static constexpr int WGS = kWgs;
  static constexpr int NSG = kWgs / 16;  // sub-groups a work-group
  static constexpr int MAXR = kMaxRows;
};

using half = sycl::half;
using h2 = sycl::vec<half, 2>;
using bf16 = sycl::ext::oneapi::bfloat16;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

// ----------------------------------------------------------------------------
// A 16-bit dtype DT (half or bf16): hidden states, norm / router /
// conv / ba / MTP fc weights, KV cache, conv state. e16 has no arithmetic, so
// every access names DT. The fp16 scratch and SLM vectors hold values rounded
// to DT where SGLang rounds to its dtype (a bf16 value in the fp16 range is
// exact in fp16): the fp8 GEMVs keep their fp16 products in both dtypes.
struct e16 {
  uint16_t u;
};
template <typename DT>
inline float f32(uint16_t u) {
  return (float)sycl::bit_cast<DT>(u);
}
template <typename DT>
inline float ld16(const e16* p, size_t i) {
  return f32<DT>(p[i].u);
}
template <typename DT>
inline uint16_t bits(float v) {
  return sycl::bit_cast<uint16_t>((DT)v);
}
template <typename DT>
inline void st16(e16* p, size_t i, float v) {
  p[i].u = bits<DT>(v);
}
template <typename DT>
inline float rnd(float v) {
  return (float)(DT)v;
}
// v rounded to DT, held in fp16
template <typename DT>
inline half toh(float v) {
  if constexpr (std::is_same_v<DT, half>)
    return (half)v;
  else
    return (half)(float)(DT)v;
}

inline uint32_t uc_load32(const uint32_t* p) {  // L1-uncached plain load
  syclex::annotated_ptr<
      const uint32_t,
      decltype(syclex::properties{
          intelex::read_hint<intelex::cache_control<intelex::cache_mode::uncached, syclex::cache_level::L1>>})>
      a(p);
  return *a;
}
// state store with L1 streaming + L3 write-back (IGC LSC builtin)
#ifdef __SYCL_DEVICE_ONLY__
typedef unsigned int lsc_u4 __attribute__((ext_vector_type(4)));
extern "C" SYCL_EXTERNAL void
__builtin_IB_lsc_store_global_uint4(__attribute__((opencl_global)) lsc_u4* base, int off, lsc_u4 val, int cache);
#endif
inline void st_store8(float* p, const sycl::vec<float, 8>& v) {
#ifdef __SYCL_DEVICE_ONLY__
  auto* g = (__attribute__((opencl_global)) lsc_u4*)(__attribute__((opencl_global)) void*)(p);
  lsc_u4 a = {
      sycl::bit_cast<unsigned>(v[0]),
      sycl::bit_cast<unsigned>(v[1]),
      sycl::bit_cast<unsigned>(v[2]),
      sycl::bit_cast<unsigned>(v[3])};
  lsc_u4 b = {
      sycl::bit_cast<unsigned>(v[4]),
      sycl::bit_cast<unsigned>(v[5]),
      sycl::bit_cast<unsigned>(v[6]),
      sycl::bit_cast<unsigned>(v[7])};
  __builtin_IB_lsc_store_global_uint4(g, 0, a, 6);  // LSC_STCC_L1S_L3WB
  __builtin_IB_lsc_store_global_uint4(g, 1, b, 6);
#else
  (void)p;
  (void)v;
#endif
}

using cnt_ref = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>;
// NOTE: the release fence also waits for the thread's outstanding LOADS ->
// issue preloads after signal().
inline void signal(sycl::nd_item<1> it, uint32_t* c) {
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
  sycl::group_barrier(it.get_group());
  if (it.get_local_linear_id() == 0) cnt_ref(*c).fetch_add(1u, sycl::memory_order::release);
}
// The consumer polls with a relaxed load; the producer's release is the fence
// in signal() and the release RMW. After the loop the polling work-item takes
// one device-scope acquire fence, and the work-group barrier then orders the
// data for the rest of the work-group (the fence synchronizes with the
// producer's release, the barrier with the other threads; happens-before is
// transitive). One fence a WG lowers to one L1 invalidate; a fence in every
// thread would invalidate once per thread and delay the hand-off more.
// Measured against the same wait without the fence on two B70
// (bf16, 8 rotating layer copies, A B A B twice): GDN layers 0 us, attention
// layers of 1 row +0.9 us (1K context) / +1.3 us (32K context), i.e. +0.8 %
// each, 8 rows and the verify kernels within 1 us: about 0.2 % of a decode step.
// The counters and targets wrap at 2^32 (32 producers a call: after 2^27
// calls of a layer), so compare the signed difference.
inline void wait_for(sycl::nd_item<1> it, uint32_t* c, uint32_t target) {
  if (it.get_local_linear_id() == 0) {
    while ((int32_t)(cnt_ref(*c).load(sycl::memory_order::relaxed) - target) < 0) {
    }
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
  }
  sycl::group_barrier(it.get_group());
}
// in-WG barrier with SLM-only memory semantics (group_barrier also fences
// global memory = waits for loads)
inline void wg_bar(sycl::group<1> g) {
#if defined(__SYCL_DEVICE_ONLY__)
  __spirv_ControlBarrier(
      __spv::Scope::Workgroup,
      __spv::Scope::Workgroup,
      __spv::MemorySemanticsMask::AcquireRelease | __spv::MemorySemanticsMask::WorkgroupMemory);
#else
  sycl::group_barrier(g);
#endif
}

inline uint32_t pack_h2(half a, half b) {
  return (uint32_t)sycl::bit_cast<uint16_t>(a) | ((uint32_t)sycl::bit_cast<uint16_t>(b) << 16);
}
inline uint32_t pack16(uint16_t a, uint16_t b) {
  return (uint32_t)a | ((uint32_t)b << 16);
}

}  // namespace sgl_kernel::stage::xe20
