// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace sgl_kernel::stage::xe20::mailbox {

// Each half carries a 16-bit payload and the same 16-bit tag. This
// sentinel has different half-tags so it cannot match any invocation.
inline constexpr uint64_t never = 0x00000000ffff0000ull;
using Atomic = sycl::atomic_ref<
    uint64_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>;

inline uint64_t word2(uint32_t payload, uint32_t tag) {
  const uint64_t t = tag & 0xffffu;
  return (t << 48) | (uint64_t(payload >> 16) << 32) | (t << 16) | (payload & 0xffffu);
}

inline bool match(uint64_t word, uint32_t tag) {
  const uint32_t t = tag & 0xffffu;
  return uint32_t(word >> 48) == t && ((word >> 16) & 0xffffu) == t;
}

inline uint32_t payload(uint64_t word) {
  return (uint32_t(word >> 32) << 16) | (word & 0xffffu);
}

inline uint64_t poll(uint64_t* address) {
  return Atomic(*address).load();
}

inline void put(uint64_t* address, uint64_t word) {
#ifdef __SYCL_DEVICE_ONLY__
  const auto addr = reinterpret_cast<uint64_t>(address);
  asm volatile(
      "{\n"
      ".decl mailbox_addr v_type=G type=uq num_elts=16 alias=<%0,0>\n"
      ".decl mailbox_data v_type=G type=uq num_elts=16 alias=<%1,0>\n"
      "lsc_store.ugm.uc.uc.uc (M1, 16) flat[mailbox_addr]:a64 mailbox_data:d64\n"
      "}\n"
      :
      : "rw"(addr), "rw"(word)
      : "memory");
#else
  Atomic(*address).store(word);
#endif
}

}  // namespace sgl_kernel::stage::xe20::mailbox
