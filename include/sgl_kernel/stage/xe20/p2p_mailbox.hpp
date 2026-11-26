// SPDX-License-Identifier: Apache-2.0
//
// Word format, transport and waits of the P2P mailbox the TP all-reduces of two
// ranks use: each rank writes its values into the peer's mailbox as tagged
// 64-bit words and polls its own mailbox for the peer's words of the same call.
#pragma once
#include "sgl_kernel/stage/xe20/common.hpp"

namespace sgl_kernel::stage::xe20 {

// health words: [0] FAILED (sticky: a wait gave up), [1] waits that gave up,
// [2] FAILED of the peer (written through P2P by a rank that gave up), and of
// the first wait that gave up: [3] the tag it waited for (a counter wait: the
// arrival count), [4] / [5] the word it last read (low / high half; a counter
// wait: the count), [6] the word's offset in the mailbox region (in words from
// the region start; the low half of the word's address when the region is not
// set; a counter wait: the counter's index), [7] the low half of the
// address of the user's counters (which user); [8] / [9] the address of the host copy of
// words 0 .. 7 and 10 (host memory, written once by the wait that marks the
// device failed, for a host thread to poll), or 0; [10] the kind of the first
// wait that gave up (H_KIND)
enum {
  H_FAILED = 0,
  H_TIMEOUTS,
  H_PEER_FAILED,
  H_TAG,
  H_SEEN_LO,
  H_SEEN_HI,
  H_SLOT,
  H_LAYER,
  H_HOST_LO,
  H_HOST_HI,
  H_KIND,  // of the first wait that gave up: 0 a mailbox wait, 1 a counter wait
  H_WORDS = 16
};
// The control block of a mailbox user: CTL_WORDS words at word CTL of its
// counters (the waits take the counters and CTL):
// [0..1] the address of the device's health words (0: not registered),
// [2..3] the wait deadline in cycles of the sub-group clock (0: the default),
// [4..5] the address of the peer's health words (P2P: its symmetric-memory
// copy; 0: the peer is not told, its own waits give up at their deadline),
// [6..7] the address of the user's mailbox region (for the health record; 0:
// the record keeps the word's address).
enum { CTL_HEALTH_PTR = 0, CTL_DEADLINE = 2, CTL_PEER_HEALTH_PTR = 4, CTL_MBOX_PTR = 6, CTL_WORDS = 12 };
constexpr uint64_t WAIT_CYCLES_DEFAULT = 5600000000ull;  // 2 s

using mb_ref = sycl::atomic_ref<
    uint64_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>;
// A peer write may land as two 4-byte halves (e.g. GPUs on different CPU root
// ports): each half carries the 16-bit tag (word = tag|hi16, tag|lo16), so a
// word that mixes two writes never matches.
inline uint64_t mb_word(uint32_t v, uint32_t tag) {
  const uint64_t t = tag & 0xffffu;
  return (t << 48) | ((uint64_t)(v >> 16) << 32) | (t << 16) | (v & 0xffffu);
}
inline bool mb_match(uint64_t w, uint32_t tag) {
  const uint32_t t = tag & 0xffffu;
  return (uint32_t)(w >> 48) == t && ((uint32_t)(w >> 16) & 0xffffu) == t;
}
inline uint32_t mb_val(uint64_t w) {
  return ((uint32_t)(w >> 32) << 16) | ((uint32_t)w & 0xffffu);
}
// initial value of every mailbox word: its two halves
// carry different tags, so it matches no tag
constexpr uint64_t MB_NEVER = 0x00000000ffff0000ull;
static_assert(((MB_NEVER >> 48) & 0xffffu) != ((MB_NEVER >> 16) & 0xffffu), "MB_NEVER must match no tag");
// Send of a word to the peer: one uncached 64-bit store (a posted PCIe write
// that reaches the peer's memory without a flush of this card's caches; a
// cached store may stay in this card's cache and miss the peer's wait).
// SIMD16 vISA: the kernels run 16-wide sub-groups.
inline void mb_put(uint64_t* peer, uint64_t w) {
#if defined(__SYCL_DEVICE_ONLY__)
  const uint64_t a = reinterpret_cast<uint64_t>(peer);
  asm volatile(
      "{\n"
      ".decl MB_A v_type=G type=uq num_elts=16 alias=<%0,0>\n"
      ".decl MB_D v_type=G type=uq num_elts=16 alias=<%1,0>\n"
      "lsc_store.ugm.uc.uc.uc (M1, 16) flat[MB_A]:a64 MB_D:d64\n"
      "}\n"
      :
      : "rw"(a), "rw"(w)
      : "memory");
#else
  (void)peer;
  (void)w;
#endif
}
// the same for a 32-bit word (the peer's health FAILED flag)
inline void uc_put32(int* peer, int v) {
#if defined(__SYCL_DEVICE_ONLY__)
  const uint64_t a = reinterpret_cast<uint64_t>(peer);
  const uint32_t d = (uint32_t)v;
  asm volatile(
      "{\n"
      ".decl UP_A v_type=G type=uq num_elts=16 alias=<%0,0>\n"
      ".decl UP_D v_type=G type=ud num_elts=16 alias=<%1,0>\n"
      "lsc_store.ugm.uc.uc.uc (M1, 16) flat[UP_A]:a64 UP_D:d32\n"
      "}\n"
      :
      : "rw"(a), "rw"(d)
      : "memory");
#else
  (void)peer;
  (void)v;
#endif
}
using h_ref = sycl::
    atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;
using hs_ref = sycl::
    atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>;
// Health check of a slow wait, every 256 polls: false while it may go on;
// true when it gives up: this device already failed (health FAILED, sticky:
// every later wait exits after a few hundred polls), the peer reported a
// failure, or the deadline passed. The wait that marks the device failed
// records why (tag: what it waited for, seen: what it last read, slot: the
// mailbox word's offset, or the counter's index for a counter wait; kind:
// H_KIND),
// tells the peer and publishes the record to the host copy.
template <int CTL>
inline bool wait_gives_up(const uint32_t* cnt, uint64_t& t0, uint32_t tag, uint64_t seen, int slot, int kind) {
  int* h = reinterpret_cast<int*>(*reinterpret_cast<const volatile uint64_t*>(cnt + CTL + CTL_HEALTH_PTR));
  if (h == nullptr) return false;  // not registered
  const uint64_t limit = *reinterpret_cast<const volatile uint64_t*>(cnt + CTL + CTL_DEADLINE);
  const bool peer_failed = hs_ref(h[H_PEER_FAILED]).load() != 0;
  const bool failed = h_ref(h[H_FAILED]).load() != 0 || peer_failed;
  // the sub-group clock is per hardware thread slot: a preempted and resumed
  // thread may read an earlier value; restart the deadline then
  const uint64_t now = syclex::clock<syclex::clock_scope::sub_group>();
  if ((int64_t)(now - t0) < 0) t0 = now;
  if (!failed && now - t0 <= (limit ? limit : WAIT_CYCLES_DEFAULT)) return false;
  h_ref(h[H_TIMEOUTS]).fetch_add(1);
  if (h_ref(h[H_FAILED]).exchange(1) == 0) {  // this wait marks the device failed
    if (!peer_failed) {                       // it timed out: record it and tell the peer
      h_ref(h[H_TAG]).store((int)tag);
      h_ref(h[H_SEEN_LO]).store((int)(uint32_t)seen);
      h_ref(h[H_SEEN_HI]).store((int)(uint32_t)(seen >> 32));
      h_ref(h[H_SLOT]).store(slot);
      h_ref(h[H_LAYER]).store((int)(uint32_t) reinterpret_cast<uintptr_t>(cnt));
      h_ref(h[H_KIND]).store(kind);
      // the peer's health words (P2P), so its waits give up at once
      int* ph = reinterpret_cast<int*>(*reinterpret_cast<const volatile uint64_t*>(cnt + CTL + CTL_PEER_HEALTH_PTR));
      if (ph != nullptr) uc_put32(ph + H_PEER_FAILED, 1);  // uncached, as mb_put
    }
    int* host = reinterpret_cast<int*>(
        (uint64_t)(uint32_t)h_ref(h[H_HOST_LO]).load() | ((uint64_t)(uint32_t)h_ref(h[H_HOST_HI]).load() << 32));
    if (host != nullptr) {
      for (int k = H_TIMEOUTS; k <= H_LAYER; ++k)
        hs_ref(host[k]).store(h_ref(h[k]).load());
      hs_ref(host[H_KIND]).store(h_ref(h[H_KIND]).load());
      hs_ref(host[H_PEER_FAILED]).store(peer_failed ? 1 : 0);
      sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
      hs_ref(host[H_FAILED]).store(1);
    }
  }
  return true;
}
// The slow path of a mailbox wait (the first poll missed): polls until the
// word matches or the wait gives up (wait_gives_up). False: gave up.
template <int CTL>
inline bool mailbox_wait_slow(const uint64_t* mine, uint32_t tag, uint64_t& w, const uint32_t* cnt) {
  uint64_t t0 = syclex::clock<syclex::clock_scope::sub_group>();
  for (uint32_t n = 1;; ++n) {
    w = mb_ref(*const_cast<uint64_t*>(mine)).load();
    if (mb_match(w, tag)) return true;
    if ((n & 255u) == 0) {
      const uint64_t* region =
          reinterpret_cast<const uint64_t*>(*reinterpret_cast<const volatile uint64_t*>(cnt + CTL + CTL_MBOX_PTR));
      const int slot = region ? (int)(mine - region) : (int)(uint32_t) reinterpret_cast<uintptr_t>(mine);
      if (wait_gives_up<CTL>(cnt, t0, tag & 0xffffu, w, slot, 0)) return false;
    }
  }
}
// The slow path of a counter wait (wait_for: another work-group's arrivals on
// this card): polls until counter `ci` of `cnt` reaches target or the wait
// gives up, e.g. when not all work-groups of the kernel are resident.
template <int CTL>
inline void counter_wait_slow(const uint32_t* cnt, int ci, uint32_t target) {
  uint64_t t0 = syclex::clock<syclex::clock_scope::sub_group>();
  uint32_t* c = const_cast<uint32_t*>(cnt) + ci * 16;
  for (uint32_t n = 1;; ++n) {
    const uint32_t v = cnt_ref(*c).load(sycl::memory_order::relaxed);
    if ((int32_t)(v - target) >= 0) return;
    if ((n & 255u) == 0 && wait_gives_up<CTL>(cnt, t0, target, v, ci, 1)) return;
  }
}
// waits for this call's word (tag) in own mailbox word `mine`; on a give-up the
// word reads as all ones, a NaN in each 16- or 32-bit value, so the outputs of
// a failed step are NaN (and the device's health records the failure)
template <int CTL>
inline uint64_t mailbox_wait(const uint64_t* mine, uint32_t tag, const uint32_t* cnt) {
  uint64_t w = mb_ref(*const_cast<uint64_t*>(mine)).load();
  if (!mb_match(w, tag) && !mailbox_wait_slow<CTL>(mine, tag, w, cnt)) w = ~0ull;
  return w;
}
// all-reduce of two values: one mailbox word of two bf16 + tag. Both ranks add
// the same bf16-rounded partials (a + b == b + a), so their sums are bitwise
// equal. A failed wait gives NaN (0xffff halves).
template <int CTL>
inline void mailbox_ar2(uint64_t* peer, uint64_t* mine, float& v0, float& v1, uint32_t tag, const uint32_t* cnt) {
  const bf16 a0 = (bf16)v0, a1 = (bf16)v1;
  const uint32_t v = (uint32_t)sycl::bit_cast<uint16_t>(a0) | ((uint32_t)sycl::bit_cast<uint16_t>(a1) << 16);
  mb_put(peer, mb_word(v, tag));
  const uint32_t p = mb_val(mailbox_wait<CTL>(mine, tag, cnt));
  v0 = (float)a0 + (float)sycl::bit_cast<bf16>((uint16_t)(p & 0xffffu));
  v1 = (float)a1 + (float)sycl::bit_cast<bf16>((uint16_t)(p >> 16));
}

}  // namespace sgl_kernel::stage::xe20
