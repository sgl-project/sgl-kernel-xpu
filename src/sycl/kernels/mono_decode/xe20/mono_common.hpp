// SPDX-License-Identifier: Apache-2.0
//
// Mono decode (one resident kernel per decoder layer) for Qwen3.5 / Qwen3.6
// MoE (35B-A3B shapes) served by SGLang at TP2 on two Intel Arc Pro B70 (Xe2,
// 32 Xe cores): shapes, the layer dtype, the arguments in SGLang's layouts, the
// P2P mailbox and the Xe2 primitives shared by every layer kernel.
//
// Weights (SGLang block-FP8 checkpoint, used in place):
// - fp8 e4m3 linears [N][K] with 128x128 block scales [N / 128][K / 128] fp32:
//   GDN in_proj_qkvz / out_proj, attention qkv_proj / o_proj, the shared
//   expert, and the routed experts w13 [E][2 I][H] / w2 [E][H][I] with scales
//   [E][2 I / 128][H / 128] / [E][H / 128][I / 128] (SGLang FusedMoE layout).
// - 16-bit (the layer dtype DT): norms, router, shared-expert gate, GDN
//   in_proj_ba (b | a) and conv1d, the MTP fc.
// Caches (SGLang pools, used in place):
// - attention: K and V buffers [slots][1][256] each (token pitch 512 B),
//   addressed through a page table [requests][pages] of page indices;
// - GatedDeltaNet: conv state [slots][4096][3 (+ draft tokens)] (channel
//   major, taps innermost), recurrent state [slots][16][128 v][128 k] fp32.
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

namespace sgl_mono {

using half = sycl::half;
using h2 = sycl::vec<half, 2>;
using bf16 = sycl::ext::oneapi::bfloat16;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

constexpr int NWG = 32, WGS = 512, NSG = 32;
constexpr int HID = 2048, NE = 256, TOPK = 8,
              INTER = 256;        // per rank (TP2): expert / shared inter 512 / 2
constexpr int QB = 128;           // fp8 block size (N and K)
constexpr int KB_HID = HID / QB;  // K blocks of a hidden-size row (16)
constexpr float EPS = 1e-6f;
// GDN (per rank): 8 k heads, 16 v heads, 128 dims
constexpr int NKH = 8, NVH = 16, DK = 128, DV = 128;
constexpr int KD = NKH * DK, VD = NVH * DV, CONVD = 2 * KD + VD, QKVZ = CONVD + VD;
// full attention (per rank): 8 q heads, 1 kv head, head dim 256, partial RoPE
// 64 dims
constexpr int NQ = 8, HD = 256, QKV_ROWS = NQ * 2 * HD + 2 * HD;  // 4608
constexpr int GROWS = NQ * HD + 2 * HD;                           // gate + k + v rows
constexpr float SM_SCALE = 0.0625f * 1.4426950408889634f, THETA = 1.0e7f;
constexpr int GKV_MID = 2, GKV_ROWS = 2,
              GKV_CHUNKS = (GROWS - GKV_MID * NWG * NSG) / GKV_ROWS;  // 256
constexpr int SPW = INTER / NWG;                                      // shared inter rows per WG (8)
constexpr int SHR = 2 * INTER / NSG;                                  // shared gate|up rows per SG in the partial (16)
// MTP fc (16-bit, the same full [2048][4096] on both ranks; each rank computes
// half of the output rows)
constexpr int MTP_FC_IN = 2 * HID, MTP_FC_OUT = HID / 2;

enum { C_A = 0, C_B, C_C, C_O, C_GU, C_TKT, NCNT };  // GDN: A = qkvz, B = core ; ATTN: A = q, B = att, C = merge
constexpr int EPOCH = NCNT * 16;
// mailbox tag counters, one per kernel configuration (rows 1, 2, 4, 8 at MBT +
// 0 .. 3; rows 3, 6 at MBT2 + 0, 1), each with its own mailbox region:
// consecutive uses of a region carry consecutive tags, so a word left from an
// older use never matches the 16-bit tag
constexpr int MBT = EPOCH + 4, MBT2 = EPOCH + 16;
// two words: the address of the device's health words (mono_health), written
// at registration and read only by a wait that is taking long
constexpr int HEALTH_PTR = EPOCH + 28;
// health words: [0] FAILED (sticky: a wait gave up), [1] waits that gave up,
// [2] FAILED of the peer (written through P2P by a rank that gave up), and of
// the first wait that gave up: [3] the tag it waited for, [4] / [5] the word it
// last read (low / high half), [6] the word's offset in the mailbox region (in
// words from the region start), [7] the low half of the address of the
// layer's counters (which layer); [8] / [9] the address of the host copy of
// words 0 .. 7 (host memory, written once by the wait that marks the device
// failed, for a host thread to poll), or 0
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
  H_WORDS = 16
};
// two words after the health pointer: the wait deadline in cycles of the
// sub-group clock (~2.8 GHz), set per device (mono_set_deadline)
constexpr int DEADLINE = EPOCH + 30;
// two words: the address of the peer's health words (P2P mapping), or 0
constexpr int PEER_HEALTH_PTR = EPOCH + 32;
// two words: the address of this layer's mailbox region (for the health record)
constexpr int MBOX_PTR = EPOCH + 34;
constexpr int CNT_WORDS = EPOCH + 40;
constexpr uint64_t WAIT_CYCLES_DEFAULT = 14000000000ull;  // 5 s

// ----------------------------------------------------------------------------
// The layer's 16-bit dtype DT (half or bf16): hidden states, norm / router /
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

// ----------------------------------------------------------------------------
// arguments (e16: the layer's dtype DT; scales: fp32 128x128 block scales)
struct Ptrs {
  // common
  const e16 *hs_in, *res_in;  // layer input hidden / residual (zeros for layer 0)
  e16 *hid_out, *res_out;     // MoE output (after all-reduce #2), residual after the mixer
  const e16 *in_norm_w, *post_norm_w, *router_w, *sgate_w;
  // router / in_proj_ba in fp16 for the DPAS paths (batched and verify kernels):
  // the layer's own tensors in an fp16 layer, rounded copies in a bf16 one
  const e16 *router_h, *ba_h;
  // out_proj / o_proj [2048][2048], w13 [E][512][2048], w2 [E][2048][256],
  // shared w13 [512][2048], shared w2 [2048][256]
  const uint8_t *out_w, *w13, *w2, *sw13, *sw2;
  // [16][16], [E][4][16], [E][16][2], [4][16], [16][2]
  const float *out_s, *w13_s, *w2_s, *sw13_s, *sw2_s;
  // GDN
  const uint8_t* qkvz_w;  // [6144][2048] (q | k | v | z)
  const float* qkvz_s;    // [48][16]
  const e16* ba_w;        // [32][2048] (b | a), 16-bit
  const float* A_log;
  const e16 *conv_w, *dt_bias, *gnorm_w;  // conv_w [4096][4]
  e16* conv_cache;                        // [slots][4096][conv_ch] (slot stride conv_stride)
  float* ssm_cache;                       // [slots][16][128 v][128 k] (slot stride ssm_stride)
  int64_t conv_stride, ssm_stride;
  int conv_ch;           // elements per channel of a conv state slot (3 + draft tokens)
  const int* state_idx;  // [rows] state slot of each row (< 0: padding)
  // verify (SGLang scratch convention): the state after token t of request b
  // goes to intermediate slot inter_idx[b], token t
  float* inter_ssm;  // [slots][T][16][128][128]
  e16* inter_conv;   // [slots][T][4096][3]
  int64_t inter_ssm_stride, inter_ssm_tstride, inter_conv_stride, inter_conv_tstride;
  const int* inter_idx;  // [requests]
  // ATTN
  const uint8_t* qkv_w;  // [4608][2048]: per head q (256) | gate (256), then k, v
  const float* qkv_s;    // [36][16]
  const e16 *q_norm_w, *k_norm_w;
  e16 *kc, *vc;                       // K / V buffers [slots][256]
  int ps;                             // tokens per page (multiple of 16)
  const int *block_table, *seq_lens;  // page table [requests][bt_stride], sequence lengths (with the new tokens)
  int bt_stride;
  const int64_t *slot, *pos;  // out_cache_loc [rows], positions [rows]
  // MTP prologue
  const e16* fc_w;  // [2048][4096]
};

struct Scratch {
  uint32_t* cnt;     // CNT_WORDS
  float* qkv_raw;    // ATTN: QKV_ROWS fp32 GEMV outputs
  float* part_ml;    // ATTN: NWG * NQ * 2
  float* part_o;     // ATTN: NWG * NQ * HD
  half* attn_out;    // ATTN: NQ * HD (gated attention output)
  half* qkv;         // GDN : CONVD
  half* core;        // GDN : VD
  half* zbuf;        // GDN : VD
  half* ba;          // GDN : 2 * NVH
  float* part_rt;    // NWG * NE
  float* part_sh;    // NWG * 2 * INTER
  float* part_sg;    // NWG * 2
  half* hbuf;        // (TOPK + 1) * INTER
  e16* dummy_conv;   // GDN padding slot: CONVD * 8
  float* dummy_ssm;  // GDN padding slot: NVH * DV * DK
  uint64_t* mbox;    // words written by the peer card
  uint64_t* peer;    // the peer card's mbox (P2P mapping)
};

// bytes of one layer's scratch (everything except mbox / peer), 256-B aligned
// pieces
inline size_t scratch_bytes() {
  size_t n = 0;
  auto add = [&](size_t b) { n += (b + 255) & ~size_t(255); };
  add(CNT_WORDS * 4);
  add(QKV_ROWS * 4);
  add(NWG * NQ * 2 * 4);
  add((size_t)NWG * NQ * HD * 4);
  add(NQ * HD * 2);
  add(CONVD * 2);
  add(VD * 2);
  add(VD * 2);
  add(2 * NVH * 2);
  add((size_t)NWG * NE * 4);
  add((size_t)NWG * 2 * INTER * 4);
  add(NWG * 2 * 4);
  add((TOPK + 1) * INTER * 2);
  add(8 * CONVD * 2);
  add((size_t)NVH * DV * DK * 4);
  return n;
}
// carve a zero-initialized device buffer of scratch_bytes() into the Scratch
// pointers
inline Scratch scratch_carve(void* base, uint64_t* mbox, uint64_t* peer) {
  Scratch S{};
  uint8_t* p = static_cast<uint8_t*>(base);
  auto take = [&](size_t b) {
    void* r = p;
    p += (b + 255) & ~size_t(255);
    return r;
  };
  S.cnt = (uint32_t*)take(CNT_WORDS * 4);
  S.qkv_raw = (float*)take(QKV_ROWS * 4);
  S.part_ml = (float*)take(NWG * NQ * 2 * 4);
  S.part_o = (float*)take((size_t)NWG * NQ * HD * 4);
  S.attn_out = (half*)take(NQ * HD * 2);
  S.qkv = (half*)take(CONVD * 2);
  S.core = (half*)take(VD * 2);
  S.zbuf = (half*)take(VD * 2);
  S.ba = (half*)take(2 * NVH * 2);
  S.part_rt = (float*)take((size_t)NWG * NE * 4);
  S.part_sh = (float*)take((size_t)NWG * 2 * INTER * 4);
  S.part_sg = (float*)take(NWG * 2 * 4);
  S.hbuf = (half*)take((TOPK + 1) * INTER * 2);
  S.dummy_conv = (e16*)take(8 * CONVD * 2);
  S.dummy_ssm = (float*)take((size_t)NVH * DV * DK * 4);
  S.mbox = mbox;
  S.peer = peer;
  return S;
}
constexpr size_t MBOX_WORDS = 2 * HID;  // per layer, rows-1 region

// ----------------------------------------------------------------------------
// helpers
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
// A read-modify-write, not a store: it completes at the peer card's memory, so
// the word cannot linger in this card's cache (a store to the peer's memory may
// stay there until evicted and miss the peer's wait).
inline void mb_put(uint64_t* peer, uint64_t w) {
  (void)mb_ref(*peer).exchange(w);
}
inline void mb_send(uint64_t* peer, float v, uint32_t tag) {
  mb_put(peer, mb_word(sycl::bit_cast<uint32_t>(v), tag));
}
using h_ref = sycl::
    atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;
using hs_ref = sycl::
    atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>;
// The slow path of a mailbox wait (the first poll missed): polls until the
// word matches, or gives up when this device already failed (health FAILED,
// sticky: every later wait exits after a few hundred polls), when the peer
// reported a failure, or after the deadline. The wait that marks the device
// failed records why, tells the peer and publishes the record to the host
// copy. False: gave up.
inline bool mb_wait_slow(const uint64_t* mine, uint32_t tag, uint64_t& w, const uint32_t* cnt) {
  uint64_t t0 = syclex::clock<syclex::clock_scope::sub_group>();
  for (uint32_t n = 1;; ++n) {
    w = mb_ref(*const_cast<uint64_t*>(mine)).load();
    if (mb_match(w, tag)) return true;
    if ((n & 255u) == 0) {
      int* h = reinterpret_cast<int*>(*reinterpret_cast<const volatile uint64_t*>(cnt + HEALTH_PTR));
      if (h == nullptr) continue;  // not registered
      const uint64_t limit = *reinterpret_cast<const volatile uint64_t*>(cnt + DEADLINE);
      const bool peer_failed = hs_ref(h[H_PEER_FAILED]).load() != 0;
      const bool failed = h_ref(h[H_FAILED]).load() != 0 || peer_failed;
      // the sub-group clock is per hardware thread slot: a preempted and resumed
      // thread may read an earlier value; restart the deadline then
      const uint64_t now = syclex::clock<syclex::clock_scope::sub_group>();
      if ((int64_t)(now - t0) < 0) t0 = now;
      if (failed || now - t0 > (limit ? limit : WAIT_CYCLES_DEFAULT)) {
        h_ref(h[H_TIMEOUTS]).fetch_add(1);
        if (h_ref(h[H_FAILED]).exchange(1) == 0) {  // this wait marks the device failed
          if (!peer_failed) {                       // it timed out: record it and tell the peer
            h_ref(h[H_TAG]).store((int)(tag & 0xffffu));
            h_ref(h[H_SEEN_LO]).store((int)(uint32_t)w);
            h_ref(h[H_SEEN_HI]).store((int)(uint32_t)(w >> 32));
            const uint64_t* region =
                reinterpret_cast<const uint64_t*>(*reinterpret_cast<const volatile uint64_t*>(cnt + MBOX_PTR));
            h_ref(h[H_SLOT]).store(region ? (int)(mine - region) : -1);
            h_ref(h[H_LAYER]).store((int)(uint32_t) reinterpret_cast<uintptr_t>(cnt));
            // the peer's health words (P2P), so its waits give up at once
            int* ph = reinterpret_cast<int*>(*reinterpret_cast<const volatile uint64_t*>(cnt + PEER_HEALTH_PTR));
            if (ph != nullptr) hs_ref(ph[H_PEER_FAILED]).store(1);
          }
          int* host = reinterpret_cast<int*>(
              (uint64_t)(uint32_t)h_ref(h[H_HOST_LO]).load() | ((uint64_t)(uint32_t)h_ref(h[H_HOST_HI]).load() << 32));
          if (host != nullptr) {
            for (int k = H_TIMEOUTS; k <= H_LAYER; ++k)
              hs_ref(host[k]).store(h_ref(h[k]).load());
            hs_ref(host[H_PEER_FAILED]).store(peer_failed ? 1 : 0);
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            hs_ref(host[H_FAILED]).store(1);
          }
        }
        return false;
      }
    }
  }
}
// waits for this call's word (tag) in own mailbox word `mine`; on a give-up the
// word reads as all ones, a NaN in each 16- or 32-bit value, so the outputs of
// a failed step are NaN (and the device's health records the failure)
inline uint64_t mb_wait(const uint64_t* mine, uint32_t tag, const uint32_t* cnt) {
  uint64_t w = mb_ref(*const_cast<uint64_t*>(mine)).load();
  if (!mb_match(w, tag) && !mb_wait_slow(mine, tag, w, cnt)) w = ~0ull;
  return w;
}
inline float mb_recv(uint64_t* mine, uint32_t tag, const uint32_t* cnt) {
  return sycl::bit_cast<float>(mb_val(mb_wait(mine, tag, cnt)));
}

inline uint32_t fp8x2_lo(uint32_t w) {  // bytes 0,2 -> two fp16 = e4m3 / 256
  uint32_t t = w & 0x00ff00ffu;
  return ((t & 0x00800080u) << 8) | ((t & 0x007f007fu) << 7);
}
inline uint32_t fp8x2_hi(uint32_t w) {  // bytes 1,3
  uint32_t t = (w >> 8) & 0x00ff00ffu;
  return ((t & 0x00800080u) << 8) | ((t & 0x007f007fu) << 7);
}
inline h2 as_h2(uint32_t u) {
  return sycl::bit_cast<h2>(u);
}

// 16 fp8 weights . 16 activations stored permuted (x0,x2,x1,x3 per 4) in SLM.
inline float dot16_fp8(sycl::uint4 w, const uint32_t* xp) {
  sycl::uint4 xa = *reinterpret_cast<const sycl::uint4*>(xp);
  sycl::uint4 xb = *reinterpret_cast<const sycl::uint4*>(xp + 4);
  h2 acc = as_h2(fp8x2_lo(w.x())) * as_h2(xa.x());
  acc += as_h2(fp8x2_hi(w.x())) * as_h2(xa.y());
  acc += as_h2(fp8x2_lo(w.y())) * as_h2(xa.z());
  acc += as_h2(fp8x2_hi(w.y())) * as_h2(xa.w());
  acc += as_h2(fp8x2_lo(w.z())) * as_h2(xb.x());
  acc += as_h2(fp8x2_hi(w.z())) * as_h2(xb.y());
  acc += as_h2(fp8x2_lo(w.w())) * as_h2(xb.z());
  acc += as_h2(fp8x2_hi(w.w())) * as_h2(xb.w());
  return (float)acc[0] + (float)acc[1];
}
// 16 fp8 weights . 16 activations in natural order (registers)
inline float dot16_fp8_nat(sycl::uint4 w, const sycl::vec<half, 16>& x) {
  h2 acc = as_h2(fp8x2_lo(w.x())) * h2(x[0], x[2]);
  acc += as_h2(fp8x2_hi(w.x())) * h2(x[1], x[3]);
  acc += as_h2(fp8x2_lo(w.y())) * h2(x[4], x[6]);
  acc += as_h2(fp8x2_hi(w.y())) * h2(x[5], x[7]);
  acc += as_h2(fp8x2_lo(w.z())) * h2(x[8], x[10]);
  acc += as_h2(fp8x2_hi(w.z())) * h2(x[9], x[11]);
  acc += as_h2(fp8x2_lo(w.w())) * h2(x[12], x[14]);
  acc += as_h2(fp8x2_hi(w.w())) * h2(x[13], x[15]);
  return (float)acc[0] + (float)acc[1];
}
// 8 16-bit (DT) weights . 8 activations in natural order (fp16 in SLM), fp32
template <typename DT>
inline float dot8_w16(sycl::uint4 w, const half* x) {
  const sycl::vec<uint16_t, 8> wh = sycl::bit_cast<sycl::vec<uint16_t, 8>>(w);
  const sycl::vec<half, 8> xv = *reinterpret_cast<const sycl::vec<half, 8>*>(x);
  float a = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    a += f32<DT>(wh[i]) * (float)xv[i];
  return a;
}
template <int KCH>
inline void load_row(sycl::uint4 (&w)[KCH], const uint8_t* row, int lane) {
#pragma unroll
  for (int j = 0; j < KCH; ++j)
    w[j] = *reinterpret_cast<const sycl::uint4*>(row + j * 256 + lane * 16);
}
// The block scales of this lane's chunks of a K = KCH * 256 fp8 row: chunk j
// covers K bytes j * 256 + lane * 16 .. + 16, i.e. K block 2 j + lane / 8.
// srow: the row's [K / 128] scales.
template <int KCH>
inline void load_scales(float (&sc)[KCH], const float* srow, int lane) {
#pragma unroll
  for (int j = 0; j < KCH; ++j)
    sc[j] = srow[2 * j + (lane >> 3)];
}
// sum over a K = KCH * 256 fp8 row (registers) . x (permuted SLM), the lanes'
// chunks scaled by their K block, reduced over the sub-group: * 256 = output
template <int KCH>
inline float
dot_row_regs(sycl::sub_group sg, int lane, const sycl::uint4 (&w)[KCH], const uint32_t* xp, const float (&sc)[KCH]) {
  float sacc = 0.f;
#pragma unroll
  for (int j = 0; j < KCH; ++j)
    sacc += dot16_fp8(w[j], xp + (j * 256 + lane * 16) / 2) * sc[j];
  return sycl::reduce_over_group(sg, sacc, sycl::plus<float>());
}
// x[k] -> permuted slot (x0,x2,x1,x3 per 4): swap bits 0 and 1 of k.
inline void store_perm(half* xp, int k, half v) {
  xp[(k & ~3) | ((k & 1) << 1) | ((k >> 1) & 1)] = v;
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
// No device-scope acquire fence after the barrier: it lowers to an L1
// invalidate on every thread, which delays the hand-off. Not needed here: a WG
// reads another WG's hand-off data only after the matching wait and never
// touches those lines earlier in the same launch, and L1 starts clean at every
// launch.
// The counters and targets wrap at 2^32 (32 producers a call: after 2^27
// calls of a layer), so compare the signed difference.
inline void wait_for(sycl::nd_item<1> it, uint32_t* c, uint32_t target) {
  if (it.get_local_linear_id() == 0)
    while ((int32_t)(cnt_ref(*c).load(sycl::memory_order::relaxed) - target) < 0) {
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

// ---- Xe2 DPAS + 2D block loads (inline vISA / IGC builtins, as the sycl-tla
// arch primitives): joint_matrix element access (get_wi_data) does not build
// with IGC 2.38.
#if defined(__clang__)
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned int v8u __attribute__((ext_vector_type(8)));
typedef unsigned int v4u __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));
#else
// the host compiler pass (g++) parses the kernels too: a stand-in with the
// operations they use
template <typename T, int N>
struct hvec {
  T v[N];
  hvec() = default;
  explicit hvec(T x) {
    for (int i = 0; i < N; ++i)
      v[i] = x;
  }
  T& operator[](int i) {
    return v[i];
  }
  const T& operator[](int i) const {
    return v[i];
  }
  hvec& operator+=(const hvec& o) {
    for (int i = 0; i < N; ++i)
      v[i] += o.v[i];
    return *this;
  }
  hvec operator*(T s) const {
    hvec r;
    for (int i = 0; i < N; ++i)
      r.v[i] = v[i] * s;
    return r;
  }
};
using v8f = hvec<float, 8>;
using v8u = hvec<unsigned int, 8>;
using v4u = hvec<unsigned int, 4>;
using v2i = hvec<int, 2>;
#endif
#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL v8u __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(
    intptr_t base, int width_m1, int height_m1, int pitch_m1, v2i xy);
extern "C" SYCL_EXTERNAL v8u __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
    intptr_t base, int width_m1, int height_m1, int pitch_m1, v2i xy);
#endif
// d[8 rows x 16 cols] (lane = column, element i = row i) += a[8 x 16 half] .
// b[16 x 16 half]
//   a: dpas A layout, lane l element j = (A[2j + l/8][2(l%8)], A[2j +
//   l/8][2(l%8) + 1]) b: VNNI, lane n element p = (B[2p][n], B[2p + 1][n])
inline void dpas_hf(v8f& d, const v4u& a, const v8u& b) {
#ifdef __SYCL_DEVICE_ONLY__
  asm("{\n"
      ".decl DST     v_type=G type=f num_elts=128 alias=<%0,0>\n"
      ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"
      ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"
      "dpas.hf.hf.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"
      "}\n"
      : "+rw"(d)
      : "rw"(a), "rw"(b));
#else
  (void)d;
  (void)a;
  (void)b;
#endif
}
// the same for DT operands (attention: q, P, K and V in the KV cache dtype)
template <typename DT>
inline void dpas16(v8f& d, const v4u& a, const v8u& b) {
  if constexpr (std::is_same_v<DT, half>) {
    dpas_hf(d, a, b);
  } else {
#ifdef __SYCL_DEVICE_ONLY__
    asm("{\n"
        ".decl DST     v_type=G type=f num_elts=128 alias=<%0,0>\n"
        ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n"
        ".decl SRC2_UD v_type=G type=UD num_elts=64 alias=<%1,0>\n"
        "dpas.bf.bf.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)\n"
        "}\n"
        : "+rw"(d)
        : "rw"(a), "rw"(b));
#else
    (void)d;
    (void)a;
    (void)b;
#endif
  }
}
// K tile: 16 token rows x 16 dims at dims dw*2.. (dw in dwords) of the K
// buffer (token pitch HD halves) -> VNNI B of K^T (lane = token)
inline v8u load_k_tile(const void* row0, int dw) {
#ifdef __SYCL_DEVICE_ONLY__
  return __builtin_IB_subgroup_block_read_flat_transpose_u32_k8((intptr_t)row0, HD * 2 - 1, 15, HD * 2 - 1, v2i{dw, 0});
#else
  (void)row0;
  (void)dw;
  return v8u{};
#endif
}
// V tile: 16 token rows x 16 dims (VNNI-transformed) of a 128-dim half of the
// V buffer (token pitch HD halves)
inline v8u load_v_tile(const void* row0, int col) {
#ifdef __SYCL_DEVICE_ONLY__
  return __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
      (intptr_t)row0, 128 * 2 - 1, 15, HD * 2 - 1, v2i{col, 0});
#else
  (void)row0;
  (void)col;
  return v8u{};
#endif
}
inline uint32_t pack_h2(half a, half b) {
  return (uint32_t)sycl::bit_cast<uint16_t>(a) | ((uint32_t)sycl::bit_cast<uint16_t>(b) << 16);
}
inline uint32_t pack16(uint16_t a, uint16_t b) {
  return (uint32_t)a | ((uint32_t)b << 16);
}

// token t of a request -> its slot (page table of page indices, ps tokens a
// page); a 16-token tile never crosses a page (ps % 16 == 0)
inline int64_t kv_slot(const int* btab, int ps, int t) {
  return (int64_t)btab[t / ps] * ps + (t % ps);
}

// one head (256) -> (GemmaRMS + NeoX RoPE on dims 0..63 if nw) -> DT dst (and
// dst2).
template <typename DT>
inline void
head_norm_rope(sycl::sub_group sg, int lane, const float* src, const e16* nw, const float* cs, e16* dst, e16* dst2) {
  float y[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    y[i] = rnd<DT>(src[lane * 16 + i]);
  if (nw) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i)
      ss += y[i] * y[i];
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    const float rs = sycl::rsqrt(ss / HD + EPS);
#pragma unroll
    for (int i = 0; i < 16; ++i)
      y[i] = y[i] * rs * (1.f + ld16<DT>(nw, lane * 16 + i));
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float other = sycl::permute_group_by_xor(sg, y[i], 2);
      if (lane < 4) {
        const int fi = (lane & 1) * 16 + i;
        y[i] = lane < 2 ? y[i] * cs[fi] - other * cs[32 + fi] : y[i] * cs[fi] + other * cs[32 + fi];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    st16<DT>(dst, lane * 16 + i, y[i]);
    if (dst2) st16<DT>(dst2, lane * 16 + i, y[i]);
  }
}

// RoPE cos / sin of position p for the 32 rotated pairs (i < 32): Cody-Waite
// reduction by 2 pi, then cos / sin of |r| <= pi
inline void rope_cs(float p, int i, float* cs) {
  const float inv = 1.0f / sycl::pow(THETA, (float)(2 * i) / 64.f);
  const float f = p * inv;
  const float k = sycl::rint(f * 0.15915494309189535f);
  float r = sycl::fma(-k, 6.28125f, f);
  r = sycl::fma(-k, 1.9353071795864769e-3f, r);
  cs[i] = sycl::cos(r);
  cs[32 + i] = sycl::sin(r);
}

}  // namespace sgl_mono
