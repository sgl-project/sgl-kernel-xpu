// SPDX-License-Identifier: Apache-2.0
//
// Mono decode (one resident kernel per decoder layer) for Qwen3.5 / Qwen3.6
// MoE (35B-A3B shapes) served by SGLang at TP2 on two Intel Arc Pro B70 (Xe2,
// 32 Xe cores): the counters, the arguments in SGLang's layouts and the
// model-shaped helpers shared by every layer kernel. The shapes are in
// shape.hpp; the shape-free pieces (mailbox, hand-offs, GEMV and DPAS
// primitives) are the Xe2 stage library's (include/sgl_kernel/stage/xe20/).
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
#include "sgl_kernel/stage/xe20/common.hpp"
#include "sgl_kernel/stage/xe20/gemv.hpp"
#include "sgl_kernel/stage/xe20/p2p_mailbox.hpp"
#include "sgl_kernel/stage/xe20/xmx.hpp"
#include "shape.hpp"

namespace sgl_mono {

using namespace sgl_kernel::stage::xe20;  // NOLINT: the shape-free stage pieces

enum { C_A = 0, C_B, C_C, C_O, C_GU, C_TKT, NCNT };  // GDN: A = qkvz, B = core ; ATTN: A = q, B = att, C = merge
constexpr int EPOCH = NCNT * 16;
// mailbox tag counters, one per kernel configuration (rows 1, 2, 4, 8 at MBT +
// 0 .. 3; rows 3, 6 at MBT2 + 0, 1), each with its own mailbox region:
// consecutive uses of a region carry consecutive tags, so a word left from an
// older use never matches the 16-bit tag
constexpr int MBT = EPOCH + 4, MBT2 = EPOCH + 16;
// the MTP prologue's tag counters (its own scratch): one per row count 1, 2, 3,
// 4, 6, 8, each with its own mailbox region (mtp.hpp)
constexpr int MTPT = EPOCH + 18;
// the mailbox control block (stage p2p_mailbox.hpp): the address of the
// device's health words (mono_health, written at registration and read only by
// a wait that is taking long), the wait deadline (mono_set_deadline), the
// peer's health words (reserved: the host does not set them yet, so the
// peer's own wait gives up at its deadline) and this layer's mailbox region
constexpr int CTL = EPOCH + 28;
constexpr int HEALTH_PTR = CTL + CTL_HEALTH_PTR, DEADLINE = CTL + CTL_DEADLINE;
constexpr int PEER_HEALTH_PTR = CTL + CTL_PEER_HEALTH_PTR, MBOX_PTR = CTL + CTL_MBOX_PTR;
constexpr int CNT_WORDS = CTL + CTL_WORDS;
static_assert(CNT_WORDS == EPOCH + 40, "counter layout");

// ----------------------------------------------------------------------------
// arguments (e16: the layer's dtype DT; scales: fp32 128x128 block scales)
struct Ptrs {
  // common
  const e16 *hs_in, *res_in;  // layer input hidden / residual (zeros for layer 0)
  e16 *hid_out, *res_out;     // MoE output (after all-reduce #2), residual after the mixer
  const e16 *in_norm_w, *post_norm_w, *router_w, *sgate_w;
  // in_proj_ba in fp16 for the DPAS paths: the layer's own tensor in an fp16
  // layer, a rounded copy in a bf16 one (the router is read in the model's dtype
  // and rounded in the batched / verify kernels' loads)
  const e16* ba_h;
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
  int64_t inter_ssm_stride, inter_ssm_tstride, inter_conv_stride, inter_conv_cstride;
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

// the mailbox waits of a layer (its counters S.cnt hold the control block at CTL)
inline uint64_t mb_wait(const uint64_t* mine, uint32_t tag, const uint32_t* cnt) {
  return mailbox_wait<CTL>(mine, tag, cnt);
}
inline void mb_ar2(uint64_t* peer, uint64_t* mine, float& v0, float& v1, uint32_t tag, const uint32_t* cnt) {
  mailbox_ar2<CTL>(peer, mine, v0, v1, tag, cnt);
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
