// SPDX-License-Identifier: Apache-2.0
//
// The arguments, scratch and counters of the Qwen3.5 / Qwen3.6 MoE decoder-layer
// kernels (Xe2): the stage arrival counters, the mailbox tag counters and the
// mailbox control block of a kernel, the weight / cache / metadata pointers
// (Ptrs, in SGLang's layouts) and the per-device scratch (Scratch).
//
// Stage code: every function that depends on the model's shape takes the
// traits Tr (Dims<Shape>, dims.hpp) as its first template argument.
#pragma once
#include "sgl_kernel/stage/xe20/common.hpp"
#include "sgl_kernel/stage/xe20/dims.hpp"
#include "sgl_kernel/stage/xe20/gemv.hpp"
#include "sgl_kernel/stage/xe20/p2p_mailbox.hpp"
#include "sgl_kernel/stage/xe20/xmx.hpp"

namespace sgl_kernel::stage::xe20 {

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
template <class Tr>
inline size_t scratch_bytes() {
  size_t n = 0;
  auto add = [&](size_t b) { n += (b + 255) & ~size_t(255); };
  add(CNT_WORDS * 4);
  add(Tr::QKV_ROWS * 4);
  add(Tr::NWG * Tr::NQ * 2 * 4);
  add((size_t)Tr::NWG * Tr::NQ * Tr::HD * 4);
  add(Tr::NQ * Tr::HD * 2);
  add(Tr::CONVD * 2);
  add(Tr::VD * 2);
  add(Tr::VD * 2);
  add(2 * Tr::NVH * 2);
  add((size_t)Tr::NWG * Tr::NE * 4);
  add((size_t)Tr::NWG * 2 * Tr::INTER * 4);
  add(Tr::NWG * 2 * 4);
  add((Tr::TOPK + 1) * Tr::INTER * 2);
  add(8 * Tr::CONVD * 2);
  add((size_t)Tr::NVH * Tr::DV * Tr::DK * 4);
  return n;
}
// carve a zero-initialized device buffer of scratch_bytes() into the Scratch
// pointers
template <class Tr>
inline Scratch scratch_carve(void* base, uint64_t* mbox, uint64_t* peer) {
  Scratch S{};
  uint8_t* p = static_cast<uint8_t*>(base);
  auto take = [&](size_t b) {
    void* r = p;
    p += (b + 255) & ~size_t(255);
    return r;
  };
  S.cnt = (uint32_t*)take(CNT_WORDS * 4);
  S.qkv_raw = (float*)take(Tr::QKV_ROWS * 4);
  S.part_ml = (float*)take(Tr::NWG * Tr::NQ * 2 * 4);
  S.part_o = (float*)take((size_t)Tr::NWG * Tr::NQ * Tr::HD * 4);
  S.attn_out = (half*)take(Tr::NQ * Tr::HD * 2);
  S.qkv = (half*)take(Tr::CONVD * 2);
  S.core = (half*)take(Tr::VD * 2);
  S.zbuf = (half*)take(Tr::VD * 2);
  S.ba = (half*)take(2 * Tr::NVH * 2);
  S.part_rt = (float*)take((size_t)Tr::NWG * Tr::NE * 4);
  S.part_sh = (float*)take((size_t)Tr::NWG * 2 * Tr::INTER * 4);
  S.part_sg = (float*)take(Tr::NWG * 2 * 4);
  S.hbuf = (half*)take((Tr::TOPK + 1) * Tr::INTER * 2);
  S.dummy_conv = (e16*)take(8 * Tr::CONVD * 2);
  S.dummy_ssm = (float*)take((size_t)Tr::NVH * Tr::DV * Tr::DK * 4);
  S.mbox = mbox;
  S.peer = peer;
  return S;
}

// the mailbox waits of a layer (its counters S.cnt hold the control block at CTL)
inline uint64_t mb_wait(const uint64_t* mine, uint32_t tag, const uint32_t* cnt) {
  return mailbox_wait<CTL>(mine, tag, cnt);
}
inline void mb_ar2(uint64_t* peer, uint64_t* mine, float& v0, float& v1, uint32_t tag, const uint32_t* cnt) {
  mailbox_ar2<CTL>(peer, mine, v0, v1, tag, cnt);
}

// ----------------------------------------------------------------------------
// Split launches (modes 1 and 2): the layer as one kernel a segment, as a
// bitwise reference of the one-kernel layer (mode 0). A segment ends where the
// one-kernel layer hands data over between work-groups through global memory
// (a wait_for) or, in mode 2, at an all-reduce, which the caller then runs on
// `ar` between the launches.
//
// A split kernel (SPLIT = true; seg a runtime argument) runs the whole layer
// code but only its segment's work: the stage signals, the writes of global
// outputs and of the caches, the mailbox all-reduces. Everything a segment
// needs that the one-kernel layer keeps in registers or SLM across the
// hand-off (weights and states prefetched before it, the input norm, the
// routing) is computed again from global memory by the code before the
// segment, as the one-kernel layer computes it, so the values are bitwise the
// same. The per-call counters (epoch, mailbox tag) advance after the last
// segment (split_advance), not inside one: the work-groups of a later segment
// kernel read them at their start.
enum SplitSeg : int {
  SEG_ALL = -1,  // mode 0: the whole layer
  SEG_F1 = 0,    // the mixer's first stage (GDN: in_proj qkvz ; attention: q, k, v rows)
  SEG_F2,        // GDN: conv1d + gated delta rule ; attention: the context split
  SEG_F3,        // attention: merge + gate (GDN: none)
  SEG_T1,        // the mixer's last stage + tail up to the router partials (mode 1)
  SEG_T1A,       // mode 2: ... up to the o_proj / out_proj partials (-> ar)
  SEG_T1B,       // mode 2: from the all-reduced o_proj / out_proj (ar) to the router partials
  SEG_S56,       // reduction + top-k + routed gate|up
  SEG_S7,        // down + all-reduce #2 (mode 1)
  SEG_S7A,       // mode 2: down partials (-> ar)
  SEG_S7B,       // mode 2: the all-reduced down (ar) -> hid_out
  SEG_COUNT
};
struct Split {
  int seg = SEG_ALL;
  float* ar = nullptr;  // mode 2: [rows][HID] fp32, the values to all-reduce / all-reduced
  // >= 0: the work-group this call runs as (a launch with fewer work-groups
  // than NWG runs the NWG work-groups' parts in turn); -1: the launch's own
  int vwg = -1;
};
// The segment guards are written out as (!SPLIT || X.seg == k), and mode 2's
// as (SPLIT && X.ar): with SPLIT false the compiler folds them while it emits
// the code, so the one-kernel layer's code is that of an unguarded layer (a
// guard function, folded only by the later passes, changed the register
// allocation of a one-kernel layer).
// a value as the mailbox all-reduce sends it (each rank's partial rounded to
// bf16, then the two added in fp32): mode 2 writes this into ar, and the
// caller's fp32 sum of the two ranks' ar is the mailbox's result
inline float ar_send(float v) {
  return (float)(bf16)v;
}

}  // namespace sgl_kernel::stage::xe20
