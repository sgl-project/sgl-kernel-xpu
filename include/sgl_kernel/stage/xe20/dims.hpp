// SPDX-License-Identifier: Apache-2.0
//
// The traits of the Qwen3.5 / Qwen3.6 MoE decoder-layer stages: Dims<Base>
// derives, from the model's base shape (Base: Launch<NWG, WGS, MAXR> plus the
// per-rank sizes below), the sizes the stages compute with and the layouts of
// their SLM, scratch and mailboxes. The stages are written for this layer
// structure (GatedDeltaNet / full-attention mixer, top-k MoE with a shared
// expert, TP2 in-kernel all-reduces); a model of another structure has its own
// stages. Base provides: H, EPS, NE, TOPK, INTER, QB, NKH, NVH, DK, DV, NQ, HD,
// THETA, MTP_FC_IN.
#pragma once
#include <cstddef>

namespace sgl_kernel::stage::xe20 {

template <class Base>
struct Dims : Base {
  // launch: work-groups, work-items, sub-groups a work-group
  static constexpr int NWG = Base::NWG, WGS = Base::WGS, NSG = Base::NSG;
  // hidden, MoE (per rank), fp8 block
  static constexpr int HID = Base::H, NE = Base::NE, TOPK = Base::TOPK, INTER = Base::INTER;
  static constexpr int QB = Base::QB;
  static constexpr int KB_HID = HID / QB;  // K blocks of a hidden-size row
  static constexpr float EPS = Base::EPS;
  // GatedDeltaNet (per rank)
  static constexpr int NKH = Base::NKH, NVH = Base::NVH, DK = Base::DK, DV = Base::DV;
  static constexpr int KD = NKH * DK, VD = NVH * DV, CONVD = 2 * KD + VD, QKVZ = CONVD + VD;
  // full attention (per rank)
  static constexpr int NQ = Base::NQ, HD = Base::HD, QKV_ROWS = NQ * 2 * HD + 2 * HD;
  static constexpr int GROWS = NQ * HD + 2 * HD;  // gate + k + v rows
  // softmax scale 1 / sqrt(HD) in log2 units
  static constexpr float SM_SCALE = 0.0625f * 1.4426950408889634f, THETA = Base::THETA;
  static_assert(HD == 256, "SM_SCALE is 1 / sqrt(256)");
  static constexpr int GKV_MID = 2, GKV_ROWS = 2, GKV_CHUNKS = (GROWS - GKV_MID * NWG * NSG) / GKV_ROWS;
  static constexpr int SPW = INTER / NWG;      // shared inter rows per WG
  static constexpr int SHR = 2 * INTER / NSG;  // shared gate|up rows per SG in the partial
  // MTP fc (each rank computes half of the output rows)
  static constexpr int MTP_FC_IN = Base::MTP_FC_IN, MTP_FC_OUT = HID / 2;

  static constexpr size_t MBOX_WORDS = 2 * HID;  // per layer, rows-1 region

  static constexpr int NBMAX = Base::MAXR;
  static constexpr int UMAX = NBMAX * TOPK;     // experts in the union
  static constexpr int ACH = 80;                // cached tokens per SG pair and pass
  static constexpr int APAIRS = NWG * NSG / 2;  // 512
  static constexpr int APASS = NBMAX;  // partial slots reserved for up to NBMAX passes (an upper bound of the buffers)
  // mailbox words of one layer: a region per configuration (rows 1: 2 HID fp32
  // words; rows NB: 2 all-reduces x NB HID / 2 words of two fp16 each), rows 2 /
  // 4 / 8 / 3 / 6 at 2 / 4 / 8 / 16 / 19 HID
  static constexpr size_t MBOX_WORDS_B = 32 * (size_t)HID;
  static_assert(MBOX_WORDS_B >= MBOX_WORDS, "the batched mailbox covers the single-row one");

  // SLM layout GDN
  static constexpr int G_X = 0, G_XN = 4096, G_QK = 8192, G_RED = G_QK + 2048, G_ROUTE = G_RED + 512,
                       G_AO = G_ROUTE + 128, G_MR = G_AO + 8192, G_HB = G_MR + 16384,
                       G_BYTES = G_HB + (TOPK + 1) * INTER * 2;
  // ATTN
  static constexpr int A_X = 0, A_XN = 8192, A_Q = A_XN + 4096, A_KV = A_Q + 8192, A_RED = A_KV + 2048,
                       A_ROUTE = A_RED + 512, A_CS = A_ROUTE + 128, A_ST = A_CS + 256, A_QPK = A_ST + 2048,
                       A_AO = A_QPK + 4096, A_MR = A_AO + 8192, A_HB = A_MR + 16384, A_BYTES = A_AO + 65536;
  static_assert(A_HB + (TOPK + 1) * INTER * 2 <= A_BYTES, "tail SLM inside the attention partial area");

  // -------------------------------------------------------- SLM layout (batched)
  static constexpr int B_XA = 0,      // A tiles of the current activations: 128 chunks x 256 B
      B_RED = B_XA + 32768,           // SG partials: [32][8][16] f32
      B_QK = B_RED + 16384,           // GDN: per row q[128] k[128] v[64] beta g (384 f32 per row)
      B_SM = B_QK + 8 * 384 * 4,      // small reductions: 1024 f32
      B_RB = B_SM + 4096,             // tail: res2 block [8][64] f32
      B_XAN = B_RB + 2048,            // tail: router A tiles (nat, 4 chunks)
      B_XAP = B_XAN + 1024,           // tail: shared gate|up A tiles (perm, 4 chunks)
      B_LG = B_XAP + 1024,            // logits [8][256] f32
      B_S6R = B_RED,                  // S6 SG partials [32][2 tiles][8][16] f32 / S7 [32][4][8][16]
                                      // over RED .. LG (dead in S6 / S7)
      B_RT = B_RED + 65536,           // route_e [8][8] int, route_w [8][8] f32
      B_U = B_RT + 512,               // union: e [64] int, w [64][8] f32, count
      B_SGT = B_U + 256 + 2048 + 64,  // shared-expert gate per row [8] f32
      B_CS = B_SGT + 64,              // GDN: conv states of the WG's 320 channels [8][3][320] f16
      B_BYTES = B_CS + 8 * 3 * 320 * 2;
  static_assert(B_LG + 8192 <= B_RT, "S6 partials alias only dead tail SLM");

  // ---------------------------------------------- full-attention layer (batched)
  // SLM (beyond the shared batched layout): q per row in DPAS A order, RoPE
  // tables, new k|v, normalized q, merge weights
  static constexpr int ADYN_MIN = 96 * 1024;  // cached tokens per step from which the
  static constexpr int AB_QPK = 49152, AB_CS = AB_QPK + NBMAX * 4096, AB_KVN = AB_CS + NBMAX * 256,
                       AB_QS = AB_KVN + NBMAX * 1024, AB_W = AB_QS + NQ * HD * 2, AB_SM3 = AB_W + 2048,
                       AB_LST =
                           AB_SM3 + 1024,  // [8][512] i16 (no reader in the current kernels; kept so that the offsets
                                           // after it, and the SLM layout of the measured kernels, do not move)
      AB_WL = AB_LST + NBMAX * 512 * 2,    // [8][512] f32 (likewise)
      AB_TK = AB_WL + NBMAX * 512 * 4,     // ticket ring [16][2] + ack [16]
      AB_BYTES = AB_TK + 256;
  static_assert(AB_BYTES <= 131072, "SLM of the batched attention layer");
  static constexpr int AB_SLM = AB_BYTES > B_BYTES ? AB_BYTES : B_BYTES;  // the tail reuses the (dead) attention data
  static_assert(AB_SLM <= 131072, "SLM of the batched attention layer + tail");
};

}  // namespace sgl_kernel::stage::xe20
