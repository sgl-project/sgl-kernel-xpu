// SPDX-License-Identifier: Apache-2.0
//
// The per-rank shapes of Qwen3.5 / Qwen3.6 MoE (35B-A3B) at TP2 and the launch
// shape of its Xe2 decode kernels: the base shape Shape, the stage traits Q35
// derived from it (Dims<Shape>), and the flat names of the traits' members.
#pragma once
#include "sgl_kernel/stage/xe20/common.hpp"
#include "sgl_kernel/stage/xe20/dims.hpp"

namespace sgl_mono {

// 32 work-groups x 512 work-items, up to 8 rows a launch
struct Shape : sgl_kernel::stage::xe20::Launch<32, 512, 8> {
  // hidden, RMSNorm eps
  static constexpr int H = 2048;
  static constexpr float EPS = 1e-6f;
  // MoE: 256 experts top-8, intermediate 512 (routed and shared) / 2 per rank
  static constexpr int NE = 256, TOPK = 8, INTER = 256;
  // fp8 block size (N and K)
  static constexpr int QB = 128;
  // GatedDeltaNet (per rank): 8 k heads, 16 v heads, 128 dims
  static constexpr int NKH = 8, NVH = 16, DK = 128, DV = 128;
  // full attention (per rank): 8 q heads, 1 kv head, head dim 256, partial
  // RoPE (64 dims, theta 1e7)
  static constexpr int NQ = 8, HD = 256;
  static constexpr float THETA = 1.0e7f;
  // MTP fc input (embedding | hidden)
  static constexpr int MTP_FC_IN = 2 * H;
};

// the stages' traits of this model (include/sgl_kernel/stage/xe20/dims.hpp)
using Q35 = sgl_kernel::stage::xe20::Dims<Shape>;

// the flat names of the traits' members, for the host code and the MTP
// prologue (model code outside the stages)
constexpr auto AB_BYTES = Q35::AB_BYTES;
constexpr auto AB_CS = Q35::AB_CS;
constexpr auto AB_KVN = Q35::AB_KVN;
constexpr auto AB_LST = Q35::AB_LST;
constexpr auto AB_QPK = Q35::AB_QPK;
constexpr auto AB_QS = Q35::AB_QS;
constexpr auto AB_SLM = Q35::AB_SLM;
constexpr auto AB_SM3 = Q35::AB_SM3;
constexpr auto AB_TK = Q35::AB_TK;
constexpr auto AB_W = Q35::AB_W;
constexpr auto AB_WL = Q35::AB_WL;
constexpr auto ACH = Q35::ACH;
constexpr auto ADYN_MIN = Q35::ADYN_MIN;
constexpr auto APAIRS = Q35::APAIRS;
constexpr auto APASS = Q35::APASS;
constexpr auto A_AO = Q35::A_AO;
constexpr auto A_BYTES = Q35::A_BYTES;
constexpr auto A_CS = Q35::A_CS;
constexpr auto A_HB = Q35::A_HB;
constexpr auto A_KV = Q35::A_KV;
constexpr auto A_MR = Q35::A_MR;
constexpr auto A_Q = Q35::A_Q;
constexpr auto A_QPK = Q35::A_QPK;
constexpr auto A_RED = Q35::A_RED;
constexpr auto A_ROUTE = Q35::A_ROUTE;
constexpr auto A_ST = Q35::A_ST;
constexpr auto A_X = Q35::A_X;
constexpr auto A_XN = Q35::A_XN;
constexpr auto B_BYTES = Q35::B_BYTES;
constexpr auto B_CS = Q35::B_CS;
constexpr auto B_LG = Q35::B_LG;
constexpr auto B_QK = Q35::B_QK;
constexpr auto B_RB = Q35::B_RB;
constexpr auto B_RED = Q35::B_RED;
constexpr auto B_RT = Q35::B_RT;
constexpr auto B_S6R = Q35::B_S6R;
constexpr auto B_SGT = Q35::B_SGT;
constexpr auto B_SM = Q35::B_SM;
constexpr auto B_U = Q35::B_U;
constexpr auto B_XA = Q35::B_XA;
constexpr auto B_XAN = Q35::B_XAN;
constexpr auto B_XAP = Q35::B_XAP;
constexpr auto CONVD = Q35::CONVD;
constexpr auto DK = Q35::DK;
constexpr auto DV = Q35::DV;
constexpr auto EPS = Q35::EPS;
constexpr auto GKV_CHUNKS = Q35::GKV_CHUNKS;
constexpr auto GKV_MID = Q35::GKV_MID;
constexpr auto GKV_ROWS = Q35::GKV_ROWS;
constexpr auto GROWS = Q35::GROWS;
constexpr auto G_AO = Q35::G_AO;
constexpr auto G_BYTES = Q35::G_BYTES;
constexpr auto G_HB = Q35::G_HB;
constexpr auto G_MR = Q35::G_MR;
constexpr auto G_QK = Q35::G_QK;
constexpr auto G_RED = Q35::G_RED;
constexpr auto G_ROUTE = Q35::G_ROUTE;
constexpr auto G_X = Q35::G_X;
constexpr auto G_XN = Q35::G_XN;
constexpr auto HD = Q35::HD;
constexpr auto HID = Q35::HID;
constexpr auto INTER = Q35::INTER;
constexpr auto KB_HID = Q35::KB_HID;
constexpr auto KD = Q35::KD;
constexpr auto MBOX_WORDS = Q35::MBOX_WORDS;
constexpr auto MBOX_WORDS_B = Q35::MBOX_WORDS_B;
constexpr auto MTP_FC_IN = Q35::MTP_FC_IN;
constexpr auto MTP_FC_OUT = Q35::MTP_FC_OUT;
constexpr auto NBMAX = Q35::NBMAX;
constexpr auto NE = Q35::NE;
constexpr auto NKH = Q35::NKH;
constexpr auto NQ = Q35::NQ;
constexpr auto NSG = Q35::NSG;
constexpr auto NVH = Q35::NVH;
constexpr auto NWG = Q35::NWG;
constexpr auto QB = Q35::QB;
constexpr auto QKVZ = Q35::QKVZ;
constexpr auto QKV_ROWS = Q35::QKV_ROWS;
constexpr auto SHR = Q35::SHR;
constexpr auto SM_SCALE = Q35::SM_SCALE;
constexpr auto SPW = Q35::SPW;
constexpr auto THETA = Q35::THETA;
constexpr auto TOPK = Q35::TOPK;
constexpr auto UMAX = Q35::UMAX;
constexpr auto VD = Q35::VD;
constexpr auto WGS = Q35::WGS;

}  // namespace sgl_mono
