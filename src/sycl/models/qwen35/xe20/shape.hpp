// SPDX-License-Identifier: Apache-2.0
//
// The per-rank shapes of Qwen3.5 / Qwen3.6 MoE (35B-A3B) at TP2 and the launch
// shape of its Xe2 decode kernels (32 resident work-groups x 512 work-items).
#pragma once

namespace sgl_mono {

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

}  // namespace sgl_mono
