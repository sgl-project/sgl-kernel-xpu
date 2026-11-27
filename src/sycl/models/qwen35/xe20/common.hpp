// SPDX-License-Identifier: Apache-2.0
//
// Mono decode (one resident kernel per decoder layer) for Qwen3.5 / Qwen3.6
// MoE (35B-A3B shapes) served by SGLang at TP2 on two Intel Arc Pro B70 (Xe2,
// 32 Xe cores): the model's side of the Xe2 decoder-layer stages
// (include/sgl_kernel/stage/xe20/: the GatedDeltaNet and full-attention
// mixers, the MoE tail, the mailbox), instantiated with the traits Q35 of
// shape.hpp.
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
#include "sgl_kernel/stage/xe20/attn.hpp"
#include "sgl_kernel/stage/xe20/gdn.hpp"
#include "shape.hpp"

namespace sgl_mono {

using namespace sgl_kernel::stage::xe20;  // NOLINT: the stages

}  // namespace sgl_mono
