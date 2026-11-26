/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Host side of the Qwen3.5 / Qwen3.6 MoE decode kernels (TP2, Xe2): one kernel
// a decoder layer, launched on the current torch XPU queue. The launchers are
// the qwen35_*_launch_* sources next to this file, one translation unit (and
// library) per kernel kind and dtype.
//
// Nothing is kept across calls. The caller owns:
//  - the layer table: qwen35_layer_desc() builds a layer's descriptor (the
//    weight, cache and mailbox addresses) into a CPU uint8 tensor; the caller
//    stacks them into a CPU tensor [layers, desc_bytes] (each launch reads its
//    layer's row on the host and passes it to the kernel) and keeps every
//    tensor a descriptor points to alive (qwen35_prepare_layer() gives the
//    tensor list, with the copies the kernels need);
//  - workspace (uint8, zeroed once), shared by all launches of a device, one
//    launch at a time;
//  - counters (int32 [kernels, CNT_WORDS], qwen35_init_counters): row l is
//    kernel l's stage epoch and mailbox tags, kept from one step to the next,
//    and the control block of its waits; health words (int32 device + pinned
//    host);
//  - the P2P mailboxes (int64 [mailbox_words] a kernel, filled with
//    qwen35_mailbox_never(); the peer's views e.g. from torch symmetric
//    memory).
#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <array>
#include <atomic>
#include <cstring>
#include <optional>
#include <sycl/sycl.hpp>
#include <vector>

#include "../../../Utils.h"
#include "b60/launch.hpp"
#include "b60/mtp.hpp"
#include "launch.hpp"
#include "mtp.hpp"
#include "sgl_kernel_ops.h"  // the ops, exported

// the kernels are compiled in their own translation units (one per layer kind
// and dtype)
namespace sgl_mono {
#define MONO_EXTERN_LAUNCHERS(DT)                                                                                    \
  extern template void launch_gdn<DT>(sycl::queue&, int, const Ptrs&, const Scratch&, const ScratchB&);              \
  extern template void launch_attn<DT>(sycl::queue&, int, const Ptrs&, const Scratch&, const ScratchB&);             \
  extern template bool launch_gdn_verify<DT>(sycl::queue&, int, int, const Ptrs&, const Scratch&, const ScratchB&);  \
  extern template bool launch_attn_verify<DT>(sycl::queue&, int, int, const Ptrs&, const Scratch&, const ScratchB&); \
  extern template bool launch_mtp<DT>(sycl::queue&, int, const Ptrs&, const Scratch&, int);                          \
  extern template bool launch_gdn_split<DT>(                                                                         \
      sycl::queue&, int, const Ptrs&, const Scratch&, const ScratchB&, const Split&);                                \
  extern template bool launch_attn_split<DT>(                                                                        \
      sycl::queue&, int, const Ptrs&, const Scratch&, const ScratchB&, const Split&);                                \
  extern template bool launch_gdn_verify_split<DT>(                                                                  \
      sycl::queue&, int, int, const Ptrs&, const Scratch&, const ScratchB&, const Split&);                           \
  extern template bool launch_attn_verify_split<DT>(                                                                 \
      sycl::queue&, int, int, const Ptrs&, const Scratch&, const ScratchB&, const Split&);                           \
  extern template bool b60_launch_gdn<DT>(                                                                           \
      sycl::queue&, int, int, const Ptrs&, const Scratch&, const ScratchB&, const Split&, const Segs&);              \
  extern template bool b60_launch_attn<DT>(                                                                          \
      sycl::queue&, int, int, const Ptrs&, const Scratch&, const ScratchB&, const Split&, const Segs&);              \
  extern template bool b60_launch_gdn_verify<DT>(                                                                    \
      sycl::queue&, int, int, int, const Ptrs&, const Scratch&, const ScratchB&, const Split&, const Segs&);         \
  extern template bool b60_launch_attn_verify<DT>(                                                                   \
      sycl::queue&, int, int, int, const Ptrs&, const Scratch&, const ScratchB&, const Split&, const Segs&);         \
  extern template bool b60_launch_mtp<DT>(sycl::queue&, int, int, const Ptrs&, const Scratch&, int);
MONO_EXTERN_LAUNCHERS(half)
MONO_EXTERN_LAUNCHERS(bf16)
#undef MONO_EXTERN_LAUNCHERS
}  // namespace sgl_mono

namespace {

using sgl_mono::bf16;
using sgl_mono::e16;
using sgl_mono::Ptrs;
using sgl_mono::Scratch;
using sgl_mono::ScratchB;
using sgl_mono::Split;

// -----------------------------------------------------------------------------
// device check: Xe2 (Battlemage) with at least 20 Xe cores. A launch keeps one
// work-group a Xe core resident, and the work-groups wait on each other: with
// 32 or more Xe cores (Arc Pro B70) the layer kernels run their NWG = 32
// work-groups; with 20 to 31 (Arc Pro B60: 20) the b60/ kernels run the 32
// work-groups' parts on as many work-groups as there are Xe cores.
constexpr int kMinWorkGroups = 20;

int xe_cores_of(at::DeviceIndex d) {
  if (!is_bmg(d)) return 0;
  const auto& dev = c10::xpu::get_raw_device(d);
  return dev.get_info<sycl::ext::intel::info::device::gpu_slices>() *
         dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
}

int xe_cores(int64_t device) {                      // cached; 0: not an Xe2 GPU
  static std::array<std::atomic<int>, 64> known{};  // 0 unknown, else cores + 1
  const at::DeviceIndex d = device < 0 ? c10::xpu::current_device() : (at::DeviceIndex)device;
  if (d < 0 || d >= (at::DeviceIndex)known.size()) return xe_cores_of(d);
  int v = known[d].load(std::memory_order_relaxed);
  if (v == 0) {
    v = xe_cores_of(d) + 1;
    known[d].store(v, std::memory_order_relaxed);
  }
  return v - 1;
}

bool xe2_device(int64_t device) {
  return xe_cores(device) >= kMinWorkGroups;
}

// test hook (qwen35_force_work_groups): launch on at most this many
// work-groups (kMinWorkGroups .. NWG), e.g. 20 on a B70 to run the b60/
// kernels; 0: the device's
std::atomic<int> g_force_work_groups{0};

// work-groups of a launch on device d: NWG (the layer kernels) or fewer (b60/),
// never more than the device's Xe cores (one resident work-group a core: a
// launch of more would wait on work-groups that cannot run)
int work_groups(int64_t device) {
  const int n = std::min(xe_cores(device), (int)sgl_mono::NWG);
  const int f = g_force_work_groups.load(std::memory_order_relaxed);
  return f > 0 ? std::min(f, n) : n;
}

void check_device(const at::Tensor& t, const char* op) {
  TORCH_CHECK(t.is_xpu(), "qwen35: ", op, ": tensors must be on an XPU device, got ", t.device());
  TORCH_CHECK(
      xe2_device(t.get_device()),
      "qwen35: ",
      op,
      ": unsupported device (the decode kernels need an Xe2 (Battlemage) GPU with at least 20 Xe cores; "
      "check qwen35_supported() first).");
}

// wait deadline of the in-kernel mailbox waits, in cycles of the sub-group
// clock (about 2.8 GHz on B70)
constexpr double kClockHz = 2.8e9;

// -----------------------------------------------------------------------------
// descriptors
enum : int32_t { K_GDN = 0, K_ATTN = 1, K_MTP = 2 };

// A kernel's descriptor (qwen35_layer_desc / qwen35_mtp_desc), the bytes of a
// CPU uint8 tensor
struct LayerDesc {
  int32_t magic;
  int32_t kind;          // K_GDN, K_ATTN or K_MTP
  int32_t bf;            // the layer's dtype: 1 bf16, 0 fp16
  int32_t rank;          // MTP: this rank's half of the fc output rows
  int32_t caches_bound;  // decoder layers: qwen35_bind_*_caches ran
  int32_t inter_tokens;  // GDN: tokens a request of the bound verify scratch (0: none)
  int64_t inter_rows;    // and its rows (requests)
  uint64_t* mbox;        // this kernel's mailbox (words written by the peer)
  uint64_t* peer;        // the peer's mailbox of this kernel (P2P mapping)
  Ptrs P;                // the weights and caches (per-step fields unset)
};
constexpr int32_t kDescMagic = 0x51333544;  // "Q35D"

at::Tensor to_bytes(const LayerDesc& v) {
  at::Tensor t = at::zeros({(int64_t)sizeof(LayerDesc)}, at::TensorOptions().dtype(at::kByte));
  std::memcpy(t.data_ptr(), &v, sizeof(LayerDesc));
  return t;
}
LayerDesc& from_bytes(const at::Tensor& t, const char* name) {
  TORCH_CHECK(
      t.is_cpu() && t.scalar_type() == at::kByte && t.is_contiguous() && t.numel() == (int64_t)sizeof(LayerDesc),
      "qwen35: ",
      name,
      " must be a CPU uint8 [",
      sizeof(LayerDesc),
      "] descriptor (qwen35_layer_desc)");
  LayerDesc& D = *reinterpret_cast<LayerDesc*>(t.data_ptr());
  TORCH_CHECK(D.magic == kDescMagic, "qwen35: ", name, " is not a qwen35 descriptor");
  return D;
}

// -----------------------------------------------------------------------------
// tensor checks
const char* dtype_name(at::ScalarType t) {
  return c10::toString(t);
}

// fp8 [N][K] linear weight: row-major [N, K], or its [K, N] transpose view
const uint8_t* w_nk(const at::Tensor& w, int64_t n, int64_t k, const char* name) {
  TORCH_CHECK(w.scalar_type() == at::kFloat8_e4m3fn, "qwen35: ", name, " must be float8_e4m3fn, got ", w.scalar_type());
  const bool nk = w.dim() == 2 && w.size(0) == n && w.size(1) == k && w.stride(0) == k && w.stride(1) == 1;
  const bool kn = w.dim() == 2 && w.size(0) == k && w.size(1) == n && w.stride(0) == 1 && w.stride(1) == k;
  TORCH_CHECK(
      nk || kn,
      "qwen35: ",
      name,
      " must be a row-major [",
      n,
      ", ",
      k,
      "] fp8 tensor (or its transpose view), got ",
      w.sizes(),
      " strides ",
      w.strides());
  return static_cast<const uint8_t*>(w.data_ptr());
}

// 128x128 block scales as the kernels read them: row-major fp32 [nb, kb]
// (qwen35_prepare_layer makes that copy of SGLang's scales)
const float* scale_rows(const at::Tensor& s, int64_t nb, int64_t kb, const char* name) {
  TORCH_CHECK(
      s.scalar_type() == at::kFloat && s.is_contiguous() && s.dim() == 2 && s.size(0) == nb && s.size(1) == kb,
      "qwen35: ",
      name,
      " must be contiguous fp32 [",
      nb,
      ", ",
      kb,
      "] 128x128 block scales (qwen35_prepare_layer), got ",
      s.scalar_type(),
      " ",
      s.sizes());
  return s.data_ptr<float>();
}
// expert block scales, contiguous fp32 [E][nb][kb]
const float* scale_exp_rows(const at::Tensor& s, int64_t e, int64_t nb, int64_t kb, const char* name) {
  TORCH_CHECK(
      s.scalar_type() == at::kFloat && s.is_contiguous() && s.dim() == 3 && s.size(0) == e && s.size(1) == nb &&
          s.size(2) == kb,
      "qwen35: ",
      name,
      " must be contiguous fp32 [",
      e,
      ", ",
      nb,
      ", ",
      kb,
      "] 128x128 block scales (qwen35_prepare_layer), got ",
      s.scalar_type(),
      " ",
      s.sizes());
  return s.data_ptr<float>();
}

// 2-D block scales -> row-major fp32 [nb, kb]. SGLang keeps them logical [nb,
// kb] (stored column-major) or as the transpose: the tensor itself when it
// already is, else a copy.
at::Tensor scale_blk(const at::Tensor& s, int64_t nb, int64_t kb, const char* name) {
  TORCH_CHECK(s.dim() == 2, "qwen35: ", name, " must be 2-D block scales, got ", s.sizes());
  at::Tensor t = (s.size(0) == nb && s.size(1) == kb) ? s : (s.size(0) == kb && s.size(1) == nb) ? s.t() : s;
  TORCH_CHECK(
      t.size(0) == nb && t.size(1) == kb,
      "qwen35: ",
      name,
      " must be [",
      nb,
      ", ",
      kb,
      "] 128x128 block scales, got ",
      s.sizes());
  return (t.scalar_type() == at::kFloat && t.is_contiguous()) ? t : t.to(at::kFloat).contiguous();
}
// expert block scales [E][nb][kb] -> contiguous fp32
at::Tensor scale_exp(const at::Tensor& s, int64_t e, int64_t nb, int64_t kb, const char* name) {
  TORCH_CHECK(
      s.dim() == 3 && s.size(0) == e && s.size(1) == nb && s.size(2) == kb,
      "qwen35: ",
      name,
      " must be [",
      e,
      ", ",
      nb,
      ", ",
      kb,
      "] 128x128 block scales, got ",
      s.sizes());
  return (s.scalar_type() == at::kFloat && s.is_contiguous()) ? s : s.to(at::kFloat).contiguous();
}

// the layer's dtype, from its first 16-bit tensor
at::ScalarType dtype16(const at::Tensor& t, const char* name) {
  TORCH_CHECK(
      t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16,
      "qwen35: ",
      name,
      " must be fp16 or bf16, got ",
      t.scalar_type());
  return t.scalar_type();
}
// a contiguous 16-bit tensor in the layer's dtype: activations, norm / router /
// conv / ba weights all share it (no mixed fp16 / bf16 layers)
e16* t16(const at::Tensor& t, int64_t numel, const char* name, at::ScalarType dt) {
  TORCH_CHECK(
      t.scalar_type() == dt && t.numel() == numel && t.is_contiguous(),
      "qwen35: ",
      name,
      " must be contiguous ",
      dtype_name(dt),
      " (the layer's dtype) with ",
      numel,
      " elements, got ",
      t.scalar_type(),
      " ",
      t.sizes());
  return reinterpret_cast<e16*>(t.data_ptr());
}

// Every tensor of an op is an XPU tensor of the layer's device: a CPU or
// other-device tensor would be handed to the kernels as a device pointer.
void check_on(const at::Tensor& t, int device, const char* op, const char* name) {
  TORCH_CHECK(
      t.is_xpu() && t.get_device() == device,
      "qwen35: ",
      op,
      ": ",
      name,
      " must be an XPU tensor on device ",
      device,
      ", got ",
      t.device());
}
void check_on(const std::optional<at::Tensor>& t, int device, const char* op, const char* name) {
  if (t.has_value()) check_on(*t, device, op, name);
}
// a metadata vector the kernels read as a contiguous array [>= n]
void check_vec(const at::Tensor& t, at::ScalarType dt, int64_t n, const char* name) {
  TORCH_CHECK(
      t.scalar_type() == dt && t.dim() == 1 && t.size(0) >= n && t.stride(0) == 1,
      "qwen35: ",
      name,
      " must be a unit-stride ",
      dt,
      " vector with at least ",
      n,
      " elements, got ",
      t.scalar_type(),
      " ",
      t.sizes(),
      " ",
      t.strides());
}
// positions: [T], or mrope [3, T] of which the kernels read row 0
void check_positions(const at::Tensor& t, int64_t n) {
  TORCH_CHECK(
      t.scalar_type() == at::kLong && (t.dim() == 1 || t.dim() == 2) && t.size(-1) >= n && t.stride(-1) == 1,
      "qwen35: positions must be an int64 [>= ",
      n,
      "] (or mrope [3, >= ",
      n,
      "]) tensor with unit stride on the token axis, got ",
      t.scalar_type(),
      " ",
      t.sizes(),
      " ",
      t.strides());
}
uint64_t* mailbox_ptr(const at::Tensor& t, int device, const char* name) {
  TORCH_CHECK(
      t.is_xpu() && t.scalar_type() == at::kLong && t.is_contiguous() && t.numel() == (int64_t)sgl_mono::MBOX_WORDS_B,
      "qwen35: ",
      name,
      " must be a contiguous int64 [",
      sgl_mono::MBOX_WORDS_B,
      "] XPU tensor (qwen35_sizes()[3] words), got ",
      t.scalar_type(),
      " ",
      t.sizes(),
      " on ",
      t.device());
  // the peer's mailbox is a P2P mapping of the peer card's memory (device < 0:
  // any device); this kernel's own is on its device
  TORCH_CHECK(
      device < 0 || t.get_device() == device, "qwen35: ", name, " must be on device ", device, ", got ", t.device());
  return reinterpret_cast<uint64_t*>(t.data_ptr());
}

// -----------------------------------------------------------------------------
// workspace: the layers' intermediates, one block a device shared by all
// launches (one at a time on the queue; each kernel writes an intermediate
// before it reads it). Pieces, 256-B aligned: scratch | batched scratch |
// batched attention chunk partials | two ping-pong (hidden, residual) pairs of
// a multi-layer launch.
constexpr size_t al256(size_t b) {
  return (b + 255) & ~size_t(255);
}
constexpr size_t kPingBytes = (size_t)sgl_mono::NBMAX * sgl_mono::HID * 2;  // [8][2048] 16-bit
size_t workspace_bytes() {
  return al256(sgl_mono::scratch_bytes<sgl_mono::Q35>()) + al256(sgl_mono::scratch_b_bytes<sgl_mono::Q35>()) +
         al256(sgl_mono::attn_part_bytes<sgl_mono::Q35>()) + 4 * kPingBytes;
}
struct Carved {
  Scratch S;
  ScratchB SB;
  e16* ping[4];  // hidden 0, residual 0, hidden 1, residual 1
};
// the workspace pieces; counters: the kernel's row of the counters
Carved carve(void* ws, uint32_t* counters, uint64_t* mbox, uint64_t* peer) {
  using namespace sgl_mono;
  Carved c{};
  uint8_t* p = static_cast<uint8_t*>(ws);
  c.S = scratch_carve<Q35>(p, mbox, peer);
  c.S.cnt = counters;  // the kernel's own counters, not the scratch's
  p += al256(scratch_bytes<Q35>());
  c.SB = scratch_b_carve<Q35>(p);
  p += al256(scratch_b_bytes<Q35>());
  c.SB.pa_o = reinterpret_cast<sycl::half*>(p);
  c.SB.pa_ml = reinterpret_cast<float*>(p + (size_t)APASS * APAIRS * NQ * HD * 2);
  p += al256(attn_part_bytes<Q35>());
  for (int i = 0; i < 4; ++i)
    c.ping[i] = reinterpret_cast<e16*>(p + i * kPingBytes);
  return c;
}

void check_workspace(const at::Tensor& ws, int device, const char* op) {
  check_on(ws, device, op, "workspace");
  TORCH_CHECK(
      ws.scalar_type() == at::kByte && ws.is_contiguous() && ws.numel() == (int64_t)workspace_bytes(),
      "qwen35: ",
      op,
      ": workspace must be a contiguous uint8 [",
      workspace_bytes(),
      "] tensor (qwen35_sizes()[0]), got ",
      ws.scalar_type(),
      " ",
      ws.sizes());
}
// counters: int32 [kernels, CNT_WORDS] (rows = true) or one kernel's row
void check_counters(const at::Tensor& c, int device, const char* op, bool rows) {
  check_on(c, device, op, "counters");
  TORCH_CHECK(
      c.scalar_type() == at::kInt && c.is_contiguous() && c.size(-1) == sgl_mono::CNT_WORDS &&
          c.dim() == (rows ? 2 : 1),
      "qwen35: ",
      op,
      ": counters must be a contiguous int32 ",
      rows ? "[kernels, " : "[",
      sgl_mono::CNT_WORDS,
      "] tensor (qwen35_init_counters), got ",
      c.scalar_type(),
      " ",
      c.sizes());
}

int64_t rows_of(const at::Tensor& t) {
  return t.dim() == 2 ? t.size(0) : 1;
}

// -----------------------------------------------------------------------------
// split launches (modes 1 / 2): the plan of a step, one item a segment kernel
// (stage args.hpp, SplitSeg), an all-reduce of the caller (mode 2) or the
// per-call counter advance after a layer's last segment
constexpr int ITEM_AR = -2, ITEM_ADVANCE = -3;
struct PlanItem {
  int l;    // layer
  int seg;  // SplitSeg, ITEM_AR or ITEM_ADVANCE
};
std::vector<PlanItem> split_plan(const LayerDesc* table, int l0, int l1, bool ext_ar) {
  using namespace sgl_mono;
  std::vector<PlanItem> v;
  for (int l = l0; l < l1; ++l) {
    v.push_back({l, SEG_F1});
    v.push_back({l, SEG_F2});
    if (table[l].kind == K_ATTN) v.push_back({l, SEG_F3});
    if (ext_ar) {
      for (int k : {(int)SEG_T1A, ITEM_AR, (int)SEG_T1B, (int)SEG_S56, (int)SEG_S7A, ITEM_AR, (int)SEG_S7B})
        v.push_back({l, k});
    } else {
      for (int k : {SEG_T1, SEG_S56, SEG_S7})
        v.push_back({l, k});
    }
    v.push_back({l, ITEM_ADVANCE});
  }
  return v;
}
// the mailbox tag counter of a row count (stage rows.hpp mb_tag, and MBT for
// one row)
int tag_word(int nb) {
  using namespace sgl_mono;
  switch (nb) {
    case 1:
      return MBT;
    case 2:
      return MBT + 1;
    case 4:
      return MBT + 2;
    case 8:
      return MBT + 3;
    case 3:
      return MBT2;
    default:
      return MBT2 + 1;  // 6
  }
}

// the per-call counters of one layer kernel (its counters row lc), as the
// one-kernel layer advances them after its last stage: the epoch and the
// mailbox tag of the row count nb
void advance_counters(sycl::queue& q, uint32_t* lc, int nb) {
  using namespace sgl_mono;
  const int tw = tag_word(nb);
  q.single_task<class Qwen35SplitAdvanceK>([=] {
    lc[EPOCH] += 1u;
    lc[tw] += 1u;
  });
}

}  // namespace

// -----------------------------------------------------------------------------
// ops
bool qwen35_supported(int64_t device) {
  return xe2_device(device);
}

int64_t qwen35_max_context() {
  return 512 * 1024;
}

// [workspace bytes, counter words a kernel, health words, mailbox words a
// kernel, descriptor bytes]
std::vector<int64_t> qwen35_sizes() {
  return {
      (int64_t)workspace_bytes(),
      (int64_t)sgl_mono::CNT_WORDS,
      (int64_t)sgl_mono::H_WORDS,
      (int64_t)sgl_mono::MBOX_WORDS_B,
      (int64_t)sizeof(LayerDesc)};
}

// the initial value of every mailbox word (it matches no tag)
int64_t qwen35_mailbox_never() {
  return (int64_t)sgl_mono::MB_NEVER;
}

// The tensor list of qwen35_layer_desc from a layer's SGLang tensors (used in
// place where the kernels read them as they are, else copies the caller keeps
// with the descriptor): the 128x128 block scales as row-major fp32, and for a
// GatedDeltaNet layer in_proj_ba in fp16 appended (the DPAS paths read it in
// fp16: the tensor itself in an fp16 layer, a copy rounded once in a bf16 one).
// Input order, both kinds: 0 input_layernorm [2048], 1 post_attention_layernorm
// [2048], 2 router [256, 2048], 3 shared_expert_gate [1, 2048], 4 o_proj /
// out_proj fp8 [2048, K], 5 its scales, 6 w13 fp8 [256, 512, 2048], 7 its
// scales [256, 4, 16], 8 w2 fp8 [256, 2048, 256], 9 its scales [256, 16, 2], 10
// shared gate_up fp8 [512, 2048], 11 its scales, 12 shared down fp8 [2048,
// 256], 13 its scales; attention: 14 qkv_proj fp8 [4608, 2048], 15 its scales,
// 16 q_norm [256], 17 k_norm [256]; GatedDeltaNet: 14 in_proj_qkvz fp8 [6144,
// 2048], 15 its scales, 16 in_proj_ba [32, 2048], 17 conv1d [4096, (1,) 4], 18
// A_log fp32 [16], 19 dt_bias [16], 20 norm [128].
std::vector<at::Tensor> qwen35_prepare_layer(bool attn, const std::vector<at::Tensor>& t) {
  using namespace sgl_mono;
  TORCH_CHECK(t.size() == (attn ? 18u : 21u), "qwen35_prepare_layer: ", attn ? 18 : 21, " tensors expected");
  std::vector<at::Tensor> o(t.begin(), t.end());
  const int64_t ok = attn ? NQ * HD : VD;
  o[5] = scale_blk(t[5], HID / QB, ok / QB, "out scale");
  o[7] = scale_exp(t[7], NE, 2 * INTER / QB, HID / QB, "w13 scale");
  o[9] = scale_exp(t[9], NE, HID / QB, INTER / QB, "w2 scale");
  o[11] = scale_blk(t[11], 2 * INTER / QB, HID / QB, "shared gate_up scale");
  o[13] = scale_blk(t[13], HID / QB, INTER / QB, "shared down scale");
  if (attn) {
    o[15] = scale_blk(t[15], QKV_ROWS / QB, HID / QB, "qkv scale");
  } else {
    o[15] = scale_blk(t[15], QKVZ / QB, HID / QB, "qkvz scale");
    o.push_back(t[16].scalar_type() == at::kHalf ? t[16] : t[16].to(at::kHalf).contiguous());
  }
  return o;
}

// A decoder layer's descriptor (CPU uint8): the tensors of
// qwen35_prepare_layer (checked; the caller keeps them alive), this kernel's
// mailbox and the peer's view of it (int64 [mailbox_words]). Caches unbound.
at::Tensor
qwen35_layer_desc(bool attn, const std::vector<at::Tensor>& t, const at::Tensor& mailbox, const at::Tensor& peer) {
  using namespace sgl_mono;
  TORCH_CHECK(
      t.size() == (attn ? 18u : 22u),
      "qwen35_layer_desc: ",
      attn ? 18 : 22,
      " tensors expected (qwen35_prepare_layer), got ",
      t.size());
  check_device(t[0], "qwen35_layer_desc");
  const int dev = t[0].get_device();
  for (const at::Tensor& w : t)
    check_on(w, dev, "qwen35_layer_desc", "a weight");
  LayerDesc D{};
  D.magic = kDescMagic;
  D.kind = attn ? K_ATTN : K_GDN;
  const at::ScalarType dt = dtype16(t[0], "input_norm_w");
  D.bf = dt == at::kBFloat16;
  Ptrs& P = D.P;
  P.in_norm_w = t16(t[0], HID, "input_norm_w", dt);
  P.post_norm_w = t16(t[1], HID, "post_norm_w", dt);
  P.router_w = t16(t[2], (int64_t)NE * HID, "router_w", dt);
  P.sgate_w = t16(t[3], HID, "shared_gate_w", dt);
  const int64_t ok = attn ? NQ * HD : VD;
  P.out_w = w_nk(t[4], HID, ok, attn ? "o_proj" : "out_proj");
  P.out_s = scale_rows(t[5], HID / QB, ok / QB, "out scale");
  TORCH_CHECK(
      t[6].scalar_type() == at::kFloat8_e4m3fn && t[6].is_contiguous() && t[6].dim() == 3 && t[6].size(0) == NE &&
          t[6].size(1) == 2 * INTER && t[6].size(2) == HID,
      "qwen35: w13 must be contiguous [256, 512, 2048] e4m3 (SGLang FusedMoE layout per rank), got ",
      t[6].sizes());
  TORCH_CHECK(
      t[8].scalar_type() == at::kFloat8_e4m3fn && t[8].is_contiguous() && t[8].dim() == 3 && t[8].size(0) == NE &&
          t[8].size(1) == HID && t[8].size(2) == INTER,
      "qwen35: w2 must be contiguous [256, 2048, 256] e4m3, got ",
      t[8].sizes());
  P.w13 = static_cast<const uint8_t*>(t[6].data_ptr());
  P.w13_s = scale_exp_rows(t[7], NE, 2 * INTER / QB, HID / QB, "w13 scale");
  P.w2 = static_cast<const uint8_t*>(t[8].data_ptr());
  P.w2_s = scale_exp_rows(t[9], NE, HID / QB, INTER / QB, "w2 scale");
  P.sw13 = w_nk(t[10], 2 * INTER, HID, "shared gate_up_proj");
  P.sw13_s = scale_rows(t[11], 2 * INTER / QB, HID / QB, "shared gate_up scale");
  P.sw2 = w_nk(t[12], HID, INTER, "shared down_proj");
  P.sw2_s = scale_rows(t[13], HID / QB, INTER / QB, "shared down scale");
  if (attn) {
    P.qkv_w = w_nk(t[14], QKV_ROWS, HID, "qkv_proj");
    P.qkv_s = scale_rows(t[15], QKV_ROWS / QB, HID / QB, "qkv scale");
    P.q_norm_w = t16(t[16], HD, "q_norm_w", dt);
    P.k_norm_w = t16(t[17], HD, "k_norm_w", dt);
  } else {
    P.qkvz_w = w_nk(t[14], QKVZ, HID, "in_proj_qkvz");
    P.qkvz_s = scale_rows(t[15], QKVZ / QB, HID / QB, "qkvz scale");
    P.ba_w = t16(t[16], (int64_t)2 * NVH * HID, "in_proj_ba", dt);
    P.ba_h = t16(t[21], (int64_t)2 * NVH * HID, "in_proj_ba (fp16 copy)", at::kHalf);
    P.conv_w = t16(t[17], (int64_t)CONVD * 4, "conv1d", dt);
    TORCH_CHECK(
        t[18].scalar_type() == at::kFloat && t[18].numel() == NVH && t[18].is_contiguous(),
        "qwen35: A_log must be contiguous fp32 [16]");
    P.A_log = t[18].data_ptr<float>();
    P.dt_bias = t16(t[19], NVH, "dt_bias", dt);
    P.gnorm_w = t16(t[20], DV, "norm_w", dt);
  }
  D.mbox = mailbox_ptr(mailbox, dev, "mailbox");
  D.peer = mailbox_ptr(peer, -1, "peer mailbox");
  return to_bytes(D);
}

// The MTP drafter prologue's descriptor (CPU uint8): [fc [2048, 4096],
// pre_fc_norm_embedding [2048], pre_fc_norm_hidden [2048]], all in the model's
// 16-bit dtype (the caller keeps them alive); `rank` (0 / 1) picks this rank's
// half of the fc output rows.
at::Tensor
qwen35_mtp_desc(const std::vector<at::Tensor>& t, const at::Tensor& mailbox, const at::Tensor& peer, int64_t rank) {
  using namespace sgl_mono;
  TORCH_CHECK(t.size() == 3u, "qwen35_mtp_desc: takes [fc, pre_fc_norm_embedding, pre_fc_norm_hidden]");
  check_device(t[0], "qwen35_mtp_desc");
  const int dev = t[0].get_device();
  for (const at::Tensor& w : t)
    check_on(w, dev, "qwen35_mtp_desc", "a weight");
  TORCH_CHECK(rank == 0 || rank == 1, "qwen35: rank must be 0 or 1, got ", rank);
  TORCH_CHECK(
      t[0].dim() == 2 && t[0].size(0) == HID && t[0].size(1) == MTP_FC_IN && t[0].is_contiguous(),
      "qwen35: MTP fc must be contiguous [2048, 4096] (the full weight on each rank), got ",
      t[0].sizes());
  LayerDesc D{};
  D.magic = kDescMagic;
  D.kind = K_MTP;
  D.rank = (int32_t)rank;
  const at::ScalarType dt = dtype16(t[0], "mtp fc");
  D.bf = dt == at::kBFloat16;
  D.P.fc_w = t16(t[0], (int64_t)HID * MTP_FC_IN, "mtp fc", dt);
  D.P.in_norm_w = t16(t[1], HID, "pre_fc_norm_embedding", dt);
  D.P.post_norm_w = t16(t[2], HID, "pre_fc_norm_hidden", dt);
  D.mbox = mailbox_ptr(mailbox, dev, "mailbox");
  D.peer = mailbox_ptr(peer, -1, "peer mailbox");
  return to_bytes(D);
}

// Binds a GatedDeltaNet layer's caches into its descriptor (SGLang pools, used
// in place; checked): conv_state [slots, 4096, >= 3], ssm_state [slots, 16,
// 128, 128] fp32, and for speculative decoding the verify scratch
// (intermediate_ssm [rows, T, 16, 128, 128] fp32, intermediate conv windows
// [rows, T, 4096, 3]). Only updates the descriptor: run again whenever the
// pools move, on the row of the layer table the launches read (a stacked
// table holds copies of the descriptors).
void qwen35_bind_gdn_caches(
    at::Tensor& desc,
    const at::Tensor& conv,
    const at::Tensor& ssm,
    const std::optional<at::Tensor>& inter_ssm,
    const std::optional<at::Tensor>& inter_conv) {
  using namespace sgl_mono;
  LayerDesc& D = from_bytes(desc, "desc");
  TORCH_CHECK(D.kind == K_GDN, "qwen35_bind_gdn_caches: not a GatedDeltaNet layer");
  check_device(conv, "qwen35_bind_gdn_caches");
  const int dev = conv.get_device();
  check_on(ssm, dev, "qwen35_bind_gdn_caches", "ssm state");
  check_on(inter_ssm, dev, "qwen35_bind_gdn_caches", "intermediate ssm state");
  check_on(inter_conv, dev, "qwen35_bind_gdn_caches", "intermediate conv windows");
  TORCH_CHECK(
      inter_ssm.has_value() == inter_conv.has_value(), "qwen35: the verify scratch takes both intermediate buffers");
  const at::ScalarType dt = D.bf ? at::kBFloat16 : at::kHalf;
  const at::Tensor& cs = conv;
  const at::Tensor& ss = ssm;
  TORCH_CHECK(
      cs.scalar_type() == dt && cs.dim() == 3 && cs.size(1) == CONVD && cs.size(2) == 3 && cs.stride(2) == 1 &&
          cs.stride(1) == cs.size(2),
      "qwen35: conv state must be [slots, 4096, 3] in the layer's dtype (channel major, taps innermost; the kernels "
      "keep the 3 history taps in entries 0..2), got ",
      cs.scalar_type(),
      " ",
      cs.sizes(),
      " ",
      cs.strides());
  TORCH_CHECK(
      (reinterpret_cast<uintptr_t>(cs.data_ptr()) & 15) == 0 && cs.stride(0) % 8 == 0,
      "qwen35: conv state slots must be 16-B aligned");
  TORCH_CHECK(
      ss.scalar_type() == at::kFloat && ss.dim() == 4 && ss.size(1) == NVH && ss.size(2) == DV && ss.size(3) == DK &&
          ss.stride(3) == 1 && ss.stride(2) == DK && ss.stride(1) == DV * DK,
      "qwen35: ssm state must be [slots, 16, 128, 128] fp32, got ",
      ss.sizes(),
      " ",
      ss.strides());
  Ptrs& P = D.P;
  P.conv_cache = reinterpret_cast<e16*>(cs.data_ptr());
  P.conv_stride = cs.stride(0);
  P.conv_ch = (int)cs.size(2);
  P.ssm_cache = ss.data_ptr<float>();
  P.ssm_stride = ss.stride(0);
  D.inter_tokens = 0;
  D.inter_rows = 0;
  P.inter_ssm = nullptr;
  P.inter_conv = nullptr;
  if (inter_ssm.has_value()) {
    const at::Tensor& is = *inter_ssm;
    const at::Tensor& ic = *inter_conv;
    const int64_t tok = is.dim() == 5 ? is.size(1) : 0;
    TORCH_CHECK(
        is.scalar_type() == at::kFloat && is.dim() == 5 && is.size(2) == NVH && is.size(3) == DV && is.size(4) == DK &&
            is.stride(4) == 1 && is.stride(3) == DK && is.stride(2) == DV * DK && is.stride(1) == NVH * DV * DK,
        "qwen35: intermediate ssm state must be [rows, T, 16, 128, 128] fp32, got ",
        is.sizes(),
        " ",
        is.strides());
    // SGLang's windows: a [rows][4096][T + 2] buffer viewed as [rows][T][4096][3]
    TORCH_CHECK(
        ic.scalar_type() == dt && ic.dim() == 4 && ic.size(1) == tok && ic.size(2) == CONVD && ic.size(3) == 3 &&
            ic.stride(1) == 1 && ic.stride(3) == 1 && ic.stride(2) == tok + 2,
        "qwen35: intermediate conv windows must be the [rows, T, 4096, 3] view of a [rows, 4096, T + 2] buffer "
        "in the layer's dtype, got ",
        ic.scalar_type(),
        " ",
        ic.sizes(),
        " ",
        ic.strides());
    P.inter_ssm = is.data_ptr<float>();
    P.inter_ssm_stride = is.stride(0);
    P.inter_ssm_tstride = is.stride(1);
    P.inter_conv = reinterpret_cast<e16*>(ic.data_ptr());
    P.inter_conv_stride = ic.stride(0);
    // T0 + 2 of the bound scratch: a step of T <= T0 tokens writes T + 2 of them
    P.inter_conv_cstride = ic.stride(2);
    P.inter_idx = nullptr;  // a request's scratch row is its batch position
    D.inter_tokens = (int32_t)tok;
    D.inter_rows = std::min(is.size(0), ic.size(0));
  }
  D.caches_bound = 1;
}

// Binds an attention layer's caches into its descriptor: k_cache, v_cache
// ([slots, (1,) 256] each, the layer's dtype) with `page_size` tokens a page.
void qwen35_bind_attn_caches(at::Tensor& desc, const at::Tensor& k, const at::Tensor& v, int64_t page_size) {
  using namespace sgl_mono;
  LayerDesc& D = from_bytes(desc, "desc");
  TORCH_CHECK(D.kind == K_ATTN, "qwen35_bind_attn_caches: not an attention layer");
  check_device(k, "qwen35_bind_attn_caches");
  check_on(v, k.get_device(), "qwen35_bind_attn_caches", "v_cache");
  const at::ScalarType dt = D.bf ? at::kBFloat16 : at::kHalf;
  for (int i = 0; i < 2; ++i) {
    const at::Tensor& c = i ? v : k;
    TORCH_CHECK(
        c.scalar_type() == dt,
        "qwen35: ",
        i ? "v" : "k",
        "_cache must have the layer's dtype ",
        dtype_name(dt),
        " (an fp16 / bf16 mix of activations and KV cache is not supported), got ",
        c.scalar_type());
    TORCH_CHECK(
        (c.dim() == 3 && c.size(1) == 1 && c.size(2) == HD && c.stride(0) == HD && c.stride(2) == 1) ||
            (c.dim() == 2 && c.size(1) == HD && c.stride(0) == HD && c.stride(1) == 1),
        "qwen35: ",
        i ? "v" : "k",
        "_cache must be [slots, 1, 256] contiguous token rows, got ",
        c.sizes(),
        " ",
        c.strides());
    TORCH_CHECK((reinterpret_cast<uintptr_t>(c.data_ptr()) & 63) == 0, "qwen35: KV buffers must be 64-B aligned");
  }
  TORCH_CHECK(page_size >= 16 && page_size % 16 == 0, "qwen35: page size must be a multiple of 16, got ", page_size);
  D.P.kc = reinterpret_cast<e16*>(k.data_ptr());
  D.P.vc = reinterpret_cast<e16*>(v.data_ptr());
  D.P.ps = (int)page_size;
  D.caches_bound = 1;
}

// Counters and health words of a device: zeroes them and links every
// counters row to the health words, to the peer's health words (peer_health:
// the peer rank's copy mapped on this device, e.g. symmetric memory; none: the
// peer is not told of a failure), the health words to their host copy
// (pinned), and sets the wait deadline. Enqueued on the current stream.
void qwen35_init_counters(
    at::Tensor& counters,
    at::Tensor& health,
    at::Tensor& health_host,
    double deadline_s,
    const std::optional<at::Tensor>& peer_health) {
  using namespace sgl_mono;
  check_device(counters, "qwen35_init_counters");
  const int dev = counters.get_device();
  check_counters(counters, dev, "qwen35_init_counters", true);
  check_on(health, dev, "qwen35_init_counters", "health");
  TORCH_CHECK(
      health.scalar_type() == at::kInt && health.is_contiguous() && health.numel() == H_WORDS,
      "qwen35: health must be int32 [",
      H_WORDS,
      "] on the counters' device");
  TORCH_CHECK(
      health_host.is_cpu() && health_host.is_pinned() && health_host.scalar_type() == at::kInt &&
          health_host.is_contiguous() && health_host.numel() == H_WORDS,
      "qwen35: health_host must be a pinned int32 [",
      H_WORDS,
      "] host tensor");
  TORCH_CHECK(deadline_s > 0.0 && deadline_s <= 600.0, "qwen35: deadline_s must be in (0, 600], got ", deadline_s);
  uint64_t php = 0;
  if (peer_health.has_value()) {
    const at::Tensor& ph = *peer_health;
    TORCH_CHECK(
        ph.is_xpu() && ph.scalar_type() == at::kInt && ph.is_contiguous() && ph.numel() == H_WORDS,
        "qwen35: peer_health must be an int32 [",
        H_WORDS,
        "] XPU tensor (the peer's health words mapped on this device)");
    php = reinterpret_cast<uint64_t>(ph.data_ptr());
    TORCH_CHECK(php % 8 == 0 && reinterpret_cast<uint64_t>(health.data_ptr()) != php, "qwen35: bad peer_health");
  }
  const int64_t n = counters.size(0);
  at::Tensor c = at::zeros({n, CNT_WORDS}, at::TensorOptions().dtype(at::kInt));
  const uint64_t hp = reinterpret_cast<uint64_t>(health.data_ptr());
  const uint64_t dl = (uint64_t)(deadline_s * kClockHz);
  for (int64_t i = 0; i < n; ++i) {
    int32_t* cw = c.data_ptr<int32_t>() + i * CNT_WORDS;
    std::memcpy(cw + HEALTH_PTR, &hp, 8);
    std::memcpy(cw + DEADLINE, &dl, 8);
    std::memcpy(cw + PEER_HEALTH_PTR, &php, 8);
  }
  at::Tensor h = at::zeros({H_WORDS}, at::TensorOptions().dtype(at::kInt));
  const uint64_t hh = reinterpret_cast<uint64_t>(health_host.data_ptr());
  h.data_ptr<int32_t>()[H_HOST_LO] = (int32_t)(uint32_t)hh;
  h.data_ptr<int32_t>()[H_HOST_HI] = (int32_t)(uint32_t)(hh >> 32);
  const c10::DeviceGuard guard(counters.device());
  // a kernel still running may write the host copy: zero it after the stream
  c10::xpu::getCurrentXPUStream(dev).synchronize();
  health_host.zero_();
  counters.copy_(c);
  health.copy_(h);
}

// One decode step of layers [l0, l1) of the table `layers` (CPU uint8 [L,
// desc_bytes], caches bound) on the current XPU stream, one kernel a layer:
// layer l0 reads hidden / residual, layer l1 - 1 writes hid_out (MoE output,
// all-reduced) / res_out (residual after the mixer); [rows, 2048] each in the
// layers' dtype, rows = requests x tokens (request-major). tokens 1: a decode
// step (1, 2, 3, 4, 6 or 8 rows); tokens >= 2: a speculative-decoding verify
// step (GatedDeltaNet: the state after each token goes to the bound verify
// scratch, row = batch position; the committed state is read, not written).
// Metadata: GatedDeltaNet state_idx [>= requests] int32; attention page_table
// [>= requests, n] int32 (page indices), seq_lens [>= requests] int32 (with the
// new tokens), out_loc [>= rows] int64, positions [rows] (or mrope [3, rows])
// int64. counters: int32 [L, CNT_WORDS] (row l: layer l's). Returns false
// (nothing launched) for a shape without a kernel.
bool qwen35_decode_step(
    int64_t mode,
    int64_t i0,
    int64_t i1,
    const std::optional<at::Tensor>& ar,
    const at::Tensor& layers,
    int64_t l0,
    int64_t l1,
    const at::Tensor& hidden,
    const at::Tensor& residual,
    at::Tensor& hid_out,
    at::Tensor& res_out,
    const at::Tensor& positions,
    at::Tensor& workspace,
    at::Tensor& counters,
    int64_t tokens,
    const std::optional<at::Tensor>& state_idx,
    const std::optional<at::Tensor>& page_table,
    const std::optional<at::Tensor>& seq_lens,
    const std::optional<at::Tensor>& out_loc) {
  using namespace sgl_mono;
  const char* op = "qwen35_decode_step";
  check_device(hidden, op);
  const int dev = hidden.get_device();
  TORCH_CHECK(
      layers.is_cpu() && layers.scalar_type() == at::kByte && layers.is_contiguous() && layers.dim() == 2 &&
          layers.size(1) == (int64_t)sizeof(LayerDesc),
      "qwen35: layers must be a CPU uint8 [L, ",
      sizeof(LayerDesc),
      "] table (qwen35_layer_desc rows)");
  TORCH_CHECK(0 <= l0 && l0 < l1 && l1 <= layers.size(0), "qwen35: bad layer range [", l0, ", ", l1, ")");
  check_counters(counters, dev, op, true);
  TORCH_CHECK(counters.size(0) >= l1, "qwen35: counters must have a row for every layer of the range");
  check_workspace(workspace, dev, op);
  TORCH_CHECK(mode >= 0 && mode <= 2, "qwen35: mode must be 0, 1 or 2, got ", mode);
  check_on(ar, dev, op, "ar");
  check_on(residual, dev, op, "residual");
  check_on(hid_out, dev, op, "hid_out");
  check_on(res_out, dev, op, "res_out");
  check_on(positions, dev, op, "positions");
  check_on(state_idx, dev, op, "state_idx");
  check_on(page_table, dev, op, "page_table");
  check_on(seq_lens, dev, op, "seq_lens");
  check_on(out_loc, dev, op, "out_loc");
  const int64_t nb = rows_of(hidden);
  TORCH_CHECK(tokens >= 1 && nb % tokens == 0, "qwen35: rows = requests x tokens, got ", nb, " x ", tokens);
  const int64_t nrq = nb / tokens;
  if (tokens == 1)
    TORCH_CHECK(
        nb == 1 || nb == 2 || nb == 3 || nb == 4 || nb == 6 || nb == 8,
        "qwen35: batch of ",
        nb,
        " rows (supported: 1, 2, 3, 4, 6, 8)");
  const auto* table = reinterpret_cast<const LayerDesc*>(layers.data_ptr());
  TORCH_CHECK(table[l0].magic == kDescMagic, "qwen35: layer ", l0, " of the table is not a qwen35 descriptor");
  const bool bf = table[l0].bf != 0;
  const at::ScalarType dt = bf ? at::kBFloat16 : at::kHalf;
  const e16* in_h = t16(hidden, nb * HID, "hidden", dt);
  const e16* in_r = t16(residual, nb * HID, "residual", dt);
  e16* out_h = t16(hid_out, nb * HID, "hid_out", dt);
  e16* out_r = t16(res_out, nb * HID, "res_out", dt);
  bool any_attn = false, any_gdn = false;
  for (int64_t l = l0; l < l1; ++l) {
    const LayerDesc& D = table[l];
    TORCH_CHECK(D.magic == kDescMagic, "qwen35: layer ", l, " of the table is not a qwen35 descriptor");
    TORCH_CHECK(D.kind == K_GDN || D.kind == K_ATTN, "qwen35: kernel ", l, " is an MTP prologue, not a decoder layer");
    TORCH_CHECK((D.bf != 0) == bf, "qwen35: the layers of a launch share one dtype");
    TORCH_CHECK(D.caches_bound, "qwen35: layer ", l, " has no caches bound");
    any_attn |= D.kind == K_ATTN;
    any_gdn |= D.kind == K_GDN;
    if (D.kind == K_GDN && tokens > 1 && nb <= NBMAX) {  // (no verify kernel above NBMAX rows)
      TORCH_CHECK(
          D.inter_tokens >= tokens, "qwen35: layer ", l, " has no verify scratch for ", tokens, " tokens bound");
      TORCH_CHECK(
          nrq <= D.inter_rows,
          "qwen35: layer ",
          l,
          "'s verify scratch has ",
          D.inter_rows,
          " rows, the step ",
          nrq,
          " requests");
    }
  }
  if (tokens > 1 && nb > NBMAX) return false;  // no verify kernel for the shape
  Ptrs M{};                                    // the step's metadata
  if (any_attn) {
    TORCH_CHECK(
        page_table.has_value() && seq_lens.has_value() && out_loc.has_value(),
        "qwen35: the attention layers need page_table, seq_lens and out_loc");
    const at::Tensor& bt = *page_table;
    TORCH_CHECK(
        bt.scalar_type() == at::kInt && bt.dim() == 2 && bt.stride(1) == 1 && bt.size(0) >= nrq,
        "qwen35: page table must be int32 [>= requests, n] (unit stride)");
    check_vec(*seq_lens, at::kInt, nrq, "seq_lens (one a request)");
    check_vec(*out_loc, at::kLong, nb, "out_loc (one a row)");
    check_positions(positions, nb);
    M.block_table = bt.data_ptr<int>();
    M.bt_stride = (int)bt.stride(0);
    M.seq_lens = seq_lens->data_ptr<int>();
    M.slot = out_loc->data_ptr<int64_t>();
    M.pos = positions.data_ptr<int64_t>();  // [T] or mrope [3, T]: row 0
  }
  if (any_gdn) {
    TORCH_CHECK(state_idx.has_value(), "qwen35: the GatedDeltaNet layers need state_idx");
    check_vec(*state_idx, at::kInt, nrq, "state_idx (one a request)");
    M.state_idx = state_idx->data_ptr<int>();
  }

  const c10::DeviceGuard guard(hidden.device());
  sycl::queue& q = c10::xpu::getCurrentXPUStream(dev).queue();
  uint32_t* cnt = reinterpret_cast<uint32_t*>(counters.data_ptr<int32_t>());
  // a multi-layer launch passes the intermediate streams through the
  // workspace's ping-pong pairs
  const Carved c0 = carve(workspace.data_ptr(), cnt, nullptr, nullptr);
  // layer l's arguments: its descriptor, the step's metadata and streams
  auto layer_args = [&](int64_t l, Ptrs& P, Carved& c) {
    const LayerDesc& D = table[l];
    c = carve(workspace.data_ptr(), cnt + l * CNT_WORDS, D.mbox, D.peer);
    P = D.P;
    const int pi = (int)((l - l0) & 1) * 2;  // this layer's output pair
    const int po = 2 - pi;                   // the previous layer's
    P.hs_in = l == l0 ? in_h : c0.ping[po];
    P.res_in = l == l0 ? in_r : c0.ping[po + 1];
    P.hid_out = l == l1 - 1 ? out_h : c0.ping[pi];
    P.res_out = l == l1 - 1 ? out_r : c0.ping[pi + 1];
    if (D.kind == K_ATTN) {
      P.block_table = M.block_table;
      P.bt_stride = M.bt_stride;
      P.seq_lens = M.seq_lens;
      P.slot = M.slot;
      P.pos = M.pos;
    } else {
      P.state_idx = M.state_idx;
    }
  };
  const int pw = work_groups(dev);
  if (pw < NWG) {
    // fewer Xe cores than NWG (b60/): every mode runs the segments of the split
    // plan on pw work-groups, one kernel a layer (mode 0) or a segment (modes
    // 1 / 2; mode 2 runs plan items [i0, i1))
    const std::vector<PlanItem> plan = split_plan(table, (int)l0, (int)l1, mode == 2);
    size_t b = 0, e = plan.size();
    Split X;
    if (mode == 2) {
      TORCH_CHECK(
          ar.has_value() && ar->scalar_type() == at::kFloat && ar->is_contiguous() && ar->numel() >= nb * HID,
          "qwen35: mode 2 needs ar, a contiguous fp32 [>= rows, ",
          HID,
          "] tensor");
      TORCH_CHECK(0 <= i0 && i0 <= i1, "qwen35: bad plan range [", i0, ", ", i1, ")");
      X.ar = ar->data_ptr<float>();
      b = std::min((size_t)i0, plan.size());
      e = std::min((size_t)i1, plan.size());
    }
    bool launched = false;
    Segs G;
    auto flush = [&](int l) -> bool {  // launches G's segments of layer l
      if (G.n == 0) return true;
      const LayerDesc& D = table[l];
      Ptrs P;
      Carved c;
      layer_args(l, P, c);
      bool ok;
      if (tokens == 1)
        ok = D.kind == K_ATTN ? (bf ? b60_launch_attn<bf16>(q, pw, (int)nb, P, c.S, c.SB, X, G)
                                    : b60_launch_attn<sycl::half>(q, pw, (int)nb, P, c.S, c.SB, X, G))
                              : (bf ? b60_launch_gdn<bf16>(q, pw, (int)nb, P, c.S, c.SB, X, G)
                                    : b60_launch_gdn<sycl::half>(q, pw, (int)nb, P, c.S, c.SB, X, G));
      else
        ok = D.kind == K_ATTN
                 ? (bf ? b60_launch_attn_verify<bf16>(q, pw, (int)nb, (int)tokens, P, c.S, c.SB, X, G)
                       : b60_launch_attn_verify<sycl::half>(q, pw, (int)nb, (int)tokens, P, c.S, c.SB, X, G))
                 : (bf ? b60_launch_gdn_verify<bf16>(q, pw, (int)nb, (int)tokens, P, c.S, c.SB, X, G)
                       : b60_launch_gdn_verify<sycl::half>(q, pw, (int)nb, (int)tokens, P, c.S, c.SB, X, G));
      G.n = 0;
      // every kind has the same shapes: only the first launch can refuse
      TORCH_CHECK(ok || !launched, "qwen35: a shape refused after the first launch");
      launched |= ok;
      return ok;
    };
    for (size_t i = b; i < e; ++i) {
      const PlanItem& it = plan[i];
      if (it.seg == ITEM_AR) continue;
      if (it.seg == ITEM_ADVANCE) {
        if (!flush(it.l)) return false;
        advance_counters(q, cnt + it.l * CNT_WORDS, (int)nb);
        continue;
      }
      G.seg[G.n++] = it.seg;
      if (mode != 0 && !flush(it.l)) return false;
    }
    return true;
  }
  if (mode == 0) {
    for (int64_t l = l0; l < l1; ++l) {
      const LayerDesc& D = table[l];
      Ptrs P;
      Carved c;
      layer_args(l, P, c);
      bool ok = true;
      if (tokens == 1) {
        if (D.kind == K_ATTN)
          bf ? launch_attn<bf16>(q, (int)nb, P, c.S, c.SB) : launch_attn<sycl::half>(q, (int)nb, P, c.S, c.SB);
        else
          bf ? launch_gdn<bf16>(q, (int)nb, P, c.S, c.SB) : launch_gdn<sycl::half>(q, (int)nb, P, c.S, c.SB);
      } else if (D.kind == K_ATTN) {
        ok = bf ? launch_attn_verify<bf16>(q, (int)nb, (int)tokens, P, c.S, c.SB)
                : launch_attn_verify<sycl::half>(q, (int)nb, (int)tokens, P, c.S, c.SB);
      } else {
        ok = bf ? launch_gdn_verify<bf16>(q, (int)nb, (int)tokens, P, c.S, c.SB)
                : launch_gdn_verify<sycl::half>(q, (int)nb, (int)tokens, P, c.S, c.SB);
      }
      // the verify shapes are the same for both kinds: only the first layer can
      // refuse, before anything was launched
      if (!ok) {
        TORCH_CHECK(l == l0, "qwen35: a verify shape refused after the first layer");
        return false;
      }
    }
    return true;
  }

  // modes 1 / 2: one kernel a segment; mode 2 runs plan items [i0, i1), the
  // caller all-reduces ar ([>= rows][2048] fp32) at the ITEM_AR items
  // (qwen35_split_ar_slots)
  const std::vector<PlanItem> plan = split_plan(table, (int)l0, (int)l1, mode == 2);
  size_t b = 0, e = plan.size();
  Split X;
  if (mode == 2) {
    TORCH_CHECK(
        ar.has_value() && ar->scalar_type() == at::kFloat && ar->is_contiguous() && ar->numel() >= nb * HID,
        "qwen35: mode 2 needs ar, a contiguous fp32 [>= rows, ",
        HID,
        "] tensor");
    TORCH_CHECK(0 <= i0 && i0 <= i1, "qwen35: bad plan range [", i0, ", ", i1, ")");
    X.ar = ar->data_ptr<float>();
    b = std::min((size_t)i0, plan.size());
    e = std::min((size_t)i1, plan.size());
  }
  bool launched = false;
  for (size_t i = b; i < e; ++i) {
    const PlanItem& it = plan[i];
    if (it.seg == ITEM_AR) continue;
    if (it.seg == ITEM_ADVANCE) {  // the per-call counters, as the one-kernel layer advances them
      advance_counters(q, cnt + it.l * CNT_WORDS, (int)nb);
      continue;
    }
    const LayerDesc& D = table[it.l];
    Ptrs P;
    Carved c;
    layer_args(it.l, P, c);
    X.seg = it.seg;
    bool ok;
    if (tokens == 1)
      ok = D.kind == K_ATTN ? (bf ? launch_attn_split<bf16>(q, (int)nb, P, c.S, c.SB, X)
                                  : launch_attn_split<sycl::half>(q, (int)nb, P, c.S, c.SB, X))
                            : (bf ? launch_gdn_split<bf16>(q, (int)nb, P, c.S, c.SB, X)
                                  : launch_gdn_split<sycl::half>(q, (int)nb, P, c.S, c.SB, X));
    else
      ok = D.kind == K_ATTN ? (bf ? launch_attn_verify_split<bf16>(q, (int)nb, (int)tokens, P, c.S, c.SB, X)
                                  : launch_attn_verify_split<sycl::half>(q, (int)nb, (int)tokens, P, c.S, c.SB, X))
                            : (bf ? launch_gdn_verify_split<bf16>(q, (int)nb, (int)tokens, P, c.S, c.SB, X)
                                  : launch_gdn_verify_split<sycl::half>(q, (int)nb, (int)tokens, P, c.S, c.SB, X));
    // every kind has the same split shapes: only the first segment can refuse
    if (!ok) {
      TORCH_CHECK(!launched, "qwen35: a split shape refused after the first segment");
      return false;
    }
    launched = true;
  }
  return true;
}

// The plan items of a mode-2 step of layers [l0, l1) of the table where the
// caller all-reduces ar, then the plan length.
std::vector<int64_t> qwen35_split_ar_slots(const at::Tensor& layers, int64_t l0, int64_t l1) {
  TORCH_CHECK(
      layers.is_cpu() && layers.scalar_type() == at::kByte && layers.is_contiguous() && layers.dim() == 2 &&
          layers.size(1) == (int64_t)sizeof(LayerDesc),
      "qwen35: layers must be a CPU uint8 [L, ",
      sizeof(LayerDesc),
      "] table");
  TORCH_CHECK(0 <= l0 && l0 < l1 && l1 <= layers.size(0), "qwen35: bad layer range [", l0, ", ", l1, ")");
  const auto* table = reinterpret_cast<const LayerDesc*>(layers.data_ptr());
  for (int64_t l = l0; l < l1; ++l)
    TORCH_CHECK(
        table[l].magic == kDescMagic && (table[l].kind == K_GDN || table[l].kind == K_ATTN),
        "qwen35: layer ",
        l,
        " of the table is not a decoder layer descriptor");
  const std::vector<PlanItem> v = split_plan(table, (int)l0, (int)l1, true);
  std::vector<int64_t> o;
  for (size_t i = 0; i < v.size(); ++i)
    if (v[i].seg == ITEM_AR) o.push_back((int64_t)i);
  o.push_back((int64_t)v.size());
  return o;
}

// The MTP drafter prologue (its own kernel, qwen35_mtp_desc): out = fc([
// norm_e(embed); norm_h(hidden)]) for rows embed / hidden [rows, 2048] (the
// model's dtype). counters: its int32 [CNT_WORDS] row. False (nothing
// launched) for a row count without a kernel.
bool qwen35_mtp_prologue(
    const at::Tensor& desc,
    const at::Tensor& embed,
    const at::Tensor& hidden,
    at::Tensor& out,
    at::Tensor& workspace,
    at::Tensor& counters) {
  using namespace sgl_mono;
  const char* op = "qwen35_mtp_prologue";
  const LayerDesc& D = from_bytes(desc, "desc");
  TORCH_CHECK(D.kind == K_MTP, "qwen35: ", op, ": not an MTP prologue descriptor");
  check_device(embed, op);
  const int dev = embed.get_device();
  check_on(hidden, dev, op, "hidden");
  check_on(out, dev, op, "out");
  check_workspace(workspace, dev, op);
  check_counters(counters, dev, op, false);
  const int64_t nb = rows_of(embed);
  TORCH_CHECK(rows_of(hidden) >= nb && rows_of(out) >= nb, "qwen35: MTP hidden / out smaller than the batch");
  const at::ScalarType dt = D.bf ? at::kBFloat16 : at::kHalf;
  Ptrs P = D.P;
  P.hs_in = t16(embed, nb * HID, "embed", dt);
  P.res_in = t16(hidden, nb * HID, "hidden", dt);
  P.hid_out = t16(out, nb * HID, "out", dt);
  const Carved c =
      carve(workspace.data_ptr(), reinterpret_cast<uint32_t*>(counters.data_ptr<int32_t>()), D.mbox, D.peer);
  const c10::DeviceGuard guard(embed.device());
  sycl::queue& q = c10::xpu::getCurrentXPUStream(dev).queue();
  const int pw = work_groups(dev);
  if (pw < NWG)
    return D.bf ? b60_launch_mtp<bf16>(q, pw, (int)nb, P, c.S, D.rank)
                : b60_launch_mtp<sycl::half>(q, pw, (int)nb, P, c.S, D.rank);
  return D.bf ? launch_mtp<bf16>(q, (int)nb, P, c.S, D.rank) : launch_mtp<sycl::half>(q, (int)nb, P, c.S, D.rank);
}

// The work-groups a launch on `device` has: 32 (the layer kernels; >= 32 Xe
// cores) or the Xe cores of a smaller part (20 on an Arc Pro B60; the b60/
// kernels run the 32 work-groups' parts in turn).
int64_t qwen35_work_groups(int64_t device) {
  return work_groups(device);
}

// Test hook: launch every kernel on at most n work-groups (20 .. 32; never
// more than the device's Xe cores), e.g. 20 on an Arc Pro B70 to run the
// kernels of a 20-core part; 0: the device's. Process-wide, between steps (the
// kernels of one step must agree).
void qwen35_force_work_groups(int64_t n) {
  TORCH_CHECK(
      n == 0 || (n >= kMinWorkGroups && n <= sgl_mono::NWG),
      "qwen35_force_work_groups: n must be 0 or ",
      kMinWorkGroups,
      " .. ",
      sgl_mono::NWG,
      ", got ",
      n);
  g_force_work_groups.store((int)n, std::memory_order_relaxed);
}

// Test hook (fault injection): shifts one kernel's mailbox tags on this rank
// by `delta` (its int32 [CNT_WORDS] counters row; queued on the current
// stream), so the peer never sees the words it waits for: both ranks give up
// their waits and record the failure. arrivals: sets its stage arrival
// counters back by `delta` instead, so its next call waits for work-group
// arrivals on this card that never come (as when not all work-groups are
// resident): the counter waits give up.
void qwen35_fault_inject(at::Tensor& counters, int64_t delta, bool arrivals) {
  using namespace sgl_mono;
  check_device(counters, "qwen35_fault_inject");
  check_counters(counters, counters.get_device(), "qwen35_fault_inject", false);
  uint32_t* cnt = reinterpret_cast<uint32_t*>(counters.data_ptr<int32_t>());
  const uint32_t d = (uint32_t)delta;
  const c10::DeviceGuard guard(counters.device());
  sycl::queue& q = c10::xpu::getCurrentXPUStream(counters.get_device()).queue();
  q.single_task<class Qwen35FaultInjectK>([=] {
    if (arrivals) {  // the stage arrival counters fall behind the epoch: the waits
                     // for them never complete (the tickets and the epoch stay)
      for (int c : {C_A, C_B, C_C, C_O, C_GU})
        cnt[c * 16] -= d;
      return;
    }
    for (int i = 0; i < 4; ++i)
      cnt[MBT + i] += d;  // rows 1, 2, 4, 8
    for (int i = 0; i < 2; ++i)
      cnt[MBT2 + i] += d;  // rows 3, 6
    for (int i = 0; i < 6; ++i)
      cnt[MTPT + i] += d;  // the MTP prologue's row counts (unused words in a decoder layer)
  });
}
