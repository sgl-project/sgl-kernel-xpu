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

// Host side of the mono decode layers (Qwen3.5 / Qwen3.6 MoE, TP2, Xe2): the
// device check, the IPC mailbox regions, the per-layer registry (SGLang
// weights registered once, caches bound separately and again whenever the
// pools move, one scratch per layer) and the per-step launch on the current
// torch XPU queue. The kernels are in kernels/mono_decode/xe20/, one
// translation unit (and library) per kernel kind and dtype.
#include <ATen/ATen.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <level_zero/ze_api.h>
#include <sys/syscall.h>
#include <torch/all.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

#include "Utils.h"
#include "kernels/mono_decode/xe20/mono_layers_s.hpp"
#include "kernels/mono_decode/xe20/mono_mtp.hpp"
#include "sgl_kernel_ops.h"  // the ops, exported

// the kernels are compiled in their own translation units (one per layer kind
// and dtype)
namespace sgl_mono {
#define MONO_EXTERN_LAUNCHERS(DT)                                                                                    \
  extern template void launch_gdn<DT>(sycl::queue&, int, const Ptrs&, const Scratch&, const ScratchB&);              \
  extern template void launch_attn<DT>(sycl::queue&, int, const Ptrs&, const Scratch&, const ScratchB&);             \
  extern template bool launch_gdn_verify<DT>(sycl::queue&, int, int, const Ptrs&, const Scratch&, const ScratchB&);  \
  extern template bool launch_attn_verify<DT>(sycl::queue&, int, int, const Ptrs&, const Scratch&, const ScratchB&); \
  extern template bool launch_mtp<DT>(sycl::queue&, int, const Ptrs&, const Scratch&, int);
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

#define ZE_CHECK(call)                                                                                        \
  do {                                                                                                        \
    ze_result_t _r = (call);                                                                                  \
    TORCH_CHECK(_r == ZE_RESULT_SUCCESS, #call " failed: ze_result 0x", std::hex, static_cast<uint32_t>(_r)); \
  } while (0)

// -----------------------------------------------------------------------------
// device check: Xe2 (Battlemage) with at least 32 Xe cores. The layer kernels
// keep 32 work-groups resident, one per Xe core, that wait on each other: a
// part with fewer Xe cores (BMG-G21: 20; Arc Pro B65) cannot run them.
constexpr int kResidentWorkGroups = sgl_mono::NWG;

bool supported_device(at::DeviceIndex d) {
  if (!is_bmg(d)) return false;
  const auto& dev = c10::xpu::get_raw_device(d);
  const int xe_cores = dev.get_info<sycl::ext::intel::info::device::gpu_slices>() *
                       dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  return xe_cores >= kResidentWorkGroups;
}

bool xe2_device(int64_t device) {
  static std::array<std::atomic<int8_t>, 64> known{};  // 0 unknown, 1 / -1
  const at::DeviceIndex d = device < 0 ? c10::xpu::current_device() : (at::DeviceIndex)device;
  if (d < 0 || d >= (at::DeviceIndex)known.size()) return supported_device(d);
  int8_t v = known[d].load(std::memory_order_relaxed);
  if (v == 0) {
    v = supported_device(d) ? 1 : -1;
    known[d].store(v, std::memory_order_relaxed);
  }
  return v == 1;
}

void check_device(const char* op, int64_t device = -1) {
  TORCH_CHECK(
      xe2_device(device),
      op,
      ": unsupported device (the mono decode kernels need an Xe2 (Battlemage) GPU with at least 32 Xe cores; "
      "check mono_decode_supported() first).");
}

// -----------------------------------------------------------------------------
// IPC regions
constexpr int64_t kCookieOff = 0;  // first 64 B of a region: cookie (mailboxes start at 64 B)
constexpr int64_t kHdr = 64;

struct Region {
  ze_context_handle_t ctx = nullptr;
  ze_device_handle_t dev = nullptr;
  int device = -1;  // the torch XPU device that created it: its queue serves every later use
  char* own = nullptr;
  char* peer = nullptr;
  int64_t bytes = 0;
  ze_ipc_mem_handle_t ipc{};
  bool exported = false;
  uint64_t cookie = 0;
  int imported_fd = -1;
};
std::mutex g_mu;
std::vector<std::unique_ptr<Region>> g_regions;

Region& region(int64_t h) {
  std::lock_guard<std::mutex> lk(g_mu);
  TORCH_CHECK(h >= 0 && h < (int64_t)g_regions.size() && g_regions[h], "mono: invalid IPC handle ", h);
  return *g_regions[h];
}

// wait deadline of the in-kernel mailbox waits, in cycles of the sub-group
// clock (about 2.8 GHz on B70)
constexpr double kClockHz = 2.8e9;
std::atomic<uint64_t> g_deadline_cycles{sgl_mono::WAIT_CYCLES_DEFAULT};

struct Health {
  int* device = nullptr;  // health words (device memory)
  int* host = nullptr;    // the host copy of words 0 .. 7 (host memory)
};
Health& health(int device) {
  static std::mutex mu;
  static std::unordered_map<int, Health> words;
  std::lock_guard<std::mutex> lk(mu);
  Health& w = words[device];
  if (!w.device) {
    sycl::queue& q = c10::xpu::getCurrentXPUStream(device).queue();
    w.host = sycl::malloc_host<int>(sgl_mono::H_WORDS, q);
    TORCH_CHECK(w.host, "mono: host allocation of the health words failed");
    std::memset(w.host, 0, sgl_mono::H_WORDS * sizeof(int));
    int init[sgl_mono::H_WORDS] = {};
    const uint64_t hp = reinterpret_cast<uint64_t>(w.host);
    init[sgl_mono::H_HOST_LO] = (int)(uint32_t)hp;
    init[sgl_mono::H_HOST_HI] = (int)(uint32_t)(hp >> 32);
    w.device = sycl::malloc_device<int>(sgl_mono::H_WORDS, q);
    q.memcpy(w.device, init, sizeof(init)).wait();
  }
  return w;
}
int* health_words(int device) {
  return health(device).device;
}

// -----------------------------------------------------------------------------
// layers
struct Layer {
  bool attn = false;
  bool mtp = false;  // MTP drafter prologue (mono_register_mtp)
  int rank = 0;      // MTP: this rank's half of the fc output rows
  bool caches_bound = false;
  int inter_tokens = 0;              // GatedDeltaNet: tokens a request of the bound verify scratch (0: none)
  int64_t inter_rows = 0;            // and its rows (requests)
  at::ScalarType dtype = at::kHalf;  // fp16 or bf16: every 16-bit tensor
  Ptrs P{};
  Scratch S{};
  ScratchB SB{};
  void* scratch = nullptr;    // shared by the layers of a device (shared_scratch), or the MTP prologue's own
  void* scratch_b = nullptr;  // shared (shared_scratch)
  void* cnt_mem = nullptr;    // this layer's counters and mailbox tags (CNT_WORDS words)
  int device = -1;
  std::vector<at::Tensor> keep;  // fp32 copies of 16-bit scales
};
std::unordered_map<int64_t, Layer> g_layers;
// batched attention chunk partials, one buffer per device shared by all
// attention layers (they run one at a time on the device's queue)
std::unordered_map<int, void*> g_attn_part;

// The layers' intermediates (scratch, scratch_b) are one pair of buffers per
// device shared by all layers: the layers' kernels run one at a time on the
// device's queue, and each kernel writes an intermediate before it reads it.
// Only the counters (stage arrival counts, epoch, mailbox tags) are the layer's
// own: they carry state from one step to the next (zero when allocated).
struct SharedScratch {
  void* scratch = nullptr;
  void* scratch_b = nullptr;
};
std::unordered_map<int, SharedScratch> g_shared_scratch;
SharedScratch& shared_scratch(int device, sycl::queue& q) {
  SharedScratch& sh = g_shared_scratch[device];
  if (!sh.scratch) {
    sh.scratch = sycl::malloc_device(sgl_mono::scratch_bytes(), q);
    TORCH_CHECK(sh.scratch, "mono: device allocation of the scratch failed");
    q.memset(sh.scratch, 0, sgl_mono::scratch_bytes()).wait();
    sh.scratch_b = sycl::malloc_device(sgl_mono::scratch_b_bytes(), q);
    TORCH_CHECK(sh.scratch_b, "mono: device allocation of the batched scratch failed");
    q.memset(sh.scratch_b, 0, sgl_mono::scratch_b_bytes()).wait();
  }
  return sh;
}
constexpr size_t cnt_bytes() {
  return (sgl_mono::CNT_WORDS * 4 + 255) & ~size_t(255);
}

// the device's health words and the wait deadline, for the slow path of the
// layer's mailbox waits
void set_health(sycl::queue& q, const Layer& L) {
  const uint64_t h = reinterpret_cast<uint64_t>(health_words(L.device));
  const uint64_t dl = g_deadline_cycles.load();
  q.memcpy(L.S.cnt + sgl_mono::HEALTH_PTR, &h, sizeof(h)).wait();
  q.memcpy(L.S.cnt + sgl_mono::DEADLINE, &dl, sizeof(dl)).wait();
  const uint64_t mb = reinterpret_cast<uint64_t>(L.S.mbox);
  q.memcpy(L.S.cnt + sgl_mono::MBOX_PTR, &mb, sizeof(mb)).wait();
}

const char* dtype_name(at::ScalarType t) {
  return c10::toString(t);
}

// fp8 [N][K] linear weight: row-major [N, K], or its [K, N] transpose view
const uint8_t* w_nk(const at::Tensor& w, int64_t n, int64_t k, const char* name) {
  TORCH_CHECK(w.scalar_type() == at::kFloat8_e4m3fn, "mono: ", name, " must be float8_e4m3fn, got ", w.scalar_type());
  const bool nk = w.dim() == 2 && w.size(0) == n && w.size(1) == k && w.stride(0) == k && w.stride(1) == 1;
  const bool kn = w.dim() == 2 && w.size(0) == k && w.size(1) == n && w.stride(0) == 1 && w.stride(1) == k;
  TORCH_CHECK(
      nk || kn,
      "mono: ",
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
// 128x128 block scales, logical [N / 128][K / 128] (or [K / 128][N / 128]):
// a non-contiguous or 16-bit tensor is copied to row-major fp32 (kept by the
// layer)
const float* scale_blk(Layer& L, const at::Tensor& s, int64_t nb, int64_t kb, const char* name) {
  TORCH_CHECK(s.dim() == 2, "mono: ", name, " must be 2-D block scales, got ", s.sizes());
  // SGLang keeps them logical [N / 128, K / 128] (stored column-major)
  at::Tensor t = (s.size(0) == nb && s.size(1) == kb) ? s : (s.size(0) == kb && s.size(1) == nb) ? s.t() : s;
  TORCH_CHECK(
      t.size(0) == nb && t.size(1) == kb,
      "mono: ",
      name,
      " must be [",
      nb,
      ", ",
      kb,
      "] 128x128 block scales, got ",
      s.sizes());
  if (t.scalar_type() != at::kFloat || !t.is_contiguous()) {
    t = t.to(at::kFloat).contiguous();
    L.keep.push_back(t);
  }
  return t.data_ptr<float>();
}
// expert block scales [E][nb][kb]
const float* scale_exp(Layer& L, const at::Tensor& s, int64_t e, int64_t nb, int64_t kb, const char* name) {
  TORCH_CHECK(
      s.dim() == 3 && s.size(0) == e && s.size(1) == nb && s.size(2) == kb,
      "mono: ",
      name,
      " must be [",
      e,
      ", ",
      nb,
      ", ",
      kb,
      "] 128x128 block scales, got ",
      s.sizes());
  at::Tensor t = s;
  if (t.scalar_type() != at::kFloat || !t.is_contiguous()) {
    t = t.to(at::kFloat).contiguous();
    L.keep.push_back(t);
  }
  return t.data_ptr<float>();
}
// the layer's dtype, from its first 16-bit tensor
at::ScalarType dtype16(const at::Tensor& t, const char* name) {
  TORCH_CHECK(
      t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16,
      "mono: ",
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
      "mono: ",
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

// fp16 view of a checked 16-bit weight for the DPAS paths: the tensor itself
// in an fp16 layer, else a copy rounded to fp16 once (the rounding the kernels
// would otherwise apply on every call), kept with the layer
const e16* half_copy(Layer& L, const at::Tensor& t, const e16* own) {
  if (t.scalar_type() == at::kHalf) return own;
  at::Tensor h = t.to(at::kHalf).contiguous();
  L.keep.push_back(h);
  return reinterpret_cast<const e16*>(h.data_ptr());
}

Layer& layer_of(int64_t layer, const at::Tensor& like) {
  auto it = g_layers.find(layer);
  TORCH_CHECK(it != g_layers.end(), "mono: layer ", layer, " not registered");
  TORCH_CHECK(like.get_device() == it->second.device, "mono: layer ", layer, " registered on another device");
  return it->second;
}

// Every tensor of an op is an XPU tensor of the layer's device: a CPU or
// other-device tensor would be handed to the kernels as a device pointer.
void check_on(const at::Tensor& t, int device, const char* op, const char* name) {
  TORCH_CHECK(
      t.is_xpu() && t.get_device() == device,
      "mono: ",
      op,
      ": ",
      name,
      " must be an XPU tensor on device ",
      device,
      " (the layer's), got ",
      t.device());
}
void check_on(const std::optional<at::Tensor>& t, int device, const char* op, const char* name) {
  if (t.has_value()) check_on(*t, device, op, name);
}
// a metadata vector the kernels read as a contiguous array [>= n]
void check_vec(const at::Tensor& t, at::ScalarType dt, int64_t n, const char* name) {
  TORCH_CHECK(
      t.scalar_type() == dt && t.dim() == 1 && t.size(0) >= n && t.stride(0) == 1,
      "mono: ",
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
      "mono: positions must be an int64 [>= ",
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

}  // namespace

// -----------------------------------------------------------------------------
// ops
bool mono_decode_supported(int64_t device) {
  return xe2_device(device);
}

int64_t mono_max_context() {
  return 512 * 1024;
}

at::Tensor mono_health(int64_t device) {
  check_device("mono_health", device);
  const int d = device < 0 ? (int)c10::xpu::current_device() : (int)device;
  return torch::from_blob(
      health_words(d), {sgl_mono::H_WORDS}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kXPU, d));
}

at::Tensor mono_health_host(int64_t device) {
  check_device("mono_health_host", device);
  const int d = device < 0 ? (int)c10::xpu::current_device() : (int)device;
  return torch::from_blob(health(d).host, {sgl_mono::H_WORDS}, torch::TensorOptions().dtype(torch::kInt32));
}

void mono_set_deadline(double seconds) {
  TORCH_CHECK(seconds > 0.0 && seconds <= 600.0, "mono: deadline must be in (0, 600] s, got ", seconds);
  const uint64_t dl = (uint64_t)(seconds * kClockHz);
  g_deadline_cycles.store(dl);
  for (auto& kv : g_layers) {
    sycl::queue& q = c10::xpu::getCurrentXPUStream(kv.second.device).queue();
    q.memcpy(kv.second.S.cnt + sgl_mono::DEADLINE, &dl, sizeof(dl)).wait();
  }
}

int64_t mono_ipc_create(int64_t bytes) {
  check_device("mono_ipc_create");
  auto r = std::make_unique<Region>();
  r->device = (int)c10::xpu::current_device();
  sycl::queue& q = c10::xpu::getCurrentXPUStream(r->device).queue();
  r->ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
  r->dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
  r->bytes = kHdr + bytes;
  ze_device_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  void* p = nullptr;
  ZE_CHECK(zeMemAllocDevice(r->ctx, &desc, r->bytes, 4096, r->dev, &p));
  r->own = static_cast<char*>(p);
  q.memset(r->own, 0, r->bytes).wait();
  r->cookie = ((uint64_t)getpid() << 32) ^ (uint64_t)(uintptr_t)r->own ^ 0x9e3779b97f4a7c15ull;
  q.memcpy(r->own + kCookieOff, &r->cookie, sizeof(uint64_t)).wait();
  std::lock_guard<std::mutex> lk(g_mu);
  g_regions.push_back(std::move(r));
  return (int64_t)g_regions.size() - 1;
}

// [pid, fd, cookie lo, cookie hi, raw ze_ipc_mem_handle_t words]
std::vector<int64_t> mono_ipc_export(int64_t h) {
  Region& r = region(h);
  if (!r.exported) {
    ZE_CHECK(zeMemGetIpcHandle(r.ctx, r.own, &r.ipc));
    r.exported = true;
  }
  uint64_t fd = 0;
  if (zeMemGetFileDescriptorFromIpcHandleExp(r.ctx, r.ipc, &fd) != ZE_RESULT_SUCCESS) {
    int fd32 = 0;  // the Intel handle starts with the dma-buf fd
    std::memcpy(&fd32, r.ipc.data, sizeof(fd32));
    fd = (uint64_t)fd32;
  }
  std::vector<int64_t> out{
      (int64_t)getpid(), (int64_t)fd, (int64_t)(r.cookie & 0xffffffffu), (int64_t)(r.cookie >> 32)};
  uint32_t w[ZE_MAX_IPC_HANDLE_SIZE / 4];
  std::memcpy(w, r.ipc.data, sizeof(w));
  for (uint32_t v : w)
    out.push_back(v);
  return out;
}

// local_fd >= 0: the peer's dma-buf fd, already passed into this process (unix
// socket SCM_RIGHTS); the handle is opened with its fd field patched. Otherwise
// tries (0) the raw peer handle (the driver imports the fd itself), then (1)
// pidfd_getfd (needs ptrace permission). A mapping is accepted only if the
// peer's cookie reads back.
int64_t mono_ipc_open(int64_t h, std::vector<int64_t> peer, int64_t local_fd) {
  Region& r = region(h);
  TORCH_CHECK(r.peer == nullptr, "mono: peer already opened");
  TORCH_CHECK(peer.size() == 4 + ZE_MAX_IPC_HANDLE_SIZE / 4, "mono: bad peer handle info");
  const uint64_t peer_cookie = (uint64_t)(uint32_t)peer[2] | ((uint64_t)(uint32_t)peer[3] << 32);
  sycl::queue& q = c10::xpu::getCurrentXPUStream(r.device).queue();  // the region's own device
  auto verify = [&](void* p) {
    uint64_t c = 0;
    q.memcpy(&c, static_cast<char*>(p) + kCookieOff, sizeof(c)).wait();
    if (c == peer_cookie) return true;
    zeMemCloseIpcHandle(r.ctx, p);
    return false;
  };
  ze_ipc_mem_handle_t raw{};
  uint32_t w[ZE_MAX_IPC_HANDLE_SIZE / 4];
  for (size_t i = 0; i < ZE_MAX_IPC_HANDLE_SIZE / 4; ++i)
    w[i] = (uint32_t)peer[4 + i];
  std::memcpy(raw.data, w, sizeof(w));
  void* p = nullptr;
  int64_t mode = -1;
  // the handle with its fd (and the exporter pid) patched to a local fd
  auto open_with_fd = [&](int fd) {
    ze_ipc_mem_handle_t ph = raw;
    int fd32 = fd;
    std::memcpy(ph.data, &fd32, sizeof(fd32));
    uint32_t pw[ZE_MAX_IPC_HANDLE_SIZE / 4];
    std::memcpy(pw, ph.data, sizeof(pw));
    for (int i = 1; i < ZE_MAX_IPC_HANDLE_SIZE / 4; ++i)
      if (pw[i] == (uint32_t)peer[0]) pw[i] = (uint32_t)getpid();
    std::memcpy(ph.data + 4, pw + 1, sizeof(pw) - 4);
    const ze_result_t rr = zeMemOpenIpcHandle(r.ctx, r.dev, ph, 0, &p);
    return rr == ZE_RESULT_SUCCESS && verify(p);
  };
  if (local_fd >= 0) {  // fd received over a unix socket: no ptrace permission needed
    TORCH_CHECK(open_with_fd((int)local_fd), "mono: could not open the peer IPC handle with the received fd");
    r.imported_fd = (int)local_fd;
    r.peer = static_cast<char*>(p);
    return 2;
  }
  ze_result_t r0 = zeMemOpenIpcHandle(r.ctx, r.dev, raw, 0, &p);
  if (r0 == ZE_RESULT_SUCCESS && verify(p)) mode = 0;
  if (mode < 0) {
    int pidfd = (int)syscall(SYS_pidfd_open, (pid_t)peer[0], 0);
    TORCH_CHECK(pidfd >= 0, "mono: pidfd_open(", peer[0], ") failed, errno ", errno);
    int fd = (int)syscall(SYS_pidfd_getfd, pidfd, (int)peer[1], 0);
    const int err = errno;
    close(pidfd);
    TORCH_CHECK(fd >= 0, "mono: pidfd_getfd failed, errno ", err);
    if (open_with_fd(fd)) {
      mode = 1;
      r.imported_fd = fd;
    } else
      close(fd);
    TORCH_CHECK(mode >= 0, "mono: could not open the peer IPC handle (raw 0x", std::hex, (uint32_t)r0, ")");
  }
  r.peer = static_cast<char*>(p);
  return mode;
}

int64_t mono_ipc_ptr(int64_t h, bool peer) {
  Region& r = region(h);
  char* p = peer ? r.peer : r.own;
  TORCH_CHECK(p != nullptr, "mono: region not opened");
  return (int64_t)(uintptr_t)(p + kHdr);
}

int64_t mono_mbox_bytes() {
  return (int64_t)(sgl_mono::MBOX_WORDS_B * sizeof(uint64_t));
}

// Registers layer `layer`'s weights (SGLang tensors, used in place; checked).
// Registering a layer again keeps its scratch (and so its mailbox tags).
void mono_register_layer(int64_t layer, bool attn, int64_t mbox, int64_t peer, const std::vector<at::Tensor>& t) {
  using namespace sgl_mono;
  TORCH_CHECK(t.size() == (attn ? 18u : 21u), "mono: wrong number of tensors for layer ", layer);
  TORCH_CHECK(t[0].is_xpu(), "mono: mono_register_layer: the weights must be XPU tensors, got ", t[0].device());
  Layer L;
  L.attn = attn;
  L.device = t[0].get_device();
  c10::DeviceGuard guard(t[0].device());  // the scratch and the queue are the weights' device's
  check_device("mono_register_layer", L.device);
  for (const at::Tensor& w : t)
    check_on(w, L.device, "mono_register_layer", "a weight");
  const at::ScalarType dt = L.dtype = dtype16(t[0], "input_norm_w");
  Ptrs& P = L.P;
  P.in_norm_w = t16(t[0], HID, "input_norm_w", dt);
  P.post_norm_w = t16(t[1], HID, "post_norm_w", dt);
  P.router_w = t16(t[2], (int64_t)NE * HID, "router_w", dt);
  P.sgate_w = t16(t[3], HID, "shared_gate_w", dt);
  const int64_t ok = attn ? NQ * HD : VD;
  P.out_w = w_nk(t[4], HID, ok, attn ? "o_proj" : "out_proj");
  P.out_s = scale_blk(L, t[5], HID / QB, ok / QB, "out scale");
  TORCH_CHECK(
      t[6].scalar_type() == at::kFloat8_e4m3fn && t[6].is_contiguous() && t[6].dim() == 3 && t[6].size(0) == NE &&
          t[6].size(1) == 2 * INTER && t[6].size(2) == HID,
      "mono: w13 must be contiguous [256, 512, 2048] e4m3 (SGLang FusedMoE layout per rank), got ",
      t[6].sizes());
  TORCH_CHECK(
      t[8].scalar_type() == at::kFloat8_e4m3fn && t[8].is_contiguous() && t[8].dim() == 3 && t[8].size(0) == NE &&
          t[8].size(1) == HID && t[8].size(2) == INTER,
      "mono: w2 must be contiguous [256, 2048, 256] e4m3, got ",
      t[8].sizes());
  P.w13 = static_cast<const uint8_t*>(t[6].data_ptr());
  P.w13_s = scale_exp(L, t[7], NE, 2 * INTER / QB, HID / QB, "w13 scale");
  P.w2 = static_cast<const uint8_t*>(t[8].data_ptr());
  P.w2_s = scale_exp(L, t[9], NE, HID / QB, INTER / QB, "w2 scale");
  P.sw13 = w_nk(t[10], 2 * INTER, HID, "shared gate_up_proj");
  P.sw13_s = scale_blk(L, t[11], 2 * INTER / QB, HID / QB, "shared gate_up scale");
  P.sw2 = w_nk(t[12], HID, INTER, "shared down_proj");
  P.sw2_s = scale_blk(L, t[13], HID / QB, INTER / QB, "shared down scale");
  if (attn) {
    P.qkv_w = w_nk(t[14], QKV_ROWS, HID, "qkv_proj");
    P.qkv_s = scale_blk(L, t[15], QKV_ROWS / QB, HID / QB, "qkv scale");
    P.q_norm_w = t16(t[16], HD, "q_norm_w", dt);
    P.k_norm_w = t16(t[17], HD, "k_norm_w", dt);
  } else {
    P.qkvz_w = w_nk(t[14], QKVZ, HID, "in_proj_qkvz");
    P.qkvz_s = scale_blk(L, t[15], QKVZ / QB, HID / QB, "qkvz scale");
    P.ba_w = t16(t[16], (int64_t)2 * NVH * HID, "in_proj_ba", dt);
    P.ba_h = half_copy(L, t[16], P.ba_w);
    P.conv_w = t16(t[17], (int64_t)CONVD * 4, "conv1d", dt);
    TORCH_CHECK(t[18].scalar_type() == at::kFloat && t[18].numel() == NVH, "mono: A_log must be fp32 [16]");
    P.A_log = t[18].data_ptr<float>();
    P.dt_bias = t16(t[19], NVH, "dt_bias", dt);
    P.gnorm_w = t16(t[20], DV, "norm_w", dt);
  }
  sycl::queue& q = c10::xpu::getCurrentXPUStream(L.device).queue();
  auto it = g_layers.find(layer);
  if (it != g_layers.end() && it->second.cnt_mem) {  // re-registration: keep the counters (mailbox tags)
    L.cnt_mem = it->second.cnt_mem;
    SharedScratch& sh = shared_scratch(L.device, q);
    L.scratch = sh.scratch;
    L.scratch_b = sh.scratch_b;
    if (it->second.caches_bound && it->second.attn == attn) {  // and the cache binding
      Ptrs& O = it->second.P;
      P.kc = O.kc;
      P.vc = O.vc;
      P.ps = O.ps;
      P.conv_cache = O.conv_cache;
      P.ssm_cache = O.ssm_cache;
      P.conv_stride = O.conv_stride;
      P.ssm_stride = O.ssm_stride;
      P.conv_ch = O.conv_ch;
      P.inter_ssm = O.inter_ssm;
      P.inter_ssm_stride = O.inter_ssm_stride;
      P.inter_ssm_tstride = O.inter_ssm_tstride;
      P.inter_conv = O.inter_conv;
      P.inter_conv_stride = O.inter_conv_stride;
      L.inter_tokens = it->second.inter_tokens;
      L.inter_rows = it->second.inter_rows;
      L.caches_bound = true;
    }
  } else {
    SharedScratch& sh = shared_scratch(L.device, q);
    L.scratch = sh.scratch;
    L.scratch_b = sh.scratch_b;
    L.cnt_mem = sycl::malloc_device(cnt_bytes(), q);
    TORCH_CHECK(L.cnt_mem, "mono: device allocation of the layer counters failed");
    q.memset(L.cnt_mem, 0, cnt_bytes()).wait();
  }
  L.SB = scratch_b_carve(L.scratch_b);
  if (attn) {
    void*& pa = g_attn_part[L.device];
    if (!pa) pa = sycl::malloc_device(attn_part_bytes(), q);
    L.SB.pa_o = static_cast<sycl::half*>(pa);
    L.SB.pa_ml = reinterpret_cast<float*>(static_cast<char*>(pa) + (size_t)APASS * APAIRS * NQ * HD * 2);
  }
  L.S = scratch_carve(L.scratch, reinterpret_cast<uint64_t*>(mbox), reinterpret_cast<uint64_t*>(peer));
  L.S.cnt = static_cast<uint32_t*>(L.cnt_mem);  // the layer's own counters, not the shared scratch's
  set_health(q, L);
  g_layers[layer] = std::move(L);
}

// Registers the MTP drafter prologue as kernel `layer` (its own scratch and
// mailbox region): [fc [2048, 4096], pre_fc_norm_embedding [2048],
// pre_fc_norm_hidden [2048]], all in the model's 16-bit dtype; `rank` (0 / 1)
// picks this rank's half of the fc output rows.
void mono_register_mtp(int64_t layer, int64_t mbox, int64_t peer, const std::vector<at::Tensor>& t, int64_t rank) {
  using namespace sgl_mono;
  TORCH_CHECK(t.size() == 3u, "mono: the MTP prologue takes [fc, pre_fc_norm_embedding, pre_fc_norm_hidden]");
  TORCH_CHECK(t[0].is_xpu(), "mono: mono_register_mtp: the weights must be XPU tensors, got ", t[0].device());
  c10::DeviceGuard guard(t[0].device());
  check_device("mono_register_mtp", t[0].get_device());
  for (const at::Tensor& w : t)
    check_on(w, t[0].get_device(), "mono_register_mtp", "a weight");
  TORCH_CHECK(rank == 0 || rank == 1, "mono: rank must be 0 or 1, got ", rank);
  TORCH_CHECK(
      t[0].dim() == 2 && t[0].size(0) == HID && t[0].size(1) == MTP_FC_IN && t[0].is_contiguous(),
      "mono: MTP fc must be contiguous [2048, 4096] (the full weight on each rank), got ",
      t[0].sizes());
  Layer L;
  L.mtp = true;
  L.rank = (int)rank;
  L.device = t[0].get_device();
  const at::ScalarType dt = L.dtype = dtype16(t[0], "mtp fc");
  L.P.fc_w = t16(t[0], (int64_t)HID * MTP_FC_IN, "mtp fc", dt);
  L.P.in_norm_w = t16(t[1], HID, "pre_fc_norm_embedding", dt);
  L.P.post_norm_w = t16(t[2], HID, "pre_fc_norm_hidden", dt);
  sycl::queue& q = c10::xpu::getCurrentXPUStream(L.device).queue();
  auto it = g_layers.find(layer);
  if (it != g_layers.end() && it->second.scratch) {  // re-registration: keep the scratch (mailbox tags)
    L.scratch = it->second.scratch;
  } else {
    L.scratch = sycl::malloc_device(scratch_bytes(), q);
    q.memset(L.scratch, 0, scratch_bytes()).wait();
  }
  L.S = scratch_carve(L.scratch, reinterpret_cast<uint64_t*>(mbox), reinterpret_cast<uint64_t*>(peer));
  set_health(q, L);
  g_layers[layer] = std::move(L);
}

// Binds layer `layer`'s caches (SGLang pools, used in place; checked): an
// attention layer [k_cache, v_cache] ([slots, (1,) 256] each, the layer's
// dtype) with `page_size` tokens a page; a GatedDeltaNet layer [conv_state
// [slots, 4096, >= 3], ssm_state [slots, 16, 128, 128] fp32]. Only updates
// pointers (no allocation, no wait): may run again whenever the pools move.
void mono_bind_caches(int64_t layer, const std::vector<at::Tensor>& t, int64_t page_size) {
  using namespace sgl_mono;
  TORCH_CHECK(t.size() == 2u || t.size() == 4u, "mono: bind_caches takes 2 tensors (or 4: with the verify scratch)");
  Layer& L = layer_of(layer, t[0]);
  for (const at::Tensor& c : t)
    check_on(c, L.device, "mono_bind_caches", "a cache");
  TORCH_CHECK(!L.mtp, "mono: kernel ", layer, " is an MTP prologue, it has no caches");
  const at::ScalarType dt = L.dtype;
  Ptrs& P = L.P;
  if (L.attn) {
    for (int i = 0; i < 2; ++i) {
      const at::Tensor& c = t[i];
      TORCH_CHECK(
          c.scalar_type() == dt,
          "mono: ",
          i ? "v" : "k",
          "_cache must have the layer's dtype ",
          dtype_name(dt),
          " (an fp16 / bf16 mix of activations and KV cache is not supported), got ",
          c.scalar_type());
      TORCH_CHECK(
          (c.dim() == 3 && c.size(1) == 1 && c.size(2) == HD && c.stride(0) == HD && c.stride(2) == 1) ||
              (c.dim() == 2 && c.size(1) == HD && c.stride(0) == HD && c.stride(1) == 1),
          "mono: ",
          i ? "v" : "k",
          "_cache must be [slots, 1, 256] contiguous token rows, got ",
          c.sizes(),
          " ",
          c.strides());
      TORCH_CHECK((reinterpret_cast<uintptr_t>(c.data_ptr()) & 63) == 0, "mono: KV buffers must be 64-B aligned");
    }
    TORCH_CHECK(page_size >= 16 && page_size % 16 == 0, "mono: page size must be a multiple of 16, got ", page_size);
    P.kc = reinterpret_cast<e16*>(t[0].data_ptr());
    P.vc = reinterpret_cast<e16*>(t[1].data_ptr());
    P.ps = (int)page_size;
  } else {
    const at::Tensor& cs = t[0];
    const at::Tensor& ss = t[1];
    TORCH_CHECK(
        cs.scalar_type() == dt && cs.dim() == 3 && cs.size(1) == CONVD && cs.size(2) == 3 && cs.stride(2) == 1 &&
            cs.stride(1) == cs.size(2),
        "mono: conv state must be [slots, 4096, 3] in the layer's dtype (channel major, taps innermost; the kernels "
        "keep "
        "the 3 history taps in entries 0..2), got ",
        cs.scalar_type(),
        " ",
        cs.sizes(),
        " ",
        cs.strides());
    TORCH_CHECK(
        (reinterpret_cast<uintptr_t>(cs.data_ptr()) & 15) == 0 && cs.stride(0) % 8 == 0,
        "mono: conv state slots must be 16-B aligned");
    TORCH_CHECK(
        ss.scalar_type() == at::kFloat && ss.dim() == 4 && ss.size(1) == NVH && ss.size(2) == DV && ss.size(3) == DK &&
            ss.stride(3) == 1 && ss.stride(2) == DK && ss.stride(1) == DV * DK,
        "mono: ssm state must be [slots, 16, 128, 128] fp32, got ",
        ss.sizes(),
        " ",
        ss.strides());
    P.conv_cache = reinterpret_cast<e16*>(cs.data_ptr());
    P.conv_stride = cs.stride(0);
    P.conv_ch = (int)cs.size(2);
    P.ssm_cache = ss.data_ptr<float>();
    P.ssm_stride = ss.stride(0);
    L.inter_tokens = 0;
    L.inter_rows = 0;
    if (t.size() == 4u) {  // verify scratch: intermediate_ssm, intermediate_conv_window
      const at::Tensor& is = t[2];
      const at::Tensor& ic = t[3];
      const int64_t tok = is.dim() == 5 ? is.size(1) : 0;
      TORCH_CHECK(
          is.scalar_type() == at::kFloat && is.dim() == 5 && is.size(2) == NVH && is.size(3) == DV &&
              is.size(4) == DK && is.stride(4) == 1 && is.stride(3) == DK && is.stride(2) == DV * DK &&
              is.stride(1) == NVH * DV * DK,
          "mono: intermediate ssm state must be [rows, T, 16, 128, 128] fp32, got ",
          is.sizes(),
          " ",
          is.strides());
      // SGLang's windows: a [rows][4096][T + 2] buffer viewed as [rows][T][4096][3]
      TORCH_CHECK(
          ic.scalar_type() == dt && ic.dim() == 4 && ic.size(1) == tok && ic.size(2) == CONVD && ic.size(3) == 3 &&
              ic.stride(1) == 1 && ic.stride(3) == 1 && ic.stride(2) == tok + 2,
          "mono: intermediate conv windows must be the [rows, T, 4096, 3] view of a [rows, 4096, T + 2] buffer "
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
      P.inter_idx = nullptr;  // a request's scratch row is its batch position
      L.inter_tokens = (int)tok;
      L.inter_rows = std::min(is.size(0), ic.size(0));
    }
  }
  L.caches_bound = true;
}

// One decode step (one token a request, 1, 2, 3, 4, 6 or 8 rows) of layer `layer` on
// the current XPU stream. hidden / residual / hid_out / res_out: [rows, 2048]
// in the layer's dtype; hid_out = MoE output (all-reduced), res_out = residual
// after the mixer. attention: m0 page table [>= rows, n] int32 (page indices),
// m1 sequence lengths [>= rows] int32 (with the new token), m2 out_cache_loc
// [>= rows] int64; GatedDeltaNet: m0 state slots [>= rows] int32.
void mono_decode_layer(
    int64_t layer,
    const at::Tensor& hidden,
    const at::Tensor& residual,
    at::Tensor& hid_out,
    at::Tensor& res_out,
    const at::Tensor& positions,
    const at::Tensor& m0,
    const std::optional<at::Tensor>& m1,
    const std::optional<at::Tensor>& m2) {
  using namespace sgl_mono;
  Layer& L = layer_of(layer, hidden);
  TORCH_CHECK(!L.mtp, "mono: kernel ", layer, " is an MTP prologue, not a decoder layer");
  TORCH_CHECK(L.caches_bound, "mono: layer ", layer, " has no caches bound");
  check_on(residual, L.device, "mono_decode_layer", "residual");
  check_on(hid_out, L.device, "mono_decode_layer", "hid_out");
  check_on(res_out, L.device, "mono_decode_layer", "res_out");
  check_on(positions, L.device, "mono_decode_layer", "positions");
  check_on(m0, L.device, "mono_decode_layer", "m0");
  check_on(m1, L.device, "mono_decode_layer", "m1");
  check_on(m2, L.device, "mono_decode_layer", "m2");
  const int64_t nb = hidden.dim() == 2 ? hidden.size(0) : 1;
  TORCH_CHECK(
      nb == 1 || nb == 2 || nb == 3 || nb == 4 || nb == 6 || nb == 8,
      "mono: batch of ",
      nb,
      " rows (supported: 1, 2, 3, 4, 6, 8)");
  Ptrs P = L.P;
  P.hs_in = t16(hidden, nb * HID, "hidden", L.dtype);
  P.res_in = t16(residual, nb * HID, "residual", L.dtype);
  P.hid_out = t16(hid_out, nb * HID, "hid_out", L.dtype);
  P.res_out = t16(res_out, nb * HID, "res_out", L.dtype);
  const bool bf = L.dtype == at::kBFloat16;
  c10::DeviceGuard guard(c10::Device(c10::kXPU, L.device));
  sycl::queue& q = c10::xpu::getCurrentXPUStream(L.device).queue();
  if (L.attn) {
    TORCH_CHECK(m1.has_value() && m2.has_value(), "mono: attention layer needs seq_lens and out_cache_loc");
    TORCH_CHECK(m0.scalar_type() == at::kInt && m0.stride(-1) == 1, "mono: page table must be int32 (unit stride)");
    TORCH_CHECK(m0.dim() == 2 && m0.size(0) >= nb, "mono: page table must be [>= rows, n]");
    check_vec(*m1, at::kInt, nb, "seq_lens");
    check_vec(*m2, at::kLong, nb, "out_cache_loc");
    check_positions(positions, nb);
    P.block_table = m0.data_ptr<int>();
    P.bt_stride = (int)m0.stride(0);
    P.seq_lens = m1->data_ptr<int>();
    P.slot = m2->data_ptr<int64_t>();
    P.pos = positions.data_ptr<int64_t>();  // [T] or mrope [3, T]: row 0
    if (bf)
      launch_attn<bf16>(q, (int)nb, P, L.S, L.SB);
    else
      launch_attn<sycl::half>(q, (int)nb, P, L.S, L.SB);
  } else {
    check_vec(m0, at::kInt, nb, "state indices");
    P.state_idx = m0.data_ptr<int>();
    if (bf)
      launch_gdn<bf16>(q, (int)nb, P, L.S, L.SB);
    else
      launch_gdn<sycl::half>(q, (int)nb, P, L.S, L.SB);
  }
}

// MTP drafter prologue of kernel `layer` (mono_register_mtp): out = fc([
// norm_e(embed); norm_h(hidden)]) for rows embed / hidden [rows, 2048] (the
// model's dtype). False (nothing launched) for a row count without a kernel.
bool mono_mtp_prologue(int64_t layer, const at::Tensor& embed, const at::Tensor& hidden, at::Tensor& out) {
  using namespace sgl_mono;
  Layer& L = layer_of(layer, embed);
  TORCH_CHECK(L.mtp, "mono: kernel ", layer, " is not an MTP prologue");
  check_on(hidden, L.device, "mono_mtp_prologue", "hidden");
  check_on(out, L.device, "mono_mtp_prologue", "out");
  const int64_t nb = embed.dim() == 2 ? embed.size(0) : 1;
  TORCH_CHECK(hidden.size(0) >= nb && out.size(0) >= nb, "mono: MTP hidden / out smaller than the batch");
  Ptrs P = L.P;
  P.hs_in = t16(embed, nb * HID, "embed", L.dtype);
  P.res_in = t16(hidden, nb * HID, "hidden", L.dtype);
  P.hid_out = t16(out, nb * HID, "out", L.dtype);
  c10::DeviceGuard guard(c10::Device(c10::kXPU, L.device));
  sycl::queue& q = c10::xpu::getCurrentXPUStream(L.device).queue();
  return L.dtype == at::kBFloat16 ? launch_mtp<bf16>(q, (int)nb, P, L.S, L.rank)
                                  : launch_mtp<sycl::half>(q, (int)nb, P, L.S, L.rank);
}

// Speculative-decoding verify step of layer `layer`: hidden holds requests x
// `tokens` rows (request-major). attention: m0 page table [>= requests, n]
// int32, m1 sequence lengths [>= requests] int32 (with the `tokens` new ones),
// m2 out_cache_loc [rows] int64; GatedDeltaNet: m0 state slots [>= requests]
// int32 (read, not written: the state after each token goes to the bound
// verify scratch, row = batch position). Returns false (nothing launched) for a
// shape without a kernel.
bool mono_decode_layer_verify(
    int64_t layer,
    int64_t tokens,
    const at::Tensor& hidden,
    const at::Tensor& residual,
    at::Tensor& hid_out,
    at::Tensor& res_out,
    const at::Tensor& positions,
    const at::Tensor& m0,
    const std::optional<at::Tensor>& m1,
    const std::optional<at::Tensor>& m2) {
  using namespace sgl_mono;
  Layer& L = layer_of(layer, hidden);
  TORCH_CHECK(!L.mtp, "mono: kernel ", layer, " is an MTP prologue, not a decoder layer");
  TORCH_CHECK(L.caches_bound, "mono: layer ", layer, " has no caches bound");
  check_on(residual, L.device, "mono_decode_layer_verify", "residual");
  check_on(hid_out, L.device, "mono_decode_layer_verify", "hid_out");
  check_on(res_out, L.device, "mono_decode_layer_verify", "res_out");
  check_on(positions, L.device, "mono_decode_layer_verify", "positions");
  check_on(m0, L.device, "mono_decode_layer_verify", "m0");
  check_on(m1, L.device, "mono_decode_layer_verify", "m1");
  check_on(m2, L.device, "mono_decode_layer_verify", "m2");
  const int64_t nb = hidden.dim() == 2 ? hidden.size(0) : 1;
  TORCH_CHECK(tokens >= 2 && nb % tokens == 0, "mono: verify needs rows = requests x tokens, got ", nb, " x ", tokens);
  const int64_t nrq = nb / tokens;
  Ptrs P = L.P;
  P.hs_in = t16(hidden, nb * HID, "hidden", L.dtype);
  P.res_in = t16(residual, nb * HID, "residual", L.dtype);
  P.hid_out = t16(hid_out, nb * HID, "hid_out", L.dtype);
  P.res_out = t16(res_out, nb * HID, "res_out", L.dtype);
  const bool bf = L.dtype == at::kBFloat16;
  c10::DeviceGuard guard(c10::Device(c10::kXPU, L.device));
  sycl::queue& q = c10::xpu::getCurrentXPUStream(L.device).queue();
  if (L.attn) {
    TORCH_CHECK(m1.has_value() && m2.has_value(), "mono: attention layer needs seq_lens and out_cache_loc");
    TORCH_CHECK(
        m0.scalar_type() == at::kInt && m0.dim() == 2 && m0.stride(1) == 1 && m0.size(0) >= nrq,
        "mono: page table must be int32 [>= requests, n]");
    check_vec(*m1, at::kInt, nrq, "seq_lens (one a request)");
    check_vec(*m2, at::kLong, nb, "out_cache_loc (one a row)");
    check_positions(positions, nb);
    P.block_table = m0.data_ptr<int>();
    P.bt_stride = (int)m0.stride(0);
    P.seq_lens = m1->data_ptr<int>();
    P.slot = m2->data_ptr<int64_t>();
    P.pos = positions.data_ptr<int64_t>();
    return bf ? launch_attn_verify<bf16>(q, (int)nb, (int)tokens, P, L.S, L.SB)
              : launch_attn_verify<sycl::half>(q, (int)nb, (int)tokens, P, L.S, L.SB);
  }
  TORCH_CHECK(L.inter_tokens >= tokens, "mono: layer ", layer, " has no verify scratch for ", tokens, " tokens bound");
  TORCH_CHECK(
      nrq <= L.inter_rows,
      "mono: layer ",
      layer,
      "'s verify scratch has ",
      L.inter_rows,
      " rows, the step ",
      nrq,
      " requests");
  check_vec(m0, at::kInt, nrq, "state indices (one a request)");
  P.state_idx = m0.data_ptr<int>();
  return bf ? launch_gdn_verify<bf16>(q, (int)nb, (int)tokens, P, L.S, L.SB)
            : launch_gdn_verify<sycl::half>(q, (int)nb, (int)tokens, P, L.S, L.SB);
}

// Test / debug hook: layer `layer`'s counter words (int32 view, device
// memory): the stage arrival counters, the epoch and the mailbox tags.
at::Tensor mono_layer_counters(int64_t layer) {
  auto it = g_layers.find(layer);
  TORCH_CHECK(it != g_layers.end(), "mono: layer ", layer, " not registered");
  return torch::from_blob(
      it->second.S.cnt,
      {sgl_mono::CNT_WORDS},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kXPU, it->second.device));
}

// Test hook (fault injection): shifts layer `layer`'s mailbox tags on this rank
// by `delta` (queued on the current stream), so the peer never sees the words
// it waits for: both ranks give up their waits and record the failure.
void mono_fault_inject(int64_t layer, int64_t delta) {
  using namespace sgl_mono;
  auto it = g_layers.find(layer);
  TORCH_CHECK(it != g_layers.end(), "mono: layer ", layer, " not registered");
  uint32_t* cnt = it->second.S.cnt;
  const uint32_t d = (uint32_t)delta;
  sycl::queue& q = c10::xpu::getCurrentXPUStream(it->second.device).queue();
  q.single_task<class MonoFaultInjectK>([=] {
    for (int i = 0; i < 4; ++i)
      cnt[MBT + i] += d;  // rows 1, 2, 4, 8
    for (int i = 0; i < 2; ++i)
      cnt[MBT2 + i] += d;  // rows 3, 6
    for (int i = 0; i < 6; ++i)
      cnt[MTPT + i] += d;  // the MTP prologue's row counts (unused words in a decoder layer)
  });
}
