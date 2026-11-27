// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include <array>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "arguments.hpp"
#include "sgl_kernel_export.h"
#include "shape.hpp"

namespace sgl_kernel::minicpm5::xe20 {
using at::Tensor;
using Bf16 = sycl::ext::oneapi::bfloat16;
using Fp16 = sycl::half;
using Weights = WeightPointers<Bf16>;
using Workspace = WorkspacePointers<Bf16>;
using Step = StepPointers<Bf16>;

#define DECLARE_LAUNCHER(NAME, SCALAR) \
  namespace NAME {                     \
  SGL_KERNEL_EXPORT void dispatch(     \
      sycl::queue&,                    \
      WeightPointers<SCALAR>,          \
      WorkspacePointers<SCALAR>,       \
      StepPointers<SCALAR>,            \
      int rows,                        \
      int tp,                          \
      bool dense);                     \
  }
DECLARE_LAUNCHER(b70_fp16, Fp16)
DECLARE_LAUNCHER(b70_bf16, Bf16)
DECLARE_LAUNCHER(b60_fp16, Fp16)
DECLARE_LAUNCHER(b60_bf16, Bf16)
DECLARE_LAUNCHER(b70_fp16_split, Fp16)
DECLARE_LAUNCHER(b70_bf16_split, Bf16)
DECLARE_LAUNCHER(b60_fp16_split, Fp16)
DECLARE_LAUNCHER(b60_bf16_split, Bf16)
#undef DECLARE_LAUNCHER

namespace b70_draft_bf16 {
SGL_KERNEL_EXPORT void dispatch(sycl::queue&, DraftWeights, Workspace, Step, int tp);
}

namespace {
constexpr int H = Shape::hidden, D = Shape::head_dim, E = Shape::experts, Top = Shape::topk;
constexpr int R = Shape::max_rows;

struct Layer {
  Weights weights;
  int device;
  int tp;
  at::ScalarType dtype;
  bool dense;
  bool draft = false;
  DraftWeights draft_weights{};
};
static_assert(std::is_trivially_copyable_v<Layer>);

bool is_dtype(at::ScalarType dtype) {
  return dtype == at::kHalf || dtype == at::kBFloat16;
}

void check(const Tensor& tensor, int device, at::ScalarType dtype, at::IntArrayRef shape, const char* name) {
  TORCH_CHECK(
      tensor.is_xpu() && tensor.get_device() == device && tensor.scalar_type() == dtype && tensor.sizes() == shape &&
          tensor.is_contiguous(),
      "MiniCPM5 mono invalid ",
      name,
      ": expected contiguous ",
      dtype,
      " ",
      shape,
      " on xpu:",
      device);
}

const uint8_t* nk(const Tensor& t, int n, int k, int device) {
  TORCH_CHECK(
      t.is_xpu() && t.get_device() == device && t.scalar_type() == at::kFloat8_e4m3fn &&
          t.sizes() == at::IntArrayRef({k, n}) && t.strides() == at::IntArrayRef({1, k}) &&
          reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0,
      "MiniCPM5 mono needs an aligned FP8 [K,N] view of contiguous [N,K] storage");
  return static_cast<const uint8_t*>(t.const_data_ptr());
}

bool check_expert_weight(const Tensor& t, int n, int k, int device) {
  TORCH_CHECK(
      t.is_xpu() && t.get_device() == device && t.scalar_type() == at::kFloat8_e4m3fn &&
          t.sizes() == at::IntArrayRef({E, n, k}),
      "MiniCPM5 mono expects FP8 expert weights shaped [E,N,K] on the layer device");
  const bool k_major = !t.is_contiguous();
  TORCH_CHECK(
      !k_major || t.strides() == at::IntArrayRef({int64_t(n) * k, 1, n}),
      "MiniCPM5 mono expert weights must be contiguous [E,N,K] or a [E,N,K] view of contiguous [E,K,N]");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 64 == 0, "MiniCPM5 mono expert arrays must be 64-byte aligned");
  return k_major;
}

const float* fp32_scale(const Tensor& t, int n, int device) {
  TORCH_CHECK(t.is_contiguous(), "MiniCPM5 mono requires contiguous scales");
  check(t.view({-1}), device, at::kFloat, {n}, "scale");
  return t.const_data_ptr<float>();
}

ProjectionScale projection_scale(const Tensor& t, int n, int device) {
  const bool scalar = t.numel() == 1;
  return {fp32_scale(t, scalar ? 1 : n, device), scalar ? 0 : 1};
}

Bf16* ptr(const Tensor& t) {
  return reinterpret_cast<Bf16*>(t.data_ptr());
}

uint32_t* words(const Tensor& t) {
  return reinterpret_cast<uint32_t*>(t.data_ptr<int32_t>());
}

uint64_t* mailbox(const Tensor& t) {
  return reinterpret_cast<uint64_t*>(t.data_ptr<int64_t>());
}

int geometry(int device) {
  if (device < 0 || device >= c10::xpu::device_count()) return 0;
  auto& queue = c10::xpu::getCurrentXPUStream(device).queue();
  if (!queue.is_in_order() || queue.get_backend() != sycl::backend::ext_oneapi_level_zero) return 0;
  const auto gpu = queue.get_device();
  using sycl::ext::oneapi::experimental::architecture;
  const auto arch = gpu.get_info<sycl::ext::oneapi::experimental::info::device::architecture>();
  if (arch != architecture::intel_gpu_bmg_g21 && arch != architecture::intel_gpu_bmg_g31) return 0;
  if (gpu.get_info<sycl::info::device::max_work_group_size>() < B70::work_group_size) return 0;
  const auto slm = gpu.get_info<sycl::info::device::local_mem_size>();
  const auto cores = gpu.get_info<sycl::ext::intel::info::device::gpu_slices>() *
                     gpu.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  if (cores >= 32 && slm >= 128 * 1024) return B70::work_groups;
  return cores >= 20 && slm >= 64 * 1024 ? B60::work_groups : 0;
}

Tensor serialize_layer(const Layer& layer) {
  auto result = at::empty({sizeof(Layer)}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  std::memcpy(result.data_ptr(), &layer, sizeof(layer));
  return result;
}

Tensor layer_desc(const std::vector<Tensor>& t, int64_t tp, bool dense) {
  TORCH_CHECK((tp == 1 || tp == 2) && t.size() == (dense ? 10 : 16), "Invalid MiniCPM5 layer weights or TP");
  const auto device = t[0].get_device();
  const auto dtype = t[0].scalar_type();
  const int groups = geometry(device);
  TORCH_CHECK(is_dtype(dtype) && groups, "MiniCPM5 mono requires BF16/FP16 and a supported Xe2 GPU");
  check(t[0], device, dtype, {H}, "input norm");
  check(t[1], device, dtype, {H}, "post-attention norm");
  const int qdim = Shape::query_heads * D / tp;
  const int qkv = 2 * (qdim + Shape::kv_heads * D / tp);
  Weights w{};
  w.input_norm = ptr(t[0]);
  w.post_norm = ptr(t[1]);
  w.qkv = nk(t[2], qkv, H, device);
  w.qkv_scale = projection_scale(t[3], qkv, device);
  w.out = nk(t[4], H, qdim, device);
  w.out_scale = projection_scale(t[5], H, device);
  if (dense) {
    const int inter = Shape::dense_intermediate / tp;
    w.w13 = nk(t[6], 2 * inter, H, device);
    w.dense13_scale = projection_scale(t[7], 2 * inter, device);
    w.w2 = nk(t[8], H, inter, device);
    w.dense2_scale = projection_scale(t[9], H, device);
  } else {
    const int inter = Shape::expert_intermediate / tp;
    check(t[6], device, dtype, {E, H}, "router");
    check(t[7], device, at::kFloat, {E}, "router bias");
    w.expert_k_major = check_expert_weight(t[8], 2 * inter, H, device);
    const bool down_k_major = check_expert_weight(t[10], H, inter, device);
    TORCH_CHECK(w.expert_k_major == down_k_major, "MiniCPM5 mono expert weights must share one layout");
    TORCH_CHECK(
        !w.expert_k_major || (dtype == at::kBFloat16 && groups == B70::work_groups),
        "MiniCPM5 K-major experts require the 32-work-group BF16 kernel");
    w.router = ptr(t[6]);
    w.bias = t[7].const_data_ptr<float>();
    w.w13 = static_cast<const uint8_t*>(t[8].const_data_ptr());
    w.w13_scale = fp32_scale(t[9], E, device);
    w.w2 = static_cast<const uint8_t*>(t[10].const_data_ptr());
    w.w2_scale = fp32_scale(t[11], E, device);
    w.shared13 = nk(t[12], 2 * inter, H, device);
    w.shared13_scale = projection_scale(t[13], 2 * inter, device);
    w.shared2 = nk(t[14], H, inter, device);
    w.shared2_scale = projection_scale(t[15], H, device);
  }
  return serialize_layer({w, device, int(tp), dtype, dense});
}

Tensor draft_layer_desc(const std::vector<Tensor>& tensors, int64_t tp) {
  const bool fp8 = tensors.size() == 12;
  TORCH_CHECK((tp == 1 || tp == 2) && (fp8 || tensors.size() == 8), "Invalid MiniCPM5 DSpark draft weights or TP");
  const int device = tensors[0].get_device();
  TORCH_CHECK(geometry(device) == B70::work_groups, "MiniCPM5 DSpark draft requires the 32-work-group grid");
  constexpr auto dtype = at::kBFloat16;
  const int qdim = Shape::query_heads * D / tp;
  const int qkv = (Shape::query_heads + 2 * Shape::kv_heads) * D / tp;
  const int intermediate = Shape::dense_intermediate / tp;
  check(tensors[0], device, dtype, {H}, "draft input norm");
  check(tensors[1], device, dtype, {H}, "draft post-attention norm");
  check(tensors[2], device, dtype, {D}, "draft query norm");
  check(tensors[3], device, dtype, {D}, "draft key norm");
  const auto projection = [&](int index, int n, int k, const char* name) -> DraftProjection {
    const int i = 4 + index * (fp8 ? 2 : 1);
    const auto& weight = tensors[i];
    if (fp8) {
      const auto* data = nk(weight, n, k, device);
      TORCH_CHECK(reinterpret_cast<uintptr_t>(data) % 64 == 0, "Draft FP8 projections must be aligned");
      return {data, projection_scale(tensors[i + 1], n, device)};
    }
    check(weight, device, dtype, {n, k}, name);
    const auto* data = static_cast<const uint8_t*>(weight.const_data_ptr());
    TORCH_CHECK(reinterpret_cast<uintptr_t>(data) % 64 == 0, "Draft BF16 projections must be aligned");
    return {data, {}};
  };
  return serialize_layer(
      {{},
       device,
       int(tp),
       dtype,
       true,
       true,
       {ptr(tensors[0]),
        ptr(tensors[1]),
        ptr(tensors[2]),
        ptr(tensors[3]),
        projection(0, qkv, H, "draft QKV projection"),
        projection(1, H, qdim, "draft output projection"),
        projection(2, 2 * intermediate, H, "draft gate-up projection"),
        projection(3, H, intermediate, "draft down projection")}});
}

Workspace workspace_of(const std::vector<Tensor>& w, int device, at::ScalarType dtype, int tp) {
  TORCH_CHECK(w.size() == 10, "MiniCPM5 mono requires 10 scratch buffers");
  const int q = Shape::query_heads / tp, inter = Shape::expert_intermediate / tp;
  check(w[0], device, dtype, {R, H}, "norm scratch");
  check(w[1], device, dtype, {R, 2 * D * (q + Shape::kv_heads / tp)}, "QKV scratch");
  check(w[2], device, dtype, {R, q * D}, "attention scratch");
  check(w[3], device, at::kFloat, {R, 32 * tp, q, D + 2}, "attention partial scratch");
  check(w[4], device, dtype, {R, (Top + 1) * inter}, "intermediate scratch");
  check(w[5], device, at::kFloat, {R, E}, "router scratch");
  check(w[6], device, at::kFloat, {R, Top}, "routing scratch");
  check(w[7], device, at::kInt, {R, Top}, "expert index scratch");
  check(w[8], device, at::kFloat, {R, Top, 2 * inter, 8}, "gate/up partial scratch");
  check(w[9], device, dtype, {R, Top + 1, H}, "expert output scratch");
  return {
      ptr(w[0]),
      ptr(w[1]),
      ptr(w[2]),
      ptr(w[4]),
      ptr(w[9]),
      w[3].data_ptr<float>(),
      w[5].data_ptr<float>(),
      w[6].data_ptr<float>(),
      w[8].data_ptr<float>(),
      w[7].data_ptr<int>()};
}

void decode_step(
    int64_t work_groups,
    int64_t tp,
    int64_t l0,
    int64_t l1,
    const Tensor& table,
    const Tensor& hidden,
    const std::optional<Tensor>& residual,
    const Tensor& positions,
    const Tensor& slots,
    const Tensor& blocks,
    const Tensor& lengths,
    const Tensor& rope,
    const std::vector<Tensor>& keys,
    const std::vector<Tensor>& values,
    const std::vector<Tensor>& scratch,
    Tensor counters,
    Tensor health,
    Tensor host_health,
    const std::optional<Tensor>& local_mailbox,
    const std::optional<Tensor>& peer_mailbox,
    Tensor output,
    Tensor res_out,
    int64_t part_id,
    const std::optional<Tensor>& ar) {
  TORCH_CHECK(hidden.dim() == 2 && (tp == 1 || tp == 2), "Invalid MiniCPM5 mono input or TP");
  const auto rows = hidden.size(0);
  const auto device = hidden.get_device();
  const auto dtype = hidden.scalar_type();
  const auto supported = geometry(device);
  TORCH_CHECK(
      is_dtype(dtype) && supported && (work_groups == supported || work_groups == B60::work_groups),
      "MiniCPM5 mono grid does not fit this Xe2 GPU");
  TORCH_CHECK(rows >= 1 && rows <= R, "MiniCPM5 mono requires 1–8 rows");
  TORCH_CHECK(
      table.device().is_cpu() && table.scalar_type() == at::kByte && table.is_contiguous() && table.dim() == 2 &&
          table.size(1) == sizeof(Layer) && 0 <= l0 && l0 < l1 && l1 <= table.size(0) && keys.size() == table.size(0) &&
          values.size() == keys.size(),
      "Invalid MiniCPM5 mono layer table or bounds");
  TORCH_CHECK(0 <= part_id && part_id <= static_cast<int>(Part::FeedForwardPartial), "Invalid MiniCPM5 launch part");
  const auto part = static_cast<Part>(part_id);
  const bool split = part != Part::Full;
  const bool external = part == Part::AttentionPartial || part == Part::FeedForwardPartial;
  const bool use_mailbox = tp == 2 && !external;
  TORCH_CHECK(!split || l1 == l0 + 1, "Reference launch parts require a single layer");
  TORCH_CHECK(ar.has_value() == external, "Reference partial launch requires an FP32 all-reduce buffer");
  if (ar) check(*ar, device, at::kFloat, {rows, H}, "external all-reduce buffer");
  check(hidden, device, dtype, {rows, H}, "input");
  if (residual) check(*residual, device, dtype, {rows, H}, "residual");
  check(output, device, dtype, {2, rows, H}, "output buffers");
  check(res_out, device, dtype, {2, rows, H}, "residual output buffers");
  check(positions, device, at::kLong, {rows}, "positions");
  check(slots, device, at::kLong, {rows}, "cache slots");
  check(lengths, device, at::kInt, {lengths.numel()}, "sequence lengths");
  TORCH_CHECK(
      blocks.dim() == 2 && blocks.size(1) > 0 && blocks.size(1) <= MaxContext, "Invalid MiniCPM5 mono block table");
  TORCH_CHECK(blocks.size(0) == rows || blocks.size(0) == 1, "Wrong number of MiniCPM5 block-table rows");
  check(blocks, device, at::kInt, blocks.sizes(), "block table");
  TORCH_CHECK(rope.dim() == 2 && rope.size(0) > 0, "Invalid MiniCPM5 mono RoPE table");
  check(rope, device, dtype, {rope.size(0), D}, "RoPE table");
  const auto control_banks = tp == 2 ? Shape::tp2_control_banks : 1;
  check(counters, device, at::kInt, {table.size(0), control_banks, Shape::control_words}, "counters");
  check(health, device, at::kInt, {1}, "device health");
  TORCH_CHECK(
      host_health.device().is_cpu() && host_health.is_pinned() && host_health.is_contiguous() &&
          host_health.scalar_type() == at::kInt && host_health.numel() == 1,
      "Expected pinned MiniCPM5 health");
  if (use_mailbox) {
    TORCH_CHECK(local_mailbox && peer_mailbox, "MiniCPM5 TP2 requires symmetric mailboxes");
    const std::array<int64_t, 2> size{table.size(0), Shape::mailbox_words};
    check(*local_mailbox, device, at::kLong, size, "local mailbox");
    TORCH_CHECK(
        peer_mailbox->is_xpu() && peer_mailbox->scalar_type() == at::kLong && peer_mailbox->sizes() == size &&
            peer_mailbox->is_contiguous(),
        "Invalid MiniCPM5 peer mailbox");
  }
  // Each mailbox word is rewritten once per epoch in its bank. A different
  // row width or launch part can therefore never age a word across tag wrap.
  // Fused launches use both phases. Split launches have separate phase
  // storage, with an independent epoch for each of their two launch parts.
  const auto control_bank = use_mailbox ? 1 + part_id * R + (rows - 1) : 0;
  const auto mailbox_offset = (rows * (rows - 1) / 2 + (split ? Shape::mailbox_rows : 0)) * H;
  auto work = workspace_of(scratch, device, dtype, tp);
  auto& queue = c10::xpu::getCurrentXPUStream(device).queue();
  std::vector<Layer> layers;
  layers.reserve(l1 - l0);
  for (int64_t index = l0; index < l1; ++index) {
    Layer layer;
    std::memcpy(&layer, table.const_data_ptr<uint8_t>() + index * sizeof(Layer), sizeof(Layer));
    TORCH_CHECK(layer.device == device && layer.tp == tp && layer.dtype == dtype, "Incompatible MiniCPM5 descriptor");
    TORCH_CHECK(lengths.size(0) == (layer.draft ? 1 : rows), "Wrong number of MiniCPM5 sequence lengths");
    TORCH_CHECK(
        !layer.weights.expert_k_major || work_groups == B70::work_groups,
        "MiniCPM5 K-major experts require the 32-work-group BF16 kernel");
    TORCH_CHECK(
        !layer.draft || (work_groups == B70::work_groups && !split && rows == Shape::draft_rows && blocks.size(0) == 1),
        "MiniCPM5 DSpark draft requires seven queries of one request and the full BF16 kernel");
    const auto& k = keys[index];
    const auto& v = values[index];
    const int64_t kv_heads = Shape::kv_heads / tp;
    TORCH_CHECK(k.dim() == 4 && k.size(0) > 0 && (k.size(1) == 64 || k.size(1) == 128), "Expected paged MiniCPM5 KV");
    const auto page = k.size(1);
    check(k, device, dtype, {k.size(0), page, kv_heads, D}, "key cache");
    check(v, device, dtype, k.sizes(), "value cache");
    layers.push_back(layer);
  }
  for (int64_t index = l0; index < l1; ++index) {
    const auto& layer = layers[index - l0];
    const auto& k = keys[index];
    const auto& v = values[index];
    const auto page = k.size(1);
    const size_t row = (index - l0) % 2;
    const auto previous = row ^ 1;
    Step step{
        index == l0 ? ptr(hidden) : ptr(output) + previous * rows * H,
        index == l0 ? (residual ? ptr(*residual) : nullptr) : ptr(res_out) + previous * rows * H,
        ptr(output) + row * rows * H,
        ptr(res_out) + row * rows * H,
        ptr(k),
        ptr(v),
        positions.const_data_ptr<int64_t>(),
        slots.const_data_ptr<int64_t>(),
        blocks.const_data_ptr<int>(),
        lengths.const_data_ptr<int>(),
        int(page),
        blocks.size(0) == 1 ? 0 : int(blocks.stride(0)),
        k.size(0),
        blocks.size(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        words(counters) + (index * control_banks + control_bank) * Shape::control_words,
        use_mailbox ? mailbox(*local_mailbox) + index * Shape::mailbox_words + mailbox_offset : nullptr,
        use_mailbox ? mailbox(*peer_mailbox) + index * Shape::mailbox_words + mailbox_offset : nullptr,
        words(health),
        ptr(rope),
        rope.size(0),
        part,
        ar ? ar->data_ptr<float>() : nullptr};
    const bool b60 = work_groups == B60::work_groups;
    if (layer.draft) {
      b70_draft_bf16::dispatch(queue, layer.draft_weights, work, step, tp);
    } else if (dtype == at::kHalf) {
      const auto dispatch = split ? (b60 ? b60_fp16_split::dispatch : b70_fp16_split::dispatch)
                                  : (b60 ? b60_fp16::dispatch : b70_fp16::dispatch);
      dispatch(queue, layer.weights.as<Fp16>(), work.as<Fp16>(), step.as<Fp16>(), rows, tp, layer.dense);
    } else {
      const auto dispatch = split ? (b60 ? b60_bf16_split::dispatch : b70_bf16_split::dispatch)
                                  : (b60 ? b60_bf16::dispatch : b70_bf16::dispatch);
      dispatch(queue, layer.weights, work, step, rows, tp, layer.dense);
    }
  }
  queue.memcpy(host_health.data_ptr<int>(), health.const_data_ptr<int>(), sizeof(int));
}
}  // namespace

SGL_KERNEL_EXPORT void register_ops(torch::Library& m) {
  m.def("minicpm5_work_groups(int device) -> int", [](int64_t device) -> int64_t { return geometry(device); });
  m.def("minicpm5_layer_desc(Tensor[] weights, int tp, bool dense) -> Tensor", layer_desc);
  m.def("minicpm5_draft_layer_desc(Tensor[] weights, int tp) -> Tensor", draft_layer_desc);
  m.def(
      "minicpm5_decode_step(int work_groups, int tp, int l0, int l1, Tensor layers, Tensor hidden, Tensor? residual, "
      "Tensor positions, Tensor slots, Tensor blocks, Tensor lengths, Tensor rope, Tensor(a!)[] keys, "
      "Tensor(b!)[] values, Tensor(c!)[] scratch, Tensor(d!) counters, Tensor(e!) health, Tensor(f!) host_health, "
      "Tensor(g!)? local_mailbox, Tensor(h!)? peer_mailbox, Tensor(i!) output, Tensor(j!) res_out, "
      "int part=0, Tensor(k!)? ar=None) -> ()",
      decode_step);
}
}  // namespace sgl_kernel::minicpm5::xe20
