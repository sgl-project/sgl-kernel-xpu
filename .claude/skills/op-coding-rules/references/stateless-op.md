# Op coding rules (OP-1 … OP-6, TLA-1)

OP-7 (precision) is in [precision.md](precision.md).

The result of an op depends only on its input arguments. Graph replay does
not run host code: every host-side decision is frozen at capture time, and
every device address used at capture time is reused at replay.

## OP-1: no host-device sync

Why: the host stalls until the device drains, so the CPU cannot run ahead
and queue the next launches; the whole workload slows, decode most. Inside
graph capture, a sync fails or records the wrong thing.

- The host never waits for the device. Disallowed in op code:
  - Python: `Tensor.item()`, `.cpu()`, `.tolist()`, `.numpy()`,
    `torch.xpu.synchronize()`;
  - C++: `at::Tensor::item<T>()`, `.cpu()`, `.to(at::kCPU)`, and any D2H
    copy whose result the host reads;
  - SYCL: `queue::wait()`, `queue::wait_and_throw()`, `event::wait()`, and
    `sycl::free()` (it waits for device work before releasing; raw
    allocation is OP-5).
- The host never reads or writes device tensor data; shape, stride, dtype,
  device and `data_ptr()` are fine.
- Exception: an API whose documented purpose is a host transfer (e.g. a KV
  cache host↔device copy op) or host-side planning, called outside the
  captured region. State it in the docstring.

## OP-2: graph-capture compatible

Why: graph replay does not run host code. Grid size and kernel choice stay
as they were at capture even when per-call data changes, and work on a
queue that never joins the current one is not recorded.

- Grid size and kernel selection use only host-known information: tensor
  shapes, scalar arguments, and CPU tensors passed in by the user. Per-call
  values on the device (`cu_seqlens`, token counts, routing results) are read
  by the kernel: size the grid from a host-known upper bound (`max_seqlen`)
  and let work-groups exit early.
- Submit all work to the user's current queue
  (`at::xpu::getCurrentXPUStream().queue()`). Never use a default/global
  queue (`compat::get_default_queue()`, `dpct::get_default_queue()`, a
  `compat::launch` without a queue argument).
- A side queue must come from the torch stream pool
  (`c10::xpu::getStreamFromPool()`), never from a `sycl::queue` constructed
  in op code, and must join back to the current queue through event
  dependencies so capture records it.

```cpp
// ALLOWED: grid from a host-known upper bound
auto& queue = at::xpu::getCurrentXPUStream().queue();
const int grid_m = ceil_div(max_seqlen_q, kTileM);
launch<Kernel>(grid(grid_m, num_heads, batch), queue, params_with_cu_seqlens);
// kernel: reads cu_seqlens, early-exits tiles beyond the real length

// ALLOWED: pool side queue joined to the current queue both ways
sycl::queue& main_q = at::xpu::getCurrentXPUStream().queue();
c10::xpu::XPUStream side = c10::xpu::getStreamFromPool();
sycl::queue& side_q = side.queue();
side_q.ext_oneapi_submit_barrier({main_q.ext_oneapi_submit_barrier()});
const sycl::event side_done = launch_aux_kernel(side_q, aux_params);
launch_main_kernel(main_q, main_params);
main_q.ext_oneapi_submit_barrier({side_done});

// DISALLOWED
auto host_cu = cu_seqlens.cpu();                     // D2H + sync (OP-1)
int max_len = /* max over host_cu */;
launch<Kernel>(grid(ceil_div(max_len, kTileM), ...)); // frozen at capture
int splits = (seqused_k.max().item<int>() > 4096) ? 8 : 1; // sync (OP-1)
sycl::queue side_q(ctx, dev);                        // not from the pool
```

## OP-3: SYCL-TLA API layer

| Layer | Typical signals | Status | Why |
|---|---|---|---|
| CuTe | `cute::` tensors, layouts, `TiledMma`, `TiledCopy`, own mainloop | **Preferred** | Full control for complex algorithms and fusions |
| Collective | `cutlass::gemm::collective::`, `cutlass::epilogue::collective::`, `CollectiveBuilder` | Allowed | No need to write mainloop/epilogue from scratch; fixed flow, so a poor fit for complex fusions |
| Kernel | `cutlass::gemm::kernel::GemmUniversal`, `cutlass::*::kernel::` launched as the whole kernel | Allowed | Fewer tuning knobs; mainloop/pipeline hidden |
| Device | `cutlass::gemm::device::`, `GemmUniversalAdapter`, `can_implement` + `get_workspace_size` + `initialize` + `run` | **Discouraged** for new code: give a reason in the PR | Host overhead (visible in decode); hidden workspace state (e.g. StreamK flags need a memset before each launch); a default compat queue breaks OP-2 |

If a Device adapter is used, it must run on the current queue, and its
workspace must come from `at::empty` (OP-5). Launching a kernel functor
yourself with `compat::experimental::launch<...>(policy, queue, params)` on
the current queue is fine; calls that omit the queue use the global compat
queue and break OP-2.

## OP-4: plain layout unless declared

Why: a blocked input makes every caller add a packing kernel per call (an
extra launch plus a full read and write), which often costs more than the
kernel gains. Weights do not change in inference, so packing them once at
load time is cheap.

- All input and output tensors use plain layout: standard strided torch
  tensors, contiguous or with the strides the interface documents.
- A blocked or packed layout is allowed only when the interface declares it
  in the parameter name and the documentation (or mirrors an upstream CUDA
  `sgl-kernel` op that declares the same layout).
- Packed weights come from a dedicated prepack API that frameworks call
  once at weight-load time. The packed format is documented.
- Per-call activations, scales and outputs are never required to be in a
  blocked or packed layout. If the kernel needs a blocked layout for them,
  transform inside the op (in the kernel, or fused).

```python
# ALLOWED: declared prepack API, called once at load time
w_packed = sgl_kernel.prepack_foo_weight(w)
out = sgl_kernel.foo_gemm(a, w_packed)       # interface declares B as prepacked

# DISALLOWED: caller must pack activations per call
a_packed = pack_activation_to_kernel_layout(a)
out = foo_gemm(a_packed, w, a_scale)
```

## OP-5: global memory through the torch allocator

Why: `sycl::free` waits for the device (OP-1). Raw allocations are not in
`torch.xpu.memory_stats()`, so SGLang plans memory (e.g. the KV cache budget)
wrong and can run out of memory while serving. Raw pointers also leak on
error paths; torch tensors do not.

- Outputs are either passed by the user or allocated with torch APIs
  (`at::empty`, `at::zeros`, `torch.empty`) and returned.
- Workspaces (split-KV partials, LSE, semaphores, counters) are allocated
  with torch APIs per call. If a buffer must start zeroed, use `at::zeros`
  or a memset kernel on the current queue.
- An op may take an optional workspace (`std::optional<at::Tensor>`); when
  it is omitted, allocate with torch APIs.
- Raw device allocation is disallowed in op code: `sycl::malloc_device`,
  `sycl::malloc_shared`, `sycl::malloc_host`, `sycl::aligned_alloc_*`,
  `sycl::free`, `compat::malloc`, `cutlass::DeviceAllocation`,
  `cutlass::device_memory::allocate`, or any other wrapper on top of
  `sycl::malloc_*`, including fallbacks used when the caller passes no
  workspace.

```cpp
// ALLOWED
at::Tensor out = out_.has_value() ? *out_ : at::empty_like(q);
auto o_accum   = at::empty({b, num_splits, s_q, h_q, d_v}, q.options().dtype(at::kFloat));
auto lse_accum = at::empty({b, num_splits, s_q, h_q},      q.options().dtype(at::kFloat));
launch_splitkv(q, kv, o_accum, lse_accum, queue);
launch_reduce(o_accum, lse_accum, out, queue);

// DISALLOWED
void* ws = sycl::malloc_device(bytes, queue);   // raw USM
launch(kernel, ws); queue.wait(); sycl::free(ws, queue);
void* ws = user_ws ? user_ws : sycl::malloc_device(bytes, queue);  // fallback
cutlass::DeviceAllocation<int> buf; buf.reset(n); buf.copy_from_host(h);
```

## OP-6: no persistent device tensors

Why: a captured graph keeps the device addresses it saw at capture, so
reallocating a cached buffer leaves the graph reading freed memory. A stored
`data_ptr()` dangles once the user frees the tensor or the allocator reuses
it.

- Device tensors created by an op must not survive the call in `static`,
  `thread_local`, global, singleton, or long-lived member storage:
  workspaces, constant tables (`arange`, LUTs) and scratch alike.
- After the call returns, the op holds no `at::Tensor` and no `data_ptr()`
  to the device memory of any tensor, user-passed or op-created.
- A constant or persistent tensor the op needs is an explicit argument (a
  buffer registered on the caller's `torch.nn.Module`, a prepacked weight).
- Allowed: thread-safe host-side metadata caches keyed by device/shape that
  do not change results (device properties, tuned kernel config).
- Library state (e.g. communication buffers for custom all-reduce) is
  allowed only when the caller creates it through an explicit API and
  passes it to the op, so the caller owns its lifetime.

```cpp
// ALLOWED
auto ws = at::empty(ws_shape, opts);                 // back to caching allocator
static std::unordered_map<Key, KernelConfig> config_cache;  // host metadata, mutex-guarded

// DISALLOWED
thread_local at::Tensor cached_ws;                   // cached workspace
if (!cached_ws.defined() || cached_ws.sizes() != ws_shape)
  cached_ws = at::empty(ws_shape, opts);             // frees memory a graph uses
static std::map<int, at::Tensor> const_cache;        // persistent device constant
static void* last_out = out.data_ptr();              // dangles after user frees out
```

## TLA-1: sycl-tla public headers only

- Include only headers from the sycl-tla `include/` tree (`cute/...`,
  `cutlass/...`) and `tools/util/include/` (`cutlass/util/...`), which are
  the include paths `CMakeLists.txt` adds.
- Do not include or copy-paste kernel code from sycl-tla `applications/` or
  `examples/` as a dependency. If a kernel from there is needed, port it into
  `src/sycl/` on top of the public API, so a sycl-tla bump does not silently
  change or break it.
