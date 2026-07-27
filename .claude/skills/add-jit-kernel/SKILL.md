---
name: add-jit-kernel
description: Step-by-step tutorial for adding a new lightweight SYCL JIT kernel for Intel XPU to sgl-kernel-xpu's python/sgl_kernel/jit module (SYCL source + Python wrapper + tests + benchmark)
---

# Tutorial: Adding a New SYCL/XPU JIT Kernel to sgl-kernel-xpu

This tutorial walks through adding a new runtime-compiled SYCL kernel to the
`sgl-kernel-xpu` repository. JIT kernels are compiled on-demand with Intel's
`icpx` compiler (via `load_jit_sycl`), cached to disk and in-memory, and exposed
to Python through a thin `ctypes` wrapper.

We'll use a simple element-wise `scale(x, factor) = x * factor` operation to
demonstrate the complete workflow.

## Goal

Add a new SYCL kernel that:

1. Lives as a header in `include/sgl_kernel/jit_kernel/`
2. Is JIT-compiled and cached through `python/sgl_kernel/jit/compiler.py::load_jit_sycl`
3. Is exposed via a Python wrapper in `python/sgl_kernel/jit/`
4. Has accuracy tests in `tests/test_jit_kernels.py`
5. (Optionally) has an AOT-vs-JIT benchmark in `benchmark/`

---

## Repository Layout for JIT Kernels

| Concern | Location |
|---------|----------|
| SYCL kernel headers | `include/sgl_kernel/jit_kernel/` (subdirs: `elementwise/`, `diffusion/`, plus shared `memory.hpp`) |
| JIT compiler / loader | `python/sgl_kernel/jit/compiler.py` (`load_jit_sycl`, `SYCLModule`, module cache) |
| Cache decorator | `python/sgl_kernel/jit/utils.py` (`cache_once`) |
| Python kernel wrappers | `python/sgl_kernel/jit/norm.py`, `rope.py`, `timestep_embedding.py` |
| Public exports | `python/sgl_kernel/jit/__init__.py` (guarded by `is_xpu()`) |
| Tests | `tests/test_jit_kernels.py` |
| Benchmarks | `benchmark/bench_jit_*.py` |

`sycl_files` passed to `load_jit_sycl` are resolved **relative to
`include/sgl_kernel/jit_kernel/`**.

---

## Reference Implementations — Consult These FIRST

Before writing a new kernel, **read the closest existing implementation** to
reuse proven layouts, math, and thread mappings. Optimized SYCL is much easier
to port from an existing kernel than to write from scratch.

### 1. Existing JIT SYCL headers (primary reference — copy their structure)

| Kernel | Header | Techniques to reuse |
|--------|--------|---------------------|
| RMSNorm | `include/sgl_kernel/jit_kernel/elementwise/rmsnorm.hpp` | `aligned_vector` vectorization, `reduce_over_group`, `[[sycl::reqd_sub_group_size]]`, vectorized+fallback kernel selection, float32 accumulation |
| QKNorm | `include/sgl_kernel/jit_kernel/elementwise/qknorm.hpp` | per-head normalization, compile-time `head_dim` |
| RoPE | `include/sgl_kernel/jit_kernel/elementwise/rope.hpp` | neox/interleave specialization, strided head access, fused KV-cache store |
| Timestep Embedding | `include/sgl_kernel/jit_kernel/diffusion/timestep_embedding.hpp` | vectorized register compute + scalar tail loop |
| Shared helpers | `include/sgl_kernel/jit_kernel/memory.hpp` | `sgl::sycl::aligned_vector<T, N>` for SIMD loads/stores |

### 2. AOT SYCL sources (production kernels — best optimization reference)

The compiled (ahead-of-time) SYCL kernels in `src/sycl/` are the most
performance-tuned versions. When adding a JIT kernel, **port the math and thread
mapping from the matching AOT source**:

- `src/sycl/RMSNorm.cpp`, `src/sycl/Rope.cpp`, `src/sycl/FusedQKNormRope.cpp`,
  `src/sycl/Norm.h` — element-wise / normalization references.
- `src/sycl/SYCLHelpers.h`, `src/sycl/MemoryAccess.h`, `src/sycl/Utils.h` —
  device query, vectorized access, and reduction helpers used by AOT kernels
  (mirror their strategy, but keep the JIT header self-contained).
- `src/torch_extension_sycl.cc` — the op schema + `torch::kXPU` binding for each
  AOT op (use this to confirm exact op names/signatures when benchmarking).

Comments in the JIT headers already cite this lineage (e.g. rmsnorm.hpp:
"matches AOT `reduce_over_group(group)`"). Keep JIT results **numerically close**
to the AOT kernel.

### 3. CUDA references (algorithm reference, not XPU-optimized)

When no SYCL version exists, use the CUDA JIT kernels as the algorithmic
blueprint and translate with the CUDA -> SYCL mapping table below. In this
workspace the upstream `sglang` repo sits next to `sgl-kernel-xpu`:

- CUDA JIT Python wrappers: `sglang/python/sglang/jit_kernel/*.py`.
- CUDA kernel sources: `sglang/sgl-kernel/csrc/**/*.cuh`
  (e.g. `sglang/sgl-kernel/csrc/elementwise/pos_enc.cuh` for RoPE).
- Translate warp-level primitives (`__shfl_*`, `__syncwarp`, 32-lane warps) to
  SYCL sub-group collectives (`reduce_over_group`, `shift_group_left`, pinned
  sub-group size).

**Workflow**: find AOT SYCL first -> else find CUDA `.cuh` -> translate to a
JIT SYCL header using the patterns and mapping in this guide.

---

## Step 1: Write the SYCL Kernel Header

Create `include/sgl_kernel/jit_kernel/elementwise/scale.hpp`.

### Critical SYCL Namespace Rule

**Always use the `::sycl::` prefix for math functions**, on both host and device
code. `<sycl/sycl.hpp>` defines iostream proxies (e.g. `std::clog`) that collide
with `<cmath>` names like `std::log`.

```cpp
// WRONG - may collide with SYCL iostream proxies
float r = std::exp(x) * std::log(y);

// CORRECT
float r = ::sycl::exp(x) * ::sycl::log(y);
```

Affected functions (always prefix): `log`, `exp`, `sqrt`, `rsqrt`, `sin`, `cos`,
`tan`, `pow`, `abs`, `min`, `max`, and any other `<cmath>` function.

### Kernel Structure

Match the conventions used by existing headers
(`elementwise/rmsnorm.hpp`, `elementwise/rope.hpp`, `diffusion/timestep_embedding.hpp`):

- `#pragma once` include guard
- `namespace sgl { namespace sycl_kernel { ... } }`
- A kernel **functor** class with `operator()(::sycl::nd_item<1> item) const`
- A templated **host launcher** that submits to a `::sycl::queue`
- An `extern "C"` **C API** with the SYCL `queue` pointer as the first argument
- One exported symbol per (dtype, compile-time-specialization) combination,
  typically produced with a macro

```cpp
#pragma once

#include <sycl/sycl.hpp>

#include "../memory.hpp"  // shared helpers (e.g. aligned_vector), if needed

namespace sgl {
namespace sycl_kernel {

// --- Kernel functor ---
template <typename T>
class ScaleKernel {
 public:
  ScaleKernel(const T* in_ptr, T* out_ptr, float factor, int64_t numel)
      : in_ptr_(in_ptr), out_ptr_(out_ptr), factor_(factor), numel_(numel) {}

  void operator()(::sycl::nd_item<1> item) const {
    const size_t idx = item.get_global_id(0);
    if (idx >= static_cast<size_t>(numel_)) return;
    out_ptr_[idx] = static_cast<T>(static_cast<float>(in_ptr_[idx]) * factor_);
  }

 private:
  const T* in_ptr_;
  T* out_ptr_;
  float factor_;
  int64_t numel_;
};

// --- Host launcher ---
template <typename T>
void scale_launcher(::sycl::queue& queue, const void* input, void* output, float factor, int64_t numel) {
  if (numel <= 0) return;

  const T* in_ptr = static_cast<const T*>(input);
  T* out_ptr = static_cast<T*>(output);

  constexpr size_t kThreadsPerGroup = 256;
  const size_t num_groups = (numel + kThreadsPerGroup - 1) / kThreadsPerGroup;

  queue.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(
        ::sycl::nd_range<1>(::sycl::range<1>(num_groups * kThreadsPerGroup), ::sycl::range<1>(kThreadsPerGroup)),
        ScaleKernel<T>(in_ptr, out_ptr, factor, numel));
  });
  // NOTE: do NOT call .wait() here. The kernel runs on the current XPU stream,
  // and synchronization is handled on the PyTorch side. Adding .wait() serializes
  // execution and hurts performance.
}

// --- C API for the Python ctypes wrapper ---
#define DEFINE_SCALE_FORWARD(DTYPE_SUFFIX, DTYPE)                                  \
  extern "C" void scale_forward_##DTYPE_SUFFIX(                                    \
      void* queue_ptr, const void* input, void* output, float factor, int64_t numel) { \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);                         \
    scale_launcher<DTYPE>(queue, input, output, factor, numel);                    \
  }

DEFINE_SCALE_FORWARD(fp32, float)
DEFINE_SCALE_FORWARD(fp16, ::sycl::half)
DEFINE_SCALE_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16)

#undef DEFINE_SCALE_FORWARD

}  // namespace sycl_kernel
}  // namespace sgl
```

**Compile-time specialization**: If the kernel needs a compile-time constant
(as `rmsnorm.hpp` uses `SGL_RMSNORM_HIDDEN_SIZE`, `rope.hpp` uses `SGL_ROPE_DIM`),
gate the exported symbol on a `-D` macro and pass it via `extra_sycl_cflags` in
the Python loader. This keeps each compiled `.so` specialized to one config.

### CUDA → SYCL API Mapping

| CUDA | SYCL | Notes |
|------|------|-------|
| `threadIdx.x` | `item.get_local_id(0)` | Thread index within work-group |
| `blockIdx.x` | `item.get_group(0)` | Work-group index |
| `blockDim.x` | `item.get_local_range(0)` | Work-group size |
| `gridDim.x` | `item.get_group_range(0)` | Number of work-groups |
| global index | `item.get_global_id(0)` | Flattened global index |
| `__global__` | functor class with `operator()` | Kernel object |
| `__device__` | regular function | No qualifier needed |
| `__shared__` | `::sycl::local_accessor` | Set up on the handler |
| `__syncthreads()` | `item.barrier()` | Work-group barrier |
| `half` | `::sycl::half` | fp16 |
| `nv_bfloat16` | `::sycl::ext::oneapi::bfloat16` | bf16 |
| `__shfl_*` / warp reduce | `::sycl::reduce_over_group` / `::sycl::shift_group_left` | Sub-group collectives |
| warp (32 lanes) | sub-group (8/16/32 on Intel) | Pin with `[[sycl::reqd_sub_group_size(N)]]` |

---

## XPU Optimization Playbook

These are the techniques that make the difference between a correct kernel and an
**optimized** one. Apply them by default; the existing headers demonstrate each.

### 1. Vectorize memory access with `aligned_vector`

Load/store multiple elements per instruction to maximize bandwidth. Use the
shared `sgl::sycl::aligned_vector<T, N>` from `memory.hpp` and pick `N` from the
contiguous dimension (8 if divisible by 8, else 4, else 2, else 1) — see
`get_vec_size<T, kHiddenSize>()` in rmsnorm.hpp.

```cpp
using Vec = aligned_vector<T, kVecSize>;
const Vec* in_vec = reinterpret_cast<const Vec*>(input_ptr);
Vec v = in_vec[i];              // one vectorized load
#pragma unroll
for (int e = 0; e < kVecSize; ++e) { /* work on v[e] */ }
```

Only reinterpret to `Vec*` when the base pointer and the per-row length are
properly aligned; otherwise use a scalar tail loop for the remainder (see the
`vec_half_dim` tail handling in timestep_embedding.hpp).

### 2. Accumulate reductions in float32

For fp16/bf16 inputs, **always cast to `float` for sums/means/variance** and
store back in the native dtype. This matches the AOT kernels and keeps accuracy
within the `1e-2` test tolerance.

### 3. Use sub-group / work-group collectives, not manual loops

Replace hand-rolled reductions with `::sycl::reduce_over_group(item.get_group(),
val, ::sycl::plus<float>())`. Pin the sub-group size for deterministic codegen:

```cpp
static constexpr int kSubGroupSize = 16;  // Intel supports 8/16/32
[[sycl::reqd_sub_group_size(kSubGroupSize)]]
void operator()(::sycl::nd_item<1> item) const { ... }
```

### 4. Map one work-group per output row/token

The established pattern is `group = item.get_group(0)` -> token/row index, and
threads within the group cooperate over the feature/hidden dimension with a
grid-stride loop `for (i = tid; i < N; i += num_threads)`. This gives coalesced
access and cheap intra-row reductions. `kThreadsPerBlock = 256` is a good default
(cap at the hidden/rope dimension).

### 5. Specialize at compile time

Template on shape constants (`kHiddenSize`, `rope_dim`, `head_dim`, neox flag)
and select them via `-D` macros in `extra_sycl_cflags`. Compile-time constants
let the compiler unroll loops and choose vector widths. Provide a **vectorized
kernel + a fallback kernel** and pick between them with `if constexpr` in the
launcher (see `use_vectorized` in rmsnorm.hpp).

### 6. Unroll hot loops

Add `#pragma unroll` to fixed-trip-count inner loops (over `kVecSize`, small
dims) for better instruction-level parallelism.

### 7. Keep launches asynchronous

Do **not** call `.wait()` in the launcher — submit on the current XPU stream and
let PyTorch handle synchronization (see the pitfalls section).

### 8. Match AOT numerics

Prefer `::sycl::rsqrt`, `::sycl::exp`, `::sycl::cos/sin` (fast device math). The
repo compiles with `-fhonor-nans -fhonor-infinities -fno-associative-math
-no-ftz` (see `DEFAULT_SYCL_CFLAGS`), so results stay close to the AOT/CUDA
reference. Do not reorder floating-point reductions in ways that diverge from
the reference.

---

## Step 2: Add the Python Wrapper

Create `python/sgl_kernel/jit/scale.py`. Follow the pattern in `norm.py` /
`rope.py`: a `@cache_once` module loader plus a wrapper class that resolves the
exported C function by name and calls it with the XPU SYCL queue.

```python
"""XPU/SYCL element-wise scale kernel wrapper."""

from __future__ import annotations

import torch

from .compiler import load_jit_sycl
from .utils import cache_once

_SUPPORTED_SCALE_DTYPES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


@cache_once
def _jit_scale_module_xpu(dtype: torch.dtype):
    """Compile/load the XPU/SYCL scale module for the given dtype."""
    if dtype not in _SUPPORTED_SCALE_DTYPES:
        raise ValueError(
            f"Unsupported dtype for XPU scale: {dtype}. "
            f"Supported: {list(_SUPPORTED_SCALE_DTYPES)}"
        )

    dtype_str = _SUPPORTED_SCALE_DTYPES[dtype]

    # sycl_files are resolved relative to include/sgl_kernel/jit_kernel/
    module = load_jit_sycl(
        "scale",
        dtype_str,
        sycl_files=["elementwise/scale.hpp"],
    )
    return _XPUScaleWrapper(module, dtype_str)


class _XPUScaleWrapper:
    """Wrapper matching a simple in/out kernel API."""

    def __init__(self, module, dtype_str: str):
        import ctypes

        self._module = module
        self._func_name = f"scale_forward_{dtype_str}"
        self._argtypes = [
            ctypes.c_void_p,  # queue
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # output
            ctypes.c_float,  # factor
            ctypes.c_int64,  # numel
        ]

    def scale(self, input: torch.Tensor, output: torch.Tensor, factor: float) -> None:
        # Validate layout assumptions the SYCL kernel relies on.
        if not input.is_contiguous() or not output.is_contiguous():
            raise ValueError("XPU scale requires contiguous input/output tensors")

        # The SYCL queue backing the current XPU stream.
        queue = torch.xpu.current_stream().sycl_queue

        func = self._module.get_function(self._func_name, self._argtypes)
        func(queue, input.data_ptr(), output.data_ptr(), float(factor), input.numel())


def scale(input: torch.Tensor, factor: float) -> torch.Tensor:
    """Element-wise scale with XPU JIT kernel and a PyTorch fallback."""
    output = torch.empty_like(input)

    if hasattr(torch, "xpu") and input.device.type == "xpu":
        module = _jit_scale_module_xpu(input.dtype)
        module.scale(input, output, factor)
        return output

    # PyTorch fallback for non-XPU devices.
    return input * factor
```

Key conventions (verify against `norm.py` / `rope.py`):

- **Cache with `cache_once`** (from `.utils`), not `functools.lru_cache` — it is
  `torch.compile`-friendly.
- **First C argument is always the queue pointer**, obtained via
  `torch.xpu.current_stream().sycl_queue`.
- Resolve the exported function with
  `module.get_function(func_name, argtypes)` where `argtypes` is a list of
  `ctypes` types; `SYCLModule` caches the configured function.
- Encode dtype (and any compile-time specialization) into the **exported symbol
  name** so one `.so` maps to one configuration.
- Validate tensor layout (contiguity, storage offset, shape) in Python before
  calling into SYCL — the kernel assumes these invariants.

### Passing compile-time constants

When a kernel is specialized on a compile-time value, add it to both the
`load_jit_sycl` cache key (as a positional `*args` marker) and the compiler
flags, exactly like `rmsnorm`/`rope`:

```python
module = load_jit_sycl(
    "scale",
    str(some_dim),   # part of the module identity / cache key
    dtype_str,
    sycl_files=["elementwise/scale.hpp"],
    extra_sycl_cflags=[f"-DSGL_SCALE_DIM={some_dim}"],
)
```

---

## Step 3: Export from the JIT Package

Register the new entry point in `python/sgl_kernel/jit/__init__.py` inside the
`if is_xpu():` block so it is only imported on XPU:

```python
if is_xpu():
    ...
    from .scale import scale
    ...
    __all__ = [
        ...
        "scale",
    ]
```

---

## Step 4: Add Accuracy Tests

Add a test to `tests/test_jit_kernels.py`. Follow the existing structure:
skip markers for `HAS_SGLANG_JIT` / `HAS_SGL_KERNEL` / `HAS_XPU`, a PyTorch
reference, and a `torch.testing.assert_close` comparison.

```python
@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires JIT compilation")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_scale_jit_vs_reference():
    from sgl_kernel.jit import scale as jit_scale

    device = "xpu"
    x = torch.randn(1024, dtype=torch.float16, device=device)
    factor = 2.5

    y_ref = x * factor
    y_jit = jit_scale(x.clone(), factor)

    torch.testing.assert_close(y_jit, y_ref, rtol=1e-2, atol=1e-2)
```

**Use loose tolerances** (`rtol=1e-2, atol=1e-2`) for fp16/bf16: XPU math
functions may differ slightly in the last bits from a PyTorch/CUDA reference.

---

## Step 5: (Optional) Add a Benchmark

Mirror the existing `benchmark/bench_jit_*.py` files (e.g. `bench_jit_rope.py`,
`bench_jit_rmsnorm.py`). Important gotchas learned from those benchmarks:

- **Import `sgl_kernel` before touching `torch.ops.sgl_kernel.*`.** The AOT
  operators register lazily on `import sgl_kernel`; without the import, ops like
  `torch.ops.sgl_kernel.rotary_embedding` raise `AttributeError`.
- Confirm the **exact AOT op name and signature** against the compiled binary
  (e.g. `src/torch_extension_sycl.cc` and `src/sycl/*.cpp`). Python wrappers in
  `python/sgl_kernel/` can drift out of sync with the built `common_ops` `.so`.
- Match each kernel's dtype requirements (e.g. AOT `rotary_embedding` uses a
  `cos_sin_cache` in the same dtype as q/k; the JIT RoPE path uses float32).

---

## The JIT Compilation Flow

`load_jit_sycl` (in `python/sgl_kernel/jit/compiler.py`) does the following:

1. **Guards**: requires `torch.xpu.is_available()` and `icpx` on `PATH`
   (`is_icpx_available()`); otherwise raises `RuntimeError`.
2. **Sources**: reads each `sycl_files` entry relative to
   `include/sgl_kernel/jit_kernel/` and generates a small `.cpp` that
   `#include`s them.
3. **Flags**: `DEFAULT_SYCL_CFLAGS` + AOT target flags from
   `_get_sycl_aot_flags()` + any `extra_sycl_cflags`.
   - AOT target auto-detects the device (Xe2 / Battlemage → `intel_gpu_bmg_g21`),
     falling back to generic `spir64`. Override with the
     `SGLANG_SYCL_AOT_TARGETS` environment variable.
4. **Cache key**: a hash of (markers + source filenames + source contents +
   flags + `icpx` version + relevant env vars). The compiled artifact is
   `<module>_<hash>.so` in the cache directory.
5. **Compile**: invokes `icpx`, publishing the `.so` atomically via
   `os.replace` from a temp file. Recompiles only when the cache key changes.
6. **Load & cache**: wraps the `.so` in `SYCLModule` and stores it in an
   in-memory LRU (`_LOADED_MODULES_CACHE`) to avoid repeated `dlopen`.

### Safe module lifecycle

The in-memory `_LRUModuleCache` intentionally **never unloads** a module
(`_close_module` is a no-op). This is deliberate: `@cache_once`-wrapped loaders
hand out long-lived `SYCLModule` objects and bound `ctypes` function pointers,
so `dlclose`-ing the `.so` would dangle those pointers. Eviction only drops the
LRU's own reference; the module survives until Python GC reclaims it. Use
`clear_module_cache()` only in tests or to intentionally free memory.

---

## Common SYCL/XPU Pitfalls

1. **Host-side math namespace**: use `::sycl::exp()` not `std::exp()` (avoids
   iostream-proxy collisions from `<sycl/sycl.hpp>`).
2. **No `.wait()` in launchers**: let PyTorch manage stream synchronization;
   `.wait()` serializes and hurts performance.
3. **Pointer casts**: cast `void*` explicitly with `static_cast<T*>()`.
4. **Include guards**: always `#pragma once` in headers.
5. **`icpx` on PATH**: the JIT path needs Intel oneAPI; `source setvars.sh`
   first. Missing `icpx` raises
   `RuntimeError: icpx compiler not found`.
6. **Loose test tolerances**: prefer `atol=1e-2, rtol=1e-2` for fp16/bf16.
7. **Layout validation in Python**: check contiguity / storage offset / shape
   before calling the kernel; SYCL kernels assume these invariants.
8. **First compile is slow** (tens of seconds); subsequent runs reuse the cached
   `.so`.

---

## Debugging Compilation Failures

`load_jit_sycl` raises a `RuntimeError` containing the full `icpx` command plus
`stdout`/`stderr` when compilation fails:

```python
try:
    module = _jit_scale_module_xpu(torch.float16)
except RuntimeError as e:
    print(e)  # includes the icpx command line and compiler diagnostics
```

To force a clean rebuild, call `clear_module_cache()` (drops the in-memory
cache) and delete the stale `.so` from the JIT cache directory.

---

## Checklist for a New JIT Kernel

- [ ] Consulted the closest reference: AOT SYCL in `src/sycl/`, existing JIT
      header in `include/sgl_kernel/jit_kernel/`, or CUDA `.cuh` for the algorithm.
- [ ] Header in `include/sgl_kernel/jit_kernel/<category>/<name>.hpp`
      (`sgl::sycl_kernel` namespace, `::sycl::` math, `extern "C"` API with queue
      first arg, per-dtype exported symbols).
- [ ] Applied the optimization playbook: `aligned_vector` vectorization, float32
      accumulation, `reduce_over_group` + pinned sub-group size, one work-group
      per row, compile-time specialization, `#pragma unroll`, no `.wait()`.
- [ ] Python wrapper in `python/sgl_kernel/jit/<name>.py` (`@cache_once` loader,
      `ctypes` wrapper, layout validation, PyTorch fallback).
- [ ] Exported from `python/sgl_kernel/jit/__init__.py` under `is_xpu()`.
- [ ] Accuracy test in `tests/test_jit_kernels.py` (loose tolerances), compared
      against the PyTorch reference or the AOT kernel.
- [ ] (Optional) Benchmark in `benchmark/bench_jit_<name>.py`
      (remember `import sgl_kernel` for AOT ops).
- [ ] Verified end-to-end after `source setvars.sh` (so `icpx` is available).
