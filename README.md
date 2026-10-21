# SGL Kernel for XPU

sglang-kernel-xpu is the official kernel library of the [SGLang](https://github.com/sgl-project/sglang) framework for Intel XPU, which delivers high-performance, production-ready compute primitives on Intel XPU platforms.

[![PyPI](https://img.shields.io/pypi/v/sgl-kernel)](https://pypi.org/project/sgl-kernel)

## Installation

Currently we only support building from source. To use on Intel GPUs, you need to install the Intel GPUs driver first. For installation guide, visit [Intel GPUs Driver Installation](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-8.html#driver-installation).

## Build from source

Development build:

```bash
source /PATH/TO/ONEAPI/setvars.sh
pip install -v .
```

### Optional Build Toggles

You can selectively disable feature groups during build with these environment
variables (default is `ON` for all):

| Variable | Default | Set to `OFF`/`0` to disable |
| --- | --- | --- |
| `USE_MOE` | `ON` | MoE group gemm kernels |
| `USE_FMHA` | `ON` | Flash attention kernels (`fwd`) |
| `USE_GDN` | `ON` | Gated DeltaNet attention kernels |
| `USE_MLA` | `ON` | MLA decode/prefill/sparse-decode kernels |
| `USE_QSA` | `ON` | BF16 QSA sparse attention and indexer |
| `USE_MLA_SPARSE_FUSED` | `OFF` | Fused (single-pass) sparse MLA decode kernel (optimization track) |

There is also an opt-in switch (default `OFF`):

| Variable | Default | Set to `ON`/`1` to enable |
| --- | --- | --- |
| `USE_SYCL_JIT` | `OFF` | Runtime-JIT kernels (FMHA/MLA/MoE/GDN compiled on demand instead of AOT) |

Notes:
- `OFF` and `0` are both accepted.
- Values are case-insensitive (`off`, `Off`, `OFF` all work).

Examples:

```bash
# Disable MoE only
USE_MOE=OFF pip install -v .

# Same as above
USE_MOE=0 pip install -v .

# Disable all three
USE_MOE=0 USE_FMHA=OFF USE_MLA=off pip install -v .
```

### Runtime-JIT Kernels (`USE_SYCL_JIT`)

By default all kernels are compiled ahead-of-time (AOT). Setting the
`USE_SYCL_JIT` environment variable (or CMake option) to `ON`/`1` opts into the
runtime-JIT path: the FMHA, MLA, MoE grouped-GEMM and GDN kernels are **not**
instantiated at build time. Instead their `*.cpp.in` templates are compiled on
demand with `icpx` on first use and cached as `.so` files, which shortens the
build and shrinks the wheel.

```bash
source /PATH/TO/ONEAPI/setvars.sh
USE_SYCL_JIT=ON pip install -v .
```

Notes:
- The **first** call into a JIT kernel triggers a synchronous `icpx` compile of
  that configuration; it can take seconds to minutes and may look like a hang.
  Subsequent calls hit the on-disk cache and start immediately.
- Torch include/lib paths are exported automatically at import time. Optional
  runtime overrides: `SGL_JIT_CACHE_DIR` (compiled-`.so` cache dir),
  `SGL_JIT_CUTLASS_INCLUDE` (CUTLASS-SYCL include dirs, needed if the headers
  were not packaged), `SGL_JIT_INCLUDE_ROOT` (override the package include root).


### Build with [ccache](https://github.com/ccache/ccache)
```bash
# or `yum install -y ccache`.
apt-get install -y ccache
# Building with ccache is enabled when ccache is installed and CCACHE_DIR is set.
export CCACHE_DIR=/path/to/your/ccache/dir
export CCACHE_BACKEND=""
export CCACHE_KEEP_LOCAL_STORAGE="TRUE"
unset CCACHE_READONLY
python -m uv build --wheel -Cbuild-dir=build --color=always .
```

### Parallel Build

We highly recommend you build sgl-kernel-xpu with Ninja. Ninja can automatically build sgl-kernel in parallel.
And if you build the sgl-kernel-xpu with cmake, you need to add `CMAKE_BUILD_PARALLEL_LEVEL` for parallel build like:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) python -m uv build --wheel -Cbuild-dir=build --color=always .
```

SYCL device-link parallelism inside each compiler invocation defaults to one
job, independently of Ninja's outer build parallelism. This avoids multiplying
the outer job count by every host CPU during large AOT builds. To override it
explicitly, pass `-DSYCL_MAX_PARALLEL_LINK_JOBS=N` to CMake or
`--config-settings=cmake.define.SYCL_MAX_PARALLEL_LINK_JOBS=N` to `pip`.

### Kernel Development

Steps to add a new kernel:

1. Implement the kernel in [csrc](https://github.com/sgl-project/sgl-kernel-xpu/tree/main/src/)
2. Expose the interface in [include/sgl_kernel_ops.h](https://github.com/sgl-project/sgl-kernel-xpu/blob/main/include/sgl_kernel_ops.h)
3. Create torch extension in [csrc/common_extension.cc](https://github.com/sgl-project/sgl-kernel-xpu/blob/main/src/torch_extension_sycl.cc)
4. Update [CMakeLists.txt](https://github.com/sgl-project/sgl-kernel-xpu/blob/main/CMakeLists.txt) to include new source files
5. Expose Python interface in [python](https://github.com/sgl-project/sgl-kernel-xpu/blob/main/python/sgl_kernel)

### Xe35 Small MXFP4 Permutation

`sgl_kernel.mxfp4_small_permutation_xe35` writes padded expert counts, source
rows, the expanded-to-permuted map, and the signed scale-layout map in one
launch. It requires `DPCPP_SYCL_TARGET=cri` and `USE_MOE=ON`. All outputs are
preallocated contiguous XPU int32 tensors; input expert IDs may be int32 or
int64 and must be in range. The interface supports up to 256 pairs,
power-of-two expert counts through 1024, and alignment from 1 through 64.

```bash
pytest -q tests/test_mxfp4_small_permutation.py
python benchmark/bench_mxfp4_small_permutation.py \
  --baseline-source /path/to/sglang/srt/layers/quantization/mxfp4.py
```

The benchmark uses B8/topk8 and defaults to E256/alignment4. It loads the
unchanged `_get_xpu_w4a4_small_permutation_kernel` factory from the supplied
historical SGLang source, reports its SHA256, checks all four outputs exactly,
and alternates candidate and baseline XPU-event measurements over five rounds.
Both paths exclude output allocation and use the same inputs and capacities.
Each measurement has a synchronized two-second cooldown. A candidate median
above the baseline median fails by default; `--max-regression` explicitly
changes that gate. `--routing` and `--index-dtype` select routing distributions
and paired index dtypes. The serving scope is B8/ISL4096/OSL1024/TP4/EP1; this
is an operator benchmark, not an end-to-end model run.

For comparison against the older sgl-kernel `mxfp4_small_permutation` operator,
use separate Python environments with normally installed complete packages:

```bash
"$OLD_PYTHON" benchmark/bench_mxfp4_small_permutation.py \
  --installed-operator legacy --routing balanced --index-dtype int32 \
  --result-file results/r0-int32-balanced-legacy.json
"$NEW_PYTHON" benchmark/bench_mxfp4_small_permutation.py \
  --installed-operator candidate --routing balanced --index-dtype int32 \
  --result-file results/r0-int32-balanced-candidate.json
python benchmark/compare_mxfp4_small_permutation_packages.py --results-dir results
```

Create the results directory first. Collect rounds 0–4, alternating package
order per round, for both index dtypes and all three routing distributions.
The worker records 201 XPU-profiler kernel durations, package/library provenance,
inputs and all four outputs. The comparison checks exact inputs and outputs
across packages and rounds, then rejects any median kernel-time regression.
Use the same idle GPU and exclusive GPU lock throughout the paired sweep.
Keep each package's compatible PyTorch/SYCL dependencies isolated and report
their versions; profiler kernel durations and event-bracketed operator latency
are different metrics and must not be compared against each other.

### Development Tips

1. When implementing kernels, only define pure SYCL files and C++ interfaces. If you need to use `Torch::tensor`, use `<torch/all.h>` instead of `<torch/extension.h>`. Using `<torch/extension.h>` will cause compilation errors when using SABI.

2. When creating torch extensions, add the function definition with `m.def`, and device binding with `m.impl`:
- Using torch.compile need `m.def` with schema, it helps auto capture the custom kernel. Reference: [How to add FakeTensor](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.ptttacy8y1u9)

- How to write schema: [Schema reference](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func)

### Integrating Third-Party Libraries with Data Type Conversion

When integrating new third-party libraries like flash-attention, you may encounter data type compatibility issues between the C++ interface and PyTorch bindings. For example, the third-party code might use `float` or `int` types, while PyTorch requires `double` and `int64_t`.

> The reason we need `double` and `int64_t` in torch binding is that TORCH_LIBRARY handles the `Python-to-C++` conversion process. Python's `float` data type actually corresponds to `double` in C++, while Python's `int` corresponds to `int64_t` in C++.

To address this issue, we provide the `make_pytorch_shim` function in [sgl_kernel_torch_shim](https://github.com/sgl-project/sgl-kernel-xpu/blob/main/include/sgl_kernel_torch_shim.h) that handles data type conversions automatically.

When you need to support new data type conversions, you can easily add conversion functions like this:

```cpp
// Map `int` -> `int64_t`
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    TORCH_CHECK(arg <= std::numeric_limits<int>::max(), "int64_t value is too large to be converted  to int");
    TORCH_CHECK(arg >= std::numeric_limits<int>::min(), "int64_t value is too small to be converted to int");
    return arg;
  }
};
```

To use this with your library functions, simply wrap them with make_pytorch_shim:

```cpp
/*
 * From flash-attention
 */
 m.impl("fwd", torch::kXPU, make_pytorch_shim(&mha_fwd));
```

### Contributing

We welcome contributions of all kinds!
Please read our [Contributing Guidelines](./CONTRIBUTING.md) before submitting a pull request.

### Testing & Benchmarking

1. Add pytest tests in [tests/](https://github.com/sgl-project/sgl-kernel-xpu/tree/main/tests), if you need to skip some test, please use `@pytest.mark.skipif`

```python
@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
```

2. Add benchmarks using [triton benchmark](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) in [benchmark/](https://github.com/sgl-project/sglang/tree/main/sgl-kernel/benchmark)
3. Run test suite

### Release new version

Update version in [pyproject.toml](https://github.com/sgl-project/sgl-kernel-xpu/blob/main/pyproject.toml) and [version.py](https://github.com/sgl-project/sgl-kernel-xpu/blob/main/python/sgl_kernel/version.py)
