# SGL Kernel for XPU

sglang-kernel-xpu is the official kernel library of the [SGLang](https://github.com/sgl-project/sglang) framework for Intel XPU, which delivers high-performance, production-ready compute primitives on Intel XPU platforms.

[![PyPI](https://img.shields.io/pypi/v/sgl-kernel)](https://pypi.org/project/sgl-kernel)

## Two-stage ConvRot INT8 (BMG / Xe2)

The standalone ConvRot kernels do not depend on comfy-kitchen and do not change
SGLang model dispatch:

```python
from sgl_kernel import convrot_rotate_quantize, convrot_int8_linear_prequant

# x: contiguous BF16 [M, K]; weight: pre-rotated INT8 [N, K].
q, row_scale = convrot_rotate_quantize(x, group_size=256)
y = convrot_int8_linear_prequant(q, weight, row_scale, weight_scale, bias=None)
```

These inference-only kernels currently support BMG devices, K divisible by 256
in `[256, 8192]`, and positive N divisible by 32. Empty M is supported.
Weights must already use the regular-Hadamard ConvRot basis; ordinary
Walsh-Hadamard weights are not interchangeable. No online weight quantization
or weight transpose/packing is performed.

The first kernel applies normalized regular H256 independently to each group,
rounds the rotation to BF16, then quantizes **the entire row with one scale**.
It returns contiguous INT8 `[M, K]` and FP32 `[M]`. Scale is
`max(absmax * float32(1/127), 1e-30)`; the division scale and quotient are
rounded to BF16 before round-to-even and saturation to `[-128, 127]`.
Inputs must be finite and remain finite through the rotation.

**Precision limitation:** FP32 butterfly summation reorders the additions
relative to a dense Torch matmul. At BF16 rounding boundaries this can change
the quantized values; the first stage is not guaranteed bit-exact with the
eager rotation. Model-quality acceptance is still required before model
integration.

The second kernel uses SYCL-TLA signed INT8 DPAS with INT32 accumulation and
a fused epilogue: `BF16(float32(acc) * (row_scale * weight_scale))`, followed by
optional BF16 bias addition. It writes BF16 `[M, N]` directly, without a full
INT32 output or Torch scaling/casting/concatenation. Scales are contiguous
FP32 `[M]` and `[N]` (or `[N, 1]` for weight scales); optional bias is contiguous
BF16 `[N]`. All tensors must be on the same device. Unsupported metadata raises
an error rather than silently selecting an eager backend.

Run correctness tests and the synthetic H3-shape benchmark after building:

```bash
ZE_AFFINITY_MASK=4,5,6,7 python -m pytest tests/test_convrot_int8.py -q
ZE_AFFINITY_MASK=4,5,6,7 python benchmark/bench_convrot_int8.py \
  --devices 0 1 2 3 --rows 63744 66048 53376 --iterations 5 \
  --output /tmp/convrot-benchmark.json --trace /tmp/convrot.trace.json.gz
```

The benchmark checks the prequantized GEMM independently with exact tensor
equality, and reports end-to-end differences against eager dense rotation.
For its seeded synthetic inputs, it requires equal row scales, quantized
mismatch fraction at most `1e-6`, INT8 difference at most 2, and final-output
relative L2 error at most `1e-4`. These are benchmark checks, not a guarantee
for arbitrary input distributions or a replacement for model-quality tests.
It reports synchronized host latency and incremental peak allocated memory.
Devices are exercised sequentially: this is not a TP4 model benchmark.

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

### Kernel Development

Steps to add a new kernel:

1. Implement the kernel in [csrc](https://github.com/sgl-project/sgl-kernel-xpu/tree/main/src/)
2. Expose the interface in [include/sgl_kernel_ops.h](https://github.com/sgl-project/sgl-kernel-xpu/blob/main/include/sgl_kernel_ops.h)
3. Create torch extension in [csrc/common_extension.cc](https://github.com/sgl-project/sgl-kernel-xpu/blob/main/src/torch_extension_sycl.cc)
4. Update [CMakeLists.txt](https://github.com/sgl-project/sgl-kernel-xpu/blob/main/CMakeLists.txt) to include new source files
5. Expose Python interface in [python](https://github.com/sgl-project/sgl-kernel-xpu/blob/main/python/sgl_kernel)

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
