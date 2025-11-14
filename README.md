# SGL Kernel for XPU

A fork of [Kernel Library](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) for SGLang support on Intel GPU backend

[![PyPI](https://img.shields.io/pypi/v/sgl-kernel)](https://pypi.org/project/sgl-kernel)

## Installation

Currently we only support building from source. To use on Intel GPUs, you need to install the Intel GPUs driver first. For installation guide, visit [Intel GPUs Driver Installation](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-8.html#driver-installation).

## Build from source

Development build:

```bash
source /PATH/TO/ONEAPI/setvars.sh
pip install -v .
```


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
