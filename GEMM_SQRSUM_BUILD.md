# GEMM + Square Sum Kernel - Build Integration

This document describes how the `gemm_sqrsum` kernel has been integrated into the build system.

## Overview

The kernel computes two operations in one pass:
1. **GEMM**: `C = A @ B`
2. **Row-wise square sum**: `sqrsum[i] = sum(A[i,:]^2)`

## Files Created/Modified

### New Files

#### Kernel Implementation
- `src/sycl/kernels/gemm_sqrsum/collective/xe_gemm_sqrsum_mainloop.hpp` - Core mainloop logic
- `src/sycl/kernels/gemm_sqrsum/kernel/xe_gemm_sqrsum_kernel.hpp` - Kernel wrapper
- `src/sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_types.hpp` - Type definitions and runner
- `src/sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_dispatch.hpp` - Dispatch declarations
- `src/sycl/kernels/gemm_sqrsum/device/gemm_sqrsum_example.hpp` - Example usage (reference)
- `src/sycl/kernels/gemm_sqrsum/README.md` - Kernel documentation

#### PyTorch Integration
- `src/sycl/gemm_sqrsum.cpp` - PyTorch binding and dispatch
- `src/sycl/gemm_sqrsum_kernel.cpp.in` - Template for generated instantiations

#### Build System
- `src/GemmSqrSumXe20.cmake` - CMake configuration for kernel generation

#### Testing
- `test_gemm_sqrsum.py` - Python test script
- `src/sycl/kernels/gemm_sqrsum/test_gemm_sqrsum.cpp` - C++ test skeleton

### Modified Files
- `src/CMakeLists.txt` - Added `include(GemmSqrSumXe20.cmake)`
- `src/torch_extension_sycl.cc` - Added PyTorch binding registration

## Build Instructions

```bash
cd /home/gta/frameworks.ai.pytorch.sgl-kernel-xpu/sgl-kernel-xpu

# Clean previous build (if needed)
rm -rf build

# Configure with CMake
mkdir -p build && cd build
cmake ..

# Build
make -j$(nproc)

# The kernel will be compiled into the sgl_kernel PyTorch extension
```

## Generated Files

During build, CMake will generate:
- `build/src/sycl/gemm_sqrsum_kernel_half_256x256x16.cpp`
- `build/src/sycl/gemm_sqrsum_kernel_bf16_256x256x16.cpp`
- `build/src/sycl/gemm_sqrsum_kernel_float_256x256x16.cpp`

Each file instantiates the kernel template for a specific element type.

## Usage from Python

```python
import torch
import sgl_kernel

# Create inputs
M, K, N = 1024, 512, 2048
A = torch.randn(M, K, dtype=torch.float16, device="xpu")
B = torch.randn(K, N, dtype=torch.float16, device="xpu")

# Allocate outputs
C = torch.empty(M, N, dtype=torch.float16, device="xpu")
sqrsum = torch.empty(M, dtype=torch.float16, device="xpu")

# Run kernel
sgl_kernel.gemm_sqrsum(C, sqrsum, A, B)

# C now contains A @ B
# sqrsum[i] contains sum(A[i,:]^2) for each row i
```

## Testing

```bash
# Run Python test
cd /home/gta/frameworks.ai.pytorch.sgl-kernel-xpu/sgl-kernel-xpu
python test_gemm_sqrsum.py
```

## Current Status

### ✓ Completed
- File structure and organization
- CMake build system integration
- PyTorch binding registration
- Basic mainloop structure
- Test scaffolding

### ⚠️ Known Issues / TODOs

1. **Kernel Implementation**
   - Currently uses PyTorch fallback (`at::matmul`) instead of custom CUTLASS kernel
   - Need to complete CUTLASS/CUTE MMA atom configuration
   - Need to implement proper epilogue for writing outputs
   - Fragment reduction logic needs verification with actual CUTE types

2. **Square Sum Computation**
   - Current implementation uses `reduce<2>()` which may not match the actual fragment layout
   - Needs testing with real CUTE tensor partitioning
   - May need cross-subgroup reduction for correctness

3. **Tile Configuration**
   - Currently hardcoded to 256x256x16
   - Should add more tile size options
   - Need to tune for specific hardware (BMG)

4. **Epilogue**
   - No epilogue implementation yet
   - Outputs are not actually written to global memory in the kernel
   - Need to add proper epilogue collective similar to MLA kernel

5. **Testing**
   - C++ test is skeleton only
   - Need comprehensive correctness tests
   - Need performance benchmarks

## Next Steps to Complete

### Priority 1: Get it compiling and running with fallback
1. ✓ Build system integration (done)
2. ✓ PyTorch binding (done)
3. Test that it compiles
4. Test Python binding with fallback implementation
5. Verify fallback produces correct results

### Priority 2: Implement actual CUTLASS kernel
1. Configure proper MMA atom for XPU (DPAS)
2. Complete TiledMMA configuration
3. Implement epilogue for writing C and sqrsum
4. Test with simple small matrices
5. Debug any fragment layout issues

### Priority 3: Optimize
1. Add multiple tile sizes
2. Tune for specific problem sizes
3. Add prefetching hints
4. Optimize cross-subgroup reductions
5. Benchmark against naive implementation

## File Structure Summary

```
sgl-kernel-xpu/
├── src/
│   ├── CMakeLists.txt                              [MODIFIED]
│   ├── GemmSqrSumXe20.cmake                        [NEW]
│   ├── torch_extension_sycl.cc                     [MODIFIED]
│   └── sycl/
│       ├── gemm_sqrsum.cpp                         [NEW]
│       ├── gemm_sqrsum_kernel.cpp.in               [NEW]
│       └── kernels/
│           └── gemm_sqrsum/
│               ├── README.md                       [NEW]
│               ├── test_gemm_sqrsum.cpp            [NEW]
│               ├── collective/
│               │   └── xe_gemm_sqrsum_mainloop.hpp [NEW]
│               ├── kernel/
│               │   └── xe_gemm_sqrsum_kernel.hpp   [NEW]
│               └── device/
│                   ├── gemm_sqrsum_types.hpp       [NEW]
│                   ├── gemm_sqrsum_dispatch.hpp    [NEW]
│                   └── gemm_sqrsum_example.hpp     [NEW]
└── test_gemm_sqrsum.py                             [NEW]
```

## Troubleshooting

### Build Errors

**Missing includes:**
- Ensure CUTLASS and CUTE are in the include path
- Check that `#include <cute/tensor.hpp>` resolves correctly

**Template errors:**
- The mainloop uses placeholder types (void) that will cause compile errors
- Replace with proper DPAS MMA atoms when implementing the full kernel

**Linking errors:**
- Ensure `gemm_sqrsum.cpp` is in the source list
- Check that generated kernel files are being compiled

### Runtime Errors

**"Operation gemm_sqrsum not found":**
- Extension not built/installed correctly
- Try rebuilding with `pip install -e .`

**Wrong results:**
- Currently expected - using fallback implementation
- Check test tolerance settings

**Device errors:**
- Ensure XPU device is available
- Check `torch.xpu.is_available()`

## References

- MLA kernel: `src/sycl/mla_decode.cpp` and related files
- CUTLASS GEMM: `src/sycl/HcPreGemm.cpp`
- Build patterns: `src/MlaDecodeXe20.cmake`
