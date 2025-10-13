# FusedMoE Kernels for Intel XPU

This implementation provides optimized Mixture of Experts (MoE) kernels for Intel XPU devices, leveraging CUTLASS templates and Intel XE Matrix Extensions (XMX) for high-performance inference.

## Overview

The fusedMoE implementation consists of specialized kernels that optimize the complete MoE forward pass, including:

- **Expert Routing**: Efficient TopK selection and token-to-expert assignment
- **Grouped GEMM**: Batched matrix operations for multiple experts
- **Fused Activations**: Combined SiLU activation and element-wise operations
- **Output Reduction**: Weighted aggregation of expert outputs

## Architecture

### Core Components

1. **Main Kernel** (`src/sycl/fused_moe.cpp`)
   - Entry point for fusedMoE operations
   - Parameter validation and device management
   - CUTLASS kernel configuration and dispatch

2. **Specialized Kernels** (`src/sycl/kernels/fused_moe/`)
   - `xe_fused_moe.hpp`: Main MoE execution kernel
   - `xe_moe_gemm.hpp`: Grouped GEMM operations
   - `xe_moe_routing.hpp`: Token routing and TopK selection
   - `xe_moe_reduction.hpp`: Expert output aggregation

3. **Header Declarations** (`include/sgl_moe_kernel_ops.h`)
   - Function signatures for all MoE operations
   - Parameter structures and type definitions

4. **Python Interface** (`python/sgl_kernel/moe.py`)
   - High-level Python API functions
   - Input validation and error handling
   - Automatic fallback mechanisms

## Features

### Performance Optimizations

- **Intel XPU Specific**: Optimized for Intel XE Matrix Extensions (XMX)
- **Memory Efficiency**: Block-aligned access patterns and vectorized operations
- **Fusion**: Eliminates intermediate memory traffic between operations
- **CUTLASS Integration**: Template-based kernel architecture for flexibility

### Data Type Support

- **Primary**: `bfloat16` (recommended for Intel XPU)
- **Secondary**: `float16`, `float32`
- **Mixed Precision**: Automatic accumulation in higher precision for numerical stability

### Flexible Configuration

- Configurable tile sizes (M, N, K dimensions)
- Adjustable pipeline stages for latency/throughput trade-offs
- Support for different expert topk values
- Optional weight renormalization

## API Reference

### Core Functions

#### `fused_moe_forward`

```python
def fused_moe_forward(
    hidden_states: torch.Tensor,        # [num_tokens, hidden_size]
    gate_weights: torch.Tensor,         # [num_experts, hidden_size, intermediate_size]
    up_weights: torch.Tensor,           # [num_experts, hidden_size, intermediate_size]
    down_weights: torch.Tensor,         # [num_experts, intermediate_size, hidden_size]
    topk_weights: torch.Tensor,         # [num_tokens, top_k]
    topk_indices: torch.Tensor,         # [num_tokens, top_k]
    top_k: int = 2,
    renormalize: bool = True,
    inplace: bool = False,
    use_grouped_topk: bool = True,
) -> torch.Tensor:
```

Complete MoE forward pass with all optimizations.

#### `grouped_gemm_moe`

```python
def grouped_gemm_moe(
    A: torch.Tensor,                    # [num_tokens, K]
    B: torch.Tensor,                    # [num_experts, K, N] or [num_experts, N, K]
    C: torch.Tensor,                    # [num_experts, N] (bias)
    D: torch.Tensor,                    # [num_tokens, N] (output)
    topk_weights: torch.Tensor,         # [num_tokens, top_k]
    topk_indices: torch.Tensor,         # [num_tokens, top_k]
    top_k: int,
    trans_b: bool = False,
) -> None:
```

Optimized grouped GEMM for expert processing.

#### `silu_and_mul_moe`

```python
def silu_and_mul_moe(
    gate_output: torch.Tensor,          # [*, intermediate_size] (modified in-place)
    up_output: torch.Tensor,            # [*, intermediate_size]
) -> None:
```

Fused SiLU activation and element-wise multiplication.

#### `moe_align_block_size`

```python
def moe_align_block_size(
    topk_ids: torch.Tensor,             # [num_tokens, top_k]
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,     # Output buffer
    experts_ids: torch.Tensor,          # Output buffer
    num_tokens_post_pad: torch.Tensor,  # Output buffer
) -> None:
```

Align token routing for efficient block processing.

## Usage Examples

### Basic Usage

```python
import torch
from sgl_kernel.moe import fused_moe_forward

# Initialize on Intel XPU
device = torch.device("xpu:0")
dtype = torch.bfloat16

# Create sample data
num_tokens, hidden_size, intermediate_size = 1024, 4096, 11008
num_experts, top_k = 32, 2

hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
gate_weights = torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype, device=device)
up_weights = torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype, device=device)
down_weights = torch.randn(num_experts, intermediate_size, hidden_size, dtype=dtype, device=device)

# Assume we have routing information
topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=device)
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # Normalize
topk_indices = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)

# Run fused MoE
output = fused_moe_forward(
    hidden_states=hidden_states,
    gate_weights=gate_weights,
    up_weights=up_weights,
    down_weights=down_weights,
    topk_weights=topk_weights,
    topk_indices=topk_indices,
    top_k=top_k,
    renormalize=True,
)

print(f"Output shape: {output.shape}")  # [1024, 4096]
```

### Integration with PyTorch Module

```python
import torch.nn as nn
from sgl_kernel.moe import fused_moe_forward

class IntelXPUMoELayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, top_k=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gating network
        self.gate_proj = nn.Linear(hidden_size, num_experts)
        
        # Expert weights
        self.gate_weights = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size))
        self.up_weights = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size))
        self.down_weights = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size))
    
    def forward(self, hidden_states):
        # Compute routing
        gate_logits = self.gate_proj(hidden_states)
        topk_weights, topk_indices = torch.topk(torch.softmax(gate_logits, dim=-1), self.top_k, dim=-1)
        
        # Fused MoE forward
        return fused_moe_forward(
            hidden_states=hidden_states,
            gate_weights=self.gate_weights,
            up_weights=self.up_weights,
            down_weights=self.down_weights,
            topk_weights=topk_weights,
            topk_indices=topk_indices,
            top_k=self.top_k,
        )

# Usage
model = IntelXPUMoELayer(hidden_size=4096, intermediate_size=11008, num_experts=32)
model = model.to("xpu:0").bfloat16()

input_tensor = torch.randn(1024, 4096, dtype=torch.bfloat16, device="xpu:0")
output = model(input_tensor)
```

## Performance Characteristics

### Expected Speedups

Based on the CUTLASS architecture and Intel XPU optimizations:

- **vs PyTorch Baseline**: 2-5x speedup for typical MoE configurations
- **Memory Bandwidth**: ~3x improvement through fusion and optimized access patterns
- **Expert Utilization**: Better load balancing through grouped operations

### Recommended Configurations

#### Small Models (7B parameters)
```python
hidden_size = 4096
intermediate_size = 11008  
num_experts = 8
top_k = 2
dtype = torch.bfloat16
```

#### Medium Models (13B-30B parameters)
```python
hidden_size = 5120
intermediate_size = 13824
num_experts = 16
top_k = 2
dtype = torch.bfloat16
```

#### Large Models (70B+ parameters)
```python
hidden_size = 8192
intermediate_size = 28672
num_experts = 32
top_k = 2
dtype = torch.bfloat16
```

## Testing

### Unit Tests

```bash
# Run MoE-specific tests
python -m pytest tests/test_fused_moe.py -v

# Run with XPU device
python -m pytest tests/test_fused_moe.py::TestFusedMoE::test_basic_fused_moe_forward -v
```

### Benchmarks

```bash
# Run comprehensive benchmarks
python benchmark/bench_fused_moe.py

# Run specific benchmark
python benchmark/bench_fused_moe.py --test moe
python benchmark/bench_fused_moe.py --test silu
```

### Integration Example

```bash
# Run the integration example
python examples/moe_integration_example.py
```

## Build Requirements

### System Dependencies

- Intel oneAPI Base Toolkit (2024.0 or later)
- Intel Extension for PyTorch with XPU support
- CUTLASS library (included in Intel oneAPI)
- Python 3.8+ with PyTorch 2.0+

### Build Configuration

The kernels are automatically included in the build process through CMake glob patterns:

```cmake
file(GLOB device_cpp "sycl/*.cpp" "sycl/*.sycl")
```

Ensure your build environment has:

```bash
export SYCL_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/linux/lib
export CMAKE_PREFIX_PATH=/opt/intel/oneapi/compiler/latest/linux/cmake
```

## Limitations and Known Issues

### Current Limitations

1. **Device Support**: Currently only supports Intel XPU devices
2. **Precision**: Mixed-precision accumulation may affect numerical precision for very deep expert networks
3. **Memory**: Large expert counts may exceed device memory for very large models

### Fallback Behavior

The implementation gracefully falls back to PyTorch baseline operations when:

- Intel XPU is not available
- Unsupported data types are used
- Kernel compilation fails
- Runtime errors occur

### Future Improvements

- [ ] Support for dynamic expert selection
- [ ] Load balancing optimizations
- [ ] Support for expert parallelism across multiple XPU devices
- [ ] Integration with model parallelism frameworks

## Contributing

When contributing to the fusedMoE kernels:

1. **Follow Patterns**: Use the established CUTLASS template patterns
2. **Test Thoroughly**: Include unit tests for new functionality
3. **Document Changes**: Update this README for API changes
4. **Performance**: Include benchmark comparisons for new optimizations

## License

Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0. See the LICENSE file for details.