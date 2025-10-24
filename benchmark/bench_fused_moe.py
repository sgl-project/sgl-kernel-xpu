"""
Benchmark for fused MoE kernels on Intel XPU
"""

import argparse
import time
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

try:
    from sgl_kernel.moe import fused_moe_forward, silu_and_mul_moe
    SGL_KERNEL_AVAILABLE = True
except ImportError:
    SGL_KERNEL_AVAILABLE = False
    print("Warning: sgl_kernel not available. Only running baseline benchmarks.")


def create_test_data(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create test data for MoE benchmarking."""
    
    data = {}
    
    # Input hidden states
    data["hidden_states"] = torch.randn(
        num_tokens, hidden_size, dtype=dtype, device=device
    )
    
    # Expert weights
    data["gate_weights"] = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=dtype, device=device
    )
    data["up_weights"] = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=dtype, device=device
    )
    data["down_weights"] = torch.randn(
        num_experts, intermediate_size, hidden_size, dtype=dtype, device=device
    )
    
    # Routing information
    data["topk_weights"] = torch.rand(num_tokens, top_k, dtype=torch.float32, device=device)
    data["topk_weights"] = data["topk_weights"] / data["topk_weights"].sum(dim=-1, keepdim=True)
    
    data["topk_indices"] = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    
    return data


def baseline_moe_forward(
    hidden_states: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Reference MoE implementation using standard PyTorch operations."""
    
    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, intermediate_size = gate_weights.shape
    
    # Initialize output
    output = torch.zeros_like(hidden_states)
    
    # Process each token
    for token_idx in range(num_tokens):
        token_hidden = hidden_states[token_idx:token_idx+1]  # [1, hidden_size]
        token_output = torch.zeros_like(token_hidden)
        
        # Process each selected expert for this token
        for k in range(top_k):
            expert_idx = topk_indices[token_idx, k].item()
            expert_weight = topk_weights[token_idx, k].item()
            
            # Gate projection
            gate_proj = torch.matmul(token_hidden, gate_weights[expert_idx])  # [1, intermediate_size]
            
            # Up projection
            up_proj = torch.matmul(token_hidden, up_weights[expert_idx])  # [1, intermediate_size]
            
            # SiLU activation and gating
            gate_activated = gate_proj * torch.sigmoid(gate_proj)  # SiLU
            intermediate = gate_activated * up_proj
            
            # Down projection
            expert_output = torch.matmul(intermediate, down_weights[expert_idx])  # [1, hidden_size]
            
            # Weighted accumulation
            token_output += expert_weight * expert_output
        
        output[token_idx] = token_output
    
    return output


def benchmark_silu_and_mul():
    """Benchmark SiLU and multiply operation."""
    
    print("\n" + "="*60)
    print("Benchmarking SiLU and Multiply")
    print("="*60)
    
    sizes = [
        (1024, 4096),
        (2048, 8192),
        (4096, 11008),  # LLaMA-style
        (8192, 22016),
    ]
    
    device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
    dtype = torch.bfloat16
    num_warmup = 10
    num_iters = 100
    
    for batch_size, intermediate_size in sizes:
        print(f"\nSize: [{batch_size}, {intermediate_size}]")
        
        # Create test data
        gate = torch.randn(batch_size, intermediate_size, dtype=dtype, device=device)
        up = torch.randn(batch_size, intermediate_size, dtype=dtype, device=device)
        
        # Baseline implementation
        gate_baseline = gate.clone()
        up_baseline = up.clone()
        
        # Warmup
        for _ in range(num_warmup):
            result_baseline = gate_baseline * torch.sigmoid(gate_baseline) * up_baseline
        
        # Benchmark baseline
        torch.xpu.synchronize() if device.type == "xpu" else None
        start_time = time.time()
        for _ in range(num_iters):
            result_baseline = gate_baseline * torch.sigmoid(gate_baseline) * up_baseline
        torch.xpu.synchronize() if device.type == "xpu" else None
        baseline_time = (time.time() - start_time) / num_iters * 1000  # ms
        
        print(f"  Baseline: {baseline_time:.3f} ms")
        
        # Fused implementation (if available)
        if SGL_KERNEL_AVAILABLE and device.type == "xpu":
            gate_fused = gate.clone()
            up_fused = up.clone()
            
            # Warmup
            try:
                for _ in range(num_warmup):
                    gate_test = gate_fused.clone()
                    silu_and_mul_moe(gate_test, up_fused)
                
                # Benchmark fused
                torch.xpu.synchronize()
                start_time = time.time()
                for _ in range(num_iters):
                    gate_test = gate_fused.clone()
                    silu_and_mul_moe(gate_test, up_fused)
                torch.xpu.synchronize()
                fused_time = (time.time() - start_time) / num_iters * 1000  # ms
                
                print(f"  Fused:    {fused_time:.3f} ms")
                print(f"  Speedup:  {baseline_time / fused_time:.2f}x")
                
            except Exception as e:
                print(f"  Fused:    Failed ({e})")
        else:
            print(f"  Fused:    Not available")


def benchmark_full_moe():
    """Benchmark full MoE forward pass."""
    
    print("\n" + "="*60)
    print("Benchmarking Full MoE Forward")
    print("="*60)
    
    configs = [
        # (num_tokens, hidden_size, intermediate_size, num_experts, top_k)
        (512, 1024, 4096, 8, 2),      # Small
        (1024, 2048, 8192, 16, 2),    # Medium
        (2048, 4096, 11008, 32, 2),   # Large (LLaMA-style)
    ]
    
    device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
    dtype = torch.bfloat16
    num_warmup = 5
    num_iters = 20
    
    for num_tokens, hidden_size, intermediate_size, num_experts, top_k in configs:
        print(f"\nConfig: tokens={num_tokens}, hidden={hidden_size}, intermediate={intermediate_size}, experts={num_experts}, top_k={top_k}")
        
        # Create test data
        data = create_test_data(
            num_tokens, hidden_size, intermediate_size, num_experts, top_k, dtype, device
        )
        
        # Baseline implementation
        # Warmup
        for _ in range(num_warmup):
            result_baseline = baseline_moe_forward(
                data["hidden_states"],
                data["gate_weights"],
                data["up_weights"],
                data["down_weights"],
                data["topk_weights"],
                data["topk_indices"],
                top_k,
            )
        
        # Benchmark baseline
        torch.xpu.synchronize() if device.type == "xpu" else None
        start_time = time.time()
        for _ in range(num_iters):
            result_baseline = baseline_moe_forward(
                data["hidden_states"],
                data["gate_weights"],
                data["up_weights"],
                data["down_weights"],
                data["topk_weights"],
                data["topk_indices"],
                top_k,
            )
        torch.xpu.synchronize() if device.type == "xpu" else None
        baseline_time = (time.time() - start_time) / num_iters * 1000  # ms
        
        print(f"  Baseline: {baseline_time:.3f} ms")
        
        # Fused implementation (if available)
        if SGL_KERNEL_AVAILABLE and device.type == "xpu":
            # Warmup
            try:
                for _ in range(num_warmup):
                    result_fused = fused_moe_forward(
                        data["hidden_states"],
                        data["gate_weights"],
                        data["up_weights"],
                        data["down_weights"],
                        data["topk_weights"],
                        data["topk_indices"],
                        top_k,
                        renormalize=True,
                    )
                
                # Benchmark fused
                torch.xpu.synchronize()
                start_time = time.time()
                for _ in range(num_iters):
                    result_fused = fused_moe_forward(
                        data["hidden_states"],
                        data["gate_weights"],
                        data["up_weights"],
                        data["down_weights"],
                        data["topk_weights"],
                        data["topk_indices"],
                        top_k,
                        renormalize=True,
                    )
                torch.xpu.synchronize()
                fused_time = (time.time() - start_time) / num_iters * 1000  # ms
                
                print(f"  Fused:    {fused_time:.3f} ms")
                print(f"  Speedup:  {baseline_time / fused_time:.2f}x")
                
                # Check correctness (loose tolerance due to different computation order)
                max_diff = torch.max(torch.abs(result_baseline - result_fused)).item()
                rel_diff = max_diff / torch.max(torch.abs(result_baseline)).item()
                print(f"  Max relative diff: {rel_diff:.6f}")
                
            except Exception as e:
                print(f"  Fused:    Failed ({e})")
        else:
            print(f"  Fused:    Not available")


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused MoE kernels")
    parser.add_argument("--test", choices=["silu", "moe", "all"], default="all",
                       help="Which test to run")
    parser.add_argument("--device", choices=["xpu", "cpu"], default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Running benchmarks on {device}")
    print(f"SGL Kernel available: {SGL_KERNEL_AVAILABLE}")
    
    if device.type == "xpu":
        print(f"XPU device: {torch.xpu.get_device_name()}")
        print(f"XPU memory: {torch.xpu.get_device_properties().total_memory / 1e9:.1f} GB")
    
    # Run benchmarks
    if args.test in ["silu", "all"]:
        benchmark_silu_and_mul()
    
    if args.test in ["moe", "all"]:
        benchmark_full_moe()
    
    print("\nBenchmarking completed!")


if __name__ == "__main__":
    main()