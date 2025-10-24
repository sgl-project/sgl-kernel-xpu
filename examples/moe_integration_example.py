"""
Example of integrating fusedMoE kernel with SGLang for Intel XPU

This example shows how to create a simple MoE backend for SGLang that uses 
the optimized Intel XPU fusedMoE kernels.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import warnings

try:
    from sgl_kernel.moe import fused_moe_forward, silu_and_mul_moe
    SGL_KERNEL_AVAILABLE = True
except ImportError:
    SGL_KERNEL_AVAILABLE = False
    warnings.warn("sgl_kernel not available. Using PyTorch fallback.")


class IntelXPUMoELayer(nn.Module):
    """
    Mixture of Experts layer optimized for Intel XPU using SGLang fused kernels.
    
    This implementation demonstrates how to integrate the fusedMoE kernels
    with a standard PyTorch/SGLang model architecture.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int, 
        num_experts: int,
        top_k: int = 2,
        gate_bias: bool = False,
        expert_bias: bool = False,
        renormalize: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize
        
        # Auto-detect device
        if device is None:
            device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
        self.device = device
        self.dtype = dtype
        
        # Check if Intel XPU optimizations are available
        self.use_fused_kernels = (
            SGL_KERNEL_AVAILABLE and 
            device.type == "xpu" and 
            dtype in [torch.bfloat16, torch.float16, torch.float32]
        )
        
        if not self.use_fused_kernels:
            warnings.warn(
                f"Fused kernels not available (device={device}, dtype={dtype}). "
                "Using PyTorch fallback implementation."
            )
        
        # Gating network
        self.gate_proj = nn.Linear(
            hidden_size, num_experts, bias=gate_bias, dtype=dtype, device=device
        )
        
        # Expert weights - separate projections for better memory layout
        self.gate_weights = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size, dtype=dtype, device=device)
        )
        self.up_weights = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size, dtype=dtype, device=device)
        )
        self.down_weights = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size, dtype=dtype, device=device)
        )
        
        # Optional expert biases
        if expert_bias:
            self.gate_bias = nn.Parameter(
                torch.zeros(num_experts, intermediate_size, dtype=dtype, device=device)
            )
            self.up_bias = nn.Parameter(
                torch.zeros(num_experts, intermediate_size, dtype=dtype, device=device)
            )
            self.down_bias = nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=dtype, device=device)
            )
        else:
            self.register_parameter('gate_bias', None)
            self.register_parameter('up_bias', None)
            self.register_parameter('down_bias', None)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize expert weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.gate_weights)
        nn.init.xavier_uniform_(self.up_weights)
        nn.init.xavier_uniform_(self.down_weights)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE layer.
        
        Args:
            hidden_states: Input tensor [batch_size * seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size * seq_len, hidden_size]
        """
        batch_size, hidden_size = hidden_states.shape
        
        # Ensure input is on the correct device and dtype
        if hidden_states.device != self.device:
            hidden_states = hidden_states.to(self.device)
        if hidden_states.dtype != self.dtype:
            hidden_states = hidden_states.to(self.dtype)
        
        # Compute gating scores
        gate_logits = self.gate_proj(hidden_states)  # [batch_size, num_experts]
        
        # TopK expert selection
        gate_probs = torch.softmax(gate_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Renormalize weights if requested
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Route through experts
        if self.use_fused_kernels:
            # Use optimized Intel XPU fused kernels
            output = self._fused_moe_forward(hidden_states, topk_weights, topk_indices)
        else:
            # Fallback to PyTorch implementation
            output = self._pytorch_moe_forward(hidden_states, topk_weights, topk_indices)
        
        return output
    
    def _fused_moe_forward(
        self, 
        hidden_states: torch.Tensor, 
        topk_weights: torch.Tensor, 
        topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass using Intel XPU fused kernels."""
        
        try:
            output = fused_moe_forward(
                hidden_states=hidden_states,
                gate_weights=self.gate_weights,
                up_weights=self.up_weights,
                down_weights=self.down_weights,
                topk_weights=topk_weights,
                topk_indices=topk_indices,
                top_k=self.top_k,
                renormalize=self.renormalize,
                inplace=False,
                use_grouped_topk=True,
            )
            return output
            
        except Exception as e:
            warnings.warn(f"Fused kernel failed: {e}. Falling back to PyTorch implementation.")
            return self._pytorch_moe_forward(hidden_states, topk_weights, topk_indices)
    
    def _pytorch_moe_forward(
        self, 
        hidden_states: torch.Tensor, 
        topk_weights: torch.Tensor, 
        topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """Fallback PyTorch implementation."""
        
        batch_size, hidden_size = hidden_states.shape
        output = torch.zeros_like(hidden_states)
        
        # Process each token
        for i in range(batch_size):
            token_output = torch.zeros(1, hidden_size, dtype=self.dtype, device=self.device)
            
            # Process each selected expert
            for k in range(self.top_k):
                expert_idx = topk_indices[i, k].item()
                expert_weight = topk_weights[i, k].item()
                
                if expert_weight < 1e-8:  # Skip negligible weights
                    continue
                
                # Expert forward pass
                token_hidden = hidden_states[i:i+1]
                
                # Gate and up projections
                gate_out = torch.matmul(token_hidden, self.gate_weights[expert_idx])
                up_out = torch.matmul(token_hidden, self.up_weights[expert_idx])
                
                # Add bias if present
                if self.gate_bias is not None:
                    gate_out = gate_out + self.gate_bias[expert_idx]
                if self.up_bias is not None:
                    up_out = up_out + self.up_bias[expert_idx]
                
                # SiLU activation and gating
                gate_activated = gate_out * torch.sigmoid(gate_out)  # SiLU
                intermediate = gate_activated * up_out
                
                # Down projection
                expert_output = torch.matmul(intermediate, self.down_weights[expert_idx])
                
                # Add bias if present
                if self.down_bias is not None:
                    expert_output = expert_output + self.down_bias[expert_idx]
                
                # Weighted accumulation
                token_output += expert_weight * expert_output
            
            output[i] = token_output.squeeze(0)
        
        return output
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics in MB."""
        
        def tensor_memory(tensor):
            return tensor.numel() * tensor.element_size() / (1024 * 1024)
        
        memory_stats = {
            "gate_weights": tensor_memory(self.gate_weights),
            "up_weights": tensor_memory(self.up_weights), 
            "down_weights": tensor_memory(self.down_weights),
            "gate_proj": tensor_memory(self.gate_proj.weight),
        }
        
        if self.gate_bias is not None:
            memory_stats["expert_biases"] = (
                tensor_memory(self.gate_bias) + 
                tensor_memory(self.up_bias) + 
                tensor_memory(self.down_bias)
            )
        
        memory_stats["total"] = sum(memory_stats.values())
        return memory_stats
    
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, "
            f"num_experts={self.num_experts}, top_k={self.top_k}, "
            f"fused_kernels={self.use_fused_kernels}, device={self.device}"
        )


def create_sample_moe_model(
    hidden_size: int = 1024,
    intermediate_size: int = 4096,
    num_experts: int = 8,
    top_k: int = 2,
    dtype: torch.dtype = torch.bfloat16,
) -> IntelXPUMoELayer:
    """Create a sample MoE model for testing."""
    
    device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
    
    model = IntelXPUMoELayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        dtype=dtype,
        device=device,
    )
    
    return model


def run_example():
    """Run a simple example of the Intel XPU MoE layer."""
    
    print("Intel XPU MoE Layer Example")
    print("=" * 40)
    
    # Check device availability
    device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "xpu":
        print(f"XPU device: {torch.xpu.get_device_name()}")
    
    print(f"Fused kernels available: {SGL_KERNEL_AVAILABLE}")
    
    # Create model
    model = create_sample_moe_model(
        hidden_size=512,
        intermediate_size=2048,
        num_experts=8,
        top_k=2,
        dtype=torch.bfloat16,
    )
    
    print(f"\nModel: {model}")
    
    # Show memory usage
    memory_stats = model.get_memory_usage()
    print(f"\nMemory usage:")
    for key, value in memory_stats.items():
        print(f"  {key}: {value:.2f} MB")
    
    # Create sample input
    batch_size = 16
    hidden_states = torch.randn(
        batch_size, model.hidden_size, 
        dtype=model.dtype, device=device
    )
    
    print(f"\nInput shape: {hidden_states.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(hidden_states)
    
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output device: {output.device}")
    
    # Check output is finite
    if torch.isfinite(output).all():
        print("✓ Output is finite")
    else:
        print("✗ Output contains non-finite values")
    
    # Compute some basic statistics
    output_mean = output.mean().item()
    output_std = output.std().item()
    output_max = output.max().item()
    output_min = output.min().item()
    
    print(f"\nOutput statistics:")
    print(f"  Mean: {output_mean:.6f}")
    print(f"  Std:  {output_std:.6f}")
    print(f"  Max:  {output_max:.6f}")
    print(f"  Min:  {output_min:.6f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    run_example()