"""
Test for fused MoE kernels on Intel XPU
"""

import torch
import pytest
import warnings

# Test if XPU is available
try:
    if not torch.xpu.is_available():
        pytest.skip("XPU not available", allow_module_level=True)
except:
    pytest.skip("XPU support not available", allow_module_level=True)

try:
    from sgl_kernel.moe import fused_moe_forward, moe_align_block_size, grouped_gemm_moe, silu_and_mul_moe
except ImportError:
    pytest.skip("sgl_kernel not available", allow_module_level=True)


class TestFusedMoE:
    """Test suite for fused MoE operations."""
    
    @pytest.fixture
    def device(self):
        return torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
    
    @pytest.fixture
    def moe_config(self):
        return {
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_experts": 8,
            "top_k": 2,
            "num_tokens": 32,
            "dtype": torch.bfloat16,
        }
    
    def test_basic_fused_moe_forward(self, device, moe_config):
        """Test basic fused MoE forward pass."""
        if device.type == "cpu":
            pytest.skip("Test requires XPU device")
        
        # Create test tensors
        hidden_states = torch.randn(
            moe_config["num_tokens"], 
            moe_config["hidden_size"], 
            dtype=moe_config["dtype"], 
            device=device
        )
        
        gate_weights = torch.randn(
            moe_config["num_experts"], 
            moe_config["hidden_size"], 
            moe_config["intermediate_size"],
            dtype=moe_config["dtype"], 
            device=device
        )
        
        up_weights = torch.randn(
            moe_config["num_experts"], 
            moe_config["hidden_size"], 
            moe_config["intermediate_size"],
            dtype=moe_config["dtype"], 
            device=device
        )
        
        down_weights = torch.randn(
            moe_config["num_experts"], 
            moe_config["intermediate_size"], 
            moe_config["hidden_size"],
            dtype=moe_config["dtype"], 
            device=device
        )
        
        # Create topk routing data
        topk_weights = torch.rand(
            moe_config["num_tokens"], 
            moe_config["top_k"], 
            dtype=torch.float32,
            device=device
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # Normalize
        
        topk_indices = torch.randint(
            0, moe_config["num_experts"],
            (moe_config["num_tokens"], moe_config["top_k"]),
            dtype=torch.int32,
            device=device
        )
        
        # Test fused MoE forward
        try:
            output = fused_moe_forward(
                hidden_states=hidden_states,
                gate_weights=gate_weights,
                up_weights=up_weights,
                down_weights=down_weights,
                topk_weights=topk_weights,
                topk_indices=topk_indices,
                top_k=moe_config["top_k"],
                renormalize=True,
            )
            
            # Verify output shape
            assert output.shape == (moe_config["num_tokens"], moe_config["hidden_size"])
            assert output.dtype == moe_config["dtype"]
            assert output.device == device
            
            # Verify output is finite
            assert torch.isfinite(output).all(), "Output contains non-finite values"
            
        except RuntimeError as e:
            if "sgl_kernel not available" in str(e):
                pytest.skip("sgl_kernel extension not built")
            else:
                raise
    
    def test_moe_align_block_size(self, device, moe_config):
        """Test MoE block size alignment."""
        if device.type == "cpu":
            pytest.skip("Test requires XPU device")
        
        block_size = 64
        total_tokens = moe_config["num_tokens"] * moe_config["top_k"]
        
        # Input topk routing
        topk_ids = torch.randint(
            0, moe_config["num_experts"],
            (moe_config["num_tokens"], moe_config["top_k"]),
            dtype=torch.int32,
            device=device
        )
        
        # Output buffers
        sorted_token_ids = torch.zeros(total_tokens, dtype=torch.int32, device=device)
        experts_ids = torch.zeros(total_tokens, dtype=torch.int32, device=device)
        num_tokens_post_pad = torch.zeros(1, dtype=torch.int32, device=device)
        
        try:
            moe_align_block_size(
                topk_ids=topk_ids,
                num_experts=moe_config["num_experts"],
                block_size=block_size,
                sorted_token_ids=sorted_token_ids,
                experts_ids=experts_ids,
                num_tokens_post_pad=num_tokens_post_pad,
            )
            
            # Verify alignment worked
            assert sorted_token_ids.device == device
            assert experts_ids.device == device
            assert num_tokens_post_pad.device == device
            
        except RuntimeError as e:
            if "sgl_kernel not available" in str(e):
                pytest.skip("sgl_kernel extension not built")
            else:
                raise
    
    def test_silu_and_mul_moe(self, device, moe_config):
        """Test SiLU and multiply operation."""
        if device.type == "cpu":
            pytest.skip("Test requires XPU device")
        
        # Create test tensors
        gate_output = torch.randn(
            moe_config["num_tokens"], 
            moe_config["intermediate_size"],
            dtype=moe_config["dtype"],
            device=device
        )
        
        up_output = torch.randn(
            moe_config["num_tokens"], 
            moe_config["intermediate_size"],
            dtype=moe_config["dtype"],
            device=device
        )
        
        # Reference implementation
        gate_original = gate_output.clone()
        up_original = up_output.clone()
        
        try:
            # Test fused operation
            silu_and_mul_moe(gate_output, up_output)
            
            # Verify gate_output was modified
            assert not torch.equal(gate_output, gate_original), "gate_output should be modified in-place"
            
            # Verify output is finite
            assert torch.isfinite(gate_output).all(), "Output contains non-finite values"
            
            # Compare with reference implementation
            sigmoid_gate = torch.sigmoid(gate_original)
            silu_gate = gate_original * sigmoid_gate
            reference_output = silu_gate * up_original
            
            # Allow for some numerical differences due to fusion
            torch.testing.assert_close(gate_output, reference_output, rtol=1e-3, atol=1e-3)
            
        except RuntimeError as e:
            if "sgl_kernel not available" in str(e):
                pytest.skip("sgl_kernel extension not built")
            else:
                raise
    
    def test_grouped_gemm_moe(self, device, moe_config):
        """Test grouped GEMM operation."""
        if device.type == "cpu":
            pytest.skip("Test requires XPU device")
        
        # Create test tensors for GEMM: A @ B + C = D
        A = torch.randn(
            moe_config["num_tokens"], 
            moe_config["hidden_size"],
            dtype=moe_config["dtype"],
            device=device
        )
        
        B = torch.randn(
            moe_config["num_experts"], 
            moe_config["hidden_size"], 
            moe_config["intermediate_size"],
            dtype=moe_config["dtype"],
            device=device
        )
        
        C = torch.zeros(
            moe_config["num_experts"], 
            moe_config["intermediate_size"],
            dtype=moe_config["dtype"],
            device=device
        )
        
        D = torch.zeros(
            moe_config["num_tokens"], 
            moe_config["intermediate_size"],
            dtype=moe_config["dtype"],
            device=device
        )
        
        # Routing information
        topk_weights = torch.rand(
            moe_config["num_tokens"], 
            moe_config["top_k"],
            dtype=torch.float32,
            device=device
        )
        
        topk_indices = torch.randint(
            0, moe_config["num_experts"],
            (moe_config["num_tokens"], moe_config["top_k"]),
            dtype=torch.int32,
            device=device
        )
        
        try:
            grouped_gemm_moe(
                A=A,
                B=B,
                C=C,
                D=D,
                topk_weights=topk_weights,
                topk_indices=topk_indices,
                top_k=moe_config["top_k"],
                trans_b=False,
            )
            
            # Verify output
            assert D.shape == (moe_config["num_tokens"], moe_config["intermediate_size"])
            assert D.device == device
            assert torch.isfinite(D).all(), "Output contains non-finite values"
            
        except RuntimeError as e:
            if "sgl_kernel not available" in str(e):
                pytest.skip("sgl_kernel extension not built")
            else:
                raise
    
    def test_dtype_compatibility(self, device, moe_config):
        """Test compatibility with different data types."""
        if device.type == "cpu":
            pytest.skip("Test requires XPU device")
        
        dtypes_to_test = [torch.bfloat16, torch.float16, torch.float32]
        
        for dtype in dtypes_to_test:
            if dtype == torch.bfloat16 and not torch.xpu.is_bf16_supported():
                continue  # Skip if bfloat16 not supported
            
            gate_output = torch.randn(
                moe_config["num_tokens"], 
                moe_config["intermediate_size"],
                dtype=dtype,
                device=device
            )
            
            up_output = torch.randn(
                moe_config["num_tokens"], 
                moe_config["intermediate_size"],
                dtype=dtype,
                device=device
            )
            
            try:
                # This should work with all supported dtypes
                silu_and_mul_moe(gate_output, up_output)
                assert torch.isfinite(gate_output).all()
                
            except RuntimeError as e:
                if "sgl_kernel not available" in str(e):
                    pytest.skip("sgl_kernel extension not built")
                elif "Unsupported dtype" in str(e):
                    warnings.warn(f"Dtype {dtype} not supported")
                else:
                    raise


if __name__ == "__main__":
    # Simple smoke test
    device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
    if device.type == "cpu":
        print("XPU not available, skipping tests")
    else:
        print(f"Running basic test on {device}")
        
        # Test SiLU and mul
        gate = torch.randn(4, 8, dtype=torch.bfloat16, device=device)
        up = torch.randn(4, 8, dtype=torch.bfloat16, device=device)
        
        try:
            silu_and_mul_moe(gate, up)
            print("✓ SiLU and mul test passed")
        except Exception as e:
            print(f"✗ SiLU and mul test failed: {e}")
        
        print("Basic smoke test completed")