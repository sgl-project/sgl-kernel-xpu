import itertools
import unittest
from typing import Tuple

import pytest
import torch

try:
    HAS_XPU = torch.xpu.is_available()
except (ImportError, AttributeError):
    HAS_XPU = False

try:
    from sglang.srt.layers.quantization.fp8_kernel import (
        per_token_group_quant_8bit as triton_per_token_group_quant_8bit,
    )
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_8bit


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round scale to nearest power of 2 for UE8M0 format."""
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs().clamp(min=1e-10))))


# Pure PyTorch reference implementations
def per_token_group_quant_fp8_ref(
    x: torch.Tensor, group_size: int = 128, eps: float = 1e-10, scale_ue8m0: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for per-token group FP8 quantization."""
    assert x.dim() == 2 and x.size(1) % group_size == 0
    num_tokens, hidden_dim = x.shape
    x_view = x.view(num_tokens, -1, group_size)
    x_amax = x_view.abs().float().amax(dim=2).clamp(min=eps)
    scales = x_amax / 448.0  # FP8 E4M3 max

    if scale_ue8m0:
        scales = ceil_to_ue8m0(scales)

    x_quantized = (x_view / scales.unsqueeze(2)).to(torch.float8_e4m3fn)
    return x_quantized.view(num_tokens, hidden_dim), scales


def per_token_group_quant_int8_ref(
    x: torch.Tensor, group_size: int = 128, eps: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for per-token group INT8 quantization."""
    assert x.dim() == 2 and x.size(1) % group_size == 0
    num_tokens, hidden_dim = x.shape
    x_view = x.view(num_tokens, -1, group_size)
    x_amax = x_view.abs().float().amax(dim=2).clamp(min=eps)
    scales = x_amax / 127.0  # INT8 max
    x_scaled = (x_view / scales.unsqueeze(2)).clamp(-127.0, 127.0)
    x_quantized = x_scaled.to(torch.int8)
    return x_quantized.view(num_tokens, hidden_dim), scales


class TestPerTokenGroupQuantXPU(unittest.TestCase):
    """Test suite for XPU per-token group quantization."""

    @classmethod
    def setUpClass(cls):
        if not HAS_XPU:
            raise unittest.SkipTest("XPU is not available")
        torch.xpu.set_device(0)

    def setUp(self):
        self.device = torch.device("xpu")
        self.eps = 1e-10

    def _test_against_reference(
        self,
        num_tokens: int,
        hidden_dim: int,
        group_size: int,
        dst_dtype: torch.dtype,
        column_major_scales: bool = False,
        scale_ue8m0: bool = False,
        seed: int = 42,
    ):
        """Test XPU implementation against PyTorch reference."""
        torch.manual_seed(seed)
        x_cpu = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
        x_xpu = x_cpu.to(self.device)

        # Get reference output
        if dst_dtype == torch.float8_e4m3fn:
            x_q_ref, scales_ref = per_token_group_quant_fp8_ref(x_cpu, group_size, self.eps, scale_ue8m0)
        else:
            x_q_ref, scales_ref = per_token_group_quant_int8_ref(x_cpu, group_size, self.eps)

        # Run XPU implementation
        x_q_xpu, scales_xpu = sglang_per_token_group_quant_8bit(
            x=x_xpu,
            masked_m=None,
            group_size=group_size,
            eps=self.eps,
            dst_dtype=dst_dtype,
            column_major_scales=column_major_scales,
            scale_ue8m0=scale_ue8m0,
            enable_v2=False,
        )

        # Compare
        x_q_xpu_cpu = x_q_xpu.cpu()
        scales_xpu_cpu = scales_xpu.cpu()

        self.assertEqual(x_q_xpu_cpu.shape, x_q_ref.shape)
        self.assertEqual(x_q_xpu_cpu.dtype, dst_dtype)

        torch.testing.assert_close(
            scales_xpu_cpu, scales_ref, rtol=1e-3, atol=1e-5,
            msg=f"Scales mismatch"
        )

        # Dequantize and compare
        num_groups = hidden_dim // group_size
        x_dq_ref = (x_q_ref.view(num_tokens, num_groups, group_size).to(torch.float32)
                    * scales_ref.unsqueeze(2)).view(num_tokens, hidden_dim)
        x_dq_xpu = (x_q_xpu_cpu.view(num_tokens, num_groups, group_size).to(torch.float32)
                    * scales_xpu_cpu.unsqueeze(2)).view(num_tokens, hidden_dim)

        rtol, atol = (1e-1, 1e-1) if dst_dtype == torch.float8_e4m3fn else (1e-2, 1e-2)
        torch.testing.assert_close(x_dq_xpu, x_dq_ref, rtol=rtol, atol=atol)

    def test_fp8_basic(self):
        """Test basic FP8 quantization."""
        self._test_against_reference(128, 1024, 128, torch.float8_e4m3fn)

    def test_int8_basic(self):
        """Test basic INT8 quantization."""
        self._test_against_reference(128, 1024, 128, torch.int8)

    def test_various_sizes(self):
        """Test various tensor sizes."""
        configs = [
            (64, 512, 64, torch.float8_e4m3fn),
            (128, 2048, 128, torch.float8_e4m3fn),
            (256, 4096, 64, torch.int8),
        ]
        for num_tokens, hidden_dim, group_size, dtype in configs:
            with self.subTest(num_tokens=num_tokens, hidden_dim=hidden_dim,
                              group_size=group_size, dtype=dtype):
                self._test_against_reference(num_tokens, hidden_dim, group_size, dtype)

    def test_column_major_scales(self):
        """Test column-major scale layout."""
        self._test_against_reference(128, 1024, 128, torch.float8_e4m3fn, column_major_scales=True)

    def test_scale_ue8m0(self):
        """Test UE8M0 scale format with column-major layout."""
        self._test_against_reference(128, 1024, 128, torch.float8_e4m3fn,
                                     column_major_scales=True, scale_ue8m0=True)

    def test_edge_cases(self):
        """Test edge cases (small/large values)."""
        for scale_factor in [1e-3, 100.0]:
            with self.subTest(scale_factor=scale_factor):
                torch.manual_seed(42)
                x = torch.randn(64, 512, dtype=torch.bfloat16) * scale_factor
                x_xpu = x.to(self.device)
                x_q, scales = sglang_per_token_group_quant_8bit(
                    x=x_xpu, masked_m=None, group_size=64, eps=self.eps,
                    dst_dtype=torch.float8_e4m3fn, column_major_scales=False,
                    scale_ue8m0=False, enable_v2=False
                )
                self.assertEqual(x_q.shape, x.shape)
                self.assertTrue((scales > 0).all() and torch.isfinite(scales).all())


@pytest.mark.skipif(not HAS_XPU, reason="XPU not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestAgainstTriton:
    """Optional tests comparing XPU implementation against Triton."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.xpu.set_device(0)
        self.device = torch.device("xpu")
        self.eps = 1e-10

    @pytest.mark.parametrize(
        "num_tokens,hidden_dim,group_size,dst_dtype,column_major,scale_ue8m0",
        [
            (128, 1024, 64, torch.float8_e4m3fn, False, False),
            (256, 2048, 128, torch.float8_e4m3fn, True, False),
            (512, 4096, 64, torch.int8, False, False),
            (128, 1024, 128, torch.float8_e4m3fn, True, True),
        ],
    )
    def test_xpu_vs_triton(self, num_tokens, hidden_dim, group_size, dst_dtype, column_major, scale_ue8m0):
        """Compare XPU implementation against Triton reference."""
        torch.manual_seed(42)
        x_cpu = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
        x_xpu = x_cpu.to(self.device)

        # Run Triton on CPU
        x_q_triton, scales_triton = triton_per_token_group_quant_8bit(
            x=x_cpu, masked_m=None, group_size=group_size, eps=self.eps,
            dst_dtype=dst_dtype, column_major_scales=column_major, scale_ue8m0=scale_ue8m0
        )

        # Run XPU
        x_q_xpu, scales_xpu = sglang_per_token_group_quant_8bit(
            x=x_xpu, masked_m=None, group_size=group_size, eps=self.eps,
            dst_dtype=dst_dtype, column_major_scales=column_major,
            scale_ue8m0=scale_ue8m0, enable_v2=False
        )

        # Compare
        x_q_xpu_cpu = x_q_xpu.cpu()
        scales_xpu_cpu = scales_xpu.cpu()

        torch.testing.assert_close(
            scales_xpu_cpu.contiguous(), scales_triton.contiguous(),
            rtol=1e-3, atol=1e-5
        )

        # Compare quantized values (convert to float for comparison)
        x_dq_triton = (x_q_triton.view(num_tokens, -1, group_size).to(torch.float32)
                       * scales_triton.unsqueeze(2)).view(num_tokens, hidden_dim)
        x_dq_xpu = (x_q_xpu_cpu.view(num_tokens, -1, group_size).to(torch.float32)
                    * scales_xpu_cpu.unsqueeze(2)).view(num_tokens, hidden_dim)

        rtol, atol = (1e-1, 1e-1) if dst_dtype == torch.float8_e4m3fn else (1e-2, 1e-2)
        torch.testing.assert_close(x_dq_xpu, x_dq_triton, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
