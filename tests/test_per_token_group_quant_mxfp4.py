# SPDX-License-Identifier: Apache-2.0
"""
Tests for MXFP4 (E2M1) Per-Token Group Quantization on Intel XPU

MXFP4 follows the OpenCompute MX (Microscaling) format specification:
- Data type: E2M1 (4-bit float with 2-bit exponent, 1-bit mantissa)
- Block size: 32 elements per scale factor
- Scale format: UE8M0 (unsigned 8-bit exponent-only, no mantissa)

Usage:
    pytest test_per_token_group_quant_mxfp4.py -v
"""

import pytest
import torch

MXFP4_BLOCK_SIZE = 32
FLOAT4_E2M1_MAX = 6.0

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def quantize_to_e2m1(tensor: torch.Tensor) -> torch.Tensor:
    e2m1_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    sign = (tensor < 0).to(torch.uint8)
    abs_val = torch.clamp(tensor.abs(), max=6.0)
    abs_val_expanded = abs_val.unsqueeze(-1)
    e2m1_expanded = e2m1_values.view(*([1] * abs_val.dim()), -1)
    distances = (abs_val_expanded - e2m1_expanded).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)
    quantized = (sign << 3) | indices
    return quantized


def pack_fp4(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape[-1] % 2 == 0
    shape = tensor.shape[:-1] + (tensor.shape[-1] // 2, 2)
    paired = tensor.reshape(shape)
    packed = (paired[..., 0] & 0x0F) | ((paired[..., 1] & 0x0F) << 4)
    return packed.to(torch.uint8)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    return unpacked


def dequantize_e2m1(
    quantized: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    sign = ((quantized >> 3) & 1).to(torch.bool)
    magnitude_idx = (quantized & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=quantized.device)
    magnitude = kE2M1[magnitude_idx]
    result = torch.where(sign, -magnitude, magnitude)
    return result.to(dtype)


def quantize_to_mxfp4(
    tensor: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE
) -> tuple:
    assert tensor.dim() == 2
    m, k = tensor.shape
    assert k % block_size == 0
    assert k % 2 == 0

    tensor_fp32 = tensor.float()
    num_blocks = k // block_size
    tensor_blocks = tensor_fp32.reshape(m, num_blocks, block_size)

    block_max = tensor_blocks.abs().max(dim=-1, keepdim=True).values
    block_max = torch.clamp(block_max, min=1e-12)

    log2_max = torch.log2(block_max / FLOAT4_E2M1_MAX)
    exponent = torch.ceil(log2_max).clamp(min=-127, max=127).to(torch.int32)
    scales_ue8m0 = (exponent + 127).to(torch.uint8).squeeze(-1)

    scale_values = torch.pow(2.0, exponent.float())
    scaled_tensor = tensor_blocks / scale_values
    quantized_blocks = quantize_to_e2m1(scaled_tensor)
    quantized = quantized_blocks.reshape(m, k)
    packed = pack_fp4(quantized)

    return packed, scales_ue8m0


def dequantize_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> torch.Tensor:
    m, packed_k = packed.shape
    k = packed_k * 2

    unpacked = unpack_fp4(packed)
    dequantized = dequantize_e2m1(unpacked, dtype)

    num_blocks = k // block_size
    dequantized_blocks = dequantized.reshape(m, num_blocks, block_size)

    scale_exp = scales.to(torch.int32) - 127
    scale_values = torch.pow(2.0, scale_exp.float()).unsqueeze(-1)
    scaled = dequantized_blocks * scale_values

    return scaled.reshape(m, k).to(dtype)


class TestMXFP4ReferenceQuantization:
    def test_e2m1_roundtrip(self):
        device = torch.device("cpu")
        test_values = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        quantized = quantize_to_e2m1(test_values)
        dequantized = dequantize_e2m1(quantized)
        torch.testing.assert_close(dequantized, test_values, atol=0.0, rtol=0.0)

    def test_pack_unpack_roundtrip(self):
        device = torch.device("cpu")
        m, k = 16, 64
        original = torch.randint(0, 16, (m, k), dtype=torch.uint8, device=device)
        packed = pack_fp4(original)
        unpacked = unpack_fp4(packed)
        torch.testing.assert_close(unpacked, original)

    def test_mxfp4_quantization_shape(self):
        device = torch.device("cpu")
        m, k = 32, 128
        original = torch.randn(m, k, dtype=torch.float32, device=device)
        packed, scales = quantize_to_mxfp4(original)
        assert packed.shape == (m, k // 2)
        assert scales.shape == (m, k // MXFP4_BLOCK_SIZE)
        assert packed.dtype == torch.uint8
        assert scales.dtype == torch.uint8

    def test_mxfp4_dequantization_accuracy(self):
        device = torch.device("cpu")
        m, k = 32, 128
        original = torch.randn(m, k, dtype=torch.float32, device=device) * 3.0
        packed, scales = quantize_to_mxfp4(original)
        dequantized = dequantize_mxfp4(packed, scales, torch.float32)
        assert dequantized.shape == original.shape
        relative_error = (dequantized - original).abs() / (original.abs() + 1e-6)
        mean_error = relative_error.mean().item()
        assert mean_error < 0.5


@pytest.mark.skipif(not is_xpu_available(), reason="XPU not available")
class TestPerTokenGroupQuantFP4XPU:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = "xpu"
        self.eps = 1e-10

    def _import_kernel(self):
        try:
            from sgl_kernel import sgl_per_token_group_quant_fp4

            return sgl_per_token_group_quant_fp4
        except ImportError:
            pytest.skip("sgl_per_token_group_quant_fp4 kernel not available")

    def _test_against_reference(
        self,
        num_tokens: int,
        hidden_dim: int,
        src_dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
    ):
        sgl_per_token_group_quant_fp4 = self._import_kernel()
        group_size = MXFP4_BLOCK_SIZE

        torch.manual_seed(seed)

        x_cpu = torch.randn(num_tokens, hidden_dim, dtype=src_dtype, device="cpu")
        x_q_ref, scales_ref = quantize_to_mxfp4(x_cpu.float(), group_size)

        x_xpu = x_cpu.to(self.device)
        x_q_xpu, scales_xpu = sgl_per_token_group_quant_fp4(
            x=x_xpu,
            group_size=group_size,
            eps=self.eps,
        )

        x_q_xpu_cpu = x_q_xpu.cpu()
        scales_xpu_cpu = scales_xpu.cpu()

        assert (
            x_q_xpu_cpu.shape == x_q_ref.shape
        ), f"Quantized shape mismatch: {x_q_xpu_cpu.shape} vs {x_q_ref.shape}"
        assert (
            scales_xpu_cpu.shape == scales_ref.shape
        ), f"Scales shape mismatch: {scales_xpu_cpu.shape} vs {scales_ref.shape}"
        assert x_q_xpu_cpu.dtype == torch.uint8
        assert scales_xpu_cpu.dtype == torch.uint8

        x_dq_ref = dequantize_mxfp4(x_q_ref, scales_ref, torch.float32, group_size)
        x_dq_xpu = dequantize_mxfp4(
            x_q_xpu_cpu, scales_xpu_cpu, torch.float32, group_size
        )
        torch.testing.assert_close(x_dq_xpu, x_dq_ref, rtol=0.2, atol=0.5)

        scale_exp_ref = scales_ref.to(torch.int32) - 127
        scale_exp_xpu = scales_xpu_cpu.to(torch.int32) - 127
        exp_diff = (scale_exp_ref - scale_exp_xpu).abs()
        assert (
            exp_diff.max() <= 1
        ), f"Scale exponent difference too large: {exp_diff.max()}"

    @pytest.mark.parametrize(
        "num_tokens,hidden_dim,src_dtype",
        [
            (128, 256, torch.bfloat16),
            (64, 128, torch.float16),
            (64, 128, torch.float32),
            (256, 2048, torch.bfloat16),
        ],
    )
    def test_quantization_vs_reference(self, num_tokens, hidden_dim, src_dtype):
        self._test_against_reference(num_tokens, hidden_dim, src_dtype)

    def test_quantize_dequantize_roundtrip(self):
        sgl_per_token_group_quant_fp4 = self._import_kernel()

        torch.manual_seed(42)
        m, k = 128, 256

        x_cpu = torch.randn(m, k, dtype=torch.bfloat16, device="cpu")
        x_xpu = x_cpu.to(self.device)

        x_q, scales = sgl_per_token_group_quant_fp4(
            x=x_xpu, group_size=MXFP4_BLOCK_SIZE
        )

        x_dq = dequantize_mxfp4(
            x_q.cpu(), scales.cpu(), torch.float32, MXFP4_BLOCK_SIZE
        )

        correlation = torch.corrcoef(
            torch.stack([x_dq.flatten(), x_cpu.float().flatten()])
        )[0, 1]
        assert correlation > 0.9, f"Correlation too low: {correlation}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
