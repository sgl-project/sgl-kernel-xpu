# SPDX-License-Identifier: Apache-2.0
"""
Tests for MXFP4 (E2M1) Per-Token Group Quantization on Intel XPU

MXFP4 follows the OpenCompute MX (Microscaling) format specification:
- Data type: E2M1 (4-bit float with 2-bit exponent, 1-bit mantissa)
- Block size: 32 elements per scale factor
- Scale format: UE8M0 (unsigned 8-bit exponent-only, no mantissa)

Rounding: Per OCP MX spec (section 5.3.3), FP4 conversion uses
roundTiesToEven — at midpoints between representable values, the
value with even mantissa (mantissa bit = 0) is chosen.

Usage:
    pytest test_per_token_group_quant_mxfp4.py -v
"""

import pytest
import torch

MXFP4_BLOCK_SIZE = 32
FLOAT4_E2M1_MAX = 6.0

# E2M1 format parameters (from Microsoft microxcaling formats.py)
E2M1_EBITS = 2
E2M1_MBITS = 3  # includes sign bit and implicit one
E2M1_EMAX = 2 ** (E2M1_EBITS - 1)  # = 2
E2M1_MAX_NORM = (
    2**E2M1_EMAX * float(2 ** (E2M1_MBITS - 1) - 1) / 2 ** (E2M1_MBITS - 2)
)  # = 6.0

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)  # 2^(-126)

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _round_mantissa_even(A: torch.Tensor) -> torch.Tensor:
    """Round mantissa using roundTiesToEven (from Microsoft microxcaling).

    At exact 0.5 midpoints (i.e., values like 0.5, 2.5, 4.5, ...),
    round to the nearest even integer (the one whose LSB is 0).

    Ref: https://github.com/microsoft/microxcaling/blob/main/mx/elemwise_ops.py
    """
    absA = torch.abs(A)
    # Identify exact midpoints: 0.5, 2.5, 4.5, ...  i.e. (absA - 0.5) % 2 == 0
    maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
    # round half up, then subtract 1 at midpoints to get even
    return torch.sign(A) * (torch.floor(absA + 0.5) - maskA)


def _quantize_elemwise_core_e2m1(
    A: torch.Tensor, saturate_normals: bool = True
) -> torch.Tensor:
    """Element-wise quantization to E2M1 using Microsoft microxcaling's
    _quantize_elemwise_core algorithm with round='even'.

    E2M1 format: ebits=2, mbits=3, emax=2, max_norm=6.0
    min_exp = -(2^(ebits-1)) + 2 = 0

    Algorithm (from Microsoft microxcaling elemwise_ops.py):
      1. Compute per-element private exponent = floor(log2(|A|)),
         clamped to min_exp.
      2. Left-shift: out = A / 2^private_exp * 2^(mbits-2)
      3. Round mantissa with roundTiesToEven
      4. Right-shift: out = out / 2^(mbits-2) * 2^private_exp
      5. Clamp to [-max_norm, max_norm] if saturate_normals

    Ref: https://github.com/microsoft/microxcaling/blob/main/mx/elemwise_ops.py
    """
    ebits = E2M1_EBITS  # 2
    mbits = E2M1_MBITS  # 3
    max_norm = E2M1_MAX_NORM  # 6.0

    # min representable exponent: -(2^(ebits-1)) + 2 = 0
    min_exp = -(2 ** (ebits - 1)) + 2  # 0

    out = A.clone()

    # Per-element private exponent: floor(log2(|A|))
    # Add guard for zeros: log2(0) is -inf, we use (A==0) to avoid that
    private_exp = torch.floor(torch.log2(torch.abs(A) + (A == 0).type(A.dtype)))
    private_exp = private_exp.clip(min=min_exp)

    # Left-shift: scale up so mantissa bits land in integer portion
    # out = A / 2^private_exp * 2^(mbits-2)
    shift = mbits - 2  # = 1
    out = out / (2**private_exp) * (2**shift)

    # Round mantissa with roundTiesToEven
    out = _round_mantissa_even(out)

    # Right-shift: undo scaling
    # out = out / 2^(mbits-2) * 2^private_exp
    out = out / (2**shift) * (2**private_exp)

    # Saturate to [-max_norm, max_norm]
    if saturate_normals:
        out = torch.clamp(out, min=-max_norm, max=max_norm)

    return out


def _float_to_e2m1_code(val: torch.Tensor) -> torch.Tensor:
    """Convert quantized float values back to E2M1 4-bit codes.

    After _quantize_elemwise_core_e2m1, values are one of the 8 representable
    E2M1 magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
    This maps them to 4-bit codes (sign in bit 3, magnitude in bits 0-2).
    """
    sign = (val < 0).to(torch.uint8)
    abs_val = val.abs()

    # Map representable magnitudes to 3-bit indices via the kE2M1ToFloat LUT.
    # Use a tolerance-based comparison since values are exact after quantization.
    indices = torch.zeros_like(abs_val, dtype=torch.uint8)
    lut = kE2M1ToFloat.to(device=val.device)
    for i in range(8):
        indices = torch.where(
            torch.isclose(abs_val, lut[i].expand_as(abs_val), atol=1e-6, rtol=0),
            torch.tensor(i, dtype=torch.uint8, device=val.device),
            indices,
        )

    return (sign << 3) | indices


def quantize_to_e2m1(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor values to E2M1 format (4-bit indices).

    Uses the Microsoft microxcaling _quantize_elemwise_core algorithm
    with roundTiesToEven, then maps the resulting float values to 4-bit codes.

    Ref: https://github.com/microsoft/microxcaling/blob/main/mx/elemwise_ops.py
    """
    quantized_float = _quantize_elemwise_core_e2m1(
        tensor.float(), saturate_normals=True
    )
    return _float_to_e2m1_code(quantized_float)


def pack_fp4(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape[-1] % 2 == 0
    shape = tensor.shape[:-1] + (tensor.shape[-1] // 2, 2)
    paired = tensor.reshape(shape)
    packed = (paired[..., 0] & 0x0F) | ((paired[..., 1] & 0x0F) << 4)
    return packed.to(torch.uint8)


def _normalize_packed_fp4_signed_zero(packed: torch.Tensor) -> torch.Tensor:
    """Canonicalize signed zeros in packed FP4 bytes.

    In E2M1, code 0x0 is +0.0 and code 0x8 is -0.0.  Both represent
    the same value, but different implementations may emit either form.
    This helper rewrites every -0.0 nibble (0x8) to +0.0 (0x0) so that
    byte-level comparisons are not tripped up by this harmless difference.
    """
    # For each nibble, 0x8 is the only code that equals -0.0
    # (sign=1, exponent=0, mantissa=0).  Clear bit 3 whenever the
    # lower 3 bits (magnitude) are zero — i.e. the nibble is 0x0 or 0x8.
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    lo = torch.where(lo == 0x08, torch.zeros_like(lo), lo)
    hi = torch.where(hi == 0x08, torch.zeros_like(hi), hi)
    return (lo | (hi << 4)).to(torch.uint8)


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


def _shared_exponents(A: torch.Tensor, axis: int) -> torch.Tensor:
    """Compute shared exponents per block using Microsoft microxcaling's
    _shared_exponents algorithm with method="max".

    Algorithm:
      1. shared_exp = max(|A|) along axis (per block)
      2. shared_exp = floor(log2(shared_exp + FP32_MIN_NORMAL * (shared_exp == 0)))
         The FP32_MIN_NORMAL guard ensures log2(0) doesn't produce -inf.
      3. Offset by emax: shared_exp = shared_exp - emax

    Ref: https://github.com/microsoft/microxcaling/blob/main/mx/mx_ops.py
    """
    shared_exp = torch.max(torch.abs(A), dim=axis, keepdim=True).values

    # floor(log2(...)) with zero-guard from microxcaling
    shared_exp = torch.floor(
        torch.log2(
            shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
        )
    )

    # Offset by the largest representable exponent in E2M1
    shared_exp = shared_exp - E2M1_EMAX

    return shared_exp


def quantize_to_mxfp4(
    tensor: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE, eps: float = 1e-10
) -> tuple:
    """Quantize to MXFP4 using Microsoft microxcaling's _quantize_mx algorithm.

    Algorithm (from mx_ops.py _quantize_mx):
      1. Reshape into blocks
      2. Compute shared exponent per block via _shared_exponents
      3. Clamp shared_exp to scale_emax range [-127, 127]
      4. Scale elements: A = A / 2^shared_exp
      5. Quantize element-wise with _quantize_elemwise_core (saturate_normals=True)
      6. Rescale: A = A * 2^shared_exp (implicitly stored in UE8M0 scale)

    Ref: https://github.com/microsoft/microxcaling/blob/main/mx/mx_ops.py
    """
    assert tensor.dim() == 2
    m, k = tensor.shape
    assert k % block_size == 0
    assert k % 2 == 0

    tensor_fp32 = tensor.float()
    num_blocks = k // block_size
    tensor_blocks = tensor_fp32.reshape(m, num_blocks, block_size)

    # Compute shared exponents (microxcaling _shared_exponents + offset by emax)
    shared_exp = _shared_exponents(tensor_blocks, axis=-1)

    # Clamp to UE8M0 scale range: scale_bits=8, scale_emax = 2^(8-1)-1 = 127
    scale_emax = 127
    shared_exp = shared_exp.clamp(min=-scale_emax, max=scale_emax)

    # Encode as UE8M0: stored_scale = shared_exp + 127
    scales_ue8m0 = (shared_exp.to(torch.int32) + 127).to(torch.uint8).squeeze(-1)

    # Scale elements by shared exponent: A = A / 2^shared_exp
    scaled_tensor = tensor_blocks / (2.0**shared_exp)

    # Quantize element-wise with microxcaling core (roundTiesToEven, saturate)
    quantized_float = _quantize_elemwise_core_e2m1(scaled_tensor, saturate_normals=True)

    # Convert quantized float values to 4-bit E2M1 codes
    quantized_blocks = _float_to_e2m1_code(quantized_float)

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

    def test_e2m1_round_ties_to_even(self):
        """Test that midpoints between representable values round to even (m=0).

        Per OCP MX spec section 5.3.3, FP4 must use roundTiesToEven.
        At midpoints, the value with even mantissa (m=0) is chosen.
        """
        device = torch.device("cpu")
        # Midpoint values and their expected quantized results
        # (midpoint_value, expected_dequantized_value)
        midpoint_tests = [
            (0.25, 0.0),  # midpoint of (0.0, 0.5) -> 0.0 (m=0, even)
            (0.75, 1.0),  # midpoint of (0.5, 1.0) -> 1.0 (m=0, even)
            (1.25, 1.0),  # midpoint of (1.0, 1.5) -> 1.0 (m=0, even)
            (1.75, 2.0),  # midpoint of (1.5, 2.0) -> 2.0 (m=0, even)
            (2.5, 2.0),  # midpoint of (2.0, 3.0) -> 2.0 (m=0, even)
            (3.5, 4.0),  # midpoint of (3.0, 4.0) -> 4.0 (m=0, even)
            (5.0, 4.0),  # midpoint of (4.0, 6.0) -> 4.0 (m=0, even)
            # Negative midpoints
            (-0.25, 0.0),  # -> -0.0 = 0.0
            (-0.75, -1.0),
            (-1.25, -1.0),
            (-1.75, -2.0),
            (-2.5, -2.0),
            (-3.5, -4.0),
            (-5.0, -4.0),
        ]
        for midpoint, expected in midpoint_tests:
            tensor = torch.tensor([midpoint], dtype=torch.float32, device=device)
            quantized = quantize_to_e2m1(tensor)
            dequantized = dequantize_e2m1(quantized)
            # For -0.25, dequantized is -0.0 which equals 0.0
            assert dequantized.item() == expected or (
                expected == 0.0 and abs(dequantized.item()) == 0.0
            ), f"Midpoint {midpoint}: expected {expected}, got {dequantized.item()}"

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
        x_q_ref, scales_ref = quantize_to_mxfp4(x_cpu.float(), group_size, eps=self.eps)

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

        # Compare quantized values directly (packed uint8).
        # Normalise signed zeros first: in E2M1 code 0x0 (+0.0) and 0x8
        # (-0.0) are semantically identical.  The kernel may preserve the
        # sign of the original float while the reference always emits +0.0,
        # so we canonicalise before comparing.
        x_q_xpu_norm = _normalize_packed_fp4_signed_zero(x_q_xpu_cpu)
        x_q_ref_norm = _normalize_packed_fp4_signed_zero(x_q_ref)
        q_match = torch.equal(x_q_xpu_norm, x_q_ref_norm)
        if not q_match:
            q_mismatches = (x_q_xpu_norm != x_q_ref_norm).sum().item()
            total = x_q_ref_norm.numel()
            assert (
                q_mismatches / total < 0.05
            ), f"Too many quantized value mismatches: {q_mismatches}/{total}"

        # Compare scale exponents (allow ±1 difference due to rounding)
        scale_exp_ref = scales_ref.to(torch.int32) - 127
        scale_exp_xpu = scales_xpu_cpu.to(torch.int32) - 127
        exp_diff = (scale_exp_ref - scale_exp_xpu).abs()
        assert exp_diff.max() == 0, f"Scale exponent difference: {exp_diff.max()}"

        # Compare dequantized outputs
        x_dq_ref = dequantize_mxfp4(x_q_ref, scales_ref, torch.float32, group_size)
        x_dq_xpu = dequantize_mxfp4(
            x_q_xpu_cpu, scales_xpu_cpu, torch.float32, group_size
        )
        torch.testing.assert_close(x_dq_xpu, x_dq_ref, rtol=0.0, atol=0.0)

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

    def test_round_ties_to_even_on_xpu(self):
        """Test that the kernel implements roundTiesToEven at midpoints."""
        sgl_per_token_group_quant_fp4 = self._import_kernel()

        # Create a tensor of exactly 32 elements (one group) containing
        # midpoint values. Scale will be 1.0 (exponent=0) since max abs is 5.0
        # which maps to scale = 2^(floor(log2(5.0)) - 2) = 2^(2 - 2) = 2^0 = 1.0
        midpoints = [
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -3.5,
            -5.0,
        ]
        # Pad to 32 elements with zeros
        padded = midpoints + [0.0] * (32 - len(midpoints))
        x = torch.tensor([padded], dtype=torch.float32, device=self.device)

        x_q, scales = sgl_per_token_group_quant_fp4(
            x=x, group_size=MXFP4_BLOCK_SIZE, eps=self.eps
        )

        # Reference
        x_q_ref, scales_ref = quantize_to_mxfp4(
            x.cpu().float(), MXFP4_BLOCK_SIZE, eps=self.eps
        )

        x_dq_xpu = dequantize_mxfp4(
            x_q.cpu(), scales.cpu(), torch.float32, MXFP4_BLOCK_SIZE
        )
        x_dq_ref = dequantize_mxfp4(
            x_q_ref, scales_ref, torch.float32, MXFP4_BLOCK_SIZE
        )

        torch.testing.assert_close(x_dq_xpu, x_dq_ref, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
