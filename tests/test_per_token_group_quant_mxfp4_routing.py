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


def is_xpu_available_xe35() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _round_mantissa_even_xe35(A: torch.Tensor) -> torch.Tensor:
    """Round mantissa using roundTiesToEven (from Microsoft microxcaling).

    At exact 0.5 midpoints (i.e., values like 0.5, 2.5, 4.5, ...),
    round to the nearest even integer (the one whose LSB is 0).

    Ref: https://github.com/microsoft/microxcaling/blob/main_xe35/mx/elemwise_ops.py
    """
    absA = torch.abs(A)
    # Identify exact midpoints: 0.5, 2.5, 4.5, ...  i.e. (absA - 0.5) % 2 == 0
    maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
    # round half up, then subtract 1 at midpoints to get even
    return torch.sign(A) * (torch.floor(absA + 0.5) - maskA)


def _quantize_elemwise_core_e2m1_xe35(
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

    Ref: https://github.com/microsoft/microxcaling/blob/main_xe35/mx/elemwise_ops.py
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
    out = _round_mantissa_even_xe35(out)

    # Right-shift: undo scaling
    # out = out / 2^(mbits-2) * 2^private_exp
    out = out / (2**shift) * (2**private_exp)

    # Saturate to [-max_norm, max_norm]
    if saturate_normals:
        out = torch.clamp(out, min=-max_norm, max=max_norm)

    return out


def _float_to_e2m1_code_xe35(val: torch.Tensor) -> torch.Tensor:
    """Convert quantized float values back to E2M1 4-bit codes.

    After _quantize_elemwise_core_e2m1_xe35, values are one of the 8 representable
    E2M1 magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
    This maps them to 4-bit codes (sign in bit 3, magnitude in bits 0-2).
    """
    sign = torch.signbit(val).to(torch.uint8)
    abs_val = val.abs()

    # Map representable magnitudes to 3-bit indices via the kE2M1ToFloat LUT.
    indices = torch.zeros_like(abs_val, dtype=torch.uint8)
    lut = kE2M1ToFloat.to(device=val.device)
    for i in range(8):
        indices = torch.where(
            abs_val == lut[i],
            torch.tensor(i, dtype=torch.uint8, device=val.device),
            indices,
        )

    return (sign << 3) | indices


def quantize_to_e2m1_xe35(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor values to E2M1 format (4-bit indices).

    Uses the Microsoft microxcaling _quantize_elemwise_core algorithm
    with roundTiesToEven, then maps the resulting float values to 4-bit codes.

    Ref: https://github.com/microsoft/microxcaling/blob/main_xe35/mx/elemwise_ops.py
    """
    quantized_float = _quantize_elemwise_core_e2m1_xe35(
        tensor.float(), saturate_normals=True
    )
    return _float_to_e2m1_code_xe35(quantized_float)


def pack_fp4_xe35(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape[-1] % 2 == 0
    shape = tensor.shape[:-1] + (tensor.shape[-1] // 2, 2)
    paired = tensor.reshape(shape)
    packed = (paired[..., 0] & 0x0F) | ((paired[..., 1] & 0x0F) << 4)
    return packed.to(torch.uint8)


def unpack_fp4_xe35(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    return unpacked


def dequantize_e2m1_xe35(
    quantized: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    sign = ((quantized >> 3) & 1).to(torch.bool)
    magnitude_idx = (quantized & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=quantized.device)
    magnitude = kE2M1[magnitude_idx]
    result = torch.where(sign, -magnitude, magnitude)
    return result.to(dtype)


def _shared_exponents_xe35(A: torch.Tensor, axis: int) -> torch.Tensor:
    """Compute shared exponents per block using Microsoft microxcaling's
    _shared_exponents_xe35 algorithm with method="max".

    Algorithm:
      1. shared_exp = max(|A|) along axis (per block)
      2. shared_exp = floor(log2(shared_exp + FP32_MIN_NORMAL * (shared_exp == 0)))
         The FP32_MIN_NORMAL guard ensures log2(0) doesn't produce -inf.
      3. Offset by emax: shared_exp = shared_exp - emax

    Ref: https://github.com/microsoft/microxcaling/blob/main_xe35/mx/mx_ops.py
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


def quantize_to_mxfp4_xe35(
    tensor: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE, eps: float = 1e-10
) -> tuple:
    """Quantize to MXFP4 using Microsoft microxcaling's _quantize_mx algorithm.

    Algorithm (from mx_ops.py _quantize_mx):
      1. Reshape into blocks
      2. Compute shared exponent per block via _shared_exponents_xe35
      3. Clamp shared_exp to scale_emax range [-127, 127]
      4. Scale elements: A = A / 2^shared_exp
      5. Quantize element-wise with _quantize_elemwise_core (saturate_normals=True)
      6. Rescale: A = A * 2^shared_exp (implicitly stored in UE8M0 scale)

    Ref: https://github.com/microsoft/microxcaling/blob/main_xe35/mx/mx_ops.py
    """
    assert tensor.dim() == 2
    m, k = tensor.shape
    assert k % block_size == 0
    assert k % 2 == 0

    tensor_fp32 = tensor.float()
    num_blocks = k // block_size
    tensor_blocks = tensor_fp32.reshape(m, num_blocks, block_size)

    # Compute shared exponents (microxcaling _shared_exponents_xe35 + offset by emax)
    shared_exp = _shared_exponents_xe35(tensor_blocks, axis=-1)

    # Clamp to UE8M0 scale range: scale_bits=8, scale_emax = 2^(8-1)-1 = 127
    scale_emax = 127
    shared_exp = shared_exp.clamp(min=-scale_emax, max=scale_emax)

    # Encode as UE8M0: stored_scale = shared_exp + 127
    scales_ue8m0 = (shared_exp.to(torch.int32) + 127).to(torch.uint8).squeeze(-1)

    # Scale elements by shared exponent: A = A / 2^shared_exp
    scaled_tensor = tensor_blocks / (2.0**shared_exp)

    # Quantize element-wise with microxcaling core (roundTiesToEven, saturate)
    quantized_float = _quantize_elemwise_core_e2m1_xe35(scaled_tensor, saturate_normals=True)

    # Convert quantized float values to 4-bit E2M1 codes
    quantized_blocks = _float_to_e2m1_code_xe35(quantized_float)

    quantized = quantized_blocks.reshape(m, k)
    packed = pack_fp4_xe35(quantized)

    return packed, scales_ue8m0


def dequantize_mxfp4_xe35(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> torch.Tensor:
    m, packed_k = packed.shape
    k = packed_k * 2

    unpacked = unpack_fp4_xe35(packed)
    dequantized = dequantize_e2m1_xe35(unpacked, dtype)

    num_blocks = k // block_size
    dequantized_blocks = dequantized.reshape(m, num_blocks, block_size)

    scale_exp = scales.to(torch.int32) - 127
    scale_values = torch.pow(2.0, scale_exp.float()).unsqueeze(-1)
    scaled = dequantized_blocks * scale_values

    return scaled.reshape(m, k).to(dtype)


class TestMXFP4ReferenceQuantization:
    def test_e2m1_zero_codes_exact_xe35(self):
        values = torch.tensor([-0.25, -0.125, -0.0, 0.0, 0.125, 0.25])
        expected = torch.tensor([8, 8, 0, 0, 0, 0], dtype=torch.uint8)
        torch.testing.assert_close(quantize_to_e2m1_xe35(values), expected, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "dtype,gate,up",
        [
            (torch.bfloat16, 0.5, 2.40625),
            (torch.float16, 0.004001617431640625, 374.0),
        ],
    )
    def test_swiglu_output_rounding_changes_fp4_xe35(self, dtype, gate, up):
        """The separate silu_and_mul writes dtype before MXFP4 quantization."""
        x = torch.zeros(1, 64, dtype=dtype)
        x[0, 0], x[0, 32] = gate, up
        x[0, 1], x[0, 33] = 4.0, 1.25  # Keep the block scale at 1.
        activated_fp32 = torch.nn.functional.silu(x[:, :32].float()) * x[:, 32:].float()
        separate = activated_fp32.to(dtype).float()
        assert activated_fp32[0, 0] < 0.75
        assert separate[0, 0] == 0.75
        rounded_q, rounded_s = quantize_to_mxfp4_xe35(separate)
        direct_q, direct_s = quantize_to_mxfp4_xe35(activated_fp32)
        assert torch.equal(rounded_s, direct_s)
        assert rounded_s[0, 0] == 127
        assert unpack_fp4_xe35(rounded_q)[0, 0] == 2  # 0.75 ties to 1.0.
        assert unpack_fp4_xe35(direct_q)[0, 0] == 1  # Below 0.75 gives 0.5.

    def test_e2m1_roundtrip_xe35(self):
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
        quantized = quantize_to_e2m1_xe35(test_values)
        dequantized = dequantize_e2m1_xe35(quantized)
        torch.testing.assert_close(dequantized, test_values, atol=0.0, rtol=0.0)

    def test_e2m1_round_ties_to_even_xe35(self):
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
            quantized = quantize_to_e2m1_xe35(tensor)
            dequantized = dequantize_e2m1_xe35(quantized)
            # For -0.25, dequantized is -0.0 which equals 0.0
            assert dequantized.item() == expected or (
                expected == 0.0 and abs(dequantized.item()) == 0.0
            ), f"Midpoint {midpoint}: expected {expected}, got {dequantized.item()}"

    def test_pack_unpack_roundtrip_xe35(self):
        device = torch.device("cpu")
        m, k = 16, 64
        original = torch.randint(0, 16, (m, k), dtype=torch.uint8, device=device)
        packed = pack_fp4_xe35(original)
        unpacked = unpack_fp4_xe35(packed)
        torch.testing.assert_close(unpacked, original)

    def test_mxfp4_quantization_shape_xe35(self):
        device = torch.device("cpu")
        m, k = 32, 128
        original = torch.randn(m, k, dtype=torch.float32, device=device)
        packed, scales = quantize_to_mxfp4_xe35(original)
        assert packed.shape == (m, k // 2)
        assert scales.shape == (m, k // MXFP4_BLOCK_SIZE)
        assert packed.dtype == torch.uint8
        assert scales.dtype == torch.uint8

    def test_mxfp4_dequantization_accuracy_xe35(self):
        device = torch.device("cpu")
        m, k = 32, 128
        original = torch.randn(m, k, dtype=torch.float32, device=device) * 3.0
        packed, scales = quantize_to_mxfp4_xe35(original)
        dequantized = dequantize_mxfp4_xe35(packed, scales, torch.float32)
        assert dequantized.shape == original.shape
        relative_error = (dequantized - original).abs() / (original.abs() + 1e-6)
        mean_error = relative_error.mean().item()
        assert mean_error < 0.5


def is_cri_device_xe35() -> bool:
    from sgl_kernel import is_xe3_arch

    return is_xe3_arch()


@pytest.mark.skipif(
    not is_xpu_available_xe35() or not is_cri_device_xe35(),
    reason="MXFP4 routing quant kernels require Intel Xe3.5",
)
class TestPerTokenGroupQuantFP4XPU:
    @pytest.fixture(autouse=True)
    def setup_xe35(self):
        import utils

        self.device = utils.get_device()
        self.eps = 1e-10

    def _import_kernel_xe35(self):
        try:
            from sgl_kernel import sgl_per_token_group_quant_fp4_xe35

            return sgl_per_token_group_quant_fp4_xe35
        except ImportError:
            pytest.skip("sgl_per_token_group_quant_fp4_xe35 kernel not available")

    @pytest.mark.parametrize(
        "dtype,gate,up",
        [
            (torch.bfloat16, 0.5, 2.40625),
            (torch.float16, 0.004001617431640625, 374.0),
        ],
    )
    def test_fused_swiglu_matches_separate_dtype_surface_xe35(self, dtype, gate, up):
        from sgl_kernel import silu_and_mul

        x = torch.zeros(1, 64, dtype=dtype, device=self.device)
        x[0, 0], x[0, 32] = gate, up
        x[0, 1], x[0, 33] = 4.0, 1.25
        activated = torch.empty(1, 32, dtype=dtype, device=self.device)
        silu_and_mul(x, out=activated)
        quant = self._import_kernel_xe35()
        values_cpu = x.cpu().float()
        expected_activation = (
            torch.nn.functional.silu(values_cpu[:, :32]) * values_cpu[:, 32:]
        ).to(dtype)
        expected_q, expected_s = quantize_to_mxfp4_xe35(expected_activation)
        separate_q, separate_s = quant(activated)
        actual_q, actual_s = quant(x, fuse_silu_and_mul=True)
        # silu_and_mul keeps bf16 intermediates, so the separate path is checked
        # against the quantization of its own activation, not the fp32 reference.
        separate_ref_q, separate_ref_s = quantize_to_mxfp4_xe35(activated.cpu())
        torch.testing.assert_close(separate_q.cpu(), separate_ref_q, rtol=0, atol=0)
        torch.testing.assert_close(separate_s.cpu(), separate_ref_s, rtol=0, atol=0)
        torch.testing.assert_close(actual_s.cpu(), expected_s, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_q.cpu(),
            expected_q,
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("hidden", [256, 8192])
    def test_fused_swiglu_cpu_reference_xe35(self, dtype, hidden):
        # Muse decode is B8/hidden8192; hidden256 exercises a generic small surface.
        from sgl_kernel import silu_and_mul

        generator = torch.Generator().manual_seed(20261016 + hidden)
        values = torch.randn(8, 2 * hidden, generator=generator).to(dtype)
        activation = (
            torch.nn.functional.silu(values[:, :hidden].float())
            * values[:, hidden:].float()
        ).to(dtype)
        expected_q, expected_s = quantize_to_mxfp4_xe35(activation)
        values_xpu = values.to(self.device)
        separate_activation = silu_and_mul(values_xpu)
        quant = self._import_kernel_xe35()
        separate_q, separate_s = quant(separate_activation)
        fused_q, fused_s = quant(values_xpu, fuse_silu_and_mul=True)
        # silu_and_mul keeps bf16 intermediates, so the separate path is checked
        # against the quantization of its own activation, not the fp32 reference.
        separate_ref_q, separate_ref_s = quantize_to_mxfp4_xe35(
            separate_activation.cpu()
        )
        torch.testing.assert_close(separate_q.cpu(), separate_ref_q, rtol=0, atol=0)
        torch.testing.assert_close(separate_s.cpu(), separate_ref_s, rtol=0, atol=0)
        torch.testing.assert_close(fused_q.cpu(), expected_q, rtol=0, atol=0)
        torch.testing.assert_close(fused_s.cpu(), expected_s, rtol=0, atol=0)

    def _test_against_reference_xe35(
        self,
        num_tokens: int,
        hidden_dim: int,
        src_dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
    ):
        sgl_per_token_group_quant_fp4_xe35 = self._import_kernel_xe35()
        group_size = MXFP4_BLOCK_SIZE

        torch.manual_seed(seed)

        x_cpu = torch.randn(num_tokens, hidden_dim, dtype=src_dtype, device="cpu")
        x_q_ref, scales_ref = quantize_to_mxfp4_xe35(x_cpu.float(), group_size, eps=self.eps)

        x_xpu = x_cpu.to(self.device)
        x_q_xpu, scales_xpu = sgl_per_token_group_quant_fp4_xe35(
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

        torch.testing.assert_close(x_q_xpu_cpu, x_q_ref, rtol=0, atol=0)

        scale_exp_ref = scales_ref.to(torch.int32) - 127
        scale_exp_xpu = scales_xpu_cpu.to(torch.int32) - 127
        exp_diff = (scale_exp_ref - scale_exp_xpu).abs()
        assert exp_diff.max() == 0, f"Scale exponent difference: {exp_diff.max()}"

        # Compare dequantized outputs
        x_dq_ref = dequantize_mxfp4_xe35(x_q_ref, scales_ref, torch.float32, group_size)
        x_dq_xpu = dequantize_mxfp4_xe35(
            x_q_xpu_cpu, scales_xpu_cpu, torch.float32, group_size
        )
        torch.testing.assert_close(x_dq_xpu, x_dq_ref, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        "num_tokens,hidden_dim,src_dtype",
        [
            # Muse BF16 decode surface is B8/hidden8192; other shapes are generic edges.
            (8, 8192, torch.bfloat16),
            (128, 256, torch.bfloat16),
            (64, 128, torch.float16),
            (64, 128, torch.float32),
            (256, 2048, torch.bfloat16),
            # Past the small-surface cutoff: one lane per group.
            (4096, 4096, torch.bfloat16),
        ],
    )
    def test_quantization_vs_reference_xe35(self, num_tokens, hidden_dim, src_dtype):
        self._test_against_reference_xe35(num_tokens, hidden_dim, src_dtype)

    def test_quantize_dequantize_roundtrip_xe35(self):
        sgl_per_token_group_quant_fp4_xe35 = self._import_kernel_xe35()

        torch.manual_seed(42)
        m, k = 128, 256

        x_cpu = torch.randn(m, k, dtype=torch.bfloat16, device="cpu")
        x_xpu = x_cpu.to(self.device)

        x_q, scales = sgl_per_token_group_quant_fp4_xe35(
            x=x_xpu, group_size=MXFP4_BLOCK_SIZE
        )
        expected_q, expected_scales = quantize_to_mxfp4_xe35(x_cpu, MXFP4_BLOCK_SIZE)
        torch.testing.assert_close(x_q.cpu(), expected_q, rtol=0, atol=0)
        torch.testing.assert_close(scales.cpu(), expected_scales, rtol=0, atol=0)

        x_dq = dequantize_mxfp4_xe35(
            x_q.cpu(), scales.cpu(), torch.float32, MXFP4_BLOCK_SIZE
        )

        correlation = torch.corrcoef(
            torch.stack([x_dq.flatten(), x_cpu.float().flatten()])
        )[0, 1]
        assert correlation > 0.9, f"Correlation too low: {correlation}"

    @pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("map_dtype", [torch.int32, torch.int64])
    def test_padded_gather_matches_index_select_then_quantize_xe35(
        self, src_dtype, map_dtype
    ):
        from sgl_kernel import (
            sgl_padded_gather_per_token_group_quant_fp4_xe35,
            sgl_per_token_group_quant_fp4_xe35,
        )

        torch.manual_seed(123)
        source = torch.randn(7, 256, dtype=src_dtype, device=self.device)
        # Repeated source row 0 represents padding; the remaining entries cover
        # a nontrivial permutation with further repeated routed rows.
        row_map = torch.tensor(
            [3, 1, 0, 0, 6, 2, 0, 5, 4, 0, 3],
            dtype=map_dtype,
            device=self.device,
        )

        expected_q, expected_s = sgl_per_token_group_quant_fp4_xe35(
            source.index_select(0, row_map.to(torch.int64))
        )
        actual_q, actual_s = sgl_padded_gather_per_token_group_quant_fp4_xe35(
            source, row_map
        )

        assert torch.equal(actual_q.cpu(), expected_q.cpu())
        assert torch.equal(actual_s.cpu(), expected_s.cpu())

    def test_padded_gather_empty_map_xe35(self):
        from sgl_kernel import sgl_padded_gather_per_token_group_quant_fp4_xe35

        source = torch.empty(2, 64, dtype=torch.bfloat16, device=self.device)
        row_map = torch.empty(0, dtype=torch.int32, device=self.device)
        output_q, output_s = sgl_padded_gather_per_token_group_quant_fp4_xe35(
            source, row_map
        )
        assert output_q.shape == (0, 32)
        assert output_s.shape == (0, 2)

    @pytest.mark.parametrize("fuse_silu_and_mul", [False, True])
    def test_scale_map_writes_mn_major_blocks_xe35(self, fuse_silu_and_mul):
        """`scale_map` rows (start, rows) store MN-major per-block scales,
        (start, -rows) pad rows store only a zero scale, (r, 0) slack nothing."""
        from sgl_kernel import (
            sgl_padded_gather_per_token_group_quant_fp4_xe35,
            sgl_per_token_group_quant_fp4_xe35,
        )

        torch.manual_seed(7)
        groups = 8
        # Two 4-row blocks, the first with two pad rows, the second with one,
        # followed by three slack rows.
        scale_map = torch.tensor(
            [[0, 4], [0, 4], [0, -4], [0, -4], [4, 4], [4, 4], [4, 4], [4, -4]]
            + [[8, 0], [9, 0], [10, 0]],
            dtype=torch.int32,
            device=self.device,
        )
        routed = (scale_map[:, 1] > 0).cpu()
        if fuse_silu_and_mul:
            x = torch.randn(
                11, 2 * groups * 32, dtype=torch.bfloat16, device=self.device
            )
            ref_q, ref_s = sgl_per_token_group_quant_fp4_xe35(x, fuse_silu_and_mul=True)
            q, s = sgl_per_token_group_quant_fp4_xe35(
                x, fuse_silu_and_mul=True, scale_map=scale_map
            )
        else:
            x = torch.randn(5, groups * 32, dtype=torch.bfloat16, device=self.device)
            row_map = torch.tensor(
                [3, 1, 0, 0, 4, 2, 0, 0, 0, 0, 0], dtype=torch.int32, device=self.device
            )
            ref_q, ref_s = sgl_padded_gather_per_token_group_quant_fp4_xe35(x, row_map)
            q, s = sgl_padded_gather_per_token_group_quant_fp4_xe35(
                x, row_map, scale_map=scale_map
            )

        ref_s = ref_s.cpu()
        ref_s[scale_map[:, 1].cpu() < 0] = 0
        want = torch.cat([ref_s[0:4].t().flatten(), ref_s[4:8].t().flatten()])
        assert torch.equal(q.cpu()[routed], ref_q.cpu()[routed])
        assert torch.equal(s.cpu().flatten()[: 8 * groups], want)

    def test_round_ties_to_even_on_xpu_xe35(self):
        """Test that the kernel implements roundTiesToEven at midpoints."""
        sgl_per_token_group_quant_fp4_xe35 = self._import_kernel_xe35()

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

        x_q, scales = sgl_per_token_group_quant_fp4_xe35(
            x=x, group_size=MXFP4_BLOCK_SIZE, eps=self.eps
        )

        # Reference
        x_q_ref, scales_ref = quantize_to_mxfp4_xe35(
            x.cpu().float(), MXFP4_BLOCK_SIZE, eps=self.eps
        )
        torch.testing.assert_close(x_q.cpu(), x_q_ref, rtol=0, atol=0)
        torch.testing.assert_close(scales.cpu(), scales_ref, rtol=0, atol=0)

        x_dq_xpu = dequantize_mxfp4_xe35(
            x_q.cpu(), scales.cpu(), torch.float32, MXFP4_BLOCK_SIZE
        )
        x_dq_ref = dequantize_mxfp4_xe35(
            x_q_ref, scales_ref, torch.float32, MXFP4_BLOCK_SIZE
        )

        torch.testing.assert_close(x_dq_xpu, x_dq_ref, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
