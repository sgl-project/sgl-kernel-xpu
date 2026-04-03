# SPDX-License-Identifier: Apache-2.0
"""Tests for MXFP8 block-scaled grouped GEMM (MoE) on Intel XPU.

MXFP8: FP8 E4M3 data, float32 scales, block size 128.
A: (M,K) RowMajor, scales (M, K//128) col-major.
B: (N,K) ColumnMajor, scales (N//128, K//128) row-major.
Output: (M,N) RowMajor, float32.
"""

import pytest
import torch

MXFP8_BLOCK_SIZE = 128
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

MNK_FACTORS = [
    (128, 128, 128),
    (128, 256, 256),
    (256, 256, 256),
    (256, 512, 512),
    (512, 512, 512),
    (256, 256, 1024),
]


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_cri_device() -> bool:
    """Check whether the current XPU device is CRI (Xe3P)."""
    if not is_xpu_available():
        return False
    try:
        from sgl_kernel import is_xe3_arch

        return is_xe3_arch()
    except ImportError:
        return False


def skip_if_no_xpu():
    if not is_xpu_available():
        pytest.skip("Intel XPU not available")


# ---------------------------------------------------------------------------
# Quantization / dequantization helpers
# ---------------------------------------------------------------------------


def quantize_to_fp8_e4m3(
    tensor: torch.Tensor, block_size: int = MXFP8_BLOCK_SIZE
) -> tuple:
    """Quantize float32 to FP8 E4M3 with per-block scales along K.

    Returns (quantized [float8_e4m3fn], scales [float32 (rows, cols//block_size)]).
    """
    assert tensor.dim() == 2, "Input must be 2-dimensional"
    rows, cols = tensor.shape
    assert (
        cols % block_size == 0
    ), f"Columns ({cols}) must be divisible by block_size ({block_size})"

    tensor_fp32 = tensor.float()
    num_blocks = cols // block_size
    tensor_blocks = tensor_fp32.reshape(rows, num_blocks, block_size)

    block_amax = tensor_blocks.abs().amax(dim=-1)
    block_amax = torch.clamp(block_amax, min=1e-12)
    scales = (block_amax / FP8_E4M3_MAX).float()

    scale_expanded = scales.unsqueeze(-1)
    scaled_blocks = tensor_blocks / scale_expanded
    clamped = scaled_blocks.clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    quantized = clamped.reshape(rows, cols).to(torch.float8_e4m3fn)

    return quantized, scales


def dequantize_fp8_e4m3(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    block_size: int = MXFP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize FP8 E4M3 tensor using per-block scales."""
    rows, cols = quantized.shape
    num_blocks = cols // block_size

    dq = quantized.to(torch.float32).reshape(rows, num_blocks, block_size)
    scaled = dq * scales.float().unsqueeze(-1)
    return scaled.reshape(rows, cols).to(dtype)


def quantize_matrix_blockwise_2d(
    tensor: torch.Tensor,
    block_k: int = MXFP8_BLOCK_SIZE,
    block_n: int = MXFP8_BLOCK_SIZE,
):
    """Quantize (N,K) matrix to FP8 with 2D blocking.

    Returns (quantized [float8_e4m3fn], scales [N//block_n, K//block_k]).
    """
    assert tensor.dim() == 2
    n, k = tensor.shape
    assert n % block_n == 0, f"N ({n}) must be divisible by block_n ({block_n})"
    assert k % block_k == 0, f"K ({k}) must be divisible by block_k ({block_k})"

    n_blocks = n // block_n
    k_blocks = k // block_k

    blocked = tensor.float().reshape(n_blocks, block_n, k_blocks, block_k)
    block_amax = blocked.abs().amax(dim=(1, 3))
    block_amax = torch.clamp(block_amax, min=1e-12)
    scales = (block_amax / FP8_E4M3_MAX).float()

    scale_expanded = scales.unsqueeze(1).unsqueeze(3)
    clamped = (blocked / scale_expanded).clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    quantized = clamped.reshape(n, k).to(torch.float8_e4m3fn)

    return quantized, scales


def dequantize_matrix_blockwise_2d(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block_k: int = MXFP8_BLOCK_SIZE,
    block_n: int = MXFP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize (N,K) FP8 matrix with 2D blocking."""
    n, k = quantized.shape
    n_blocks = n // block_n
    k_blocks = k // block_k

    dq = quantized.to(torch.float32).reshape(n_blocks, block_n, k_blocks, block_k)
    result = (dq * scales.float().unsqueeze(1).unsqueeze(3)).reshape(n, k)
    return result.to(dtype)


# ---------------------------------------------------------------------------
# Data creation helpers
# ---------------------------------------------------------------------------


def create_random_mxfp8_data(rows: int, cols: int, device: str, seed: int = 42):
    """Create random FP8 quantized data with block scales.

    Returns (quantized, scales, original).
    """
    torch.manual_seed(seed)
    original = torch.randn(rows, cols, dtype=torch.float32) * 2.0
    quantized, scales = quantize_to_fp8_e4m3(original)
    return quantized.to(device), scales.to(device), original


def _create_b_quantized(n: int, k: int, seed: int):
    """Create a 2D-block-quantized B matrix.  Returns (b_q, sb) on CPU."""
    torch.manual_seed(seed)
    b_orig = torch.randn(n, k, dtype=torch.float32) * 2.0
    return quantize_matrix_blockwise_2d(b_orig)


def ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def reference_grouped_gemm_2d_blocked(
    a_list: list,
    b_list: list,
    scales_a_list: list,
    scales_b_list: list,
    target_device: str = "cpu",
) -> list:
    """Reference grouped GEMM with 1D-blocked A and 2D-blocked B scales."""
    outputs = []
    for a_q, b_q, sa, sb in zip(a_list, b_list, scales_a_list, scales_b_list):
        a_dq = dequantize_fp8_e4m3(a_q.cpu(), sa.cpu(), torch.float32)
        b_dq = dequantize_matrix_blockwise_2d(b_q.cpu(), sb.cpu())
        outputs.append(torch.matmul(a_dq, b_dq.t()).to(target_device))
    return outputs


# ---------------------------------------------------------------------------
# Kernel input preparation & invocation helpers
# ---------------------------------------------------------------------------


def prepare_kernel_inputs(
    a_list: list,
    b_list: list,
    scales_a_list: list,
    scales_b_list: list,
    device: str,
):
    """Stack per-expert tensors and build pointer / metadata arrays for the kernel."""
    num_experts = len(a_list)
    m, k = a_list[0].shape
    n, k_b = b_list[0].shape
    assert k == k_b

    a_stack = torch.stack([ensure_contiguous(a) for a in a_list]).contiguous()
    b_stack = torch.stack([ensure_contiguous(b) for b in b_list]).contiguous()
    # Transpose A scales from row-major (M, K//128) to column-major (K//128, M)
    # to match the kernel's StrideScaleA = Stride<_1, M, 0> expectation.
    scales_a_stack = torch.stack(
        [ensure_contiguous(s.t().contiguous()) for s in scales_a_list]
    ).contiguous()
    scales_b_stack = torch.stack(
        [ensure_contiguous(s) for s in scales_b_list]
    ).contiguous()

    output = torch.zeros((num_experts, m, n), dtype=torch.float32, device=device)

    def _ptrs(t):
        return torch.tensor(
            [t[i].data_ptr() for i in range(num_experts)],
            dtype=torch.uint64,
            device=device,
        )

    return {
        "output": output,
        "a_ptrs": _ptrs(a_stack),
        "b_ptrs": _ptrs(b_stack),
        "out_ptrs": _ptrs(output),
        "a_scales_ptrs": _ptrs(scales_a_stack),
        "b_scales_ptrs": _ptrs(scales_b_stack),
        "a_stack": a_stack,
        "b_stack": b_stack,
        "scales_a_stack": scales_a_stack,
        "scales_b_stack": scales_b_stack,
        "stride_a": torch.full((num_experts,), k, dtype=torch.int64, device=device),
        "stride_b": torch.full((num_experts,), k, dtype=torch.int64, device=device),
        "stride_c": torch.full((num_experts,), n, dtype=torch.int64, device=device),
        "layout_sfa": torch.empty((num_experts, 5), dtype=torch.int32, device=device),
        "layout_sfb": torch.empty((num_experts, 5), dtype=torch.int32, device=device),
        "problem_sizes": torch.tensor(
            [[m, n, k]] * num_experts, dtype=torch.int32, device=device
        ),
        "expert_offsets": torch.arange(num_experts, dtype=torch.int32, device=device),
        "workspace": torch.empty(
            (1024 * 1024 * 1024,), dtype=torch.uint8, device=device
        ),
    }


def _call_kernel(inputs):
    """Invoke ``fp8_blockwise_scaled_grouped_mm`` with the dict returned by
    ``prepare_kernel_inputs``."""
    from sgl_kernel import fp8_blockwise_scaled_grouped_mm

    fp8_blockwise_scaled_grouped_mm(
        inputs["output"],
        inputs["a_ptrs"],
        inputs["b_ptrs"],
        inputs["out_ptrs"],
        inputs["a_scales_ptrs"],
        inputs["b_scales_ptrs"],
        inputs["a_stack"],
        inputs["b_stack"],
        inputs["scales_a_stack"],
        inputs["scales_b_stack"],
        inputs["stride_a"],
        inputs["stride_b"],
        inputs["stride_c"],
        inputs["layout_sfa"],
        inputs["layout_sfb"],
        inputs["problem_sizes"],
        inputs["expert_offsets"],
        inputs["workspace"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_xpu_available(), reason="Intel XPU not available")
@pytest.mark.skipif(
    is_xpu_available() and not is_cri_device(),
    reason="MXFP8 blockwise scaled grouped GEMM requires a CRI (Xe3P) device",
)
class TestMXFP8BlockwiseScaledGroupedMM:
    """Tests for the MXFP8 MoE CUTLASS kernel on Intel XPU."""

    @pytest.fixture(autouse=True)
    def check_kernel_available(self):
        skip_if_no_xpu()
        if not is_cri_device():
            pytest.skip(
                "MXFP8 blockwise scaled grouped GEMM requires a CRI (Xe3P) device"
            )

    # ------------------------------------------------------------------
    # Accuracy tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("m,n,k", MNK_FACTORS)
    @pytest.mark.parametrize("num_experts", [2, 4, 8])
    @torch.inference_mode()
    def test_kernel_vs_reference(self, m: int, n: int, k: int, num_experts: int):
        """Compare kernel output to reference dequantized fp32 GEMM."""
        device = "xpu"

        a_list, b_list, sa_list, sb_list = [], [], [], []
        for i in range(num_experts):
            a_q, sa, _ = create_random_mxfp8_data(m, k, "cpu", seed=42 + i)
            b_q, sb = _create_b_quantized(n, k, seed=100 + i)
            a_list.append(ensure_contiguous(a_q))
            b_list.append(ensure_contiguous(b_q))
            sa_list.append(ensure_contiguous(sa))
            sb_list.append(ensure_contiguous(sb))

        ref_outputs = reference_grouped_gemm_2d_blocked(
            a_list, b_list, sa_list, sb_list, target_device="cpu"
        )

        inputs = prepare_kernel_inputs(
            [x.to(device) for x in a_list],
            [x.to(device) for x in b_list],
            [x.to(device) for x in sa_list],
            [x.to(device) for x in sb_list],
            device,
        )
        _call_kernel(inputs)

        for i in range(num_experts):
            kernel_out = inputs["output"][i].cpu()
            ref_out = ref_outputs[i].cpu()

            assert not torch.isnan(kernel_out).any(), f"Expert {i}: NaN"
            assert not torch.isinf(kernel_out).any(), f"Expert {i}: Inf"

            torch.testing.assert_close(kernel_out, ref_out, atol=1.0, rtol=0.15)

            ref_mag = ref_out.abs().mean()
            if ref_mag > 1e-6:
                ratio = kernel_out.abs().mean() / ref_mag
                assert (
                    0.7 < ratio < 1.3
                ), f"Expert {i}: magnitude ratio {ratio:.4f} out of range"

            if ref_out.numel() > 1 and ref_out.flatten().std() > 1e-6:
                corr = torch.corrcoef(
                    torch.stack([kernel_out.flatten(), ref_out.flatten()])
                )[0, 1]
                assert corr > 0.95, f"Expert {i}: correlation {corr:.4f} too low"

    @torch.inference_mode()
    def test_single_expert(self):
        """Single-expert grouped GEMM."""
        device = "xpu"
        m, n, k = 256, 256, 256

        a_q, sa, _ = create_random_mxfp8_data(m, k, "cpu", seed=77)
        b_q, sb = _create_b_quantized(n, k, seed=78)

        inputs = prepare_kernel_inputs(
            [ensure_contiguous(a_q).to(device)],
            [ensure_contiguous(b_q).to(device)],
            [ensure_contiguous(sa).to(device)],
            [ensure_contiguous(sb).to(device)],
            device,
        )
        _call_kernel(inputs)

        kernel_out = inputs["output"][0].cpu()
        assert not torch.isnan(kernel_out).any()
        assert not torch.isinf(kernel_out).any()

        ref_out = torch.matmul(
            dequantize_fp8_e4m3(a_q, sa),
            dequantize_matrix_blockwise_2d(b_q, sb).t(),
        )
        torch.testing.assert_close(kernel_out, ref_out, atol=1.0, rtol=0.15)

    @torch.inference_mode()
    def test_identity_like_pattern(self):
        """Structured pattern: A = ones, B ≈ identity."""
        device = "xpu"
        m, n, k = 128, 128, 128

        a_q, sa = quantize_to_fp8_e4m3(torch.ones(m, k, dtype=torch.float32))
        b_q, sb = quantize_matrix_blockwise_2d(
            torch.eye(n, k, dtype=torch.float32) * 2.0
        )

        ref_out = torch.matmul(
            dequantize_fp8_e4m3(a_q, sa),
            dequantize_matrix_blockwise_2d(b_q, sb).t(),
        )

        inputs = prepare_kernel_inputs(
            [ensure_contiguous(a_q).to(device)],
            [ensure_contiguous(b_q).to(device)],
            [ensure_contiguous(sa).to(device)],
            [ensure_contiguous(sb).to(device)],
            device,
        )
        _call_kernel(inputs)

        kernel_out = inputs["output"][0].cpu()
        assert not torch.isnan(kernel_out).any()
        assert not torch.isinf(kernel_out).any()
        torch.testing.assert_close(kernel_out, ref_out, atol=0.5, rtol=0.15)

    @torch.inference_mode()
    def test_larger_k_dimension(self):
        """Exercise multi-tile-K with larger K dimension."""
        device = "xpu"
        m, n, k = 256, 256, 1024
        num_experts = 2

        a_list, b_list, sa_list, sb_list = [], [], [], []
        for i in range(num_experts):
            a_q, sa, _ = create_random_mxfp8_data(m, k, "cpu", seed=500 + i)
            b_q, sb = _create_b_quantized(n, k, seed=600 + i)
            a_list.append(ensure_contiguous(a_q))
            b_list.append(ensure_contiguous(b_q))
            sa_list.append(ensure_contiguous(sa))
            sb_list.append(ensure_contiguous(sb))

        ref_outputs = reference_grouped_gemm_2d_blocked(
            a_list, b_list, sa_list, sb_list, target_device="cpu"
        )

        inputs = prepare_kernel_inputs(
            [x.to(device) for x in a_list],
            [x.to(device) for x in b_list],
            [x.to(device) for x in sa_list],
            [x.to(device) for x in sb_list],
            device,
        )
        _call_kernel(inputs)

        for i in range(num_experts):
            kernel_out = inputs["output"][i].cpu()
            ref_out = ref_outputs[i].cpu()
            assert not torch.isnan(kernel_out).any()
            torch.testing.assert_close(kernel_out, ref_out, atol=2.0, rtol=0.2)

    # ------------------------------------------------------------------
    # Property / dtype tests
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def test_output_dtype_is_float32(self):
        """Verify kernel output is float32."""
        device = "xpu"
        m, n, k = 128, 128, 128

        a_q, sa, _ = create_random_mxfp8_data(m, k, "cpu", seed=99)
        b_q, sb = _create_b_quantized(n, k, seed=100)

        inputs = prepare_kernel_inputs(
            [ensure_contiguous(a_q).to(device)],
            [ensure_contiguous(b_q).to(device)],
            [ensure_contiguous(sa).to(device)],
            [ensure_contiguous(sb).to(device)],
            device,
        )
        _call_kernel(inputs)

        assert inputs["output"].dtype == torch.float32

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def test_input_validation_wrong_dtype(self):
        """Kernel rejects non-FP8 inputs."""
        from sgl_kernel import fp8_blockwise_scaled_grouped_mm

        device = "xpu"
        m, n, k = 128, 128, 128

        a_q, sa, _ = create_random_mxfp8_data(m, k, "cpu", seed=50)
        b_q, sb = _create_b_quantized(n, k, seed=51)

        # Use float32 A instead of fp8 — should be rejected
        a_wrong = torch.randn(m, k, dtype=torch.float32, device=device)
        b_correct = ensure_contiguous(b_q).to(device)

        a_stack = a_wrong.unsqueeze(0).contiguous()
        b_stack = b_correct.unsqueeze(0).contiguous()
        sa_stack = ensure_contiguous(sa).to(device).unsqueeze(0).contiguous()
        sb_stack = ensure_contiguous(sb).to(device).unsqueeze(0).contiguous()
        output = torch.zeros((1, m, n), dtype=torch.float32, device=device)

        def _p(t):
            return torch.tensor([t[0].data_ptr()], dtype=torch.uint64, device=device)

        with pytest.raises(RuntimeError, match="float8_e4m3fn"):
            fp8_blockwise_scaled_grouped_mm(
                output,
                _p(a_stack),
                _p(b_stack),
                _p(output),
                _p(sa_stack),
                _p(sb_stack),
                a_stack,
                b_stack,
                sa_stack,
                sb_stack,
                torch.full((1,), k, dtype=torch.int64, device=device),
                torch.full((1,), k, dtype=torch.int64, device=device),
                torch.full((1,), n, dtype=torch.int64, device=device),
                torch.empty((1, 5), dtype=torch.int32, device=device),
                torch.empty((1, 5), dtype=torch.int32, device=device),
                torch.tensor([[m, n, k]], dtype=torch.int32, device=device),
                torch.tensor([0], dtype=torch.int32, device=device),
                torch.empty((1024 * 1024,), dtype=torch.uint8, device=device),
            )

    @torch.inference_mode()
    def test_input_validation_wrong_scale_dtype(self):
        """Kernel rejects non-float32 scales."""
        from sgl_kernel import fp8_blockwise_scaled_grouped_mm

        device = "xpu"
        m, n, k = 128, 128, 128

        a_q, sa, _ = create_random_mxfp8_data(m, k, "cpu", seed=51)
        b_q, sb = _create_b_quantized(n, k, seed=52)

        a_stack = ensure_contiguous(a_q).to(device).unsqueeze(0).contiguous()
        b_stack = ensure_contiguous(b_q).to(device).unsqueeze(0).contiguous()
        # Cast A-scales to float16 — should be rejected
        sa_stack = sa.to(torch.float16).to(device).unsqueeze(0).contiguous()
        sb_stack = ensure_contiguous(sb).to(device).unsqueeze(0).contiguous()
        output = torch.zeros((1, m, n), dtype=torch.float32, device=device)

        def _p(t):
            return torch.tensor([t[0].data_ptr()], dtype=torch.uint64, device=device)

        with pytest.raises(RuntimeError, match="float32"):
            fp8_blockwise_scaled_grouped_mm(
                output,
                _p(a_stack),
                _p(b_stack),
                _p(output),
                _p(sa_stack),
                _p(sb_stack),
                a_stack,
                b_stack,
                sa_stack,
                sb_stack,
                torch.full((1,), k, dtype=torch.int64, device=device),
                torch.full((1,), k, dtype=torch.int64, device=device),
                torch.full((1,), n, dtype=torch.int64, device=device),
                torch.empty((1, 5), dtype=torch.int32, device=device),
                torch.empty((1, 5), dtype=torch.int32, device=device),
                torch.tensor([[m, n, k]], dtype=torch.int32, device=device),
                torch.tensor([0], dtype=torch.int32, device=device),
                torch.empty((1024 * 1024,), dtype=torch.uint8, device=device),
            )
