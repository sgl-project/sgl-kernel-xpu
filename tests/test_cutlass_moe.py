# SPDX-License-Identifier: Apache-2.0
"""E2E accuracy test for sgl_kernel.cutlass_fused_experts_fp8 on Intel XPU.

Compares the wrapper output against a per-token, per-topk-choice dequantized
reference: for each (token, expert) pair, dequantize A and W, run the two
GEMMs + SiLU, then topk-weighted sum.
"""

import pytest
import torch

MXFP8_BLOCK_SIZE = 128
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_cri_device() -> bool:
    if not is_xpu_available():
        return False
    try:
        from sgl_kernel import is_xe3_arch

        return is_xe3_arch()
    except ImportError:
        return False


def _quantize_to_fp8_per_block_along_k(
    x: torch.Tensor, block_size: int = MXFP8_BLOCK_SIZE
):
    """(..., K) -> fp8_e4m3 (..., K) + fp32 scales (..., K//block_size)."""
    assert x.shape[-1] % block_size == 0
    fp32 = x.float()
    leading = fp32.shape[:-1]
    K = fp32.shape[-1]
    blocks = fp32.reshape(*leading, K // block_size, block_size)
    amax = torch.clamp(blocks.abs().amax(dim=-1), min=1e-12)
    scales = (amax / FP8_E4M3_MAX).float()
    scaled = (blocks / scales.unsqueeze(-1)).clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    return scaled.reshape(*leading, K).to(torch.float8_e4m3fn), scales


def _quantize_weight_2d_block_NK(
    w_nk: torch.Tensor,
    block_size: int = MXFP8_BLOCK_SIZE,
):
    """(N, K) row-major -> (fp8_e4m3 (N, K), fp32 scales (N//BS, K//BS))."""
    assert w_nk.dim() == 2
    N, K = w_nk.shape
    assert N % block_size == 0 and K % block_size == 0
    nb_n = N // block_size
    nb_k = K // block_size
    fp32 = w_nk.float().reshape(nb_n, block_size, nb_k, block_size)
    amax = torch.clamp(fp32.abs().amax(dim=(1, 3)), min=1e-12)
    scales = (amax / FP8_E4M3_MAX).float()  # (nb_n, nb_k)
    scaled = (fp32 / scales.unsqueeze(1).unsqueeze(3)).clamp(
        min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX
    )
    w_q = scaled.reshape(N, K).to(torch.float8_e4m3fn)
    return w_q, scales


def _dequantize_w_NK(w_q_nk: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of _quantize_weight_2d_block_NK -> (N, K) fp32."""
    BS = MXFP8_BLOCK_SIZE
    N, K = w_q_nk.shape
    fp32 = w_q_nk.to(torch.float32).reshape(N // BS, BS, K // BS, BS)
    return (fp32 * scales.unsqueeze(1).unsqueeze(3)).reshape(N, K)


def _dequantize_a(a_q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of per-block-along-K quant; scales shape (..., K/BS)."""
    BS = MXFP8_BLOCK_SIZE
    leading = a_q.shape[:-1]
    K = a_q.shape[-1]
    fp32 = a_q.to(torch.float32).reshape(*leading, K // BS, BS)
    return (fp32 * scales.unsqueeze(-1)).reshape(*leading, K)


def _reference_moe_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_q: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-token reference. Weights in (E, N, K) row-major matching the
    kernel: w1=(E, n*2, k), w2=(E, k, n)."""
    m, k_dim = a.shape
    E, n_eff_x2, k_dim_w1 = w1_q.shape
    E2, k_dim_w2, n_eff = w2_q.shape
    assert E == E2 and k_dim == k_dim_w1 == k_dim_w2
    assert n_eff_x2 == 2 * n_eff
    topk = topk_ids.size(1)

    a_fp32 = a.float()
    out = torch.zeros((m, k_dim), dtype=torch.float32, device=a.device)
    w1_dq = torch.stack(
        [_dequantize_w_NK(w1_q[e].cpu(), w1_scale[e].cpu()) for e in range(E)]
    ).to(a.device)
    w2_dq = torch.stack(
        [_dequantize_w_NK(w2_q[e].cpu(), w2_scale[e].cpu()) for e in range(E)]
    ).to(a.device)

    for i in range(m):
        for kk in range(topk):
            e = int(topk_ids[i, kk].item())
            gate_up = a_fp32[i] @ w1_dq[e].t()
            gate, up = gate_up.chunk(2, dim=-1)
            inter = torch.nn.functional.silu(gate) * up
            down = inter @ w2_dq[e].t()
            out[i] = out[i] + topk_weights[i, kk].float() * down
    return out.to(a.dtype)


def _build_fp8_moe_inputs(
    m: int,
    k_dim: int,
    n_eff: int,
    num_experts: int,
    topk: int,
    device: str,
    seed: int = 0,
):
    """Returns (a, w1_q, w1_scale, w2_q, w2_scale, topk_weights, topk_ids)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(m, k_dim, generator=g, dtype=torch.float32) * 0.5
    a = a.to(device).to(torch.bfloat16)

    # Kernel B layout: (E, N, K) row-major. w1: (E, n*2, k), w2: (E, k, n).
    w1_fp32 = torch.randn(num_experts, 2 * n_eff, k_dim, generator=g) * 0.3
    w2_fp32 = torch.randn(num_experts, k_dim, n_eff, generator=g) * 0.3

    w1_q_l, w1_s_l = [], []
    w2_q_l, w2_s_l = [], []
    for e in range(num_experts):
        wq, ws = _quantize_weight_2d_block_NK(w1_fp32[e])
        w1_q_l.append(wq)
        w1_s_l.append(ws)
        wq, ws = _quantize_weight_2d_block_NK(w2_fp32[e])
        w2_q_l.append(wq)
        w2_s_l.append(ws)
    w1_q = torch.stack(w1_q_l).to(device)
    w1_scale = torch.stack(w1_s_l).to(device)
    w2_q = torch.stack(w2_q_l).to(device)
    w2_scale = torch.stack(w2_s_l).to(device)

    g_dev = torch.Generator(device=device).manual_seed(seed + 1)
    topk_ids = torch.randint(
        0, num_experts, (m, topk), generator=g_dev, dtype=torch.int32, device=device
    )
    topk_weights = torch.softmax(
        torch.randn(m, topk, generator=g, dtype=torch.float32)
        .to(device)
        .to(torch.bfloat16),
        dim=-1,
    )
    return a, w1_q, w1_scale, w2_q, w2_scale, topk_weights, topk_ids


@pytest.mark.skipif(not is_xpu_available(), reason="Intel XPU not available")
@pytest.mark.skipif(
    is_xpu_available() and not is_cri_device(),
    reason="cutlass_fused_experts_fp8 requires CRI (Xe3P) on XPU",
)
class TestCutlassFusedExpertsFp8:
    """E2E accuracy for the sglang-compatible wrapper on XPU."""

    @torch.inference_mode()
    def test_e2e_accuracy_small(self):
        """4 tokens, 4 experts, topk=2 — exercises the flat-2D ragged path."""
        from sgl_kernel import cutlass_fused_experts_fp8

        device = "xpu"
        m, k_dim, n_eff, num_experts, topk = 4, 128, 128, 4, 2

        a, w1_q, w1_scale, w2_q, w2_scale, topk_weights, topk_ids = (
            _build_fp8_moe_inputs(m, k_dim, n_eff, num_experts, topk, device, seed=0)
        )

        # Placeholders; XPU wrapper ignores the CUDA-only stride/ptr args.
        expert_offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
        problem_sizes1 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)
        problem_sizes2 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)
        zeros_stride = torch.zeros(num_experts, dtype=torch.int64, device=device)
        zeros_ptrs = torch.zeros(num_experts, dtype=torch.uint64, device=device)
        workspace = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=device)

        out = cutlass_fused_experts_fp8(
            a=a,
            w1_q=w1_q,
            w2_q=w2_q,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            a1_strides=zeros_stride,
            c1_strides=zeros_stride,
            a2_strides=zeros_stride,
            c2_strides=zeros_stride,
            workspace=workspace,
            a_ptrs=zeros_ptrs,
            b_ptrs=zeros_ptrs,
            out_ptrs=zeros_ptrs,
            a_scales_ptrs=zeros_ptrs,
            b_scales_ptrs=zeros_ptrs,
            expert_offsets=expert_offsets,
            problem_sizes1=problem_sizes1,
            problem_sizes2=problem_sizes2,
        )

        ref = _reference_moe_fp8(
            a, w1_q, w1_scale, w2_q, w2_scale, topk_weights, topk_ids
        )

        assert out.shape == (m, k_dim)
        assert out.dtype == a.dtype

        out_fp32 = out.to(torch.float32).cpu()
        ref_fp32 = ref.to(torch.float32).cpu()
        assert not torch.isnan(out_fp32).any(), "NaN in output"
        assert not torch.isinf(out_fp32).any(), "Inf in output"

        ref_mag = ref_fp32.abs().mean()
        if ref_mag > 1e-4:
            ratio = out_fp32.abs().mean() / ref_mag
            assert 0.5 < ratio < 1.5, (
                f"Magnitude ratio {ratio:.3f} out of range; "
                f"out={out_fp32.abs().mean():.4f}  ref={ref_mag:.4f}"
            )

        if ref_fp32.numel() > 1 and ref_fp32.flatten().std() > 1e-4:
            corr = torch.corrcoef(
                torch.stack([out_fp32.flatten(), ref_fp32.flatten()])
            )[0, 1]
            assert corr > 0.9, f"Correlation {corr:.3f} too low"

    @torch.inference_mode()
    def test_e2e_sglang_call_site_layout(self):
        """Mirrors sglang/srt/layers/quantization/fp8.py:1549-1552 which
        passes .transpose(1, 2)'d weights+scales to the wrapper. Catches the
        layout-mismatch gap that test_e2e_accuracy_small can't (its fixture
        builds tensors directly in the wrapper-expected layout)."""
        from sgl_kernel import cutlass_fused_experts_fp8

        device = "xpu"
        m, k_dim, n_eff, num_experts, topk = 4, 128, 128, 4, 2

        a, w1_q_NK, w1_scale_NK, w2_q_NK, w2_scale_NK, topk_weights, topk_ids = (
            _build_fp8_moe_inputs(m, k_dim, n_eff, num_experts, topk, device, seed=0)
        )
        # Mirror sglang's dispatcher transpose.
        w1_q = w1_q_NK.transpose(1, 2).contiguous()
        w2_q = w2_q_NK.transpose(1, 2).contiguous()
        w1_scale = w1_scale_NK.transpose(1, 2).contiguous()
        w2_scale = w2_scale_NK.transpose(1, 2).contiguous()

        ref = _reference_moe_fp8(
            a, w1_q_NK, w1_scale_NK, w2_q_NK, w2_scale_NK, topk_weights, topk_ids
        )

        expert_offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
        problem_sizes1 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)
        problem_sizes2 = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)
        zeros_stride = torch.zeros(num_experts, dtype=torch.int64, device=device)
        zeros_ptrs = torch.zeros(num_experts, dtype=torch.uint64, device=device)
        workspace = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=device)

        out = cutlass_fused_experts_fp8(
            a=a,
            w1_q=w1_q,
            w2_q=w2_q,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            a1_strides=zeros_stride,
            c1_strides=zeros_stride,
            a2_strides=zeros_stride,
            c2_strides=zeros_stride,
            workspace=workspace,
            a_ptrs=zeros_ptrs,
            b_ptrs=zeros_ptrs,
            out_ptrs=zeros_ptrs,
            a_scales_ptrs=zeros_ptrs,
            b_scales_ptrs=zeros_ptrs,
            expert_offsets=expert_offsets,
            problem_sizes1=problem_sizes1,
            problem_sizes2=problem_sizes2,
        )

        assert out.shape == (m, k_dim)
        out_fp32 = out.to(torch.float32).cpu()
        ref_fp32 = ref.to(torch.float32).cpu()
        assert not torch.isnan(out_fp32).any(), "NaN in output"

        ref_mag = ref_fp32.abs().mean()
        if ref_mag > 1e-4:
            ratio = out_fp32.abs().mean() / ref_mag
            assert 0.5 < ratio < 1.5, (
                f"Magnitude ratio {ratio:.3f} out of range; "
                f"out_mean_abs={out_fp32.abs().mean():.4f}  "
                f"ref_mean_abs={ref_mag:.4f}"
            )
        if ref_fp32.numel() > 1 and ref_fp32.flatten().std() > 1e-4:
            corr = torch.corrcoef(
                torch.stack([out_fp32.flatten(), ref_fp32.flatten()])
            )[0, 1]
            assert corr > 0.9, f"Correlation {corr:.3f} too low"


# ---------------------------------------------------------------------------
# MXFP4 wrapper E2E
# ---------------------------------------------------------------------------

MXFP4_BLOCK_SIZE = 32


def _build_mxfp4_moe_inputs(
    m: int,
    k_dim: int,
    n_eff: int,
    num_experts: int,
    topk: int,
    device: str,
    seed: int = 0,
):
    """Hidden states + MXFP4 weights/scales in kernel layout (E, N, K/2)."""
    from test_mxfp4_blockwise_moe import quantize_to_mxfp4

    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(m, k_dim, generator=g, dtype=torch.float32) * 0.5
    a = a.to(device).to(torch.bfloat16)

    w1_fp32 = torch.randn(num_experts, 2 * n_eff, k_dim, generator=g) * 0.3
    w2_fp32 = torch.randn(num_experts, k_dim, n_eff, generator=g) * 0.3

    w1_q_l, w1_s_l, w2_q_l, w2_s_l = [], [], [], []
    for e in range(num_experts):
        wq, ws = quantize_to_mxfp4(w1_fp32[e])
        w1_q_l.append(wq)
        w1_s_l.append(ws)
        wq, ws = quantize_to_mxfp4(w2_fp32[e])
        w2_q_l.append(wq)
        w2_s_l.append(ws)
    w1_q = torch.stack(w1_q_l).to(device)
    w1_scale = torch.stack(w1_s_l).to(device)
    w2_q = torch.stack(w2_q_l).to(device)
    w2_scale = torch.stack(w2_s_l).to(device)

    g_dev = torch.Generator(device=device).manual_seed(seed + 1)
    topk_ids = torch.randint(
        0, num_experts, (m, topk), generator=g_dev, dtype=torch.int32, device=device
    )
    topk_weights = torch.softmax(
        torch.randn(m, topk, generator=g, dtype=torch.float32)
        .to(device)
        .to(torch.bfloat16),
        dim=-1,
    )
    return a, w1_q, w1_scale, w2_q, w2_scale, topk_weights, topk_ids


def _reference_moe_mxfp4(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_q: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-token, per-topk dequantized fp32 reference."""
    from test_mxfp4_blockwise_moe import dequantize_mxfp4

    m, k_dim = a.shape
    E, n_eff_x2, half_k = w1_q.shape
    E2, k_dim_w2, half_n = w2_q.shape
    assert E == E2 and k_dim == k_dim_w2
    n_eff = half_n * 2
    assert n_eff_x2 == 2 * n_eff
    topk = topk_ids.size(1)

    a_fp32 = a.float()
    out = torch.zeros((m, k_dim), dtype=torch.float32, device=a.device)
    w1_dq = torch.stack(
        [
            dequantize_mxfp4(w1_q[e].cpu(), w1_scale[e].cpu(), torch.float32)
            for e in range(E)
        ]
    ).to(a.device)
    w2_dq = torch.stack(
        [
            dequantize_mxfp4(w2_q[e].cpu(), w2_scale[e].cpu(), torch.float32)
            for e in range(E)
        ]
    ).to(a.device)

    for i in range(m):
        for kk in range(topk):
            e = int(topk_ids[i, kk].item())
            gate_up = a_fp32[i] @ w1_dq[e].t()
            gate, up = gate_up.chunk(2, dim=-1)
            inter = torch.nn.functional.silu(gate) * up
            down = inter @ w2_dq[e].t()
            out[i] = out[i] + topk_weights[i, kk].float() * down
    return out.to(a.dtype)


@pytest.mark.skipif(not is_xpu_available(), reason="Intel XPU not available")
@pytest.mark.skipif(
    is_xpu_available() and not is_cri_device(),
    reason="cutlass_fused_experts_mxfp4 requires CRI (Xe3P) on XPU",
)
class TestCutlassFusedExpertsMxfp4:
    """E2E accuracy for the MXFP4 wrapper."""

    @torch.inference_mode()
    def test_e2e_accuracy_small(self):
        """64 tokens, 2 experts, topk=1, balanced routing -> m_i=32 per expert.
        MXFP4 mainloop faults on m_i < 32; wrapper-side padding for arbitrary
        routing is a follow-up."""
        from sgl_kernel import cutlass_fused_experts_mxfp4

        device = "xpu"
        m, k_dim, n_eff, num_experts, topk = 64, 64, 64, 2, 1

        (a, w1_q, w1_scale, w2_q, w2_scale, topk_weights, topk_ids) = (
            _build_mxfp4_moe_inputs(m, k_dim, n_eff, num_experts, topk, device, seed=0)
        )
        # Force balanced routing: first half -> expert 0, rest -> expert 1.
        topk_ids = torch.zeros((m, topk), dtype=torch.int32, device=device)
        topk_ids[m // 2 :, 0] = 1
        topk_weights = torch.ones((m, topk), dtype=torch.bfloat16, device=device)

        out = cutlass_fused_experts_mxfp4(
            a=a,
            w1_q=w1_q,
            w2_q=w2_q,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        ref = _reference_moe_mxfp4(
            a, w1_q, w1_scale, w2_q, w2_scale, topk_weights, topk_ids
        )

        assert out.shape == (m, k_dim)
        assert out.dtype == a.dtype

        out_fp32 = out.to(torch.float32).cpu()
        ref_fp32 = ref.to(torch.float32).cpu()
        assert not torch.isnan(out_fp32).any(), "NaN in output"
        assert not torch.isinf(out_fp32).any(), "Inf in output"

        # MXFP4 lossier than fp8 -> looser tolerances.
        ref_mag = ref_fp32.abs().mean()
        if ref_mag > 1e-4:
            ratio = out_fp32.abs().mean() / ref_mag
            assert 0.5 < ratio < 1.5, (
                f"Magnitude ratio {ratio:.3f} out of range; "
                f"out={out_fp32.abs().mean():.4f}  ref={ref_mag:.4f}"
            )
        if ref_fp32.numel() > 1 and ref_fp32.flatten().std() > 1e-4:
            corr = torch.corrcoef(
                torch.stack([out_fp32.flatten(), ref_fp32.flatten()])
            )[0, 1]
            assert corr > 0.85, f"Correlation {corr:.3f} too low"
