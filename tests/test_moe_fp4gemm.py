import itertools
from typing import Callable

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel.cutlass_moe import cutlass_fused_experts_mxfp4

MXFP4_BLOCK_SIZE = 32
FLOAT4_E2M1_MAX = 6.0

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


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


def apply_act_and_mul(
    x: torch.Tensor, act_func: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    d = x.shape[-1] // 2
    return act_func(x[..., :d]) * x[..., d:]


def quantize_weight_3d_mxfp4(w: torch.Tensor):
    """(E, N, K) bf16/fp32 -> (packed (E, N, K/2) uint8, scales (E, N, K/32) uint8)."""
    e, n, k = w.shape
    assert k % MXFP4_BLOCK_SIZE == 0
    packed = torch.empty((e, n, k // 2), dtype=torch.uint8, device="cpu")
    scales = torch.empty((e, n, k // MXFP4_BLOCK_SIZE), dtype=torch.uint8, device="cpu")
    for i in range(e):
        p, s = quantize_to_mxfp4(w[i].cpu())
        packed[i] = p
        scales[i] = s
    return packed, scales


def dequantize_weight_3d_mxfp4(
    packed: torch.Tensor, scales: torch.Tensor, dtype=torch.float32
) -> torch.Tensor:
    e = packed.shape[0]
    return torch.stack(
        [dequantize_mxfp4(packed[i].cpu(), scales[i].cpu(), dtype) for i in range(e)]
    )


def torch_naive_moe(
    a,
    w1,
    w2,
    topk_ids,
    topk_weight,
    topk,
    activations="silu",
    routed_scaling_factor=None,
):
    """Per-expert reference: (A @ W1.T) -> silu*mul -> (@ W2.T) -> topk-weighted sum."""
    B, D = a.shape
    a_rep = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    topk_weight_flat = topk_weight.view(-1)
    topk_ids_flat = topk_ids.view(-1)

    act_func = (
        F.silu if activations == "silu" else lambda x: F.gelu(x, approximate="tanh")
    )
    for i in range(w1.shape[0]):
        mask = topk_ids_flat == i
        if mask.sum():
            gemm1 = (a_rep[mask] @ w1[i].transpose(0, 1)).float()
            tmp = apply_act_and_mul(gemm1.to(a.dtype), act_func)
            gemm2 = (tmp @ w2[i].transpose(0, 1)).float()
            out[mask] = gemm2.to(a.dtype)

    result = (
        out.view(B, -1, w2.shape[1]) * topk_weight_flat.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)
    if routed_scaling_factor is not None:
        result = result * routed_scaling_factor
    return result


@pytest.mark.parametrize(
    "num_tokens,topk,num_experts,hidden_size,intermediate_size",
    list(
        itertools.product(
            [64],  # num_tokens
            [1],  # topk
            [2],  # num_experts
            [64],  # hidden_size
            [64],  # intermediate_size
        )
    ),
)
def test_moe_gemm(num_tokens, topk, num_experts, hidden_size, intermediate_size):
    """E2E accuracy for cutlass_fused_experts_mxfp4. Balanced routing keeps
    m_i = num_tokens / num_experts; the MXFP4 mainloop requires m_i >= 32."""
    torch.xpu.manual_seed_all(0)
    device = "xpu"

    a_bf16 = (torch.randn(num_tokens, hidden_size, dtype=torch.float32) * 0.5).to(
        torch.bfloat16
    )
    w1_fp32 = torch.randn(num_experts, 2 * intermediate_size, hidden_size) * 0.3
    w2_fp32 = torch.randn(num_experts, hidden_size, intermediate_size) * 0.3

    w1_q_cpu, w1_scale_cpu = quantize_weight_3d_mxfp4(w1_fp32)
    w2_q_cpu, w2_scale_cpu = quantize_weight_3d_mxfp4(w2_fp32)
    w1_dq = dequantize_weight_3d_mxfp4(w1_q_cpu, w1_scale_cpu, torch.float32).to(
        torch.bfloat16
    )
    w2_dq = dequantize_weight_3d_mxfp4(w2_q_cpu, w2_scale_cpu, torch.float32).to(
        torch.bfloat16
    )

    # Balanced routing: each expert gets exactly num_tokens // num_experts rows.
    assert num_tokens % num_experts == 0
    rows_per_expert = num_tokens // num_experts
    assert rows_per_expert >= 32, "MXFP4 mainloop requires m_i >= 32"
    topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int32)
    for e in range(num_experts):
        topk_ids[e * rows_per_expert : (e + 1) * rows_per_expert, 0] = e
    topk_weights = torch.ones((num_tokens, topk), dtype=torch.bfloat16)

    ref = torch_naive_moe(
        a_bf16, w1_dq, w2_dq, topk_ids, topk_weights, topk, activations="silu"
    )

    a_xpu = a_bf16.to(device)
    w1_q = w1_q_cpu.to(device)
    w1_scale = w1_scale_cpu.to(device)
    w2_q = w2_q_cpu.to(device)
    w2_scale = w2_scale_cpu.to(device)
    topk_ids_xpu = topk_ids.to(device)
    topk_weights_xpu = topk_weights.to(device)

    out = cutlass_fused_experts_mxfp4(
        a=a_xpu,
        w1_q=w1_q,
        w2_q=w2_q,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights_xpu,
        topk_ids=topk_ids_xpu,
    )

    assert out.shape == (num_tokens, hidden_size)
    assert out.dtype == a_xpu.dtype

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
        corr = torch.corrcoef(torch.stack([out_fp32.flatten(), ref_fp32.flatten()]))[
            0, 1
        ]
        assert corr > 0.85, f"Correlation {corr:.3f} too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
