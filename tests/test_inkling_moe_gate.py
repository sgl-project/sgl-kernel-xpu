import pytest
import torch

from sgl_kernel import (
    inkling_moe_gate_gemv,
    inkling_moe_gate_gemv_fused,
    inkling_moe_gate_topk_renorm,
)

HIDDEN = 6144
N_ROUTED = 256
N_SHARED = 2
N_TOTAL = 258
N_PADDED = 264
TOPK = 6
ROUTE_SCALE = 8.0

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="Inkling MoE gate tests require XPU",
)


def _make_inputs(tokens: int, *, seed: int = 0):
    torch.manual_seed(seed)
    device = torch.device("xpu")
    x = (torch.randn((tokens, HIDDEN), device=device) * 0.05).to(torch.bfloat16)
    weight = (torch.randn((N_PADDED, HIDDEN), device=device) * 0.02).to(
        torch.bfloat16
    )
    weight[N_TOTAL:].zero_()
    bias = torch.randn((N_ROUTED,), dtype=torch.float32, device=device) * 0.1
    global_scale = torch.tensor([1.25], dtype=torch.float32, device=device)
    logits = torch.mm(x.float(), weight.float().T)[:, :N_TOTAL]
    return x, weight, bias, global_scale, logits


def _ref_gate(logits: torch.Tensor, bias: torch.Tensor, global_scale: torch.Tensor):
    logits_cpu = logits.detach().cpu().float()
    bias_cpu = bias.detach().cpu().float()
    scale = ROUTE_SCALE * float(global_scale.detach().cpu()[0])
    routed_w = torch.empty((logits_cpu.shape[0], TOPK), dtype=torch.float32)
    indices = torch.empty((logits_cpu.shape[0], TOPK), dtype=torch.int32)
    shared_w = torch.empty((logits_cpu.shape[0], N_SHARED), dtype=torch.float32)

    for row in range(logits_cpu.shape[0]):
        scores = torch.sigmoid(logits_cpu[row, :N_ROUTED]) + bias_cpu
        selected = []
        active = []
        for _ in range(TOPK):
            best_idx = min(
                range(N_ROUTED), key=lambda idx: (-float(scores[idx]), idx)
            )
            selected.append(best_idx)
            active.append(float(torch.sigmoid(logits_cpu[row, best_idx])))
            scores[best_idx] = -float("inf")
        active.extend(
            float(x) for x in torch.sigmoid(logits_cpu[row, N_ROUTED:N_TOTAL])
        )
        active_t = torch.tensor(active, dtype=torch.float32)
        weights = active_t * (scale / float(active_t.sum()))
        routed_w[row] = weights[:TOPK]
        indices[row] = torch.tensor(selected, dtype=torch.int32)
        shared_w[row] = weights[TOPK:]

    return routed_w.to(logits.device), indices.to(logits.device), shared_w.to(
        logits.device
    )


def _unpack(packed: torch.Tensor):
    packed_cpu = packed.detach().cpu().to(torch.int32)
    indices = (packed_cpu >> 16).to(torch.int32)
    weight_bits = (packed_cpu & 0xFFFF) << 16
    weights = weight_bits.view(torch.float32)
    return weights.to(packed.device), indices.to(packed.device)


def _assert_gate_close(out, logits, bias, global_scale, *, packed: bool):
    routed_w, indices, shared_w, packed_out = out
    ref_w, ref_idx, ref_shared = _ref_gate(logits, bias, global_scale)
    if packed:
        routed_w, indices = _unpack(packed_out)
        atol = 2e-2
    else:
        atol = 1e-5
    torch.testing.assert_close(indices, ref_idx, atol=0, rtol=0)
    torch.testing.assert_close(routed_w, ref_w, atol=atol, rtol=1e-3)
    torch.testing.assert_close(shared_w, ref_shared, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("tokens", [0, 1, 3, 8, 17, 64])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("padded_stride", [False, True])
def test_topk_renorm_matches_reference(tokens: int, packed: bool, padded_stride: bool):
    _, _, bias, global_scale, logits = _make_inputs(tokens)
    if padded_stride:
        padded = torch.empty((tokens, N_PADDED), dtype=torch.float32, device="xpu")
        padded[:, :N_TOTAL] = logits
        logits = padded[:, :N_TOTAL]
    out = inkling_moe_gate_topk_renorm(
        logits, bias, global_scale, ROUTE_SCALE, return_packed=packed
    )
    _assert_gate_close(out, logits, bias, global_scale, packed=packed)


def test_topk_renorm_ties_choose_lower_expert_id():
    logits = torch.zeros((2, N_TOTAL), dtype=torch.float32, device="xpu")
    bias = torch.zeros((N_ROUTED,), dtype=torch.float32, device="xpu")
    global_scale = torch.tensor([1.0], dtype=torch.float32, device="xpu")
    _, indices, _, _ = inkling_moe_gate_topk_renorm(
        logits, bias, global_scale, ROUTE_SCALE
    )
    expected = torch.arange(TOPK, dtype=torch.int32, device="xpu").expand_as(indices)
    torch.testing.assert_close(indices, expected, atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [0, 1, 3, 8, 17])
def test_gate_gemv_matches_torch_mm(tokens: int):
    x, weight, _, _, logits_ref = _make_inputs(tokens)
    logits = inkling_moe_gate_gemv(x, weight)
    assert logits.shape == (tokens, N_TOTAL)
    assert logits.stride(0) == N_PADDED
    torch.testing.assert_close(logits, logits_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("tokens", [0, 1, 3, 8, 64])
@pytest.mark.parametrize("packed", [False, True])
def test_gate_gemv_fused_matches_split(tokens: int, packed: bool):
    x, weight, bias, global_scale, _ = _make_inputs(tokens)
    logits = inkling_moe_gate_gemv(x, weight)
    split = inkling_moe_gate_topk_renorm(
        logits, bias, global_scale, ROUTE_SCALE, return_packed=packed
    )
    fused = inkling_moe_gate_gemv_fused(
        x, weight, bias, global_scale, ROUTE_SCALE, return_packed=packed
    )
    for a, b in zip(split, fused):
        if a is not None:
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    _assert_gate_close(fused, logits, bias, global_scale, packed=packed)
