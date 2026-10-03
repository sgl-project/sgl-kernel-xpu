import pytest
import torch
from sgl_kernel import inkling_moe_gate_projection, inkling_moe_gate_topk_renorm

HIDDEN = 6144
ROUTED = 256
SHARED = 2
ACTIVE = ROUTED + SHARED
PADDED = 264
TOPK = 6


def _tolerance(dtype):
    if dtype == torch.bfloat16:
        return 5e-3
    if dtype == torch.float16:
        return 1e-3
    return 2e-4


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="Inkling MoE gate tests require XPU",
)


def _gate_reference(logits, bias, global_scale, route_scale):
    scores = torch.sigmoid(logits.float())
    ranking = scores[:, :ROUTED] + bias
    _, indices = torch.topk(ranking, TOPK, dim=-1, sorted=True)
    routed = scores[:, :ROUTED].gather(1, indices)
    shared = scores[:, ROUTED:ACTIVE]
    scale = (
        route_scale
        * global_scale
        / (routed.sum(dim=1, keepdim=True) + shared.sum(dim=1, keepdim=True))
    )
    return routed * scale, indices.to(torch.int32), shared * scale


def _unpack_routes(packed):
    packed = packed.to(torch.int32)
    indices = packed >> 16
    weight_bits = (packed & 0xFFFF) << 16
    return weight_bits.view(torch.float32), indices


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [0, 1, 3, 17, 127, 512, 513])
def test_projection_matches_fp32_mm(dtype, tokens):
    torch.manual_seed(tokens)
    x = (torch.randn(tokens, HIDDEN, device="xpu") * 0.05).to(dtype)
    weight = (torch.randn(PADDED, HIDDEN, device="xpu") * 0.02).to(dtype)
    actual = inkling_moe_gate_projection(x, weight)
    expected = torch.mm(x, weight.T, out_dtype=torch.float32)[:, :ACTIVE]
    assert actual.shape == (tokens, ACTIVE)
    assert actual.stride() == (PADDED, 1)
    tolerance = _tolerance(dtype)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_projection_rejects_incorrect_padding():
    x = torch.zeros(1, HIDDEN, device="xpu", dtype=torch.bfloat16)
    weight = torch.zeros(ACTIVE, HIDDEN, device="xpu", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="weight must have shape"):
        inkling_moe_gate_projection(x, weight)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_projection_production_shape(dtype):
    torch.manual_seed(4096)
    x = (torch.randn(4096, HIDDEN, device="xpu") * 0.05).to(dtype)
    weight = (torch.randn(PADDED, HIDDEN, device="xpu") * 0.02).to(dtype)
    actual = inkling_moe_gate_projection(x, weight)
    expected = torch.mm(x, weight.T, out_dtype=torch.float32)[:, :ACTIVE]
    tolerance = _tolerance(dtype)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [0, 1, 3, 17, 127, 4096])
@pytest.mark.parametrize("padded_stride", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_topk_matches_reference(dtype, tokens, padded_stride, packed):
    torch.manual_seed(17 + tokens)
    storage_columns = PADDED if padded_stride else ACTIVE
    storage = torch.randn(tokens, storage_columns, device="xpu", dtype=dtype)
    logits = storage[:, :ACTIVE]
    bias = torch.linspace(-0.25, 0.25, ROUTED, device="xpu", dtype=torch.float32)
    global_scale = torch.tensor([1.25], device="xpu", dtype=torch.float32)
    route_scale = 8.0
    output = inkling_moe_gate_topk_renorm(
        logits,
        bias,
        global_scale,
        route_scale,
        return_packed=packed,
    )
    routed, indices, shared, packed_routes = output
    ref_routed, ref_indices, ref_shared = _gate_reference(
        logits, bias, global_scale, route_scale
    )
    if packed:
        routed, indices = _unpack_routes(packed_routes)
        reference_packed = ref_routed.to(torch.bfloat16).float()
        torch.testing.assert_close(routed, reference_packed, rtol=5e-3, atol=5e-3)
    else:
        tolerance = _tolerance(dtype)
        torch.testing.assert_close(routed, ref_routed, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(indices, ref_indices, rtol=0, atol=0)
    tolerance = _tolerance(dtype)
    torch.testing.assert_close(shared, ref_shared, rtol=tolerance, atol=tolerance)


def test_topk_ties_choose_lower_expert_id():
    logits = torch.zeros(3, ACTIVE, device="xpu", dtype=torch.float32)
    bias = torch.zeros(ROUTED, device="xpu", dtype=torch.float32)
    global_scale = torch.ones(1, device="xpu", dtype=torch.float32)
    _, indices, _, _ = inkling_moe_gate_topk_renorm(logits, bias, global_scale, 8.0)
    expected = torch.arange(TOPK, device="xpu", dtype=torch.int32).expand_as(indices)
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)
