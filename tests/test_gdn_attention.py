"""Smoke test for the fused GDN attention op (Intel Xe2 / BMG).

Exercises ``torch.ops.sgl_kernel.gdn_attention`` on synthetic Qwen3-Next/3.5
shaped inputs for both the decode and prefill paths, asserting the in-place
outputs have the expected shapes and are finite. This is a self-contained
smoke test (no vLLM dependency); bit-exact validation vs the vLLM reference op
lives in the analysis benchmark scripts.
"""

import pytest
import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel.gdn_attention
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not hasattr(torch.ops.sgl_kernel, "gdn_attention"),
    reason="Requires Intel XPU build with gdn_attention op",
)

# Canonical Qwen3-Next GDN shape (single TP rank).
NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_WIDTH = 4
TP_SIZE = 1


def _make_inputs(mode, batch_size, seqlen, dtype, device):
    nk, nv, hk, hv, w = NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, CONV_WIDTH
    if mode == "decode":
        num_decodes, num_prefills, num_actual = batch_size, 0, batch_size
        per_seq = torch.ones(batch_size, dtype=torch.int64)
    else:
        num_decodes, num_prefills = 0, batch_size
        num_actual = batch_size * seqlen
        per_seq = torch.full((batch_size,), seqlen, dtype=torch.int64)

    cache_bs = max(64, batch_size * 2)
    qkvz_size = nk * (2 * hk + 2 * hv * nv // nk)
    ba_size = nk * (2 * nv // nk)
    qkv_size = nk * (2 * hk + hv * nv // nk)

    g = torch.Generator(device=device).manual_seed(0)
    rnd = lambda *s: torch.randn(*s, dtype=dtype, device=device, generator=g)

    qkvz = rnd(num_actual, qkvz_size)
    ba = rnd(num_actual, ba_size)
    conv_state = rnd(cache_bs, w - 1, qkv_size)
    ssm_state = rnd(cache_bs, nv, hv, hk)
    conv_w = rnd(qkv_size, w)
    conv_b = rnd(qkv_size)
    A_log = torch.randn(nv, dtype=torch.float32, device=device, generator=g)
    dt_bias = rnd(nv)

    qsl = (
        torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(per_seq, 0)])
        .to(torch.int32)
        .to(device)
    )
    has_init = torch.ones(batch_size, dtype=torch.bool, device=device)
    state_idx = torch.arange(batch_size, dtype=torch.int32, device=device)

    core_attn_out = torch.zeros(num_actual, nv, hv, dtype=dtype, device=device)
    z = torch.empty_like(core_attn_out)

    return dict(
        core_attn_out=core_attn_out,
        z=z,
        qkvz=qkvz,
        ba=ba,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_w=conv_w,
        conv_b=conv_b,
        A_log=A_log,
        dt_bias=dt_bias,
        qsl=qsl,
        has_init=has_init,
        state_idx=state_idx,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_actual=num_actual,
    )


@pytest.mark.parametrize(
    "mode,batch_size,seqlen",
    [("decode", 1, 1), ("decode", 4, 1), ("prefill", 1, 256), ("prefill", 2, 128)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gdn_attention_smoke(mode, batch_size, seqlen, dtype):
    device = torch.device("xpu")
    i = _make_inputs(mode, batch_size, seqlen, dtype, device)

    torch.ops.sgl_kernel.gdn_attention(
        i["core_attn_out"],
        i["z"],
        i["qkvz"],
        i["ba"],
        NUM_K_HEADS,
        NUM_V_HEADS,
        HEAD_K_DIM,
        HEAD_V_DIM,
        i["conv_state"],
        i["ssm_state"],
        i["conv_w"],
        i["conv_b"],
        "silu",
        i["A_log"],
        i["dt_bias"],
        i["num_prefills"],
        i["num_decodes"],
        0,
        i["has_init"],
        i["qsl"],
        None,
        i["state_idx"],
        None,
        None,
        None,
        None,
        i["num_actual"],
        TP_SIZE,
        False,
    )
    torch.xpu.synchronize()

    assert i["core_attn_out"].shape == (i["num_actual"], NUM_V_HEADS, HEAD_V_DIM)
    assert i["z"].shape == i["core_attn_out"].shape
    assert torch.isfinite(i["core_attn_out"].float()).all()
    assert torch.isfinite(i["z"].float()).all()
    assert torch.isfinite(i["ssm_state"].float()).all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
