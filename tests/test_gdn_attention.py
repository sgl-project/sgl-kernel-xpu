"""Smoke test for the fused GDN attention op (Intel Xe2 / BMG).

Exercises ``torch.ops.sgl_kernel.gdn_attention`` on synthetic Qwen3-Next/3.5
shaped inputs for both the decode and prefill paths, asserting the in-place
outputs have the expected shapes and are finite.
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


def _make_inputs(
    mode, batch_size, seqlen, dtype, device, nk=NUM_K_HEADS, nv=NUM_V_HEADS
):
    hk, hv, w = HEAD_K_DIM, HEAD_V_DIM, CONV_WIDTH
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


def _run_op(
    i,
    conv_state,
    ssm_state,
    state_idx,
    reorder_input,
    nk=NUM_K_HEADS,
    nv=NUM_V_HEADS,
    tp_size=TP_SIZE,
):
    torch.ops.sgl_kernel.gdn_attention(
        i["core_attn_out"],
        i["z"],
        i["qkvz"],
        i["ba"],
        nk,
        nv,
        HEAD_K_DIM,
        HEAD_V_DIM,
        conv_state,
        ssm_state,
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
        state_idx,
        None,
        None,
        None,
        None,
        i["num_actual"],
        tp_size,
        reorder_input,
    )
    torch.xpu.synchronize()


@pytest.mark.parametrize(
    "mode,batch_size,seqlen",
    [("decode", 1, 1), ("decode", 4, 1), ("prefill", 1, 256), ("prefill", 2, 128)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gdn_attention_dim_major_conv_layout(mode, batch_size, seqlen, dtype):
    """The kernels index conv_state via explicit strides, so SGLang's native
    ``[cache, dim, width-1]`` pool (passed as a transposed view) must produce the
    same result as the contiguous ``[cache, width-1, dim]`` layout, including
    the in-place updated conv state."""
    device = torch.device("xpu")

    # Reference: contiguous [cache, width-1, dim] layout.
    ref = _make_inputs(mode, batch_size, seqlen, dtype, device)
    conv_orig = ref["conv_state"].clone()  # pristine initial conv state
    ssm_orig = ref["ssm_state"].clone()
    conv_ref = conv_orig.clone()
    ssm_ref = ssm_orig.clone()
    _run_op(ref, conv_ref, ssm_ref, ref["state_idx"], reorder_input=False)
    out_ref = ref["core_attn_out"].clone()
    z_ref = ref["z"].clone()

    # Candidate: SGLang pool [cache, dim, width-1]; pass a transposed view so the
    # op sees logical [cache, width-1, dim] while indexing via strides. Start from
    # the same pristine initial conv/ssm state as the reference.
    cand = _make_inputs(mode, batch_size, seqlen, dtype, device)
    conv_dim_major = conv_orig.transpose(1, 2).contiguous()  # [cache, dim, width-1]
    conv_view = conv_dim_major.transpose(1, 2)  # logical [cache, width-1, dim] view
    ssm_cand = ssm_orig.clone()
    _run_op(cand, conv_view, ssm_cand, cand["state_idx"], reorder_input=False)

    torch.testing.assert_close(cand["core_attn_out"], out_ref, rtol=0, atol=0)
    torch.testing.assert_close(cand["z"], z_ref, rtol=0, atol=0)
    torch.testing.assert_close(ssm_cand, ssm_ref, rtol=0, atol=0)
    # Updated conv state must match after converting the dim-major pool back.
    torch.testing.assert_close(
        conv_dim_major.transpose(1, 2).contiguous(), conv_ref, rtol=0, atol=0
    )


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("reorder_input", [False, True])
@pytest.mark.parametrize(
    "mode,batch_size,seqlen",
    [("decode", 1, 1), ("decode", 4, 1), ("prefill", 1, 256), ("prefill", 2, 128)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gdn_attention_tp_shard_equivalence(
    mode, batch_size, seqlen, reorder_input, tp_size, dtype
):
    """Simulates one TP rank's local head shard on a single GPU and checks
    that calling the op with (GLOBAL head counts, real tp_size) on that
    rank's already-sharded tensors gives bit-identical results to calling it
    with (LOCAL head counts, tp_size=1) on the same tensors.

    This is exactly the calling convention ``forward_fused_gdn`` needs for
    TP>1: ``layer.num_k_heads``/``layer.num_v_heads`` stay the GLOBAL,
    un-sharded config counts, while the actual weight/state/projection
    tensors a rank holds are already local-shard sized (via
    ``ColumnParallelLinear``/``// attn_tp_size`` in ``qwen3_5.py``). If the
    kernel's internal ``num_k_heads / tp_size`` derivation is self-consistent,
    both calls below must produce identical outputs -- this test does NOT
    require multiple GPUs/processes since TP is just a head-count split with
    no cross-rank state dependency in GDN.
    """
    assert NUM_K_HEADS % tp_size == 0 and NUM_V_HEADS % tp_size == 0
    nk_local = NUM_K_HEADS // tp_size
    nv_local = NUM_V_HEADS // tp_size
    device = torch.device("xpu")

    # "Ground truth": treat this rank's shard as if it were the WHOLE model
    # (local head counts, tp_size=1) -- the well-tested TP=1 path above.
    ref = _make_inputs(
        mode, batch_size, seqlen, dtype, device, nk=nk_local, nv=nv_local
    )
    conv_ref = ref["conv_state"].clone()
    ssm_ref = ref["ssm_state"].clone()
    _run_op(
        ref,
        conv_ref,
        ssm_ref,
        ref["state_idx"],
        reorder_input,
        nk=nk_local,
        nv=nv_local,
        tp_size=1,
    )

    # "Candidate": same local-shard tensors, but call with GLOBAL head counts
    # + the real tp_size, exactly as `forward_fused_gdn` would after the
    # TP>1 fix (passing `layer.num_k_heads`/`layer.attn_tp_size` unchanged).
    cand = _make_inputs(
        mode, batch_size, seqlen, dtype, device, nk=nk_local, nv=nv_local
    )
    conv_cand = ref["conv_state"].clone()  # same pristine initial state as ref
    ssm_cand = ref["ssm_state"].clone()
    # Use the identical random tensors as `ref` so only the head-count/tp_size
    # calling convention differs, not the data.
    for key in ("qkvz", "ba", "conv_w", "conv_b", "A_log", "dt_bias"):
        cand[key] = ref[key].clone()
    _run_op(
        cand,
        conv_cand,
        ssm_cand,
        cand["state_idx"],
        reorder_input,
        nk=NUM_K_HEADS,
        nv=NUM_V_HEADS,
        tp_size=tp_size,
    )

    torch.testing.assert_close(
        cand["core_attn_out"], ref["core_attn_out"], rtol=0, atol=0
    )
    torch.testing.assert_close(cand["z"], ref["z"], rtol=0, atol=0)
    torch.testing.assert_close(ssm_cand, ssm_ref, rtol=0, atol=0)
    torch.testing.assert_close(conv_cand, conv_ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode,batch_size,seqlen",
    [("decode", 1, 1), ("decode", 4, 1), ("prefill", 1, 256), ("prefill", 2, 128)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gdn_attention_head_slice_equivalence(mode, batch_size, seqlen, dtype):
    """TP correctness invariant: GDN is fully independent per k/v-head group
    (own conv/ssm state, own weights, no cross-head reduction anywhere in the
    kernels), so running only a *subset* of k-heads (as one TP rank would,
    with its own already-sharded weights/cache) must give bit-identical
    results to running ALL heads on one rank and slicing the output down to
    that same head range. This directly validates the invariant TP relies on
    (each rank computes its own head slice independently), without needing
    multiple GPUs/processes.

    Uses ``reorder_input=True`` (the GQA head-grouped layout actually used in
    production, see ``qwen3_5.py::fused_qkvzba_split_reshape_cat_contiguous``
    and ``gdn_backend.py::forward_fused_gdn``), where ``mixed_qkvz`` is laid
    out as 4 GLOBAL per-component blocks ``[Q_all | K_all | V_all | Z_all]``
    (each internally per-head-contiguous), ``mixed_ba`` as 2 global blocks
    ``[B_all | A_all]``, and ``conv_weights``/``conv_bias``/``conv_states``
    as 3 global blocks ``[Q_all | K_all | V_all]`` (this 3-block convention
    is unconditional, independent of ``reorder_input``). Slicing a
    contiguous k/v-head range therefore requires slicing *within each global
    block* and concatenating, not a single contiguous column slice.
    """
    device = torch.device("xpu")
    nk_total, nv_total, hk, hv, w = (
        NUM_K_HEADS,
        NUM_V_HEADS,
        HEAD_K_DIM,
        HEAD_V_DIM,
        CONV_WIDTH,
    )
    kv_ratio = nv_total // nk_total
    nk_local = nk_total // 2
    nv_local = nk_local * kv_ratio
    k_lo, k_hi = 0, nk_local  # simulate "rank 0" of a TP=2 split
    v_lo, v_hi = k_lo * kv_ratio, k_hi * kv_ratio

    qkvz_dim_per_head = 2 * hk + 2 * hv * kv_ratio
    ba_dim_per_head = 2 * kv_ratio
    qkv_dim_per_head = 2 * hk + hv * kv_ratio

    if mode == "decode":
        num_decodes, num_prefills, num_actual = batch_size, 0, batch_size
        per_seq = torch.ones(batch_size, dtype=torch.int64)
    else:
        num_decodes, num_prefills = 0, batch_size
        num_actual = batch_size * seqlen
        per_seq = torch.full((batch_size,), seqlen, dtype=torch.int64)
    cache_bs = max(64, batch_size * 2)
    qsl = (
        torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(per_seq, 0)])
        .to(torch.int32)
        .to(device)
    )
    has_init = torch.ones(batch_size, dtype=torch.bool, device=device)
    state_idx = torch.arange(batch_size, dtype=torch.int32, device=device)

    g = torch.Generator(device=device).manual_seed(0)
    rnd = lambda *s: torch.randn(*s, dtype=dtype, device=device, generator=g)

    # Full (all-heads) tensors. mixed_qkvz: 4 global blocks [Q_all|K_all|V_all|Z_all],
    # each internally per-head-contiguous. mixed_ba: 2 global blocks [B_all|A_all].
    # conv_weights/conv_bias/conv_states: 3 global blocks [Q_all|K_all|V_all]
    # (unconditional on reorder_input).
    qkvz_full = rnd(num_actual, nk_total * qkvz_dim_per_head)
    ba_full = rnd(num_actual, nk_total * ba_dim_per_head)
    conv_state_full = rnd(cache_bs, w - 1, nk_total * qkv_dim_per_head)
    ssm_state_full = rnd(cache_bs, nv_total, hv, hk)
    conv_w_full = rnd(nk_total * qkv_dim_per_head, w)
    conv_b_full = rnd(nk_total * qkv_dim_per_head)
    A_log_full = torch.randn(nv_total, dtype=torch.float32, device=device, generator=g)
    dt_bias_full = rnd(nv_total)

    def run(nk, nv, qkvz, ba, conv_state, ssm_state, conv_w, conv_b, A_log, dt_bias):
        core_attn_out = torch.zeros(num_actual, nv, hv, dtype=dtype, device=device)
        z = torch.empty_like(core_attn_out)
        torch.ops.sgl_kernel.gdn_attention(
            core_attn_out,
            z,
            qkvz,
            ba,
            nk,
            nv,
            hk,
            hv,
            conv_state,
            ssm_state,
            conv_w,
            conv_b,
            "silu",
            A_log,
            dt_bias,
            num_prefills,
            num_decodes,
            0,
            has_init,
            qsl,
            None,
            state_idx,
            None,
            None,
            None,
            None,
            num_actual,
            1,
            True,
        )
        torch.xpu.synchronize()
        return core_attn_out, z, ssm_state

    # Reference: all heads on one rank, then slice the output.
    out_full, z_full, ssm_full = run(
        nk_total,
        nv_total,
        qkvz_full,
        ba_full,
        conv_state_full.clone(),
        ssm_state_full.clone(),
        conv_w_full,
        conv_b_full,
        A_log_full,
        dt_bias_full,
    )

    # Candidate: only the [k_lo, k_hi) head range, as a TP rank would own.
    def slice_global_blocks(t, dim, block_sizes_full, block_sizes_local, k_scale):
        """Slice `t` along `dim`, where the axis is a concatenation of
        `len(block_sizes_full)` global blocks, each itself per-head
        contiguous with `k_scale` heads-per-block-unit. Keeps the
        [k_lo*k_scale, k_hi*k_scale) sub-range of each block."""
        pieces = []
        offset = 0
        idx = [slice(None)] * t.dim()
        for full_size, per_head in zip(block_sizes_full, block_sizes_local):
            lo = offset + k_lo * per_head
            hi = offset + k_hi * per_head
            idx[dim] = slice(lo, hi)
            pieces.append(t[tuple(idx)])
            offset += full_size
        return torch.cat(pieces, dim=dim).contiguous()

    qkvz_slice = slice_global_blocks(
        qkvz_full,
        1,
        [nk_total * hk, nk_total * hk, nv_total * hv, nv_total * hv],
        [hk, hk, hv * kv_ratio, hv * kv_ratio],
        None,
    )
    ba_slice = slice_global_blocks(
        ba_full, 1, [nv_total, nv_total], [kv_ratio, kv_ratio], None
    )
    conv_state_slice = slice_global_blocks(
        conv_state_full,
        2,
        [nk_total * hk, nk_total * hk, nv_total * hv],
        [hk, hk, hv * kv_ratio],
        None,
    )
    conv_w_slice = slice_global_blocks(
        conv_w_full,
        0,
        [nk_total * hk, nk_total * hk, nv_total * hv],
        [hk, hk, hv * kv_ratio],
        None,
    )
    conv_b_slice = slice_global_blocks(
        conv_b_full,
        0,
        [nk_total * hk, nk_total * hk, nv_total * hv],
        [hk, hk, hv * kv_ratio],
        None,
    )
    ssm_state_slice = ssm_state_full[:, v_lo:v_hi].contiguous()
    A_log_slice = A_log_full[v_lo:v_hi].contiguous()
    dt_bias_slice = dt_bias_full[v_lo:v_hi].contiguous()

    out_cand, z_cand, ssm_cand = run(
        nk_local,
        nv_local,
        qkvz_slice,
        ba_slice,
        conv_state_slice,
        ssm_state_slice,
        conv_w_slice,
        conv_b_slice,
        A_log_slice,
        dt_bias_slice,
    )

    torch.testing.assert_close(out_cand, out_full[:, v_lo:v_hi], rtol=0, atol=0)
    torch.testing.assert_close(z_cand, z_full[:, v_lo:v_hi], rtol=0, atol=0)
    torch.testing.assert_close(ssm_cand, ssm_full[:, v_lo:v_hi], rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
