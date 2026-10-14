"""Correctness tests and CPU references for the Inkling SConv operators."""

import pytest
import torch
from sgl_kernel.inkling_sconv import (
    HIS_ONES,
    HIS_PREFIX,
    HIS_SEQ_MINUS_EXT,
    HIS_ZEROS,
    PAD_SLOT_ID,
    fused_decode_sconv_metadata,
    fused_extend_sconv_metadata,
    precompute_helion_decode_metadata,
    precompute_helion_extend_metadata,
)

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Inkling SConv ops are XPU-only",
)


# CPU references shared by the SConv tests.


def rand(shape, dtype, scale=1.0):
    return (torch.randn(shape, dtype=torch.float32) * scale).to(
        device="xpu", dtype=dtype
    )


def tol(dtype: torch.dtype):
    if dtype is torch.bfloat16:
        return 5.0e-3, 5.0e-3
    if dtype is torch.float16:
        return 1.0e-3, 1.0e-3
    return 1.0e-4, 1.0e-4


def assert_close(actual, expected, dtype=torch.float32):
    atol, rtol = tol(dtype)
    torch.testing.assert_close(
        actual.detach().cpu().float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
        check_dtype=False,
    )


def silu(x):
    return x * torch.sigmoid(x)


def causal_conv1d_ref(
    x,
    weight,
    cache,
    cache_mask,
    safe_idx,
    cu,
    si,
    *,
    activation=None,
    use_residual=True,
    is_decode=False,
):
    x = x.detach().cpu().float()
    weight = weight.detach().cpu().float()
    cache = cache.detach().cpu().float()
    cache_mask = cache_mask.detach().cpu().reshape(cache_mask.shape[0], -1)[:, 0].bool()
    safe_idx = safe_idx.detach().cpu().long()
    cu = cu.detach().cpu().long()
    si = si.detach().cpu().int()
    T, D = x.shape
    W = weight.shape[1]
    out = torch.zeros((T, D), dtype=torch.float32)
    for t in range(T):
        seq = int(si[t])
        bos = int(cu[seq])
        slot = int(safe_idx[seq])
        mask = bool(is_decode or cache_mask[seq])
        for d in range(D):
            acc = 0.0
            for iw in range(W):
                shifted = t - (W - 1) + iw
                tap = 0.0
                if shifted >= bos and shifted < T:
                    tap = float(x[shifted, d])
                else:
                    prefix_pos = shifted - bos + (W - 1)
                    if shifted < bos and 0 <= prefix_pos < W - 1 and mask:
                        tap = float(cache[slot, prefix_pos, d])
                acc += tap * float(weight[d, iw])
            y = torch.tensor(acc, dtype=torch.float32)
            if activation in ("silu", "swish"):
                y = silu(y)
            if use_residual:
                y = y + x[t, d]
            out[t, d] = y
    return out


def update_cache_ref(x, cache, cache_indices, has_initial_state, query_start_loc):
    x_cpu = x.detach().cpu()
    out = cache.detach().cpu().clone()
    cache_indices = cache_indices.detach().cpu().int()
    has_initial_state = has_initial_state.detach().cpu().bool()
    query_start_loc = query_start_loc.detach().cpu().int()
    W1 = out.shape[1]
    for b, slot_t in enumerate(cache_indices):
        slot = int(slot_t)
        start = int(query_start_loc[b])
        end = int(query_start_loc[b + 1])
        qlen = end - start
        if slot == -1 or qlen <= 0:
            continue
        old = out[slot] if bool(has_initial_state[b]) else torch.zeros_like(out[slot])
        virtual = torch.cat([old, x_cpu[start:end]], dim=0)
        out[slot] = virtual[-W1:]
    return out


def fused_decode_ref(
    x,
    weight,
    cache,
    cache_indices,
    cache_mask,
    *,
    activation=None,
    use_residual=True,
    track_mask=None,
    track_indices=None,
):
    x_cpu = x.detach().cpu()
    x_f = x_cpu.float()
    weight = weight.detach().cpu().float()
    out_cache = cache.detach().cpu().clone()
    cache_before = cache.detach().cpu().float()
    cache_indices = cache_indices.detach().cpu().int()
    cache_mask = cache_mask.detach().cpu().reshape(-1).bool()
    if track_mask is not None:
        track_mask = track_mask.detach().cpu().reshape(-1).bool()
        track_indices = track_indices.detach().cpu().long()
    T, D = x_f.shape
    W = weight.shape[1]
    y = torch.zeros((T, D), dtype=torch.float32)
    for t in range(T):
        slot = int(cache_indices[t])
        valid = slot != -1
        safe_slot = slot if valid else 0
        mask = bool(cache_mask[t])
        for d in range(D):
            acc = 0.0
            for iw in range(W - 1):
                if mask:
                    acc += float(cache_before[safe_slot, iw, d]) * float(weight[d, iw])
            acc += float(x_f[t, d]) * float(weight[d, W - 1])
            val = torch.tensor(acc, dtype=torch.float32)
            if activation in ("silu", "swish"):
                val = silu(val)
            if use_residual:
                val = val + x_f[t, d]
            y[t, d] = val
        if valid:
            updated = torch.empty_like(out_cache[slot])
            for iw in range(W - 1):
                if iw < W - 2:
                    updated[iw] = cache_before[slot, iw + 1] if mask else 0
                else:
                    updated[iw] = x_cpu[t]
            out_cache[slot] = updated
            if track_mask is not None and bool(track_mask[t]):
                out_cache[int(track_indices[t])] = updated
    return y, out_cache


def gather_scatter_ref(hidden_states, cache, track_conv_indices, mask, dst_indices):
    out = cache.detach().cpu().clone()
    hidden = hidden_states.detach().cpu()
    track_conv_indices = track_conv_indices.detach().cpu().int()
    mask = mask.detach().cpu().bool()
    dst_indices = dst_indices.detach().cpu().long()
    for b in range(mask.numel()):
        dst = int(dst_indices[b])
        if not bool(mask[b]) or dst == -1:
            continue
        for w in range(track_conv_indices.shape[1]):
            out[dst, w] = hidden[int(track_conv_indices[b, w])]
    return out


def draft_extend_ref(
    hidden_states,
    cache,
    cache_indices,
    num_accepted_tokens,
    *,
    draft_token_num,
    do_tracking=False,
    crossed=None,
    track_step=None,
    mamba_track_indices=None,
):
    hidden = (
        hidden_states.detach()
        .cpu()
        .reshape(cache_indices.numel(), draft_token_num, cache.shape[2])
    )
    before = cache.detach().cpu().clone()
    out = before.clone()
    cache_indices = cache_indices.detach().cpu().int()
    num_accepted_tokens = num_accepted_tokens.detach().cpu().int()
    if do_tracking:
        crossed = crossed.detach().cpu().bool()
        track_step = track_step.detach().cpu().int()
        mamba_track_indices = mamba_track_indices.detach().cpu().long()
    W1 = cache.shape[1]
    for b, slot_t in enumerate(cache_indices):
        slot = int(slot_t)
        accepted = int(num_accepted_tokens[b])
        if slot == -1 or accepted < 0:
            continue
        virtual = torch.cat([before[slot], hidden[b]], dim=0)
        if do_tracking and bool(crossed[b]):
            dst = int(mamba_track_indices[b])
            if dst != -1:
                at = int(track_step[b])
                out[dst] = virtual[at : at + W1]
        out[slot] = virtual[accepted : accepted + W1]
    return out


def windows_ref(
    cache, hidden_states, cache_indices, out_shape, *, batch_size, draft_token_num
):
    hidden = (
        hidden_states.detach()
        .cpu()
        .reshape(batch_size, draft_token_num, cache.shape[2])
    )
    cache_cpu = cache.detach().cpu()
    cache_indices = cache_indices.detach().cpu().int()
    out = torch.empty(out_shape, dtype=cache_cpu.dtype)
    W1 = out_shape[2]
    for b, slot_t in enumerate(cache_indices):
        slot = int(slot_t)
        if slot == -1:
            out[b].zero_()
            continue
        virtual = torch.cat([cache_cpu[slot], hidden[b]], dim=0)
        for t in range(draft_token_num):
            out[b, t] = virtual[t + 1 : t + 1 + W1]
    return out


# Causal forward and prefill


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("W,D", [(3, 7), (4, 8)])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("use_residual", [False, True])
def test_causal_conv1d_matches_inkling_pr_semantics(
    dtype, W, D, activation, use_residual
):
    from sgl_kernel.inkling_sconv import causal_conv1d

    torch.manual_seed(0)
    B = 3
    T = 7
    x = rand((T, D), dtype)
    weight = rand((D, W), dtype, scale=0.2)
    cache = rand((B, W - 1, D), dtype, scale=0.1)
    cache_mask = torch.tensor(
        [True, False, True], dtype=torch.bool, device="xpu"
    ).reshape(B, 1, 1)
    safe_idx = torch.arange(B, dtype=torch.int64, device="xpu")
    cu = torch.tensor([0, 1, 5, 7], dtype=torch.int64, device="xpu")
    si = torch.tensor([0, 1, 1, 1, 1, 2, 2], dtype=torch.int32, device="xpu")

    actual = causal_conv1d(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation=activation,
        use_residual=use_residual,
    )
    expected = causal_conv1d_ref(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation=activation,
        use_residual=use_residual,
    )
    assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "lengths,D",
    [
        ([1, 128, 1, 145], 17),  # Ragged sequences and non-vector channels.
        ([64, 64, 64, 64], 64),  # Prefill dispatch threshold with four sequences.
        ([0, 1, 255, 1], 31),  # Empty sequence and a tail past the threshold.
    ],
)
def test_causal_conv1d_rolling_window_boundaries_and_tails(dtype, lengths, D):
    """Cover prefill blocks, packed-sequence resets, cache masks, and tails."""
    from sgl_kernel.inkling_sconv import causal_conv1d

    torch.manual_seed(1)
    B, T, W = len(lengths), sum(lengths), 4
    x = rand((T, D), dtype)
    weight = rand((D, W), dtype, scale=0.2)
    cache = rand((B, W - 1, D), dtype, scale=0.1)
    cache_mask = torch.tensor(
        [True, False, True, False], dtype=torch.bool, device="xpu"
    ).reshape(B, 1, 1)
    safe_idx = torch.tensor([3, 1, 0, 2], dtype=torch.int64, device="xpu")
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int64,
        device="xpu",
    )
    si = torch.repeat_interleave(
        torch.arange(B, dtype=torch.int32, device="xpu"),
        torch.tensor(lengths, dtype=torch.int64, device="xpu"),
    )

    actual = causal_conv1d(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation="silu",
        use_residual=True,
    )
    expected = causal_conv1d_ref(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation="silu",
        use_residual=True,
    )
    assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "T,D",
    [
        (255, 32),  # Below the prefill specialization threshold.
        (256, 32),  # Small channels use the block path.
        (257, 32),  # One-token tail on the block path.
        (255, 40),  # D=40 still uses the block path below T=256.
        (256, 40),  # First shape eligible for the packed path.
        (257, 40),  # One-token tail on the packed path.
        (259, 40),  # Tail with several channel groups.
        (260, 36),  # Both dtypes use the block path below D=40.
        (256, 34),  # Both dtypes fall back for unaligned channels.
    ],
)
def test_causal_conv1d_vector_prefill_tail(dtype, T, D):
    """Check the packed and block paths around the prefill dispatch boundary."""
    from sgl_kernel.inkling_sconv import causal_conv1d

    torch.manual_seed(3)
    W = 4
    x = rand((T, D), dtype)
    weight = rand((D, W), dtype, scale=0.2)
    cache = rand((1, W - 1, D), dtype, scale=0.1)
    cache_mask = torch.ones((1, 1, 1), dtype=torch.bool, device="xpu")
    safe_idx = torch.zeros(1, dtype=torch.int64, device="xpu")
    cu = torch.tensor([0, T], dtype=torch.int64, device="xpu")
    si = torch.zeros(T, dtype=torch.int32, device="xpu")

    actual = causal_conv1d(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation="silu",
        use_residual=True,
    )
    expected = causal_conv1d_ref(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation="silu",
        use_residual=True,
    )
    assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D", [32, 40, 6144])
def test_causal_conv1d_inkling_prefill_shape(dtype, D):
    """Exercise long single-sequence prefills around the vector D boundary."""
    from sgl_kernel.inkling_sconv import causal_conv1d

    torch.manual_seed(2)
    T, W = 4096, 4
    x = rand((T, D), dtype)
    weight = rand((D, W), dtype, scale=0.2)
    cache = rand((1, W - 1, D), dtype, scale=0.1)
    cache_mask = torch.ones((1, 1, 1), dtype=torch.bool, device="xpu")
    safe_idx = torch.zeros(1, dtype=torch.int64, device="xpu")
    cu = torch.tensor([0, T], dtype=torch.int64, device="xpu")
    si = torch.zeros(T, dtype=torch.int32, device="xpu")

    actual = causal_conv1d(
        x,
        weight,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation="silu",
        use_residual=True,
    )

    # A vectorized FP32 reference makes checking every output
    # practical, including cache-prefixed rows and every tail/channel tile.
    x_float = x.float()
    weight_float = weight.float()
    cache_float = cache[0].float()
    acc = torch.zeros_like(x_float)
    for iw in range(W):
        offset = iw - (W - 1)
        if offset < 0:
            prefix_len = -offset
            acc[:prefix_len] += cache_float[W - 1 - prefix_len :] * weight_float[:, iw]
            acc[prefix_len:] += x_float[: T - prefix_len] * weight_float[:, iw]
        else:
            acc += x_float * weight_float[:, iw]
    expected = torch.nn.functional.silu(acc) + x_float

    assert_close(actual, expected.cpu(), dtype)


# Decode metadata

# cross the BLOCK=1024 grid boundary and hit odd sizes
DECODE_BATCH_SIZES = [1, 2, 3, 17, 64, 160, 257, 1023, 1024, 1025]


def _reference(B: int, cache_indices: torch.Tensor):
    device = cache_indices.device
    query_start_loc = torch.arange(B + 1, dtype=torch.int32, device=device)
    has_initial_state = torch.ones(B, dtype=torch.bool, device=device)
    precomputed = precompute_helion_decode_metadata(
        B=B, W=4, cache_indices=cache_indices, has_initial_state=has_initial_state
    )
    return query_start_loc, has_initial_state, precomputed


def _decode_metadata_out(B: int, T: int):
    return {
        "query_start_loc": torch.empty(B + 1, dtype=torch.int32, device="xpu"),
        "has_initial_state": torch.empty(B, dtype=torch.bool, device="xpu"),
        "cache_mask": torch.empty((B, 1, 1), dtype=torch.bool, device="xpu"),
        "safe_idx": torch.empty(B, dtype=torch.int64, device="xpu"),
        "cu": torch.empty(B + 1, dtype=torch.int64, device="xpu"),
        "si": torch.empty(T, dtype=torch.int32, device="xpu"),
    }


def _assert_decode_uses_out(got, out):
    for got_tensor, out_tensor in (
        (got[0], out["query_start_loc"]),
        (got[1], out["has_initial_state"]),
        (got[2]["cache_mask"], out["cache_mask"]),
        (got[2]["safe_idx"], out["safe_idx"]),
        (got[2]["cu"], out["cu"]),
        (got[2]["si"], out["si"]),
    ):
        assert got_tensor.data_ptr() == out_tensor.data_ptr()


@pytest.mark.parametrize("b", DECODE_BATCH_SIZES)
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_matches_unfused(b: int, idx_dtype: torch.dtype):
    torch.manual_seed(b)
    cache_indices = torch.randint(0, 4096, (b,), dtype=idx_dtype, device="xpu")
    # sprinkle PAD slots (cudagraph padding lanes)
    pad = torch.rand(b, device="xpu") < 0.25
    cache_indices[pad] = PAD_SLOT_ID

    ref_qsl, ref_his, ref_meta = _reference(b, cache_indices)
    qsl, his, meta = fused_decode_sconv_metadata(B=b, cache_indices=cache_indices)

    for tag, got, ref in (
        ("query_start_loc", qsl, ref_qsl),
        ("has_initial_state", his, ref_his),
        ("cache_mask", meta["cache_mask"], ref_meta["cache_mask"]),
        ("safe_idx", meta["safe_idx"], ref_meta["safe_idx"]),
        ("cu", meta["cu"], ref_meta["cu"]),
        ("si", meta["si"], ref_meta["si"]),
    ):
        assert got.dtype == ref.dtype, (tag, got.dtype, ref.dtype)
        assert got.shape == ref.shape, (tag, got.shape, ref.shape)
        assert torch.equal(got, ref), tag


def test_decode_writes_graph_static_outputs():
    b = 17
    cache_indices = torch.tensor(
        [0, 1, PAD_SLOT_ID, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        dtype=torch.int32,
        device="xpu",
    )
    out = _decode_metadata_out(b, b)
    got = fused_decode_sconv_metadata(B=b, cache_indices=cache_indices, out=out)

    _assert_decode_uses_out(got, out)
    ref = _reference(b, cache_indices)
    for got_tensor, ref_tensor in (
        (got[0], ref[0]),
        (got[1], ref[1]),
        (got[2]["cache_mask"], ref[2]["cache_mask"]),
        (got[2]["safe_idx"], ref[2]["safe_idx"]),
        (got[2]["cu"], ref[2]["cu"]),
        (got[2]["si"], ref[2]["si"]),
    ):
        assert torch.equal(got_tensor, ref_tensor)


def test_all_pad():
    cache_indices = torch.full((8,), PAD_SLOT_ID, dtype=torch.int32, device="xpu")
    _, _, meta = fused_decode_sconv_metadata(B=8, cache_indices=cache_indices)
    assert not meta["cache_mask"].any()
    assert (meta["safe_idx"] == 0).all()


# Extend metadata

# cross si tiles (BLOCK_T=256) and the single-tile B bound
EXTEND_BATCH_SIZES = [1, 2, 7, 64, 257, 1023]


def _ref_extend(B, extend_seq_lens, his_mode, his_src, cache_indices, T):
    device = cache_indices.device
    query_start_loc = torch.zeros(B + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = extend_seq_lens.cumsum(dim=0)
    if his_mode == HIS_ZEROS:
        has_initial_state = torch.zeros(B, dtype=torch.bool, device=device)
    elif his_mode == HIS_PREFIX:
        has_initial_state = his_src > 0
    else:  # HIS_SEQ_MINUS_EXT
        has_initial_state = (his_src[:B] - extend_seq_lens) > 0
    meta = precompute_helion_extend_metadata(
        B=B,
        T=T,
        W=4,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        query_start_loc=query_start_loc,
    )
    return query_start_loc, has_initial_state, meta


def _ref_verify(B, draft_token_num, cache_indices):
    device = cache_indices.device
    query_start_loc = torch.arange(
        0, (B + 1) * draft_token_num, draft_token_num, dtype=torch.int32, device=device
    )
    has_initial_state = torch.ones(B, dtype=torch.bool, device=device)
    meta = precompute_helion_extend_metadata(
        B=B,
        T=B * draft_token_num,
        W=4,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        query_start_loc=query_start_loc,
    )
    return query_start_loc, has_initial_state, meta


def _assert_equal(got, ref):
    for tag, g, r in (
        ("query_start_loc", got[0], ref[0]),
        ("has_initial_state", got[1], ref[1]),
        ("cache_mask", got[2]["cache_mask"], ref[2]["cache_mask"]),
        ("safe_idx", got[2]["safe_idx"], ref[2]["safe_idx"]),
        ("cu", got[2]["cu"], ref[2]["cu"]),
        ("si", got[2]["si"], ref[2]["si"]),
    ):
        assert g.dtype == r.dtype, (tag, g.dtype, r.dtype)
        assert g.shape == r.shape, (tag, g.shape, r.shape)
        assert torch.equal(g, r), tag


def _cache_indices(b, idx_dtype):
    ci = torch.randint(0, 4096, (b,), dtype=idx_dtype, device="xpu")
    pad = torch.rand(b, device="xpu") < 0.25
    ci[pad] = PAD_SLOT_ID
    return ci


def _extend_metadata_out(B, T):
    return {
        "query_start_loc": torch.empty(B + 1, dtype=torch.int32, device="xpu"),
        "has_initial_state": torch.empty(B, dtype=torch.bool, device="xpu"),
        "cache_mask": torch.empty((B, 1, 1), dtype=torch.bool, device="xpu"),
        "safe_idx": torch.empty(B, dtype=torch.int64, device="xpu"),
        "cu": torch.empty(B + 1, dtype=torch.int64, device="xpu"),
        "si": torch.empty(T, dtype=torch.int32, device="xpu"),
    }


def _assert_extend_uses_out(got, out):
    for got_tensor, out_tensor in (
        (got[0], out["query_start_loc"]),
        (got[1], out["has_initial_state"]),
        (got[2]["cache_mask"], out["cache_mask"]),
        (got[2]["safe_idx"], out["safe_idx"]),
        (got[2]["cu"], out["cu"]),
        (got[2]["si"], out["si"]),
    ):
        assert got_tensor.data_ptr() == out_tensor.data_ptr()


@pytest.mark.parametrize("b", EXTEND_BATCH_SIZES)
@pytest.mark.parametrize("his_mode", [HIS_ZEROS, HIS_PREFIX, HIS_SEQ_MINUS_EXT])
@pytest.mark.parametrize("lens_dtype", [torch.int32, torch.int64])
def test_extend_matches_unfused(b, his_mode, lens_dtype):
    torch.manual_seed(b * 10 + his_mode)
    lens = torch.randint(0, 33, (b,), dtype=lens_dtype, device="xpu")
    lens[torch.rand(b, device="xpu") < 0.2] = 0  # zero-length sequences
    T = int(lens.sum().item())
    cache_indices = _cache_indices(b, torch.int32)
    if his_mode == HIS_PREFIX:
        his_src = torch.randint(0, 3, (b,), dtype=lens_dtype, device="xpu")
    elif his_mode == HIS_SEQ_MINUS_EXT:
        his_src = lens + torch.randint(0, 2, (b,), dtype=lens_dtype, device="xpu")
    else:
        his_src = None

    ref = _ref_extend(b, lens, his_mode, his_src, cache_indices, T)
    got = fused_extend_sconv_metadata(
        B=b,
        T=T,
        cache_indices=cache_indices,
        his_mode=his_mode,
        extend_seq_lens=lens,
        his_src=his_src,
    )
    assert got is not None
    _assert_equal(got, ref)


def test_extend_writes_graph_static_outputs():
    b = 7
    lens = torch.tensor([3, 0, 4, 1, 2, 0, 5], dtype=torch.int32, device="xpu")
    cache_indices = torch.tensor(
        [0, PAD_SLOT_ID, 2, 3, 4, 5, 6], dtype=torch.int32, device="xpu"
    )
    his_src = lens + 1
    T = int(lens.sum().item())
    out = _extend_metadata_out(b, T)
    got = fused_extend_sconv_metadata(
        B=b,
        T=T,
        cache_indices=cache_indices,
        his_mode=HIS_SEQ_MINUS_EXT,
        extend_seq_lens=lens,
        his_src=his_src,
        out=out,
    )

    assert got is not None
    _assert_extend_uses_out(got, out)
    _assert_equal(
        got, _ref_extend(b, lens, HIS_SEQ_MINUS_EXT, his_src, cache_indices, T)
    )


@pytest.mark.parametrize("b", EXTEND_BATCH_SIZES)
@pytest.mark.parametrize("draft_token_num", [1, 9])
def test_verify_matches_unfused(b, draft_token_num):
    torch.manual_seed(b)
    cache_indices = _cache_indices(b, torch.int64)
    ref = _ref_verify(b, draft_token_num, cache_indices)
    got = fused_extend_sconv_metadata(
        B=b,
        T=b * draft_token_num,
        cache_indices=cache_indices,
        his_mode=HIS_ONES,
        draft_token_num=draft_token_num,
    )
    assert got is not None
    _assert_equal(got, ref)


def test_cu_not_spanning_T():
    """Dummy capture sequences: cu stops short of T; trailing si rows clamp to
    B-1 exactly like the reference's searchsorted + clamp."""
    b = 5
    lens = torch.tensor([3, 0, 4, 0, 2], dtype=torch.int64, device="xpu")
    T = int(lens.sum().item()) + 17
    cache_indices = _cache_indices(b, torch.int32)
    seq_lens = lens + 1
    ref = _ref_extend(b, lens, HIS_SEQ_MINUS_EXT, seq_lens, cache_indices, T)
    got = fused_extend_sconv_metadata(
        B=b,
        T=T,
        cache_indices=cache_indices,
        his_mode=HIS_SEQ_MINUS_EXT,
        extend_seq_lens=lens,
        his_src=seq_lens,
    )
    assert got is not None
    _assert_equal(got, ref)


def test_fallback_past_batch_bound():
    b = 1024  # > _FUSED_EXTEND_MAX_B
    lens = torch.ones(b, dtype=torch.int64, device="xpu")
    got = fused_extend_sconv_metadata(
        B=b,
        T=b,
        cache_indices=_cache_indices(b, torch.int32),
        his_mode=HIS_ZEROS,
        extend_seq_lens=lens,
    )
    assert got is None


# Fused decode and cache update


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("use_residual", [False, True])
def test_forward_decode_matches_fused_decode(dtype, activation, use_residual):
    from sgl_kernel.inkling_sconv import (
        causal_conv1d,
        fused_causal_conv1d_update_decode,
    )

    torch.manual_seed(1)
    B, D, W = 5, 8, 4
    x = rand((B, D), dtype)
    weight = rand((D, W), dtype, scale=0.2)
    cache = rand((B, W - 1, D), dtype, scale=0.1)
    cache_mask = torch.ones(B, dtype=torch.bool, device="xpu")
    cache_indices = torch.arange(B, dtype=torch.int32, device="xpu")
    safe_idx = cache_indices.to(torch.int64)
    cu = torch.arange(B + 1, dtype=torch.int64, device="xpu")
    si = torch.arange(B, dtype=torch.int32, device="xpu")

    forward = causal_conv1d(
        x,
        weight,
        cache,
        cache_mask.reshape(B, 1, 1),
        safe_idx,
        cu,
        si,
        activation=activation,
        use_residual=use_residual,
        is_decode=True,
    )
    fused_cache = cache.clone()
    fused = fused_causal_conv1d_update_decode(
        x,
        weight,
        fused_cache,
        cache_indices,
        cache_mask,
        activation=activation,
        use_residual=use_residual,
    )
    assert_close(forward, fused.detach().cpu().float(), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "W,D",
    [(3, 7), (4, 8), (4, 512), (4, 520), (4, 513)],
)
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("use_residual", [False, True])
def test_fused_decode_update_matches_inkling_pr_semantics(
    dtype, W, D, activation, use_residual
):
    from sgl_kernel.inkling_sconv import fused_causal_conv1d_update_decode

    torch.manual_seed(3)
    T = 5
    x = rand((T, D), dtype)
    weight = rand((D, W), dtype, scale=0.2)
    cache = rand((9, W - 1, D), dtype, scale=0.1)
    cache_indices = torch.tensor([0, 1, -1, 3, 4], dtype=torch.int32, device="xpu")
    cache_mask = torch.tensor(
        [True, False, False, True, True], dtype=torch.bool, device="xpu"
    )
    track_mask = torch.tensor(
        [True, True, False, False, True], dtype=torch.bool, device="xpu"
    )
    track_indices = torch.tensor([6, 7, 8, 8, 5], dtype=torch.int64, device="xpu")
    expected_y, expected_cache = fused_decode_ref(
        x,
        weight,
        cache,
        cache_indices,
        cache_mask,
        activation=activation,
        use_residual=use_residual,
        track_mask=track_mask,
        track_indices=track_indices,
    )

    actual = fused_causal_conv1d_update_decode(
        x,
        weight,
        cache,
        cache_indices,
        cache_mask,
        activation=activation,
        use_residual=use_residual,
        track_mask=track_mask,
        track_indices=track_indices,
    )
    assert_close(actual, expected_y, dtype)
    torch.testing.assert_close(
        cache.detach().cpu(), expected_cache, atol=0, rtol=0, check_dtype=False
    )


# Gather, scatter, and draft extension


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gather_scatter_sconv_cache_matches_inkling_pr_semantics(dtype):
    from sgl_kernel.inkling_sconv import fused_gather_scatter_to_sconv_cache

    torch.manual_seed(4)
    hidden = rand((12, 6), dtype)
    cache = rand((8, 3, 6), dtype, scale=0.1)
    track_idx = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=torch.int32, device="xpu"
    )
    mask = torch.tensor([True, False, True, True], dtype=torch.bool, device="xpu")
    dst = torch.tensor([5, 6, -1, 2], dtype=torch.int64, device="xpu")
    expected = gather_scatter_ref(hidden, cache, track_idx, mask, dst)

    fused_gather_scatter_to_sconv_cache(hidden, cache, track_idx, mask, dst)
    torch.testing.assert_close(
        cache.detach().cpu(), expected, atol=0, rtol=0, check_dtype=False
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden_layout", ["btd", "flat"])
def test_draft_extend_sconv_cache_matches_inkling_pr_semantics(dtype, hidden_layout):
    from sgl_kernel.inkling_sconv import fused_draft_extend_sconv_cache

    torch.manual_seed(5)
    B, draft_token_num, D, W1 = 4, 5, 8, 3
    hidden_btd = rand((B, draft_token_num, D), dtype)
    hidden = (
        hidden_btd
        if hidden_layout == "btd"
        else hidden_btd.reshape(B * draft_token_num, D)
    )
    cache = rand((8, W1, D), dtype, scale=0.1)
    cache_indices = torch.tensor([0, 1, -1, 4], dtype=torch.int32, device="xpu")
    accepted = torch.tensor([0, 2, 3, 5], dtype=torch.int32, device="xpu")
    crossed = torch.tensor([True, True, True, False], dtype=torch.bool, device="xpu")
    track_step = torch.tensor([1, 0, 2, 2], dtype=torch.int32, device="xpu")
    track_dst = torch.tensor([3, 6, 2, 7], dtype=torch.int64, device="xpu")
    expected = draft_extend_ref(
        hidden,
        cache,
        cache_indices,
        accepted,
        draft_token_num=draft_token_num,
        do_tracking=True,
        crossed=crossed,
        track_step=track_step,
        mamba_track_indices=track_dst,
    )

    fused_draft_extend_sconv_cache(
        hidden,
        cache,
        cache_indices,
        num_accepted_tokens=accepted,
        draft_token_num=draft_token_num,
        do_tracking=True,
        crossed=crossed,
        track_step=track_step,
        mamba_track_indices=track_dst,
    )
    torch.testing.assert_close(
        cache.detach().cpu(), expected, atol=0, rtol=0, check_dtype=False
    )


# Intermediate windows


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden_layout", ["btd", "flat"])
def test_save_intermediate_conv_windows_matches_inkling_pr_semantics(
    dtype, hidden_layout
):
    from sgl_kernel.inkling_sconv import save_intermediate_conv_windows

    torch.manual_seed(6)
    B, draft_token_num, D, W1 = 4, 5, 8, 3
    cache = rand((8, W1, D), dtype, scale=0.1)
    hidden_btd = rand((B, draft_token_num, D), dtype)
    hidden = (
        hidden_btd
        if hidden_layout == "btd"
        else hidden_btd.reshape(B * draft_token_num, D)
    )
    cache_indices = torch.tensor([0, 2, -1, 5], dtype=torch.int32, device="xpu")
    out = torch.zeros((B, draft_token_num, W1, D), dtype=dtype, device="xpu")
    expected = windows_ref(
        cache,
        hidden,
        cache_indices,
        out.shape,
        batch_size=B,
        draft_token_num=draft_token_num,
    )

    save_intermediate_conv_windows(
        cache,
        hidden,
        cache_indices,
        out,
        batch_size=B,
        draft_token_num=draft_token_num,
    )
    torch.testing.assert_close(
        out.detach().cpu(), expected, atol=0, rtol=0, check_dtype=False
    )


# Cache update


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_update_sconv_cache_matches_inkling_pr_semantics(dtype):
    from sgl_kernel.inkling_sconv import update_sconv_cache

    torch.manual_seed(2)
    x = rand((7, 6), dtype)
    cache = rand((5, 3, 6), dtype, scale=0.1)
    cache_indices = torch.tensor([0, 1, -1, 3], dtype=torch.int32, device="xpu")
    has_initial_state = torch.tensor(
        [True, False, True, True], dtype=torch.bool, device="xpu"
    )
    query_start_loc = torch.tensor([0, 1, 4, 4, 7], dtype=torch.int32, device="xpu")
    expected = update_cache_ref(
        x, cache, cache_indices, has_initial_state, query_start_loc
    )

    update_sconv_cache(x, cache, cache_indices, has_initial_state, query_start_loc)
    torch.testing.assert_close(
        cache.detach().cpu(), expected, atol=0, rtol=0, check_dtype=False
    )
