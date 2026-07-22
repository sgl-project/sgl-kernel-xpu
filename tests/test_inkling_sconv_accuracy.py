import pytest
import torch


pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Inkling sconv ops are XPU-only",
)


def _rand(shape, dtype, scale=1.0):
    return (torch.randn(shape, dtype=torch.float32) * scale).to(device="xpu", dtype=dtype)


def _tol(dtype: torch.dtype):
    if dtype is torch.bfloat16:
        return 3.0e-2, 3.0e-2
    if dtype is torch.float16:
        return 3.0e-3, 3.0e-3
    return 1.0e-4, 1.0e-4


def _assert_close(actual, expected, dtype=torch.float32):
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(
        actual.detach().cpu().float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
        check_dtype=False,
    )


def _silu(x):
    return x * torch.sigmoid(x)


def _causal_conv1d_ref(
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
                y = _silu(y)
            if use_residual:
                y = y + x[t, d]
            out[t, d] = y
    return out


def _update_cache_ref(x, cache, cache_indices, has_initial_state, query_start_loc):
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


def _fused_decode_ref(
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
                val = _silu(val)
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


def _gather_scatter_ref(hidden_states, cache, track_conv_indices, mask, dst_indices):
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


def _draft_extend_ref(
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
    hidden = hidden_states.detach().cpu().reshape(cache_indices.numel(), draft_token_num, cache.shape[2])
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


def _windows_ref(cache, hidden_states, cache_indices, out_shape, *, batch_size, draft_token_num):
    hidden = hidden_states.detach().cpu().reshape(batch_size, draft_token_num, cache.shape[2])
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("W,D", [(3, 7), (4, 8)])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("use_residual", [False, True])
def test_causal_conv1d_matches_inkling_pr_semantics(dtype, W, D, activation, use_residual):
    from sgl_kernel.inkling_sconv import causal_conv1d

    torch.manual_seed(0)
    B = 3
    lens = [1, 4, 2]
    T = sum(lens)
    x = _rand((T, D), dtype)
    weight = _rand((D, W), dtype, scale=0.2)
    cache = _rand((B, W - 1, D), dtype, scale=0.1)
    cache_mask = torch.tensor([True, False, True], dtype=torch.bool, device="xpu").reshape(B, 1, 1)
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
    expected = _causal_conv1d_ref(
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
    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("use_residual", [False, True])
def test_forward_decode_matches_fused_decode(dtype, activation, use_residual):
    from sgl_kernel.inkling_sconv import causal_conv1d, fused_causal_conv1d_update_decode

    torch.manual_seed(1)
    B, D, W = 5, 8, 4
    x = _rand((B, D), dtype)
    weight = _rand((D, W), dtype, scale=0.2)
    cache = _rand((B, W - 1, D), dtype, scale=0.1)
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
    _assert_close(forward, fused.detach().cpu().float(), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_update_sconv_cache_matches_inkling_pr_semantics(dtype):
    from sgl_kernel.inkling_sconv import update_sconv_cache

    torch.manual_seed(2)
    x = _rand((7, 6), dtype)
    cache = _rand((5, 3, 6), dtype, scale=0.1)
    cache_indices = torch.tensor([0, 1, -1, 3], dtype=torch.int32, device="xpu")
    has_initial_state = torch.tensor([True, False, True, True], dtype=torch.bool, device="xpu")
    query_start_loc = torch.tensor([0, 1, 4, 4, 7], dtype=torch.int32, device="xpu")
    expected = _update_cache_ref(x, cache, cache_indices, has_initial_state, query_start_loc)

    update_sconv_cache(x, cache, cache_indices, has_initial_state, query_start_loc)
    torch.testing.assert_close(cache.detach().cpu(), expected, atol=0, rtol=0, check_dtype=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("W,D", [(3, 7), (4, 8)])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("use_residual", [False, True])
def test_fused_decode_update_matches_inkling_pr_semantics(dtype, W, D, activation, use_residual):
    from sgl_kernel.inkling_sconv import fused_causal_conv1d_update_decode

    torch.manual_seed(3)
    T = 5
    x = _rand((T, D), dtype)
    weight = _rand((D, W), dtype, scale=0.2)
    cache = _rand((9, W - 1, D), dtype, scale=0.1)
    cache_indices = torch.tensor([0, 1, -1, 3, 4], dtype=torch.int32, device="xpu")
    cache_mask = torch.tensor([True, False, False, True, True], dtype=torch.bool, device="xpu")
    track_mask = torch.tensor([True, True, False, False, True], dtype=torch.bool, device="xpu")
    track_indices = torch.tensor([6, 7, 8, 8, 5], dtype=torch.int64, device="xpu")
    expected_y, expected_cache = _fused_decode_ref(
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
    _assert_close(actual, expected_y, dtype)
    torch.testing.assert_close(cache.detach().cpu(), expected_cache, atol=0, rtol=0, check_dtype=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gather_scatter_sconv_cache_matches_inkling_pr_semantics(dtype):
    from sgl_kernel.inkling_sconv import fused_gather_scatter_to_sconv_cache

    torch.manual_seed(4)
    hidden = _rand((12, 6), dtype)
    cache = _rand((8, 3, 6), dtype, scale=0.1)
    track_idx = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=torch.int32, device="xpu")
    mask = torch.tensor([True, False, True, True], dtype=torch.bool, device="xpu")
    dst = torch.tensor([5, 6, -1, 2], dtype=torch.int64, device="xpu")
    expected = _gather_scatter_ref(hidden, cache, track_idx, mask, dst)

    fused_gather_scatter_to_sconv_cache(hidden, cache, track_idx, mask, dst)
    torch.testing.assert_close(cache.detach().cpu(), expected, atol=0, rtol=0, check_dtype=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden_layout", ["btd", "flat"])
def test_draft_extend_sconv_cache_matches_inkling_pr_semantics(dtype, hidden_layout):
    from sgl_kernel.inkling_sconv import fused_draft_extend_sconv_cache

    torch.manual_seed(5)
    B, draft_token_num, D, W1 = 4, 5, 8, 3
    hidden_btd = _rand((B, draft_token_num, D), dtype)
    hidden = hidden_btd if hidden_layout == "btd" else hidden_btd.reshape(B * draft_token_num, D)
    cache = _rand((8, W1, D), dtype, scale=0.1)
    cache_indices = torch.tensor([0, 1, -1, 4], dtype=torch.int32, device="xpu")
    accepted = torch.tensor([0, 2, 3, 5], dtype=torch.int32, device="xpu")
    crossed = torch.tensor([True, True, True, False], dtype=torch.bool, device="xpu")
    track_step = torch.tensor([1, 0, 2, 2], dtype=torch.int32, device="xpu")
    track_dst = torch.tensor([3, 6, 2, 7], dtype=torch.int64, device="xpu")
    expected = _draft_extend_ref(
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
    torch.testing.assert_close(cache.detach().cpu(), expected, atol=0, rtol=0, check_dtype=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden_layout", ["btd", "flat"])
def test_save_intermediate_conv_windows_matches_inkling_pr_semantics(dtype, hidden_layout):
    from sgl_kernel.inkling_sconv import save_intermediate_conv_windows

    torch.manual_seed(6)
    B, draft_token_num, D, W1 = 4, 5, 8, 3
    cache = _rand((8, W1, D), dtype, scale=0.1)
    hidden_btd = _rand((B, draft_token_num, D), dtype)
    hidden = hidden_btd if hidden_layout == "btd" else hidden_btd.reshape(B * draft_token_num, D)
    cache_indices = torch.tensor([0, 2, -1, 5], dtype=torch.int32, device="xpu")
    out = torch.zeros((B, draft_token_num, W1, D), dtype=dtype, device="xpu")
    expected = _windows_ref(
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
    torch.testing.assert_close(out.detach().cpu(), expected, atol=0, rtol=0, check_dtype=False)
