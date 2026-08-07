import torch


def rand(shape, dtype, scale=1.0):
    return (torch.randn(shape, dtype=torch.float32) * scale).to(
        device="xpu", dtype=dtype
    )


def tol(dtype: torch.dtype):
    if dtype is torch.bfloat16:
        return 3.0e-2, 3.0e-2
    if dtype is torch.float16:
        return 3.0e-3, 3.0e-3
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
