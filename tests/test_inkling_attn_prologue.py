import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Inkling attention prologue ops are XPU-only",
)


def rand(shape, dtype, scale=1.0):
    return (torch.randn(shape, dtype=torch.float32) * scale).to(
        device="xpu", dtype=dtype
    )


def tol(dtype):
    if dtype is torch.bfloat16:
        return 3.0e-2, 3.0e-2
    if dtype is torch.float16:
        return 3.0e-3, 3.0e-3
    return 1.0e-4, 1.0e-4


def assert_close_xpu(actual, expected, dtype):
    atol, rtol = tol(dtype)
    torch.testing.assert_close(
        actual.detach().cpu().float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
        check_dtype=False,
    )


def round_to_dtype(x, dtype):
    return x.to(dtype).float() if dtype is not torch.float32 else x.float()


def silu(x):
    return x * torch.sigmoid(x)


def rmsnorm_ref(x, gamma, eps, dtype):
    x = x.float()
    gamma = gamma.float()
    out = torch.empty_like(x, dtype=torch.float32)
    for h in range(x.shape[1] // 128):
        sl = slice(h * 128, (h + 1) * 128)
        vals = x[:, sl]
        inv = torch.rsqrt(vals.pow(2).mean(dim=1, keepdim=True) + eps)
        out[:, sl] = round_to_dtype(vals * inv * gamma, dtype)
    return out


def conv_prefix_ref(
    packed,
    cache,
    cache_indices,
    cache_mask,
    weight,
    *,
    offset,
    W,
    cu,
    si,
    activation,
    use_residual,
    dtype,
):
    packed = packed.detach().cpu().float()
    cache = cache.detach().cpu().float()
    cache_indices = cache_indices.detach().cpu().int()
    cache_mask = cache_mask.detach().cpu().bool().reshape(-1)
    weight = weight.detach().cpu().float()
    cu = cu.detach().cpu().long()
    si = si.detach().cpu().int()
    T = packed.shape[0]
    D = cache.shape[2]
    out = torch.empty((T, D), dtype=torch.float32)
    for t in range(T):
        seq = int(si[t])
        bos = int(cu[seq])
        slot_raw = int(cache_indices[seq])
        valid = slot_raw != -1
        slot = slot_raw if valid else 0
        gate = bool(valid and cache_mask[seq])
        for d in range(D):
            x_cur = packed[t, offset + d]
            acc = 0.0
            for iw in range(W - 1):
                shifted = t - (W - 1) + iw
                tap = 0.0
                if shifted >= bos:
                    tap = float(packed[shifted, offset + d])
                else:
                    prefix_pos = shifted - bos + (W - 1)
                    if prefix_pos >= 0 and gate:
                        tap = float(cache[slot, prefix_pos, d])
                acc += tap * float(weight[d, iw])
            acc += float(x_cur) * float(weight[d, W - 1])
            val = torch.tensor(acc, dtype=torch.float32)
            if activation in ("silu", "swish"):
                val = silu(val)
            if use_residual:
                val = val + x_cur
            out[t, d] = val
    return out


def verify_windows_ref(packed, cache, cache_indices, draft_tokens, offset):
    packed = packed.detach().cpu()
    cache = cache.detach().cpu()
    cache_indices = cache_indices.detach().cpu().int()
    B = cache_indices.numel()
    W1 = cache.shape[1]
    D = cache.shape[2]
    out = torch.zeros((B, draft_tokens, W1, D), dtype=packed.dtype)
    for b in range(B):
        slot = int(cache_indices[b])
        if slot == -1:
            continue
        bos = b * draft_tokens
        for tq in range(draft_tokens):
            t = bos + tq
            for w in range(W1):
                position = tq + 1 + w
                if position < W1:
                    out[b, tq, w] = cache[slot, position]
                else:
                    g = bos + position - W1
                    out[b, tq, w] = packed[g, offset : offset + D]
    return out


def decode_ref(
    packed,
    cache,
    cache_indices,
    cache_mask,
    weight,
    *,
    offset,
    activation,
    use_residual,
    dtype,
    track_mask=None,
    track_indices=None,
):
    packed = packed.detach().cpu().float()
    out_cache = cache.detach().cpu().clone()
    cache_before = cache.detach().cpu().float()
    cache_indices = cache_indices.detach().cpu().int()
    cache_mask = cache_mask.detach().cpu().bool().reshape(-1)
    weight = weight.detach().cpu().float()
    if track_mask is not None:
        track_mask = track_mask.detach().cpu().bool().reshape(-1)
        track_indices = track_indices.detach().cpu().long()
    T = packed.shape[0]
    W = weight.shape[1]
    D = cache.shape[2]
    y = torch.empty((T, D), dtype=torch.float32)
    for t in range(T):
        slot_raw = int(cache_indices[t])
        valid = slot_raw != -1
        slot = slot_raw if valid else 0
        gate = bool(valid and cache_mask[t])
        for d in range(D):
            x_cur = packed[t, offset + d]
            acc = 0.0
            for iw in range(W - 1):
                if gate:
                    acc += float(cache_before[slot, iw, d]) * float(weight[d, iw])
            acc += float(x_cur) * float(weight[d, W - 1])
            val = torch.tensor(acc, dtype=torch.float32)
            if activation in ("silu", "swish"):
                val = silu(val)
            if use_residual:
                val = val + x_cur
            y[t, d] = val
        if valid:
            updated = torch.empty_like(out_cache[slot])
            for iw in range(W - 1):
                if iw < W - 2:
                    updated[iw] = cache_before[slot, iw + 1] if gate else 0
                else:
                    updated[iw] = packed[t, offset : offset + D].to(cache.dtype)
            out_cache[slot] = updated
            if track_mask is not None and bool(track_mask[t]):
                out_cache[int(track_indices[t])] = updated
    return y, out_cache


def apply_kv_store(
    k_out, v_out, loc, k_buf, v_buf, *, valid_mask=None, dtype=torch.float32
):
    expected_k = k_buf.detach().cpu().clone()
    expected_v = v_buf.detach().cpu().clone()
    loc = loc.detach().cpu().long()
    for t, slot_t in enumerate(loc):
        if valid_mask is not None and not bool(valid_mask[t]):
            continue
        slot = int(slot_t)
        if slot >= 0:
            expected_k[slot] = round_to_dtype(k_out[t], dtype).reshape_as(
                expected_k[slot]
            )
            expected_v[slot] = round_to_dtype(v_out[t], dtype).reshape_as(
                expected_v[slot]
            )
    return expected_k, expected_v


def e8m0_to_float(byte_tensor):
    byte_tensor = byte_tensor.to(torch.int32)
    return torch.ldexp(
        torch.ones_like(byte_tensor, dtype=torch.float32), byte_tensor - 127
    )


def e4m3fn_to_float(byte_tensor):
    byte_tensor = byte_tensor.to(torch.uint8)
    sign = torch.where((byte_tensor & 0x80) != 0, -1.0, 1.0)
    exp = ((byte_tensor >> 3) & 0x0F).to(torch.int32)
    mant = (byte_tensor & 0x07).to(torch.float32)
    out = sign * torch.ldexp(1.0 + mant * 0.125, exp - 7)
    subnormal = exp == 0
    out = torch.where(subnormal, sign * mant * (2.0**-9), out)
    out = torch.where((byte_tensor & 0x7F) == 0, torch.zeros_like(out), out)
    return out


def dequant_q_mxfp8(q_out, q_scale):
    q_bytes = q_out.detach().cpu().view(torch.uint8)
    scale_bytes = q_scale.detach().cpu().view(torch.uint8)
    T, dq = q_bytes.shape
    hq = dq // 128
    values = e4m3fn_to_float(q_bytes.view(T, hq, 4, 32))
    scales = e8m0_to_float(scale_bytes.view(T, hq, 4)).unsqueeze(-1)
    return (values * scales).view(T, dq)


def kv_scale_offset(kv_slot, ch, dkv, page_size):
    hkv = dkv // 128
    page_chunks = page_size // 32
    sf_dim = 4
    page = kv_slot // page_size
    po = kv_slot % page_size
    head = ch // 128
    block = (ch % 128) // 32
    return (
        ((page * hkv + head) * (32 * page_chunks * sf_dim))
        + ((po % 32) * (page_chunks * sf_dim))
        + ((po // 32) * sf_dim)
        + block
    )


def dequant_kv_mxfp8_row(buf, sf, slot, dkv, page_size):
    buf_bytes = buf.detach().cpu().view(torch.uint8).view(-1, dkv)
    sf_bytes = sf.detach().cpu().view(torch.uint8).reshape(-1)
    out = torch.empty((dkv,), dtype=torch.float32)
    for ch in range(0, dkv, 32):
        scale = e8m0_to_float(sf_bytes[kv_scale_offset(slot, ch, dkv, page_size)])
        vals = e4m3fn_to_float(buf_bytes[slot, ch : ch + 32]) * scale
        out[ch : ch + 32] = vals
    return out


def assert_mxfp8_close(actual, expected):
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=8.0e-2,
        rtol=3.0e-1,
        check_dtype=False,
    )


def mxfp8_buffers(slots, dkv, page_size):
    hkv = dkv // 128
    k_buf = torch.empty((slots, hkv, 128), dtype=torch.float8_e4m3fn, device="xpu")
    v_buf = torch.empty_like(k_buf)
    k_buf.view(torch.uint8).fill_(0xA5)
    v_buf.view(torch.uint8).fill_(0x3C)
    sf_shape = (slots // page_size, hkv, 32, page_size // 32, 4)
    sfk = torch.empty(sf_shape, dtype=torch.float8_e8m0fnu, device="xpu")
    sfv = torch.empty_like(sfk)
    sfk.view(torch.uint8).fill_(0x7F)
    sfv.view(torch.uint8).fill_(0x7F)
    return k_buf, v_buf, sfk, sfv


def apply_tau_ref(q_ref, tau, dtype):
    return round_to_dtype(q_ref * tau.detach().cpu().float().view(-1, 1), dtype)


def packed_q_view(T, dq, dkv, dtype):
    gap = 8
    total = dq + gap + dkv + gap + dkv
    packed = rand((T, total), dtype, scale=0.2)
    q = packed[:, :dq]
    return packed, q, 0, dq + gap, dq + gap + dkv


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("dq,dkv", [(128, 128), (384, 256)])
def test_attn_prologue_verify_matches_reference(dtype, activation, dq, dkv):
    from sgl_kernel.inkling_attn_prologue import inkling_attn_prologue_verify

    torch.manual_seed(11)
    B, draft, W = 3, 2, 4
    T = B * draft
    packed, q, q_off, k_off, v_off = packed_q_view(T, dq, dkv, dtype)
    k_cache = rand((8, W - 1, dkv), dtype, scale=0.1)
    v_cache = rand((8, W - 1, dkv), dtype, scale=0.1)
    cache_indices = torch.tensor([0, -1, 2], dtype=torch.int32, device="xpu")
    cache_mask = torch.tensor([True, True, False], dtype=torch.bool, device="xpu")
    k_weight = rand((dkv, W), dtype, scale=0.1)
    v_weight = rand((dkv, W), dtype, scale=0.1)
    q_gamma = rand((128,), dtype, scale=0.2) + 1
    k_gamma = rand((128,), dtype, scale=0.2) + 1
    k_inter = torch.full((B, draft, W - 1, dkv), -3.0, dtype=dtype, device="xpu")
    v_inter = torch.full_like(k_inter, 7.0)
    k_inter_before = k_inter.clone()
    v_inter_before = v_inter.clone()
    loc = torch.tensor([0, 1, -1, 3, 4, 5], dtype=torch.int64, device="xpu")
    k_buf = torch.full((10, dkv // 128, 128), -5.0, dtype=dtype, device="xpu")
    v_buf = torch.full((10, dkv // 128, 128), 9.0, dtype=dtype, device="xpu")

    q_out, k_out, v_out, q_scale = inkling_attn_prologue_verify(
        q,
        k_cache,
        v_cache,
        cache_indices,
        cache_mask,
        k_weight,
        v_weight,
        k_inter,
        v_inter,
        q_gamma,
        k_gamma,
        1.0e-5,
        loc,
        k_buf,
        v_buf,
        q_off,
        k_off,
        v_off,
        dq,
        dkv,
        draft,
        activation=activation,
    )

    cu = torch.arange(B + 1, dtype=torch.int64) * draft
    si = torch.arange(T, dtype=torch.int32) // draft
    q_ref = rmsnorm_ref(
        packed.detach().cpu()[:, q_off : q_off + dq], q_gamma.cpu(), 1.0e-5, dtype
    )
    k_conv = conv_prefix_ref(
        packed,
        k_cache,
        cache_indices,
        cache_mask,
        k_weight,
        offset=k_off,
        W=W,
        cu=cu,
        si=si,
        activation=activation,
        use_residual=True,
        dtype=dtype,
    )
    v_ref = round_to_dtype(
        conv_prefix_ref(
            packed,
            v_cache,
            cache_indices,
            cache_mask,
            v_weight,
            offset=v_off,
            W=W,
            cu=cu,
            si=si,
            activation=activation,
            use_residual=True,
            dtype=dtype,
        ),
        dtype,
    )
    k_ref = rmsnorm_ref(round_to_dtype(k_conv, dtype), k_gamma.cpu(), 1.0e-5, dtype)
    expected_k_buf, expected_v_buf = apply_kv_store(
        k_ref, v_ref, loc, k_buf, v_buf, dtype=dtype
    )

    assert q_scale is None
    assert_close_xpu(q_out, q_ref, dtype)
    assert_close_xpu(k_out, k_ref, dtype)
    assert_close_xpu(v_out, v_ref, dtype)
    valid_seq = cache_indices.cpu() != -1
    expected_k_inter = k_inter_before.cpu()
    expected_v_inter = v_inter_before.cpu()
    expected_k_inter[valid_seq] = verify_windows_ref(
        packed, k_cache, cache_indices, draft, k_off
    )[valid_seq]
    expected_v_inter[valid_seq] = verify_windows_ref(
        packed, v_cache, cache_indices, draft, v_off
    )[valid_seq]
    torch.testing.assert_close(k_inter.cpu(), expected_k_inter)
    torch.testing.assert_close(v_inter.cpu(), expected_v_inter)
    torch.testing.assert_close(k_buf.cpu(), expected_k_buf, check_dtype=False)
    torch.testing.assert_close(v_buf.cpu(), expected_v_buf, check_dtype=False)


def test_attn_prologue_verify_applies_log_tau_without_mxfp8():
    from sgl_kernel.inkling_attn_prologue import inkling_attn_prologue_verify

    torch.manual_seed(21)
    dtype = torch.bfloat16
    B, draft, W, dq, dkv = 2, 2, 4, 128, 128
    T = B * draft
    packed, q, q_off, k_off, v_off = packed_q_view(T, dq, dkv, dtype)
    k_cache = rand((4, W - 1, dkv), dtype, scale=0.1)
    v_cache = rand((4, W - 1, dkv), dtype, scale=0.1)
    cache_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    cache_mask = torch.ones((B,), dtype=torch.bool, device="xpu")
    k_weight = rand((dkv, W), dtype, scale=0.1)
    v_weight = rand((dkv, W), dtype, scale=0.1)
    q_gamma = rand((128,), dtype, scale=0.2) + 1
    k_gamma = rand((128,), dtype, scale=0.2) + 1
    k_inter = torch.empty((B, draft, W - 1, dkv), dtype=dtype, device="xpu")
    v_inter = torch.empty_like(k_inter)
    loc = torch.full((T,), -1, dtype=torch.int64, device="xpu")
    k_buf = torch.empty((4, dkv // 128, 128), dtype=dtype, device="xpu")
    v_buf = torch.empty_like(k_buf)
    tau = torch.tensor([0.5, 1.25, 0.75, 1.5], dtype=torch.float32, device="xpu")

    q_out, _, _, q_scale = inkling_attn_prologue_verify(
        q,
        k_cache,
        v_cache,
        cache_indices,
        cache_mask,
        k_weight,
        v_weight,
        k_inter,
        v_inter,
        q_gamma,
        k_gamma,
        1.0e-5,
        loc,
        k_buf,
        v_buf,
        q_off,
        k_off,
        v_off,
        dq,
        dkv,
        draft,
        do_store=False,
        log_scaling_tau=tau,
    )

    q_ref = rmsnorm_ref(
        packed.detach().cpu()[:, q_off : q_off + dq], q_gamma.cpu(), 1.0e-5, dtype
    )
    assert q_scale is None
    assert_close_xpu(q_out, apply_tau_ref(q_ref, tau, dtype), dtype)


def test_attn_prologue_verify_mxfp8_quantizes_q_and_stores_kv():
    from sgl_kernel.inkling_attn_prologue import inkling_attn_prologue_verify

    torch.manual_seed(22)
    dtype = torch.bfloat16
    B, draft, W, dq, dkv, page_size = 3, 2, 4, 384, 256, 128
    T = B * draft
    packed, q, q_off, k_off, v_off = packed_q_view(T, dq, dkv, dtype)
    k_cache = rand((8, W - 1, dkv), dtype, scale=0.1)
    v_cache = rand((8, W - 1, dkv), dtype, scale=0.1)
    cache_indices = torch.tensor([0, -1, 2], dtype=torch.int32, device="xpu")
    cache_mask = torch.tensor([True, True, False], dtype=torch.bool, device="xpu")
    k_weight = rand((dkv, W), dtype, scale=0.1)
    v_weight = rand((dkv, W), dtype, scale=0.1)
    q_gamma = rand((128,), dtype, scale=0.2) + 1
    k_gamma = rand((128,), dtype, scale=0.2) + 1
    k_inter = torch.empty((B, draft, W - 1, dkv), dtype=dtype, device="xpu")
    v_inter = torch.empty_like(k_inter)
    loc = torch.tensor([0, 1, -1, 3, 4, 5], dtype=torch.int64, device="xpu")
    k_buf, v_buf, sfk, sfv = mxfp8_buffers(page_size, dkv, page_size)
    tau = torch.linspace(0.5, 1.5, T, dtype=torch.float32, device="xpu")

    q_out, k_out, v_out, q_scale = inkling_attn_prologue_verify(
        q,
        k_cache,
        v_cache,
        cache_indices,
        cache_mask,
        k_weight,
        v_weight,
        k_inter,
        v_inter,
        q_gamma,
        k_gamma,
        1.0e-5,
        loc,
        k_buf,
        v_buf,
        q_off,
        k_off,
        v_off,
        dq,
        dkv,
        draft,
        mxfp8_quant=True,
        sfk=sfk,
        sfv=sfv,
        page_size=page_size,
        log_scaling_tau=tau,
    )

    cu = torch.arange(B + 1, dtype=torch.int64) * draft
    si = torch.arange(T, dtype=torch.int32) // draft
    q_ref = rmsnorm_ref(
        packed.detach().cpu()[:, q_off : q_off + dq], q_gamma.cpu(), 1.0e-5, dtype
    )
    q_ref = apply_tau_ref(q_ref, tau, dtype)
    k_conv = conv_prefix_ref(
        packed,
        k_cache,
        cache_indices,
        cache_mask,
        k_weight,
        offset=k_off,
        W=W,
        cu=cu,
        si=si,
        activation=None,
        use_residual=True,
        dtype=dtype,
    )
    v_ref = round_to_dtype(
        conv_prefix_ref(
            packed,
            v_cache,
            cache_indices,
            cache_mask,
            v_weight,
            offset=v_off,
            W=W,
            cu=cu,
            si=si,
            activation=None,
            use_residual=True,
            dtype=dtype,
        ),
        dtype,
    )
    k_ref = rmsnorm_ref(round_to_dtype(k_conv, dtype), k_gamma.cpu(), 1.0e-5, dtype)

    assert q_out.dtype == torch.float8_e4m3fn
    assert q_scale.dtype == torch.float8_e8m0fnu
    assert q_scale.shape == (T, dq // 128, 4)
    assert_mxfp8_close(dequant_q_mxfp8(q_out, q_scale), q_ref)
    assert_close_xpu(k_out, k_ref, dtype)
    assert_close_xpu(v_out, v_ref, dtype)
    for t, slot_t in enumerate(loc.cpu().tolist()):
        if slot_t >= 0:
            assert_mxfp8_close(
                dequant_kv_mxfp8_row(k_buf, sfk, slot_t, dkv, page_size),
                k_ref[t],
            )
            assert_mxfp8_close(
                dequant_kv_mxfp8_row(v_buf, sfv, slot_t, dkv, page_size),
                v_ref[t],
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dq,dkv", [(128, 128), (384, 256)])
def test_attn_prologue_decode_updates_cache_and_store(dtype, dq, dkv):
    from sgl_kernel.inkling_attn_prologue import inkling_attn_prologue_decode

    torch.manual_seed(12)
    T, W = 4, 4
    packed, q, q_off, k_off, v_off = packed_q_view(T, dq, dkv, dtype)
    k_cache = rand((8, W - 1, dkv), dtype, scale=0.1)
    v_cache = rand((8, W - 1, dkv), dtype, scale=0.1)
    k_cache_before = k_cache.clone()
    v_cache_before = v_cache.clone()
    cache_indices = torch.tensor([0, 1, -1, 3], dtype=torch.int32, device="xpu")
    cache_mask = torch.tensor([True, False, True, True], dtype=torch.bool, device="xpu")
    track_mask = torch.tensor([True, False, True, True], dtype=torch.bool, device="xpu")
    track_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64, device="xpu")
    k_weight = rand((dkv, W), dtype, scale=0.1)
    v_weight = rand((dkv, W), dtype, scale=0.1)
    q_gamma = rand((128,), dtype, scale=0.2) + 1
    k_gamma = rand((128,), dtype, scale=0.2) + 1
    loc = torch.tensor([0, -1, 2, 3], dtype=torch.int64, device="xpu")
    k_buf = torch.full((10, dkv // 128, 128), -5.0, dtype=dtype, device="xpu")
    v_buf = torch.full((10, dkv // 128, 128), 9.0, dtype=dtype, device="xpu")

    q_out, k_out, v_out, _ = inkling_attn_prologue_decode(
        q,
        k_cache,
        v_cache,
        cache_indices,
        cache_mask,
        k_weight,
        v_weight,
        q_gamma,
        k_gamma,
        1.0e-5,
        loc,
        k_buf,
        v_buf,
        q_off,
        k_off,
        v_off,
        dq,
        dkv,
        activation="silu",
        track_mask=track_mask,
        track_indices=track_indices,
    )

    q_ref = rmsnorm_ref(
        packed.detach().cpu()[:, q_off : q_off + dq], q_gamma.cpu(), 1.0e-5, dtype
    )
    k_conv, expected_k_cache = decode_ref(
        packed,
        k_cache_before,
        cache_indices,
        cache_mask,
        k_weight,
        offset=k_off,
        activation="silu",
        use_residual=True,
        dtype=dtype,
        track_mask=track_mask,
        track_indices=track_indices,
    )
    v_ref, expected_v_cache = decode_ref(
        packed,
        v_cache_before,
        cache_indices,
        cache_mask,
        v_weight,
        offset=v_off,
        activation="silu",
        use_residual=True,
        dtype=dtype,
        track_mask=track_mask,
        track_indices=track_indices,
    )
    k_ref = rmsnorm_ref(round_to_dtype(k_conv, dtype), k_gamma.cpu(), 1.0e-5, dtype)
    valid = cache_indices.detach().cpu() != -1
    expected_k_buf, expected_v_buf = apply_kv_store(
        k_ref, v_ref, loc, k_buf, v_buf, valid_mask=valid, dtype=dtype
    )

    assert_close_xpu(q_out, q_ref, dtype)
    assert_close_xpu(k_out, k_ref, dtype)
    assert_close_xpu(v_out, round_to_dtype(v_ref, dtype), dtype)
    torch.testing.assert_close(k_cache.cpu(), expected_k_cache, check_dtype=False)
    torch.testing.assert_close(v_cache.cpu(), expected_v_cache, check_dtype=False)
    torch.testing.assert_close(k_buf.cpu(), expected_k_buf, check_dtype=False)
    torch.testing.assert_close(v_buf.cpu(), expected_v_buf, check_dtype=False)


def test_attn_prologue_decode_mxfp8_updates_valid_slots_only():
    from sgl_kernel.inkling_attn_prologue import inkling_attn_prologue_decode

    torch.manual_seed(23)
    dtype = torch.bfloat16
    T, W, dq, dkv, page_size = 4, 4, 128, 128, 128
    packed, q, q_off, k_off, v_off = packed_q_view(T, dq, dkv, dtype)
    k_cache = rand((8, W - 1, dkv), dtype, scale=0.1)
    v_cache = rand((8, W - 1, dkv), dtype, scale=0.1)
    k_cache_before = k_cache.clone()
    v_cache_before = v_cache.clone()
    cache_indices = torch.tensor([0, 1, -1, 3], dtype=torch.int32, device="xpu")
    cache_mask = torch.tensor([True, False, True, True], dtype=torch.bool, device="xpu")
    k_weight = rand((dkv, W), dtype, scale=0.1)
    v_weight = rand((dkv, W), dtype, scale=0.1)
    q_gamma = rand((128,), dtype, scale=0.2) + 1
    k_gamma = rand((128,), dtype, scale=0.2) + 1
    loc = torch.tensor([0, -1, 2, 3], dtype=torch.int64, device="xpu")
    k_buf, v_buf, sfk, sfv = mxfp8_buffers(page_size, dkv, page_size)
    k_buf_before = k_buf.view(torch.uint8).clone()
    tau = torch.tensor([1.0, 0.75, 1.25, 1.5], dtype=torch.float32, device="xpu")

    q_out, k_out, v_out, q_scale = inkling_attn_prologue_decode(
        q,
        k_cache,
        v_cache,
        cache_indices,
        cache_mask,
        k_weight,
        v_weight,
        q_gamma,
        k_gamma,
        1.0e-5,
        loc,
        k_buf,
        v_buf,
        q_off,
        k_off,
        v_off,
        dq,
        dkv,
        mxfp8_quant=True,
        sfk=sfk,
        sfv=sfv,
        page_size=page_size,
        log_scaling_tau=tau,
    )

    q_ref = rmsnorm_ref(
        packed.detach().cpu()[:, q_off : q_off + dq], q_gamma.cpu(), 1.0e-5, dtype
    )
    q_ref = apply_tau_ref(q_ref, tau, dtype)
    k_conv, _ = decode_ref(
        packed,
        k_cache_before,
        cache_indices,
        cache_mask,
        k_weight,
        offset=k_off,
        activation=None,
        use_residual=True,
        dtype=dtype,
    )
    v_ref, _ = decode_ref(
        packed,
        v_cache_before,
        cache_indices,
        cache_mask,
        v_weight,
        offset=v_off,
        activation=None,
        use_residual=True,
        dtype=dtype,
    )
    k_ref = rmsnorm_ref(round_to_dtype(k_conv, dtype), k_gamma.cpu(), 1.0e-5, dtype)
    v_ref = round_to_dtype(v_ref, dtype)

    assert q_scale is not None
    assert_mxfp8_close(dequant_q_mxfp8(q_out, q_scale), q_ref)
    assert_close_xpu(k_out, k_ref, dtype)
    assert_close_xpu(v_out, v_ref, dtype)
    valid = cache_indices.cpu() != -1
    for t, slot_t in enumerate(loc.cpu().tolist()):
        if slot_t >= 0 and bool(valid[t]):
            assert_mxfp8_close(
                dequant_kv_mxfp8_row(k_buf, sfk, slot_t, dkv, page_size),
                k_ref[t],
            )
        elif slot_t >= 0:
            torch.testing.assert_close(
                k_buf.view(torch.uint8)[slot_t].cpu(), k_buf_before[slot_t].cpu()
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dq,dkv", [(128, 128), (384, 256)])
def test_attn_prologue_extend_updates_cache_and_track(dtype, dq, dkv):
    from sgl_kernel.inkling_attn_prologue import inkling_attn_prologue_extend

    torch.manual_seed(13)
    W = 4
    cu = torch.tensor([0, 2, 2, 5], dtype=torch.int64, device="xpu")
    si = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int32, device="xpu")
    T, B = 5, 3
    packed, q, q_off, k_off, v_off = packed_q_view(T, dq, dkv, dtype)
    k_cache = rand((8, W - 1, dkv), dtype, scale=0.1)
    v_cache = rand((8, W - 1, dkv), dtype, scale=0.1)
    k_cache_before = k_cache.clone()
    v_cache_before = v_cache.clone()
    cache_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device="xpu")
    cache_mask = torch.tensor([True, False, True], dtype=torch.bool, device="xpu")
    has_initial_state = torch.tensor(
        [True, False, True], dtype=torch.bool, device="xpu"
    )
    track_rows = torch.tensor(
        [[0, 1, 0], [0, 0, 0], [2, 3, 4]], dtype=torch.int64, device="xpu"
    )
    track_mask = torch.tensor([True, False, True], dtype=torch.bool, device="xpu")
    track_dst = torch.tensor([4, 5, 6], dtype=torch.int64, device="xpu")
    k_weight = rand((dkv, W), dtype, scale=0.1)
    v_weight = rand((dkv, W), dtype, scale=0.1)
    q_gamma = rand((128,), dtype, scale=0.2) + 1
    k_gamma = rand((128,), dtype, scale=0.2) + 1
    loc = torch.arange(T, dtype=torch.int64, device="xpu")
    k_buf = torch.full((10, dkv // 128, 128), -5.0, dtype=dtype, device="xpu")
    v_buf = torch.full((10, dkv // 128, 128), 9.0, dtype=dtype, device="xpu")

    q_out, k_out, v_out, _ = inkling_attn_prologue_extend(
        q,
        k_cache,
        v_cache,
        cache_indices,
        cache_mask,
        has_initial_state,
        cu,
        si,
        k_weight,
        v_weight,
        track_rows,
        track_mask,
        track_dst,
        q_gamma,
        k_gamma,
        1.0e-5,
        loc,
        k_buf,
        v_buf,
        q_off,
        k_off,
        v_off,
        dq,
        dkv,
    )

    q_ref = rmsnorm_ref(
        packed.detach().cpu()[:, q_off : q_off + dq], q_gamma.cpu(), 1.0e-5, dtype
    )
    k_conv = conv_prefix_ref(
        packed,
        k_cache_before,
        cache_indices,
        cache_mask,
        k_weight,
        offset=k_off,
        W=W,
        cu=cu.cpu(),
        si=si.cpu(),
        activation=None,
        use_residual=True,
        dtype=dtype,
    )
    v_ref = round_to_dtype(
        conv_prefix_ref(
            packed,
            v_cache_before,
            cache_indices,
            cache_mask,
            v_weight,
            offset=v_off,
            W=W,
            cu=cu.cpu(),
            si=si.cpu(),
            activation=None,
            use_residual=True,
            dtype=dtype,
        ),
        dtype,
    )
    k_ref = rmsnorm_ref(round_to_dtype(k_conv, dtype), k_gamma.cpu(), 1.0e-5, dtype)
    expected_k_buf, expected_v_buf = apply_kv_store(
        k_ref, v_ref, loc, k_buf, v_buf, dtype=dtype
    )

    expected_k_cache = extend_cache_update_ref(
        packed,
        k_cache_before,
        cache_indices,
        has_initial_state,
        cu,
        track_rows,
        track_mask,
        track_dst,
        k_off,
    )
    expected_v_cache = extend_cache_update_ref(
        packed,
        v_cache_before,
        cache_indices,
        has_initial_state,
        cu,
        track_rows,
        track_mask,
        track_dst,
        v_off,
    )

    assert_close_xpu(q_out, q_ref, dtype)
    assert_close_xpu(k_out, k_ref, dtype)
    assert_close_xpu(v_out, v_ref, dtype)
    torch.testing.assert_close(k_cache.cpu(), expected_k_cache, check_dtype=False)
    torch.testing.assert_close(v_cache.cpu(), expected_v_cache, check_dtype=False)
    torch.testing.assert_close(k_buf.cpu(), expected_k_buf, check_dtype=False)
    torch.testing.assert_close(v_buf.cpu(), expected_v_buf, check_dtype=False)

    k_cache_skip = k_cache_before.clone()
    v_cache_skip = v_cache_before.clone()
    q_skip, k_skip, v_skip, _ = inkling_attn_prologue_extend(
        q,
        k_cache_skip,
        v_cache_skip,
        cache_indices,
        cache_mask,
        has_initial_state,
        cu,
        si,
        k_weight,
        v_weight,
        track_rows,
        track_mask,
        track_dst,
        q_gamma,
        k_gamma,
        1.0e-5,
        loc,
        k_buf,
        v_buf,
        q_off,
        k_off,
        v_off,
        dq,
        dkv,
        do_store=False,
        do_cache_update=False,
        log_scaling_tau=torch.empty(0, dtype=torch.float32, device="xpu"),
    )

    assert_close_xpu(q_skip, q_ref, dtype)
    assert_close_xpu(k_skip, k_ref, dtype)
    assert_close_xpu(v_skip, v_ref, dtype)
    torch.testing.assert_close(
        k_cache_skip.cpu(), k_cache_before.cpu(), check_dtype=False
    )
    torch.testing.assert_close(
        v_cache_skip.cpu(), v_cache_before.cpu(), check_dtype=False
    )


def test_attn_prologue_extend_mxfp8_can_skip_store():
    from sgl_kernel.inkling_attn_prologue import inkling_attn_prologue_extend

    torch.manual_seed(24)
    dtype = torch.bfloat16
    W, dq, dkv, page_size = 4, 128, 128, 128
    cu = torch.tensor([0, 2, 5], dtype=torch.int64, device="xpu")
    si = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32, device="xpu")
    T, B = 5, 2
    packed, q, q_off, k_off, v_off = packed_q_view(T, dq, dkv, dtype)
    k_cache = rand((8, W - 1, dkv), dtype, scale=0.1)
    v_cache = rand((8, W - 1, dkv), dtype, scale=0.1)
    k_cache_before = k_cache.clone()
    v_cache_before = v_cache.clone()
    cache_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    cache_mask = torch.tensor([True, True], dtype=torch.bool, device="xpu")
    has_initial_state = torch.tensor([True, False], dtype=torch.bool, device="xpu")
    track_rows = torch.empty((0,), dtype=torch.int64, device="xpu")
    track_mask = torch.empty((0,), dtype=torch.bool, device="xpu")
    track_dst = torch.empty((0,), dtype=torch.int64, device="xpu")
    k_weight = rand((dkv, W), dtype, scale=0.1)
    v_weight = rand((dkv, W), dtype, scale=0.1)
    q_gamma = rand((128,), dtype, scale=0.2) + 1
    k_gamma = rand((128,), dtype, scale=0.2) + 1
    loc = torch.arange(T, dtype=torch.int64, device="xpu")
    k_buf, v_buf, sfk, sfv = mxfp8_buffers(page_size, dkv, page_size)
    k_buf_before = k_buf.view(torch.uint8).clone()
    v_buf_before = v_buf.view(torch.uint8).clone()

    q_out, k_out, v_out, q_scale = inkling_attn_prologue_extend(
        q,
        k_cache,
        v_cache,
        cache_indices,
        cache_mask,
        has_initial_state,
        cu,
        si,
        k_weight,
        v_weight,
        track_rows,
        track_mask,
        track_dst,
        q_gamma,
        k_gamma,
        1.0e-5,
        loc,
        k_buf,
        v_buf,
        q_off,
        k_off,
        v_off,
        dq,
        dkv,
        do_store=False,
        do_cache_update=False,
        mxfp8_quant=True,
        sfk=sfk,
        sfv=sfv,
        page_size=page_size,
    )

    q_ref = rmsnorm_ref(
        packed.detach().cpu()[:, q_off : q_off + dq], q_gamma.cpu(), 1.0e-5, dtype
    )
    k_conv = conv_prefix_ref(
        packed,
        k_cache_before,
        cache_indices,
        cache_mask,
        k_weight,
        offset=k_off,
        W=W,
        cu=cu.cpu(),
        si=si.cpu(),
        activation=None,
        use_residual=True,
        dtype=dtype,
    )
    v_ref = round_to_dtype(
        conv_prefix_ref(
            packed,
            v_cache_before,
            cache_indices,
            cache_mask,
            v_weight,
            offset=v_off,
            W=W,
            cu=cu.cpu(),
            si=si.cpu(),
            activation=None,
            use_residual=True,
            dtype=dtype,
        ),
        dtype,
    )
    k_ref = rmsnorm_ref(round_to_dtype(k_conv, dtype), k_gamma.cpu(), 1.0e-5, dtype)

    assert q_scale is not None
    assert_mxfp8_close(dequant_q_mxfp8(q_out, q_scale), q_ref)
    assert_close_xpu(k_out, k_ref, dtype)
    assert_close_xpu(v_out, v_ref, dtype)
    torch.testing.assert_close(k_buf.view(torch.uint8).cpu(), k_buf_before.cpu())
    torch.testing.assert_close(v_buf.view(torch.uint8).cpu(), v_buf_before.cpu())
    torch.testing.assert_close(k_cache.cpu(), k_cache_before.cpu(), check_dtype=False)
    torch.testing.assert_close(v_cache.cpu(), v_cache_before.cpu(), check_dtype=False)


def extend_cache_update_ref(
    packed,
    cache,
    cache_indices,
    has_initial_state,
    cu,
    track_rows,
    track_mask,
    track_dst,
    offset,
):
    packed = packed.detach().cpu()
    out = cache.detach().cpu().clone()
    cache_indices = cache_indices.detach().cpu().int()
    has_initial_state = has_initial_state.detach().cpu().bool()
    cu = cu.detach().cpu().long()
    track_rows = track_rows.detach().cpu().long()
    track_mask = track_mask.detach().cpu().bool()
    track_dst = track_dst.detach().cpu().long()
    W1 = out.shape[1]
    D = out.shape[2]
    for b, slot_t in enumerate(cache_indices):
        slot = int(slot_t)
        qlen = int(cu[b + 1] - cu[b])
        if slot != -1 and qlen > 0:
            old = out[slot].clone()
            for w in range(W1):
                if qlen >= W1 - w:
                    row = int(cu[b + 1] - W1 + w)
                    out[slot, w] = packed[row, offset : offset + D]
                elif bool(has_initial_state[b]):
                    out[slot, w] = old[w + qlen]
                else:
                    out[slot, w] = 0
        if bool(track_mask[b]):
            dst = int(track_dst[b])
            if dst >= 0:
                for w in range(W1):
                    row = int(track_rows[b, w])
                    out[dst, w] = packed[row, offset : offset + D]
    return out
