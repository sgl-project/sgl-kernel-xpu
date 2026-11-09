"""Shared helpers of the mono decode kernel tests: random Qwen3.5 / Qwen3.6 MoE
(35B-A3B shapes, one TP2 rank) decoder layers in SGLang's layouts (block-FP8
128x128 weights, FusedMoE [E][N][K] experts, separate K / V buffers, channel
major conv state), a torch reference of a layer step, mailboxes and skip
markers. The 16-bit tensors take a dtype (fp16 or bf16); the same seed gives
the same values rounded to either."""

import socket
import time

import pytest
import torch

HID, NE, INTER, TOPK = 2048, 256, 256, 8  # INTER: per rank (512 / 2)
NKH, NVH, DK, DV = 8, 16, 128, 128  # GDN heads per rank
CONVD = 2 * NKH * DK + NVH * DV  # 4096
QKVZ = CONVD + NVH * DV  # 6144
NQ, HD = 8, 256  # attention q heads per rank, head dim (1 kv head per rank)
QKV_ROWS = NQ * 2 * HD + 2 * HD  # q | gate per head, k, v
QB = 128  # fp8 block size
EPS = 1e-6
THETA = 1.0e7


def _ops():
    import sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel)

    return torch.ops.sgl_kernel


def mono_available(device: int = 0) -> bool:
    try:
        ops = _ops()
    except Exception:
        return False
    return (
        hasattr(ops, "mono_decode_layer")
        and torch.xpu.is_available()
        and torch.xpu.device_count() > device
        and bool(ops.mono_decode_supported(device))
    )


requires_mono = pytest.mark.skipif(
    not mono_available(),
    reason="needs the mono decode kernels and an Xe2 GPU with 32 Xe cores",
)
requires_two_cards = pytest.mark.skipif(
    not (mono_available(0) and mono_available(1)),
    reason="needs two Xe2 GPUs with 32 Xe cores",
)


def mailbox(device) -> torch.Tensor:
    """One layer's mailbox (zeroed): pass its address as both the own and the
    peer mailbox for a single-card loopback (the all-reduces then return twice
    this rank's partial sums)."""
    return torch.zeros(_ops().mono_mbox_bytes(), dtype=torch.uint8, device=device)


def addr(t: torch.Tensor) -> int:
    """Device address of t as the ops' int (int64: USM addresses can be above
    2**63)."""
    p = t.data_ptr()
    return p - (1 << 64) if p >= 1 << 63 else p


def _fp8(shape, gen, device):
    w = torch.randn(shape, generator=gen, dtype=torch.float32) * 64
    return w.clamp_(-448.0, 448.0).to(torch.float8_e4m3fn).to(device)


def _bscale(shape, k, gen, device):
    """Block scales, ~1 / (64 sqrt(K)) each with a spread, fp32."""
    s = (torch.rand(shape, generator=gen) + 0.5) / (64 * k**0.5)
    return s.to(torch.float32).to(device)


def _w16(shape, gen, device, std=1.0, mean=0.0, dtype=torch.float16):
    return (
        (torch.randn(shape, generator=gen, dtype=torch.float32) * std + mean)
        .to(dtype)
        .to(device)
    )


def _lin(n, k, gen, device):
    """fp8 [N, K] linear weight and its [N / 128, K / 128] block scales (stored
    column-major, as SGLang keeps them)."""
    return _fp8((n, k), gen, device), _bscale((k // QB, n // QB), k, gen, device).t()


def common_tensors(gen, device, attn: bool, dtype=torch.float16) -> list:
    out_w, out_s = _lin(HID, NQ * HD if attn else NVH * DV, gen, device)
    sw13, sw13_s = _lin(2 * INTER, HID, gen, device)
    sw2, sw2_s = _lin(HID, INTER, gen, device)
    w16 = lambda shape, std: _w16(shape, gen, device, std, dtype=dtype)  # noqa: E731
    return [
        w16((HID,), 0.1),  # input_layernorm (Gemma: 1 + w)
        w16((HID,), 0.1),  # post_attention_layernorm
        w16((NE, HID), 0.05),  # router
        w16((1, HID), 0.05),  # shared_expert_gate
        out_w,
        out_s,
        _fp8((NE, 2 * INTER, HID), gen, device),  # w13 [E][2I][H] (gate | up)
        _bscale((NE, 2 * INTER // QB, HID // QB), HID, gen, device),
        _fp8((NE, HID, INTER), gen, device),  # w2 [E][H][I]
        _bscale((NE, HID // QB, INTER // QB), INTER, gen, device),
        sw13,
        sw13_s,
        sw2,
        sw2_s,
    ]


def gdn_tensors(gen, device, dtype=torch.float16) -> list:
    """The 21 weights of a GatedDeltaNet layer (mono_register_layer order)."""
    qkvz, qkvz_s = _lin(QKVZ, HID, gen, device)
    a_log = torch.log(torch.rand(NVH, generator=gen) * 15 + 1).to(device)
    return common_tensors(gen, device, False, dtype) + [
        qkvz,
        qkvz_s,
        _w16((2 * NVH, HID), gen, device, 0.05, dtype=dtype),  # in_proj_ba (b | a)
        _w16((CONVD, 1, 4), gen, device, 0.3, dtype=dtype),  # conv1d
        a_log,
        _w16((NVH,), gen, device, 1.0, dtype=dtype),  # dt_bias
        _w16((DV,), gen, device, 0.1, 1.0, dtype=dtype),  # gated norm weight
    ]


def gdn_caches(gen, device, slots, dtype=torch.float16, taps=3):
    conv = _w16((slots, CONVD, taps), gen, device, 0.5, dtype=dtype)
    ssm = (torch.randn((slots, NVH, DV, DK), generator=gen) * 0.05).to(device)
    return conv, ssm


def attn_tensors(gen, device, dtype=torch.float16) -> list:
    """The 18 weights of a full-attention layer (mono_register_layer order)."""
    qkv, qkv_s = _lin(QKV_ROWS, HID, gen, device)
    return common_tensors(gen, device, True, dtype) + [
        qkv,
        qkv_s,
        _w16((HD,), gen, device, 0.1, dtype=dtype),  # q_norm (Gemma: 1 + w)
        _w16((HD,), gen, device, 0.1, dtype=dtype),  # k_norm
    ]


def kv_caches(gen, device, slots, dtype=torch.float16):
    return (
        _w16((slots, 1, HD), gen, device, 1.0, dtype=dtype),
        _w16((slots, 1, HD), gen, device, 1.0, dtype=dtype),
    )


def rows(n, gen, device, dtype=torch.float16):
    """n hidden rows and n residual rows."""
    return (
        _w16((n, HID), gen, device, 1.0, dtype=dtype),
        _w16((n, HID), gen, device, 1.0, dtype=dtype),
    )


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


# ------------------------------------------------------------------ reference
def deq(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Block-FP8 dequantization: w [.., N, K] fp8, s [.., N / 128, K / 128]."""
    s = s.float().repeat_interleave(QB, -2).repeat_interleave(QB, -1)
    return w.float() * s


def _r(x: torch.Tensor, dtype) -> torch.Tensor:
    return x.to(dtype).float()


def _gemma_rms(x, w, dtype):
    x = x.float()
    return _r(
        x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1 + w.float()), dtype
    )


def _rope(x, pos):
    """NeoX RoPE on dims 0..63 of [n, heads, 256]."""
    inv = 1.0 / THETA ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64)
    f = pos.double()[:, None] * inv[None]
    c, s = f.cos().float()[:, None], f.sin().float()[:, None]
    x1, x2 = x[..., :32], x[..., 32:64]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, x[..., 64:]], -1)


def _moe_tail(t, mix, res, dtype, ar):
    """o_proj / out_proj -> all-reduce -> residual -> MoE -> all-reduce.
    mix [n, K] (the mixer output, rounded to dtype); ar(x) the all-reduce of a
    rank's partial. Returns (hid_out, res_out)."""
    o = ar(mix @ deq(t[4], t[5]).t())
    res2 = _r(o + res, dtype)
    x2 = _gemma_rms(res2, t[1], dtype)
    logits = x2 @ t[2].float().t()
    top_v, top_i = logits.topk(TOPK, dim=-1)
    top_w = torch.softmax(top_v, dim=-1)
    sgu = x2 @ deq(t[10], t[11]).t()
    sh = _r(torch.nn.functional.silu(sgu[:, :INTER]) * sgu[:, INTER:], dtype)
    shared = (sh @ deq(t[12], t[13]).t()) * torch.sigmoid(x2 @ t[3].float().t())
    moe = shared.clone()
    for b in range(x2.shape[0]):
        for k in range(TOPK):
            e = int(top_i[b, k])
            gu = x2[b] @ deq(t[6][e], t[7][e]).t()
            h = _r(torch.nn.functional.silu(gu[:INTER]) * gu[INTER:], dtype)
            moe[b] += (h @ deq(t[8][e], t[9][e]).t()) * top_w[b, k]
    return _r(ar(moe), dtype), res2, (logits, top_i)


def cpu(ts):
    return [x.cpu() for x in ts]


def ref_gdn(t, conv, ssm, h, r, slots, dtype, ar):
    """One decode step of a GatedDeltaNet layer for rows h / r (state slots
    `slots`), updating conv / ssm like the kernel (CPU tensors). Returns
    (hid_out, res_out, route info)."""
    res = _r(h.float() + r.float(), dtype)
    x = _gemma_rms(res, t[0], dtype)
    qkvz = _r(x @ deq(t[14], t[15]).t(), dtype)
    ba = _r(x @ t[16].float().t(), dtype)
    cw = t[17].float().view(CONVD, 4)
    core = []
    for b, sl in enumerate(slots):
        sl = int(sl)
        mixed = qkvz[b, :CONVD]
        win = torch.cat([conv[sl, :, :3].float(), mixed[:, None]], dim=1)  # [4096, 4]
        y = _r(torch.nn.functional.silu((win * cw).sum(-1)), dtype)
        conv[sl, :, 0] = conv[sl, :, 1].clone()
        conv[sl, :, 1] = conv[sl, :, 2].clone()
        conv[sl, :, 2] = mixed.to(conv.dtype)
        q = y[: NKH * DK].view(NKH, DK)
        k = y[NKH * DK : 2 * NKH * DK].view(NKH, DK)
        v = y[2 * NKH * DK :].view(NVH, DV)
        q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * DK**-0.5
        k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
        bb, aa = ba[b, :NVH], ba[b, NVH:]
        beta = torch.sigmoid(bb)
        g = torch.exp(
            -torch.exp(t[18].float()) * torch.nn.functional.softplus(aa + t[19].float())
        )
        o = torch.empty(NVH, DV)
        st = ssm[sl].float().clone()
        for hh in range(NVH):
            kk, qq = k[hh // 2], q[hh // 2]
            S = st[hh] * g[hh]
            kv = S @ kk
            delta = (v[hh] - kv) * beta[hh]
            S = S + delta[:, None] * kk[None, :]
            st[hh] = S
            o[hh] = S @ qq
        ssm[sl] = st
        o = _r(o, dtype)
        z = qkvz[b, CONVD:].view(NVH, DV)
        n = (
            o
            * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
            * t[20].float()
            * torch.nn.functional.silu(z)
        )
        core.append(_r(n, dtype).flatten())
    return _moe_tail(t, torch.stack(core), res, dtype, ar)


def ref_attn(t, kc, vc, h, r, page_table, seq_lens, out_loc, pos, page_size, dtype, ar):
    """One decode step of a full-attention layer for rows h / r (CPU tensors):
    request b has seq_lens[b] tokens with the new one (written to
    out_loc[b])."""
    res = _r(h.float() + r.float(), dtype)
    x = _gemma_rms(res, t[0], dtype)
    qkv = x @ deq(t[14], t[15]).t()
    n = x.shape[0]
    qg = qkv[:, : NQ * 2 * HD].view(n, NQ, 2 * HD)
    q, gate = _r(qg[..., :HD], dtype), qg[..., HD:]
    k = _r(qkv[:, NQ * 2 * HD : NQ * 2 * HD + HD], dtype)[:, None]
    v = _r(qkv[:, NQ * 2 * HD + HD :], dtype)
    q = _r(_rope(_gemma_rms(q, t[16], torch.float32), pos), dtype)
    k = _r(_rope(_gemma_rms(k, t[17], torch.float32), pos), dtype)[:, 0]
    outs = []
    for b in range(n):
        L = int(seq_lens[b])
        tk = torch.arange(L - 1)
        sl = page_table[b][tk // page_size].long() * page_size + tk % page_size
        kk = torch.cat([kc[sl, 0].float(), k[b : b + 1]], 0)
        vv = torch.cat([vc[sl, 0].float(), v[b : b + 1]], 0)
        kc[int(out_loc[b]), 0] = k[b].to(kc.dtype)
        vc[int(out_loc[b]), 0] = v[b].to(vc.dtype)
        p = torch.softmax((q[b] @ kk.t()) * HD**-0.5, dim=-1)
        o = p @ vv
        outs.append(_r(o * torch.sigmoid(gate[b]), dtype).flatten())
    return _moe_tail(t, torch.stack(outs), res, dtype, ar)


def loopback_ar(x):
    """The all-reduce of a single-card loopback: twice the rank's partial."""
    return 2 * x


def loopback_ar_fp16(x):
    """The batched kernels' all-reduce in loopback: two fp16-rounded partials."""
    return 2 * x.half().float()


def route_gap(info) -> torch.Tensor:
    """Per row: the gap between the 8th and 9th router logit (a near tie may
    pick another 8th expert)."""
    logits, _ = info
    top = logits.topk(TOPK + 1, dim=-1).values
    return top[:, TOPK - 1] - top[:, TOPK]


def exchange_fd(rank, path, fd):
    """Swaps the dma-buf fds of the two ranks over a unix socket
    (SCM_RIGHTS)."""
    if rank == 0:
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(1)
        conn, _ = srv.accept()
    else:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        for _ in range(600):
            try:
                conn.connect(path)
                break
            except OSError:
                time.sleep(0.05)
    with conn:
        socket.send_fds(conn, [b"f"], [fd])
        _, fds, _, _ = socket.recv_fds(conn, 1, 1)
    if rank == 0:
        srv.close()
    return fds[0]
