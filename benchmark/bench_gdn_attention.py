"""Benchmark sgl_kernel.gdn_attention (fused SYCL XE2 kernel) for Intel GPU.

Covers the full Gated-DeltaNet (GDN) forward pass:
  causal_conv1d  →  SiLU  →  Q/K l2-norm  →  g/β gating
  →  chunk (prefill) or recurrent (decode) delta rule  →  output gate

Default shapes match Qwen3.5-9B single-layer at TP=1:
  linear_num_key_heads=16, linear_num_value_heads=32,
  linear_key_head_dim=128, linear_value_head_dim=128,
  linear_conv_kernel_dim=4.

Workload sweep matches the Qwen3.5-9B serving spec BS=(1,4),
input_len=(1024,4096), output_len=1024 (see the ``workloads`` list in
``main()`` for the exact prefill/decode shape cross-product and why decode
only needs one shape per batch size).

Optionally compares against SGLang Triton kernels (requires `sglang` in path);
prints a warning and skips comparison if not available.

Run:
  ZE_AFFINITY_MASK=0 python benchmark/bench_gdn_attention.py
  ZE_AFFINITY_MASK=0 python benchmark/bench_gdn_attention.py --iters 100
  ZE_AFFINITY_MASK=0 python benchmark/bench_gdn_attention.py --dtype fp16 --iters 50
"""

import argparse
import gc
import math
import random
import warnings

# ── SYCL op (required) ──────────────────────────────────────────────────────
import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel.gdn_attention
import torch

# ── SGLang Triton pipeline (optional) ───────────────────────────────────────
_SGLANG_AVAILABLE = False
_SGLANG_MISS_REASON = ""

try:
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.mamba.causal_conv1d_triton import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )

    try:
        from sglang.srt.hardware_backend.xpu.kernels.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )
    except Exception:
        from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )

    _SGLANG_AVAILABLE = True
except ImportError as _e:
    _SGLANG_MISS_REASON = str(_e)

# ── Model shape (Qwen3.5-9B, TP=1) ──────────────────────────────────────────
# From text_config in the HF config:
#   linear_num_key_heads=16, linear_num_value_heads=32,
#   linear_key_head_dim=128, linear_value_head_dim=128,
#   linear_conv_kernel_dim=4
NK = 16  # linear_num_key_heads
NV = 32  # linear_num_value_heads
HK = 128  # linear_key_head_dim
HV = 128  # linear_value_head_dim
W = 4  # linear_conv_kernel_dim (causal conv width)
TP = 1  # tensor-parallel size

# Derived dimensions
QKV_DIM = NK * (2 * HK + HV * NV // NK)  # 8192  conv channels
QKVZ_DIM = NK * (2 * HK + 2 * HV * NV // NK)  # 12288 fused projection
BA_DIM = NK * (2 * NV // NK)  # 64    gating projection

DEVICE = "xpu"


# ── Input construction ───────────────────────────────────────────────────────


def make_inputs(mode: str, batch_size: int, seqlen: int, dtype: torch.dtype):
    """Create tensors for one GDN layer call.

    For decode: ``seqlen`` is ignored (always 1 token per request).
    For prefill: ``seqlen`` tokens per sequence.

    Returns ``(kwargs, meta)`` where ``kwargs`` is passed directly to
    ``torch.ops.sgl_kernel.gdn_attention``.
    """
    if mode == "decode":
        num_decodes, num_prefills = batch_size, 0
        n_tok = batch_size  # one new token per sequence
        per_seq = torch.ones(batch_size, dtype=torch.int64)
    elif mode == "prefill":
        num_decodes, num_prefills = 0, batch_size
        n_tok = batch_size * seqlen
        per_seq = torch.full((batch_size,), seqlen, dtype=torch.int64)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    cache_size = max(256, batch_size * 2)
    g = torch.Generator(device=DEVICE).manual_seed(42)
    rnd = lambda *s: torch.randn(*s, dtype=dtype, device=DEVICE, generator=g)
    rndf = lambda *s: torch.randn(*s, dtype=torch.float32, device=DEVICE, generator=g)

    qkvz = rnd(n_tok, QKVZ_DIM)
    ba = rnd(n_tok, BA_DIM)
    conv_state = rnd(cache_size, W - 1, QKV_DIM)  # [cache, width-1, dim]
    ssm_state = rndf(cache_size, NV, HV, HK)
    conv_w = rnd(QKV_DIM, W)
    conv_b = rnd(QKV_DIM)
    A_log = rndf(NV)
    dt_bias = rnd(NV)

    qsl = (
        torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(per_seq, 0)])
        .to(torch.int32)
        .to(DEVICE)
    )
    has_init = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)
    state_idx = torch.arange(batch_size, dtype=torch.int32, device=DEVICE)

    core_out = torch.zeros(n_tok, NV, HV, dtype=dtype, device=DEVICE)
    z = torch.empty_like(core_out)

    kwargs = dict(
        core_attn_out=core_out,
        z=z,
        projected_states_qkvz=qkvz,
        projected_states_ba=ba,
        num_k_heads=NK,
        num_v_heads=NV,
        head_k_dim=HK,
        head_v_dim=HV,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=conv_w,
        conv_bias=conv_b,
        activation="silu",
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_spec_decodes=0,
        has_initial_state=has_init,
        non_spec_query_start_loc=qsl,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=state_idx,
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=n_tok,
        tp_size=TP,
        reorder_input=False,
    )
    meta = dict(
        mode=mode,
        batch_size=batch_size,
        seqlen=seqlen,
        n_tok=n_tok,
        per_seq=per_seq,
    )
    return kwargs, meta


# ── FLOPs / bytes models ─────────────────────────────────────────────────────
#
# Ported from LMProj, which model each piece of the GDN pipeline as a separate
# graph node (conv1d+SiLU, Q/K L2-norm, GDNAttentionNode core).
# Two adjustments for how fusion changes memory traffic:
#   - L2-norm's read/write is NOT separate HBM traffic here (Q/K stay in
#     registers between conv and the delta-rule GEMMs), so we add its FLOPs
#     but not extra bytes.
#   - conv1d's output (post-conv, pre-gate activations) likewise never
#     round-trips through HBM; only its persistent conv_state does.

GDN_CHUNK_SIZE = 64  # matches FLA's chunk-wise prefill kernel default


def _gdn_decode_flops(batch: int) -> int:
    """GDN core delta-rule FLOPs for one decode step (batch tokens), assuming
    pre-normalized Q/K. Mirrors ``lmproj.ir.nodes.gdn_decode_flops``: 3 small
    GEMMs (v_predict, state update, output) + gating/elementwise (softplus,
    sigmoid, state decay, etc.)."""
    hk, hv, qk_d, v_d = NK, NV, HK, HV
    gemm_ops = batch * hv * (2 * v_d * qk_d) * 3  # v_predict, S_delta, output
    non_gemm_ops = (
        batch * hk * qk_d  # scale query
        + batch * hv  # x = a + dt_bias
        + batch * hv * 3  # softplus
        + hv * 2
        + batch * hv  # g = -exp(A_log) * softplus_x
        + batch * hv  # alpha = exp(g)
        + batch * hv  # beta = sigmoid(b)
        + batch * hv * v_d * qk_d  # S *= alpha
        + batch * hv * v_d  # v_error = v - v_predict
        + batch * hv * v_d  # v_error *= beta
        + batch * hv * v_d * qk_d  # S += S_delta
    )
    return gemm_ops + non_gemm_ops


def _gdn_prefill_flops(
    batch: int, seq_len: int, chunk_size: int = GDN_CHUNK_SIZE
) -> int:
    """GDN core delta-rule FLOPs for a chunk-wise prefill, assuming
    pre-normalized Q/K. Mirrors
    ``lmproj.ir.nodes.gdn_prefill_flops_breakdown``: per chunk of
    ``chunk_size`` tokens, 8 intra/inter-chunk GEMMs (build decay matrix,
    triangular solve, pseudo U/W, state scan, inter+intra output) plus the
    elementwise ops (decay masks, exp, cumsum) around them. Grid is over
    V-heads (gates are per-V-head), so every term scales with ``NV``.
    """
    Kd, Vd = HK, HV
    C = chunk_size
    n_chunks = max(1, (seq_len + C - 1) // C)

    gemm_per_chunk = (
        2 * C * Kd * C  # build A:      K K^T
        + 2 * C * C * Vd  # pseudo U:     T (beta V)
        + 2 * C * C * Kd  # pseudo W:     T (beta exp(gamma) K)
        + 2 * C * Kd * Vd  # scan:         W S^T
        + 2 * Kd * C * Vd  # scan:         S += K^T V_new
        + 2 * C * Kd * Vd  # output inter: (exp(gamma) Q) S^T
        + 2 * C * Kd * C  # output intra: Q K^T
        + 2 * C * C * Vd  # output intra: tril(...) V_new
    )
    solve_flops_per_chunk = (2.0 / 3.0) * C**3  # tri-solve T = (I+A)^-1

    non_gemm_per_chunk = (
        C  # cumsum(g)
        + 5 * C * C  # build A: sub + exp + decay-mul + beta-mul + tril-mask
        + C  # exp(gamma) for chunk
        + C * Vd  # beta ⊙ V
        + 2 * C * Kd  # beta ⊙ K, * exp(gamma)
        + C * Vd  # V_new = U - W S^T
        + 2 * C  # decay-normalize factor
        + C * Vd  # V_new *= decay
        + 2 * Kd * Vd  # S *= decay, S += K^T V_new
        + C  # exp(gamma) q-rows
        + C * Vd  # o *= exp(gamma)
        + 3 * C * C  # sub + exp + decay-mul (intra)
        + C * C  # tril mask (intra)
        + 3 * C * Vd  # o*scale + intra*scale
    )

    gemm_flops = batch * NV * n_chunks * (gemm_per_chunk + solve_flops_per_chunk)
    non_gemm_flops = batch * NV * n_chunks * non_gemm_per_chunk
    return int(gemm_flops + non_gemm_flops)


def estimate_flops(
    mode: str,
    n_tok: int,
    batch_size: int,
    seqlen: int,
    chunk_size: int = GDN_CHUNK_SIZE,
) -> int:
    """Approximate arithmetic FLOPs for one fused GDN forward pass.

    (1) causal_conv1d + SiLU: ``2·T·qkv_dim·width`` (MAC per tap) +
        ``4·T·qkv_dim`` (SiLU), matching ``CausalConv1DNode``.
    (2) Q/K L2-norm: ``2 · 3·T·(nk·hk)``, matching ``L2NormNode`` (applied
        once each to Q and K).
    (3) delta-rule core: chunk-wise GEMM+solve+elementwise for prefill
        (``_gdn_prefill_flops``, matches ``gdn_prefill_flops_breakdown``)
        or the single-step recurrent update for decode
        (``_gdn_decode_flops``, matches ``gdn_decode_flops``).
    """
    conv_flops = 2 * n_tok * QKV_DIM * W + 4 * n_tok * QKV_DIM
    l2norm_flops = 2 * (3 * n_tok * NK * HK)
    if mode == "decode":
        core_flops = _gdn_decode_flops(batch_size)
    elif mode == "prefill":
        core_flops = _gdn_prefill_flops(batch_size, seqlen, chunk_size)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return conv_flops + l2norm_flops + core_flops


def estimate_bytes(
    mode: str,
    n_tok: int,
    batch_size: int,
    seqlen: int,
    bpe: int,
    bpe_ssm: int = 4,
    chunk_size: int = GDN_CHUNK_SIZE,
) -> int:
    """Approximate bytes moved for one fused GDN forward pass.

    Accounts for:
      • Input read : qkvz + ba  (n_tok tokens; includes the Q/K/V/Z + a/b
        slices — L2-norm and conv don't add extra HBM traffic since they're
        fused in registers between the input read and the core kernel)
      • Output write: core_attn_out + z  (n_tok tokens)
      • Conv state  : R + W  (batch × (W-1) × qkv_dim), once per call
      • Conv weights + bias (loaded once)
      • A_log/dt_bias gate weights (loaded once per token-batch)
      • SSM state   : R + W, float32. For decode this happens once per call
        (one step). For prefill the chunk-wise algorithm reads/writes the
        recurrent state once per **chunk boundary**, not once per call —
        ``n_chunks = ceil(seqlen / chunk_size)`` round-trips per sequence,
        matching ``proj_gdn_prefill_latency_s``.
    """
    input_bytes = n_tok * (QKVZ_DIM + BA_DIM) * bpe
    output_bytes = n_tok * NV * HV * bpe * 2  # core_out + z
    conv_state_bytes = batch_size * (W - 1) * QKV_DIM * bpe * 2  # R + W
    conv_weight_bytes = (QKV_DIM * W + QKV_DIM) * bpe  # weights + bias
    gate_weight_bytes = NV * 2 * 4  # A_log (fp32) + dt_bias, loaded once

    if mode == "decode":
        n_chunks = 1
    elif mode == "prefill":
        n_chunks = max(1, (seqlen + chunk_size - 1) // chunk_size)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    state_bytes_per_roundtrip = NV * HV * HK * bpe_ssm
    ssm_state_bytes = batch_size * n_chunks * state_bytes_per_roundtrip * 2  # R + W

    return (
        input_bytes
        + output_bytes
        + conv_state_bytes
        + ssm_state_bytes
        + conv_weight_bytes
        + gate_weight_bytes
    )


# ── Timing helper ────────────────────────────────────────────────────────────


def time_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    """Return median wall time in milliseconds using XPU events."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    times = []
    for _ in range(iters):
        s = torch.xpu.Event(enable_timing=True)
        e = torch.xpu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.xpu.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]  # median


# ── SYCL runner ──────────────────────────────────────────────────────────────


def run_sycl(kwargs):
    torch.ops.sgl_kernel.gdn_attention(**kwargs)


# ── Triton pipeline (optional) ───────────────────────────────────────────────


def _split_qkvzba(kwargs, meta):
    """Decompose projected_states_qkvz/ba → mixed_qkv, a, b (for Triton)."""
    n_tok = meta["n_tok"]
    rep = NV // NK  # v-heads per k-head

    qkvz = kwargs["projected_states_qkvz"].reshape(n_tok, NK, 2 * HK + 2 * rep * HV)
    q, k, v, _ = torch.split(qkvz, [HK, HK, rep * HV, rep * HV], dim=-1)
    mixed_qkv = torch.cat(
        [
            q.reshape(n_tok, NK * HK),
            k.reshape(n_tok, NK * HK),
            v.reshape(n_tok, NV * HV),
        ],
        dim=-1,
    ).contiguous()

    ba = kwargs["projected_states_ba"].reshape(n_tok, NK, 2 * rep)
    b, a = torch.split(ba, [rep, rep], dim=-1)
    b = b.reshape(n_tok, NV).contiguous()
    a = a.reshape(n_tok, NV).contiguous()
    return mixed_qkv, a, b


def sglang_pipeline(kwargs, meta, conv_state_clone, ssm_state_clone):
    """Run the SGLang Triton GDN pipeline.  Returns (output, stage_fns_dict)."""
    n_tok = meta["n_tok"]
    qsl = kwargs["non_spec_query_start_loc"]
    cache_idx = kwargs["non_spec_state_indices_tensor"]
    A_log = kwargs["A_log"]
    dt_bias = kwargs["dt_bias"]
    conv_w = kwargs["conv_weights"]
    conv_b = kwargs["conv_bias"]
    scale = 1.0 / math.sqrt(HK)

    mixed_qkv, a, b = _split_qkvzba(kwargs, meta)

    # SGLang conv_state layout: [cache, dim, width-1] (transposed vs SYCL)
    conv_st = conv_state_clone.transpose(1, 2)

    if meta["mode"] == "prefill":

        def do_conv():
            x = mixed_qkv.transpose(0, 1).contiguous()  # [dim, n_tok]
            out = causal_conv1d_fn(
                x,
                conv_w,
                conv_b,
                conv_states=conv_st,
                query_start_loc=qsl,
                seq_lens_cpu=meta["per_seq"].to(torch.int32),
                cache_indices=cache_idx,
                has_initial_state=kwargs["has_initial_state"],
                activation="silu",
            )
            return out.transpose(0, 1)[:n_tok].contiguous()

        def do_gating(conv_out_):
            return fused_gdn_gating(A_log, a, b, dt_bias)

        def do_delta(conv_out_, g_, beta_):
            q_ = conv_out_[:, : NK * HK].view(1, n_tok, NK, HK)
            k_ = conv_out_[:, NK * HK : 2 * NK * HK].view(1, n_tok, NK, HK)
            v_ = conv_out_[:, 2 * NK * HK :].view(1, n_tok, NV, HV)
            g_v = g_.view(1, n_tok, NV)
            b_v = beta_.view(1, n_tok, NV)
            o, _, _ = chunk_gated_delta_rule(
                q=q_,
                k=k_,
                v=v_,
                g=g_v,
                beta=b_v,
                initial_state=ssm_state_clone,
                initial_state_indices=cache_idx,
                cu_seqlens=qsl,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            return o

        conv_out = do_conv()
        g, beta = do_gating(conv_out)
        out = do_delta(conv_out, g, beta)

        return out.reshape(n_tok, NV, HV), dict(
            conv=do_conv,
            gating=lambda: do_gating(conv_out),
            delta=lambda: do_delta(conv_out, g, beta),
        )

    else:  # decode

        def do_conv():
            return causal_conv1d_update(
                mixed_qkv,
                conv_st,
                conv_w,
                conv_b,
                "silu",
                conv_state_indices=cache_idx,
            )

        def do_delta(conv_out_):
            q_ = conv_out_[:, : NK * HK].view(1, n_tok, NK, HK)
            k_ = conv_out_[:, NK * HK : 2 * NK * HK].view(1, n_tok, NK, HK)
            v_ = conv_out_[:, 2 * NK * HK :].view(1, n_tok, NV, HV)
            return fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q_,
                k=k_,
                v=v_,
                b=b,
                initial_state_source=ssm_state_clone,
                initial_state_indices=cache_idx,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=qsl,
            )

        conv_out = do_conv()
        out = do_delta(conv_out)

        return out.reshape(n_tok, NV, HV), dict(
            conv=do_conv,
            delta=lambda: do_delta(conv_out),
        )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark sgl_kernel.gdn_attention (SYCL) vs SGLang Triton"
    )
    ap.add_argument(
        "--iters", type=int, default=50, help="Timing iterations (default: 50)"
    )
    ap.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations (default: 10)"
    )
    ap.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16"],
        help="Data type (default: bf16)",
    )
    ap.add_argument(
        "--no-triton",
        action="store_true",
        help="Skip SGLang Triton comparison even if available",
    )
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    bpe = dtype.itemsize
    bpe_ssm = 4  # SSM state is always float32

    compare = _SGLANG_AVAILABLE and not args.no_triton

    # ── Header ───────────────────────────────────────────────────────────────
    print("=" * 78)
    print("  sgl_kernel GDN Attention Benchmark  —  Intel Xe2 (B60/BMG)")
    print("=" * 78)
    print(f"  Model    : Qwen3.5-9B (TP=1) — {NK}×{HK}K + {NV}×{HV}V, conv_w={W}")
    print(f"  dtype    : {args.dtype}  |  warmup={args.warmup}  iters={args.iters}")
    print(
        f"  SGLang   : {'✓ found — comparison enabled' if compare else '✗ not available — SYCL only'}"
    )
    if not _SGLANG_AVAILABLE and not args.no_triton:
        warnings.warn(
            f"SGLang Triton kernels not found ({_SGLANG_MISS_REASON}). "
            "Only SYCL numbers will be shown. Pass --no-triton to suppress this warning.",
            stacklevel=2,
        )
    print()

    # ── Workloads ─────────────────────────────────────────────────────────────
    # Matches the Qwen3.5-9B serving spec: BS=(1,4), input_len(ISL)=(1024,4096),
    # output_len(OSL)=1024.
    #   Prefill: one call per (batch, ISL) — the full input_len is processed as
    #   a single chunked-prefill call.
    #   Decode: GDN's recurrent state (conv_state/ssm_state) has O(1) size that
    #   does not grow with context length, so a single decode step's cost is
    #   independent of how far into the OSL=1024 generation we are — one
    #   decode-step benchmark per batch size already represents any of the
    #   1024 output positions.
    workloads = [
        # (mode,    batch, seqlen)
        ("prefill", 1, 1024),
        ("prefill", 1, 4096),
        ("prefill", 4, 1024),
        ("prefill", 4, 4096),
        ("decode", 1, 1),
        ("decode", 4, 1),
    ]

    # ── Column widths ──────────────────────────────────────────────────────────
    C0, C1, C2, C3, C4, C5, C6 = 20, 10, 10, 9, 10, 10, 9

    def hdr_line():
        h = (
            f"{'workload':<{C0}}"
            f"{'SYCL(µs)':>{C1}}"
            f"{'BW(GB/s)':>{C2}}"
            f"{'TFLOPS':>{C3}}"
        )
        if compare:
            h += (
                f"{'Tri(µs)':>{C4}}"
                f"{'BW(GB/s)':>{C5}}"
                f"{'TFLOPS':>{C6}}"
                f"{'speedup':>8}"
            )
        return h

    print(hdr_line())
    print("-" * len(hdr_line()))

    stage_rows = []  # for per-stage Triton breakdown

    for mode, bs, sl in workloads:
        random.seed(0)
        kwargs, meta = make_inputs(mode, bs, sl, dtype)
        n_tok = meta["n_tok"]
        name = f"{mode}_b{bs}" + (f"_s{sl}" if mode == "prefill" else "")

        flops = estimate_flops(mode, n_tok, bs, sl)
        nbytes = estimate_bytes(mode, n_tok, bs, sl, bpe, bpe_ssm)

        # ── SYCL timing ──────────────────────────────────────────────────────
        conv0 = kwargs["conv_state"].clone()
        ssm0 = kwargs["ssm_state"].clone()

        def sycl_fn():
            run_sycl(kwargs)

        t_sycl_ms = time_ms(sycl_fn, warmup=args.warmup, iters=args.iters)
        t_sycl_us = t_sycl_ms * 1e3
        bw_sycl = nbytes / 1e9 / (t_sycl_ms / 1e3)  # GB/s
        tf_sycl = flops / 1e12 / (t_sycl_ms / 1e3)  # TFLOPS

        row = (
            f"{name:<{C0}}"
            f"{t_sycl_us:>{C1}.1f}"
            f"{bw_sycl:>{C2}.1f}"
            f"{tf_sycl:>{C3}.3f}"
        )

        # ── Triton timing ─────────────────────────────────────────────────────
        if compare:
            kwargs["conv_state"].copy_(conv0)
            kwargs["ssm_state"].copy_(ssm0)
            conv_s = conv0.clone()
            ssm_s = ssm0.clone()

            # warm-up Triton pipeline (compiles Triton kernels on first call)
            for _ in range(args.warmup):
                sglang_pipeline(kwargs, meta, conv_s.clone(), ssm_s.clone())
            torch.xpu.synchronize()

            def tri_fn():
                sglang_pipeline(kwargs, meta, conv_s, ssm_s)

            t_tri_ms = time_ms(tri_fn, warmup=0, iters=args.iters)
            t_tri_us = t_tri_ms * 1e3
            bw_tri = nbytes / 1e9 / (t_tri_ms / 1e3)
            tf_tri = flops / 1e12 / (t_tri_ms / 1e3)
            speedup = t_tri_us / t_sycl_us

            row += (
                f"{t_tri_us:>{C4}.1f}"
                f"{bw_tri:>{C5}.1f}"
                f"{tf_tri:>{C6}.3f}"
                f"{speedup:>7.2f}x"
            )

            # collect stage timings for breakdown table
            _, stages = sglang_pipeline(kwargs, meta, conv_s.clone(), ssm_s.clone())
            t_conv = time_ms(stages["conv"], warmup=3, iters=args.iters)
            t_gate = (
                time_ms(stages["gating"], warmup=3, iters=args.iters)
                if "gating" in stages
                else 0.0
            )
            t_delta = time_ms(stages["delta"], warmup=3, iters=args.iters)
            stage_rows.append((name, t_conv * 1e3, t_gate * 1e3, t_delta * 1e3))

        print(row)
        del kwargs, meta
        torch.xpu.empty_cache()
        gc.collect()

    # ── Per-stage Triton breakdown ────────────────────────────────────────────
    if compare and stage_rows:
        print()
        print("SGLang Triton per-stage breakdown (µs):")
        hdr2 = (
            f"{'workload':<{C0}}"
            f"{'conv1d':>10}"
            f"{'gating':>10}"
            f"{'delta':>10}"
            f"{'sum':>10}"
        )
        print(hdr2)
        print("-" * len(hdr2))
        for name, tc, tg, td in stage_rows:
            print(f"{name:<{C0}}{tc:>10.1f}{tg:>10.1f}{td:>10.1f}{tc+tg+td:>10.1f}")

    # ── Notes ─────────────────────────────────────────────────────────────────
    print()
    print("Notes:")
    print(f"  BW (GB/s) = bytes_moved / time, bytes_moved accounts for qkvz/ba input,")
    print(f"    core_attn_out+z output, conv_state R+W, conv/gate weights,")
    print(f"    and ssm_state R+W -- once per call for decode, once per chunk")
    print(f"    (chunk_size={GDN_CHUNK_SIZE}) boundary for prefill.")
    print(
        f"  TFLOPS = conv1d+SiLU + Q/K L2-norm + delta-rule core. Prefill includes the"
    )
    print(
        f"    O(chunk_size^2) intra-chunk GEMMs (build/solve/output) per {GDN_CHUNK_SIZE}-token chunk;"
    )
    print(
        f"    decode is the single-step recurrent update. Same denominator for SYCL and Triton."
    )
    if compare:
        print(f"  speedup = Triton_time / SYCL_time  (>1 means SYCL is faster).")
        print()
        print("  Apple-to-apple mapping (SYCL fuses multiple Triton stages):")
        print("    SYCL kernel           ↔  Triton stages")
        print("    chunk_causal_conv1d   ↔  fused_qkvzba_split + causal_conv1d_fn")
        print("                              + l2norm + fused_gdn_gating  (prefill)")
        print("    causal_conv1d_kernel  ↔  fused_qkvzba_split + causal_conv1d_update")
        print("                                                              (decode)")
        print("    ChunkFwdOKernel+…     ↔  chunk_gated_delta_rule          (prefill)")
        print("    gated_delta_rule      ↔  fused_recurrent_gated_delta_rule (decode)")


if __name__ == "__main__":
    main()
