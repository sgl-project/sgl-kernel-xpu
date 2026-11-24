"""Benchmark the standalone ``sgl_kernel.gated_delta_rule`` op's CHUNK vs
SEQUENTIAL dispatch across a token-length sweep, for re-tuning the
``gdn::kChunkMinTokens`` dispatch threshold.

Background
----------
``gated_delta_rule`` (the SSM-only half of the fused ``gdn_attention``, minus
conv1d) selects between two kernels for a single prefill sequence:

  * the CHUNK (parallel) kernel -- 64-token-aligned chunked delta rule, fast
    for long sequences but with per-call fixed overhead (padded-buffer
    allocation + per-chunk launches); and
  * the SEQUENTIAL (token-serial) kernel -- one recurrent step per token, no
    chunking overhead, which stays competitive (or faster) for short prompts.

The selection threshold is the C++ constant ``gdn::kChunkMinTokens``
(see ``src/sycl/gdn_attn/gdn_attn_utils.h``). Because the crossover between the
two kernels is geometry- and device-dependent (head counts, head dims, batch),
this sweep exists to RE-MEASURE it whenever that constant or the target
hardware/geometry changes, rather than to hardcode the conclusion.

What it measures
----------------
For each geometry (``nk``/``nv`` key/value head split) and token length ``T``:

  * ``natural``  -- the op's own dispatch (chunk when ``T >= kChunkMinTokens``
    and the single-sequence/basic constraints hold, else sequential); and
  * ``sequel   `` -- the SEQUENTIAL kernel forced on, by passing an identity
    ``token_indx`` remap (the op's chunk predicate requires
    ``token_indx is None``, so a remap always routes to the sequential kernel
    while producing the identical output layout).

The kink where ``natural`` switches from tracking ``sequential`` to dropping
below it is the measured crossover. Tune ``kChunkMinTokens`` to sit at/just
above it for the target geometry.

NOTE: this benchmark only exercises the standalone op on a SINGLE sequence
(the chunk path's current precondition). Multi-sequence dispatch is a separate
follow-up and is intentionally NOT covered here.

Run
---
  ZE_AFFINITY_MASK=0 python benchmark/bench_gdn_chunk_dispatch.py
  ZE_AFFINITY_MASK=0 python benchmark/bench_gdn_chunk_dispatch.py --iters 100
  ZE_AFFINITY_MASK=0 python benchmark/bench_gdn_chunk_dispatch.py --dtype fp16
"""

import argparse
import gc

import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel.gated_delta_rule
import torch

DEVICE = "xpu"

# MUST be kept in sync with gdn::kChunkMinTokens in
# src/sycl/gdn_attn/gdn_attn_utils.h. Used only to LABEL which kernel the op's
# natural dispatch is expected to pick for a given T (and to annotate the
# crossover) -- it does not control the op itself, which owns the threshold.
K_CHUNK_MIN_TOKENS = 512

HEAD_K_DIM = 128
HEAD_V_DIM = 128

# Head-count geometries. (nk, nv) = (key heads, value heads) after TP split.
#   16/32 -> TP=1 (e.g. Qwen3.5-9B single GPU)
#   4/12  -> TP=4 (e.g. Qwen3.8-27B 4-way tensor parallel)
GEOMETRIES = [
    ("nk16/nv32", 16, 32),
    ("nk4/nv12", 4, 12),
]


def _make_inputs(seq_len: int, nk: int, nv: int, dtype: torch.dtype):
    """Build a single prefill sequence of ``seq_len`` tokens for the standalone
    op. Returns ``(kwargs, token_indx_identity)`` where ``kwargs`` are the
    positional/keyword args for ``sgl_kernel.gated_delta_rule`` WITHOUT the
    optional remap, and ``token_indx_identity`` forces the sequential path."""
    total = seq_len
    cache_bs = 64

    g = torch.Generator(device=DEVICE).manual_seed(0)
    rnd = lambda *s: torch.randn(*s, dtype=dtype, device=DEVICE, generator=g)
    rndf = lambda *s: torch.randn(*s, dtype=torch.float32, device=DEVICE, generator=g)

    q = rnd(total, nk, HEAD_K_DIM)
    k = rnd(total, nk, HEAD_K_DIM)
    v = rnd(total, nv, HEAD_V_DIM)
    b = rnd(total, nv)
    a = rnd(total, nv)
    A_log = rndf(nv)
    dt_bias = rnd(nv)
    ssm_state = rndf(cache_bs, nv, HEAD_V_DIM, HEAD_K_DIM)  # committed pool (fp32)

    # One sequence -> query_start_loc = [0, total]; single state row.
    query_start_loc = torch.tensor([0, total], dtype=torch.int32, device=DEVICE)
    cache_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    core_attn_out = torch.zeros(total, nv, HEAD_V_DIM, dtype=dtype, device=DEVICE)

    def call(token_indx=None):
        # Wrapper signature: (core_attn_out, q, k, v, b, a, A_log, dt_bias,
        # ssm_state, query_start_loc, cache_indices, num_prefills, num_decodes,
        # token_indx=None, has_initial_state=None). token_indx=None lets the op
        # pick the chunk path; passing a remap always routes to the sequential
        # kernel (its chunk predicate requires token_indx is None).
        sgl_kernel.gated_delta_rule(
            core_attn_out,
            q,
            k,
            v,
            b,
            a,
            A_log,
            dt_bias,
            ssm_state,
            query_start_loc,
            cache_indices,
            num_prefills=1,
            num_decodes=0,
            token_indx=token_indx,
            has_initial_state=None,
        )

    return call, torch.arange(total, dtype=torch.int32, device=DEVICE)


def time_ms(fn, warmup: int, iters: int) -> float:
    """Median wall time in ms using XPU events (mirrors bench_gdn_attention)."""
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
    return times[len(times) // 2]


def run_geometry(
    label: str, nk: int, nv: int, lengths, dtype: torch.dtype, warmup: int, iters: int
) -> None:
    print(
        f"\n=== {label}  ({nk} key / {nv} value heads, "
        f"head_dim {HEAD_K_DIM}/{HEAD_V_DIM}, dtype={dtype}) ==="
    )
    print(
        f"  {'T':>8} {'expected':>10} {'natural(ms)':>12} "
        f"{'forced-seq(ms)':>15} {'natural/seq':>13}"
    )
    print("  " + "-" * 64)
    for T in lengths:
        call, identity = _make_inputs(T, nk, nv, dtype)

        t_natural = time_ms(lambda: call(None), warmup=warmup, iters=iters)
        # Force the sequential kernel via an identity token_indx remap.
        t_seq = time_ms(lambda: call(identity), warmup=warmup, iters=iters)

        # The op's natural path is chunk iff all its single-sequence predicate
        # constraints hold here (single seq, bf16 activations, fp32 state,
        # head_dim 128) AND T >= kChunkMinTokens. This harness always satisfies
        # the shape/dtype constraints, so the label reduces to the token count.
        expected = "chunk" if T >= K_CHUNK_MIN_TOKENS else "sequential"
        ratio = t_natural / t_seq if t_seq > 0 else float("nan")
        # ratio < 1 means natural (chunk) beats the sequential kernel.
        print(
            f"  {T:>8} {expected:>10} {t_natural:>12.4f} {t_seq:>15.4f} {ratio:>13.3f}"
        )

        del call, identity
        torch.xpu.empty_cache()
        gc.collect()


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark standalone gated_delta_rule chunk/sequential dispatch sweep"
    )
    ap.add_argument(
        "--iters", type=int, default=200, help="Timing iterations (default: 200)"
    )
    ap.add_argument(
        "--warmup", type=int, default=20, help="Warmup iterations (default: 20)"
    )
    ap.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16"],
        help="Activation dtype (default: bf16)",
    )
    ap.add_argument(
        "--lengths",
        type=int,
        nargs="*",
        default=[128, 256, 384, 512, 640, 1024, 2048, 4096, 8192],
        help="Token lengths to sweep",
    )
    args = ap.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("Intel XPU is not available in this environment.")
    if not hasattr(torch.ops.sgl_kernel, "gated_delta_rule"):
        raise SystemExit("gated_delta_rule op is not registered in this build.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print("=" * 72)
    print("  sgl_kernel GDN standalone dispatch sweep (chunk vs sequential)")
    print("=" * 72)
    print(f"  kChunkMinTokens (C++ gdn::kChunkMinTokens) = {K_CHUNK_MIN_TOKENS}")
    print(f"  dtype={args.dtype}  warmup={args.warmup}  iters={args.iters}")
    print("  natural/seq < 1.0 => natural (chunk) is faster than sequential.")
    print("  Re-tune gdn::kChunkMinTokens to the T where natural first beats")
    print("  the forced sequential kernel for the target geometry/device.")

    for label, nk, nv in GEOMETRIES:
        run_geometry(label, nk, nv, args.lengths, dtype, args.warmup, args.iters)

    print("\nDone.")


if __name__ == "__main__":
    main()
