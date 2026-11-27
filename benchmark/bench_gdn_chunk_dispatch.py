"""Benchmark the standalone ``sgl_kernel.gated_delta_rule`` op's CHUNK vs
SEQUENTIAL dispatch across a token-length sweep, for re-tuning the
``gdn::kChunkMinTokens`` dispatch threshold.

Background
----------
``gated_delta_rule`` (the SSM-only half of the fused ``gdn_attention``, minus
conv1d) selects between two kernels for a prefill batch (single- or
multi-sequence; the chunk path handles any number of sequences, each with its
own SSM-state row):

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
For each geometry (``nk``/``nv`` key/value head split) and prefill-batch shape
(single- or multi-sequence; the multi-sequence set covers ragged and balanced
batches, including the "two 4096 prefill" shape from prior reviews):

  * ``natural``  -- the op's own dispatch (chunk when the batch's TOTAL token
    count is >= kChunkMinTokens and the basic constraints hold, else
    sequential); and
  * ``seq_forced`` -- the SEQUENTIAL kernel forced on, by passing an identity
    ``token_indx`` remap (the op's chunk predicate requires
    ``token_indx is None``, so a remap always routes to the sequential kernel
    while producing the identical output layout).

For single sequences, the kink where ``natural`` switches from tracking
``sequential`` to dropping below it is the measured crossover -- tune
``kChunkMinTokens`` to sit at/just above it for the target geometry.
For multi-sequence batches, the interesting axis is inverted: the per-sequence
token count is what the sequential kernel pays for, so a batch of many modest
sequences is where the parallel chunk path should win even when each sequence
would individually be below the chunk threshold.

Run
---
  ZE_AFFINITY_MASK=0 python benchmark/bench_gdn_chunk_dispatch.py
  ZE_AFFINITY_MASK=0 python benchmark/bench_gdn_chunk_dispatch.py --iters 100
  ZE_AFFINITY_MASK=0 python benchmark/bench_gdn_chunk_dispatch.py --dtype fp16
  ZE_AFFINITY_MASK=0 python benchmark/bench_gdn_chunk_dispatch.py \
    --multi-shapes "4096+4096" "2048+2048+2048" "1025+1023"
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


def _make_inputs(seq_lens, nk: int, nv: int, dtype: torch.dtype):
    """Build a prefill batch (one or more sequences, jagged as required) for
    the standalone op. Returns ``(call, token_indx_identity)`` where ``call``
    invokes ``sgl_kernel.gated_delta_rule`` with the harness inputs (its
    ``token_indx=None`` lets the op dispatch naturally, and passing the
    identity remap always forces the sequential kernel), and
    ``token_indx_identity`` is that identity remap."""
    if isinstance(seq_lens, int):
        seq_lens = [seq_lens]
    assert len(seq_lens) >= 1 and all(l > 0 for l in seq_lens)
    total = sum(seq_lens)
    num_seqs = len(seq_lens)

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
    # Committed state pool, one DISTINCT row per sequence: this exercises
    # per-sequence read/scatter/write in the chunk path (not a shared single
    # row). Always fp32 -- the dtype the chunk path requires (the sequential
    # path accepts any dtype); the `--dtype` flag only drives q/k/v/b/a.
    cache_bs = max(64, num_seqs)
    ssm_state = rndf(cache_bs, nv, HEAD_V_DIM, HEAD_K_DIM)  # committed pool (fp32)
    ssm_state_indices = torch.arange(num_seqs, dtype=torch.int32, device=DEVICE)

    # Ragged query_start_loc (cumulative sums); all rows are pure-prefill.
    query_start_loc = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=DEVICE,
    )
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
            ssm_state_indices,
            num_prefills=num_seqs,
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


DEFAULT_MULTI_SHAPES = [
    [4096, 4096],  # "two 4096 prefill" -- the shape from the review survey
    [2048, 2048],
    [2048, 2048, 2048],
    [1024, 1024, 1024, 1024],
    [2048, 512, 512],  # ragged, total 3072
    [1025, 1023],  # ragged, straddles the total-token gate direction only
    [8192, 3],  # one long + one very short, total 8195
]


def run_geometry(
    label: str,
    nk: int,
    nv: int,
    shapes,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> None:
    print(
        f"\n=== {label}  ({nk} key / {nv} value heads, "
        f"head_dim {HEAD_K_DIM}/{HEAD_V_DIM}, dtype={dtype}) ==="
    )
    print(
        f"  {'shape':>24} {'expected':>10} {'natural(ms)':>12} "
        f"{'forced-seq(ms)':>15} {'natural/seq':>13}"
    )
    print("  " + "-" * 80)
    for shape in shapes:
        seq_lens = [int(x) for x in (shape if isinstance(shape, list) else [shape])]
        total_tokens = sum(seq_lens)
        call, identity = _make_inputs(seq_lens, nk, nv, dtype)

        t_natural = time_ms(lambda: call(None), warmup=warmup, iters=iters)
        # Force the sequential kernel via an identity token_indx remap.
        t_seq = time_ms(lambda: call(identity), warmup=warmup, iters=iters)

        # The op's natural path is chunk iff all its predicate constraints
        # hold here (pure prefill; the harness always satisfies the shape and
        # state-dtype parts) AND the batch's TOTAL token count is >=
        # kChunkMinTokens. fp16 activations never take the chunk path, so the
        # label below is correct only for --dtype bf16.
        expected = (
            "chunk"
            if total_tokens >= K_CHUNK_MIN_TOKENS and dtype == torch.bfloat16
            else "sequential"
        )
        ratio = t_natural / t_seq if t_seq > 0 else float("nan")
        # ratio < 1 means natural (chunk) beats the sequential kernel.
        print(
            f"  {'+'.join(map(str, seq_lens)):>24} {expected:>10} "
            f"{t_natural:>12.4f} {t_seq:>15.4f} {ratio:>13.3f}"
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
        help="Single-sequence token lengths to sweep",
    )
    ap.add_argument(
        "--multi-shapes",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Multi-sequence batch shapes as '+'-separated lists, e.g. "
            "'4096+4096' '2048+512+512'. Default: a fixed set covering the "
            "two-4096, balanced, and ragged cases."
        ),
    )
    args = ap.parse_args()

    if args.multi_shapes is None:
        multi_shapes = DEFAULT_MULTI_SHAPES
    else:
        multi_shapes = [
            [int(part) for part in shape.split("+")] for shape in args.multi_shapes
        ]
    shapes = list(args.lengths) + multi_shapes

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
    print(
        f"  single-seq lengths={args.lengths}; "
        f"multi-seq shapes={['+'.join(map(str, s)) for s in multi_shapes]}"
    )
    print("  natural/seq < 1.0 => natural (chunk) is faster than sequential.")
    print("  Re-tune gdn::kChunkMinTokens to the T where natural first beats")
    print("  the forced sequential kernel for the target geometry/device.")

    for label, nk, nv in GEOMETRIES:
        run_geometry(label, nk, nv, shapes, dtype, args.warmup, args.iters)

    print("\nDone.")


if __name__ == "__main__":
    main()
