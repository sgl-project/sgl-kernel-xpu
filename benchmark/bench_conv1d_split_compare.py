"""Microbenchmark: conv1d-only cost, general-purpose design vs. GDN-specific
design, isolated from the q/k/v/z/b/a split the GDN-specific kernel is
normally fused with.

Compares four kernels, matched on shape (dim=8192, width=4 — the Qwen3.5-9B
GDN conv-channel count and conv width at TP=1) and matched on total
memory footprint (conv state stored as exactly `width - 1` history steps for
both designs):

  1. General-purpose prefill: ``torch.ops.sgl_kernel.causal_conv1d_fwd``
     (channel-parallel, serial-over-time per work-item; backs the
     standalone public op).
  2. GDN-style prefill: ``torch.ops.sgl_kernel.bench_gdn_style_causal_conv1d_prefill``
     (token-parallel, no serial loop; conv-only extraction of the kernel
     that's normally fused with q/k/v/z/b/a split inside `gdn_attention`;
     needs a follow-up `update_states_kernel` launch to finalize conv
     state, unlike the general-purpose design).
  3. General-purpose decode: ``torch.ops.sgl_kernel.causal_conv1d_update``.
  4. GDN-style decode: ``torch.ops.sgl_kernel.bench_gdn_style_causal_conv1d_decode``
     (same token-parallel kernel as #2, but every batch entry has exactly
     one token so state updates happen in-place — no follow-up launch).
  5. GDN-style decode (no-scan fast path):
     ``torch.ops.sgl_kernel.bench_gdn_style_causal_conv1d_decode_fast`` --
     identical to #4 but skips the O(batch_size) per-thread linear scan
     used to look up each token's batch_id (needed for prefill, where
     multiple tokens share a batch_id, but *not* for decode, where every
     "sequence" is exactly one token so batch_id == token_id trivially).
     Added after discovering that #4 regresses vs. general-purpose at
     large decode batch (bs >= ~96) purely because of that redundant scan
     -- with it removed, GDN-style wins decode at every batch size tested,
     including bs=256.

NOTE: the GDN-style kernels here use a *fixed* elems_per_item=4 (matching
the production kernel's high-occupancy branch). Production `gdn_attention`
adaptively switches to elems_per_item=1 for small work-group counts (roughly
decode batch <= 4, see the tuned threshold in `causal_conv1d.hpp`'s
`kernel_launcher`), which this benchmark does NOT replicate. This means the
GDN-style numbers at very small decode batch sizes may look worse here than
they would in production -- interpret small-batch decode results with that
caveat in mind. This does NOT affect the scan-cost finding above: production's
own heuristic would also pick elems_per_item=4 at the batch sizes where the
regression appears (default_work_groups = batch*8 >> 64 threshold).

Layouts are NOT unified across the two designs (this is deliberate — the
two kernels have genuinely different natural layouts as part of their
respective designs, not an artifact): general-purpose uses standard NCL
conv1d layout (channel-major); GDN-style uses token-major (as produced
directly by an upstream matmul, no transpose needed in the real pipeline).
Only the kernel-launch region is timed; layout setup is untimed.

Run:
  ZE_AFFINITY_MASK=0 python benchmark/bench_conv1d_split_compare.py
  ZE_AFFINITY_MASK=0 python benchmark/bench_conv1d_split_compare.py --iters 100
"""

import argparse
import gc

import sgl_kernel  # noqa: F401  registers the torch.ops.sgl_kernel.* conv1d ops
import torch

DEVICE = "xpu"
DIM = 8192  # QKV_DIM: Qwen3.5-9B GDN conv-channel count (NK*(2*HK + HV*NV/NK)) at TP=1
WIDTH = 4  # linear_conv_kernel_dim


def time_ms(fn, warmup: int = 10, iters: int = 200) -> float:
    """Median wall time in milliseconds using XPU events (matches
    bench_gdn_attention.py's convention)."""
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


def _qsl(per_seq_lens, device):
    per_seq = torch.tensor(per_seq_lens, dtype=torch.int64)
    return (
        torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(per_seq, 0)])
        .to(torch.int32)
        .to(device)
    )


# ── general-purpose (channel-parallel) inputs ───────────────────────────────


def make_general_prefill_inputs(batch_size, seqlen, dtype, gen):
    total_tokens = batch_size * seqlen
    x = torch.randn(DIM, total_tokens, dtype=dtype, device=DEVICE, generator=gen)
    weight = torch.randn(DIM, WIDTH, dtype=dtype, device=DEVICE, generator=gen)
    bias = torch.randn(DIM, dtype=dtype, device=DEVICE, generator=gen)
    cache_size = max(256, batch_size * 2)
    conv_states = torch.randn(
        cache_size, DIM, WIDTH - 1, dtype=dtype, device=DEVICE, generator=gen
    )
    qsl = _qsl([seqlen] * batch_size, DEVICE)
    cache_indices = torch.arange(batch_size, dtype=torch.int32, device=DEVICE)
    has_init = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)
    return dict(
        x=x,
        weight=weight,
        bias_=bias,
        conv_states=conv_states,
        query_start_loc=qsl,
        cache_indices=cache_indices,
        has_initial_state=has_init,
        silu_activation=True,
        pad_slot_id=-1,
    )


def make_general_decode_inputs(batch_size, dtype, gen):
    x = torch.randn(batch_size, DIM, 1, dtype=dtype, device=DEVICE, generator=gen)
    weight = torch.randn(DIM, WIDTH, dtype=dtype, device=DEVICE, generator=gen)
    bias = torch.randn(DIM, dtype=dtype, device=DEVICE, generator=gen)
    cache_size = max(256, batch_size * 2)
    conv_state = torch.randn(
        cache_size, DIM, WIDTH - 1, dtype=dtype, device=DEVICE, generator=gen
    )
    conv_state_indices = torch.arange(batch_size, dtype=torch.int32, device=DEVICE)
    return dict(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias_=bias,
        silu_activation=True,
        cache_seqlens_=None,
        conv_state_indices_=conv_state_indices,
        pad_slot_id=-1,
    )


# ── GDN-style (token-parallel) inputs ────────────────────────────────────────


def make_gdn_prefill_inputs(batch_size, seqlen, dtype, gen):
    total_tokens = batch_size * seqlen
    x_in = torch.randn(total_tokens, DIM, dtype=dtype, device=DEVICE, generator=gen)
    out = torch.empty_like(x_in)
    weight = torch.randn(DIM, WIDTH, dtype=dtype, device=DEVICE, generator=gen)
    bias = torch.randn(DIM, dtype=dtype, device=DEVICE, generator=gen)
    cache_size = max(256, batch_size * 2)
    conv_states = torch.randn(
        cache_size, WIDTH - 1, DIM, dtype=dtype, device=DEVICE, generator=gen
    )
    qsl = _qsl([seqlen] * batch_size, DEVICE)
    cache_indices = torch.arange(batch_size, dtype=torch.int32, device=DEVICE)
    has_init = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)
    return dict(
        out=out,
        x_in=x_in,
        conv_weights=weight,
        conv_bias=bias,
        conv_states=conv_states,
        query_start_loc=qsl,
        cache_indices=cache_indices,
        has_initial_state=has_init,
        silu_activation=True,
        pad_slot_id=-1,
    )


def make_gdn_decode_inputs(batch_size, dtype, gen):
    x_in = torch.randn(batch_size, DIM, dtype=dtype, device=DEVICE, generator=gen)
    out = torch.empty_like(x_in)
    weight = torch.randn(DIM, WIDTH, dtype=dtype, device=DEVICE, generator=gen)
    bias = torch.randn(DIM, dtype=dtype, device=DEVICE, generator=gen)
    cache_size = max(256, batch_size * 2)
    conv_states = torch.randn(
        cache_size, WIDTH - 1, DIM, dtype=dtype, device=DEVICE, generator=gen
    )
    qsl = _qsl([1] * batch_size, DEVICE)
    cache_indices = torch.arange(batch_size, dtype=torch.int32, device=DEVICE)
    has_init = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)
    return dict(
        out=out,
        x_in=x_in,
        conv_weights=weight,
        conv_bias=bias,
        conv_states=conv_states,
        query_start_loc=qsl,
        cache_indices=cache_indices,
        has_initial_state=has_init,
        silu_activation=True,
        pad_slot_id=-1,
    )


def run_general_prefill(kwargs):
    torch.ops.sgl_kernel.causal_conv1d_fwd(**kwargs)


def run_general_decode(kwargs):
    torch.ops.sgl_kernel.causal_conv1d_update(**kwargs)


def run_gdn_prefill(kwargs):
    torch.ops.sgl_kernel.bench_gdn_style_causal_conv1d_prefill(**kwargs)


def run_gdn_decode(kwargs):
    torch.ops.sgl_kernel.bench_gdn_style_causal_conv1d_decode(**kwargs)


def run_gdn_decode_fast(kwargs):
    torch.ops.sgl_kernel.bench_gdn_style_causal_conv1d_decode_fast(**kwargs)


def main():
    ap = argparse.ArgumentParser(
        description="Compare general-purpose vs GDN-style conv1d-only kernels"
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print("=" * 88)
    print("  conv1d-only microbenchmark: general-purpose vs. GDN-style design")
    print("=" * 88)
    print(
        f"  dim={DIM}  width={WIDTH}  dtype={args.dtype}  warmup={args.warmup}  iters={args.iters}"
    )
    print()

    prefill_workloads = [
        (1, 1024),
        (1, 4096),
        (4, 1024),
        (4, 4096),
        (8, 1024),
        (8, 4096),
        (32, 8192),
        (128, 1024),
    ]
    decode_workloads = [1, 4, 8, 32, 64, 96, 128, 160, 192, 256]

    C0, C1, C2, C3 = 22, 16, 16, 10
    hdr = f"{'workload':<{C0}}{'general(us)':>{C1}}{'gdn-style(us)':>{C2}}{'speedup(gen/gdn)':>{C3}}"
    print(hdr)
    print("-" * len(hdr))

    for bs, sl in prefill_workloads:
        gen = torch.Generator(device=DEVICE).manual_seed(42)
        gkw = make_general_prefill_inputs(bs, sl, dtype, gen)
        gen2 = torch.Generator(device=DEVICE).manual_seed(42)
        dkw = make_gdn_prefill_inputs(bs, sl, dtype, gen2)

        t_gen = time_ms(lambda: run_general_prefill(gkw), args.warmup, args.iters) * 1e3
        t_gdn = time_ms(lambda: run_gdn_prefill(dkw), args.warmup, args.iters) * 1e3

        name = f"prefill_b{bs}_s{sl}"
        print(f"{name:<{C0}}{t_gen:>{C1}.2f}{t_gdn:>{C2}.2f}{t_gen / t_gdn:>{C3}.2f}")

        del gkw, dkw
        torch.xpu.empty_cache()
        gc.collect()

    print()
    print(
        "Decode: 3-way comparison -- general-purpose vs. GDN-style (with the "
        "O(batch) batch_id scan, matching the prefill kernel unmodified) vs. "
        "GDN-style no-scan fast path (batch_id = token_id, decode-only)."
    )
    C4 = 16
    hdr2 = (
        f"{'workload':<{C0}}{'general(us)':>{C1}}{'gdn+scan(us)':>{C2}}"
        f"{'gdn-nosc(us)':>{C4}}{'gen/gdn+scan':>{C3}}{'gen/gdn-nosc':>{C3}}"
    )
    print(hdr2)
    print("-" * len(hdr2))

    for bs in decode_workloads:
        gen = torch.Generator(device=DEVICE).manual_seed(42)
        gkw = make_general_decode_inputs(bs, dtype, gen)
        gen2 = torch.Generator(device=DEVICE).manual_seed(42)
        dkw = make_gdn_decode_inputs(bs, dtype, gen2)
        gen3 = torch.Generator(device=DEVICE).manual_seed(42)
        fkw = make_gdn_decode_inputs(bs, dtype, gen3)

        t_gen = time_ms(lambda: run_general_decode(gkw), args.warmup, args.iters) * 1e3
        t_gdn = time_ms(lambda: run_gdn_decode(dkw), args.warmup, args.iters) * 1e3
        t_fast = (
            time_ms(lambda: run_gdn_decode_fast(fkw), args.warmup, args.iters) * 1e3
        )

        name = f"decode_b{bs}"
        print(
            f"{name:<{C0}}{t_gen:>{C1}.2f}{t_gdn:>{C2}.2f}{t_fast:>{C4}.2f}"
            f"{t_gen / t_gdn:>{C3}.2f}{t_gen / t_fast:>{C3}.2f}"
        )

        del gkw, dkw, fkw
        torch.xpu.empty_cache()
        gc.collect()

    print()
    print("Notes:")
    print("  speedup > 1.0 means GDN-style (token-parallel) is faster; < 1.0 means")
    print("  general-purpose (channel-parallel) is faster.")
    print("  GDN-style prefill pays an extra kernel launch (update_states_kernel) to")
    print("  finalize conv state; general-purpose does not need one.")
    print("  gdn+scan reuses the prefill kernel's O(batch) batch_id lookup unmodified;")
    print("  it regresses vs. general-purpose past bs~96 purely from that redundant")
    print("  per-thread scan. gdn-nosc is a decode-only fast path that skips it")
    print("  (batch_id = token_id, valid because decode is always 1 token/sequence);")
    print("  it should be preferred for any real decode/verify op.")


if __name__ == "__main__":
    main()
