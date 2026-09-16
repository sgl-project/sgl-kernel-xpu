"""Name and price the per-forward host-side cast/copy ops inside
``sgl_kernel.fused_experts`` on the gpt-oss-120b MXFP4 W4A16 MoE path.

Motivation
----------
``model_enablement/openAI/gpt-oss/120b/docs/profile_vs_projection_120b_v4.md``
§10c lever 1 asks to remove "the 73 BF16 cast-copies in MoE dispatch", but §10e
records that the aten op behind them **could not be named** because the v4
traces were captured with ``ProfilerActivity.XPU`` only (no ``cpu_op`` events).
This benchmark profiles with ``CPU + XPU`` so every device kernel is attributed
to the aten op that launched it.

Shapes come from the real checkpoint
(``openai/gpt-oss-120b/config.json``: 36 layers, hidden 2880, intermediate 2880,
128 local experts, top-k 4, swiglu_limit 7.0) put through the XPU MXFP4 weight
layout in ``sglang/srt/layers/quantization/mxfp4.py:525-539``:

    intermediate_per_partition = round_up(2880 // tp, 32)     # tp=4 -> 736
    w13_weight       [E, 2*I_p, H//2]   uint8 (packed MXFP4)
    w13_weight_scale [E, 2*I_p, H//32]  uint8 (E8M0)
    w13_weight_bias  [E, 2*I_p]         bfloat16   <-- stays bf16 on XPU
    w2_weight        [E, H,     I_p//2] uint8 (packed MXFP4)
    w2_weight_scale  [E, H,     I_p//32]uint8 (E8M0)
    w2_weight_bias   [E, H]             bfloat16   <-- stays bf16 on XPU

The bf16 bias dtype is load-bearing: ``mxfp4.py:578-588,618-625`` allocates both
biases as bfloat16 and only the ``_use_aiter`` branch (``:928-933``) promotes
them to fp32, so on XPU they arrive at ``fused_experts`` in bf16 every forward.

Usage
-----
    ZE_AFFINITY_MASK=4 python benchmark/bench_moe_dispatch_casts.py
    ZE_AFFINITY_MASK=4 python benchmark/bench_moe_dispatch_casts.py --shape 1024
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Dict, List, Tuple

import torch
from sgl_kernel import fused_experts
from torch.profiler import ProfilerActivity, profile

# ---------------------------------------------------------------------------
# gpt-oss-120b, tensor-parallel rank-local MoE shapes
# ---------------------------------------------------------------------------
HIDDEN = 2880
INTERMEDIATE = 2880
NUM_EXPERTS = 128  # not expert-parallel: TP shards the intermediate dim
TOPK = 4
MXFP4_BLOCK = 32
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def build_inputs(
    num_tokens: int, tp: int, device: str = "xpu", ids_dtype: torch.dtype = torch.int32
) -> dict:
    """Allocate one rank's MXFP4 MoE weights plus a decode/prefill activation."""
    torch.manual_seed(0)
    inter_p = round_up(INTERMEDIATE // tp, MXFP4_BLOCK)

    hidden_states = torch.randn(
        (num_tokens, HIDDEN), dtype=torch.bfloat16, device=device
    )

    # Packed MXFP4 codes: the values are numerically irrelevant here (we measure
    # the dispatch overhead and gate correctness against a same-input reference),
    # but they must be real 4-bit codes so the GEMM does not hit denormal paths.
    w13 = torch.randint(
        0, 256, (NUM_EXPERTS, 2 * inter_p, HIDDEN // 2), dtype=torch.uint8
    ).to(device)
    w2 = torch.randint(
        0, 256, (NUM_EXPERTS, HIDDEN, inter_p // 2), dtype=torch.uint8
    ).to(device)
    # E8M0 exponent bytes centred on 127 (== scale 1.0) so the GEMM stays in range.
    w13_scale = torch.randint(
        120, 134, (NUM_EXPERTS, 2 * inter_p, HIDDEN // MXFP4_BLOCK), dtype=torch.uint8
    ).to(device)
    w2_scale = torch.randint(
        120, 134, (NUM_EXPERTS, HIDDEN, inter_p // MXFP4_BLOCK), dtype=torch.uint8
    ).to(device)

    # bf16 biases, exactly as mxfp4.py leaves them on XPU.
    b1 = (
        torch.randn((NUM_EXPERTS, 2 * inter_p), dtype=torch.float32, device=device)
        .mul_(0.005)
        .to(torch.bfloat16)
    )
    b2 = (
        torch.randn((NUM_EXPERTS, HIDDEN), dtype=torch.float32, device=device)
        .mul_(0.005)
        .to(torch.bfloat16)
    )

    score = torch.softmax(
        torch.randn((num_tokens, NUM_EXPERTS), dtype=torch.float32, device=device),
        dim=-1,
    )
    topk_weights, topk_ids = torch.topk(score, TOPK)
    # Every topk implementation in sglang/srt/layers/moe/topk.py returns int32
    # ids, so int32 is the real serving path and the int64->int32 copy inside
    # fused_experts does not fire. --ids-dtype int64 forces it, to price it.
    topk_ids = topk_ids.to(ids_dtype)
    topk_weights = topk_weights.to(torch.bfloat16)

    return dict(
        hidden_states=hidden_states,
        w1=w13,
        w2=w2,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        b1=b1,
        b2=b2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inter_p=inter_p,
    )


def call(inp: dict) -> torch.Tensor:
    return fused_experts(
        inp["hidden_states"],
        inp["w1"],
        inp["w2"],
        inp["topk_weights"],
        inp["topk_ids"],
        b1=inp["b1"],
        b2=inp["b2"],
        use_mxfp4_w4a16=True,
        w1_scale=inp["w1_scale"],
        w2_scale=inp["w2_scale"],
        activation="silu",
        gemm1_alpha=SWIGLU_ALPHA,
        gemm1_limit=SWIGLU_LIMIT,
        swiglu_limit=SWIGLU_LIMIT,
    )


# ---------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------
def _is_device_row(evt) -> bool:
    """True for a device (kernel) row rather than a host ``cpu_op`` row.

    ``key_averages()`` reports ``self_device_time_total`` on BOTH the launching
    aten/sgl_kernel cpu_op and the kernel it correlates to, so summing the whole
    table double-counts. Kernel rows are the ones whose own device_type is the
    accelerator."""
    return str(getattr(evt, "device_type", "")).split(".")[-1] not in (
        "CPU",
        "USER_ANNOTATION",
    )


def profile_once(
    inp: dict, iters: int
) -> Tuple[Dict[str, Tuple[int, float, bool]], float]:
    """Return {event key: (count, self_device_us, is_device_row)} and total device us."""
    for _ in range(5):
        call(inp)
    torch.xpu.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        for _ in range(iters):
            call(inp)
        torch.xpu.synchronize()

    out: Dict[str, Tuple[int, float, bool]] = {}
    total_dev = 0.0
    for evt in prof.key_averages():
        if evt.count == 0:
            continue
        dev = float(evt.self_device_time_total)
        is_dev = _is_device_row(evt)
        out[evt.key] = (int(evt.count), dev, is_dev)
        if is_dev:
            total_dev += max(dev, 0.0)
    return out, total_dev


def report(
    label: str, inp: dict, iters: int, repeats: int, top: int = 26
) -> Dict[str, object]:
    per_repeat: List[Dict[str, Tuple[int, float, bool]]] = []
    totals: List[float] = []
    for _ in range(repeats):
        ev, tot = profile_once(inp, iters)
        per_repeat.append(ev)
        totals.append(tot)

    keys = set()
    for ev in per_repeat:
        keys |= set(ev)

    rows = []
    for k in keys:
        counts = [ev.get(k, (0, 0.0, False))[0] for ev in per_repeat]
        devs = [ev.get(k, (0, 0.0, False))[1] for ev in per_repeat]
        is_dev = any(ev.get(k, (0, 0.0, False))[2] for ev in per_repeat)
        rows.append(
            dict(
                key=k,
                side="device" if is_dev else "host",
                calls_per_fwd=statistics.median(counts) / iters,
                dev_us_per_fwd=statistics.median(devs) / iters,
                dev_us_min=min(devs) / iters,
                dev_us_max=max(devs) / iters,
            )
        )
    rows.sort(key=lambda r: -r["dev_us_per_fwd"])

    print(f"\n===== {label} =====")
    print(f"iters/profile={iters}  repeats={repeats}")
    print(
        f"total DEVICE time per fused_experts call: "
        f"median {statistics.median(totals)/iters:.3f} us "
        f"[min {min(totals)/iters:.3f}, max {max(totals)/iters:.3f}]"
    )
    print(f"{'event':<74}{'side':>7}{'calls/fwd':>11}{'self dev us/fwd':>17}")
    for r in rows[:top]:
        if r["dev_us_per_fwd"] <= 0.0 and r["calls_per_fwd"] < 0.5:
            continue
        print(
            f"{r['key'][:72]:<74}{r['side']:>7}"
            f"{r['calls_per_fwd']:>11.2f}{r['dev_us_per_fwd']:>17.3f}"
        )

    return dict(
        label=label,
        iters=iters,
        repeats=repeats,
        total_dev_us_per_fwd_median=statistics.median(totals) / iters,
        total_dev_us_per_fwd_min=min(totals) / iters,
        total_dev_us_per_fwd_max=max(totals) / iters,
        rows=rows,
    )


def _time_one_op(fn, iters: int, repeats: int) -> List[float]:
    """Device us per call of ``fn``, one sample per profiler repeat.

    Only device rows are summed (see ``_is_device_row``)."""
    for _ in range(10):
        fn()
    torch.xpu.synchronize()
    samples = []
    for _ in range(repeats):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
            for _ in range(iters):
                fn()
            torch.xpu.synchronize()
        tot = sum(
            max(float(e.self_device_time_total), 0.0)
            for e in prof.key_averages()
            if _is_device_row(e)
        )
        samples.append(tot / iters)
    return samples


def isolated_cast_cost(inp: dict, iters: int, repeats: int) -> Dict[str, object]:
    """Price ``b1.float()``, ``b2.float()`` and the int64->int32 ``topk_ids``
    copy on their own, so the removable cost is attributable independently of
    the surrounding grouped GEMMs."""
    b1, b2, topk_ids = inp["b1"], inp["b2"], inp["topk_ids"]
    ids_i64 = topk_ids.to(torch.int64)
    ids_i32 = torch.empty(topk_ids.shape, dtype=torch.int32, device=topk_ids.device)

    cases = {
        "b1.float()": (lambda: b1.float(), b1.numel() * 2, b1.numel() * 4),
        "b2.float()": (lambda: b2.float(), b2.numel() * 2, b2.numel() * 4),
        "topk_ids i64->i32": (
            lambda: ids_i32.copy_(ids_i64),
            topk_ids.numel() * 8,
            topk_ids.numel() * 4,
        ),
    }

    print("\n===== isolated per-op device time =====")
    print(f"{'op':<22}{'shape':<16}{'ideal bytes':>13}{'median us':>11}{'GB/s':>9}")
    out = {}
    for name, (fn, rb, wb) in cases.items():
        s = _time_one_op(fn, iters, repeats)
        med = statistics.median(s)
        shape = (
            tuple(b1.shape)
            if name.startswith("b1")
            else tuple(b2.shape) if name.startswith("b2") else tuple(topk_ids.shape)
        )
        print(
            f"{name:<22}{str(shape):<16}{(rb+wb)/1024:>11.1f}K"
            f"{med:>11.3f}{(rb+wb)/med/1e3:>9.1f}"
        )
        out[name] = dict(
            shape=list(shape),
            read_bytes=rb,
            write_bytes=wb,
            dev_us_median=med,
            dev_us_min=min(s),
            dev_us_max=max(s),
            eff_gbps=(rb + wb) / med / 1e3,
            samples=s,
        )
    bias_pair = out["b1.float()"]["dev_us_median"] + out["b2.float()"]["dev_us_median"]
    print(f"bias pair total (removable per fused_experts call): {bias_pair:.3f} us")
    out["bias_pair_dev_us_median"] = bias_pair
    return out


def cast_size_sweep(iters: int, repeats: int, device: str = "xpu") -> Dict[str, object]:
    """Is the bf16->fp32 copy's cost bytes or launch geometry?

    The v4 traces show 73 launches/step of
    ``UnrolledElementwiseKernel<CopyScalarFunc<c10::BFloat16>, ..., LoadWithCast<1>,
    StoreWithCast<1>>`` at ~8.68 us, and show that cost barely moving between a
    1-token decode and a 4096-token prefill. That token-independence is
    sometimes read as "the cost is size-independent, so it is grid geometry".
    It is not: 72 of the 73 are casts of the MoE *bias weights*, whose size does
    not depend on the token count at all. This sweep separates the two
    explanations by varying the number of elements directly.

    ``(1, 2880)`` is the vector the geometry hypothesis assumes is being moved;
    ``(1, 201088)`` is the gpt-oss-120b logits cast (the 73rd launch,
    ``sglang/srt/layers/logits_processor.py:1054``)."""
    shapes = [
        (1, 2880),  # one hidden vector -- the geometry hypothesis's assumption
        (128, 1472),  # b1 = w13_weight_bias, gpt-oss-120b tp=4
        (128, 2880),  # b2 = w2_weight_bias
        (1, 201088),  # lm_head logits, the 73rd bf16->fp32 cast per step
        (256, 2880),  # 2x b2, to show the slope
    ]
    print("\n===== bf16 -> fp32 copy: cost vs number of elements =====")
    print(f"{'shape':<16}{'elems':>10}{'bytes moved':>14}{'median us':>11}{'GB/s':>9}")
    out = []
    for shp in shapes:
        t = torch.randn(shp, dtype=torch.float32, device=device).to(torch.bfloat16)
        s = _time_one_op(lambda: t.float(), iters, repeats)
        med = statistics.median(s)
        moved = t.numel() * 2 + t.numel() * 4
        print(
            f"{str(shp):<16}{t.numel():>10}{moved/1024:>12.1f}K"
            f"{med:>11.3f}{moved/med/1e3:>9.1f}"
        )
        out.append(
            dict(
                shape=list(shp),
                numel=t.numel(),
                bytes_moved=moved,
                dev_us_median=med,
                dev_us_min=min(s),
                dev_us_max=max(s),
                eff_gbps=moved / med / 1e3,
            )
        )
        del t
    small, big = out[0]["dev_us_median"], out[-1]["dev_us_median"]
    print(
        f"(1,2880) -> (256,2880) is {out[-1]['numel']/out[0]['numel']:.0f}x the "
        f"elements and {big/small:.1f}x the time: the cost is bytes, "
        f"with a ~{small:.2f} us fixed floor."
    )
    return dict(rows=out)


def print_full_kernel_names(inp: dict, iters: int = 5) -> List[str]:
    """Dump untruncated device-kernel names, to match the trace's template args."""
    for _ in range(3):
        call(inp)
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        for _ in range(iters):
            call(inp)
        torch.xpu.synchronize()
    names = sorted(
        {e.key for e in prof.key_averages() if _is_device_row(e) and "Copy" in e.key}
    )
    print("\n===== full device-kernel names containing 'Copy' =====")
    for n in names:
        print(f"  {n}")
    if not names:
        print("  (none -- the per-forward casts are gone)")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, nargs="+", default=[1, 1024])
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--ids-dtype", choices=["int32", "int64"], default="int32")
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="also sweep bf16->fp32 copy cost vs element count (bytes vs geometry)",
    )
    ap.add_argument(
        "--names", action="store_true", help="dump untruncated device-kernel names"
    )
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    ids_dtype = torch.int32 if args.ids_dtype == "int32" else torch.int64

    print(f"ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')}")
    print(f"torch {torch.__version__}  device {torch.xpu.get_device_name(0)}")

    results = []
    if args.sweep:
        results.append(cast_size_sweep(max(args.iters, 50), args.repeats))
    for m in args.shape:
        inp = build_inputs(m, args.tp, ids_dtype=ids_dtype)
        tag = "decode" if m == 1 else "prefill"
        res = report(
            f"{tag} M={m} tp={args.tp} (I_p={inp['inter_p']}, ids={args.ids_dtype})",
            inp,
            args.iters,
            args.repeats,
        )
        if args.names:
            res["copy_kernel_names"] = print_full_kernel_names(inp)
        res["cast"] = isolated_cast_cost(inp, max(args.iters, 50), args.repeats)
        res["num_tokens"] = m
        res["ids_dtype"] = args.ids_dtype
        results.append(res)
        del inp
        torch.xpu.empty_cache()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
