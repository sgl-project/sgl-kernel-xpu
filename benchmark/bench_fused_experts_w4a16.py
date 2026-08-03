# End-to-end routed-expert MoE benchmark: both routed-expert GEMMs + scatter +
# activation + combine. Compares SGLang's sgl_kernel fused_experts against
# vLLM's XPU XpuFusedMoe (vllm_xpu_kernels) on identical shapes and logical
# weights. Providers:
#
#   mxfp4_fused     sgl_kernel fused_experts(use_mxfp4_w4a16=True).
#   int4_fused      Same-geometry alternative quantization kernel comparison:
#                   sgl_kernel fused_experts(use_int4_w4a16=True), symmetric
#                   signed INT4 packed weights + per-group bf16 scales.
#   vllm_mxfp4      vLLM XpuFusedMoe on the same MXFP4 packed weights.
#   vllm_int4       Same-geometry alternative quantization kernel comparison:
#                   vLLM XpuFusedMoe on INT4 weights in uint4/zp8 packing.
#
# For the W4A16 grouped-GEMM portions, both providers use the Xe2 tiled
# CUTLASS/CuTe-style kernel family. This benchmark compares the resulting
# end-to-end routed-expert integrations.
#
# Shapes are TP=1 / EP=1 routed-expert model geometry:
#   gpt-oss      : experts=32,  topk=4, hidden=2880, intermediate=2880
#   deepseek-v4  : experts=256, topk=6, hidden=4096, intermediate=2048
#
# Methodology mirrors that harness so single runs are not misleading:
#   - deterministic per-route routing (softmax for gpt-oss, sqrt(softplus) +
#     x1.5 for deepseek-v4), regenerated for each --route-seed;
#   - XPU-event windows (inner_repetitions calls / window, repetitions
#     windows) -> per-route median latency;
#   - aggregate = median of the per-route medians over all route seeds.
# A single random route on a synthetic shape swings the sgl path by ~15-20%
# (uneven expert load); the multi-route aggregate removes that artifact.
#
# The sgl and vLLM providers of the same recipe consume the same logical
# weights, so the correctness pass cross-checks their outputs. Both packages
# must be importable in one env (vllm_xpu_kernels alongside sgl_kernel).
#
# Run:
#   python benchmark/bench_fused_experts_w4a16.py                  # GPT-OSS
#   SGL_MOE_BENCH_FULL_SHAPES=1 python benchmark/bench_fused_experts_w4a16.py
#   SGL_MOE_BENCH_FULL_SHAPES=1 python benchmark/bench_fused_experts_w4a16.py --profile deepseek-v4
#   python benchmark/bench_fused_experts_w4a16.py --sgl-only --tokens 1 32
#   python benchmark/bench_fused_experts_w4a16.py --json-out out.json

import gc
import math
import os
import statistics
import sys
from pathlib import Path

import sgl_kernel  # noqa: F401 — registers torch.ops.sgl_kernel
import torch
from sgl_kernel import fused_experts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from mxfp4_utils import MXFP4_BLOCK_SIZE  # noqa: E402
from mxfp4_utils import quantize_mxfp4_2d  # noqa: E402


def _import_vllm_fused_moe():
    import vllm_xpu_kernels._moe_C  # noqa: F401
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

    return XpuFusedMoe


try:
    XpuFusedMoe = _import_vllm_fused_moe()
    VLLM_REF_AVAILABLE = True
except Exception as exc:
    XpuFusedMoe = None
    VLLM_REF_AVAILABLE = False
    print(f"[vllm provider disabled: {type(exc).__name__}: {exc}]")


MODEL_PROFILES = {
    "gpt-oss": {
        "experts": 32,
        "topk": 4,
        "hidden": 2880,
        "intermediate": 2880,
        "activation": "gpt-oss",
        "router": "softmax",
    },
    "deepseek-v4": {
        "experts": 256,
        "topk": 6,
        "hidden": 4096,
        "intermediate": 2048,
        "activation": "deepseek-v4",
        "router": "sqrtsoftplus",
    },
}

# INT4 group size for the int4 providers. Matches the MXFP4 block size.
INT4_GROUP_SIZE = 32

DEFAULT_TOKENS = [1, 32, 2048]
DEFAULT_ROUTE_SEEDS = [0, 1, 2]
DEFAULT_WARMUP = 20
DEFAULT_REPETITIONS = 30
DEFAULT_INNER = 10
WEIGHT_SEED = 0
DEFAULT_CORRECTNESS_REL_TOL = 2e-2
PROVIDER_CHOICES = ["mxfp4_fused", "int4_fused", "vllm_mxfp4", "vllm_int4"]
RUN_FULL_SHAPES = os.environ.get("SGL_MOE_BENCH_FULL_SHAPES") == "1"
ENABLED_PROFILES = list(MODEL_PROFILES) if RUN_FULL_SHAPES else ["gpt-oss"]

ALL_RESULTS = []


def _recipe_for_provider(provider):
    return "int4" if "int4" in provider else "mxfp4"


def _build_and_quantize_weights_per_expert(E: int, rows: int, cols: int):
    # Quantize one expert at a time. A full (E, rows, cols) bf16 tensor
    # followed by `.float()` inside quantize_mxfp4_2d can briefly need tens of
    # GB of host RAM at large shapes and trigger the Linux OOM killer.
    packed = torch.empty((E, rows, cols // 2), dtype=torch.uint8)
    scales = torch.empty((E, rows, cols // MXFP4_BLOCK_SIZE), dtype=torch.uint8)
    for e in range(E):
        w_e_bf16 = torch.empty((rows, cols), dtype=torch.bfloat16).normal_(0, 0.01)
        p_e, s_e = quantize_mxfp4_2d(w_e_bf16.float(), MXFP4_BLOCK_SIZE)
        packed[e].copy_(p_e)
        scales[e].copy_(s_e)
        del w_e_bf16, p_e, s_e
    return packed, scales


def _build_int4_weights_per_expert(E: int, rows: int, cols: int, group_size: int):
    # Symmetric INT4 (no zero-point), one expert at a time. Each group's scale
    # is amax/7; codes are signed 4-bit two's-complement nibbles in [-8, 7] --
    # the layout fused_experts reads when use_int4_w4a16=True with no zp.
    assert cols % group_size == 0
    num_groups = cols // group_size
    packed = torch.empty((E, rows, cols // 2), dtype=torch.int8)
    scales = torch.empty((E, rows, num_groups), dtype=torch.bfloat16)
    for e in range(E):
        w_e = (
            torch.empty((rows, cols), dtype=torch.bfloat16)
            .normal_(0, 0.01)
            .float()
            .reshape(rows, num_groups, group_size)
        )
        amax = w_e.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        s_e = amax / 7.0
        codes = torch.round(w_e / s_e).clamp_(-8, 7).to(torch.int16).reshape(rows, cols)
        nibbles = torch.bitwise_and(codes, 0xF)
        p_e = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)
        packed[e].copy_(p_e.view(torch.int8))
        scales[e].copy_(s_e.reshape(rows, num_groups).to(torch.bfloat16))
        del w_e, amax, s_e, codes, nibbles, p_e
    return packed, scales


def _build_int4_weights_vllm_per_expert(E: int, rows: int, cols: int, group_size: int):
    # Same symmetric INT4 weights as _build_int4_weights_per_expert (same seed,
    # same draw order, scale = amax/7, values in [-8, 7]) but re-encoded into
    # vLLM's uint4 / zero-point-8 packing: unsigned nibbles in [0, 15] with
    # dequant (code - 8) * scale. XpuFusedMoe folds the zero point to signed s4
    # internally, so return the *unfolded* uint8 codes.
    assert cols % group_size == 0
    num_groups = cols // group_size
    packed = torch.empty((E, rows, cols // 2), dtype=torch.uint8)
    scales = torch.empty((E, rows, num_groups), dtype=torch.bfloat16)
    for e in range(E):
        w_e = (
            torch.empty((rows, cols), dtype=torch.bfloat16)
            .normal_(0, 0.01)
            .float()
            .reshape(rows, num_groups, group_size)
        )
        amax = w_e.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        s_e = amax / 7.0
        codes = (
            (torch.round(w_e / s_e).clamp_(-8, 7) + 8)
            .to(torch.int32)
            .reshape(rows, cols)
        )
        p_e = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
        packed[e].copy_(p_e)
        scales[e].copy_(s_e.reshape(rows, num_groups).to(torch.bfloat16))
        del w_e, amax, s_e, codes, p_e
    return packed, scales


def _build_weights(profile, recipe, backend):
    """Build the resident weights for one (profile, recipe, backend) on XPU.

    sgl and vLLM draw the same seeded source weights, so a recipe's two
    backends operate on identical logical weights. For the vLLM backend an
    XpuFusedMoe object is constructed once (it may fold zero points in place)
    and reused across all route seeds.
    """
    torch.manual_seed(WEIGHT_SEED)
    torch.xpu.manual_seed_all(WEIGHT_SEED)
    E = profile["experts"]
    hidden = profile["hidden"]
    inter = profile["intermediate"]
    print(
        f"[weights] quantizing {backend}/{recipe}: " f"E={E}, H={hidden}, I={inter}",
        flush=True,
    )

    if recipe == "int4":
        builder = (
            _build_int4_weights_vllm_per_expert
            if backend == "vllm"
            else _build_int4_weights_per_expert
        )
        w1_packed_cpu, w1_scale = builder(E, 2 * inter, hidden, INT4_GROUP_SIZE)
        w2_packed_cpu, w2_scale = builder(E, hidden, inter, INT4_GROUP_SIZE)
        w1_packed = w1_packed_cpu.to("xpu")
        w2_packed = w2_packed_cpu.to("xpu")
        w1_scale = w1_scale.to("xpu")
        w2_scale = w2_scale.to("xpu")
    else:
        w1_packed_cpu, w1_scale_cpu = _build_and_quantize_weights_per_expert(
            E, 2 * inter, hidden
        )
        w2_packed_cpu, w2_scale_cpu = _build_and_quantize_weights_per_expert(
            E, hidden, inter
        )
        w1_scale = w1_scale_cpu.contiguous().to("xpu")
        w2_scale = w2_scale_cpu.contiguous().to("xpu")
        if backend == "vllm":
            w1_packed = w1_packed_cpu.contiguous().to("xpu")
            w2_packed = w2_packed_cpu.contiguous().to("xpu")
        else:
            w1_packed = w1_packed_cpu.view(torch.int8).to("xpu")
            w2_packed = w2_packed_cpu.view(torch.int8).to("xpu")

    weights = {
        "w1_packed": w1_packed,
        "w2_packed": w2_packed,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "recipe": recipe,
        "backend": backend,
        "profile": profile,
    }
    if backend == "vllm":
        weights["moe"] = _build_vllm_moe(weights, profile, recipe)
    print(f"[weights] ready {backend}/{recipe} on XPU", flush=True)
    return weights


def _build_vllm_moe(weights, profile, recipe):
    # Construct XpuFusedMoe ONCE (outside the timed loop). For INT4 the
    # constructor folds the zero point into the packed weights in place, so it
    # must not be rebuilt per call. Activation matches the profile: gpt-oss
    # uses the fused swigluoai epilogue; deepseek-v4 uses silu + a gemm1 clamp.
    if recipe == "mxfp4":
        w13 = weights["w1_packed"].view(torch.float4_e2m1fn_x2)
        w2 = weights["w2_packed"].view(torch.float4_e2m1fn_x2)
        w13_scales = weights["w1_scale"].view(torch.float8_e8m0fnu)
        w2_scales = weights["w2_scale"].view(torch.float8_e8m0fnu)
    else:  # int4
        w13 = weights["w1_packed"]
        w2 = weights["w2_packed"]
        w13_scales = weights["w1_scale"]
        w2_scales = weights["w2_scale"]

    moe_kwargs = {}
    if profile["activation"] == "gpt-oss":
        moe_kwargs["activation"] = "swigluoai"
    elif profile["activation"] == "deepseek-v4":
        moe_kwargs["gemm1_clamp_limit"] = 10.0
    return XpuFusedMoe(
        w13=w13,
        w13_scales=w13_scales,
        w13_bias=None,
        w2=w2,
        w2_scales=w2_scales,
        w2_bias=None,
        n_experts_per_token=profile["topk"],
        activation=moe_kwargs.pop("activation", "silu"),
        num_experts=profile["experts"],
        **moe_kwargs,
    )


def _make_routing(tokens, profile, seed):
    """Deterministic per-route router outputs (mirrors the fair harness).

    A CPU generator seeded by (seed + tokens) makes the route byte-for-byte
    reproducible and independent of the weight RNG. deepseek-v4 uses the
    sqrt(softplus) router and scales the top-k weights by 1.5; gpt-oss uses
    plain softmax. Returns CPU tensors; the caller casts to the dtypes each
    backend expects.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + tokens)
    logits = torch.randn(tokens, profile["experts"], generator=generator)
    if profile["router"] == "softmax":
        scores = torch.softmax(logits, dim=-1)
    else:
        scores = torch.sqrt(torch.nn.functional.softplus(logits))
    topk_scores, topk_ids = torch.topk(scores, profile["topk"], dim=-1)
    topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
    if profile["activation"] == "deepseek-v4":
        topk_weights.mul_(1.5)
    return topk_ids, topk_weights


def _make_activations(tokens, profile, seed):
    # Deterministic bf16 activations for one route (seeded independently of the
    # weight RNG so the route + input are reproducible).
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1000 + seed + tokens)
    x = torch.randn(tokens, profile["hidden"], generator=generator)
    return x.to(torch.bfloat16).to("xpu")


def _routing_for_backend(topk_ids_cpu, topk_weights_cpu, backend):
    if backend == "vllm":
        return {
            "topk_ids": topk_ids_cpu.to(dtype=torch.int32, device="xpu"),
            "topk_weights": topk_weights_cpu.to(dtype=torch.float32, device="xpu"),
        }
    return {
        "topk_ids": topk_ids_cpu.to(dtype=torch.int64, device="xpu"),
        "topk_weights": topk_weights_cpu.to(dtype=torch.bfloat16, device="xpu"),
    }


def _sgl_activation_kwargs(profile):
    if profile["activation"] == "gpt-oss":
        return {"gemm1_alpha": 1.702, "gemm1_limit": 7.0}
    if profile["activation"] == "deepseek-v4":
        return {"swiglu_limit": 10.0}
    return {}


def _sgl_run(weights, x, routing, profile):
    recipe = weights["recipe"]
    quant_kwargs = (
        {"use_int4_w4a16": True} if recipe == "int4" else {"use_mxfp4_w4a16": True}
    )
    return fused_experts(
        x,
        weights["w1_packed"],
        weights["w2_packed"],
        routing["topk_weights"],
        routing["topk_ids"],
        None,
        None,
        activation="silu",
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
        **quant_kwargs,
        **_sgl_activation_kwargs(profile),
    )


def _vllm_run(weights, x, routing):
    output = torch.empty_like(x)
    weights["moe"].apply(
        output=output,
        hidden_states=x,
        topk_weights=routing["topk_weights"],
        topk_ids=routing["topk_ids"],
    )
    return output


def _make_run_fn(weights, x, routing, profile):
    if weights["backend"] == "vllm":
        return lambda: _vllm_run(weights, x, routing)
    return lambda: _sgl_run(weights, x, routing, profile)


def _time_median(run_fn, warmup, repetitions, inner):
    """Per-route median latency (ms) via XPU-event windows.

    Each window times `inner` back-to-back calls and divides by `inner`;
    the median over `repetitions` windows is returned. This matches the fair
    harness timing loop.
    """
    for _ in range(warmup):
        run_fn()
    torch.xpu.synchronize()
    timings = []
    for _ in range(repetitions):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            run_fn()
        end.record()
        torch.xpu.synchronize()
        timings.append(start.elapsed_time(end) / inner)
    return statistics.median(timings)


def _summarize(per_route_medians):
    m = per_route_medians
    if len(m) == 1:
        return {
            "ms": round(m[0], 4),
            "ms_mean": round(m[0], 4),
            "ms_min": round(m[0], 4),
            "ms_max": round(m[0], 4),
            "ms_stddev": 0.0,
        }
    return {
        "ms": round(statistics.median(m), 4),
        "ms_mean": round(statistics.mean(m), 4),
        "ms_min": round(min(m), 4),
        "ms_max": round(max(m), 4),
        "ms_stddev": round(statistics.stdev(m), 4),
    }


def benchmark_case(
    profile_name, tokens, provider, route_seeds, warmup, reps, inner, weights=None
):
    profile = MODEL_PROFILES[profile_name]
    recipe = _recipe_for_provider(provider)
    backend = "vllm" if provider.startswith("vllm_") else "sgl"

    gc.collect()
    torch.xpu.empty_cache()

    base_row = {
        "profile": profile_name,
        "tokens": tokens,
        "provider": provider,
        "experts": profile["experts"],
        "topk": profile["topk"],
        "hidden": profile["hidden"],
        "intermediate": profile["intermediate"],
    }

    print(
        f"[run] {profile_name} tokens={tokens} provider={provider}: "
        f"{len(route_seeds)} route seeds, warmup={warmup}, reps={reps}",
        flush=True,
    )

    owns_weights = weights is None
    if owns_weights:
        weights = _build_weights(profile, recipe, backend)
    per_route = []
    for seed in route_seeds:
        x = _make_activations(tokens, profile, seed)
        ids_cpu, w_cpu = _make_routing(tokens, profile, seed)
        routing = _routing_for_backend(ids_cpu, w_cpu, backend)
        run_fn = _make_run_fn(weights, x, routing, profile)
        per_route.append(_time_median(run_fn, warmup, reps, inner))
        del x, routing
    if owns_weights:
        del weights
    gc.collect()
    torch.xpu.empty_cache()

    stats = _summarize(per_route)
    # MoE flop ~ tokens * topk * 6 * hidden * intermediate (2 GEMMs).
    flop = tokens * profile["topk"] * 6 * profile["hidden"] * profile["intermediate"]
    tflops = flop / (stats["ms"] / 1e3) / 1e12

    row = dict(base_row)
    row.update(stats)
    row["tflops"] = round(tflops, 2)
    row["route_seeds"] = route_seeds
    ALL_RESULTS.append(row)
    print(
        f"[{profile_name:<12s} {tokens:>4d}tok {provider:<11s}] "
        f"median={stats['ms']:.4f} ms  mean={stats['ms_mean']:.4f}  "
        f"min={stats['ms_min']:.4f}  max={stats['ms_max']:.4f}  "
        f"stddev={stats['ms_stddev']:.4f}  ({len(route_seeds)} routes)"
    )


def _correctness_check(
    profiles,
    tokens_list,
    recipes,
    route_seed=0,
    rel_tol=DEFAULT_CORRECTNESS_REL_TOL,
):
    """Cross-check sgl_kernel vs vLLM MoE outputs on the same logical weights.

    Builds each (profile, recipe, backend) weight set once and reuses it across
    token counts. The full MoE layer still runs once per token count outside
    the timed loop. sgl_kernel and vLLM implement the same activation + combine,
    so a correct pair stays within a few percent.
    """
    if not VLLM_REF_AVAILABLE:
        print("[correctness skipped: vllm_xpu_kernels unavailable]")
        return {}
    rows = []
    results = {}
    failures = []
    for profile_name in profiles:
        profile = MODEL_PROFILES[profile_name]
        for recipe in recipes:
            rel_by_tokens = {}
            sgl_outputs = {}
            sgl_w = None
            vllm_w = None
            try:
                sgl_w = _build_weights(profile, recipe, "sgl")
                for tokens in tokens_list:
                    x = _make_activations(tokens, profile, route_seed)
                    ids_cpu, w_cpu = _make_routing(tokens, profile, route_seed)
                    sgl_outputs[tokens] = _sgl_run(
                        sgl_w, x, _routing_for_backend(ids_cpu, w_cpu, "sgl"), profile
                    ).float()
                    del x
                torch.xpu.synchronize()
                sgl_w = None
                gc.collect()
                torch.xpu.empty_cache()

                vllm_w = _build_weights(profile, recipe, "vllm")
                for tokens in tokens_list:
                    x = _make_activations(tokens, profile, route_seed)
                    ids_cpu, w_cpu = _make_routing(tokens, profile, route_seed)
                    vllm_out = _vllm_run(
                        vllm_w, x, _routing_for_backend(ids_cpu, w_cpu, "vllm")
                    ).float()
                    torch.xpu.synchronize()
                    diff = (sgl_outputs[tokens] - vllm_out).norm().item()
                    denom = vllm_out.norm().clamp_min(1e-6).item()
                    rel = round(diff / denom, 5)
                    rel_by_tokens[tokens] = rel
                    if not math.isfinite(rel) or rel > rel_tol:
                        failures.append(
                            f"{profile_name}/{tokens}tok/{recipe}: {rel} exceeds {rel_tol}"
                        )
                    del vllm_out, sgl_outputs[tokens], x
                vllm_w = None
            except Exception as exc:  # noqa: BLE001 — report, don't abort
                error = f"ERR:{type(exc).__name__}"
                for tokens in tokens_list:
                    if tokens not in rel_by_tokens:
                        rel_by_tokens[tokens] = error
                        failures.append(f"{profile_name}/{tokens}tok/{recipe}: {error}")
            finally:
                sgl_w = None
                vllm_w = None
                sgl_outputs.clear()
                gc.collect()
                torch.xpu.empty_cache()

            for tokens in tokens_list:
                rel = rel_by_tokens[tokens]
                row = {
                    "profile": profile_name,
                    "tokens": tokens,
                    "recipe": recipe,
                    "rel_l2_err": rel,
                }
                rows.append(row)
                results[(profile_name, tokens, recipe)] = rel
    import pandas as pd

    print("\n[correctness: sgl_kernel vs vLLM (same logical weights)]")
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()
    if failures:
        raise RuntimeError(
            "Correctness check failed; refusing to report benchmark timings: "
            + "; ".join(failures)
        )
    return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=ENABLED_PROFILES,
        default=None,
        help=(
            "Run only this enabled model profile. Set "
            "SGL_MOE_BENCH_FULL_SHAPES=1 to enable deepseek-v4."
        ),
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=DEFAULT_TOKENS,
        help=f"Token counts to sweep (default: {DEFAULT_TOKENS}).",
    )
    parser.add_argument(
        "--route-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_ROUTE_SEEDS,
        help=f"Synthetic route seeds to aggregate (default: {DEFAULT_ROUTE_SEEDS}).",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--inner-repetitions", type=int, default=DEFAULT_INNER)
    parser.add_argument(
        "--correctness-rel-tol",
        type=float,
        default=DEFAULT_CORRECTNESS_REL_TOL,
        help=(
            "Maximum cross-provider relative L2 error before timing aborts "
            f"(default: {DEFAULT_CORRECTNESS_REL_TOL})."
        ),
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Dump benchmark rows and their correctness result as JSON.",
    )
    parser.add_argument(
        "--sgl-only",
        action="store_true",
        help="Skip the vLLM reference providers (run only sgl_kernel).",
    )
    parser.add_argument(
        "--providers",
        choices=PROVIDER_CHOICES,
        nargs="+",
        default=None,
        help="Run only these providers (default: all available providers).",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip the sgl_kernel-vs-vLLM correctness gate before timing.",
    )
    args = parser.parse_args()

    profiles = [args.profile] if args.profile else ENABLED_PROFILES
    print(
        f"[config] fused-experts shape set: "
        f"{'full' if RUN_FULL_SHAPES else 'quick'} (profiles={profiles})",
        flush=True,
    )
    providers = args.providers or ["mxfp4_fused", "int4_fused"]
    if args.providers is None and VLLM_REF_AVAILABLE and not args.sgl_only:
        providers += ["vllm_mxfp4", "vllm_int4"]
    if args.sgl_only:
        providers = [
            provider for provider in providers if not provider.startswith("vllm_")
        ]
    if not providers:
        parser.error("no providers remain after applying --sgl-only")
    if not VLLM_REF_AVAILABLE and any(
        provider.startswith("vllm_") for provider in providers
    ):
        parser.error(
            "a vLLM provider was requested but vllm_xpu_kernels is unavailable"
        )
    providers = list(dict.fromkeys(providers))

    correctness_results = {}
    if not args.sgl_only and not args.skip_correctness:
        correctness_recipes = []
        for recipe, sgl_provider, vllm_provider in (
            ("mxfp4", "mxfp4_fused", "vllm_mxfp4"),
            ("int4", "int4_fused", "vllm_int4"),
        ):
            if sgl_provider in providers and vllm_provider in providers:
                correctness_recipes.append(recipe)
        if correctness_recipes:
            correctness_results = _correctness_check(
                profiles,
                args.tokens,
                correctness_recipes,
                route_seed=args.route_seeds[0],
                rel_tol=args.correctness_rel_tol,
            )
        else:
            print("[correctness skipped: selected providers contain no backend pair]")

    for profile_name in profiles:
        profile = MODEL_PROFILES[profile_name]
        for provider in providers:
            recipe = _recipe_for_provider(provider)
            backend = "vllm" if provider.startswith("vllm_") else "sgl"
            print(
                f"[setup] profile={profile_name} provider={provider}: "
                f"reusing weights across tokens={args.tokens}",
                flush=True,
            )
            weights = _build_weights(profile, recipe, backend)
            for tokens in args.tokens:
                benchmark_case(
                    profile_name,
                    tokens,
                    provider,
                    args.route_seeds,
                    args.warmup,
                    args.repetitions,
                    args.inner_repetitions,
                    weights,
                )
            del weights
            gc.collect()
            torch.xpu.empty_cache()

    for row in ALL_RESULTS:
        recipe = _recipe_for_provider(row["provider"])
        row["correctness_rel_l2_err"] = correctness_results.get(
            (row["profile"], row["tokens"], recipe), "not_run"
        )

    if args.json_out is not None:
        with open(args.json_out, "w") as f:
            json.dump(ALL_RESULTS, f, indent=2)
        print(f"[wrote {len(ALL_RESULTS)} rows -> {args.json_out}]")
        raise SystemExit(0)

    print("\nBenchmark finished!\n")
    import pandas as pd

    df = pd.DataFrame(ALL_RESULTS)
    pv = df.pivot_table(
        index=["profile", "tokens", "experts", "topk", "hidden", "intermediate"],
        columns="provider",
        values=["ms", "ms_stddev", "tflops"],
        aggfunc="first",
    )
    # sgl_kernel vs vLLM, per recipe. >1 means sgl_kernel is faster than vLLM.
    if ("ms", "mxfp4_fused") in pv.columns and ("ms", "vllm_mxfp4") in pv.columns:
        pv["speedup_mxfp4_sgl_vs_vllm"] = (
            pv[("ms", "vllm_mxfp4")] / pv[("ms", "mxfp4_fused")]
        ).round(2)
    if ("ms", "int4_fused") in pv.columns and ("ms", "vllm_int4") in pv.columns:
        pv["speedup_int4_sgl_vs_vllm"] = (
            pv[("ms", "vllm_int4")] / pv[("ms", "int4_fused")]
        ).round(2)
    # sgl_kernel INT4 vs MXFP4 head to head. >1 means int4_fused is faster.
    if ("ms", "mxfp4_fused") in pv.columns and ("ms", "int4_fused") in pv.columns:
        pv["speedup_int4_vs_mxfp4"] = (
            pv[("ms", "mxfp4_fused")] / pv[("ms", "int4_fused")]
        ).round(2)
    print(pv.to_markdown())
