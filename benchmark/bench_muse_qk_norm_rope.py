from muse_scope import require_muse_arch_xe35, require_muse_b8_xe35

require_muse_b8_xe35()

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from sgl_kernel import fused_qk_norm_rope

DTYPE_MAP = {"bf16": torch.bfloat16}
parser = argparse.ArgumentParser(
    description="B8 production SOURCE versus PORT graph benchmark"
)
parser.add_argument("--tokens", type=int, choices=(8, 32768), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.tokens = [args.tokens]
args.repeats, args.samples, args.warmup = 32, 21, 5
args.q_heads, args.k_heads, args.tail_heads, args.head_dim = 128, 16, 144, 64
args.dtype = "bf16"
args.profile = False


def benchmark_graph_xe35():
    require_muse_arch_xe35()
    torch.manual_seed(42)
    tokens, repeats, samples, warmup = (
        args.tokens[0],
        args.repeats,
        args.samples,
        args.warmup,
    )
    nq, nk, nv, hd = args.q_heads, args.k_heads, args.tail_heads, args.head_dim
    # Muse-Spark B8 uses T8/T32768, Q128/K16/D64 and gated tail144; other dimensions are generic corners.
    dtype = DTYPE_MAP[args.dtype]
    scale = 43.7840518911 / (hd**0.5)
    qkv = torch.randn(tokens, (nq + nk + nv) * hd, device="xpu:0", dtype=dtype)
    initial = qkv.clone()
    q_weight = torch.full((hd,), scale, device="xpu:0", dtype=dtype)
    k_weight = torch.ones(hd, device="xpu:0", dtype=dtype)
    positions = torch.arange(tokens, device="xpu:0", dtype=torch.int32) % 4096

    def fn():
        fused_qk_norm_rope(
            qkv,
            nq,
            nk,
            nv,
            hd,
            1e-5,
            q_weight,
            k_weight,
            500000.0,
            False,
            positions,
            rotary_dim=hd,
        )

    if args.profile:
        median_ms, min_ms, max_ms = profile_fused_qk_norm_rope(fn, samples=samples)
        print(
            f"profiler p50_us={median_ms * 1000:.3f} "
            f"min_us={min_ms * 1000:.3f} max_us={max_ms * 1000:.3f}"
        )
    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    qkv.copy_(initial)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(repeats):
            fn()
    torch.xpu.synchronize()

    paths = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libsgl-ops-sycl-FusedQKNormRope" in line
    }
    loaded_libraries = {}
    for path in sorted(paths):
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        loaded_libraries[path] = digest
        print(f"loaded_lib={path} sha256={digest}", flush=True)

    for _ in range(warmup):
        qkv.copy_(initial)
        graph.replay()
    torch.xpu.synchronize()
    values = []
    for _ in range(samples):
        qkv.copy_(initial)
        torch.xpu.synchronize()
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / repeats)

    ordered = sorted(values)
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    p20 = ordered[int((samples - 1) * 0.2)]
    p80 = ordered[int((samples - 1) * 0.8)]
    device_name = torch.xpu.get_device_name(0)
    print(
        f"graph tokens={tokens} dtype={args.dtype} heads={nq},{nk},{nv} hd={hd} "
        f"repeats={repeats} samples={samples} warmup={warmup} "
        f"device={device_name}",
        flush=True,
    )
    print(
        f"tokens={tokens} raw_us={','.join(f'{value:.3f}' for value in values)} "
        f"p50_us={median:.3f} mad_us={mad:.3f} p20_us={p20:.3f} p80_us={p80:.3f}",
        flush=True,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    mode="graph",
                    tokens=tokens,
                    dtype=args.dtype,
                    heads=[nq, nk, nv],
                    head_dim=hd,
                    repeats=repeats,
                    samples=samples,
                    warmup=warmup,
                    device=device_name,
                    loaded_libraries=loaded_libraries,
                    raw_us=values,
                    p50_us=median,
                    mad_us=mad,
                    p20_us=p20,
                    p80_us=p80,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    benchmark_graph_xe35()
