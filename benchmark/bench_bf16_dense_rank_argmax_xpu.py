"""Paired projection/argmax benchmark with independent rotating weight banks.

No checkpoint is downloaded. The default uses random BF16 weights and uncapped
BF16 argmax. To reproduce a serving softcap comparison, install SGLang and pass
--softcap 30; that mode calls the actual published softcap/rank helpers.

Example trained-head run (the safetensors file must already exist locally)::

    ZE_AFFINITY_MASK=1 python benchmark/bench_bf16_dense_rank_argmax_xpu.py \
        --weight-file /local/model.safetensors --softcap 30 --output results.json

Timings include projection and the final ID reduction, exclude table setup,
rotate physical weight allocations, and do not measure end-to-end serving.
"""

import argparse
import hashlib
import json
import pathlib
import statistics

import torch
import torch.nn.functional as F
from sgl_kernel import bf16_dense_rank_argmax_xe20


def make_comparison(device, softcap):
    if softcap is not None:
        from sglang.kernels.ops.sampling import softcap_argmax
        from sglang.srt.speculative.deferred_verify_logits import (
            make_verify_softcap_ranks,
        )

        ranks = make_verify_softcap_ranks(device, softcap)
        unsigned = ranks.to(torch.int64) & 0xFFFFFFFF

        def baseline(a, weight):
            return softcap_argmax(
                F.linear(a, weight), softcap=softcap, comparison_ranks=ranks
            )

        label = "torch_head_published_softcap_lut"
    else:
        # Order raw BF16 values by integer bits, merging signed zero and NaNs.
        # This also preserves the class order of nonzero BF16 subnormals.
        bits = torch.arange(65536, device=device, dtype=torch.int64)
        magnitude = bits & 0x7FFF
        unsigned = torch.where(bits & 0x8000 != 0, (~bits) & 0xFFFF, bits ^ 0x8000)
        unsigned = torch.where(magnitude == 0, 0x8000, unsigned)
        unsigned = torch.where(magnitude > 0x7F80, 0xFFFF, unsigned)

        def baseline(a, weight):
            return F.linear(a, weight).argmax(dim=-1)

        label = "torch_head_bf16_argmax"
    classes, inverse = torch.unique(unsigned, sorted=True, return_inverse=True)
    table = inverse.to(torch.uint16).contiguous()
    assert torch.equal(classes[inverse], unsigned)
    return table, baseline, label, classes.numel()


def capture(fn, banks, cycles):
    for i in range(2 * len(banks)):
        fn(banks[i % len(banks)])
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    outputs = []
    with torch.xpu.graph(graph):
        for _ in range(cycles):
            for bank in banks:
                outputs.append(fn(bank))
    return graph, outputs


def measure(graph, calls, replays):
    for _ in range(2):
        graph.replay()
    torch.xpu.synchronize()
    start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (calls * replays)


def run(args):
    assert torch.xpu.is_available(), "requires Xe2 XPU"
    assert args.bank_count >= 2 and args.cycles > 0 and args.replays > 0
    assert all(m > 0 for m in args.m)
    device = torch.device("xpu", torch.xpu.current_device())
    torch.manual_seed(args.seed)
    if args.weight_file:
        from safetensors import safe_open

        path = pathlib.Path(args.weight_file)
        if not path.is_file():
            raise FileNotFoundError(path)
        with safe_open(path, framework="pt", device="cpu") as file:
            weight = file.get_tensor(args.weight_key)
        if weight.dtype != torch.bfloat16 or weight.ndim != 2:
            raise ValueError("local checkpoint weight must be a BF16 matrix")
        first = weight.to(device).contiguous()
    else:
        first = torch.randn(args.n, args.k, device=device, dtype=torch.bfloat16) * 0.02
    banks = [first] + [first.clone() for _ in range(args.bank_count - 1)]
    pointers = [bank.untyped_storage().data_ptr() for bank in banks]
    assert len(set(pointers)) == len(
        pointers
    ), "weight banks must be independent allocations"
    n, k = first.shape
    table, baseline, label, class_count = make_comparison(device, args.softcap)
    import sgl_kernel

    library = pathlib.Path(sgl_kernel.common_ops.__file__)
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    result = {
        "device": torch.xpu.get_device_name(),
        "torch_version": torch.__version__,
        "triton_version": triton_version,
        "native_library": str(library),
        "native_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "weight_file": args.weight_file,
        "weight_key": args.weight_key if args.weight_file else None,
        "softcap": args.softcap,
        "comparison_patterns": 65536,
        "comparison_classes": class_count,
        "physical_weight_banks": len(banks),
        "rotating_weight_bytes": sum(
            bank.numel() * bank.element_size() for bank in banks
        ),
        "distinct_weight_allocations": True,
        "scope": "isolated device head timing; native and Torch accumulation may differ",
        "cases": [],
    }
    for m in args.m:
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        operations = {
            label: lambda weight: baseline(a, weight),
            "native_rank_argmax": lambda weight: bf16_dense_rank_argmax_xe20(
                a, weight, table
            ),
        }
        graphs = {
            name: capture(fn, banks, args.cycles) for name, fn in operations.items()
        }
        # Poison output and change inputs: an empty/stale capture must fail.
        for name, (graph, outputs) in graphs.items():
            for sign in (-1, -1):
                a.mul_(sign)
                expected = [operations[name](bank) for bank in banks]
                for output in outputs:
                    output.fill_(-123)
                graph.replay()
                torch.xpu.synchronize()
                for index, output in enumerate(outputs):
                    torch.testing.assert_close(
                        output, expected[index % len(banks)], atol=0, rtol=0
                    )
        native_ids = operations["native_rank_argmax"](banks[0])
        torch_ids = operations[label](banks[0])
        mismatches = int((native_ids != torch_ids).sum().item())
        order = (label, "native_rank_argmax", "native_rank_argmax", label)
        rounds = [
            {
                "label": name,
                "device_us": measure(
                    graphs[name][0], len(banks) * args.cycles, args.replays
                ),
            }
            for name in order
        ]
        means = {
            name: statistics.mean(
                row["device_us"] for row in rounds if row["label"] == name
            )
            for name in operations
        }
        result["cases"].append(
            {
                "M": m,
                "N": n,
                "K": k,
                "capture_reads_changed_input_and_restores_poison": True,
                "native_vs_torch_winner_mismatches": mismatches,
                "rounds": rounds,
                "means_us": means,
                "native_lower_percent": 100
                * (1 - means["native_rank_argmax"] / means[label]),
            }
        )
        if args.output:
            pathlib.Path(args.output).write_text(json.dumps(result, indent=2))
        del graphs
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weight-file", help="existing local safetensors checkpoint; never downloaded"
    )
    parser.add_argument(
        "--weight-key", default="model.language_model.embed_tokens.weight"
    )
    parser.add_argument("--m", type=int, nargs="+", default=[4, 32, 128, 256])
    parser.add_argument("--n", type=int, default=262144)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--bank-count", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--softcap",
        type=float,
        help="use actual SGLang softcap/LUT helpers (requires SGLang)",
    )
    parser.add_argument("--output")
    with torch.inference_mode():
        run(parser.parse_args())
