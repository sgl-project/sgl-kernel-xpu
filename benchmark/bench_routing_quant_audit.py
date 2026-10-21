"""B8 routing and MXFP4 device-time probes through one CLI."""

import argparse
import json
import statistics
import time

GRAPH_REPLAY = 1


def parse_args_xe35(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "padded-gather"), default="audit")
    parser.add_argument("--op", choices=("all", "quant", "route"), default="all")
    parser.add_argument(
        "--serving-batch", "--batch", dest="serving_batch", type=int, default=8
    )
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--osl", type=int, default=16)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--branch", action="store_true")
    parser.add_argument("--triton-route", action="store_true")
    parser.add_argument("--quant-only", action="store_true")
    parser.add_argument("--lane-only", action="store_true")
    parser.add_argument("--fused-only", action="store_true")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--rep", type=int, default=21)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--graph-replay", type=int, default=1)
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=int,
        default=(64, 128, 252, 256, 260, 368, 372, 376, 512, 1024),
    )
    parser.add_argument("--tokens", nargs="+", type=int)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--alignment", type=int, default=4)
    args = parser.parse_args(argv)
    if args.serving_batch != 8:
        parser.error("Muse-Spark serving benchmarks require --serving-batch 8")
    if (args.isl, args.osl, args.tp, args.ep) != (4096, 16, 1, 1):
        parser.error("serving scope requires ISL4096/OSL16/TP1/EP1")
    if args.warmup is None:
        args.warmup = 3 if args.mode == "padded-gather" else 5
    if args.warmup < 0 or args.rep < 1 or args.repeat < 1 or args.graph_replay < 1:
        parser.error("warmup must be nonnegative; repeat counts must be positive")
    if args.hidden < 1 or args.hidden % 32 or any(rows < 1 for rows in args.shapes):
        parser.error(
            "hidden must be positive and divisible by32; shapes must be positive"
        )
    if args.tokens is None:
        args.tokens = (8, 8 * args.isl)
    if any(tokens < 1 for tokens in args.tokens):
        parser.error("--tokens values must be positive")
    if (
        args.topk < 1
        or args.experts < 1
        or args.alignment < 1
        or args.topk > args.experts
    ):
        parser.error("topk/alignment must be positive, with topk <= experts")
    if args.mode == "padded-gather" and args.op == "route":
        parser.error("padded-gather mode does not benchmark routing")
    if args.mode == "audit" and (args.topk, args.experts, args.alignment) != (
        8,
        256,
        4,
    ):
        parser.error("audit mode requires topk8/experts256/alignment4")
    if args.op == "route" and (args.quant_only or args.lane_only or args.fused_only):
        parser.error("quantization-only flags are incompatible with --op route")
    args.quant_only = args.quant_only or args.op == "quant"
    return args


def check_route_xe35(weights, ids, logits, bias, multiplier):
    scores = torch.sigmoid((logits * multiplier).float())
    rank_scores = scores + bias
    assert bool(((ids >= 0) & (ids < logits.size(1))).all())
    assert bool((torch.diff(torch.sort(ids, dim=-1).values, dim=-1) != 0).all())
    torch.testing.assert_close(
        rank_scores.gather(1, ids.long()),
        torch.topk(rank_scores, ids.size(1), dim=-1, sorted=True).values,
        rtol=0,
        atol=0,
    )
    selected = scores.gather(1, ids.long())
    torch.testing.assert_close(
        weights,
        selected / (selected.sum(dim=-1, keepdim=True) + 1e-15),
        rtol=3e-4,
        atol=3e-4,
    )


def bench_xe35(name, fn, warmup, rep):
    global GRAPH_REPLAY
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    if GRAPH_REPLAY > 1:
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            fn()
        torch.xpu.synchronize()
    samples = []
    for _ in range(rep):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        if GRAPH_REPLAY > 1:
            for _ in range(GRAPH_REPLAY):
                graph.replay()
        else:
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / GRAPH_REPLAY)
    ordered = sorted(samples)
    print(
        json.dumps(
            {
                "name": name,
                "graph_replay": GRAPH_REPLAY,
                "samples_us": samples,
                "p20_us": ordered[round(0.2 * (len(ordered) - 1))],
                "p50_us": statistics.median(ordered),
                "p80_us": ordered[round(0.8 * (len(ordered) - 1))],
            }
        ),
        flush=True,
    )


def measure_padded_xe35(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    samples = []
    for _ in range(repeat):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def baseline_xe35(hidden_states, row_map, rows_per_expert):
    gathered = hidden_states.index_select(0, row_map)
    output_q, scales_rm = sgl_per_token_group_quant_fp4_xe35(gathered, group_size=32)
    scales_mn = torch.empty_like(scales_rm)
    mxfp4_moe_scale_to_mn_major_xe35(scales_mn, scales_rm, rows_per_expert)
    return output_q, scales_mn


def gather_quant_relayout_xe35(hidden_states, row_map, rows_per_expert):
    output_q, scales_rm = sgl_padded_gather_per_token_group_quant_fp4_xe35(
        hidden_states, row_map, group_size=32
    )
    scales_mn = torch.empty_like(scales_rm)
    mxfp4_moe_scale_to_mn_major_xe35(scales_mn, scales_rm, rows_per_expert)
    return output_q, scales_mn


def gather_quant_only_xe35(hidden_states, row_map):
    return sgl_padded_gather_per_token_group_quant_fp4_xe35(
        hidden_states, row_map, group_size=32
    )


def relayout_only_xe35(scales_rm, rows_per_expert):
    scales_mn = torch.empty_like(scales_rm)
    mxfp4_moe_scale_to_mn_major_xe35(scales_mn, scales_rm, rows_per_expert)
    return scales_mn


def swiglu_quant_relayout_xe35(gate_up, rows_per_expert):
    output_q, scales_rm = sgl_per_token_group_quant_fp4_xe35(
        gate_up, group_size=32, fuse_silu_and_mul=True
    )
    scales_mn = torch.empty_like(scales_rm)
    mxfp4_moe_scale_to_mn_major_xe35(scales_mn, scales_rm, rows_per_expert)
    return output_q, scales_mn


def logical_bytes_xe35(rows, hidden, path):
    # Count each logical stream once. The baseline_xe35 first reads and writes a
    # gathered BF16 surface, then reads it for quantization. Both paths write
    # packed FP4 values and one UE8M0 byte per group of 32 values. The legacy
    # paths additionally read and rewrite the row-major scale surface.
    bf16_bytes = 2
    fp4_bytes = 0.5
    scale_bytes = 1 / 32
    if path == "baseline_xe35":
        streams_per_element = 3 * bf16_bytes + fp4_bytes + 3 * scale_bytes
    elif path == "gather_relayout":
        streams_per_element = bf16_bytes + fp4_bytes + 3 * scale_bytes
    elif path == "swiglu_relayout":
        streams_per_element = 2 * bf16_bytes + fp4_bytes + 3 * scale_bytes
    else:
        raise ValueError(f"unknown path: {path}")
    return rows * hidden * streams_per_element


def routing_metadata_xe35(routed_rows, experts, alignment, rows, device):
    active_experts = min(experts, routed_rows)
    base, remainder = divmod(routed_rows, active_experts)
    counts = torch.zeros(experts, dtype=torch.int32)
    counts[:active_experts] = base
    counts[:remainder] += 1
    padded_counts = (
        torch.div(counts + alignment - 1, alignment, rounding_mode="floor") * alignment
    )
    used_rows = int(padded_counts.sum())
    assert used_rows <= rows
    return padded_counts.to(device), used_rows


def run_padded_gather_xe35(args):
    if not torch.xpu.is_available():
        raise RuntimeError("an XPU is required")
    device = torch.xpu.get_device_properties(0)
    print(
        f"device={device.name} device_id=0x{device.device_id:04x} "
        f"B={args.serving_batch} ISL={args.isl} OSL={args.osl} TP={args.tp} EP={args.ep}"
    )

    for token_count in args.tokens:
        torch.xpu.synchronize()
        time.sleep(2)
        routed_rows = token_count * args.topk
        rows = routed_rows + (args.alignment - 1) * min(args.experts, routed_rows)
        hidden_states = torch.randn(
            (token_count, args.hidden), dtype=torch.bfloat16, device="xpu"
        )
        gate_up = torch.randn(
            (rows, 2 * args.hidden), dtype=torch.bfloat16, device="xpu"
        )
        padded_counts, used_rows = routing_metadata_xe35(
            routed_rows,
            args.experts,
            args.alignment,
            rows,
            hidden_states.device,
        )
        row_map_i32 = torch.arange(rows, dtype=torch.int32, device="xpu") % token_count
        expected_q, expected_s = baseline_xe35(hidden_states, row_map_i32, padded_counts)
        gather_q, gather_s = gather_quant_relayout_xe35(
            hidden_states, row_map_i32, padded_counts
        )
        used_scale_bytes = used_rows * (args.hidden // 32)
        if not torch.equal(expected_q, gather_q) or not torch.equal(
            expected_s.view(-1)[:used_scale_bytes],
            gather_s.view(-1)[:used_scale_bytes],
        ):
            raise AssertionError(
                f"optimized output differs for tokens={token_count}, rows={rows}"
            )
        del expected_q, expected_s, gather_q, gather_s
        torch.xpu.synchronize()

        baseline_ms = measure_padded_xe35(
            lambda: baseline_xe35(hidden_states, row_map_i32, padded_counts),
            args.warmup,
            args.repeat,
        )
        gather_relayout_ms = measure_padded_xe35(
            lambda: gather_quant_relayout_xe35(hidden_states, row_map_i32, padded_counts),
            args.warmup,
            args.repeat,
        )
        _, scales_rm = gather_quant_only_xe35(hidden_states, row_map_i32)
        gather_quant_ms = measure_padded_xe35(
            lambda: gather_quant_only_xe35(hidden_states, row_map_i32),
            args.warmup,
            args.repeat,
        )
        relayout_ms = measure_padded_xe35(
            lambda: relayout_only_xe35(scales_rm, padded_counts),
            args.warmup,
            args.repeat,
        )
        swiglu_relayout_ms = measure_padded_xe35(
            lambda: swiglu_quant_relayout_xe35(gate_up, padded_counts),
            args.warmup,
            args.repeat,
        )
        baseline_gbps = logical_bytes_xe35(rows, args.hidden, "baseline_xe35") / baseline_ms / 1e6
        gather_relayout_gbps = (
            logical_bytes_xe35(rows, args.hidden, "gather_relayout")
            / gather_relayout_ms
            / 1e6
        )
        swiglu_relayout_gbps = (
            logical_bytes_xe35(rows, args.hidden, "swiglu_relayout")
            / swiglu_relayout_ms
            / 1e6
        )

        print(
            f"tokens={token_count} routed_rows={routed_rows} "
            f"used_rows={used_rows} padded_upper_rows={rows}: "
            f"index_select+quant+relayout={baseline_ms:.4f} ms/"
            f"{baseline_gbps:.1f} logical GB/s, "
            f"gather+quant+relayout={gather_relayout_ms:.4f} ms/"
            f"{gather_relayout_gbps:.1f} logical GB/s, "
            f"gather+quant={gather_quant_ms:.4f} ms, "
            f"scale_relayout={relayout_ms:.4f} ms, "
            f"speedup={baseline_ms / gather_relayout_ms:.3f}x"
        )
        print(
            f"  swiglu+quant+relayout={swiglu_relayout_ms:.4f} ms/"
            f"{swiglu_relayout_gbps:.1f} logical GB/s"
        )

        del gate_up, hidden_states, row_map_i32, scales_rm
        del padded_counts
        torch.xpu.synchronize()


def run_audit_xe35(args):
    global GRAPH_REPLAY
    GRAPH_REPLAY = args.graph_replay
    if args.branch:
        from sgl_kernel import (
            muse_spark_moe_routing_xe35,
            mxfp4_moe_scale_to_mn_major_xe35,
            sgl_padded_gather_per_token_group_quant_fp4_xe35,
        )

    torch.manual_seed(7)
    props = torch.xpu.get_device_properties(0)
    print(
        json.dumps(
            {
                "device": props.name,
                "device_id": props.device_id,
                "branch": args.branch,
                "hidden": args.hidden,
            }
        ),
        flush=True,
    )
    for rows in args.shapes:
        torch.xpu.synchronize()
        time.sleep(2)
        if args.graph_replay > 1:
            torch.manual_seed(7000 + rows)
        source_rows = 32768 if rows >= 32768 * 8 else 8
        if args.op != "route":
            x = torch.randn(
                (source_rows, args.hidden), device="xpu", dtype=torch.bfloat16
            )
            index = torch.arange(rows, device="xpu", dtype=torch.int32) % source_rows
            gathered = x.index_select(0, index)
            q_ref, s_ref = sgl_per_token_group_quant_fp4_xe35(gathered)
            if args.branch:
                q, s = sgl_padded_gather_per_token_group_quant_fp4_xe35(x, index)
                assert torch.equal(q, q_ref) and torch.equal(s, s_ref)
            if not args.lane_only:
                bench_xe35(
                    f"index_quant_rows{rows}",
                    lambda: sgl_per_token_group_quant_fp4_xe35(x.index_select(0, index)),
                    args.warmup,
                    args.rep,
                )
            if not args.fused_only:
                bench_xe35(
                    f"plain_quant_rows{rows}",
                    lambda: sgl_per_token_group_quant_fp4_xe35(gathered),
                    args.warmup,
                    args.rep,
                )
            if args.branch:
                if not args.fused_only:
                    bench_xe35(
                        f"gather_quant_rows{rows}",
                        lambda: sgl_padded_gather_per_token_group_quant_fp4_xe35(x, index),
                        args.warmup,
                        args.rep,
                    )
                routed_rows = source_rows * 8
                active_experts = min(256, routed_rows)
                counts = torch.zeros(256, dtype=torch.int32)
                base, rem = divmod(routed_rows, active_experts)
                counts[:active_experts] = base
                counts[:rem] += 1
                counts = (counts + 3) // 4 * 4
                used_rows = int(counts.sum())
                if used_rows <= rows and not args.lane_only:
                    counts_xpu = counts.to("xpu")

                    def index_quant_relayout():
                        q, s = sgl_per_token_group_quant_fp4_xe35(x.index_select(0, index))
                        s_mn = torch.empty_like(s)
                        mxfp4_moe_scale_to_mn_major_xe35(s_mn, s, counts_xpu)
                        return q, s_mn

                    def gather_quant_relayout_xe35():
                        q, s = sgl_padded_gather_per_token_group_quant_fp4_xe35(x, index)
                        s_mn = torch.empty_like(s)
                        mxfp4_moe_scale_to_mn_major_xe35(s_mn, s, counts_xpu)
                        return q, s_mn

                    iq, is_mn = index_quant_relayout()
                    gq, gs_mn = gather_quant_relayout_xe35()
                    assert torch.equal(iq, gq) and torch.equal(
                        is_mn.flatten()[: used_rows * (args.hidden // 32)],
                        gs_mn.flatten()[: used_rows * (args.hidden // 32)],
                    )
                    bench_xe35(
                        f"index_quant_relayout_rows{rows}",
                        index_quant_relayout,
                        args.warmup,
                        args.rep,
                    )
                    bench_xe35(
                        f"gather_quant_relayout_rows{rows}",
                        gather_quant_relayout_xe35,
                        args.warmup,
                        args.rep,
                    )
                gate_up = torch.randn(
                    (rows, 2 * args.hidden), device="xpu", dtype=torch.bfloat16
                )
                activated = silu_and_mul(gate_up)
                fused_q, fused_s = sgl_per_token_group_quant_fp4_xe35(
                    gate_up, fuse_silu_and_mul=True
                )
                separate_q, separate_s = sgl_per_token_group_quant_fp4_xe35(activated)
                assert torch.equal(fused_q, separate_q) and torch.equal(
                    fused_s, separate_s
                )
                if not args.lane_only:
                    bench_xe35(
                        f"separate_silu_quant_rows{rows}",
                        lambda: sgl_per_token_group_quant_fp4_xe35(silu_and_mul(gate_up)),
                        args.warmup,
                        args.rep,
                    )
                bench_xe35(
                    f"fused_silu_quant_rows{rows}",
                    lambda: sgl_per_token_group_quant_fp4_xe35(
                        gate_up, fuse_silu_and_mul=True
                    ),
                    args.warmup,
                    args.rep,
                )
        if not args.quant_only and rows in (256, 512, 32768 * 8 + 768):
            route_tokens = source_rows if rows != 512 else 16
            if args.graph_replay > 1:
                torch.manual_seed(17000 + route_tokens)
            logits = torch.randn((route_tokens, 256), device="xpu", dtype=torch.float32)
            bias = torch.randn((256,), device="xpu", dtype=torch.float32)
            multiplier = 0.3535533905932738

            def native():
                scores = torch.sigmoid((logits * multiplier).float())
                ids = torch.topk(scores + bias, 8, dim=-1, sorted=True)[1]
                weights = scores.gather(1, ids)
                return weights / (weights.sum(dim=-1, keepdim=True) + 1e-15), ids

            main_weights = torch.empty(
                (route_tokens, 8), device="xpu", dtype=torch.float32
            )
            main_ids = torch.empty((route_tokens, 8), device="xpu", dtype=torch.int32)

            def main_topk():
                scaled_logits = logits * multiplier
                topk_sigmoid_xe35(main_weights, main_ids, scaled_logits, True, bias)
                return main_weights, main_ids

            main_w, main_i = main_topk()
            check_route_xe35(main_w, main_i, logits, bias, multiplier)
            if args.branch:
                weights = torch.empty(
                    (route_tokens, 8), device="xpu", dtype=torch.float32
                )
                ids = torch.empty((route_tokens, 8), device="xpu", dtype=torch.int32)

                def fused():
                    muse_spark_moe_routing_xe35(
                        weights, ids, logits, bias, multiplier, 1.0, 1e-15
                    )

                fused()
                check_route_xe35(weights, ids, logits, bias, multiplier)
            bench_xe35(f"native_route_tokens{route_tokens}", native, args.warmup, args.rep)
            bench_xe35(
                f"main_topk_route_tokens{route_tokens}",
                main_topk,
                args.warmup,
                args.rep,
            )
            if args.triton_route:
                try:
                    from sglang_meta_common.ops.fused_muse_spark_routing import (
                        fused_muse_spark_routing,
                    )

                    triton_weights = torch.empty_like(main_weights)
                    triton_ids = torch.empty_like(main_ids)

                    def triton_route():
                        return fused_muse_spark_routing(
                            logits,
                            bias,
                            8,
                            1.0,
                            router_output_multiplier=multiplier,
                            router_score_normalization="sigmoid",
                            router_score_division_eps=1e-15,
                            topk_weights=triton_weights,
                            topk_ids=triton_ids,
                        )

                    triton_w, triton_i = triton_route()
                    check_route_xe35(triton_w, triton_i, logits, bias, multiplier)
                    bench_xe35(
                        f"framework_auto_route_tokens{route_tokens}",
                        triton_route,
                        args.warmup,
                        args.rep,
                    )
                except AssertionError:
                    raise
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "name": f"framework_auto_route_tokens{route_tokens}",
                                "error": repr(exc),
                            }
                        ),
                        flush=True,
                    )
            if args.branch:
                bench_xe35(f"fused_route_tokens{route_tokens}", fused, args.warmup, args.rep)
        torch.xpu.synchronize()


def main_xe35():
    args = parse_args_xe35()
    global torch, sgl_per_token_group_quant_fp4_xe35, silu_and_mul, topk_sigmoid_xe35
    global mxfp4_moe_scale_to_mn_major_xe35, sgl_padded_gather_per_token_group_quant_fp4_xe35
    import torch
    from sgl_kernel import is_xe3_arch, sgl_per_token_group_quant_fp4_xe35, silu_and_mul, topk_sigmoid_xe35

    if not torch.xpu.is_available() or not is_xe3_arch():
        raise RuntimeError("Routing/quant benchmark requires an Intel Xe3.5 XPU")

    if args.mode == "padded-gather":
        from sgl_kernel import (
            mxfp4_moe_scale_to_mn_major_xe35,
            sgl_padded_gather_per_token_group_quant_fp4_xe35,
        )

        run_padded_gather_xe35(args)
    else:
        run_audit_xe35(args)


if __name__ == "__main__":
    main_xe35()
