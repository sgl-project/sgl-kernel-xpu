import argparse

import torch
import torch.nn.functional as F
import triton

BF16_PEAK_TFLOPS = 100.0
DRAM_BANDWIDTH_GBPS = 450.0


def _bench(fn, warmup: int, rep: int) -> tuple[float, float, float]:
    return triton.testing.do_bench(
        fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8]
    )


def _roofline(name: str, flops: float, bytes_moved: float) -> None:
    intensity = flops / bytes_moved
    compute_ms = flops / (BF16_PEAK_TFLOPS * 1.0e12) * 1.0e3
    memory_ms = bytes_moved / (DRAM_BANDWIDTH_GBPS * 1.0e9) * 1.0e3
    print(
        f"{name:10s} FLOPs={flops / 1e9:9.3f} G  "
        f"bytes={bytes_moved / 1e6:8.3f} MB  AI={intensity:8.1f} FLOP/B  "
        f"roof={max(compute_ms, memory_ms):7.3f} ms "
        f"(compute={compute_ms:.3f}, memory={memory_ms:.3f})"
    )


def _result(name: str, timing: tuple[float, float, float], flops: float) -> float:
    median, p20, p80 = timing
    tflops = flops / (median * 1.0e-3) / 1.0e12
    print(
        f"{name:18s} median={median:8.4f} ms  p20={p20:8.4f} ms  "
        f"p80={p80:8.4f} ms  {tflops:7.2f} TFLOP/s  "
        f"{100.0 * tflops / BF16_PEAK_TFLOPS:6.2f}% peak"
    )
    return median


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Device-timed Inkling TP4 dense-MLP benchmark and roofline."
    )
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=6144)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--implementation", choices=("candidate", "reference"), default="candidate"
    )
    args = parser.parse_args()

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU is required")
    if args.intermediate % 16:
        raise ValueError("--intermediate must be divisible by 16")
    if args.layers < 1:
        raise ValueError("--layers must be positive")

    torch.manual_seed(20260923)
    x = torch.randn((args.m, args.hidden), device="xpu", dtype=torch.bfloat16) * 0.02
    gate_up_weight = (
        torch.randn(
            (2 * args.intermediate, args.hidden),
            device="xpu",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    down_weight = (
        torch.randn(
            (args.hidden, args.intermediate),
            device="xpu",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    gate_flops = 2.0 * args.m * (2 * args.intermediate) * args.hidden
    down_flops = 2.0 * args.m * args.hidden * args.intermediate
    element_bytes = 2.0
    gate_bytes = element_bytes * (
        args.m * args.hidden
        + args.hidden * (2 * args.intermediate)
        + args.m * args.intermediate
    )
    down_bytes = element_bytes * (
        args.m * args.intermediate
        + args.intermediate * args.hidden
        + args.m * args.hidden
    )

    print(
        f"Inkling dense MLP ({args.implementation}): "
        f"M={args.m} H={args.hidden} I={args.intermediate} "
        f"layers={args.layers}, BF16 activations/weights"
    )
    print(
        "MXFP4 note: the reduced checkpoint uses MXFP4 for routed experts; "
        "the dense MLP production tensors are BF16."
    )
    print(
        f"BMG peak ceilings: {BF16_PEAK_TFLOPS:.1f} TFLOP/s BF16, "
        f"{DRAM_BANDWIDTH_GBPS:.0f} GB/s DRAM"
    )
    _roofline("gate_up", gate_flops, gate_bytes)
    _roofline("down", down_flops, down_bytes)
    _roofline(
        "module",
        args.layers * (gate_flops + down_flops),
        args.layers * (gate_bytes + down_bytes),
    )

    if args.implementation == "reference":

        def run_gate_up():
            projected = F.linear(x, gate_up_weight)
            gate, up = projected.chunk(2, dim=-1)
            return F.silu(gate) * up

        activated = run_gate_up()

        def run_down():
            return F.linear(activated, down_weight)

        def run_module():
            current = x
            for _ in range(args.layers):
                projected = F.linear(current, gate_up_weight)
                gate, up = projected.chunk(2, dim=-1)
                current = F.linear(F.silu(gate) * up, down_weight)
            return current

    else:
        from sgl_kernel.inkling_dense_mlp import (
            inkling_down,
            inkling_gate_up,
            repack_inkling_down_weight,
            repack_inkling_gate_up_weight,
        )

        packed_gate_up = repack_inkling_gate_up_weight(gate_up_weight)
        packed_down = repack_inkling_down_weight(down_weight)
        activated_storage = torch.empty(
            (args.m, args.intermediate + 64),
            device="xpu",
            dtype=torch.bfloat16,
        )
        activated = activated_storage[:, : args.intermediate]
        outputs = (torch.empty_like(x), torch.empty_like(x))

        def run_gate_up():
            inkling_gate_up(x, packed_gate_up, out=activated)

        def run_down():
            inkling_down(activated, packed_down, out=outputs[0])

        def run_module():
            current = x
            for layer in range(args.layers):
                inkling_gate_up(current, packed_gate_up, out=activated)
                output = outputs[layer % len(outputs)]
                inkling_down(activated, packed_down, out=output)
                current = output

    torch.xpu.synchronize()
    gate_ms = _result(
        "gate_up+swiGLU", _bench(run_gate_up, args.warmup, args.rep), gate_flops
    )
    down_ms = _result("down", _bench(run_down, args.warmup, args.rep), down_flops)
    module_ms = _result(
        f"module x{args.layers}",
        _bench(run_module, args.warmup, args.rep),
        args.layers * (gate_flops + down_flops),
    )
    projected_ms = args.layers * (gate_ms + down_ms)
    print(
        f"sum of isolated ops={projected_ms:.4f} ms; measured module={module_ms:.4f} ms"
    )
    if args.implementation == "reference":
        return 0
    print("Compare this device time with the reference on the same GPU and shape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
