"""Benchmark fused Xe2 QSA preparation against equivalent Torch eager operations."""

import argparse
import itertools

import pandas as pd
import sgl_kernel
import torch
import triton

all_results = []


def rotate(x, positions, cache):
    half = x.shape[-1] // 2
    c = cache[positions, :half].to(x.dtype)
    s = cache[positions, half:].to(x.dtype)
    return torch.cat(
        [x[..., :half] * c - x[..., half:] * s, x[..., half:] * c + x[..., :half] * s],
        dim=-1,
    )


def norm(x, w):
    return (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        * (1 + w.float())
    ).to(x.dtype)


def get_benchmark(ratio: int, device: str = "xpu"):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["stage", "rows"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["fused", "eager"],
            line_names=["fused", "eager"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (ms)",
            plot_name="qsa-prepare-fused-vs-eager",
            args={"ratio": ratio},
        )
    )
    def benchmark(stage, rows, provider, ratio):
        dim, heads = 128, 4
        state_slots = max(rows, ratio)
        qk = torch.randn(rows, (heads + 1) * dim, device=device, dtype=torch.bfloat16)
        cache = torch.randn(512, dim, device=device, dtype=torch.float32)
        axes = torch.zeros(dim // 2, device=device, dtype=torch.int32)
        pos = torch.arange(rows, device=device, dtype=torch.int64) % 512
        slots = torch.arange(rows, device=device, dtype=torch.int64)
        weight = torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.03
        state = torch.randn((state_slots, dim), device=device, dtype=torch.bfloat16)
        rope_state = torch.zeros((state_slots, 3), device=device, dtype=torch.int64)
        groups = torch.arange(
            state_slots // ratio * ratio, device=device, dtype=torch.int32
        ).reshape(-1, ratio)
        writes = torch.arange(groups.shape[0], device=device, dtype=torch.int32)
        compressed = torch.empty(
            (groups.shape[0], dim), device=device, dtype=torch.bfloat16
        )

        if stage == "q":
            if provider == "fused":

                def run_op():
                    sgl_kernel.qsa_index_q_norm_rope_store(
                        qk,
                        pos,
                        cache,
                        axes,
                        weight,
                        slots,
                        state,
                        rope_state,
                        heads,
                        dim,
                        1e-6,
                        True,
                    )

            else:

                def run_op():
                    q = norm(qk[:, : heads * dim].reshape(rows, heads, dim), weight)
                    rotate(q, pos[:, None], cache)
                    state[slots] = qk[:, heads * dim :]
                    rope_state[slots] = pos[:, None].expand(-1, 3)

        else:
            # Populate state/rope_state via the fused Q stage so the K stage has
            # realistic inputs to compress, regardless of which provider runs.
            sgl_kernel.qsa_index_q_norm_rope_store(
                qk,
                pos,
                cache,
                axes,
                weight,
                slots,
                state,
                rope_state,
                heads,
                dim,
                1e-6,
                True,
            )

            if provider == "fused":

                def run_op():
                    sgl_kernel.qsa_index_k_compress_store(
                        state,
                        groups,
                        rope_state,
                        cache,
                        axes,
                        weight,
                        writes,
                        compressed,
                        ratio,
                        dim,
                        1e-6,
                        True,
                    )

            else:

                def run_op():
                    pooled = state[groups.long()].float().mean(1).to(state.dtype)
                    compressed[writes.long()] = rotate(
                        norm(pooled, weight), rope_state[groups[:, 0].long(), 0], cache
                    )

        for _ in range(10):
            run_op()
        torch.xpu.synchronize()

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(run_op, quantiles=quantiles)

        all_results.append(
            {
                "stage": stage,
                "rows": rows,
                "groups": groups.shape[0],
                "provider": provider,
                "ms": ms,
            }
        )
        print(
            f"stage={stage} rows={rows:5} groups={groups.shape[0]:5} "
            f"provider={provider:5} ms={ms:.3f}"
        )

        return ms, max_ms, min_ms

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 256, 2048])
    parser.add_argument("--ratio", type=int, default=4)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("XPU required")

    configs = list(itertools.product(["q", "k"], args.rows))
    print(f"Running {len(configs)} qsa-prepare benchmark configs")

    benchmark = get_benchmark(args.ratio, device="xpu")
    benchmark.run(print_data=False, show_plots=False, save_path=".")

    df = pd.DataFrame(all_results)
    summary_df = (
        df.pivot_table(
            index=["stage", "rows", "groups"],
            columns="provider",
            values="ms",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"fused": "fused_ms", "eager": "eager_ms"})
    )
    summary_df["speedup"] = summary_df["eager_ms"] / summary_df["fused_ms"]
    summary_df["fused_ms"] = summary_df["fused_ms"].map(lambda x: f"{x:.3f}")
    summary_df["eager_ms"] = summary_df["eager_ms"].map(lambda x: f"{x:.3f}")
    summary_df["speedup"] = summary_df["speedup"].map(lambda x: f"{x:.2f}x")

    print("Fused vs eager QSA preparation latency summary:")
    print(summary_df.to_markdown(index=False))
