import itertools

import pandas as pd
import torch
import triton
from sgl_kernel import hash_topk

all_results = []


def hash_topk_torch_native(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
):
    scores = torch.nn.functional.softplus(router_logits).sqrt()

    num_token = scores.shape[0]
    num_routed_experts = scores.shape[1]
    topk_routed = tid2eid.shape[1]
    topk = topk_routed + num_fused_shared_experts

    topk_ids = torch.zeros((num_token, topk), dtype=torch.int32, device=scores.device)
    topk_weights = torch.zeros(
        (num_token, topk), dtype=scores.dtype, device=scores.device
    )

    if num_fused_shared_experts == 1:
        topk_ids[:, :-1] = tid2eid[input_ids]
        topk_weights[:, :-1] = scores.gather(1, topk_ids[:, :-1].long())
        topk_weights[:, :-1] /= topk_weights[:, :-1].sum(dim=-1, keepdim=True)

        topk_ids[:, -1] = torch.randint(
            low=num_routed_experts,
            high=num_routed_experts + num_fused_shared_experts,
            size=(num_token,),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )

        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
    else:
        assert num_fused_shared_experts == 0
        topk_ids[:, :] = tid2eid[input_ids]
        topk_weights[:, :] = scores.gather(1, topk_ids[:, :].long())
        topk_weights[:, :] /= topk_weights[:, :].sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def get_benchmark(device: str = "xpu"):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "num_tokens",
                "num_routed_experts",
                "topk_routed",
                "num_fused_shared_experts",
                "dtype",
            ],
            x_vals=configs,
            line_arg="provider",
            line_vals=["kernel", "native"],
            line_names=["hash_topk_kernel", "hash_topk_native"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (us)",
            plot_name="hash-topk-kernel-vs-native",
            args={},
        )
    )
    def benchmark(
        num_tokens,
        num_routed_experts,
        topk_routed,
        num_fused_shared_experts,
        dtype,
        provider,
    ):
        vocab_size = 1024
        routed_scaling_factor = 2.5

        torch.manual_seed(1024)

        router_logits = torch.randn(
            num_tokens,
            num_routed_experts,
            dtype=dtype,
            device=device,
        )
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_tokens,),
            dtype=torch.int64,
            device=device,
        )
        tid2eid = torch.randint(
            low=0,
            high=num_routed_experts,
            size=(vocab_size, topk_routed),
            dtype=torch.int32,
            device=device,
        )
        topk_fused = topk_routed + num_fused_shared_experts
        topk_weights = torch.empty(
            (num_tokens, topk_fused), dtype=torch.float32, device=device
        )
        topk_ids = torch.empty(
            (num_tokens, topk_fused), dtype=torch.int32, device=device
        )

        if provider == "kernel":

            def run_op():
                hash_topk(
                    router_logits,
                    input_ids,
                    tid2eid,
                    topk_weights,
                    topk_ids,
                    routed_scaling_factor=routed_scaling_factor,
                    scoring_func="sqrtsoftplus",
                )

        elif provider == "native":

            def run_op():
                hash_topk_torch_native(
                    router_logits.float(),
                    input_ids,
                    tid2eid,
                    num_fused_shared_experts,
                    routed_scaling_factor,
                )

        else:
            raise ValueError(f"Unknown provider: {provider}")

        for _ in range(10):
            run_op()
        torch.xpu.synchronize()

        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(run_op, quantiles=quantiles)

        all_results.append(
            {
                "provider": provider,
                "num_tokens": num_tokens,
                "num_routed_experts": num_routed_experts,
                "topk_routed": topk_routed,
                "num_fused_shared_experts": num_fused_shared_experts,
                "dtype": str(dtype),
                "ms": ms,
            }
        )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    if not torch.xpu.is_available():
        raise RuntimeError("XPU is required for bench_hash_topk.py")

    sweep_params = {
        "num_tokens": [1, 64, 1024, 4096],
        "num_routed_experts": [256, 384],
        "topk_routed": [6],
        "num_fused_shared_experts": [0, 1],
        "dtype": [torch.float32],
    }
    configs = list(itertools.product(*sweep_params.values()))

    print(f"Running {len(configs)} hash_topk benchmark configs")

    benchmark = get_benchmark(device="xpu")
    benchmark.run(print_data=False, show_plots=False, save_path=".")

    df = pd.DataFrame(all_results)
    summary_key_cols = [
        "num_tokens",
        "num_routed_experts",
        "topk_routed",
        "num_fused_shared_experts",
        "dtype",
    ]
    summary_df = (
        df.pivot_table(
            index=summary_key_cols,
            columns="provider",
            values="ms",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"kernel": "kernel_ms", "native": "native_ms"})
    )
    summary_df["speedup"] = summary_df["native_ms"] / summary_df["kernel_ms"]
    summary_df["kernel_ms"] = summary_df["kernel_ms"].map(lambda x: f"{x:.6f}")
    summary_df["native_ms"] = summary_df["native_ms"].map(lambda x: f"{x:.6f}")
    summary_df["speedup"] = summary_df["speedup"].map(lambda x: f"{x:.2f}x")

    print("Kernel vs Native implementation latency summary:")
    print(summary_df.to_markdown(index=False))
