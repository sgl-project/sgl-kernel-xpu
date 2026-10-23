import argparse

import torch
import triton
from sgl_kernel import ep_moe_pre_reorder

from sglang.srt.layers.moe.ep_moe.kernels import pre_reorder_triton_kernel

batch_sizes = [64, 128, 256, 512, 640, 768, 1024, 2048, 4096]
configs = [(bs,) for bs in batch_sizes]


def benchmark(batch_size, provider, hidden_size=4096, topk=8, num_experts=256):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    start_expert_id, end_expert_id, block_size = 0, num_experts - 1, 512

    # Allocate fresh tensors for every run to match bench_moe_fused_gate style
    def alloc_tensors():
        input_ = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        gateup_input = torch.zeros(
            batch_size * topk, hidden_size, dtype=dtype, device=device
        )
        src2dst = torch.randint(
            0, batch_size * topk, (batch_size, topk), dtype=torch.int32, device=device
        )
        topk_ids = torch.randint(
            start_expert_id,
            end_expert_id + 1,
            (batch_size, topk),
            dtype=torch.int32,
            device=device,
        )
        a1_scales = torch.rand(
            end_expert_id - start_expert_id + 1, dtype=torch.float32, device=device
        )
        return input_, gateup_input, src2dst, topk_ids, a1_scales

    quantiles = [0.5, 0.2, 0.8]

    if provider == "cuda":
        inp, gout, s2d, tk_ids, scales = alloc_tensors()

        def run_cuda():
            ep_moe_pre_reorder(
                inp,
                gout,
                s2d,
                tk_ids,
                scales,
                start_expert_id,
                end_expert_id,
                topk,
                True,
            )

        ms, min_ms, max_ms = triton.testing.do_bench(run_cuda, quantiles=quantiles)

    elif provider == "triton":
        inp, gout, s2d, tk_ids, scales = alloc_tensors()

        def run_triton():
            pre_reorder_triton_kernel[(batch_size,)](
                inp.view(-1),
                gout.view(-1),
                s2d.view(-1),
                tk_ids.view(-1),
                scales,
                start_expert_id,
                end_expert_id,
                topk,
                hidden_size,
                block_size,
                True,
            )

        ms, min_ms, max_ms = triton.testing.do_bench(run_triton, quantiles=quantiles)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


def get_benchmark(configs=configs, hidden_size=4096, topk=8, num_experts=256):
    return triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["cuda", "triton"],
            line_names=["CUDA Kernel", "Triton Kernel"],
            styles=[("green", "-"), ("orange", "-")],
            ylabel="us",
            plot_name="ep-moe-pre-reorder-performance",
            args={
                "hidden_size": hidden_size,
                "topk": topk,
                "num_experts": num_experts,
            },
        )
    )(benchmark)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=batch_sizes,
        help="Batch sizes to sweep",
    )
    parser.add_argument("--hidden-size", type=int, default=4096, help="Hidden size")
    parser.add_argument("--topk", type=int, default=8, help="Experts per token")
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts"
    )
    args = parser.parse_args()

    configs = [(bs,) for bs in args.batch_sizes]
    get_benchmark(configs, args.hidden_size, args.topk, args.num_experts).run(
        print_data=True
    )
