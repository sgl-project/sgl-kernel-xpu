from itertools import product
from typing import Optional, Tuple

import torch
import triton
from sgl_kernel import sgl_per_tensor_quant_fp8


def sglang_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    is_static: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_type_: torch.dtype = torch.float8_e4m3fn
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    if scale is None:
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
    sgl_per_tensor_quant_fp8(input, output, scale, is_static)

    return output, scale


batch_size_range = [16, 32, 64, 128]
seq_len_range = [64, 128, 256, 512, 1024, 2048]
hidden_size = 1024

configs = [
    (bs, seq_len, is_static)
    for is_static in [True, False]
    for bs, seq_len in product(batch_size_range, seq_len_range)
]

all_results = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "is_static"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl_kernel"],
        line_names=["SGL Kernel"],
        styles=[("blue", "-")],
        ylabel="Time (ms)",
        plot_name="per-tensor-quant-fp8-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, is_static, provider):
    print(
        f"benchmark {provider} with batch_size={batch_size} seq_len={seq_len} is_static={is_static}"
    )
    dtype = torch.float16
    torch.set_default_device("xpu")
    torch.xpu.manual_seed_all(0)

    num_tokens = batch_size * seq_len
    x = torch.randn(
        num_tokens, hidden_size, dtype=dtype, device="xpu", requires_grad=False
    )

    scale = None
    if is_static:
        scale = torch.tensor(
            [0.01], dtype=torch.float32, device="xpu", requires_grad=False
        )
    # Warmup
    for _ in range(10):
        _ = sglang_scaled_fp8_quant(x, scale, is_static)
    torch.xpu.synchronize()

    bench_lambda = lambda: sglang_scaled_fp8_quant(x, scale, is_static)

    quantiles = [0.5, 0.25, 0.75]
    ms, _, _ = triton.testing.do_bench(
        bench_lambda, quantiles=quantiles, return_mode="median"
    )

    torch.xpu.empty_cache()
    del x
    if scale is not None:
        del scale

    # Calculate FLOPS and bandwidth
    # FP8 quantization involves: reading input, computing abs max (if dynamic),
    # scaling, and writing output
    num_elements = batch_size * seq_len * hidden_size

    # FLOPS calculation:
    # - Static: division + clamp + cast = ~1 + (1+1) + (1+1)= 5 ops per element
    # - Dynamic: static + abs + cast + max= ~ 5 + 1 + 1 + 1 ops per element

    flops_per_elem = 8 if not is_static else 5
    flop = num_elements * flops_per_elem
    tflops = flop / (ms / 1e3) / 1e12

    # Memory calculation:
    # Static: Read input (fp16=2B/elem) + Write output (fp8=1B/elem) = 3 bytes/elem
    # Dynamic: Read input for max (fp16=2B/elem) + Read input for quant (fp16=2B/elem) + Write output (fp8=1B/elem) = 5 bytes/elem
    # Scale read/write: static=4B(1 reads), dynamic=12B(2 reads + 1 write)
    bytes_per_elem = 5 if not is_static else 3
    scale_read_write = 12 if not is_static else 4
    memory = num_elements * bytes_per_elem + scale_read_write
    bandwidth = memory / (ms / 1e3) / 1e9

    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "is_static": is_static,
            "provider": provider,
            "tflops": tflops,
            "bandwidth": bandwidth,
            "ms": ms,
        }
    )
    return ms


if __name__ == "__main__":
    benchmark.run(print_data=False)
    print("Benchmark finished!")

    import pandas as pd

    df = pd.DataFrame(all_results)
    print(df.to_markdown())
