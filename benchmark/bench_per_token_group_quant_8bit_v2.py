import itertools
from typing import Tuple

import itertools
import sys
from typing import List, Optional, Tuple


import pandas as pd
import torch
import triton
import triton.language as tl
import sgl_kernel

fp8_type_ = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_type_).max
fp8_min = -fp8_max

def prepare_token_group_quant_test_data(num_tokens, hidden_dim):
    device = torch.device("xpu")
    dtype = torch.bfloat16

    seed = num_tokens * 10000 + hidden_dim
    gen_xpu = torch.Generator(device="xpu")
    gen_xpu.manual_seed(seed)

    effective_hidden_dim = hidden_dim * 2
    
    x = torch.randn(
        num_tokens,
        effective_hidden_dim,
        device=device,
        dtype=dtype,
        generator=gen_xpu,
    )
    return x

# COPIED FROM DeepGEMM
def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y

# COPIED FROM DeepGEMM
def ceil_align(x: int, y: int) -> int:
    return ceil_div(x, y) * y

def create_fp8_output_scale(
    x_shape,
    device,
    group_size,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
):
    if scale_ue8m0:
        assert column_major_scales and scale_tma_aligned
        *x_batch, x_q_mn, x_q_k = x_shape
        x_s_mn, x_s_k = x_q_mn, x_q_k // 128
        aligned_mn = ceil_align(x_s_mn, 4)
        aligned_k = ceil_align(x_s_k, 4)
        return torch.empty(
            (*x_batch, aligned_k // 4, aligned_mn),
            device=device,
            dtype=torch.int,
        ).transpose(-1, -2)[..., :x_s_mn, :]
    elif column_major_scales:
        if scale_tma_aligned:
            # TODO extract "align" function
            # aligned to 4 * sizeof(float)
            aligned_size = (x_shape[-2] + 3) // 4 * 4
            return torch.empty(
                x_shape[:-2] + (x_shape[-1] // group_size, aligned_size),
                device=device,
                dtype=torch.float32,
            ).transpose(-1, -2)[: x_shape[-2], :]
        else:
            return torch.empty(
                (x_shape[-1] // group_size,) + x_shape[:-1],
                device=device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        return torch.empty(
            x_shape[:-1] + (x_shape[-1] // group_size,),
            device=device,
            dtype=torch.float32,
        )

def sglang_per_token_group_quant_8bit_layer(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
):

    if not fuse_silu_and_mul:
        x = sgl_kernel.silu_and_mul(x.to("xpu"))

    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))

    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_type_)
    x_s = create_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )

    torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit_v2.default(
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
            fuse_silu_and_mul,
            masked_m,
    )
    
    return x_q, x_s

def calculate_diff(batch_size, seq_len, group_size, dst_dtype):
    device = torch.device("xpu")
    hidden_dim = 7168

    x = prepare_token_group_quant_test_data(batch_size * seq_len, hidden_dim)
   
    x_q_ref, x_s_ref = sglang_per_token_group_quant_8bit_layer(
        x.clone(), group_size, dst_dtype, 1e-10, True, True, True, False, None, True
    )

    x_q_sglang, x_s_sglang = sglang_per_token_group_quant_8bit_layer(
        x.clone(), group_size, dst_dtype, 1e-10, True, True, True, True, None, True
    )

    try:
        torch.testing.assert_close(
            x_s_ref.contiguous(),
            x_s_sglang.contiguous(),
            rtol=1e-3,
            atol=1e-5,
            msg=lambda message: message + f" {x_s_ref=} {x_s_sglang=}",
        )
        print(f"✅ {dst_dtype} implementations match")
    except AssertionError:
        print(
            f"{x.shape=} {x_q_ref.shape=} {x_s_ref.shape=} {x_q_sglang.shape=} {x_s_sglang.shape=}"
        )
        print(f"{x=}")
        print(f"{masked_m=}")
        print(f"{x_q_ref=}")
        print(f"{x_s_ref=}")
        print(f"{x_q_sglang=}")
        print(f"{x_s_sglang=}")
        print("❌ Implementations differ")
        raise

batch_size_range = [16, 32, 64]
seq_len_range = [128, 256, 512, 1024]
group_size_range = [128]  # For DeepSeek V3/R1
dst_dtype_range = [fp8_type_]

configs = list(
    itertools.product(
        batch_size_range, seq_len_range, group_size_range, dst_dtype_range
    )
)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "group_size", "dst_dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sglang_unfused", "sglang_fused"],
        line_names=["SGL Kernel Unfused", "SGL Kernel Fused"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="per-token-group-quant-8bit-fused-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, group_size, dst_dtype, provider):
    device = torch.device("xpu")
    hidden_dim = 7168

    x = prepare_token_group_quant_test_data(batch_size * seq_len, hidden_dim)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "sglang_unfused":
        fn = lambda: sglang_per_token_group_quant_8bit_layer(x, group_size, dst_dtype, 1e-10, True, True, True, False, None, True)
    elif provider == "sglang_fused":
        fn = lambda: sglang_per_token_group_quant_8bit_layer(x, group_size, dst_dtype, 1e-10, True, True, True, True, None, True)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":

    calculate_diff(batch_size=2, seq_len=32, group_size=128, dst_dtype=fp8_type_)

    benchmark.run(print_data=True)
