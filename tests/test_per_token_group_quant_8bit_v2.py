import itertools
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

_is_hip = False
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
fp8_dtype = fp8_type_
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max


enable_sgl_per_token_group_quant_8bit = True
import sgl_kernel

 
def create_per_token_group_quant_test_data(num_tokens, hidden_dim, num_ranks, flags):
    device = torch.device("xpu")
    dtype = torch.bfloat16

    seed = num_tokens * 10000 + hidden_dim
    gen_cpu = torch.Generator(device="cpu")
    gen_cpu.manual_seed(seed)
    gen_xpu = torch.Generator(device="xpu")
    gen_xpu.manual_seed(seed)

    if flags["fuse_silu_and_mul"]:
        effective_hidden_dim = hidden_dim * 2
    else:
        effective_hidden_dim = hidden_dim
    del hidden_dim

    if (masked_layout_mode := flags["masked_layout_mode"]) is not None:
        num_max_dispatch_tokens_per_rank = 768
        num_global_experts = 288
        num_local_experts, remainder = divmod(num_global_experts, num_ranks)
        assert remainder == 0

        # mimic DeepEP low_latency_dispatch output
        x = torch.randn(
            num_local_experts,
            num_max_dispatch_tokens_per_rank * num_ranks,
            effective_hidden_dim,
            device=device,
            dtype=dtype,
            generator=gen_xpu,
        )

        if masked_layout_mode == "balanced":
            masked_m = _compute_balanced_split(num_tokens, num_local_experts)
        elif masked_layout_mode == "imbalanced":
            masked_m = _compute_imbalanced_split(
                num_tokens, num_local_experts, gen_cpu=gen_cpu
            )
        elif masked_layout_mode == "extreme":
            masked_m = torch.tensor(
                [num_tokens] + [0] * (num_local_experts - 1), dtype=torch.int
            )
        else:
            raise NotImplementedError
        print(f"{masked_layout_mode=} {masked_m=} {x.shape=}")

        masked_m = masked_m.to(device)

        return x, masked_m
    else:
        x = torch.randn(
            num_tokens,
            effective_hidden_dim,
            device=device,
            dtype=dtype,
            generator=gen_xpu,
        )
        x[torch.randn(x.shape, device=device, generator=gen_xpu) < 0.001] *= 10
        return x, None

# COPIED FROM DeepGEMM
def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y
    
# COPIED FROM DeepGEMM
def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y

def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round scale to nearest power of 2 for UE8M0 format."""
    exp_s = torch.ceil(torch.log2(x.abs().clamp(min=1e-10)))
    y_s_quant = (exp_s.to(torch.int32) + 127).to(torch.int32)
    scale = torch.pow(2.0, torch.ceil(torch.log2(x.abs().clamp(min=1e-10))))
    return y_s_quant, scale
    #return torch.pow(2.0, torch.ceil(torch.log2(x.abs().clamp(min=1e-10))))

def create_per_token_group_quant_fp8_output_scale(
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
        aligned_mn = align(x_s_mn, 4)
        aligned_k = align(x_s_k, 4)
        # TODO(FIXME): Fix cuda kernel and recover here to empty.
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

def sgl_per_token_group_quant_8bit_kernel(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
) -> None:
    if enable_v2 is None:
        from sglang.srt.utils import get_bool_env_var

        enable_v2 = get_bool_env_var("SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2")

    if enable_v2:
        return torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit_v2.default(
            input,
            output_q,
            output_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
            fuse_silu_and_mul,
            masked_m,
        )

    assert not fuse_silu_and_mul, "only v2 support fuse_silu_and_mul"
    assert masked_m is None, "only v2 support masked_m"
    torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit.default(
        input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
    )

# For legacy usage
sgl_per_token_group_quant_fp8_kernel = sgl_per_token_group_quant_8bit_kernel
sgl_per_token_group_quant_int8_kernel = sgl_per_token_group_quant_8bit_kernel

def sglang_per_token_group_quant_int8_layer(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
    enable_v2: Optional[bool] = None,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    # Temporary
    if enable_sgl_per_token_group_quant_8bit:
        sgl_per_token_group_quant_8bit_kernel(
            x, x_q, x_s, group_size, eps, int8_min, int8_max, enable_v2=enable_v2
        )
    else:
        assert not enable_v2
        sgl_per_token_group_quant_int8_kernel(x, x_q, x_s, group_size, eps, int8_min, int8_max)

    return x_q, x_s

def sglang_per_token_group_quant_fp8_layer(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))

    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )

    if x.shape[0] > 0:
        # Temporary
        if enable_sgl_per_token_group_quant_8bit:
            sgl_per_token_group_quant_8bit_kernel(
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
                enable_v2=enable_v2,
            )
        else:
            assert not enable_v2
            sgl_per_token_group_quant_fp8_kernel(
                x, x_q, x_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
            )

    return x_q, x_s

# TODO maybe unify int8 and fp8 code later
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

    if dst_dtype == torch.int8:
        assert not column_major_scales
        assert not scale_tma_aligned
        assert not fuse_silu_and_mul
        assert masked_m is None
        return sglang_per_token_group_quant_int8_layer(
            x=x,
            group_size=group_size,
            eps=eps,
            dtype=dst_dtype,
            enable_v2=enable_v2,
        )

    return sglang_per_token_group_quant_fp8_layer(
        x=x,
        group_size=group_size,
        eps=eps,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
        enable_v2=enable_v2,
    )

configs = list(
    itertools.product(
        [1, 4, 16, 64, 127, 128, 512, 1024, 4096, 8192],  # num_tokens
        [128,  256, 384, 512, 1024, 1536, 1664, 2048, 4096, 7168, 16384],  # hidden_dim
        [16, 32, 64, 128],  # group_size
        [None],  # num_ranks
        [fp8_type_, torch.int8],  # dtype
        [
            dict(
                column_major_scales=False,
                scale_tma_aligned=False,
                scale_ue8m0=False,
                fuse_silu_and_mul=False,
                masked_layout_mode=None,
            ),
            dict(
                column_major_scales=True,
                scale_tma_aligned=False,
                scale_ue8m0=False,
                fuse_silu_and_mul=False,
                masked_layout_mode=None,
            ),
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
                fuse_silu_and_mul=False,
                masked_layout_mode=None,
            ),
            # dict(
            #     column_major_scales=True,
            #     scale_tma_aligned=True,
            #     scale_ue8m0=True,
            #     fuse_silu_and_mul=False,
            #     masked_layout_mode=None,
            # ),                                    
        ],
    )
) + list(
    itertools.product(
        [1, 4, 1 * 8, 4 * 8, 64 * 8, 256 * 8, 768 * 8],
        # TODO support more
        [2048],
        [128],
        [8, 16, 32, 48],
        [fp8_type_],
        [
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
                masked_layout_mode=None,
            ),
            # dict(
            #     column_major_scales=True,
            #     scale_tma_aligned=True,
            #     scale_ue8m0=True,
            #     fuse_silu_and_mul=True,
            #     masked_layout_mode="balanced",
            # ),
            # dict(
            #     column_major_scales=True,
            #     scale_tma_aligned=True,
            #     scale_ue8m0=True,
            #     fuse_silu_and_mul=True,
            #     masked_layout_mode="imbalanced",
            # ),
            # dict(
            #     column_major_scales=True,
            #     scale_tma_aligned=True,
            #     scale_ue8m0=True,
            #     fuse_silu_and_mul=True,
            #     masked_layout_mode="extreme",
            # ),
        ],
    )
)

# Pure PyTorch reference implementations
def per_token_group_quant_fp8_ref(
    x: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-10,
    scale_ue8m0: bool = False,
    fuse_silu_mul: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for per-token group FP8 quantization."""
    assert x.dim() == 2 and x.size(1) % group_size == 0
    num_tokens, hidden_dim = x.shape
    if fuse_silu_mul:
        assert hidden_dim % 2 == 0, "hidden_dim must be even for fused silu_mul"
        half = hidden_dim // 2
        x_primary = x[:, :half]
        x_secondary = x[:, half:]

        # reshape into groups
        x_primary = x_primary.view(num_tokens, -1, group_size)
        x_secondary = x_secondary.view(num_tokens, -1, group_size)

        # --- fused op (matches CUDA) ---
        x_view = torch.nn.functional.silu(x_primary) * x_secondary
    else:
        x_view = x.view(num_tokens, -1, group_size)
    x_amax = x_view.abs().float().amax(dim=2).clamp(min=eps)
    scales = x_amax / 448.0  # FP8 E4M3 max
    quant_scales = scales

    if scale_ue8m0:
        quant_scales, scales = ceil_to_ue8m0(scales)

    x_quantized = (x_view / scales.unsqueeze(2)).to(torch.float8_e4m3fn)
    #return x_quantized.view(num_tokens, hidden_dim), quant_scales
    if fuse_silu_mul:
        return x_quantized.reshape(num_tokens, -1), quant_scales
    else:
        return x_quantized.view(num_tokens, hidden_dim), quant_scales    

def per_token_group_quant_fp8_ref_xxx(
    x: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-10,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for per-token group FP8 quantization."""
    assert x.dim() == 2 and x.size(1) % group_size == 0
    num_tokens, hidden_dim = x.shape
    x_view = x.view(num_tokens, -1, group_size)
    x_amax = x_view.abs().float().amax(dim=2).clamp(min=eps)
    scales = x_amax / 448.0  # FP8 E4M3 max
    quant_scales = scales

    if scale_ue8m0:
        quant_scales, scales = ceil_to_ue8m0(scales)

    x_quantized = (x_view / scales.unsqueeze(2)).to(torch.float8_e4m3fn)
    return x_quantized.view(num_tokens, hidden_dim), quant_scales

def per_token_group_quant_int8_ref(
    x: torch.Tensor, group_size: int = 128, eps: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for per-token group INT8 quantization."""
    assert x.dim() == 2 and x.size(1) % group_size == 0
    num_tokens, hidden_dim = x.shape
    x_view = x.view(num_tokens, -1, group_size)
    x_amax = x_view.abs().float().amax(dim=2).clamp(min=eps)
    scales = x_amax / 127.0  # INT8 max
    x_scaled = (x_view / scales.unsqueeze(2)).clamp(-127.0, 127.0)
    x_quantized = x_scaled.to(torch.int8)
    return x_quantized.view(num_tokens, hidden_dim), scales

def sglang_per_token_group_quant_fp8_layer_ref(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))

    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )

    if x.shape[0] > 0:
        # Temporary
        if enable_sgl_per_token_group_quant_8bit:
            x_q, x_s = per_token_group_quant_fp8_ref(
                x, group_size, eps, scale_ue8m0, fuse_silu_and_mul
            )                        
        else:
            #assert not enable_v2
            x_q, x_s = per_token_group_quant_fp8_ref(
                x, group_size, eps, scale_ue8m0
            )                        

    return x_q, x_s

def sglang_per_token_group_quant_int8_layer_ref(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
    enable_v2: Optional[bool] = None,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    # Temporary
    if enable_sgl_per_token_group_quant_8bit:
        # sgl_per_token_group_quant_8bit(
        #     x, x_q, x_s, group_size, eps, int8_min, int8_max, enable_v2=enable_v2
        # )
        x_q, x_s = per_token_group_quant_int8_ref(
            x, group_size, eps
        )        
    else:
        assert not enable_v2
        #sgl_per_token_group_quant_int8(x, x_q, x_s, group_size, eps, int8_min, int8_max)
        x_q, x_s = per_token_group_quant_int8_ref(
            x, group_size, eps
        )        

    return x_q, x_s

def per_token_group_quant_8bit_ref(
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

    if dst_dtype == torch.int8:
        assert not column_major_scales
        assert not scale_tma_aligned
        assert not fuse_silu_and_mul
        assert masked_m is None
        return sglang_per_token_group_quant_int8_layer_ref(
            x=x,
            group_size=group_size,
            eps=eps,
            dtype=dst_dtype,
            enable_v2=enable_v2,
        )

    return sglang_per_token_group_quant_fp8_layer_ref(
        x=x,
        group_size=group_size,
        eps=eps,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
        enable_v2=enable_v2,
    )

def assert_all_close_or_tiny_diff(a: torch.Tensor, b: torch.Tensor):
    assert (a.shape == b.shape) and (
        a.dtype == b.dtype
    ), f"{a.shape=} {b.shape=} {a.dtype=} {b.dtype=}"
    numel = a.numel()

    if a.dtype == torch.float8_e4m3fn:
        a_u8 = a.view(torch.uint8)
        b_u8 = b.view(torch.uint8)
        diff_u8 = (a_u8.to(torch.int16) - b_u8.to(torch.int16)).abs()

        count_diff_sign = ((a_u8 >= 0) & (b_u8 < 0)).sum().item()
        count_tiny_diff = (diff_u8 == 1).sum().item()
        count_large_diff = (diff_u8 >= 2).sum().item()
    elif a.dtype == torch.int8:
        diff = (a.to(torch.int16) - a.to(torch.int16)).abs()
        count_diff_sign = ((a >= 0) & (b < 0)).sum().item()
        count_tiny_diff = (diff == 1).sum().item()
        count_large_diff = (diff >= 2).sum().item()
    else:
        raise NotImplementedError

    assert (
        (count_diff_sign == 0)
        and (count_large_diff == 0)
        and (
            (count_tiny_diff / numel < 0.005)
            or ((count_tiny_diff / numel < 0.04) and (numel <= 4096))
        )
    ), f"{count_diff_sign=} {count_tiny_diff=} {count_large_diff=} {numel=} {a=} {b=}"

def process_scales(
    x_s: torch.Tensor,
    num_tokens: int,
    column_major_scales: bool,
    scale_ue8m0: bool,
) -> torch.Tensor:

    T = num_tokens

    if scale_ue8m0:
        # ✅ FIX: ensure contiguous before reinterpret
        x_bytes = x_s.contiguous().view(torch.uint8)

        x_bytes = x_bytes.view(T, -1, 4)

        if column_major_scales:
            # column-major unpack
            x_bytes = x_bytes.permute(1, 2, 0)   # [G/4, 4, T]
            x_unpacked = x_bytes.reshape(-1, T)  # [G, T]
            x_unpacked = x_unpacked.transpose(0, 1)
        else:
            # row-major unpack
            x_unpacked = x_bytes.reshape(T, -1)

        return x_unpacked.to(torch.int32)
    else:
        return x_s

    # else:
    #     if column_major_scales:
    #         return x_s.transpose(0, 1).contiguous().to(torch.int32)
    #     else:
    #         return x_s.to(torch.int32)

@pytest.mark.parametrize(
    "num_tokens, hidden_dim, group_size, num_ranks, dst_dtype, flags", configs
)
def test_per_token_group_quant_with_column_major(
    num_tokens,
    hidden_dim,
    group_size,
    num_ranks,
    dst_dtype,
    flags,
):
    print(
        f"{num_tokens=} {hidden_dim=} {group_size=} {num_ranks=} {dst_dtype=} {flags=}"
    )
     
    if (flags["scale_ue8m0"] and (group_size != 128)) or (
        (dst_dtype == torch.int8) and flags["column_major_scales"]
    ):
        pytest.skip()
        return

    x, masked_m = create_per_token_group_quant_test_data(
        num_tokens=num_tokens, hidden_dim=hidden_dim, num_ranks=num_ranks, flags=flags
    )

    # print("hack data!!!")
    # x = torch.full_like(x, fill_value=100)

    execute_kwargs = dict(
        x=x,
        masked_m=masked_m,
        group_size=group_size,
        eps=1e-10,
        dst_dtype=dst_dtype,
        **{k: v for k, v in flags.items() if k not in ["masked_layout_mode"]},
    )

    def _postprocess(x_q, x_s):
        if masked_m is not None:
            print(f"Mask tokens after {masked_m} to be zero")
            for i in range(len(masked_m)):
                x_q[i, masked_m[i] :, :] = 0
                x_s[i, masked_m[i] :, :] = 0
        return x_q, x_s

    x_q_triton, x_s_triton = _postprocess(
        *per_token_group_quant_8bit_ref(**execute_kwargs)
    )

    x_q_sglang, x_s_sglang = _postprocess(
        *sglang_per_token_group_quant_8bit_layer(**execute_kwargs, enable_v2=True)
    )

    # Ref scale as per flags    
    x_s_sglang = process_scales(
        x_s_sglang,
        num_tokens=x_q_sglang.shape[-2],
        column_major_scales=flags["column_major_scales"],   # or False (test both)
        scale_ue8m0=flags["scale_ue8m0"]
    )

    try:
        assert_all_close_or_tiny_diff(x_q_triton, x_q_sglang)
        torch.testing.assert_close(
            x_s_triton.contiguous(),
            x_s_sglang.contiguous(),
            rtol=1e-3,
            atol=1e-5,
            msg=lambda message: message + f" {x_s_triton=} {x_s_sglang=}",
        )
    except AssertionError:
        print(
            f"{x.shape=} {x_q_triton.shape=} {x_s_triton.shape=} {x_q_sglang.shape=} {x_s_sglang.shape=}"
        )
        print(f"{x=}")
        print(f"{masked_m=}")
        print(f"{x_q_triton=}")
        print(f"{x_s_triton=}")
        print(f"{x_q_sglang=}")
        print(f"{x_s_sglang=}")

        raise

if __name__ == "__main__":
    pytest.main([__file__])