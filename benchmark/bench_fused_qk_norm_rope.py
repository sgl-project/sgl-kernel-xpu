import itertools
import math
import os

import pandas as pd
import sgl_kernel
import torch
import triton
from bench_fused_qk_rope_with_cache import create_cos_sin_cache

# Supported dtypes and their properties
DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8_e4m3fn": torch.float8_e4m3fn,
}
DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp8_e4m3fn": 1,
}


def llama_rms_norm(x, w, eps=1e-6):
    """PyTorch reference implementation of RMS normalization."""
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def apply_rotary_emb_native(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Native PyTorch rotary embedding implementation.
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, rotary_dim // 2]
        sin: [num_tokens, rotary_dim // 2]
        is_neox_style: Whether to use Neox-style or interleaved style
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    if is_neox_style:
        # Neox style: split in half along head dimension
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        # Interleaved style: even and odd indices
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def compute_inv_freq_yarn(
    head_dim: int,
    rotary_dim: int,
    base: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    device: torch.device,
):
    """Compute inverse frequencies for YARN RoPE."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )

    if factor != 1.0:
        # YARN scaling
        dim_range = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)

        # Compute linear interpolation factor
        linear_func = (dim_range - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        ramp_func = torch.clamp(linear_func, 0.0, 1.0)

        inv_freq_extrapolation = inv_freq
        inv_freq_interpolation = inv_freq / factor

        inv_freq = (
            inv_freq_interpolation * (1.0 - ramp_func)
            + inv_freq_extrapolation * ramp_func
        )

    return inv_freq


def fused_qk_norm_rope_reference(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    base: float,
    is_neox: bool,
    position_ids: torch.Tensor,
    factor: float = 1.0,
    low: float = 1.0,
    high: float = 1.0,
    attention_factor: float = 1.0,
    rotary_dim: int = None,
) -> torch.Tensor:
    """
    Reference implementation in PyTorch for testing.

    Args:
        qkv: [num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_dim]
        Other args match the kernel interface
    """
    if rotary_dim is None:
        rotary_dim = head_dim

    num_tokens = qkv.shape[0]
    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Reshape QKV to separate Q, K, V
    qkv_reshaped = qkv.view(num_tokens, total_heads, head_dim)

    q = qkv_reshaped[:, :num_heads_q, :]
    k = qkv_reshaped[:, num_heads_q : num_heads_q + num_heads_k, :]
    v = qkv_reshaped[:, num_heads_q + num_heads_k :, :]

    # Apply RMSNorm to Q and K
    q_normalized = llama_rms_norm(q, q_weight, eps)
    k_normalized = llama_rms_norm(k, k_weight, eps)

    # Compute RoPE frequencies
    inv_freq = compute_inv_freq_yarn(
        head_dim, rotary_dim, base, factor, low, high, qkv.device
    )

    # Compute cos and sin for each position
    positions = position_ids.to(torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()

    # Apply attention factor
    cos = cos * attention_factor
    sin = sin * attention_factor

    # Apply RoPE to Q and K (only to rotary_dim portion)
    q_rot = q_normalized[..., :rotary_dim]
    q_pass = q_normalized[..., rotary_dim:]
    q_rot = apply_rotary_emb_native(q_rot, cos, sin, is_neox)
    q_final = torch.cat([q_rot, q_pass], dim=-1)

    k_rot = k_normalized[..., :rotary_dim]
    k_pass = k_normalized[..., rotary_dim:]
    k_rot = apply_rotary_emb_native(k_rot, cos, sin, is_neox)
    k_final = torch.cat([k_rot, k_pass], dim=-1)

    # Concatenate Q, K, V back together
    result = torch.cat([q_final, k_final, v], dim=1)
    result = result.view(num_tokens, total_heads * head_dim)

    return result


# Benchmark configurations
batch_size_range = [1, 2, 4, 8]
seq_len_range = [64, 128, 256, 512, 1024]
# DeepSeek-V3 config: 128 Q heads, 128 KV heads (MLA)
head_config_range = [
    (32, 8, 8, 128),  # Standard MQA config
    (32, 32, 32, 128),  # Standard MHA config
    (128, 128, 128, 128),  # DeepSeek-V3 style
]
is_neox_range = [True, False]
dtype_range = ["fp16", "bf16", "fp8_e4m3fn"]

configs = []
for batch_size, seq_len, (nq, nk, nv, hd), is_neox, dtype in itertools.product(
    batch_size_range, seq_len_range, head_config_range, is_neox_range, dtype_range
):
    configs.append((batch_size, seq_len, nq, nk, nv, hd, is_neox, dtype))

all_results = []

# Support running the benchmark in "chunked" mode by setting environment variables:
# - NUM_CHUNKS: total number of chunks
# - CHUNK_IDX: index of this chunk (0-based)
num_chunks = int(os.environ.get("NUM_CHUNKS", "1"))
chunk_idx = int(os.environ.get("CHUNK_IDX", "0"))
# Support selecting a single config via env var SINGLE_CONFIG with CSV:
# "batch_size,seq_len,nq,nk,nv,head_dim,is_neox,dtype"
single_cfg = os.environ.get("SINGLE_CONFIG", "")
if single_cfg:
    parts = [p.strip() for p in single_cfg.split(",")]
    if len(parts) != 8:
        raise RuntimeError("SINGLE_CONFIG must have 8 comma-separated fields")
    bsz = int(parts[0])
    sl = int(parts[1])
    nq = int(parts[2])
    nk = int(parts[3])
    nv = int(parts[4])
    hd = int(parts[5])
    is_neox_val = parts[6].lower() in ("1", "true", "yes")
    dtype_val = parts[7]
    configs = [(bsz, sl, nq, nk, nv, hd, is_neox_val, dtype_val)]
    num_chunks = 1
    chunk_idx = 0
if num_chunks > 1:
    total = len(configs)
    chunk_size = (total + num_chunks - 1) // num_chunks
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, total)
    configs = configs[start:end]


def calculate_flops(
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    rotary_dim: int,
) -> int:
    """
    Calculate FLOPs for fused QK norm + RoPE kernel.

    RMS Norm per head: ~2*head_dim (square + normalize + multiply)
    RoPE per head: ~6*rotary_dim (sin, cos, multiply, add operations)
    """
    # RMS norm: 2 FLOPs per element (square, rsqrt, multiply, scale)
    flops_rmsnorm_q = num_tokens * num_heads_q * head_dim * 4
    flops_rmsnorm_k = num_tokens * num_heads_k * head_dim * 4

    # RoPE: ~6 FLOPs per rotary dimension (sin, cos, 4 mul/add)
    flops_rope_q = num_tokens * num_heads_q * rotary_dim * 6
    flops_rope_k = num_tokens * num_heads_k * rotary_dim * 6

    total_flops = flops_rmsnorm_q + flops_rmsnorm_k + flops_rope_q + flops_rope_k

    return total_flops


def calculate_effective_bandwidth(
    batch_size: int,
    seq_len: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    rotary_dim: int,
    time_ms: float,
    bytes_per_elem: int = 2,
) -> dict:
    """
    Calculate effective bandwidth and FLOPs for fused QK norm + RoPE kernel.

    Memory: read/write QKV tensor + read weights
    """
    num_tokens = batch_size * seq_len
    num_heads = num_heads_q + num_heads_k + num_heads_v

    # Input/output QKV tensor
    qkv_bytes = num_tokens * num_heads * head_dim * bytes_per_elem

    # Weight tensors
    weight_bytes = 2 * head_dim * bytes_per_elem  # q_weight + k_weight

    # Total bytes (read QKV + write QKV + read weights)
    total_bytes = 2 * qkv_bytes + weight_bytes

    time_s = time_ms / 1000.0
    bandwidth_gbs = (total_bytes / 1e9) / time_s

    total_flops = calculate_flops(
        num_tokens, num_heads_q, num_heads_k, head_dim, rotary_dim
    )
    gflops = (total_flops / 1e9) / time_s

    return {
        "num_tokens": num_tokens,
        "total_bytes": total_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "total_flops": total_flops,
        "gflops": gflops,
    }


def apply_rotary_emb_from_cache(
    x: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    is_neox_style: bool,
) -> torch.Tensor:
    half_rope_dim = rope_dim // 2
    pos = positions.to(device=x.device, dtype=torch.long).view(-1)
    cos = cos_sin_cache[pos, :half_rope_dim].to(dtype=x.dtype)
    sin = cos_sin_cache[pos, half_rope_dim:].to(dtype=x.dtype)
    cos = cos.view(-1, half_rope_dim).unsqueeze(-2)
    sin = sin.view(-1, half_rope_dim).unsqueeze(-2)

    x_rot = x[..., :rope_dim]
    if is_neox_style:
        x1, x2 = torch.chunk(x_rot, 2, dim=-1)
        out_rot = torch.empty_like(x_rot)
        out_rot[..., :half_rope_dim] = x1 * cos - x2 * sin
        out_rot[..., half_rope_dim:] = x2 * cos + x1 * sin
    else:
        out_rot = torch.empty_like(x_rot)
        x1 = x_rot[..., ::2]
        x2 = x_rot[..., 1::2]
        out_rot[..., ::2] = x1 * cos - x2 * sin
        out_rot[..., 1::2] = x2 * cos + x1 * sin

    if rope_dim == x.shape[-1]:
        return out_rot

    out = x.clone()
    out[..., :rope_dim] = out_rot
    return out


def native_qk_norm_rope_with_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    is_neox_style: bool,
    eps: float,
) -> None:
    q_view = q.reshape(-1, q.shape[-2], q.shape[-1])
    k_view = k.reshape(-1, k.shape[-2], k.shape[-1])

    q_view.copy_(llama_rms_norm(q_view, q_weight, eps))
    k_view.copy_(llama_rms_norm(k_view, k_weight, eps))

    q_view.copy_(
        apply_rotary_emb_from_cache(
            q_view, cos_sin_cache, positions, rope_dim, is_neox_style
        )
    )
    k_view.copy_(
        apply_rotary_emb_from_cache(
            k_view, cos_sin_cache, positions, rope_dim, is_neox_style
        )
    )


def logical_cache_bytes_per_call(
    flat_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    rope_dim: int,
    dtype_name: str,
    position_dtype_name: str,
) -> int:
    dtype_bytes = DTYPE_BYTES[dtype_name]
    position_dtype_bytes = 4 if position_dtype_name == "int32" else 8

    qk_bytes = flat_tokens * (num_heads_q + num_heads_k) * head_dim * dtype_bytes * 2
    weight_bytes = 2 * head_dim * dtype_bytes
    cache_bytes = flat_tokens * rope_dim * 4
    position_bytes = flat_tokens * position_dtype_bytes
    return qk_bytes + weight_bytes + cache_bytes + position_bytes


CACHE_CONFIGS = [
    (False, 3, 1, 4, 2, 64, 32, False, "bf16", "int32"),
    (False, 5, 1, 8, 4, 128, 64, True, "fp16", "int64"),
    (True, 2, 4, 16, 4, 128, 128, False, "bf16", "int32"),
    (True, 1, 8, 32, 8, 256, 128, True, "fp16", "int64"),
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "use_4d",
            "batch_size",
            "seq_len",
            "num_q_heads",
            "num_k_heads",
            "head_dim",
            "rope_dim",
            "is_neox",
            "dtype_name",
            "position_dtype_name",
        ],
        x_vals=[list(config) for config in CACHE_CONFIGS],
        line_arg="provider",
        line_vals=["native_split", "cache_fused"],
        line_names=["Native split", "SGL Kernel cache fused"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Latency (us)",
        plot_name="fused-qk-norm-rope-with-cache-performance",
        args={},
    )
)
def benchmark_cache(
    use_4d,
    batch_size,
    seq_len,
    num_q_heads,
    num_k_heads,
    head_dim,
    rope_dim,
    is_neox,
    dtype_name,
    position_dtype_name,
    provider,
):
    q, k, q_weight, k_weight, cos_sin_cache, positions = make_cache_inputs(
        use_4d,
        batch_size,
        seq_len,
        num_q_heads,
        num_k_heads,
        head_dim,
        rope_dim,
        dtype_name,
        position_dtype_name,
    )

    if provider == "native_split":

        def fn() -> None:
            native_qk_norm_rope_with_cache(
                q,
                k,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                rope_dim,
                is_neox,
                1e-6,
            )

    elif provider == "cache_fused":

        def fn() -> None:
            sgl_kernel.fused_inplace_qknorm_rope(
                q,
                k,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                is_neox,
                1e-6,
            )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    flat_tokens = batch_size * seq_len if use_4d else batch_size
    logical_bytes = logical_cache_bytes_per_call(
        flat_tokens,
        num_q_heads,
        num_k_heads,
        head_dim,
        rope_dim,
        dtype_name,
        position_dtype_name,
    )
    bandwidth_gbs = (logical_bytes / 1e9) / (ms / 1000.0)

    cache_results.append(
        {
            "layout": "4d" if use_4d else "3d",
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_tokens": flat_tokens,
            "num_q_heads": num_q_heads,
            "num_k_heads": num_k_heads,
            "head_dim": head_dim,
            "rope_dim": rope_dim,
            "is_neox": is_neox,
            "dtype": dtype_name,
            "position_dtype": position_dtype_name,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bandwidth_gbs,
            "logical_bytes_mb": logical_bytes / 1e6,
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "batch_size",
            "seq_len",
            "num_heads_q",
            "num_heads_k",
            "num_heads_v",
            "head_dim",
            "is_neox",
            "dtype",
        ],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "sglang"],
        line_names=["PyTorch", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="fused-qk-norm-rope-performance",
        args={},
    )
)
def benchmark(
    batch_size,
    seq_len,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    is_neox,
    dtype,
    provider,
):
    device = torch.device("xpu")
    num_tokens = batch_size * seq_len
    num_heads = num_heads_q + num_heads_k + num_heads_v
    torch_dtype = DTYPE_MAP[dtype]
    is_fp8 = dtype == "fp8_e4m3fn"

    if is_fp8:
        # FP8 tensors: create in float32, clamp to representable range, convert
        qkv = (
            torch.randn(
                num_tokens, num_heads * head_dim, device=device, dtype=torch.float32
            )
            .clamp(-448.0, 448.0)
            .to(torch_dtype)
        )
        q_weight = (
            torch.randn(head_dim, device=device, dtype=torch.float32)
            .clamp(-448.0, 448.0)
            .to(torch_dtype)
        )
        k_weight = (
            torch.randn(head_dim, device=device, dtype=torch.float32)
            .clamp(-448.0, 448.0)
            .to(torch_dtype)
        )
    else:
        qkv = torch.randn(
            num_tokens, num_heads * head_dim, device=device, dtype=torch_dtype
        )
        q_weight = torch.randn(head_dim, device=device, dtype=torch_dtype)
        k_weight = torch.randn(head_dim, device=device, dtype=torch_dtype)

    position_ids = torch.arange(num_tokens, device=device, dtype=torch.int32)

    eps = 1e-6
    base = 10000.0
    rotary_dim = head_dim

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        if is_fp8:
            # PyTorch has no native FP8 compute; upcast to float32 for reference
            qkv_ref = qkv.to(torch.float32)
            q_weight_ref = q_weight.to(torch.float32)
            k_weight_ref = k_weight.to(torch.float32)
            fn = lambda: fused_qk_norm_rope_reference(
                qkv_ref.clone(),
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_dim,
                eps,
                q_weight_ref,
                k_weight_ref,
                base,
                is_neox,
                position_ids,
                factor=1.0,
                low=1.0,
                high=1.0,
                attention_factor=1.0,
                rotary_dim=rotary_dim,
            ).to(torch_dtype)
        else:
            fn = lambda: fused_qk_norm_rope_reference(
                qkv.clone(),
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_dim,
                eps,
                q_weight,
                k_weight,
                base,
                is_neox,
                position_ids,
                factor=1.0,
                low=1.0,
                high=1.0,
                attention_factor=1.0,
                rotary_dim=rotary_dim,
            )
    elif provider == "sglang":
        fn = lambda: sgl_kernel.fused_qk_norm_rope(
            qkv.clone(),
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            eps,
            q_weight,
            k_weight,
            base,
            is_neox,
            position_ids,
            factor=1.0,
            low=1.0,
            high=1.0,
            attention_factor=1.0,
            rotary_dim=rotary_dim,
        )

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    # Calculate effective bandwidth using the correct bytes-per-element for this dtype
    bw_metrics = calculate_effective_bandwidth(
        batch_size,
        seq_len,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        ms,
        bytes_per_elem=DTYPE_BYTES[dtype],
    )

    all_results.append(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_tokens": bw_metrics["num_tokens"],
            "num_heads_q": num_heads_q,
            "num_heads_k": num_heads_k,
            "num_heads_v": num_heads_v,
            "head_dim": head_dim,
            "is_neox": is_neox,
            "dtype": dtype,
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bw_metrics["bandwidth_gbs"],
            "total_bytes_mb": bw_metrics["total_bytes"] / 1e6,
            "total_flops_m": bw_metrics["total_flops"] / 1e6,
            "gflops": bw_metrics["gflops"],
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


CACHE_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def build_cache_configs() -> (
    list[tuple[bool, int, int, int, int, int, int, bool, str, str]]
):
    return list(CACHE_CONFIGS)


cache_configs = CACHE_CONFIGS
cache_results = []


def make_cache_inputs(
    use_4d: bool,
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
    rope_dim: int,
    dtype_name: str,
    position_dtype_name: str,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    dtype = CACHE_DTYPE_MAP[dtype_name]
    position_dtype = torch.int32 if position_dtype_name == "int32" else torch.int64
    device = torch.device("xpu")

    if use_4d:
        q = torch.randn(
            batch_size, seq_len, num_q_heads, head_dim, device=device, dtype=dtype
        ).contiguous()
        k = torch.randn(
            batch_size, seq_len, num_k_heads, head_dim, device=device, dtype=dtype
        ).contiguous()
        flat_tokens = batch_size * seq_len
    else:
        q = torch.randn(
            batch_size, num_q_heads, head_dim, device=device, dtype=dtype
        ).contiguous()
        k = torch.randn(
            batch_size, num_k_heads, head_dim, device=device, dtype=dtype
        ).contiguous()
        flat_tokens = batch_size

    q_weight = torch.randn(head_dim, device=device, dtype=dtype).contiguous()
    k_weight = torch.randn(head_dim, device=device, dtype=dtype).contiguous()
    cos_sin_cache = create_cos_sin_cache(rope_dim, flat_tokens + 1, 10000.0)
    positions = torch.arange(flat_tokens, device=device, dtype=position_dtype)
    return q, k, q_weight, k_weight, cos_sin_cache, positions


# ============================================================================
# DeepSeek-V4 `fused_q_norm_rope` and `fused_k_norm_rope_flashmla` benchmarks
# ============================================================================

# DeepSeek-V4 production (head_dim, rope_dim) shapes: 128/512 pair with the
# production rope_dim=64 (see `dsv4_fused_q_norm_rope` /
# `dsv4_fused_k_norm_rope_flashmla` in
# sgl-kernel/csrc/elementwise/dsv4_norm_rope.cu). head_dim=192 uses rope_dim=84
# instead of 64 here so it actually exercises the XPU warp path (192's
# kElemsPerThread=12 doesn't divide 64, so rope_dim=64 would silently fall
# back to the CTA path).
DSV4_HEAD_ROPE_DIM = [(128, 64), (192, 84), (512, 64)]
DSV4_MAX_POSITION_EMBEDDINGS = 65536

# Only these real DSV4 (head_dim, rope_dim) shapes are benchmarked below --
# no synthetic warp-path-only shapes (e.g. head_dim 64/256).
QNORM_ROPE_CONFIGS = []
for _head_dim, _rope_dim in DSV4_HEAD_ROPE_DIM:
    QNORM_ROPE_CONFIGS.extend(
        [
            (1, 8, _head_dim, _rope_dim, f"dsv4_bz8_h{_head_dim}"),
            (1, 16, _head_dim, _rope_dim, f"dsv4_bz16_h{_head_dim}"),
            (1, 32, _head_dim, _rope_dim, f"dsv4_bz32_h{_head_dim}"),
            (1, 64, _head_dim, _rope_dim, f"dsv4_bz64_h{_head_dim}"),
            (128, 64, _head_dim, _rope_dim, f"dsv4_bz8192_h{_head_dim}"),
            (512, 64, _head_dim, _rope_dim, f"dsv4_bz32768_h{_head_dim}"),
        ]
    )

q_results = []


def native_q_norm_rope(
    q_input: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    B, H, head_dim = q_input.shape
    rope_dim = freqs_cis.shape[-1]
    nope_dim = head_dim - rope_dim

    x = q_input.float()
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(rms + eps)

    freq_rows = freqs_cis[positions.long()].unsqueeze(1)
    freq_re = freq_rows[..., 0::2]
    freq_im = freq_rows[..., 1::2]

    x_rope = x[..., nope_dim:]
    x_re = x_rope[..., 0::2]
    x_im = x_rope[..., 1::2]
    rotated_re = x_re * freq_re - x_im * freq_im
    rotated_im = x_re * freq_im + x_im * freq_re
    rotated = torch.stack([rotated_re, rotated_im], dim=-1).flatten(-2)

    out = torch.cat([x[..., :nope_dim], rotated], dim=-1)
    return out.to(q_input.dtype)


def make_q_norm_rope_inputs(
    dtype_name: str,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    max_pos: int = DSV4_MAX_POSITION_EMBEDDINGS,
):
    dtype = DTYPE_MAP[dtype_name]
    device = torch.device("xpu")

    q_input = torch.randn(
        num_tokens, num_heads, head_dim, device=device, dtype=dtype
    ).contiguous()
    q_output = torch.empty_like(q_input)
    freqs_cis_complex = torch.randn(
        max_pos, rope_dim // 2, dtype=torch.complex64, device=device
    )
    freqs_real = torch.view_as_real(freqs_cis_complex).flatten(-2).contiguous()
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int32, device=device
    )
    return q_input, q_output, freqs_real, positions


def benchmark_q_norm_rope_shape(
    dtype_name: str,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    label: str,
):
    dtype = DTYPE_MAP[dtype_name]
    itemsize = torch.tensor([], dtype=dtype).element_size()
    rows = []

    q_input, q_output, freqs_real, positions = make_q_norm_rope_inputs(
        dtype_name, num_tokens, num_heads, head_dim, rope_dim
    )

    def native_fn() -> None:
        native_q_norm_rope(q_input, freqs_real, positions, 1e-6)

    quantiles = [0.5, 0.2, 0.8]
    native_ms, _, _ = triton.testing.do_bench(native_fn, quantiles=quantiles)

    def fused_fn() -> None:
        sgl_kernel.fused_q_norm_rope(q_input, q_output, freqs_real, positions, 1e-6)

    fused_ms, _, _ = triton.testing.do_bench(fused_fn, quantiles=quantiles)

    qnorm_bytes = num_tokens * num_heads * head_dim * itemsize * 2
    for provider, latency_ms in (
        ("native_unfused", native_ms),
        ("fused_q_norm_rope", fused_ms),
    ):
        latency_us = latency_ms * 1000.0
        rows.append(
            {
                "shape_label": label,
                "provider": provider,
                "dtype": dtype_name,
                "num_tokens": num_tokens,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "rope_dim": rope_dim,
                "bytes_moved": qnorm_bytes,
                "latency_us": round(latency_us, 3),
                "gbps": round(qnorm_bytes / (latency_us * 1e-6) / 1e9, 2),
            }
        )
    return rows


def run_qnorm_rope_benchmarks(dtype_name: str = "bf16"):
    print("\n=== DeepSeek-V4 fused_q_norm_rope shape benchmark ===")
    for num_tokens, num_heads, head_dim, rope_dim, label in QNORM_ROPE_CONFIGS:
        q_results.extend(
            benchmark_q_norm_rope_shape(
                dtype_name, num_tokens, num_heads, head_dim, rope_dim, label
            )
        )

    fieldnames = list(q_results[0].keys())
    print("\t".join(fieldnames))
    for row in q_results:
        print("\t".join(str(row[column]) for column in fieldnames))


def native_k_norm_rope_flashmla(
    kv: torch.Tensor,  # (B, 512) input dtype (bf16/fp16/fp32)
    kv_weight: torch.Tensor,  # (512,) same dtype as kv
    freqs_cis: torch.Tensor,  # (max_pos, 64) fp32 interleaved [re0,im0,re1,im1,...]
    positions: torch.Tensor,  # (B,) int32 or int64
    out_loc: torch.Tensor,  # (B,) int32 cache-slot indices
    kvcache: torch.Tensor,  # (npages, kPageBytes) uint8
    eps: float,
    page_size: int,  # must be a power of 2
) -> None:
    """Pure-PyTorch reference for fused K-norm + RoPE + FlashMLA paged cache
    store. Fixed FlashMLA cache layout (head_dim=512, rope_dim=64,
    nope_dim=448): value slot (576 bytes) = [0..447] FP8 E4M3 nope (7 warps x
    64 elems) + [448..575] BF16 rope (64 elems); scale slot (8 bytes) =
    [0..6] UE8M0 exponent per nope warp + [7] padding. Vectorized (no
    per-token Python loop), stays on the input device.
    """
    fp8_max = 448.0
    head_dim, rope_dim = 512, 64
    nope_dim = head_dim - rope_dim  # 448
    nope_warps = 7  # 64 elems each
    epw = nope_dim // nope_warps  # 64
    value_bytes = 576

    assert kv.shape[-1] == head_dim, f"head_dim must be {head_dim}"
    assert freqs_cis.shape[-1] == rope_dim, f"rope_dim must be {rope_dim}"
    assert (page_size & (page_size - 1)) == 0, "page_size must be a power of 2"

    B = kv.shape[0]
    if B == 0:
        return

    device = kv.device
    page_bits = int(math.log2(page_size))
    page_bytes = kvcache.shape[1]

    def cast_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
        bits = x.float().clamp(min=1e-38).view(torch.int32)
        exp = (bits >> 23) & 0xFF
        round_up = (bits & 0x7FFFFF) != 0
        return exp + round_up.to(torch.int32)

    def inv_scale_ue8m0(ue8m0: torch.Tensor) -> torch.Tensor:
        inv_exp = (254 - ue8m0).clamp(min=0)
        inv_bits = (inv_exp << 23).to(torch.int32)
        inv_f = inv_bits.view(torch.float32)
        return torch.where(ue8m0 >= 254, torch.zeros_like(inv_f), inv_f)

    # Step 1: block-wide RMSNorm with kv_weight.
    x = kv.float()
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    norm_factor = torch.rsqrt(rms + eps)
    x = x * norm_factor * kv_weight.float().unsqueeze(0)

    # Step 2: RoPE on the last rope_dim=64 elements.
    freq = freqs_cis[positions.long()]
    freq_re = freq[:, 0::2]
    freq_im = freq[:, 1::2]

    x_nope = x[:, :nope_dim]
    x_rope = x[:, nope_dim:]
    xr = x_rope[:, 0::2]
    xi = x_rope[:, 1::2]

    rotated_re = xr * freq_re - xi * freq_im
    rotated_im = xr * freq_im + xi * freq_re
    rope_out = torch.stack([rotated_re, rotated_im], dim=-1).flatten(-2)

    # Step 3: per-warp FP8 E4M3 quantization of the nope region.
    x_warps = x_nope.reshape(B, nope_warps, epw)
    abs_max = x_warps.abs().amax(dim=-1)
    scale_raw = abs_max.clamp(min=1e-4) / fp8_max
    ue8m0 = cast_to_ue8m0(scale_raw)
    inv_scale = inv_scale_ue8m0(ue8m0)
    fp8_nope = (
        (x_warps * inv_scale.unsqueeze(-1))
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    )

    # Step 4: flat byte addresses for all scatter writes.
    valid = out_loc >= 0
    if not valid.any():
        return

    pages = out_loc[valid].long() >> page_bits
    offsets = out_loc[valid].long() & (page_size - 1)

    value_base = pages * page_bytes + offsets * value_bytes
    scale_base = pages * page_bytes + page_size * value_bytes + offsets * 8

    flat_cache = kvcache.view(-1)

    # Step 5a: scatter FP8 nope bytes.
    fp8_bytes = fp8_nope[valid].reshape(-1, nope_dim).view(torch.uint8)
    nope_cols = torch.arange(nope_dim, device=device)
    nope_idx = value_base.unsqueeze(1) + nope_cols.unsqueeze(0)
    flat_cache.index_put_((nope_idx.reshape(-1),), fp8_bytes.reshape(-1))

    # Step 5b: scatter BF16 rope bytes.
    rope_bf16 = rope_out[valid].to(torch.bfloat16)
    rope_bytes = rope_bf16.view(torch.uint8)
    rope_cols = torch.arange(128, device=device)
    rope_idx = (value_base + nope_dim).unsqueeze(1) + rope_cols.unsqueeze(0)
    flat_cache.index_put_((rope_idx.reshape(-1),), rope_bytes.reshape(-1))

    # Step 5c: scatter UE8M0 scale bytes.
    ue8m0_bytes = ue8m0[valid].to(torch.uint8)
    scale_cols = torch.arange(nope_warps, device=device)
    scale_idx = scale_base.unsqueeze(1) + scale_cols.unsqueeze(0)
    flat_cache.index_put_((scale_idx.reshape(-1),), ue8m0_bytes.reshape(-1))


FLASHMLA_K_CONFIGS = [
    (1, "dsv4_k_bz1"),
    (8, "dsv4_k_bz8"),
    (32, "dsv4_k_bz32"),
    (128, "dsv4_k_bz128"),
    (512, "dsv4_k_bz512"),
    (2048, "dsv4_k_bz2048"),
]
k_results = []


def make_flashmla_k_inputs(
    dtype_name: str,
    num_tokens: int,
    head_dim: int = DSV4_HEAD_ROPE_DIM[-1][0],  # K-path (FlashMLA) is always 512
    rope_dim: int = DSV4_HEAD_ROPE_DIM[-1][1],
    page_size: int = 256,
    max_pos: int = DSV4_MAX_POSITION_EMBEDDINGS,
):
    dtype = DTYPE_MAP[dtype_name]
    device = torch.device("xpu")

    kv = torch.randn(num_tokens, head_dim, device=device, dtype=dtype).contiguous()
    kv_weight = torch.randn(head_dim, device=device, dtype=dtype).contiguous()
    freqs_cis_complex = torch.randn(
        max_pos, rope_dim // 2, dtype=torch.complex64, device=device
    )
    freqs_real = torch.view_as_real(freqs_cis_complex).flatten(-2).contiguous()
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int32, device=device
    )
    out_loc = torch.randperm(num_tokens, dtype=torch.int32, device=device)

    npages = (num_tokens + page_size - 1) // page_size + 2
    k_page_bytes = page_size * 584
    kvcache = torch.zeros((npages, k_page_bytes), dtype=torch.uint8, device=device)

    return kv, kv_weight, freqs_real, positions, out_loc, kvcache


def benchmark_flashmla_k_shape(
    dtype_name: str,
    num_tokens: int,
    label: str,
    page_size: int = 256,
):
    dtype = DTYPE_MAP[dtype_name]
    itemsize = torch.tensor([], dtype=dtype).element_size()
    head_dim, rope_dim = DSV4_HEAD_ROPE_DIM[-1]  # K-path (FlashMLA) is always 512/64
    rows = []

    kv, kv_weight, freqs_real, positions, out_loc, kvcache = make_flashmla_k_inputs(
        dtype_name, num_tokens, head_dim, rope_dim, page_size
    )

    kvcache_test = kvcache.clone()

    def fused_k_fn():
        sgl_kernel.fused_k_norm_rope_flashmla(
            kv, kv_weight, freqs_real, positions, out_loc, kvcache_test, 1e-6, page_size
        )

    quantiles = [0.5, 0.2, 0.8]
    fused_k_ms, _, _ = triton.testing.do_bench(fused_k_fn, quantiles=quantiles)

    kvcache_ref = kvcache.clone()

    def native_k_fn():
        native_k_norm_rope_flashmla(
            kv, kv_weight, freqs_real, positions, out_loc, kvcache_ref, 1e-6, page_size
        )

    native_k_ms, _, _ = triton.testing.do_bench(native_k_fn, quantiles=quantiles)

    flashmla_k_bytes = num_tokens * (head_dim * itemsize + 584)
    fused_k_us = fused_k_ms * 1000.0
    native_k_us = native_k_ms * 1000.0

    rows.append(
        {
            "shape_label": label,
            "provider": "fused_k_norm_rope_flashmla",
            "dtype": dtype_name,
            "num_tokens": num_tokens,
            "head_dim": head_dim,
            "rope_dim": rope_dim,
            "bytes_moved": flashmla_k_bytes,
            "latency_us": round(fused_k_us, 3),
            "gbps": round(flashmla_k_bytes / (fused_k_us * 1e-6) / 1e9, 2),
        }
    )

    rows.append(
        {
            "shape_label": label,
            "provider": "native_unfused_ref",
            "dtype": dtype_name,
            "num_tokens": num_tokens,
            "head_dim": head_dim,
            "rope_dim": rope_dim,
            "bytes_moved": flashmla_k_bytes,
            "latency_us": round(native_k_us, 3),
            "gbps": round(flashmla_k_bytes / (native_k_us * 1e-6) / 1e9, 2),
        }
    )

    return rows


def run_flashmla_k_benchmarks(dtype_name: str = "bf16"):
    print("\n=== DeepSeek-V4 fused_k_norm_rope_flashmla shape benchmark ===")
    for num_tokens, label in FLASHMLA_K_CONFIGS:
        k_results.extend(benchmark_flashmla_k_shape(dtype_name, num_tokens, label))

    fieldnames = list(k_results[0].keys())
    print("\t".join(fieldnames))
    for row in k_results:
        print("\t".join(str(row[column]) for column in fieldnames))


if __name__ == "__main__":
    print("Running benchmarks...")
    benchmark.run(print_data=True)
    benchmark_cache.run(print_data=True)

    dtype_name = os.environ.get("DTYPE", "bf16")
    run_qnorm_rope_benchmarks(dtype_name)
    run_flashmla_k_benchmarks(dtype_name)

    # Ensure results dir exists and write per-chunk CSV
    os.makedirs("benchmark/results", exist_ok=True)
    df = pd.DataFrame(all_results)
    chunk_label = (
        f"chunk_{os.environ.get('CHUNK_IDX','0')}_of_{os.environ.get('NUM_CHUNKS','1')}"
    )
    out_csv = os.path.join("benchmark/results", f"results_{chunk_label}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote results CSV: {out_csv}")

    cache_df = pd.DataFrame(cache_results)
    cache_out_csv = os.path.join("benchmark/results", f"cache_{chunk_label}.csv")
    cache_df.to_csv(cache_out_csv, index=False)
    print(f"Wrote cache results CSV: {cache_out_csv}")

    # Print bandwidth results
    print("\n" + "=" * 80)
    print("Effective Bandwidth Results")
    print("=" * 80)
    df["bandwidth_gbs"] = df["bandwidth_gbs"].round(2)
    df["total_bytes_mb"] = df["total_bytes_mb"].round(2)
    df["time_us"] = df["time_us"].round(2)
    df["total_flops_m"] = df["total_flops_m"].round(2)
    df["gflops"] = df["gflops"].round(2)

    print(df.to_markdown(index=False))

    # Print summary statistics per provider
    print("\n" + "=" * 80)
    print("Summary Statistics by Provider")
    print("=" * 80)
    summary = df.groupby(["dtype", "provider"]).agg(
        {
            "bandwidth_gbs": ["mean", "min", "max"],
            "time_us": ["mean", "min", "max"],
            "gflops": ["mean", "min", "max"],
        }
    )
    print(summary.to_markdown())

    print("\n" + "=" * 80)
    print("Cache Benchmark Results")
    print("=" * 80)
    if not cache_df.empty:
        cache_df["bandwidth_gbs"] = cache_df["bandwidth_gbs"].round(2)
        cache_df["logical_bytes_mb"] = cache_df["logical_bytes_mb"].round(2)
        cache_df["time_us"] = cache_df["time_us"].round(2)
        print(cache_df.to_markdown(index=False))

    # Print best speedup
    print("\n" + "=" * 80)
    print("Speedup Analysis")
    print("=" * 80)

    # Pivot to compare providers
    pivot = df.pivot_table(
        index=[
            "batch_size",
            "seq_len",
            "num_heads_q",
            "num_heads_k",
            "num_heads_v",
            "head_dim",
            "is_neox",
            "dtype",
        ],
        columns="provider",
        values="time_us",
    )

    if "torch" in pivot.columns and "sglang" in pivot.columns:
        pivot["speedup"] = pivot["torch"] / pivot["sglang"]
        print(f"\nOverall average speedup: {pivot['speedup'].mean():.2f}x")
        print(f"Overall max speedup:     {pivot['speedup'].max():.2f}x")
        print(f"Overall min speedup:     {pivot['speedup'].min():.2f}x")

        # Per-dtype speedup breakdown
        print("\nSpeedup by dtype:")
        for dt in df["dtype"].unique():
            mask = pivot.index.get_level_values("dtype") == dt
            sp = pivot.loc[mask, "speedup"]
            if not sp.empty:
                print(
                    f"  {dt:>12s}: avg={sp.mean():.2f}x  max={sp.max():.2f}x  min={sp.min():.2f}x"
                )
    print(f"Wrote results CSV: {out_csv}")

    # === q_results (fused_q_norm_rope) summary ===
    print("\n" + "=" * 80)
    print("Q Norm+RoPE Benchmark Results")
    print("=" * 80)
    q_df = pd.DataFrame(q_results)
    q_out_csv = os.path.join("benchmark/results", f"q_norm_rope.csv")
    q_df.to_csv(q_out_csv, index=False)
    print(f"Wrote results CSV: {q_out_csv}")

    if not q_df.empty:
        q_df["latency_us"] = q_df["latency_us"].round(3)
        q_df["gbps"] = q_df["gbps"].round(2)
        print(q_df.to_markdown(index=False))

        q_summary = q_df.groupby(["dtype", "provider"]).agg(
            {"latency_us": ["mean", "min", "max"], "gbps": ["mean", "min", "max"]}
        )
        print(q_summary.to_markdown())

        q_pivot = q_df.pivot_table(
            index=["shape_label", "dtype", "num_heads", "head_dim", "rope_dim"],
            columns="provider",
            values="latency_us",
        )
        if (
            "native_unfused" in q_pivot.columns
            and "fused_q_norm_rope" in q_pivot.columns
        ):
            q_pivot["speedup"] = (
                q_pivot["native_unfused"] / q_pivot["fused_q_norm_rope"]
            )
            print(f"\nfused_q_norm_rope avg speedup: {q_pivot['speedup'].mean():.2f}x")
            print(f"fused_q_norm_rope max speedup: {q_pivot['speedup'].max():.2f}x")
            print(f"fused_q_norm_rope min speedup: {q_pivot['speedup'].min():.2f}x")

    # === k_results (fused_k_norm_rope_flashmla) summary ===
    print("\n" + "=" * 80)
    print("K Norm+RoPE FlashMLA Benchmark Results")
    print("=" * 80)
    k_df = pd.DataFrame(k_results)
    k_out_csv = os.path.join("benchmark/results", f"k_norm_rope_flashmla.csv")
    k_df.to_csv(k_out_csv, index=False)
    print(f"Wrote results CSV: {k_out_csv}")

    if not k_df.empty:
        k_df["latency_us"] = k_df["latency_us"].round(3)
        k_df["gbps"] = k_df["gbps"].round(2)
        print(k_df.to_markdown(index=False))

        k_summary = k_df.groupby(["dtype", "provider"]).agg(
            {"latency_us": ["mean", "min", "max"], "gbps": ["mean", "min", "max"]}
        )
        print(k_summary.to_markdown())

        k_pivot = k_df.pivot_table(
            index=["shape_label", "dtype", "head_dim", "rope_dim"],
            columns="provider",
            values="latency_us",
        )
        if (
            "native_unfused_ref" in k_pivot.columns
            and "fused_k_norm_rope_flashmla" in k_pivot.columns
        ):
            k_pivot["speedup"] = (
                k_pivot["native_unfused_ref"] / k_pivot["fused_k_norm_rope_flashmla"]
            )
            print(
                f"\nfused_k_norm_rope_flashmla avg speedup: {k_pivot['speedup'].mean():.2f}x"
            )
            print(
                f"fused_k_norm_rope_flashmla max speedup: {k_pivot['speedup'].max():.2f}x"
            )
            print(
                f"fused_k_norm_rope_flashmla min speedup: {k_pivot['speedup'].min():.2f}x"
            )
