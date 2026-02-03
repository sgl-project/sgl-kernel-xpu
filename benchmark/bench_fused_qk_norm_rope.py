import itertools

import pandas as pd
import torch
import triton
from sgl_kernel import fused_qk_norm_rope


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
seq_len_range = [64, 128, 256, 512, 1024, 2048]
# DeepSeek-V3 config: 128 Q heads, 128 KV heads (MLA)
head_config_range = [
    (32, 8, 8, 128),  # Standard MQA config
    (32, 32, 32, 128),  # Standard MHA config
    (128, 128, 128, 128),  # DeepSeek-V3 style
]
is_neox_range = [True, False]

configs = []
for batch_size, seq_len, (nq, nk, nv, hd), is_neox in itertools.product(
    batch_size_range, seq_len_range, head_config_range, is_neox_range
):
    configs.append((batch_size, seq_len, nq, nk, nv, hd, is_neox))

all_results = []


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
) -> dict:
    """
    Calculate effective bandwidth and FLOPs for fused QK norm + RoPE kernel.

    Memory: read/write QKV tensor + read weights
    """
    num_tokens = batch_size * seq_len
    num_heads = num_heads_q + num_heads_k + num_heads_v

    # Input/output QKV tensor (bf16)
    qkv_bytes = num_tokens * num_heads * head_dim * 2

    # Weight tensors (bf16)
    weight_bytes = 2 * head_dim * 2  # q_weight + k_weight

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
    provider,
):
    device = torch.device("xpu")
    num_tokens = batch_size * seq_len
    num_heads = num_heads_q + num_heads_k + num_heads_v

    qkv = torch.randn(
        num_tokens, num_heads * head_dim, device=device, dtype=torch.bfloat16
    )
    q_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(num_tokens, device=device, dtype=torch.int32)

    eps = 1e-6
    base = 10000.0
    rotary_dim = head_dim

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
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
        fn = lambda: fused_qk_norm_rope(
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

    # Calculate effective bandwidth
    bw_metrics = calculate_effective_bandwidth(
        batch_size,
        seq_len,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        ms,
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
            "provider": provider,
            "time_us": 1000 * ms,
            "bandwidth_gbs": bw_metrics["bandwidth_gbs"],
            "total_bytes_mb": bw_metrics["total_bytes"] / 1e6,
            "total_flops_m": bw_metrics["total_flops"] / 1e6,
            "gflops": bw_metrics["gflops"],
        }
    )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    print("Running benchmarks...")
    benchmark.run(print_data=True)

    # Print bandwidth results
    print("\n" + "=" * 80)
    print("Effective Bandwidth Results")
    print("=" * 80)

    df = pd.DataFrame(all_results)
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
    summary = df.groupby("provider").agg(
        {
            "bandwidth_gbs": ["mean", "min", "max"],
            "time_us": ["mean", "min", "max"],
            "gflops": ["mean", "min", "max"],
        }
    )
    print(summary.to_markdown())

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
        ],
        columns="provider",
        values="time_us",
    )

    if "torch" in pivot.columns and "sglang" in pivot.columns:
        pivot["speedup"] = pivot["torch"] / pivot["sglang"]
        print(f"\nAverage speedup: {pivot['speedup'].mean():.2f}x")
        print(f"Max speedup: {pivot['speedup'].max():.2f}x")
        print(f"Min speedup: {pivot['speedup'].min():.2f}x")
