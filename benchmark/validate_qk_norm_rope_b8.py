"""Validate the full Muse B8 prefill launch at batch and tail boundaries."""

from muse_scope import require_muse_arch_xe35, require_muse_b8_xe35

require_muse_b8_xe35()

import argparse
import hashlib
from pathlib import Path
import json

import torch
from sgl_kernel import fused_qk_norm_rope


def main_xe35():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--q-heads", type=int, default=128)
    parser.add_argument("--k-heads", type=int, default=16)
    parser.add_argument("--tail-heads", type=int, default=144)
    parser.add_argument("--head-dim", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--entry", choices=("generic", "xe35"), default="generic")
    args = parser.parse_args()
    if args.tokens not in (8, 32768) or (
        args.q_heads,
        args.k_heads,
        args.tail_heads,
        args.head_dim,
    ) != (128, 16, 144, 64):
        parser.error("requires B8 T8/T32768 Q128/K16/tail144/D64")
    require_muse_arch_xe35()
    if args.entry == "xe35":
        from sgl_kernel import fused_qk_norm_rope_xe35

        invoke = fused_qk_norm_rope_xe35
    else:
        invoke = fused_qk_norm_rope
    if min(args.tokens, args.q_heads, args.k_heads) <= 0 or args.tail_heads < 0:
        parser.error(
            "tokens and Q/K heads must be positive; tail-heads must be nonnegative"
        )
    torch.manual_seed(42)
    tokens = args.tokens
    heads_q, heads_k, heads_v, dim = (
        args.q_heads,
        args.k_heads,
        args.tail_heads,
        args.head_dim,
    )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    eps, base = 1e-5, 500000.0
    scale = 43.7840518911 / (dim**0.5)
    qkv = torch.randn(
        tokens,
        (heads_q + heads_k + heads_v) * dim,
        device="xpu:0",
        dtype=dtype,
    )
    # Muse-Spark B8: T8 decode/T32768 prefill, Q128/K16/D64, gated tail144.
    original = qkv.cpu()
    q_weight = torch.full((dim,), scale, device="xpu:0", dtype=dtype)
    k_weight = torch.ones(dim, device="xpu:0", dtype=dtype)
    positions = torch.arange(tokens, device="xpu:0", dtype=torch.int32) % 4096
    inv_freq = base ** (
        -torch.arange(0, dim, 2, dtype=torch.float64, device="cpu") / dim
    )
    invoke(
        qkv,
        heads_q,
        heads_k,
        heads_v,
        dim,
        eps,
        q_weight,
        k_weight,
        base,
        False,
        positions,
        rotary_dim=dim,
    )
    torch.xpu.synchronize()
    actual = qkv.cpu()
    positions_cpu = positions.cpu()
    tolerance = 5e-3 if dtype == torch.bfloat16 else 1e-3
    max_abs, sum_abs, elements, violations = 0.0, 0.0, 0, 0
    error_bins = {bound: 0 for bound in (1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2)}
    for start in range(0, tokens, 512):
        rows = (
            original[start : start + 512]
            .reshape(-1, heads_q + heads_k + heads_v, dim)
            .float()
        )
        q, k = rows[:, :heads_q], rows[:, heads_q : heads_q + heads_k]
        q = (
            q
            * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + eps)
            * q_weight.float().cpu()
        )
        k = (
            k
            * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + eps)
            * k_weight.float().cpu()
        )
        theta = (
            positions_cpu[start : start + 512].float()[:, None]
            * inv_freq.float()[None, :]
        )
        cos, sin = theta.cos()[:, None, :], theta.sin()[:, None, :]

        def rotate(values):
            even, odd = values[..., ::2], values[..., 1::2]
            return torch.stack(
                (even * cos - odd * sin, odd * cos + even * sin), dim=-1
            ).flatten(-2)

        expected = torch.cat((rotate(q), rotate(k)), dim=1)
        got = actual[start : start + 512, : (heads_q + heads_k) * dim].reshape_as(
            expected
        )
        difference = (got.float() - expected.float()).abs()
        max_abs = max(max_abs, difference.max().item())
        sum_abs += difference.sum().item()
        elements += difference.numel()
        for bound in error_bins:
            error_bins[bound] += (difference <= bound).sum().item()
        violations += (
            (
                (~torch.isfinite(got))
                | (~torch.isfinite(expected))
                | (difference > tolerance + tolerance * expected.float().abs())
            )
            .sum()
            .item()
        )
    tail_equal = torch.equal(
        actual[:, (heads_q + heads_k) * dim :].view(torch.uint16),
        original[:, (heads_q + heads_k) * dim :].view(torch.uint16),
    )
    mapped = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libsgl-ops-sycl-FusedQKNormRope" in line
        }
    )
    libraries = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in mapped
    }
    output_hash = hashlib.sha256(
        actual.view(torch.uint16).numpy().tobytes()
    ).hexdigest()
    input_hash = hashlib.sha256(
        original.view(torch.uint16).numpy().tobytes()
    ).hexdigest()
    print(
        json.dumps(
            {
                "entry": args.entry,
                "input_sha256": input_hash,
                "output_sha256": output_hash,
                "loaded_libraries": libraries,
                "tokens": tokens,
                "dtype": args.dtype,
                "heads": [heads_q, heads_k, heads_v],
                "head_dim": dim,
                "reference": "independent CPU FP32 with FP64 frequency rounded to FP32",
                "atol": tolerance,
                "rtol": tolerance,
                "elements": elements,
                "max_abs": max_abs,
                "mean_abs": sum_abs / elements,
                "error_cumulative_bins": error_bins,
                "violations": violations,
                "tail_equal": tail_equal,
            }
        ),
        flush=True,
    )
    assert violations == 0 and tail_equal


if __name__ == "__main__":
    main_xe35()
