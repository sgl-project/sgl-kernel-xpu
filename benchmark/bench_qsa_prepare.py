"""Benchmark fused Xe2 QSA preparation against equivalent Torch eager operations."""

import argparse
import time

import sgl_kernel
import torch


def time_ms(fn, repetitions=60):
    for _ in range(10):
        fn()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - start) * 1000 / repetitions


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


def bench(rows, ratio):
    dim, heads = 128, 4
    state_slots = max(rows, ratio)
    qk = torch.randn(rows, (heads + 1) * dim, device="xpu", dtype=torch.bfloat16)
    cache = torch.randn(512, dim, device="xpu", dtype=torch.float32)
    axes = torch.zeros(dim // 2, device="xpu", dtype=torch.int32)
    pos = torch.arange(rows, device="xpu", dtype=torch.int64) % 512
    slots = torch.arange(rows, device="xpu", dtype=torch.int64)
    weight = torch.randn(dim, device="xpu", dtype=torch.bfloat16) * 0.03
    state = torch.randn((state_slots, dim), device="xpu", dtype=torch.bfloat16)
    rope_state = torch.zeros((state_slots, 3), device="xpu", dtype=torch.int64)
    groups = torch.arange(
        state_slots // ratio * ratio, device="xpu", dtype=torch.int32
    ).reshape(-1, ratio)
    writes = torch.arange(groups.shape[0], device="xpu", dtype=torch.int32)
    compressed = torch.empty((groups.shape[0], dim), device="xpu", dtype=torch.bfloat16)

    def fused_q():
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

    def eager_q():
        q = norm(qk[:, : heads * dim].reshape(rows, heads, dim), weight)
        rotate(q, pos[:, None], cache)
        state[slots] = qk[:, heads * dim :]
        rope_state[slots] = pos[:, None].expand(-1, 3)

    def fused_k():
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

    def eager_k():
        pooled = state[groups.long()].float().mean(1).to(state.dtype)
        compressed[writes.long()] = rotate(
            norm(pooled, weight), rope_state[groups[:, 0].long(), 0], cache
        )

    fused_q_ms, eager_q_ms = time_ms(fused_q), time_ms(eager_q)
    fused_q()
    fused_k_ms, eager_k_ms = time_ms(fused_k), time_ms(eager_k)
    print(
        f"rows={rows:5} groups={groups.shape[0]:5} "
        f"Q fused/eager={fused_q_ms:.3f}/{eager_q_ms:.3f} ms "
        f"K fused/eager={fused_k_ms:.3f}/{eager_k_ms:.3f} ms"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 256, 2048])
    parser.add_argument("--ratio", type=int, default=4)
    args = parser.parse_args()
    for count in args.rows:
        bench(count, args.ratio)
