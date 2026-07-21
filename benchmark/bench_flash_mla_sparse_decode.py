"""Benchmark: sparse MLA fp8 decode (sgl_kernel).

Times sgl_kernel.flash_mla_sparse_decode across decode shapes
(including the extra_kv second-pool path) and reports average latency +
effective memory bandwidth.

Usage:
  python benchmark/bench_flash_mla_sparse_decode.py
"""

import torch
from sgl_kernel import flash_mla_sparse_decode

# ── sparse fp8 KV layout constants ──
NOPE_DIM = 448
ROPE_DIM = 64
D_QK = NOPE_DIM + ROPE_DIM  # 512
D_V = 512
DATA_BYTES_PER_TOKEN = NOPE_DIM + ROPE_DIM * 2  # 576
SCALE_BYTES_PER_TOKEN = 8
HEAD_BYTES = DATA_BYTES_PER_TOKEN + SCALE_BYTES_PER_TOKEN  # 584

BLOCK_SIZE = 64
NUM_BLOCKS = 256  # 16384 cached tokens

# (batch, h_q, topk, extra_topk)
CONFIGS = [
    (64, 16, 2048, 0),
    (64, 32, 2048, 0),
    (64, 128, 2048, 0),
    (128, 16, 512, 0),
    (128, 128, 512, 0),
    (64, 16, 2048, 512),
    (128, 128, 512, 512),
]


def _pack_sparse_fp8_kv_deepseek_v4(kv: torch.Tensor):
    num_blocks, block_size, num_heads, d_qk = kv.shape
    assert num_heads == 1 and d_qk == D_QK
    packed_storage = torch.empty(
        (num_blocks, block_size * HEAD_BYTES), device=kv.device, dtype=torch.uint8
    )
    dequant = torch.empty_like(kv, dtype=torch.float32)
    nope = kv[:, :, 0, :NOPE_DIM].float()
    rope = kv[:, :, 0, NOPE_DIM:].contiguous()
    scale_bytes = torch.full(
        (num_blocks, block_size, SCALE_BYTES_PER_TOKEN),
        127,
        device=kv.device,
        dtype=torch.uint8,
    )
    scale_bytes[..., 7] = 0
    nope_fp8 = nope.to(torch.float8_e4m3fn)
    dequant[:, :, 0, :NOPE_DIM] = nope_fp8.float()
    dequant[:, :, 0, NOPE_DIM:] = rope.float()
    data_region = packed_storage[:, : block_size * DATA_BYTES_PER_TOKEN].view(
        num_blocks, block_size, DATA_BYTES_PER_TOKEN
    )
    scale_region = packed_storage[:, block_size * DATA_BYTES_PER_TOKEN :].view(
        num_blocks, block_size, SCALE_BYTES_PER_TOKEN
    )
    data_region[..., :NOPE_DIM] = nope_fp8.contiguous().view(torch.uint8)
    data_region[..., NOPE_DIM:] = rope.view(torch.uint8).view(
        num_blocks, block_size, ROPE_DIM * 2
    )
    scale_region.copy_(scale_bytes)
    packed = packed_storage.as_strided(
        (num_blocks, block_size, num_heads, HEAD_BYTES),
        (block_size * HEAD_BYTES, DATA_BYTES_PER_TOKEN, HEAD_BYTES, 1),
    ).view(torch.float8_e4m3fn)
    return packed, dequant


def _make_indices(b, topk, s_kv, device):
    return torch.stack(
        [
            torch.stack([torch.randperm(s_kv, device=device)[:topk].to(torch.int32)])
            for _ in range(b)
        ]
    )  # [b, 1, topk]


def build_inputs(batch, h_q, topk, extra_topk=0, device="xpu", seed=0):
    torch.manual_seed(seed)
    b, s_q = batch, 1
    q = (torch.randn((b, s_q, h_q, D_QK), device=device, dtype=torch.float32) * 0.5).to(
        torch.bfloat16
    )
    logical_kv = (
        torch.randn(
            (NUM_BLOCKS, BLOCK_SIZE, 1, D_QK), device=device, dtype=torch.bfloat16
        )
        * 0.5
    )
    packed_kv, _ = _pack_sparse_fp8_kv_deepseek_v4(logical_kv)
    s_kv = NUM_BLOCKS * BLOCK_SIZE
    indices = _make_indices(b, topk, s_kv, device)

    extra_kv = None
    extra_indices = None
    extra_topk_length = None
    if extra_topk > 0:
        logical_extra = (
            torch.randn(
                (NUM_BLOCKS, BLOCK_SIZE, 1, D_QK), device=device, dtype=torch.bfloat16
            )
            * 0.5
        )
        extra_kv, _ = _pack_sparse_fp8_kv_deepseek_v4(logical_extra)
        extra_indices = _make_indices(b, extra_topk, s_kv, device)
        extra_topk_length = torch.full(
            (b,), extra_topk, dtype=torch.int32, device=device
        )
    return q, packed_kv, indices, extra_kv, extra_indices, extra_topk_length


def effective_bytes(batch, h_q, topk, extra_topk=0):
    # q read (bf16) + gathered fp8 kv read (packed head bytes) + out write (bf16)
    q_bytes = batch * h_q * D_QK * 2
    kv_bytes = batch * (topk + extra_topk) * HEAD_BYTES
    out_bytes = batch * h_q * D_V * 2
    return q_bytes + kv_bytes + out_bytes


def bench(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    if not torch.xpu.is_available():
        print("XPU not available")
        return

    hdr = (
        f"{'head_q':>6s} {'batch':>6s} {'topk':>6s} {'extra':>6s} "
        f"{'ms':>9s} {'GB/s':>9s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for batch, h_q, topk, extra_topk in CONFIGS:
        q, packed_kv, indices, extra_kv, extra_indices, extra_topk_length = (
            build_inputs(batch, h_q, topk, extra_topk)
        )
        sm_scale = D_QK**-0.5

        def run_sgl():
            return flash_mla_sparse_decode(
                q,
                packed_kv,
                indices,
                sm_scale=sm_scale,
                d_v=D_V,
                extra_kv=extra_kv,
                extra_indices=extra_indices,
                extra_topk_length=extra_topk_length,
                return_softmax_lse=False,
            )

        run_sgl()  # warmup / build
        avg_ms = bench(run_sgl)
        gbs = effective_bytes(batch, h_q, topk, extra_topk) / (avg_ms * 1e-3) / 1e9

        print(
            f"{h_q:6d} {batch:6d} {topk:6d} {extra_topk:6d} "
            f"{avg_ms:9.3f} {gbs:9.2f}"
        )


if __name__ == "__main__":
    main()
