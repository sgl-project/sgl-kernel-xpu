"""Benchmark: DeepSeek V4 sparse MLA fp8 decode (sgl_kernel).

Times sgl_kernel.flash_mla_sparse_decode across DeepSeek-V4 decode shapes and
reports latency + effective memory bandwidth. Also cross-checks correctness
against a float reference.

Usage:
  python benchmark/bench_flash_mla_sparse_decode.py
"""

import torch
from sgl_kernel import flash_mla_sparse_decode

# ── DeepSeek V4 sparse fp8 KV layout constants ──
NOPE_DIM = 448
ROPE_DIM = 64
D_QK = NOPE_DIM + ROPE_DIM  # 512
D_V = 512
DATA_BYTES_PER_TOKEN = NOPE_DIM + ROPE_DIM * 2  # 576
SCALE_BYTES_PER_TOKEN = 8
HEAD_BYTES = DATA_BYTES_PER_TOKEN + SCALE_BYTES_PER_TOKEN  # 584

BLOCK_SIZE = 64
NUM_BLOCKS = 256  # 16384 cached tokens

# (label, batch, h_q, topk)
CONFIGS = [
    ("TP=8  (16 heads) topk=2048", 64, 16, 2048),
    ("TP=4  (32 heads) topk=2048", 64, 32, 2048),
    ("TP=1 (128 heads) topk=2048", 64, 128, 2048),
    ("TP=8  (16 heads) topk=512", 128, 16, 512),
    ("TP=1 (128 heads) topk=512", 128, 128, 512),
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


def build_inputs(batch, h_q, topk, device="xpu", seed=0):
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
    packed_kv, dequant_kv = _pack_sparse_fp8_kv_deepseek_v4(logical_kv)
    s_kv = NUM_BLOCKS * BLOCK_SIZE
    indices = torch.stack(
        [
            torch.stack([torch.randperm(s_kv, device=device)[:topk].to(torch.int32)])
            for _ in range(b)
        ]
    )  # [b, 1, topk]
    return q, packed_kv, dequant_kv, indices


def effective_bytes(batch, h_q, topk):
    # q read (bf16) + gathered fp8 kv read (packed head bytes) + out write (bf16)
    q_bytes = batch * h_q * D_QK * 2
    kv_bytes = batch * topk * HEAD_BYTES
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


def reference_out(q, dequant_kv, indices, sm_scale):
    b, s_q, h_q, d_qk = q.shape
    kv_flat = dequant_kv.view(-1, D_QK)
    s_kv = kv_flat.shape[0]
    topk = indices.shape[2]
    idx = indices.clone()
    valid = (idx >= 0) & (idx < s_kv)
    gather = idx.masked_fill(~valid, 0)
    gk = kv_flat.index_select(0, gather.reshape(-1)).view(b, s_q, topk, D_QK).float()
    P = torch.einsum("bshd,bskd->bshk", q.float(), gk) * sm_scale
    P = P.masked_fill(~valid.unsqueeze(2), float("-inf"))
    lse = torch.logsumexp(P, dim=-1, keepdim=True)
    s = torch.exp(P - lse)
    return torch.einsum("bshk,bskd->bshd", s, gk[..., :D_V]).to(torch.bfloat16)


def main():
    if not torch.xpu.is_available():
        print("XPU not available")
        return

    hdr = (
        f"{'config':30s} {'MB/call':>9s} {'sgl ms':>9s} {'sgl GB/s':>9s} "
        f"{'max_abs':>9s} {'ok':>4s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for label, batch, h_q, topk in CONFIGS:
        q, packed_kv, dequant_kv, indices = build_inputs(batch, h_q, topk)
        sm_scale = D_QK**-0.5
        mb = effective_bytes(batch, h_q, topk) / 1e6

        def run_sgl():
            return flash_mla_sparse_decode(
                q,
                packed_kv,
                indices,
                sm_scale=sm_scale,
                d_v=D_V,
                return_softmax_lse=False,
            )

        out_sgl = run_sgl()
        ref = reference_out(q, dequant_kv, indices, sm_scale)
        max_abs = (out_sgl.float() - ref.float()).abs().max().item()
        ok = torch.allclose(out_sgl.float(), ref.float(), atol=1e-2, rtol=1e-2)

        sgl_ms = bench(run_sgl)
        sgl_gbs = effective_bytes(batch, h_q, topk) / (sgl_ms * 1e-3) / 1e9

        print(
            f"{label:30s} {mb:9.1f} {sgl_ms:9.3f} {sgl_gbs:9.2f} "
            f"{max_abs:9.2e} {str(ok):>4s}"
        )


if __name__ == "__main__":
    main()
