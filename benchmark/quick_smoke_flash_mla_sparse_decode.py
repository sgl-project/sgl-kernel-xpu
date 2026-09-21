"""Quick smoke: single call of flash_mla_sparse decode at MINIMAL params.

- Everything as small as possible: batch=1, topk=8, num_pages=8 (from 512).
- Inputs built on CPU then .to(xpu) — avoids the AubLoad-slow randn(device=xpu).
- One kernel call. The perf-print line comes from inside the C++ dispatch, so
  a single invocation is enough to see time / bandwidth / TFLOPS.
"""

import sys
import time

import torch

sys.path.insert(0, "benchmark")
from bench_flash_mla_with_kvcache import (
    H_PER_RANK,
    PAGE_SIZE,
    build_inputs,
)

from sgl_kernel import flash_mla_with_kvcache

# Minimum feasible params. topk aligns to compiled B_H=8. num_pages must be
# >= 8 for make_fp8_kv_cache's fill loop and cover topk indices.
BATCH = 1
TOPK = 8
EXTRA_TOPK = 0
H = H_PER_RANK      # 16 — kernel needs this specific head count
NUM_PAGES = 8       # from 512
PAGE = PAGE_SIZE    # 256 — matched to compiled kernel


def main() -> None:
    torch.manual_seed(42)
    cpu = torch.device("cpu")
    xpu = torch.device("xpu")

    print(
        f"[quick-decode] MIN config: batch={BATCH} topk={TOPK} H={H} "
        f"num_pages={NUM_PAGES} page_size={PAGE}",
        flush=True,
    )

    t0 = time.time()
    cpu_inputs = build_inputs(BATCH, TOPK, EXTRA_TOPK, NUM_PAGES, PAGE, H, cpu)
    print(f"[quick-decode] CPU inputs built in {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    inputs = {}
    for k, v in cpu_inputs.items():
        inputs[k] = v.to(xpu) if isinstance(v, torch.Tensor) else v
    torch.xpu.synchronize()
    print(f"[quick-decode] H2D transfer in {time.time()-t0:.2f}s", flush=True)

    print("[quick-decode] calling flash_mla_with_kvcache (1x)", flush=True)
    t0 = time.time()
    out = flash_mla_with_kvcache(
        q=inputs["q"],
        k_cache=inputs["k_cache"],
        block_table=None,
        cache_seqlens=None,
        head_dim_v=inputs["head_dim_v"],
        tile_scheduler_metadata=None,
        num_splits=None,
        softmax_scale=inputs["softmax_scale"],
        causal=False,
        is_fp8_kvcache=True,
        indices=inputs["indices"],
        attn_sink=inputs["attn_sink"],
        extra_k_cache=inputs.get("extra_k_cache"),
        extra_indices_in_kvcache=inputs.get("extra_indices"),
        topk_length=inputs["topk_length"],
        extra_topk_length=inputs.get("extra_topk_length"),
    )
    torch.xpu.synchronize()
    print(f"[quick-decode] wall: {time.time()-t0:.2f}s", flush=True)
    print(f"[quick-decode] out.shape={tuple(out.shape) if hasattr(out,'shape') else 'n/a'}", flush=True)


if __name__ == "__main__":
    main()
