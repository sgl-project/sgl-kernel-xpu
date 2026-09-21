"""Quick smoke: single call of flash_mla_sparse prefill at MINIMAL params.

- Small config: s_q=16, h_q=8, topk=8, d_qk=512 (d_qk fixed by compiled kernel).
- Inputs built on CPU then .to(xpu) — avoids AubLoad-slow randn(device=xpu)
  over the (16384,1,512) kv tensor and the 16384-element randperms.
- One kernel call. Perf-print emitted from C++ dispatch on every invocation.
"""

import sys
import time

import torch

sys.path.insert(0, "benchmark")
from bench_flash_mla_sparse_fwd import D_V, build_inputs

from sgl_kernel import flash_mla_sparse_fwd

# Minimum feasible params. d_qk must be 512 (compiled). topk aligns to B_H=8.
S_Q = 16    # from 512
H_Q = 8     # from 16
TOPK = 8    # from 2048
D_QK = 512  # compiled kernel dim


def main() -> None:
    torch.manual_seed(42)
    cpu = torch.device("cpu")
    xpu = torch.device("xpu")

    print(
        f"[quick-prefill] MIN config: s_q={S_Q} h_q={H_Q} topk={TOPK} d_qk={D_QK}",
        flush=True,
    )

    t0 = time.time()
    q_cpu, kv_cpu, indices_cpu = build_inputs(S_Q, H_Q, TOPK, D_QK, device=cpu)
    print(f"[quick-prefill] CPU inputs built in {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    q = q_cpu.to(xpu)
    kv = kv_cpu.to(xpu)
    indices = indices_cpu.to(xpu)
    torch.xpu.synchronize()
    print(f"[quick-prefill] H2D transfer in {time.time()-t0:.2f}s", flush=True)

    sm_scale = D_QK ** -0.5
    print("[quick-prefill] calling flash_mla_sparse_fwd (1x)", flush=True)
    t0 = time.time()
    out, max_logits, lse = flash_mla_sparse_fwd(q, kv, indices, sm_scale=sm_scale, d_v=D_V)
    torch.xpu.synchronize()
    print(f"[quick-prefill] wall: {time.time()-t0:.2f}s", flush=True)
    print(f"[quick-prefill] out.shape={tuple(out.shape)} dtype={out.dtype}", flush=True)


if __name__ == "__main__":
    main()
