"""Time gptq_gemm on XPU. A machine without XPU prints a skip and is not a B70 result."""

import sys

import torch


def main() -> None:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        print("XPU unavailable; this run is not an Intel Arc Pro B70 result")
        return
    from sgl_kernel import gptq_gemm

    device = torch.device("xpu")
    k, n = 4096, 4096
    for m in (1, 16):
        act = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        qweight = torch.randint(0, 2**31 - 1, (k // 8, n), device=device, dtype=torch.int32)
        qzeros = torch.randint(0, 2**31 - 1, (k // 128, n // 8), device=device, dtype=torch.int32)
        scales = torch.rand(k // 128, n, device=device, dtype=torch.bfloat16) * 0.05 + 0.01
        g_idx = (torch.arange(k, device=device) // 128).to(torch.int32)
        for _ in range(3):
            gptq_gemm(act, qweight, qzeros, scales, g_idx, False, 4, 1)
        torch.xpu.synchronize()
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        iters = 10
        start.record()
        for _ in range(iters):
            gptq_gemm(act, qweight, qzeros, scales, g_idx, False, 4, 1)
        end.record()
        torch.xpu.synchronize()
        ms = start.elapsed_time(end) / iters
        print(f"gptq_gemm M={m} K={k} N={n} {ms:.3f} ms")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"benchmark did not run: {exc}", file=sys.stderr)
        raise
