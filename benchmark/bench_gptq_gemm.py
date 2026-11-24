"""Time gptq_gemm on XPU against the existing XPU GPTQ path.

The existing path is the dense fallback: weights are dequantized once at
load and every call is a torch.matmul. The packed path repacks once at
load (gptq_repack_weights) and dequantizes on the fly per call. Load-time
work is excluded from every timing. A machine without XPU prints a skip
and is not a B70 result.
"""

import sys

import torch


def dequantize(qweight, qzeros, scales, g_idx, zero_format):
    k = qweight.shape[0] * 8
    n = qweight.shape[1]
    q = torch.empty(k, n, device=qweight.device, dtype=torch.float32)
    for i in range(8):
        q[i::8] = ((qweight >> (i * 4)) & 0x0F).float()
    stored = torch.empty(
        qzeros.shape[0], n, device=qweight.device, dtype=torch.float32
    )
    for i in range(8):
        stored[:, i::8] = ((qzeros >> (i * 4)) & 0x0F).float()
    eff = stored + 1 if zero_format == 1 else stored
    return (q - eff[g_idx]) * scales.float()[g_idx]


def time_op(fn, iters=20):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        print("XPU unavailable; this run is not an Intel Arc Pro B70 result")
        return
    from sgl_kernel import gptq_gemm, gptq_repack_weights

    device = torch.device("xpu")
    k, n = 4096, 4096
    for m in (1, 4, 16):
        act = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        qweight = torch.randint(
            0, 2**31 - 1, (k // 8, n), device=device, dtype=torch.int32
        )
        qzeros = torch.randint(
            0, 2**31 - 1, (k // 128, n // 8), device=device, dtype=torch.int32
        )
        scales = torch.rand(k // 128, n, device=device, dtype=torch.bfloat16) * 0.05 + 0.01
        g_idx = (torch.arange(k, device=device) // 128).to(torch.int32)

        packed = gptq_repack_weights(qweight, g_idx)
        dense = dequantize(qweight, qzeros, scales, g_idx, 1).to(torch.bfloat16)

        ms_packed = time_op(
            lambda: gptq_gemm(act, packed, qzeros, scales, g_idx, False, 4, 1, True)
        )
        ms_unpacked = time_op(
            lambda: gptq_gemm(act, qweight, qzeros, scales, g_idx, False, 4, 1, False)
        )
        ms_dense = time_op(lambda: act @ dense)
        print(
            f"M={m} K={k} N={n} packed {ms_packed:.3f} ms | "
            f"unpacked {ms_unpacked:.3f} ms | dense fallback {ms_dense:.3f} ms"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"benchmark did not run: {exc}", file=sys.stderr)
        raise
