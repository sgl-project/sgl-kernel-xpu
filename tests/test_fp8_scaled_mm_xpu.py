"""
Copyright (C) 2026 Intel Corporation, All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import fp8_scaled_mm


def torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
    """
    Reference FP32: matmul in FP32, then post-multiply by FP16 scales.

    The kernel computes D = diag(sa) * (A @ B^T) * diag(sb) + bias,
    where scales are applied in FP16 inside the mainloop. For the reference
    we post-multiply the FP32 matmul result by FP16 scales, which is
    mathematically equivalent but may differ due to FP16 rounding
    in the kernel's pre-scaling path.

    a: [M, K] (fp8)
    b: [N, K] (fp8)
    scale_a: [M] (fp32)
    scale_b: [N] (fp32)
    bias: [M] (out_dtype) or None
    """
    if b.shape[1] == a.shape[1]:
        b_for_linear = b
    else:
        b_for_linear = b.t()

    a32 = a.to(torch.float32).cpu()
    b32 = b_for_linear.to(torch.float32).cpu()

    # Match kernel: scales are cast to FP16
    scale_a_fp16 = scale_a.to(torch.float16).cpu()
    scale_b_fp16 = scale_b.to(torch.float16).cpu()

    o = F.linear(a32, b32)
    o = o * scale_a_fp16.view(-1, 1)
    o = o * scale_b_fp16.view(1, -1)

    if bias is not None:
        bias_cpu = bias.to(out_dtype).cpu()
        o = o + bias_cpu.to(torch.float32).view(-1, 1)

    return o.to(out_dtype)


def _get_tolerances(out_dtype):
    """Return (rtol, atol) based on output dtype.

    The kernel pre-scales FP8 inputs by FP16 scales before the FP32
    accumulation, while the reference post-multiplies. With FP8-range
    inputs (~224) and K up to 256, rounding difference can produce
    absolute errors up to ~50 on output values of ~100K which is
    within FP16 precision (~0.02% relative error).
    """
    if out_dtype == torch.float32:
        return 2e-2, 50.0
    elif out_dtype == torch.float16:
        return 3e-2, 50.0
    else:  # bfloat16
        return 4e-2, 50.0


def _test_accuracy_once(M, N, K, with_bias, out_dtype, device):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * fp8_max
    a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * fp8_max
    b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    # Compute scales from amax: scale = amax / fp8_max (always positive)
    amax_a = a_fp8.to(torch.float32).abs().amax(dim=1)  # [M]
    scale_a = (amax_a / fp8_max).to(torch.float32).to(device)

    amax_b = b_fp8.to(torch.float32).abs().amax(dim=1)  # [N]
    scale_b = (amax_b / fp8_max).to(torch.float32).to(device)

    # Bias in out_dtype (matches kernel ElementBias = ElementOutput)
    bias = torch.randn((M,), device=device, dtype=out_dtype) if with_bias else None

    ref = torch_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype, bias)
    out = fp8_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype, bias)
    out_cpu = out.to(out_dtype).cpu()

    if (~torch.isfinite(ref)).any().item():
        pytest.skip("Invalid reference run. Test skipped.")

    rtol, atol = _get_tolerances(out_dtype)
    torch.testing.assert_close(out_cpu, ref, rtol=rtol, atol=atol)
    print(f"M={M}, N={N}, K={K}, with_bias={with_bias}, out_dtype={out_dtype}: OK")


@pytest.mark.parametrize("M", [128, 256, 512])
@pytest.mark.parametrize("N", [128, 256, 512])
@pytest.mark.parametrize("K", [64, 128, 256])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_accuracy(M, N, K, with_bias, out_dtype):
    _test_accuracy_once(M, N, K, with_bias, out_dtype, "xpu")


if __name__ == "__main__":
    pytest.main([__file__])
