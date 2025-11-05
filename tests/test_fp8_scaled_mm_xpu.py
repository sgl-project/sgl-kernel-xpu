"""
Copyright (C) 2025 Intel Corporation, All rights reserved.
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

"""
Test code for sgl_kernel.fp8_scaled_mm()

Run as:
python -m pytest -v -s test_fp8_scaled_mm_xpu.py
"""

import pytest
import torch
from sgl_kernel import fp8_scaled_mm


def is_fp8_dtype(dtype):
    """Check if dtype is FP8"""
    return dtype in [torch.float8_e4m3fn, torch.float8_e5m2]


def torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
    """
    Reference implementation of scaled matrix multiplication
    """

    # Convert scales to half precision
    scale_a_half = scale_a.to(torch.float16)
    scale_b_half = scale_b.to(torch.float16)
    # Convert back to float32 for computation
    scale_a_fp32 = scale_a_half.to(torch.float32)
    scale_b_fp32 = scale_b_half.to(torch.float32)

    o = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    o = o.to(torch.float32)
    temp1 = o * scale_a_fp32.view(-1, 1)
    temp2 = temp1 * scale_b_fp32.view(1, -1)

    # Add bias before quantization
    if bias is not None:
        temp2 = temp2 + bias.to(torch.float32).view(1, -1)

    # Quantize to FP8 if needed
    if is_fp8_dtype(out_dtype):
        # Get FP8 range
        fp8_info = torch.finfo(out_dtype)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        # Ensure scales are safe (positive and non-zero)
        scale_a_safe = scale_a_fp32.abs().clamp(min=1e-6)
        scale_b_safe = scale_b_fp32.abs().clamp(min=1e-6)

        # Per-element scaling: reverse the input scaling
        scale_matrix = scale_a_safe.view(-1, 1) * scale_b_safe.view(1, -1)
        temp_unscaled = temp2 / scale_matrix

        # Handle any NaN/Inf from division
        temp_unscaled = torch.where(
            torch.isfinite(temp_unscaled),
            temp_unscaled,
            torch.zeros_like(temp_unscaled),
        )

        # Compute global quantization scale from unscaled values
        amax = temp_unscaled.abs().max()
        if amax < 1e-10 or torch.isnan(amax) or torch.isinf(amax):
            amax = torch.tensor(1e-10, device=temp_unscaled.device)

        quant_scale = amax / fp8_max

        # Quantize
        final = (temp_unscaled / quant_scale).clamp(fp8_min, fp8_max).to(out_dtype)
    else:
        final = temp2.to(out_dtype)

    return final


def _test_accuracy_once(M, N, K, with_bias, out_dtype, fp8_dtype, device):
    # Get FP8 type info
    fp8_info = torch.finfo(fp8_dtype)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    # For E5M2, restrict range for numerical stability
    if fp8_dtype == torch.float8_e5m2:
        fp8_max, fp8_min = 8.0, -8.0

    # Generate random FP8 tensors
    a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(fp8_dtype)
    b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(fp8_dtype)

    # Generate non-zero positive scales
    scale_a = (
        torch.rand((M,), device=device, dtype=torch.float32) * 0.002 + 0.0001
    )  # Min 1e-4
    scale_b = (
        torch.rand((N,), device=device, dtype=torch.float32) * 0.002 + 0.0001
    )  # Min 1e-4

    # For E5M2, use smaller scales for stability
    if fp8_dtype == torch.float8_e5m2:
        scale_a = scale_a * 0.5
        scale_b = scale_b * 0.5

    # Generate bias if needed
    if with_bias:
        if is_fp8_dtype(out_dtype):
            bias = torch.randn((N,), device=device, dtype=torch.float16)
        else:
            bias = torch.randn((N,), device=device, dtype=out_dtype)
    else:
        bias = None

    # Transpose B for matrix multiplication
    b_fp8 = b_fp8.t()

    # Compute reference output
    o = torch_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype, bias)

    # Compute kernel output
    o1 = fp8_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype, bias)

    # For FP8 output, convert to FP32 before comparison
    if is_fp8_dtype(out_dtype):
        o_cmp = o.to(torch.float32)
        o1_cmp = o1.to(torch.float32)
    else:
        o_cmp = o
        o1_cmp = o1

    # Adjust tolerance based on input and output FP8 types
    if is_fp8_dtype(out_dtype):
        if out_dtype == torch.float8_e5m2:
            # E5M2 output
            # E5M2 representable values are very sparse
            # With 2 mantissa bits, quantization steps can be large
            rtol = 0.25
            atol = 256.0
        elif fp8_dtype == torch.float8_e5m2:
            # E5M2 input, E4M3 output
            rtol = 0.04
            atol = 32.0
        else:
            # E4M3 input and output
            rtol = 0.04
            atol = 32.0
    elif fp8_dtype == torch.float8_e5m2:
        # E5M2 input, FP16/BF16 output
        rtol = 0.03
        atol = 1.5
    else:
        # E4M3 input, FP16/BF16 output
        rtol = 0.02
        atol = 1.0

    torch.testing.assert_close(o_cmp, o1_cmp, rtol=rtol, atol=atol)

    fp8_in_name = "e4m3" if fp8_dtype == torch.float8_e4m3fn else "e5m2"
    if is_fp8_dtype(out_dtype):
        fp8_out_name = "e4m3" if out_dtype == torch.float8_e4m3fn else "e5m2"
        out_name = f"fp8_{fp8_out_name}"
    else:
        out_name = str(out_dtype).split(".")[-1]

    print(
        f"M={M}, N={N}, K={K}, in_fp8={fp8_in_name}, bias={with_bias}, out={out_name}: OK"
    )


# Full test suite
@pytest.mark.parametrize("M", [1, 128, 512, 1024, 4096])
@pytest.mark.parametrize("N", [16, 128, 512, 1024, 4096])
@pytest.mark.parametrize("K", [512, 1024, 4096, 8192, 16384])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize(
    "out_dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]
)
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_accuracy(M, N, K, with_bias, out_dtype, fp8_dtype):
    _test_accuracy_once(M, N, K, with_bias, out_dtype, fp8_dtype, "xpu")


if __name__ == "__main__":
    pytest.main([__file__])
