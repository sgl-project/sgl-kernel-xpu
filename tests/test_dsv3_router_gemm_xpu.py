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
from sgl_kernel.gemm import dsv3_router_gemm


@pytest.mark.parametrize("num_tokens", [1, 4, 16])
@pytest.mark.parametrize("num_experts", [256, 384])
@pytest.mark.parametrize("hidden_dim", [7168])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_dsv3_router_gemm(num_tokens, num_experts, hidden_dim, out_dtype):
    """Test with various output dtypes."""

    mat_a = torch.randn(
        (num_tokens, hidden_dim), dtype=torch.bfloat16, device="xpu"
    ).contiguous()
    mat_b = torch.randn(
        (num_experts, hidden_dim), dtype=torch.bfloat16, device="xpu"
    ).contiguous()

    # Reference computation on CPU: use the requested dtype when supported to reduce conversion error.
    ref = F.linear(mat_a.cpu().to(out_dtype), mat_b.cpu().to(out_dtype))

    # Test output
    output = dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype)
    output_cpu = output.cpu()

    # Tolerances
    rtol = 1e-3
    atol = 1e-3
    if out_dtype != torch.float32:
        rtol = 1e-2
        atol = 1e-2

    assert torch.allclose(
        output_cpu, ref, rtol=rtol, atol=atol
    ), f"Router GEMM output mismatch! Max diff: {(output_cpu - ref).abs().max()}"


if __name__ == "__main__":
    print("=" * 60)
    print("Running DSV3 Router GEMM XPU Tests")
    print("=" * 60)
    print("Running Full Test Suite")
    pytest.main([__file__, "-v", "--tb=short"])
