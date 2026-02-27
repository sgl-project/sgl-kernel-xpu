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
import sgl_kernel
import torch
import torch.nn.functional as F

# Fixed production dimensions
K = 7168
N_VALUES = [2048, 2112]


def dsv3_fused_a_gemm(
    hidden_states: torch.Tensor, weights: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    # weights is [K, N] column-major; output is [M, N]
    output = torch.empty(
        hidden_states.shape[0],
        weights.shape[1],
        device=hidden_states.device,
        dtype=output_dtype,
    )
    torch.ops.sgl_kernel.dsv3_fused_a_gemm(output, hidden_states, weights)
    return output


@pytest.mark.parametrize("num_tokens", [i + 1 for i in range(16)])
@pytest.mark.parametrize("N", N_VALUES)
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dsv3_fused_a_gemm_all_tokens(num_tokens, N, output_dtype):
    torch.manual_seed(0)

    # Random A in bf16
    mat_a = torch.randn((num_tokens, K), device="xpu", dtype=torch.bfloat16)

    # Random B built as column-major [K, N]: stride(0)=1, stride(1)=K
    mat_b = torch.randn((N, K), device="xpu", dtype=torch.bfloat16).t()
    assert mat_b.shape == (K, N)
    assert mat_b.stride() == (1, K)

    # Reference: weight is B^T (out_features=N, in_features=K)
    ref = F.linear(mat_a.float().cpu(), mat_b.float().cpu().T).to(output_dtype)

    output = dsv3_fused_a_gemm(mat_a, mat_b, output_dtype)
    output_cpu = output.cpu()

    rtol = 1e-2
    atol = 1e-3

    assert torch.allclose(
        output_cpu, ref, rtol=rtol, atol=atol
    ), f"Fused A GEMM output mismatch (dtype={output_dtype}, M={num_tokens}, N={N})! Max diff: {(output_cpu - ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
