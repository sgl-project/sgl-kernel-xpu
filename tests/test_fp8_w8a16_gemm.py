"""
Copyright (C) 2026 Intel Corporation, All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
"""

import sys

import pytest
import torch
import utils
from sgl_kernel import fp8_w8a16_mm

pytestmark = pytest.mark.arch("xe20")

device = utils.get_device()

BLOCK = 128
# config = 10 * k_slices + tile id; tile ids 0/1/2/4 are 8/16/32/64-row tiles, 3 is 128x128.
EXPLICIT_CONFIGS = [10, 20, 40, 11, 21, 41, 12, 22, 42, 14, 24, 44, 13]


def make_problem(m, n, k, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = (torch.randn(n, k, device=device) * 64).to(torch.float8_e4m3fn)
    s = torch.rand(-(n // -BLOCK), k // BLOCK, device=device) * 1e-2 + 1e-3
    return x, w, s


def reference(x, w, s):
    n, k = w.shape
    scale = s.repeat_interleave(BLOCK, 0)[:n].repeat_interleave(BLOCK, 1)[:, :k]
    return x.float() @ (w.float() * scale).t()


def assert_close_to_reference(out, ref):
    # bf16 output rounding dominates; fp32 accumulation order differs from the reference.
    tol = 1e-2 * ref.abs().max().item()
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=tol)


@pytest.mark.parametrize("m", [1, 3, 8, 13, 16, 31, 32, 47, 64, 100, 128, 257])
@pytest.mark.parametrize(
    "n,k",
    [(2560, 5120), (5120, 2048), (5120, 6400), (192, 640), (320, 1024)],
)
def test_fp8_w8a16_mm_auto(m, n, k):
    x, w, s = make_problem(m, n, k)
    out = fp8_w8a16_mm(x, w, s)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert_close_to_reference(out, reference(x, w, s))


@pytest.mark.parametrize("config", EXPLICIT_CONFIGS)
@pytest.mark.parametrize("m", [1, 9, 40])
def test_fp8_w8a16_mm_every_config(config, m):
    n, k = 448, 4096
    x, w, s = make_problem(m, n, k, seed=config)
    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    torch.ops.sgl_kernel.fp8_w8a16_mm_xe20(out, x, w, s, config)
    assert_close_to_reference(out, reference(x, w, s))


def test_fp8_w8a16_mm_strided_input_and_output():
    m, n, k = 6, 256, 1024
    x_full, w, s = make_problem(m, n, k)
    x_big = torch.zeros(m, k + 128, device=device, dtype=torch.bfloat16)
    x_big[:, :k] = x_full
    out_big = torch.zeros(m, n + 64, device=device, dtype=torch.bfloat16)
    fp8_w8a16_mm(x_big[:, :k], w, s, out=out_big[:, :n])
    assert_close_to_reference(out_big[:, :n], reference(x_full, w, s))
    assert out_big[:, n:].abs().max().item() == 0


def test_fp8_w8a16_mm_rejects_misaligned_base():
    m, n, k = 6, 256, 1024
    x_full, w, s = make_problem(m, n, k)
    x_big = torch.zeros(m, k + 128, device=device, dtype=torch.bfloat16)
    # Row pitch stays 16B aligned; the base is 2 bytes past a 64B boundary.
    with pytest.raises(RuntimeError, match="64B aligned"):
        fp8_w8a16_mm(x_big[:, 1 : k + 1], w, s)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
