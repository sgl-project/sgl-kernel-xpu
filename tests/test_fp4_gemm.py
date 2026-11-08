"""MXFP4 dense scaled GEMM (`cutlass_scaled_fp4_mm`) on Intel XPU.

The op implements the MX flavour of FP4 — E2M1 data with UE8M0 scales over
blocks of 32 — because NVFP4 (block 16, E4M3 scales) has no Xe equivalent.
"""

import sys

import pytest
import torch
import utils
from mxfp4_utils import MXFP4_BLOCK_SIZE, dequantize_mxfp4_2d, quantize_mxfp4_2d
from sgl_kernel import cutlass_scaled_fp4_mm, sgl_per_token_group_quant_fp4

device = utils.get_device()

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
# m, n, k
SHAPES = [
    (128, 128, 128),
    (128, 256, 192),
    (256, 128, 128),
    (256, 256, 256),
]
# Shapes the tile dispatch and the scale padding have to survive: M not a
# multiple of the 4-element scale alignment, K not a multiple of the 64-wide
# K tile, and a decode-sized M.
PAD_SHAPES = [(150, 128, 128), (128, 128, 96), (1, 256, 128), (33, 128, 160)]
SHAPES.extend(PAD_SHAPES)


def is_cri_device() -> bool:
    try:
        from sgl_kernel import is_xe3_arch

        return is_xe3_arch()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not (device.type == "xpu" and is_cri_device()),
    reason="cutlass_scaled_fp4_mm requires an Xe3 (CRI) XPU",
)


def dtype_id(dtype: torch.dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def quantize(x: torch.Tensor):
    """Quantize to MXFP4 on CPU with the reference implementation."""
    packed, scales = quantize_mxfp4_2d(x.float().cpu(), MXFP4_BLOCK_SIZE)
    return packed.to(x.device), scales.to(x.device)


def reference_mm(a_packed, a_scales, b_packed, b_scales, out_dtype, alpha):
    a = dequantize_mxfp4_2d(
        a_packed.cpu(), a_scales.cpu(), torch.float32, MXFP4_BLOCK_SIZE
    )
    b = dequantize_mxfp4_2d(
        b_packed.cpu(), b_scales.cpu(), torch.float32, MXFP4_BLOCK_SIZE
    )
    return (alpha * torch.matmul(a, b.t())).to(out_dtype).to(a_packed.device)


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@torch.inference_mode()
def test_mxfp4_gemm(dtype: torch.dtype, shape: tuple[int, int, int]) -> None:
    torch.manual_seed(42)
    m, n, k = shape

    a_packed, a_scales = quantize(torch.randn((m, k), dtype=dtype, device=device))
    b_packed, b_scales = quantize(torch.randn((n, k), dtype=dtype, device=device))

    out = cutlass_scaled_fp4_mm(
        a_packed, b_packed, a_scales, b_scales, None, out_dtype=dtype
    )
    expected = reference_mm(a_packed, a_scales, b_packed, b_scales, dtype, 1.0)

    # Both sides see identical dequantized operands, so the only error is the
    # accumulation order; the magnitudes grow with K.
    torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=dtype_id)
@torch.inference_mode()
def test_mxfp4_gemm_alpha(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    m, n, k = 256, 256, 128

    a_packed, a_scales = quantize(torch.randn((m, k), dtype=dtype, device=device))
    b_packed, b_scales = quantize(torch.randn((n, k), dtype=dtype, device=device))

    alpha_value = 0.375
    alpha = torch.tensor([alpha_value], dtype=torch.float32, device=device)
    out = cutlass_scaled_fp4_mm(
        a_packed, b_packed, a_scales, b_scales, alpha, out_dtype=dtype
    )
    expected = reference_mm(a_packed, a_scales, b_packed, b_scales, dtype, alpha_value)

    torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=dtype_id)
@torch.inference_mode()
def test_mxfp4_gemm_with_xpu_quant(dtype: torch.dtype) -> None:
    """The XPU quantizer's outputs feed the GEMM as-is, with no relayout."""
    torch.manual_seed(42)
    m, n, k = 150, 128, 128

    a_packed, a_scales = sgl_per_token_group_quant_fp4(
        torch.randn((m, k), dtype=dtype, device=device)
    )
    b_packed, b_scales = sgl_per_token_group_quant_fp4(
        torch.randn((n, k), dtype=dtype, device=device)
    )

    out = cutlass_scaled_fp4_mm(
        a_packed, b_packed, a_scales, b_scales, None, out_dtype=dtype
    )
    expected = reference_mm(a_packed, a_scales, b_packed, b_scales, dtype, 1.0)

    torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
