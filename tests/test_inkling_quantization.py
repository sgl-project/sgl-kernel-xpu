import importlib
import math
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PKG = _REPO_ROOT / "python" / "sgl_kernel"
_LOCAL_EXT = _REPO_ROOT / "build" / "src" / "inkling_quantization_ops.abi3.so"

if _LOCAL_PKG.is_dir() and _LOCAL_EXT.is_file() and "sgl_kernel" not in sys.modules:
    pkg = types.ModuleType("sgl_kernel")
    pkg.__path__ = [str(_LOCAL_PKG), str(_LOCAL_EXT.parent)]
    sys.modules["sgl_kernel"] = pkg
    torch.ops.load_library(str(_LOCAL_EXT))
else:
    import sgl_kernel  # noqa: F401

    try:
        importlib.import_module("sgl_kernel.inkling_quantization_ops")
    except ImportError:
        pass

pytestmark = pytest.mark.skipif(
    not (
        hasattr(torch, "xpu")
        and torch.xpu.is_available()
        and hasattr(torch.ops.sgl_kernel, "inkling_mxfp4_mapping")
        and hasattr(torch.ops.sgl_kernel, "inkling_nvfp4_layout")
    ),
    reason="Inkling quantization helpers are XPU-only",
)

K_E2M1_MAX = 6.0
K_E4M3_MAX = 448.0
FP32_MIN_NORMAL = 2 ** -126


def _quantize_e2m1_code(x: torch.Tensor) -> torch.Tensor:
    sign = (x < 0).to(torch.uint8) << 3
    ax = x.abs()
    mag = torch.zeros_like(sign)
    mag += (ax > 0.250001).to(torch.uint8)
    mag += (ax >= 0.749999).to(torch.uint8)
    mag += (ax > 1.250001).to(torch.uint8)
    mag += (ax >= 1.749999).to(torch.uint8)
    mag += (ax > 2.500001).to(torch.uint8)
    mag += (ax >= 3.499999).to(torch.uint8)
    mag += (ax > 5.000001).to(torch.uint8)
    return torch.where(mag == 0, torch.zeros_like(mag), sign | mag)


def _pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    paired = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    return ((paired[..., 0] & 0x0F) | ((paired[..., 1] & 0x0F) << 4)).to(torch.uint8)


def _mxfp4_reference(x: torch.Tensor, column_major_scales: bool, eps: float):
    rows, cols = x.shape
    groups = cols // 32
    blocks = x.float().reshape(rows, groups, 32)
    max_abs = torch.clamp(blocks.abs().amax(dim=-1), min=eps)
    shared_exp = torch.floor(torch.log2(max_abs)).to(torch.int32) - 2
    shared_exp = torch.clamp(shared_exp, -127, 127)
    scales = (shared_exp + 127).to(torch.uint8)
    scaled = blocks / torch.pow(2.0, shared_exp.float()).unsqueeze(-1)
    packed = _pack_e2m1(_quantize_e2m1_code(scaled).reshape(rows, cols))
    if column_major_scales:
        storage = torch.empty((groups, rows), dtype=torch.uint8)
        storage.copy_(scales.T.contiguous())
        scales = torch.as_strided(storage, (rows, groups), (1, rows))
    return packed, scales


def _round_nearest_even_int(x: float) -> int:
    base = math.floor(x)
    frac = x - base
    if frac > 0.5 or (frac == 0.5 and (base & 1)):
        base += 1
    return base


def _e4m3fn_encode(x: float) -> int:
    if not x > 0.0:
        return 0
    if x >= K_E4M3_MAX:
        return 0x7E
    if x < 0.015625:
        mantissa = _round_nearest_even_int(x / 0.001953125)
        if mantissa <= 0:
            return 0
        if mantissa >= 8:
            return 0x08
        return mantissa
    exponent = math.floor(math.log2(x))
    scale = 2.0**exponent
    mantissa = _round_nearest_even_int((x / scale - 1.0) * 8.0)
    exponent_field = exponent + 7
    if mantissa >= 8:
        mantissa = 0
        exponent_field += 1
    if exponent_field >= 15 and mantissa > 6:
        return 0x7E
    if exponent_field > 15:
        return 0x7E
    return (exponent_field << 3) | mantissa


def _e4m3fn_decode(code: int) -> float:
    exponent_field = (code >> 3) & 0x0F
    mantissa = code & 0x07
    if (code & 0x7F) == 0 or (code & 0x7F) == 0x7F:
        return 0.0
    if exponent_field == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa * 0.125) * 2.0 ** (exponent_field - 7)


def _nvfp4_swizzled_scale_index(row: int, group: int, rounded_groups: int) -> int:
    row_block = row // 128
    row_rem = row - row_block * 128
    e = row_rem // 32
    d = row_rem - e * 32
    c = group // 4
    f = group - c * 4
    groups4 = rounded_groups // 4
    return (((row_block * groups4 + c) * 32 + d) * 4 + e) * 4 + f


def _nvfp4_reference(x: torch.Tensor, global_scale: float):
    rows, cols = x.shape
    groups = cols // 16
    rounded_rows = ((rows + 127) // 128) * 128
    rounded_groups = ((groups + 3) // 4) * 4
    scale_factor = global_scale / K_E2M1_MAX
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8)
    scales = torch.zeros((rounded_rows, rounded_groups), dtype=torch.uint8)
    flat_scales = scales.view(-1)

    blocks = x.float().reshape(rows, groups, 16)
    for row in range(rows):
        for group in range(groups):
            block = blocks[row, group]
            scale_byte = _e4m3fn_encode(float(block.abs().max().item()) * scale_factor)
            decoded = _e4m3fn_decode(scale_byte)
            output_scale = 0.0 if decoded == 0.0 else global_scale / decoded
            flat_scales[_nvfp4_swizzled_scale_index(row, group, rounded_groups)] = scale_byte
            packed[row, group * 8 : (group + 1) * 8] = _pack_e2m1(
                _quantize_e2m1_code(block * output_scale)
            )
    return packed, scales


def _make_input(shape, dtype):
    n = shape[0] * shape[1]
    values = torch.arange(n, dtype=torch.float32).reshape(shape)
    values = ((values % 257) - 128) / 23.0
    edge = torch.tensor(
        [
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -3.5,
            -5.0,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    )
    values.reshape(-1)[: edge.numel()] = edge
    return values.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("column_major_scales", [False, True])
@pytest.mark.parametrize("shape", [(3, 96), (9, 384), (130, 192)])
def test_mxfp4_mapping_matches_reference(dtype, column_major_scales, shape):
    x_cpu = _make_input(shape, dtype)
    packed_ref, scales_ref = _mxfp4_reference(x_cpu, column_major_scales, 1.0e-10)
    packed, scales = torch.ops.sgl_kernel.inkling_mxfp4_mapping(
        x_cpu.to("xpu").contiguous(),
        column_major_scales,
        1.0e-10,
    )
    torch.xpu.synchronize()

    if column_major_scales:
        assert scales.stride() == (1, shape[0])
    torch.testing.assert_close(packed.cpu(), packed_ref, atol=0, rtol=0)
    torch.testing.assert_close(scales.cpu(), scales_ref, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(3, 48), (33, 384), (130, 1536)])
def test_nvfp4_layout_matches_reference(dtype, shape):
    x_cpu = _make_input(shape, dtype)
    amax = float(x_cpu.float().abs().max().item())
    global_scale = K_E4M3_MAX * K_E2M1_MAX / amax if amax > 0.0 else 1.0
    packed_ref, scales_ref = _nvfp4_reference(x_cpu, global_scale)
    packed, scales = torch.ops.sgl_kernel.inkling_nvfp4_layout(
        x_cpu.to("xpu").contiguous(),
        global_scale,
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(packed.cpu(), packed_ref, atol=0, rtol=0)
    torch.testing.assert_close(scales.cpu(), scales_ref, atol=0, rtol=0)


def test_quantization_rejects_bad_group_size():
    x = torch.zeros((2, 33), dtype=torch.bfloat16, device="xpu")
    with pytest.raises(RuntimeError, match="cols must be divisible"):
        torch.ops.sgl_kernel.inkling_mxfp4_mapping(x, False, 1.0e-10)
