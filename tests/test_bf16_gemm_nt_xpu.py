import importlib.util
import os
import time
from pathlib import Path

if os.environ.get("BATCHES", "8") != "8":
    raise SystemExit("BATCHES must be 8; rejected before GPU work")

import pytest
import sgl_kernel
import torch
from sgl_kernel import bf16_gemm_nt_xe35, pack_bf16_gemm_weight_xe35

reference_source = (
    Path(__file__).resolve().parents[1] / "benchmark/bench_bf16_gemm_nt.py"
)
reference_specification = importlib.util.spec_from_file_location(
    "bf16_gemm_reference", reference_source
)
reference_module = importlib.util.module_from_spec(reference_specification)
reference_specification.loader.exec_module(reference_module)
reference_module.torch = torch
cpu_reference_xe35 = reference_module.cpu_reference_xe35


def test_bf16_gemm_cpu_reference_chunks_xe35():
    activations = torch.arange(3 * 16, device="cpu", dtype=torch.float32).reshape(3, 16)
    weight = torch.arange(5 * 16, device="cpu", dtype=torch.float32).reshape(5, 16)
    actual = cpu_reference_xe35(activations, weight, chunk_bytes=2 * 16 * 4)
    torch.testing.assert_close(actual, torch.mm(activations, weight.T), rtol=0, atol=0)


@pytest.fixture(autouse=True)
def seed_inputs_xe35():
    torch.manual_seed(42)
    if torch.xpu.is_available() and sgl_kernel.is_xe3_arch():
        torch.xpu.synchronize()
        time.sleep(2)


@pytest.mark.parametrize("capability,expected", [((3, 5), True), ((2, 0), False)])
def test_bf16_gemm_nt_dispatch_capability_without_device_name_xe35(
    monkeypatch, capability, expected
):
    from sgl_kernel.bf16_gemm import _bf16_gemm_nt_use_cute_xe35

    monkeypatch.setattr(
        torch.ops.sgl_kernel.query_device, "default", lambda index: capability
    )

    def reject_device_name(index):
        raise AssertionError(
            "dispatch must not use a device-model performance whitelist"
        )

    monkeypatch.setattr(torch.xpu, "get_device_name", reject_device_name)
    assert (
        _bf16_gemm_nt_use_cute_xe35(torch.empty(8, 4096), 8, 8192, 4096, False)
        == expected
    )


@pytest.mark.parametrize(
    "n,k,last_kernel_row,next_kernel_row",
    [
        (4608, 8192, 32, None),
        (4096, 8192, 32, None),
        (8192, 4096, 32, None),
        (8448, 4096, 32, None),
        (8192, 8192, 16, None),
        (8448, 8192, 16, None),
        (18432, 8192, 96, None),
        (202240, 8192, 16, 33),
    ],
)
def test_bf16_gemm_nt_packed_dispatch_boundaries_xe35(
    monkeypatch, n, k, last_kernel_row, next_kernel_row
):
    from sgl_kernel.bf16_gemm import _bf16_gemm_nt_use_cute_xe35

    monkeypatch.setattr(
        torch.ops.sgl_kernel.query_device, "default", lambda index: (3, 5)
    )
    activations = torch.empty(1, 16)
    assert _bf16_gemm_nt_use_cute_xe35(activations, last_kernel_row, n, k, True)
    assert not _bf16_gemm_nt_use_cute_xe35(activations, last_kernel_row + 1, n, k, True)
    if next_kernel_row is not None:
        assert _bf16_gemm_nt_use_cute_xe35(activations, next_kernel_row, n, k, True)
    assert not _bf16_gemm_nt_use_cute_xe35(activations, 0, n, k, True)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 256, 32),  # Generic small GEMM.
        (7, 257, 96),  # Generic non-divisible output tail.
        (8, 202048, 8192),  # Muse B8 decode: M8, vocabulary N202048, K8192.
        (
            8,
            8192,
            4096,
        ),  # Muse B8 MoE decompression: M8, hidden N8192, compressed K4096.
        (8, 4096, 8192),  # Muse B8 MoE compression: M8, compressed N4096, hidden K8192.
        (17, 4096, 8192),
        (33, 4608, 8192),
        (64, 4608, 8192),
        (97, 202048, 8192),
        (193, 202048, 8192),
        (64, 202048, 8192),  # Synthetic GEMM row stress, not a serving batch.
        (65, 8192, 4096),  # Generic tile-boundary stress, not a serving batch.
        (96, 8192, 4096),
        (97, 8192, 4096),
        (128, 8192, 4096),
        (129, 8192, 4096),
    ],
)
def test_bf16_gemm_nt_xe35(shape):
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")
    m, n, k = shape
    a = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
    actual = bf16_gemm_nt_xe35(a, b)
    expected = cpu_reference_xe35(a, b)
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize(
    "m,expected", [(32, False), (33, True), (64, True), (65, False)]
)
def test_bf16_gemm_nt_native_tp4_dispatch_boundary_xe35(monkeypatch, m, expected):
    from sgl_kernel.bf16_gemm import _bf16_gemm_nt_use_cute_xe35

    monkeypatch.setattr(
        torch.ops.sgl_kernel.query_device, "default", lambda index: (3, 5)
    )
    assert (
        _bf16_gemm_nt_use_cute_xe35(torch.empty(1, 16), m, 4608, 8192, False)
        == expected
    )


@pytest.mark.parametrize(
    "rows,expected",
    [
        (64, False),
        (65, True),
        (79, True),
        (96, True),
        (97, True),
        (128, True),
        (129, False),
        (193, False),
    ],
)
def test_bf16_gemm_nt_native_decompress_dispatch_xe35(monkeypatch, rows, expected):
    from sgl_kernel.bf16_gemm import _bf16_gemm_nt_use_cute_xe35

    monkeypatch.setattr(
        torch.ops.sgl_kernel.query_device, "default", lambda index: (3, 5)
    )
    assert (
        _bf16_gemm_nt_use_cute_xe35(torch.empty(1, 16), rows, 8192, 4096, False)
        == expected
    )


@pytest.mark.parametrize("n,k", [(4608, 8192), (4096, 8192), (8192, 4096)])
@pytest.mark.parametrize("m", [1, 8, 9, 16, 17, 32, 33])
def test_bf16_gemm_nt_packed_tp4_projection_xe35(m, n, k):
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")
    activations = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
    packed = pack_bf16_gemm_weight_xe35(weight)
    output = bf16_gemm_nt_xe35(activations, packed, packed_weight=True, output_n=n)
    expected = cpu_reference_xe35(activations, weight)
    torch.testing.assert_close(output.cpu().float(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize(
    "rows,expected",
    [
        (64, False),
        (65, True),
        (128, True),
        (129, False),
        (192, False),
        (193, True),
        (256, True),
        (257, False),
    ],
)
def test_bf16_gemm_nt_native_lm_dispatch_boundaries_xe35(monkeypatch, rows, expected):
    from sgl_kernel.bf16_gemm import _bf16_gemm_nt_use_cute_xe35

    monkeypatch.setattr(
        torch.ops.sgl_kernel.query_device, "default", lambda index: (3, 5)
    )
    assert (
        _bf16_gemm_nt_use_cute_xe35(torch.empty(1, 16), rows, 202048, 8192, False)
        == expected
    )


def test_bf16_gemm_nt_packed_weight_xe35():
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")
    a = torch.randn((8, 96), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((257, 96), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight_xe35(b)
    actual = bf16_gemm_nt_xe35(a, packed_b, packed_weight=True)[:, : b.shape[0]]
    expected = cpu_reference_xe35(a, b)
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("m", [8, 9, 17, 32, 33, 96, 97, 256, 257])
def test_bf16_gemm_nt_packed_dense_projection_xe35(m):
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")
    # Muse hidden projection N8192/K8192; only M8 is B8 decode, others are tile stress.
    a = torch.randn((m, 8192), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((8192, 8192), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight_xe35(b)
    actual = bf16_gemm_nt_xe35(a, packed_b, packed_weight=True)
    expected = cpu_reference_xe35(a, b)
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("m", [1, 7, 8, 9, 16, 17, 25, 32, 96, 97])
def test_bf16_gemm_nt_packed_qkv_projection_xe35(m):
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")
    # Muse Q/K/V+output-gate N18432/K8192; M8 is B8 decode, M1/7/9 are boundaries.
    a = torch.randn((m, 8192), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((18432, 8192), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight_xe35(b)
    actual = bf16_gemm_nt_xe35(a, packed_b, packed_weight=True)
    expected = cpu_reference_xe35(a, b)
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("m", [1, 8, 9, 16, 17, 33])
@pytest.mark.parametrize("n,k", [(512, 96), (4096, 256)])
def test_bf16_gemm_nt_packed_tiles_xe35(m, n, k):
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")
    a = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
    packed_b = pack_bf16_gemm_weight_xe35(b)
    actual = torch.empty((m, n), dtype=torch.bfloat16, device="xpu")
    torch.ops.sgl_kernel.bf16_gemm_nt_xe35.default(actual, a, packed_b, 2)
    expected = cpu_reference_xe35(a, b)
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("m,n,k", [(8, 512, 96), (17, 4096, 256)])
def test_bf16_gemm_nt_native_kernel_xe35(m, n, k):
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")
    a = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
    actual = torch.empty((m, n), dtype=torch.bfloat16, device="xpu")
    torch.ops.sgl_kernel.bf16_gemm_nt_xe35.default(actual, a, b, 0)
    expected = cpu_reference_xe35(a, b)
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (8, 8192, 4112),
        (9, 8192, 4112),
        (8, 16640, 4080),
        (9, 16640, 4080),
        (8, 18432, 4080),
        (9, 18432, 4080),
    ],
)
def test_bf16_gemm_nt_native_dispatch_boundary_xe35(m, n, k):
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")
    a = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
    torch.testing.assert_close(
        bf16_gemm_nt_xe35(a, b).cpu().float(),
        cpu_reference_xe35(a, b),
        rtol=5e-3,
        atol=5e-3,
    )


def test_bf16_gemm_nt_packed_logical_width_xe35():
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")
    for m, n, k in [
        (8, 272, 96),
        (8, 16384, 8192),
        (8, 202032, 8192),
        (8, 202048, 8192),
        (8, 202064, 8192),
    ]:
        torch.xpu.synchronize()
        time.sleep(2)
        a = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
        b = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
        packed_b = pack_bf16_gemm_weight_xe35(b)
        expected_width = (n + 255) // 256 * 256
        if expected_width >= 16384 and expected_width & (expected_width - 1) == 0:
            expected_width += 256
        assert packed_b.shape == (k, expected_width)
        torch.testing.assert_close(packed_b[:, :n], b.T, rtol=0, atol=0)
        actual = bf16_gemm_nt_xe35(a, packed_b, packed_weight=True, output_n=n)
        assert actual.shape == (m, n)
        torch.testing.assert_close(
            actual.cpu().float(),
            cpu_reference_xe35(a, b),
            rtol=5e-3,
            atol=5e-3,
        )


@pytest.mark.parametrize("m,n,k", [(1, 16, 16), (9, 272, 96), (17, 272, 96)])
def test_bf16_gemm_nt_forced_packed_logical_tail_xe35(m, n, k):
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")
    activations = torch.randn((m, k), dtype=torch.bfloat16, device="xpu")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="xpu")
    packed = pack_bf16_gemm_weight_xe35(weight)
    output = torch.empty((m, n), dtype=torch.bfloat16, device="xpu")
    torch.ops.sgl_kernel.bf16_gemm_nt_xe35.default(output, activations, packed, 2)
    expected = cpu_reference_xe35(activations, weight)
    torch.testing.assert_close(output.cpu().float(), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("packed_weight", [False, True])
def test_bf16_gemm_nt_fp16_fallback_xe35(packed_weight):
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")
    activations = torch.randn((8, 96), dtype=torch.float16, device="xpu")
    weight = torch.randn((257, 96), dtype=torch.float16, device="xpu")
    operand = weight.T.contiguous() if packed_weight else weight
    output = bf16_gemm_nt_xe35(activations, operand, packed_weight=packed_weight)
    expected = cpu_reference_xe35(activations, weight)
    assert output.dtype == torch.float16
    torch.testing.assert_close(output.cpu().float(), expected, rtol=5e-3, atol=5e-3)


def test_bf16_gemm_nt_output_n_validation_xe35():
    a = torch.ones((1, 16), dtype=torch.bfloat16)
    weight = torch.ones((16, 16), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires a packed weight"):
        bf16_gemm_nt_xe35(a, weight, output_n=16)
    with pytest.raises(ValueError, match="within its width"):
        bf16_gemm_nt_xe35(a, weight, packed_weight=True, output_n=17)


def test_bf16_gemm_nt_prefill_projection_xe35():
    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        pytest.skip("Xe35 XPU is required")
    # Muse B8 ISL4096 prefill: M32768, hidden projection N8192, K8192.
    activations = torch.randn((8 * 4096, 8192), dtype=torch.bfloat16, device="xpu")
    weights = torch.randn((8192, 8192), dtype=torch.bfloat16, device="xpu")
    actual = bf16_gemm_nt_xe35(activations, weights)
    expected = cpu_reference_xe35(activations, weights)
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=5e-3, atol=5e-3)
