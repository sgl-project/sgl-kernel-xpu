"""Use the actual native BF16 activation, including its intermediate rounding."""

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import bf16_dense_gelu_tanh_mul_lut_xe20, gelu_tanh_and_mul

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available()
    or not hasattr(torch.ops.sgl_kernel, "bf16_dense_gelu_tanh_mul_lut_xe20"),
    reason="requires the Xe2 BF16 fused dense GELU LUT operator",
)


@pytest.fixture(scope="module")
def gelu_table():
    gates = (
        torch.arange(65536, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .to("xpu")
    )
    return gelu_tanh_and_mul(
        torch.cat((gates, torch.ones_like(gates))).view(1, -1)
    ).flatten()


def assert_bits_or_nan(actual, expected):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    finite_or_inf = ~torch.isnan(expected)
    assert torch.equal(
        actual.view(torch.int16)[finite_or_inf],
        expected.view(torch.int16)[finite_or_inf],
    )


@pytest.mark.parametrize(
    "m,n,k",
    [(1, 64, 32), (17, 192, 96), (127, 256, 64), (128, 24576, 1536), (129, 192, 96)],
)
def test_exact_projection_and_activation(gelu_table, m, n, k):
    # Small integer operands make every FP32 accumulation exactly representable,
    # separating the epilogue's BF16 rounding contract from GEMM ordering.
    torch.manual_seed(42)
    a = torch.randint(-2, 3, (m, k), device="xpu").to(torch.bfloat16) / 4
    w = torch.randint(-2, 3, (n, k), device="xpu").to(torch.bfloat16) / 16
    expected = gelu_tanh_and_mul(F.linear(a, w))
    actual = bf16_dense_gelu_tanh_mul_lut_xe20(a, w, gelu_table)
    assert actual.shape == (m, n // 2)
    assert_bits_or_nan(actual, expected)


@pytest.mark.parametrize(
    "up", [0.0, -0.0, 0.5, -1.0, 3.0, float("inf"), -float("inf"), float("nan")]
)
def test_all_gate_patterns_through_native_projection(gelu_table, up, record_property):
    a = torch.zeros(1, 32, device="xpu", dtype=torch.bfloat16)
    a[0, 0] = 1
    w = torch.zeros(131072, 32, device="xpu", dtype=torch.bfloat16)
    w[:65536, 0] = (
        torch.arange(65536, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .to("xpu")
    )
    w[65536:, 0] = up
    # The FP32 one-hot projection preserves BF16 subnormals, matching DPAS.
    # Torch BF16 GEMM can flush them; that is a projection distinction rather
    # than an activation error. Keep all bit patterns, including signed zeros,
    # Inf and NaN, in this independent epilogue oracle.
    projection = F.linear(a.float(), w.float()).to(torch.bfloat16)
    expected = gelu_tanh_and_mul(projection)
    actual = bf16_dense_gelu_tanh_mul_lut_xe20(a, w, gelu_table)
    assert_bits_or_nan(actual, expected)

    native_projection = F.linear(a, w)
    both_nan = torch.isnan(native_projection) & torch.isnan(projection)
    mismatch = (
        native_projection.view(torch.int16) != projection.view(torch.int16)
    ) & ~both_nan
    allowed = torch.zeros_like(mismatch)
    patterns = torch.arange(65536, device="xpu", dtype=torch.int32)
    allowed[0, :65536] = ((patterns & 0x7F80) == 0) & ((patterns & 0x7F) != 0)
    assert not torch.any(mismatch & ~allowed).item()
    # Record the current implementation's behavior without requiring future
    # Torch versions to keep flushing the same patterns.
    record_property("torch_version", torch.__version__)
    record_property("bf16_projection_subnormal_difference_count", int(mismatch.sum()))


def test_graph_replay_reads_current_input_and_owned_table(gelu_table):
    a = torch.ones(128, 32, device="xpu", dtype=torch.bfloat16) / 4
    w = torch.ones(64, 32, device="xpu", dtype=torch.bfloat16) / 16
    table = gelu_table.clone()
    bf16_dense_gelu_tanh_mul_lut_xe20(a, w, table)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        output = bf16_dense_gelu_tanh_mul_lut_xe20(a, w, table)
    for scale in (0.5, 2.0):
        a.mul_(scale)
        output.fill_(float("nan"))
        graph.replay()
        torch.xpu.synchronize()
        assert_bits_or_nan(output, gelu_tanh_and_mul(F.linear(a, w)))
    table.zero_()
    graph.replay()
    torch.xpu.synchronize()
    assert torch.count_nonzero(output).item() == 0


@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "rank",
        "k",
        "n",
        "stride",
        "table_shape",
        "table_dtype",
        "table_stride",
        "table_device",
    ],
)
def test_invalid_inputs(gelu_table, case):
    a = torch.zeros(1, 32, device="xpu", dtype=torch.bfloat16)
    w = torch.zeros(64, 32, device="xpu", dtype=torch.bfloat16)
    table = gelu_table
    if case == "dtype":
        a = a.float()
    elif case == "rank":
        a = a.unsqueeze(0)
    elif case == "k":
        a, w = a[:, :16].contiguous(), w[:, :16].contiguous()
    elif case == "n":
        w = w[:63]
    elif case == "stride":
        a = torch.zeros(1, 64, device="xpu", dtype=torch.bfloat16)[:, ::2]
    elif case == "table_shape":
        table = table[:-1]
    elif case == "table_dtype":
        table = table.float()
    elif case == "table_stride":
        table = torch.zeros(131072, device="xpu", dtype=torch.bfloat16)[::2]
    elif case == "table_device":
        table = table.cpu()
    with pytest.raises(RuntimeError):
        bf16_dense_gelu_tanh_mul_lut_xe20(a, w, table)


@pytest.mark.parametrize("m,n", [(0, 64), (128, 0)])
def test_empty_output(gelu_table, m, n):
    a = torch.empty(m, 32, device="xpu", dtype=torch.bfloat16)
    w = torch.empty(n, 32, device="xpu", dtype=torch.bfloat16)
    assert bf16_dense_gelu_tanh_mul_lut_xe20(a, w, gelu_table).shape == (m, n // 2)


def test_torch_compile_fake_registration(gelu_table):
    a = torch.ones(128, 32, device="xpu", dtype=torch.bfloat16) / 4
    w = torch.ones(64, 32, device="xpu", dtype=torch.bfloat16) / 16
    compiled = torch.compile(
        bf16_dense_gelu_tanh_mul_lut_xe20, backend="eager", fullgraph=True
    )
    assert_bits_or_nan(compiled(a, w, gelu_table), gelu_tanh_and_mul(F.linear(a, w)))
