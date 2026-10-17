import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import bf16_dense_rank_argmax_xe20

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available()
    or not hasattr(torch.ops.sgl_kernel, "bf16_dense_rank_argmax_xe20"),
    reason="requires the Xe2 BF16 comparison-rank GEMM operator",
)


def comparison_table(softcap=30):
    # The operator accepts arbitrary caller-defined comparison classes. This
    # oracle exercises softcap saturation, signed-zero equality and first NaN.
    raw = torch.arange(65536, device="xpu", dtype=torch.int32)
    values = raw.to(torch.int16).view(torch.bfloat16).float()
    if softcap:
        values = (values / softcap).tanh() * softcap
    bits = values.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    ranks = torch.where(bits & 0x80000000 != 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000)
    ranks = torch.where(values == 0, 0x80000000, ranks)
    ranks = torch.where(values.isnan(), 0xFFFFFFFF, ranks)
    _, ordinal = torch.unique(ranks, sorted=True, return_inverse=True)
    return ordinal.to(torch.uint16)


def reference_ids(input, weight, table):
    logits = F.linear(input, weight)
    bits = logits.view(torch.int16).to(torch.int64) & 65535
    return table.to(torch.int64)[bits].argmax(dim=-1)


@pytest.mark.parametrize("m", [4, 32, 127, 128, 129, 256])
def test_partial_rows_and_global_ties(m):
    table = comparison_table()
    a = torch.zeros(m, 32, device="xpu", dtype=torch.bfloat16)
    a[:, 0] = 1
    weight = torch.zeros(4096, 32, device="xpu", dtype=torch.bfloat16)
    for kind in ("zero", "saturation", "nonfinite", "cross_tile_tie"):
        weight.zero_()
        if kind == "saturation":
            weight[127, 0] = 1000
            weight[128, 0] = 1010
            weight[-1, 0] = 1020
        elif kind == "nonfinite":
            weight[127, 0] = float("inf")
            weight[128, 0] = weight[-1, 0] = float("nan")
        elif kind == "cross_tile_tie":
            # Comparing packed local keys across tiles would incorrectly pick
            # column128 over column127. Rank then first global ID must win.
            weight[127, 0] = weight[128, 0] = 4
        actual = bf16_dense_rank_argmax_xe20(a, weight, table)
        torch.testing.assert_close(
            actual, reference_ids(a, weight, table), atol=0, rtol=0
        )


def test_all_bf16_patterns_and_high_unsigned_classes():
    table = comparison_table()
    assert table.to(torch.int64).max().item() > 32767
    a = torch.zeros(3, 32, device="xpu", dtype=torch.bfloat16)
    a[:, 0] = 1
    weight = torch.zeros(65536, 32, device="xpu", dtype=torch.bfloat16)
    weight[:, 0] = (
        torch.arange(65536, device="xpu", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    torch.testing.assert_close(
        bf16_dense_rank_argmax_xe20(a, weight, table),
        reference_ids(a, weight, table),
        atol=0,
        rtol=0,
    )


def test_random_projection():
    torch.manual_seed(42)
    a = torch.randn(128, 96, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(4096, 96, device="xpu", dtype=torch.bfloat16) * 0.02
    table = comparison_table()
    torch.testing.assert_close(
        bf16_dense_rank_argmax_xe20(a, weight, table),
        reference_ids(a, weight, table),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("m", [4, 128])
@pytest.mark.parametrize("negative", [False, True])
def test_subnormal_projection_uses_fp32_reference(m, negative):
    # Some BF16 GEMM backends flush nonzero subnormal projections. The native
    # DPAS contract preserves them; compare the FP32 projection rounded to BF16
    # rather than assuming arbitrary GEMM implementations are bitwise equal.
    table = comparison_table(softcap=None)
    a = torch.zeros(m, 32, device="xpu", dtype=torch.bfloat16)
    a[:, 0] = 1
    weight = torch.zeros(256, 32, device="xpu", dtype=torch.bfloat16)
    tiny = 2.0**-133
    if negative:
        weight[:, 0] = -tiny
        weight[128, 0] = 0
    else:
        weight[128, 0] = tiny
    projected = F.linear(a.float(), weight.float()).to(torch.bfloat16)
    bits = projected.view(torch.int16).to(torch.int64) & 65535
    expected = table.to(torch.int64)[bits].argmax(dim=-1)
    assert expected.eq(128).all().item()
    torch.testing.assert_close(
        bf16_dense_rank_argmax_xe20(a, weight, table), expected, atol=0, rtol=0
    )


def test_graph_replay_and_compile_use_current_input_and_table():
    a = torch.zeros(32, 32, device="xpu", dtype=torch.bfloat16)
    a[:, 0] = 1
    weight = torch.zeros(4096, 32, device="xpu", dtype=torch.bfloat16)
    weight[127, 0], weight[128, 0] = 4, -4
    table = comparison_table()
    bf16_dense_rank_argmax_xe20(a, weight, table)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        output = bf16_dense_rank_argmax_xe20(a, weight, table)
    for _ in range(2):
        a.neg_()
        output.fill_(-123)
        graph.replay()
        torch.xpu.synchronize()
        torch.testing.assert_close(
            output, reference_ids(a, weight, table), atol=0, rtol=0
        )
    # Table contents are live data too, not a compile/capture-time policy.
    table.fill_(0)
    output.fill_(-123)
    graph.replay()
    torch.xpu.synchronize()
    assert output.eq(0).all().item()
    compiled = torch.compile(
        bf16_dense_rank_argmax_xe20, backend="eager", fullgraph=True
    )
    torch.testing.assert_close(compiled(a, weight, table), output, atol=0, rtol=0)


@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "rank",
        "k_mismatch",
        "k_alignment",
        "n_alignment",
        "strides",
        "table_dtype",
        "table_shape",
        "table_strides",
        "table_device",
    ],
)
def test_invalid_inputs(case):
    a = torch.empty(4, 32, device="xpu", dtype=torch.bfloat16)
    weight = torch.empty(128, 32, device="xpu", dtype=torch.bfloat16)
    table = torch.empty(65536, device="xpu", dtype=torch.uint16)
    if case == "dtype":
        a = a.float()
    elif case == "rank":
        a = a.unsqueeze(0)
    elif case == "k_mismatch":
        weight = weight[:, :16].contiguous()
    elif case == "k_alignment":
        a, weight = a[:, :16].contiguous(), weight[:, :16].contiguous()
    elif case == "n_alignment":
        weight = weight[:127]
    elif case == "strides":
        a = torch.empty(32, 4, device="xpu", dtype=torch.bfloat16).t()
    elif case == "table_dtype":
        table = table.to(torch.int32)
    elif case == "table_shape":
        table = table[:65535]
    elif case == "table_strides":
        table = torch.empty(131072, device="xpu", dtype=torch.uint16)[::2]
    else:
        table = table.cpu()
    with pytest.raises(RuntimeError):
        bf16_dense_rank_argmax_xe20(a, weight, table)


def test_empty_rows_and_fake_contract():
    table = torch.empty(65536, device="xpu", dtype=torch.uint16)
    a = torch.empty(0, 32, device="xpu", dtype=torch.bfloat16)
    weight = torch.empty(128, 32, device="xpu", dtype=torch.bfloat16)
    assert bf16_dense_rank_argmax_xe20(a, weight, table).shape == (0,)
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        a = torch.empty(129, 1536, device="xpu", dtype=torch.bfloat16)
        weight = torch.empty(262144, 1536, device="xpu", dtype=torch.bfloat16)
        table = torch.empty(65536, device="xpu", dtype=torch.uint16)
        output = bf16_dense_rank_argmax_xe20(a, weight, table)
        assert output.shape == (129,) and output.dtype == torch.int64
        with pytest.raises(RuntimeError):
            bf16_dense_rank_argmax_xe20(a, weight[:127], table)


@pytest.mark.parametrize(
    "m,n,k,valid",
    [
        (0xFFFFFF, 128, 32, True),
        (0x1000000, 128, 32, False),
        (1, 0xFFFF80, 32, True),
        (1, 0x1000000, 32, False),
        (1, 128, 8388576, True),
        (1, 128, 8388608, False),
    ],
)
def test_fake_block_2d_descriptor_boundaries_without_allocation(m, n, k, valid):
    # Width/pitch bytes and heights have 24-bit hardware descriptor limits.
    # FakeTensor covers the exact neighboring shapes without GiB allocations.
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        a = torch.empty(m, k, device="xpu", dtype=torch.bfloat16)
        weight = torch.empty(n, k, device="xpu", dtype=torch.bfloat16)
        table = torch.empty(65536, device="xpu", dtype=torch.uint16)
        if valid:
            assert bf16_dense_rank_argmax_xe20(a, weight, table).shape == (m,)
        else:
            with pytest.raises(RuntimeError):
                bf16_dense_rank_argmax_xe20(a, weight, table)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
