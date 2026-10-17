import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import bf16_dense_gemm_xe20

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available()
    or not hasattr(torch.ops.sgl_kernel, "bf16_dense_gemm_xe20"),
    reason="requires the Xe2 BF16 dense GEMM operator",
)


def assert_reference_quality(actual, a, weight):
    reference = F.linear(a.float(), weight.float())
    native = F.linear(a, weight)
    torch.testing.assert_close(actual.float(), reference, atol=0.01, rtol=0.01)
    rms = lambda value: (value.float() - reference).square().mean().sqrt()
    assert rms(actual).item() <= rms(native).item() * 1.1 + 1e-6


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 64, 32),
        (3, 192, 64),
        (127, 192, 96),
        (129, 1536, 6144),
        (128, 1536, 12288),
        (256, 1536, 6144),
        (256, 1536, 12288),
        (512, 1536, 6144),
        (257, 1536, 12288),
    ],
)
def test_reference_and_partial_tiles(m, n, k):
    torch.manual_seed(42)
    a = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="xpu", dtype=torch.bfloat16) * 0.02
    actual = bf16_dense_gemm_xe20(a, weight)
    assert actual.shape == (m, n)
    assert actual.dtype == torch.bfloat16
    assert_reference_quality(actual, a, weight)


@pytest.mark.parametrize("m,k", [(128, 12288), (256, 6144)])
def test_graph_replay_uses_current_inputs(m, k):
    a = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(1536, k, device="xpu", dtype=torch.bfloat16) * 0.02
    bf16_dense_gemm_xe20(a, weight)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        output = bf16_dense_gemm_xe20(a, weight)
    for scale in (1.0, 0.5):
        a.mul_(scale)
        output.fill_(float("nan"))
        graph.replay()
        torch.xpu.synchronize()
        assert torch.isfinite(output).all().item()
        assert_reference_quality(output, a, weight)


@pytest.mark.parametrize("m,n", [(0, 64), (128, 0)])
def test_empty_output(m, n):
    a = torch.empty(m, 32, device="xpu", dtype=torch.bfloat16)
    weight = torch.empty(n, 32, device="xpu", dtype=torch.bfloat16)
    assert bf16_dense_gemm_xe20(a, weight).shape == (m, n)


@pytest.mark.parametrize(
    "case", ["dtype", "rank", "k_mismatch", "k_alignment", "n_alignment", "strides"]
)
def test_invalid_inputs(case):
    a = torch.empty(4, 32, device="xpu", dtype=torch.bfloat16)
    weight = torch.empty(64, 32, device="xpu", dtype=torch.bfloat16)
    if case == "dtype":
        a = a.float()
    elif case == "rank":
        a = a.unsqueeze(0)
    elif case == "k_mismatch":
        weight = weight[:, :16].contiguous()
    elif case == "k_alignment":
        a, weight = a[:, :16].contiguous(), weight[:, :16].contiguous()
    elif case == "n_alignment":
        weight = weight[:63]
    else:
        a = torch.empty(32, 4, device="xpu", dtype=torch.bfloat16).t()
    with pytest.raises(RuntimeError):
        bf16_dense_gemm_xe20(a, weight)


def test_fake_and_compile():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        a = torch.empty(128, 12288, device="xpu", dtype=torch.bfloat16)
        weight = torch.empty(1536, 12288, device="xpu", dtype=torch.bfloat16)
        actual = bf16_dense_gemm_xe20(a, weight)
        assert actual.shape == (128, 1536)
    a = torch.randn(128, 12288, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(1536, 12288, device="xpu", dtype=torch.bfloat16) * 0.02
    compiled = torch.compile(bf16_dense_gemm_xe20, backend="eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(a, weight), bf16_dense_gemm_xe20(a, weight), atol=0, rtol=0
    )
