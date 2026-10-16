import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest
import sgl_kernel
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available()
    or not hasattr(torch.ops.sgl_kernel, "qsa_sparse_attention"),
    reason="Requires an XPU build with qsa_sparse_attention enabled",
)


def reference(q, k, v, slots, scale):
    result = torch.zeros_like(q)
    group_size = q.shape[1] // k.shape[1]
    for row in range(q.shape[0]):
        valid = slots[row][(slots[row] >= 0) & (slots[row] < k.shape[0])].long()
        if valid.numel() == 0:
            continue
        keys = k[valid].float().repeat_interleave(group_size, dim=1)
        values = v[valid].float().repeat_interleave(group_size, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys) * scale
        result[row] = torch.einsum("hk,khd->hd", scores.softmax(dim=-1), values).to(
            result.dtype
        )
    return result


def test_qsa_python_wrapper_forwards_to_native(monkeypatch):
    native_op = torch.ops.sgl_kernel.qsa_sparse_attention
    calls = []

    def traced(*args):
        calls.append(args)
        return native_op(*args)

    monkeypatch.setattr(torch.ops.sgl_kernel, "qsa_sparse_attention", traced)
    q = torch.randn(1, 4, 256, device="xpu", dtype=torch.bfloat16)
    k = torch.randn(8, 2, 256, device="xpu", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    slots = torch.tensor([[0, 3, -1]], device="xpu", dtype=torch.int32)

    for scale in (None, 0.12):
        calls.clear()
        actual = sgl_kernel.qsa_sparse_attention(q, k, v, slots, scale)

        assert len(calls) == 1
        assert all(
            arg is expected for arg, expected in zip(calls[0][:4], (q, k, v, slots))
        )
        assert calls[0][4] == scale
        torch.testing.assert_close(
            actual.cpu(),
            reference(q.cpu(), k.cpu(), v.cpu(), slots.cpu(), scale or 0.0625),
            atol=0.025,
            rtol=0.025,
        )


@pytest.mark.parametrize(
    "rows,heads,kv_heads,topk,valid_count",
    [
        (0, 12, 1, 64, 0),
        (1, 12, 1, 0, 0),
        (1, 12, 1, 1, 1),
        (1, 12, 1, 16, 9),
        (2, 24, 2, 17, 10),
        (1, 12, 1, 63, 55),
        (2, 24, 2, 64, 0),
        (1, 24, 2, 65, 35),
        (1, 12, 1, 127, 116),
        (16, 24, 2, 128, 93),
        (2, 24, 2, 129, 103),
        (2, 6, 2, 512, 217),
        (1, 24, 2, 2051, 1301),
        (1, 24, 2, 2051, 2051),
    ],
)
@pytest.mark.parametrize("scale", [None, 0.0, 0.12])
def test_qsa_bf16_matches_reference(rows, heads, kv_heads, topk, valid_count, scale):
    torch.manual_seed(1234)
    q = torch.randn(rows, heads, 256).bfloat16()
    k = torch.randn(256, kv_heads, 256).bfloat16()
    v = torch.randn_like(k)
    slots = torch.full((rows, topk), -1, dtype=torch.int32)
    if valid_count:
        slots[:, :valid_count] = torch.randint(
            0, k.shape[0], (rows, valid_count), dtype=torch.int32
        )
        if valid_count > 5:
            slots[:, 3] = -1
            slots[:, valid_count - 2] = -1
    sm_scale = scale or 1 / math.sqrt(256)
    expected = reference(q, k, v, slots, sm_scale)
    actual = torch.ops.sgl_kernel.qsa_sparse_attention(
        q.xpu(), k.xpu(), v.xpu(), slots.xpu(), scale
    ).cpu()
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)


@pytest.mark.parametrize("topk", [1, 17, 65])
@pytest.mark.parametrize("cache_tokens", [0, 4])
def test_qsa_masks_out_of_range_cache_slots(topk, cache_tokens):
    torch.manual_seed(27)
    q = torch.randn(2, 4, 256, dtype=torch.bfloat16)
    k = torch.randn(cache_tokens, 2, 256, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    slots = torch.full((2, topk), -1, dtype=torch.int32)
    slots[0, 0] = 0
    slots[1, 0] = cache_tokens
    if topk > 1:
        slots[0, 1:4] = torch.tensor([4, 2**31 - 1, 2], dtype=torch.int32)
        slots[1, 1] = 2**31 - 1

    expected = reference(q, k, v, slots, 0.0625)
    actual = torch.ops.sgl_kernel.qsa_sparse_attention(
        q.xpu(), k.xpu(), v.xpu(), slots.xpu(), 0.0625
    ).cpu()
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)


def test_qsa_rejects_invalid_contract():
    q = torch.zeros(1, 12, 256, device="xpu", dtype=torch.bfloat16)
    k = torch.zeros(4, 1, 256, device="xpu", dtype=torch.bfloat16)
    slots = torch.zeros(1, 4, device="xpu", dtype=torch.int32)
    op = torch.ops.sgl_kernel.qsa_sparse_attention
    with pytest.raises(RuntimeError, match="only supports BF16"):
        op(q.float(), k, k, slots, None)
    with pytest.raises(RuntimeError, match="int32"):
        op(q, k, k, slots.long(), None)
    with pytest.raises(RuntimeError, match="token_slots must be on XPU"):
        op(q, k, k, slots.cpu(), None)
    with pytest.raises(RuntimeError, match="head dimension must be 256"):
        op(q[..., :128].contiguous(), k, k, slots, None)
    with pytest.raises(RuntimeError, match="scale must be finite"):
        op(q, k, k, slots, float("nan"))


def test_qsa_bf16_dpas_token_and_dimension_mapping():
    q = torch.zeros(1, 1, 256, dtype=torch.bfloat16, device="xpu")
    k = torch.zeros(16, 1, 256, dtype=torch.bfloat16, device="xpu")
    v = torch.zeros_like(k)
    slots = torch.arange(16, dtype=torch.int32, device="xpu").unsqueeze(0)
    for token in range(16):
        v[token, 0, token] = 1
    expected = torch.arange(16, dtype=torch.float32).softmax(dim=0)
    for dim in range(16):
        q.zero_()
        k.zero_()
        q[0, 0, dim] = 1
        k[:, 0, dim] = torch.arange(16, device="xpu")
        actual = torch.ops.sgl_kernel.qsa_sparse_attention(q, k, v, slots, 1.0)
        torch.testing.assert_close(
            actual.cpu()[0, 0, :16].float(), expected, atol=0.01, rtol=0.01
        )


@pytest.mark.parametrize("rows", [[1, 2], [3]])
def test_qsa_benchmark_sweeps_rows(rows):
    script = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "bench_qsa_sparse_attention.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--rows",
            *(str(row) for row in rows),
            "--topk",
            "16",
            "--skip-reference",
            "--repetitions",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    output = list(csv.DictReader(result.stdout.splitlines()))
    assert [(int(entry["rows"]), int(entry["topk"])) for entry in output] == [
        (row, 16) for row in rows
    ]
