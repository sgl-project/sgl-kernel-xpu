import pytest
import sgl_kernel  # noqa: F401
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available()
    or not hasattr(torch.ops.sgl_kernel, "qsa_sparse_attention"),
    reason="Requires an XPU build with qsa_sparse_attention enabled",
)

HEAD_DIM = 256


def test_sparse_attention_int32_q_output_index_overflow():
    # rows * heads * HEAD_DIM = 2^31 + 1024, so the last row's Q/output offset
    # wraps with int32 indexing.
    rows, heads = 2_097_153, 4
    q = torch.zeros(rows, heads, HEAD_DIM, device="xpu", dtype=torch.bfloat16)
    k = torch.zeros(1, 1, HEAD_DIM, device="xpu", dtype=torch.bfloat16)
    v = torch.full_like(k, 3)
    slots = torch.zeros(rows, 1, device="xpu", dtype=torch.int32)

    actual = torch.ops.sgl_kernel.qsa_sparse_attention(q, k, v, slots, 1.0)

    expected = torch.full((heads, HEAD_DIM), 3, device="xpu", dtype=torch.bfloat16)
    torch.testing.assert_close(actual[-1], expected)
    torch.testing.assert_close(actual[0], expected)


def test_sparse_attention_int32_kv_cache_index_overflow():
    # cache_tokens * kv_heads * HEAD_DIM = 2^31 + 512, so the last token's K/V
    # offset wraps with int32 indexing. Distinct per-KV-head values check that
    # the last KV head is also addressed correctly.
    tokens, kv_heads, heads = 4_194_305, 2, 4
    q = torch.zeros(1, heads, HEAD_DIM, device="xpu", dtype=torch.bfloat16)
    k = torch.zeros(tokens, kv_heads, HEAD_DIM, device="xpu", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    v[-1, 0] = 1
    v[-1, 1] = 2
    slots = torch.tensor([[tokens - 1]], device="xpu", dtype=torch.int32)

    actual = torch.ops.sgl_kernel.qsa_sparse_attention(q, k, v, slots, 1.0)

    for head, value in enumerate((1, 1, 2, 2)):
        torch.testing.assert_close(
            actual[0, head],
            torch.full((HEAD_DIM,), value, device="xpu", dtype=torch.bfloat16),
        )
