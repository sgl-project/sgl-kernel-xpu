import pytest
import sgl_kernel  # noqa: F401
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available()
    or not hasattr(torch.ops.sgl_kernel, "qsa_mqa_prefill")
    or not hasattr(torch.ops.sgl_kernel, "qsa_mqa_decode"),
    reason="Requires an XPU build with QSA MQA operators enabled",
)


def test_mqa_int32_q_index_overflow():
    rows = 4_194_305
    q = torch.zeros(rows, 4, 128, device="xpu", dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 128, device="xpu", dtype=torch.bfloat16)
    q[-1, -1, -1] = 1
    k[0, 0, -1] = 1
    starts = torch.zeros(rows, device="xpu", dtype=torch.int32)
    ends = torch.ones(rows, device="xpu", dtype=torch.int32)

    prefill = sgl_kernel.qsa_mqa_prefill(q, k, starts, ends, 1.0)
    decode = sgl_kernel.qsa_mqa_decode(
        q, k.reshape(1, 1, 1, 128), starts[:, None].contiguous(), ends, 1, 1.0
    )

    for actual in (prefill, decode):
        torch.testing.assert_close(actual[-1, 0], torch.ones((), device="xpu"))
        torch.testing.assert_close(actual[0, 0], torch.zeros((), device="xpu"))


def test_prefill_int32_k_index_overflow():
    columns = 16_777_217
    q = torch.zeros(1, 4, 128, device="xpu", dtype=torch.bfloat16)
    k = torch.zeros(columns, 1, 128, device="xpu", dtype=torch.bfloat16)
    q[0, -1, -1] = 1
    k[-1, 0, -1] = 1
    starts = torch.zeros(1, device="xpu", dtype=torch.int32)
    ends = torch.full_like(starts, columns)

    actual = sgl_kernel.qsa_mqa_prefill(q, k, starts, ends, 1.0)

    torch.testing.assert_close(actual[0, -1], torch.ones((), device="xpu"))
    torch.testing.assert_close(actual[0, 0], torch.zeros((), device="xpu"))
