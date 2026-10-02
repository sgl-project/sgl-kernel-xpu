import pytest
import sgl_kernel  # noqa: F401
import torch


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize(
    "tokens,hidden,vocab,start",
    (
        (0, 6144, 129, 23),
        (1, 7, 129, 23),
        (17, 769, 129, 23),
        (17, 770, 129, -(2**31) + 16),
        (17, 769, 0, 23),
        (17, 0, 129, 23),
        (4096, 6144, 129, 23),
    ),
)
def test_inkling_vocab_parallel_embedding(
    dtype, index_dtype, tokens, hidden, vocab, start
):
    torch.manual_seed(17)
    weight = torch.randn((vocab, hidden), dtype=dtype, device="xpu")
    ids = torch.randint(
        start - 7, start + vocab + 7, (tokens,), dtype=index_dtype, device="xpu"
    )
    if tokens >= 4:
        ids[:4] = torch.tensor(
            [start - 1, start, start + vocab - 1, start + vocab],
            dtype=index_dtype,
            device="xpu",
        )
    output = torch.empty((tokens, hidden), dtype=dtype, device="xpu")
    torch.ops.sgl_kernel.inkling_vocab_parallel_embedding(
        output, ids, weight, start, start + vocab
    )
    if vocab == 0:
        reference = torch.zeros_like(output)
    else:
        local = (ids.long() - start).clamp(0, vocab - 1)
        reference = torch.nn.functional.embedding(local, weight)
        reference.masked_fill_(
            ((ids < start) | (ids >= start + vocab)).unsqueeze(-1), 0
        )
    torch.xpu.synchronize()
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


def test_inkling_vocab_parallel_embedding_large_int64_ids():
    weight = torch.randn((3, 7), dtype=torch.bfloat16, device="xpu")
    ids = torch.tensor(
        [2**40 - 1, 2**40, 2**40 + 2, 2**40 + 3],
        dtype=torch.int64,
        device="xpu",
    )
    output = torch.empty((4, 7), dtype=weight.dtype, device="xpu")
    torch.ops.sgl_kernel.inkling_vocab_parallel_embedding(
        output, ids, weight, 2**40, 2**40 + 3
    )
    reference = torch.stack(
        (torch.zeros_like(weight[0]), weight[0], weight[2], torch.zeros_like(weight[0]))
    )
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
