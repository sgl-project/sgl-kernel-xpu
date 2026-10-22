import math
import sys

import pytest
import sgl_kernel
import torch
import utils

device = utils.get_device()

# bf16 has 8 bits of mantissa; allow a one-ulp difference from the fp32 accumulation order.
BF16_TOL = {"rtol": 2**-6, "atol": 1e-3}


def gate_activation_ref(gate):
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    return torch.sigmoid(gate)


def gate_value_ref(gate, value):
    return gate_activation_ref(gate) * value.unsqueeze(-2)


def gate_reduce_ref(key, query, value):
    gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(key.shape[-1])
    return gate_activation_ref(gate) * value.unsqueeze(-2)


def rand_bf16(*shape, scale=1.0):
    return (torch.randn(*shape) * scale).to(device).to(torch.bfloat16)


@pytest.mark.parametrize("specify_out", [True, False])
@pytest.mark.parametrize(
    "tokens, hc_count, hidden",
    [
        (1, 4, 2560),  # Qwen3.8 PLE decode
        (3, 4, 2560),
        (17, 4, 2560),
        (257, 4, 2560),
        (5, 1, 8),  # smallest hidden
        (5, 3, 24),  # work-group larger than the number of vectors
        (4, 2, 4096),
        (2, 4, 8200),  # more vectors than work-items
    ],
)
def test_qwen4_gate_value(tokens, hc_count, hidden, specify_out):
    torch.manual_seed(0)
    gate = rand_bf16(tokens, hc_count, 1, scale=2.0)
    value = rand_bf16(tokens, hidden)
    expected = gate_value_ref(gate, value)
    if specify_out:
        out = torch.empty(tokens, hc_count, hidden, dtype=torch.bfloat16, device=device)
        sgl_kernel.qwen4_gate_value(gate, value, out=out)
    else:
        out = sgl_kernel.qwen4_gate_value(gate, value)
    assert out.shape == (tokens, hc_count, hidden)
    torch.testing.assert_close(out, expected, **BF16_TOL)


def test_qwen4_gate_value_special_gates():
    gate = torch.tensor(
        [0.0, 1e-8, -1e-8, 1e-3, -1e-3, 1.0, -1.0, 100.0, -100.0, 3e38, -3e38, 5e-39],
        dtype=torch.float32,
    ).to(torch.bfloat16)
    gate = gate.reshape(3, 4, 1).to(device)
    value = rand_bf16(3, 2560)
    out = sgl_kernel.qwen4_gate_value(gate, value)
    torch.testing.assert_close(out, gate_value_ref(gate, value), **BF16_TOL)
    # Zero gate has sign 0, so the activation is sigmoid(0) = 0.5.
    torch.testing.assert_close(out[0, 0], value[0] * 0.5, **BF16_TOL)


def test_qwen4_gate_value_nan_and_inf_gates():
    nan, inf = float("nan"), float("inf")
    gate = torch.tensor([nan, inf, -inf, 0.5], dtype=torch.bfloat16)
    gate = gate.reshape(1, 4, 1).to(device)
    value = rand_bf16(1, 2560)
    out = sgl_kernel.qwen4_gate_value(gate, value)
    torch.testing.assert_close(
        out, gate_value_ref(gate, value), equal_nan=True, **BF16_TOL
    )
    assert out[0, 0].isnan().all()


def test_qwen4_gate_reduce_propagates_nan():
    key = rand_bf16(2, 4, 2560)
    query = rand_bf16(2, 4, 2560)
    value = rand_bf16(2, 2560)
    key[0, 1, 7] = float("nan")
    key[1, 2, 0] = float("inf")
    query[1, 2, 0] = 0.0  # inf * 0 = NaN
    out = sgl_kernel.qwen4_gate_reduce(key, query, value)
    torch.testing.assert_close(
        out, gate_reduce_ref(key, query, value), equal_nan=True, **BF16_TOL
    )
    assert out[0, 1].isnan().all() and out[1, 2].isnan().all()
    assert not out[0, 0].isnan().any()


@pytest.mark.parametrize("specify_out", [True, False])
@pytest.mark.parametrize(
    "tokens, hc_count, hidden",
    [
        (1, 4, 2560),  # Qwen3.8 PLE verify
        (7, 4, 2560),
        (33, 4, 2560),
        (130, 4, 2560),
        (5, 1, 8),
        (5, 3, 24),
        (4, 2, 4096),
        (2, 4, 8200),
    ],
)
def test_qwen4_gate_reduce(tokens, hc_count, hidden, specify_out):
    torch.manual_seed(0)
    key = rand_bf16(tokens, hc_count, hidden)
    query = rand_bf16(tokens, hc_count, hidden)
    value = rand_bf16(tokens, hidden)
    expected = gate_reduce_ref(key, query, value)
    if specify_out:
        out = torch.empty_like(key)
        sgl_kernel.qwen4_gate_reduce(key, query, value, out=out)
    else:
        out = sgl_kernel.qwen4_gate_reduce(key, query, value)
    torch.testing.assert_close(out, expected, **BF16_TOL)


def test_qwen4_gate_reduce_zero_and_correlated_inputs():
    torch.manual_seed(0)
    key = rand_bf16(4, 4, 2560)
    value = rand_bf16(4, 2560)
    # Zero dot product -> sigmoid(0) = 0.5.
    zeros = torch.zeros_like(key)
    out = sgl_kernel.qwen4_gate_reduce(key, zeros, value)
    torch.testing.assert_close(
        out, (0.5 * value).unsqueeze(1).expand_as(out), **BF16_TOL
    )
    # Strongly positive and negative dot products saturate the gate in both directions.
    for sign in (1.0, -1.0):
        query = (key * sign).contiguous()
        out = sgl_kernel.qwen4_gate_reduce(key, query, value)
        torch.testing.assert_close(out, gate_reduce_ref(key, query, value), **BF16_TOL)


@pytest.mark.parametrize("op", ["gate_value", "gate_reduce"])
def test_qwen4_gate_empty(op):
    if op == "gate_value":
        out = sgl_kernel.qwen4_gate_value(rand_bf16(0, 4, 1), rand_bf16(0, 2560))
    else:
        out = sgl_kernel.qwen4_gate_reduce(
            rand_bf16(0, 4, 2560), rand_bf16(0, 4, 2560), rand_bf16(0, 2560)
        )
    assert out.shape == (0, 4, 2560)


@pytest.mark.parametrize(
    "case",
    [
        "float32",
        "non_contiguous",
        "hidden_not_multiple_of_8",
        "gate_wrong_shape",
        "value_wrong_tokens",
        "output_wrong_shape",
        "misaligned",
    ],
)
def test_qwen4_gate_value_rejects_invalid(case):
    gate = rand_bf16(3, 4, 1)
    value = rand_bf16(3, 2560)
    out = None
    if case == "float32":
        value = value.float()
    elif case == "non_contiguous":
        value = rand_bf16(2560, 3).t()
    elif case == "hidden_not_multiple_of_8":
        value = rand_bf16(3, 2564 - 1)
    elif case == "gate_wrong_shape":
        gate = rand_bf16(3, 4, 2)
    elif case == "value_wrong_tokens":
        value = rand_bf16(2, 2560)
    elif case == "output_wrong_shape":
        out = torch.empty(3, 4, 2048, dtype=torch.bfloat16, device=device)
    elif case == "misaligned":
        value = rand_bf16(3 * 2560 + 8)[1 : 1 + 3 * 2560].view(3, 2560)
    with pytest.raises(RuntimeError):
        sgl_kernel.qwen4_gate_value(gate, value, out=out)


@pytest.mark.parametrize(
    "case",
    [
        "float32",
        "non_contiguous",
        "hidden_not_multiple_of_8",
        "query_wrong_shape",
        "value_wrong_shape",
        "output_wrong_shape",
        "misaligned",
    ],
)
def test_qwen4_gate_reduce_rejects_invalid(case):
    key = rand_bf16(3, 4, 2560)
    query = rand_bf16(3, 4, 2560)
    value = rand_bf16(3, 2560)
    out = None
    if case == "float32":
        key = key.float()
    elif case == "non_contiguous":
        query = rand_bf16(3, 2560, 4).transpose(1, 2)
    elif case == "hidden_not_multiple_of_8":
        key, query, value = (
            rand_bf16(3, 4, 2563),
            rand_bf16(3, 4, 2563),
            rand_bf16(3, 2563),
        )
    elif case == "query_wrong_shape":
        query = rand_bf16(3, 4, 2048)
    elif case == "value_wrong_shape":
        value = rand_bf16(3, 2048)
    elif case == "output_wrong_shape":
        out = torch.empty(3, 2, 2560, dtype=torch.bfloat16, device=device)
    elif case == "misaligned":
        key = rand_bf16(3 * 4 * 2560 + 8)[1 : 1 + 3 * 4 * 2560].view(3, 4, 2560)
    with pytest.raises(RuntimeError):
        sgl_kernel.qwen4_gate_reduce(key, query, value, out=out)


def test_qwen4_gate_value_rejects_output_overlap():
    buf = rand_bf16(32)
    value = rand_bf16(1, 8)
    with pytest.raises(RuntimeError, match="single memory location"):
        sgl_kernel.qwen4_gate_value(buf[:4].view(1, 4, 1), value, out=buf.view(1, 4, 8))
    gate = rand_bf16(1, 4, 1)
    buf = rand_bf16(32)
    with pytest.raises(RuntimeError, match="single memory location"):
        sgl_kernel.qwen4_gate_value(gate, buf[:8].view(1, 8), out=buf.view(1, 4, 8))


@pytest.mark.parametrize("operand", ["key", "query"])
def test_qwen4_gate_reduce_rejects_partial_output_overlap(operand):
    buf = rand_bf16(40)
    key, query, value = rand_bf16(1, 4, 8), rand_bf16(1, 4, 8), rand_bf16(1, 8)
    partial = buf[:32].view(1, 4, 8)
    out = buf[8:].view(1, 4, 8)
    if operand == "key":
        key = partial
    else:
        query = partial
    with pytest.raises(RuntimeError, match="single memory location"):
        sgl_kernel.qwen4_gate_reduce(key, query, value, out=out)


def test_qwen4_gate_reduce_rejects_value_overlap():
    buf = rand_bf16(32)
    key, query = rand_bf16(1, 4, 8), rand_bf16(1, 4, 8)
    with pytest.raises(RuntimeError, match="single memory location"):
        sgl_kernel.qwen4_gate_reduce(
            key, query, buf[:8].view(1, 8), out=buf.view(1, 4, 8)
        )


@pytest.mark.parametrize("operand", ["key", "query"])
def test_qwen4_gate_reduce_in_place(operand):
    key, query, value = rand_bf16(3, 4, 2560), rand_bf16(3, 4, 2560), rand_bf16(3, 2560)
    expected = gate_reduce_ref(key, query, value)
    target = key if operand == "key" else query
    out = sgl_kernel.qwen4_gate_reduce(key, query, value, out=target)
    assert out.data_ptr() == target.data_ptr()
    torch.testing.assert_close(out, expected, **BF16_TOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
