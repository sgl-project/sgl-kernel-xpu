import sys

import pytest
import sgl_kernel
import torch
import utils

device = utils.get_device()


def sigmoid_mul_ref(x, gate):
    return x * torch.sigmoid(gate.reshape(x.shape))


def strided_gate(tokens, heads, head_dim, dtype):
    # Gate as a chunk of a [N, heads, 2 * head_dim] q/gate projection, like the attention output gate.
    q_gate = torch.randn(tokens, heads, 2 * head_dim, device=device, dtype=dtype) * 4
    return q_gate[..., head_dim:]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "tokens, heads, head_dim",
    [
        (1, 24, 256),  # Qwen3.8 decode
        (17, 24, 256),
        (1031, 24, 256),
        (3, 16, 128),
        (5, 2, 4),  # head_dim a multiple of 4 but not 8: 4-element vector path
        (4, 3, 12),
        (3, 5, 36),
        (5, 2, 6),  # head_dim not a multiple of 4: scalar path
        (4, 3, 7),
    ],
)
@pytest.mark.parametrize("gate_layout", ["flat", "strided_3d", "contiguous_3d"])
def test_fused_sigmoid_mul(tokens, heads, head_dim, gate_layout, inplace, dtype):
    hidden = heads * head_dim
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    if gate_layout == "flat":
        gate = torch.randn(tokens, hidden, device=device, dtype=dtype) * 4
    elif gate_layout == "strided_3d":
        gate = strided_gate(tokens, heads, head_dim, dtype)
    else:
        gate = strided_gate(tokens, heads, head_dim, dtype).contiguous()
    ref = sigmoid_mul_ref(x, gate)
    x_in = x.clone()
    out = sgl_kernel.fused_sigmoid_mul(x_in, gate, inplace=inplace)
    if inplace:
        assert out.data_ptr() == x_in.data_ptr()
    else:
        torch.testing.assert_close(x_in, x, rtol=0, atol=0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("offset", [0, 1, 2, 4])
def test_fused_sigmoid_mul_misaligned_views(offset, dtype):
    # Offsets of 4 elements break 16-byte alignment (4-element path); odd offsets fall back to scalar.
    tokens, heads, head_dim = 5, 2, 16
    hidden = heads * head_dim
    x_buf = torch.randn(tokens * hidden + 8, device=device, dtype=dtype)
    gate_buf = torch.randn(tokens * hidden + 8, device=device, dtype=dtype) * 4
    x = x_buf[offset : offset + tokens * hidden].view(tokens, hidden)
    gate = gate_buf[offset : offset + tokens * hidden].view(tokens, hidden)
    ref = sigmoid_mul_ref(x, gate)
    sgl_kernel.fused_sigmoid_mul(x, gate, inplace=True)
    torch.testing.assert_close(x, ref, rtol=0, atol=0)


def test_fused_sigmoid_mul_special_values():
    gate = torch.tensor(
        [[0.0, -0.0, 100.0, -100.0, float("inf"), float("-inf"), float("nan"), 1.0]]
    ).to(device)
    x = torch.tensor([[1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, float("nan")]]).to(device)
    gate, x = gate.to(torch.bfloat16), x.to(torch.bfloat16)
    out = sgl_kernel.fused_sigmoid_mul(x, gate)
    torch.testing.assert_close(
        out, sigmoid_mul_ref(x, gate), rtol=0, atol=0, equal_nan=True
    )


def test_fused_sigmoid_mul_empty():
    x = torch.empty(0, 6144, device=device, dtype=torch.bfloat16)
    assert sgl_kernel.fused_sigmoid_mul(x, torch.empty_like(x)).shape == x.shape


@pytest.mark.parametrize(
    "case",
    [
        "dtype_mismatch",
        "fp32",
        "shape_mismatch",
        "gate_last_dim_strided",
        "input_not_contiguous",
        "gate_overlaps_out",
    ],
)
def test_fused_sigmoid_mul_rejects_invalid(case):
    x = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    gate = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    if case == "dtype_mismatch":
        gate = gate.half()
    elif case == "fp32":
        x, gate = x.float(), gate.float()
    elif case == "shape_mismatch":
        gate = gate[:, :32]
    elif case == "gate_last_dim_strided":
        gate = torch.randn(4, 128, device=device, dtype=torch.bfloat16)[:, ::2]
    elif case == "input_not_contiguous":
        x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)[:, :64]
    else:
        with pytest.raises(RuntimeError):
            sgl_kernel.fused_sigmoid_mul(x, x, inplace=True)
        return
    with pytest.raises(RuntimeError):
        sgl_kernel.fused_sigmoid_mul(x, gate)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
