import sys

import pytest
import sgl_kernel
import torch
import torch.nn.functional as F

device = "xpu"
CHANNELS = 10240  # Qwen3.8-Flash-Next: hc_count 4 x hidden 2560
KERNEL_SIZE = 4
DILATION = 3
STATE_LEN = (KERNEL_SIZE - 1) * DILATION


# Reference: the decode branch of Qwen4ExpPLELayer._short_conv (index_select, cat, index_put).
def short_conv_state_eager(state, state_indices, x):
    conv_input = torch.cat(
        [state.index_select(0, state_indices).to(x.dtype), x.unsqueeze(-1)], dim=-1
    )
    real = state_indices != 0
    state[state_indices[real]] = conv_input[real][:, :, 1:].to(state.dtype)
    return conv_input


# Reference: the target-verify branch of Qwen4ExpPLELayer._short_conv and forward.
def verify_conv_eager(
    x, residual, weight, state, state_indices, valid, width, dilation, intermediate
):
    requests = state_indices.numel()
    channels, state_len = state.shape[1:]
    selected = state.index_select(0, state_indices.long()).to(x.dtype)
    seq = x.view(requests, width, channels).transpose(1, 2)
    conv_input = torch.cat([selected, seq], dim=-1)
    conv = F.conv1d(conv_input, weight, dilation=dilation, groups=channels)
    activated = F.silu(conv.transpose(1, 2).reshape(-1, channels))
    output = torch.where(
        valid.unsqueeze(-1), residual + activated, torch.zeros_like(residual)
    )
    if intermediate is not None and state_len:
        windows = (
            conv_input.unfold(2, state_len, 1)[:, :, 1 : width + 1]
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        windows = torch.where(
            valid.view(requests, width, 1, 1), windows, torch.zeros_like(windows)
        )
        intermediate[:requests, :width] = windows.to(intermediate.dtype)
    return output


def make_state(slots, channels, state_len, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(slots, channels, state_len, generator=g).to(dtype).to(device)


def make_x(tokens, channels, dtype, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(tokens, channels, generator=g).to(dtype).to(device)


def decode_indices(tokens, slots, padding, seed=2):
    # Unique real slots plus `padding` rows pointing at the padding slot 0.
    g = torch.Generator().manual_seed(seed)
    real = torch.randperm(slots - 1, generator=g)[: tokens - padding] + 1
    idx = torch.cat([real, torch.zeros(padding, dtype=torch.long)])
    return idx[torch.randperm(tokens, generator=g)].to(device)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens,channels,state_len,padding",
    [
        (1, CHANNELS, STATE_LEN, 0),
        (8, CHANNELS, STATE_LEN, 3),
        (64, CHANNELS, STATE_LEN, 0),
        (5, 300, 1, 2),
        (5, 72, STATE_LEN, 1),  # vectorized, partial last channel block
        (3, 257, 16, 1),
        (4, 7, 5, 0),
    ],
)
def test_short_conv_state_matches_eager(tokens, channels, state_len, padding, dtype):
    slots = tokens + 4
    state = make_state(slots, channels, state_len, dtype)
    x = make_x(tokens, channels, dtype)
    idx = decode_indices(tokens, slots, padding)
    expected_state = state.clone()
    expected = short_conv_state_eager(expected_state, idx, x)
    padding_before = state[0].clone()

    out = sgl_kernel.qwen4_short_conv_state(state, idx, x)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    # The padding slot is skipped by the kernel, while eager would not write it either.
    torch.testing.assert_close(state[0], padding_before, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


def test_short_conv_state_misaligned_falls_back():
    # Offsetting state and out by one element breaks 16-byte alignment and selects the scalar path.
    tokens, slots, channels = 4, 9, 64
    numel = slots * channels * STATE_LEN
    state = (
        make_state(1, 1, numel + 1, torch.bfloat16)
        .view(-1)[1:]
        .view(slots, channels, STATE_LEN)
    )
    out_numel = tokens * channels * (STATE_LEN + 1)
    out = torch.empty(out_numel + 1, dtype=torch.bfloat16, device=device)[1:].view(
        tokens, channels, STATE_LEN + 1
    )
    x = make_x(tokens, channels, torch.bfloat16)
    idx = decode_indices(tokens, slots, 1)
    expected_state = state.clone()
    expected = short_conv_state_eager(expected_state, idx, x)
    sgl_kernel.qwen4_short_conv_state(state, idx, x, out=out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


def test_short_conv_state_pool_slice_16b_aligned():
    # A per-layer view into a shared pool: 16-byte aligned (fast path) but not 64-byte aligned.
    tokens, slots, channels = 6, 10, 72
    pool = make_state(1, 1, 8 + slots * channels * STATE_LEN, torch.bfloat16).view(-1)
    state = pool[8:].view(slots, channels, STATE_LEN)
    assert state.data_ptr() % 16 == 0 and state.data_ptr() % 64 != 0
    x = make_x(tokens, channels, torch.bfloat16)
    idx = decode_indices(tokens, slots, 2)
    guard = pool[:8].clone()
    expected_state = state.clone()
    expected = short_conv_state_eager(expected_state, idx, x)
    out = sgl_kernel.qwen4_short_conv_state(state, idx, x)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(pool[:8], guard, rtol=0, atol=0)


def test_short_conv_state_out_and_empty():
    state = make_state(6, 64, STATE_LEN, torch.bfloat16)
    x = make_x(3, 64, torch.bfloat16)
    idx = torch.tensor([3, 1, 5], device=device)
    expected_state = state.clone()
    expected = short_conv_state_eager(expected_state, idx, x)
    out = torch.empty(3, 64, STATE_LEN + 1, dtype=torch.bfloat16, device=device)
    assert sgl_kernel.qwen4_short_conv_state(state, idx, x, out=out) is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)

    empty = sgl_kernel.qwen4_short_conv_state(state, idx[:0], x[:0])  # no tokens: no-op
    assert empty.shape == (0, 64, STATE_LEN + 1)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


STATE_ERRORS = {
    "state_len_17": "state length must be in",
    "fp32_state": "state must be bfloat16 or float16",
    "int32_indices": "state_indices must be int64",
    "x_dtype": "x must have the same dtype as state",
    "x_shape": "x must have shape",
    "out_shape": "conv_input must have shape",
    "non_contiguous_state": "state must be contiguous",
    "cpu_x": "x must be on XPU",
    "out_overlaps_state": "refer to a single memory location",
    "x_overlaps_state": "refer to a single memory location",
}


@pytest.mark.parametrize("case", STATE_ERRORS)
def test_short_conv_state_rejects_invalid(case):
    state_len = 17 if case == "state_len_17" else STATE_LEN
    dtype = torch.float32 if case == "fp32_state" else torch.bfloat16
    state = make_state(6, 32, state_len, dtype)
    x = make_x(2, 32, dtype)
    idx = torch.tensor([1, 2], device=device)
    out = None
    if case == "int32_indices":
        idx = idx.int()
    elif case == "x_dtype":
        x = x.half()
    elif case == "x_shape":
        x = make_x(2, 31, dtype)
    elif case == "out_shape":
        out = torch.empty(2, 32, state_len, dtype=dtype, device=device)
    elif case == "non_contiguous_state":
        state = make_state(6, 32, state_len * 2, dtype)[:, :, ::2]
    elif case == "cpu_x":
        x = x.cpu()
    elif case == "out_overlaps_state":
        out = state.view(-1)[: 2 * 32 * (state_len + 1)].view(2, 32, state_len + 1)
    elif case == "x_overlaps_state":
        x = state[4].view(-1)[: 2 * 32].view(2, 32)
    with pytest.raises(RuntimeError, match=STATE_ERRORS[case]):
        sgl_kernel.qwen4_short_conv_state(state, idx, x, out=out)


def make_verify_inputs(
    requests,
    width,
    channels,
    kernel_size,
    dilation,
    state_dtype,
    index_dtype,
    seed=0,
):
    g = torch.Generator().manual_seed(seed)
    state_len = (kernel_size - 1) * dilation
    slots = requests + 3
    tokens = requests * width
    x = torch.randn(tokens, channels, generator=g).to(torch.bfloat16)
    residual = torch.randn(tokens, channels, generator=g).to(torch.bfloat16)
    weight = (torch.randn(channels, 1, kernel_size, generator=g) * 0.5).to(
        torch.bfloat16
    )
    state = torch.randn(slots, channels, state_len, generator=g).to(state_dtype)
    # Padding requests reuse slot 0; verify only reads the state.
    idx = torch.randperm(slots, generator=g)[:requests]
    idx[requests // 2 :: 3] = 0
    valid = torch.rand(tokens, generator=g) < 0.75
    return [
        t.to(device) for t in (x, residual, weight, state, idx.to(index_dtype), valid)
    ]


@pytest.mark.parametrize("inter_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "requests,width,channels,kernel_size,dilation",
    [
        (1, 1, CHANNELS, KERNEL_SIZE, DILATION),
        (4, 4, CHANNELS, KERNEL_SIZE, DILATION),
        (3, 8, 300, KERNEL_SIZE, DILATION),
        (2, 32, 64, KERNEL_SIZE, DILATION),
        (5, 3, 17, 2, 1),
        (2, 3, 16, 1, 1),  # state_len 0
        (2, 32, 300, 17, 1),  # largest tiles: state_len 16, width 32
    ],
)
def test_verify_conv_matches_eager(
    requests,
    width,
    channels,
    kernel_size,
    dilation,
    state_dtype,
    index_dtype,
    inter_dtype,
):
    x, residual, weight, state, idx, valid = make_verify_inputs(
        requests, width, channels, kernel_size, dilation, state_dtype, index_dtype
    )
    state_len = state.shape[2]
    state_before = state.clone()
    inter = expected_inter = None
    if inter_dtype is not None:
        # Larger than [R, W] so the kernel must leave the rest untouched.
        inter = torch.full(
            (requests + 2, width + 1, channels, state_len),
            7.0,
            dtype=inter_dtype,
            device=device,
        )
        expected_inter = inter.clone()
    expected = verify_conv_eager(
        x, residual, weight, state, idx, valid, width, dilation, expected_inter
    )
    out = sgl_kernel.qwen4_verify_conv(
        x, residual, weight, state, idx, valid, width, dilation, inter
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, state_before, rtol=0, atol=0)
    if inter is not None:
        torch.testing.assert_close(inter, expected_inter, rtol=0, atol=0)


def test_verify_conv_strided_state_indices_and_intermediate():
    requests, width, channels = 3, 4, 96
    x, residual, weight, _, _, valid = make_verify_inputs(
        requests, width, channels, KERNEL_SIZE, DILATION, torch.bfloat16, torch.int64
    )
    # A state view with padded columns, every other index, and a permuted intermediate buffer.
    state = make_state(8, channels, STATE_LEN + 3, torch.bfloat16)[:, :, :STATE_LEN]
    idx = torch.tensor([5, 0, 2, 9, 7, 1], device=device)[::2]
    inter = torch.zeros(
        channels, requests, width, STATE_LEN, dtype=torch.float32, device=device
    ).permute(1, 2, 0, 3)
    expected_inter = inter.clone()
    expected = verify_conv_eager(
        x, residual, weight, state, idx, valid, width, DILATION, expected_inter
    )
    out = sgl_kernel.qwen4_verify_conv(
        x, residual, weight, state, idx, valid, width, DILATION, inter
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(inter, expected_inter, rtol=0, atol=0)


def test_verify_conv_padding_slot_shared_with_intermediate():
    # Slot 0 serves both padding rows; each request still gets its own intermediate rows.
    requests, width, channels = 4, 4, 300
    x, residual, weight, state, _, valid = make_verify_inputs(
        requests, width, channels, KERNEL_SIZE, DILATION, torch.float32, torch.int64
    )
    idx = torch.tensor([0, 3, 0, 1], device=device)
    inter = torch.full(
        (requests, width, channels, STATE_LEN), 5.0, dtype=torch.float32, device=device
    )
    expected_inter = inter.clone()
    expected = verify_conv_eager(
        x, residual, weight, state, idx, valid, width, DILATION, expected_inter
    )
    out = sgl_kernel.qwen4_verify_conv(
        x, residual, weight, state, idx, valid, width, DILATION, inter
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(inter, expected_inter, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["x", "state"])
def test_verify_conv_rejects_output_overlap(name):
    requests, width, channels = 2, 4, 32
    x, residual, weight, state, idx, valid = make_verify_inputs(
        requests, width, channels, KERNEL_SIZE, DILATION, torch.bfloat16, torch.int64
    )
    src = x if name == "x" else state
    buf = torch.zeros(max(x.numel(), src.numel()), dtype=torch.bfloat16, device=device)
    buf[: src.numel()] = src.reshape(-1)
    output = buf[: x.numel()].view_as(x)
    if name == "x":
        x = buf[: x.numel()].view_as(x)
    else:
        state = buf[: state.numel()].view_as(state)
    with pytest.raises(RuntimeError, match="refer to a single memory location"):
        torch.ops.sgl_kernel.qwen4_verify_conv(
            output, None, x, residual, weight, state, idx, valid, width, DILATION
        )


def test_verify_conv_output_may_alias_residual():
    requests, width, channels = 3, 4, 300
    x, residual, weight, state, idx, valid = make_verify_inputs(
        requests, width, channels, KERNEL_SIZE, DILATION, torch.bfloat16, torch.int64
    )
    expected = verify_conv_eager(
        x, residual, weight, state, idx, valid, width, DILATION, None
    )
    torch.ops.sgl_kernel.qwen4_verify_conv(
        residual, None, x, residual, weight, state, idx, valid, width, DILATION
    )
    torch.testing.assert_close(residual, expected, rtol=0, atol=0)


def test_verify_conv_special_values():
    # silu must follow eager for inf, -inf and NaN conv outputs. Specials stay in step 0 only: the XPU
    # oneDNN conv1d reference turns a neighbouring step's inf into NaN.
    requests, width, channels = 1, 2, 8
    x, residual, _, state, idx, _ = make_verify_inputs(
        requests, width, channels, 1, 1, torch.bfloat16, torch.int64
    )
    weight = torch.ones(channels, 1, 1, dtype=torch.bfloat16, device=device)
    x[0, :3] = torch.tensor([float("inf"), float("-inf"), float("nan")])
    valid = torch.tensor([True, False], device=device)
    expected = verify_conv_eager(x, residual, weight, state, idx, valid, width, 1, None)
    out = sgl_kernel.qwen4_verify_conv(x, residual, weight, state, idx, valid, width, 1)
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)


def test_verify_conv_empty():
    x, residual, weight, state, idx, valid = make_verify_inputs(
        1, 4, 32, KERNEL_SIZE, DILATION, torch.bfloat16, torch.int64
    )
    out = sgl_kernel.qwen4_verify_conv(
        x[:0], residual[:0], weight, state, idx[:0], valid[:0], 4, DILATION
    )
    assert out.shape == (0, 32)


VERIFY_ERRORS = {
    "fp16_x": "x must be a 2-D bfloat16 tensor",
    "fp16_state": "state must be bfloat16 or float32",
    "width_mismatch": "x must have shape",
    "width_33": "width must be in",
    "valid_dtype": "valid must be a bool tensor",
    "residual_shape": "residual must be bfloat16 with shape",
    "intermediate_small": "intermediate must have shape",
    "intermediate_dtype": "intermediate must be bfloat16 or float32",
    "intermediate_dtype_no_state": "intermediate must be bfloat16 or float32",
    "intermediate_overlaps_x": "refer to a single memory location",
    "intermediate_overlaps_state": "refer to a single memory location",
    "dilation_mismatch": "dilation must equal state length",
    "cpu_state": "state must be on XPU",
}


@pytest.mark.parametrize("case", VERIFY_ERRORS)
def test_verify_conv_rejects_invalid(case):
    requests, width, channels = 2, 4, 32
    x, residual, weight, state, idx, valid = make_verify_inputs(
        requests, width, channels, KERNEL_SIZE, DILATION, torch.bfloat16, torch.int64
    )
    inter = None
    dilation = DILATION
    if case == "width_33":
        width = 33
        x, residual, weight, state, idx, valid = make_verify_inputs(
            requests,
            width,
            channels,
            KERNEL_SIZE,
            DILATION,
            torch.bfloat16,
            torch.int64,
        )
    elif case == "fp16_x":
        x = x.half()
    elif case == "fp16_state":
        state = state.half()
    elif case == "width_mismatch":
        width = 3
    elif case == "valid_dtype":
        valid = valid.int()
    elif case == "residual_shape":
        residual = residual[:-1]
    elif case == "intermediate_small":
        inter = torch.zeros(
            requests, width - 1, channels, STATE_LEN, dtype=x.dtype, device=device
        )
    elif case == "intermediate_dtype":
        inter = torch.zeros(
            requests, width, channels, STATE_LEN, dtype=torch.half, device=device
        )
    elif case == "intermediate_dtype_no_state":
        x, residual, weight, state, idx, valid = make_verify_inputs(
            requests, width, channels, 1, 1, torch.bfloat16, torch.int64
        )
        dilation = 1
        inter = torch.zeros(
            requests, width, channels, 0, dtype=torch.half, device=device
        )
    elif case in ("intermediate_overlaps_x", "intermediate_overlaps_state"):
        # The intermediate buffer shares storage with the input it must not clobber.
        src = x if case == "intermediate_overlaps_x" else state
        buf = torch.zeros(
            requests * width * channels * STATE_LEN, dtype=src.dtype, device=device
        )
        buf[: src.numel()] = src.reshape(-1)
        inter = buf.view(requests, width, channels, STATE_LEN)
        if case == "intermediate_overlaps_x":
            x = buf[: x.numel()].view_as(x)
        else:
            state = buf[: state.numel()].view_as(state)
    elif case == "dilation_mismatch":
        dilation = 2
    elif case == "cpu_state":
        state = state.cpu()
    with pytest.raises(RuntimeError, match=VERIFY_ERRORS[case]):
        sgl_kernel.qwen4_verify_conv(
            x, residual, weight, state, idx, valid, width, dilation, inter
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
