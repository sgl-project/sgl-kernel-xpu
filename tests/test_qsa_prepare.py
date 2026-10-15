import pytest
import sgl_kernel
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available()
    or not hasattr(torch.ops.sgl_kernel, "qsa_index_q_norm_rope_store"),
    reason="Requires USE_QSA on Xe2",
)


def rope_reference(x, positions, cos_sin, axis_map, rotary, neox):
    half = rotary // 2
    output = x.clone()
    for pair in range(half):
        axis = int(axis_map[pair])
        cache_row = cos_sin[int(positions[axis])]
        c = cache_row[pair].to(x.dtype)
        s = cache_row[half + pair].to(x.dtype)
        a, b = (pair, pair + half) if neox else (2 * pair, 2 * pair + 1)
        output[a] = (x[a] * c - x[b] * s).to(x.dtype)
        output[b] = (x[b] * c + x[a] * s).to(x.dtype)
    return output


def norm_reference(x, weight, eps):
    scale = torch.rsqrt(x.float().square().mean() + eps)
    return (x.float() * scale * (1 + weight.float())).to(x.dtype)


@pytest.mark.parametrize("dim,rotary", [(64, 32), (128, 128), (256, 64)])
@pytest.mark.parametrize("neox", [True, False])
@pytest.mark.parametrize("axes", [1, 3])
def test_query_store_and_rope(dim, rotary, neox, axes):
    torch.manual_seed(153)
    tokens, heads, slots = 5, 4, 11
    qk = torch.randn(tokens, (heads + 1) * dim, dtype=torch.bfloat16, device="xpu")
    weight = (torch.randn(dim, dtype=torch.bfloat16, device="xpu") * 0.05).contiguous()
    angle = torch.arange(30, device="xpu")[:, None] / (
        10000 ** (torch.arange(rotary // 2, device="xpu")[None, :] / rotary)
    )
    cache = torch.cat([angle.cos(), angle.sin()], dim=1).float().contiguous()
    axis = (
        torch.arange(rotary // 2, device="xpu", dtype=torch.int32) % axes
    ).contiguous()
    positions = torch.tensor(
        [[2, 3, 4, 5, 6], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24]][:axes],
        device="xpu",
    ).long()
    if axes == 1:
        positions = positions[0]
    else:
        wide_positions = torch.zeros((3, tokens + 2), device="xpu", dtype=torch.int64)
        wide_positions[:, :tokens] = positions
        positions = wide_positions[:, :tokens]
    locs = torch.tensor([6, 2, -1, 9, 4], device="xpu", dtype=torch.int64)
    state = torch.full((slots, dim), 9, device="xpu", dtype=qk.dtype)
    rope_state = torch.full((slots, 3), -7, device="xpu", dtype=torch.int64)
    actual = sgl_kernel.qsa_index_q_norm_rope_store(
        qk,
        positions,
        cache,
        axis,
        weight,
        locs,
        state,
        rope_state,
        heads,
        rotary,
        1e-6,
        neox,
        q_heads_padded=6,
        out_dtype=torch.bfloat16,
    )
    expected = torch.zeros_like(actual)
    for token in range(tokens):
        pos = positions[:, token] if axes == 3 else positions[token].repeat(3)
        for head in range(heads):
            source = qk[token, head * dim : (head + 1) * dim]
            expected[token, head] = rope_reference(
                norm_reference(source, weight, 1e-6), pos, cache, axis, rotary, neox
            )
        if locs[token] >= 0:
            slot = int(locs[token])
            torch.testing.assert_close(state[slot], qk[token, heads * dim :])
            torch.testing.assert_close(rope_state[slot], pos)
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
    assert (state[0] == 9).all()
    assert (rope_state[0] == -7).all()
    assert (actual[:, heads:] == 0).all()


@pytest.mark.parametrize("ratio,neox", [(2, False), (4, True), (8, False), (16, True)])
def test_compress_staged_ring_groups(ratio, neox):
    torch.manual_seed(214)
    dim, rotary = 128, 64
    source = torch.randn(64, dim, device="xpu", dtype=torch.bfloat16)
    rope_pos = torch.randint(0, 20, (64, 3), device="xpu", dtype=torch.int64)
    cos_sin = torch.randn(20, rotary, device="xpu", dtype=torch.float32)
    axis = (torch.arange(rotary // 2, device="xpu", dtype=torch.int32) % 3).contiguous()
    weight = torch.randn(dim, device="xpu", dtype=torch.bfloat16) * 0.08
    groups = torch.tensor(
        [[(start + j) % 64 for j in range(ratio)] for start in [60, 9, 29, 37, 47]],
        device="xpu",
        dtype=torch.int32,
    )
    groups[2, -1] = -1
    writes = torch.tensor([4, -1, 1, 6, 2], device="xpu", dtype=torch.int32)
    dst = torch.full((8, dim), 7, device="xpu", dtype=torch.bfloat16)
    result = sgl_kernel.qsa_index_k_compress_store(
        source,
        groups,
        rope_pos,
        cos_sin,
        axis,
        weight,
        writes,
        dst,
        ratio,
        rotary,
        1e-6,
        neox,
    )
    assert result is None
    for g in (0, 3, 4):
        pooled = source[groups[g].long()].float().mean(0).to(torch.bfloat16)
        expected = rope_reference(
            norm_reference(pooled, weight, 1e-6),
            rope_pos[int(groups[g, 0])],
            cos_sin,
            axis,
            rotary,
            neox,
        )
        torch.testing.assert_close(
            dst[int(writes[g])], expected, atol=0.016, rtol=0.016
        )
    assert (dst[1] == 7).all()
    assert (dst[0] == 7).all()


def test_rejects_unsupported_configuration():
    qk = torch.ones((1, 5 * 128), device="xpu", dtype=torch.bfloat16)
    weight = torch.ones(128, device="xpu", dtype=torch.bfloat16)
    cache = torch.zeros((3, 64), device="xpu")
    axis = torch.zeros(32, device="xpu", dtype=torch.int32)
    pos = torch.zeros(1, device="xpu", dtype=torch.int64)
    slots = torch.zeros((2, 128), device="xpu", dtype=torch.bfloat16)
    rope_pos = torch.zeros((2, 3), device="xpu", dtype=torch.int64)
    args = (qk, pos, cache, axis, weight, pos, slots, rope_pos, 4, 64, 1e-6, True)
    direct = torch.ops.sgl_kernel.qsa_index_q_norm_rope_store(*args, 6, torch.bfloat16)
    assert direct.shape == (1, 6, 128)
    with pytest.raises(NotImplementedError, match="BF16 output only"):
        sgl_kernel.qsa_index_q_norm_rope_store(*args, out_dtype=torch.float8_e4m3fn)
    with pytest.raises(RuntimeError, match="BF16 output only"):
        torch.ops.sgl_kernel.qsa_index_q_norm_rope_store(
            *args, None, torch.float8_e4m3fn
        )
    with pytest.raises(RuntimeError, match="head count or buffer shapes"):
        sgl_kernel.qsa_index_q_norm_rope_store(*args, q_heads_padded=3)
    with pytest.raises(RuntimeError, match="BF16"):
        sgl_kernel.qsa_index_q_norm_rope_store(qk.float(), *args[1:])
    with pytest.raises(RuntimeError, match="compressed_k_buffer"):
        sgl_kernel.qsa_index_k_compress_store(
            slots,
            torch.zeros((1, 2), device="xpu", dtype=torch.int32),
            rope_pos,
            cache,
            axis,
            weight,
            torch.zeros(1, device="xpu", dtype=torch.int32),
            torch.empty((2, 128), device="xpu", dtype=torch.float8_e4m3fn),
            2,
            64,
            1e-6,
            True,
        )


def test_empty_rows_and_default_output_shape():
    dim = 64
    cache = torch.ones((2, dim), device="xpu")
    axis = torch.zeros(dim // 2, device="xpu", dtype=torch.int32)
    weight = torch.zeros(dim, device="xpu", dtype=torch.bfloat16)
    state = torch.ones((2, dim), device="xpu", dtype=torch.bfloat16)
    rope_pos = torch.zeros((2, 3), device="xpu", dtype=torch.int64)
    q = sgl_kernel.qsa_index_q_norm_rope_store(
        torch.empty((0, 3 * dim), device="xpu", dtype=torch.bfloat16),
        torch.empty((0,), device="xpu", dtype=torch.int64),
        cache,
        axis,
        weight,
        torch.empty(0, device="xpu", dtype=torch.int64),
        state,
        rope_pos,
        2,
        dim,
        1e-6,
        True,
    )
    assert q.shape == (0, 2, dim)
    dst = torch.ones((2, dim), device="xpu", dtype=torch.bfloat16)
    sgl_kernel.qsa_index_k_compress_store(
        state,
        torch.empty((0, 4), device="xpu", dtype=torch.int32),
        rope_pos,
        cache,
        axis,
        weight,
        torch.empty(0, device="xpu", dtype=torch.int32),
        dst,
        4,
        dim,
        1e-6,
        True,
    )
    assert (dst == 1).all()
