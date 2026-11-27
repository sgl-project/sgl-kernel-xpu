# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-free coverage for the seven-query Xe2 draft layer."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sgl_kernel import minicpm5_decode as mono
from test_decode import reduce_fp32


@pytest.fixture(scope="module")
def draft_device():
    if not torch.xpu.is_available() or not mono.is_available():
        pytest.skip("requires sgl-kernel-xpu mono")
    tp = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("xpu", rank)
    torch.xpu.set_device(device)
    if tp == 2:
        dist.init_process_group("gloo")
    try:
        groups = torch.tensor(mono.work_groups(device), device="cpu")
        if tp == 2:
            dist.all_reduce(groups, op=dist.ReduceOp.MIN)
        if groups.item() != 32:
            pytest.skip("requires the B70 grid on all ranks")
        yield device, tp, rank
    finally:
        if tp == 2:
            dist.destroy_process_group()


@pytest.mark.parametrize(
    "prefix,page,scale_layout",
    [
        (62, 64, None),
        (126, 128, None),
        (32767, 64, None),
        (62, 64, "tensor"),
        (126, 128, "channel"),
        (32767, 64, "channel"),
    ],
)
def test_draft_shared_history(draft_device, prefix, page, scale_layout):
    device, tp, rank = draft_device
    dtype = torch.bfloat16
    hidden_size, rows, head = 2048, 7, 128
    query_heads, kv_heads, intermediate = 32 // tp, 2 // tp, 8192 // tp
    q_size, kv_size = query_heads * head, kv_heads * head
    generator = torch.Generator(device=device).manual_seed(63)

    def rand(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=generator)

    def norm(x, weight):
        return F.rms_norm(x.float(), (x.shape[-1],), weight.float(), 1e-6).to(dtype)

    hidden, residual = rand(rows, hidden_size) * 0.2, rand(rows, hidden_size) * 0.2
    norm0, norm1 = (1 + rand(hidden_size) * 0.1 for _ in range(2))
    norm_q, norm_k = (1 + rand(head) * 0.1 for _ in range(2))
    generator.manual_seed(117 + rank)
    qkv = rand(q_size + 2 * kv_size, hidden_size) * 0.01
    out = rand(hidden_size, q_size) * 0.01
    gate_up = rand(2 * intermediate, hidden_size) * 0.01
    down = rand(hidden_size, intermediate) * 0.01

    weights = [norm0, norm1, norm_q, norm_k]

    def prepare(weight):
        if scale_layout is None:
            weights.append(weight)
            return weight
        data = weight.float()
        if scale_layout == "channel":
            amax = data.abs().amax(dim=1, keepdim=True)
        else:
            amax = data.abs().amax()
        scale = amax / 448
        fp8 = (data / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        weights.extend((fp8.t(), scale))
        return (fp8.float() * scale).to(dtype)

    qkv, out, gate_up, down = map(prepare, (qkv, out, gate_up, down))
    table = mono.LayerTable([weights], [True], tp, draft=True)
    workspace = mono.Workspace(
        device=device,
        dtype=dtype,
        layers=1,
        tp=tp,
        group=dist.group.WORLD if tp == 2 else None,
    )
    length = prefix + rows
    pages = (length + page - 1) // page
    positions = torch.arange(prefix, length, dtype=torch.int64, device=device)
    lengths = torch.full((1,), length, dtype=torch.int32, device=device)
    blocks = torch.arange(1, pages + 1, dtype=torch.int32, device=device)[None]
    slots = positions + page
    phase = rand(length, head // 2).float() * 0.1
    rope = torch.cat((phase.cos(), phase.sin()), dim=-1).to(dtype)
    keys = rand(pages + 1, page, kv_heads, head) * 0.05
    vals = rand(*keys.shape) * 0.05
    ref_keys, ref_vals = keys.clone(), vals.clone()

    def rotary(x):
        a, b = x.float().chunk(2, dim=-1)
        cs, sn = rope[positions].float().unsqueeze(1).chunk(2, dim=-1)
        return torch.cat((a * cs - b * sn, b * cs + a * sn), dim=-1).to(dtype)

    h = (hidden.float() + residual.float()).to(dtype)
    q, k, v = F.linear(norm(h, norm0), qkv).split((q_size, kv_size, kv_size), dim=-1)
    q = rotary(norm(q.view(rows, query_heads, head), norm_q))
    k = rotary(norm(k.view(rows, kv_heads, head), norm_k))
    ref_keys.view(-1, kv_heads, head)[slots] = k
    ref_vals.view(-1, kv_heads, head)[slots] = v.view(rows, kv_heads, head)
    history_k = ref_keys[1:].view(-1, kv_heads, head)[:length].float()
    history_v = ref_vals[1:].view(-1, kv_heads, head)[:length].float()
    group_size = query_heads // kv_heads
    scores = torch.einsum(
        "rhd,thd->rht", q.float(), history_k.repeat_interleave(group_size, dim=1)
    ) * (head**-0.5)
    attn = torch.einsum(
        "rht,thd->rhd",
        scores.softmax(dim=-1),
        history_v.repeat_interleave(group_size, dim=1),
    ).to(dtype)
    o = F.linear(attn.reshape(rows, q_size), out).float()
    if tp == 2:
        reduce_fp32(o)
    ref_res = (o.to(dtype).float() + h.float()).to(dtype)
    g, u = F.linear(norm(ref_res, norm1), gate_up).chunk(2, dim=-1)
    ref_out = F.linear((F.silu(g.float()) * u.float()).to(dtype), down).float()
    if tp == 2:
        reduce_fp32(ref_out)

    def forward():
        return workspace.decode_step(
            table,
            0,
            1,
            hidden=hidden,
            residual=residual,
            positions=positions,
            slots=slots,
            blocks=blocks,
            lengths=lengths,
            rope=rope,
            keys=[keys],
            values=[vals],
        )

    torch.xpu.synchronize()
    if tp == 2:
        dist.barrier()
    result, res = forward()
    torch.xpu.synchronize()
    workspace.check_health()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        replayed, replayed_res = forward()
    if tp == 2:
        dist.barrier()
    graph.replay()
    torch.xpu.synchronize()
    workspace.check_health()
    torch.testing.assert_close(result, replayed, rtol=0, atol=0)
    torch.testing.assert_close(res, replayed_res, rtol=0, atol=0)
    torch.testing.assert_close(res, ref_res, atol=0.025, rtol=0.025)
    torch.testing.assert_close(result, ref_out.to(dtype), atol=0.025, rtol=0.025)
    torch.testing.assert_close(keys, ref_keys, atol=0.025, rtol=0.025)
    torch.testing.assert_close(vals, ref_vals, atol=0.025, rtol=0.025)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
