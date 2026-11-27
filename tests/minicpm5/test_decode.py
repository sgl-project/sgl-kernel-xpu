# SPDX-License-Identifier: Apache-2.0
"""Weight-free MiniCPM5 decode coverage for the Xe2 opt-in build."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sgl_kernel import minicpm5_decode as mono

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not mono.is_available(),
    reason="requires an Xe2 XPU and USE_MONO_DECODE=ON",
)

HIDDEN, EXPERTS, TOPK, HEAD = 2048, 160, 16, 128
EPOCH = 16 * 16


def reduce_fp32(tensor):
    cpu = tensor.cpu()
    dist.all_reduce(cpu)
    tensor.copy_(cpu)


class Inputs:
    def __init__(
        self, tp, rank, dtype, dense, channelwise, verify=False, rows=1, shard_rank=None
    ):
        self.tp, self.dtype, self.dense, self.rows = tp, dtype, dense, rows
        self.device = torch.device("xpu", rank)
        torch.xpu.set_device(self.device)
        self.rng = torch.Generator(device=self.device).manual_seed(7)
        self.hidden = self.rand((rows, HIDDEN))
        self.residual = 0.2 * self.rand((rows, HIDDEN))
        self.router = self.rand((EXPERTS, HIDDEN)) * 0.02
        self.bias = torch.linspace(-0.5, 0.5, EXPERTS, device=self.device)
        self.qheads, self.kvheads = 32 // tp, 2 // tp
        self.i = (8192 if dense else 512) // tp
        qkv = 2 * (self.qheads + self.kvheads) * HEAD
        self.channels = channelwise
        self.norm0 = 1 + 0.1 * self.rand((HIDDEN,))
        self.norm1 = 1 + 0.1 * self.rand((HIDDEN,))
        self.rng.manual_seed(171 + (rank if shard_rank is None else shard_rank))
        self.qkv = self.linear(qkv, HIDDEN)
        self.out = self.linear(HIDDEN, self.qheads * HEAD)
        self.gate_up = self.linear(2 * self.i, HIDDEN)
        self.down = self.linear(HIDDEN, self.i)
        self.weights = [self.norm0, self.norm1, *self.qkv, *self.out]
        if not dense:
            self.experts_gate_up = self.fp8((EXPERTS, 2 * self.i, HIDDEN))
            self.experts_down = self.fp8((EXPERTS, HIDDEN, self.i))
            self.expert_scales = torch.linspace(
                0.003, 0.005, EXPERTS, device=self.device
            )
            self.weights += [
                self.router,
                self.bias,
                self.experts_gate_up,
                self.expert_scales,
                self.experts_down,
                self.expert_scales,
            ]
        self.weights += [*self.gate_up, *self.down]
        start = 62 if verify else 66
        self.positions = torch.arange(start, start + rows, device=self.device)
        if verify:
            self.blocks = torch.tensor([[1, 2]], device=self.device, dtype=torch.int32)
            self.slots = 64 + self.positions
            pages = 3
        else:
            self.blocks = torch.arange(
                1, 1 + 2 * rows, device=self.device, dtype=torch.int32
            ).reshape(rows, 2)
            self.slots = self.blocks[:, 1].long() * 64 + self.positions % 64
            pages = 1 + 2 * rows
        self.lengths = (self.positions + 1).int()
        self.keys = self.rand((pages, 64, self.kvheads, HEAD)) * 0.1
        self.values = self.rand(self.keys.shape) * 0.1
        phase = self.rand((128, 64)).float() * 0.1
        self.rope = torch.cat((phase.cos(), phase.sin()), dim=-1).to(self.dtype)

    def rand(self, shape):
        return torch.randn(
            shape, device=self.device, dtype=self.dtype, generator=self.rng
        )

    def fp8(self, shape):
        values = torch.randint(
            0, 2, shape, dtype=torch.uint8, device=self.device, generator=self.rng
        )
        return ((values << 7) | 0x38).view(torch.float8_e4m3fn)

    def linear(self, n, k):
        scale = torch.linspace(
            0.003, 0.005, n if self.channels else 1, device=self.device
        )
        return self.fp8((n, k)).t(), scale

    @staticmethod
    def project(x, weights):
        weight, scale = weights
        return F.linear(x.float(), weight.t().float() * scale.reshape(-1, 1)).to(
            x.dtype
        )

    def ref(self, key, value):
        dtype = self.dtype
        h = (self.hidden.float() + self.residual.float()).to(dtype)
        norm = F.rms_norm(h.float(), (HIDDEN,), self.norm0.float(), 1e-6).to(dtype)
        qkv = self.project(norm, self.qkv)
        qg, k, v = qkv.split(
            (2 * self.qheads * HEAD, self.kvheads * HEAD, self.kvheads * HEAD), dim=1
        )
        q, gate = qg.reshape(self.rows, self.qheads, 2 * HEAD).split(HEAD, dim=-1)
        k, v = k.reshape(self.rows, self.kvheads, HEAD), v.reshape(
            self.rows, self.kvheads, HEAD
        )
        cs, sn = self.rope[self.positions].float().unsqueeze(1).chunk(2, dim=-1)

        def rotate(x):
            a, b = x.float().chunk(2, dim=-1)
            return torch.cat((a * cs - b * sn, b * cs + a * sn), dim=-1).to(dtype)

        q, k = rotate(q), rotate(k)
        key.view(-1, self.kvheads, HEAD)[self.slots] = k
        value.view(-1, self.kvheads, HEAD)[self.slots] = v
        key, value = key.float(), value.float()
        attn = []
        for row in range(self.rows):
            pages = self.blocks[min(row, self.blocks.size(0) - 1)].long()
            length = int(self.lengths[row])
            history_k = (
                key[pages]
                .reshape(-1, self.kvheads, HEAD)[:length]
                .repeat_interleave(16, dim=1)
            )
            history_v = (
                value[pages]
                .reshape(-1, self.kvheads, HEAD)[:length]
                .repeat_interleave(16, dim=1)
            )
            scores = torch.einsum("hd,thd->ht", q[row].float(), history_k) * (
                HEAD**-0.5
            )
            mixed = torch.einsum("ht,thd->hd", scores.softmax(dim=-1), history_v)
            attn.append(
                (mixed.to(dtype) * gate[row].float().sigmoid().to(dtype)).flatten()
            )
        o = self.project(torch.stack(attn), self.out)
        if self.tp == 2:
            o_cpu = o.float().cpu()
            dist.all_reduce(o_cpu)
            o = o_cpu.to(device=self.device, dtype=dtype)
        res_out = (o.float() + h.float()).to(dtype)
        x = F.rms_norm(res_out.float(), (HIDDEN,), self.norm1.float(), 1e-6).to(dtype)
        g, u = self.project(x, self.gate_up).chunk(2, dim=-1)
        out = self.project((F.silu(g.float()) * u.float()).to(dtype), self.down)
        if not self.dense:
            scores = F.linear(x.float(), self.router.float()).sigmoid()
            ids = (scores + self.bias).topk(TOPK).indices
            picked = scores.gather(1, ids)
            picked = picked * (3.66 / picked.sum(dim=-1, keepdim=True))
            for row in range(self.rows):
                ew = (
                    self.experts_gate_up[ids[row]].float()
                    * self.expert_scales[ids[row], None, None]
                )
                dw = (
                    self.experts_down[ids[row]].float()
                    * self.expert_scales[ids[row], None, None]
                )
                g, u = torch.matmul(ew, x[row].float()).to(dtype).chunk(2, dim=-1)
                per_expert = torch.bmm(
                    dw, (F.silu(g.float()) * u.float()).to(dtype).float().unsqueeze(-1)
                )
                combined = (
                    (per_expert.squeeze(-1).to(dtype).float() * picked[row, :, None])
                    .sum(dim=0)
                    .to(dtype)
                )
                out[row] = combined + out[row]
        if self.tp == 2:
            out_cpu = out.float().cpu()
            dist.all_reduce(out_cpu)
            out = out_cpu.to(device=self.device, dtype=dtype)
        return out, res_out, ids if not self.dense else None

    def step(self, workspace, table, mode=0, rows=None):
        rows = self.rows if rows is None else rows
        return workspace.decode_step(
            table,
            0,
            1,
            hidden=self.hidden[:rows],
            residual=self.residual[:rows],
            positions=self.positions[:rows],
            slots=self.slots[:rows],
            blocks=self.blocks[:rows],
            lengths=self.lengths[:rows],
            rope=self.rope,
            keys=[self.keys],
            values=[self.values],
            mode=mode,
            all_reduce=reduce_fp32 if mode == 2 and self.tp == 2 else None,
        )

    def compare(self, groups=None, capture=False, split=False):
        if not mono.work_groups(self.device):
            pytest.skip("requires a supported Xe2 grid")
        if groups == 32 and mono.work_groups(self.device) < 32:
            pytest.skip("32-group path requires a B70-class GPU")
        table = mono.LayerTable([self.weights], [self.dense], self.tp)
        ws = mono.Workspace(
            device=self.device,
            dtype=self.dtype,
            layers=1,
            tp=self.tp,
            group=dist.group.WORLD if self.tp == 2 else None,
            groups=groups,
        )
        kres, vres = self.keys.clone(), self.values.clone()
        ref, rref, ids = self.ref(kres, vres)
        torch.xpu.synchronize()
        if self.tp == 2:
            dist.barrier()

        out, res = self.step(ws, table)
        torch.xpu.synchronize()
        ws.check_health()
        if capture:
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                out, res = self.step(ws, table)
            for _ in range(2):
                graph.replay()
            torch.xpu.synchronize()
            ws.check_health()

        tol = 0.025 if self.dtype == torch.bfloat16 else 0.005
        torch.testing.assert_close(res, rref, atol=tol, rtol=tol)
        torch.testing.assert_close(self.keys, kres, atol=tol, rtol=tol)
        torch.testing.assert_close(self.values, vres, atol=tol, rtol=tol)
        if ids is not None:
            actual_ids = ws.scratch[7][: self.rows].sort(dim=-1).values
            torch.testing.assert_close(actual_ids, ids.sort(dim=-1).values.int())
        torch.testing.assert_close(out, ref, atol=tol, rtol=tol)
        bank = self.rows if self.tp == 2 else 0
        calls = 3 if capture else 1
        assert ws.counters[0, bank, EPOCH].item() == calls
        if split:
            keys = self.keys.clone()
            values = self.values.clone()
            for mode in (1, 2):
                split_out, split_res = self.step(ws, table, mode)
                torch.xpu.synchronize()
                ws.check_health()
                assert torch.equal(split_out, out)
                assert torch.equal(split_res, res)
                assert torch.equal(self.keys, keys)
                assert torch.equal(self.values, values)
            assert ws.counters[0, bank, EPOCH].item() == (
                calls if self.tp == 2 else calls + 4
            )
        if self.tp == 2:
            pair = [torch.empty_like(ref.cpu()) for _ in range(2)]
            dist.all_gather(pair, out.cpu().contiguous())
            assert torch.equal(*pair)
        return out, res


@pytest.mark.parametrize("groups", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
def test_dense(rows, dtype, groups):
    Inputs(1, 0, dtype, dense=True, channelwise=rows != 1, rows=rows).compare(groups)


@pytest.mark.parametrize("groups", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
def test_moe(rows, dtype, groups):
    Inputs(1, 0, dtype, dense=False, channelwise=True, rows=rows).compare(
        groups, capture=rows == 4, split=True
    )


@pytest.mark.parametrize("groups", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_verify_causal(dtype, groups):
    rows = 8
    Inputs(1, 0, dtype, dense=False, channelwise=True, rows=rows, verify=True).compare(
        groups, split=True
    )


def check_mailbox_wrap(rank):
    data = Inputs(2, rank, torch.bfloat16, dense=True, channelwise=True, rows=4)
    table = mono.LayerTable([data.weights], [True], 2)
    workspace = mono.Workspace(
        device=data.device,
        dtype=data.dtype,
        layers=1,
        tp=2,
        group=dist.group.WORLD,
    )
    # Populate tag 1 in the four-row bank. Advance the one-row bank through
    # wrap and change the wide input before reusing its mailbox.
    data.step(workspace, table)
    narrow = workspace.counters[0, 1]
    narrow[:EPOCH:16].fill_(0xFFFF * workspace.groups)
    narrow[EPOCH] = 0xFFFF
    data.step(workspace, table, rows=1)
    data.hidden[1:].mul_(0.5)
    out, residual = data.step(workspace, table)
    ref, ref_residual = data.step(workspace, table, mode=2)
    torch.xpu.synchronize()
    workspace.check_health()
    assert workspace.counters[0, 1, EPOCH].item() == 0x10000
    assert workspace.counters[0, 4, EPOCH].item() == 2
    assert torch.equal(out, ref)
    assert torch.equal(residual, ref_residual)


def set_long_shared_history(data, start):
    data.positions += start - 62
    data.lengths = (data.positions + 1).int()
    pages = (start + data.rows + 63) // 64
    data.blocks = torch.arange(1, pages + 1, device=data.device, dtype=torch.int32)[
        None
    ]
    data.slots = data.positions + 64
    data.keys = data.rand((pages + 1, 64, data.kvheads, HEAD)) * 0.03
    data.values = data.rand(data.keys.shape) * 0.03
    phase = data.rand((start + data.rows, HEAD // 2)).float() * 0.1
    data.rope = torch.cat((phase.cos(), phase.sin()), dim=-1).to(data.dtype)
    for index in (8, 10):
        weight = data.weights[index]
        data.weights[index] = weight.transpose(-1, -2).contiguous().transpose(-1, -2)


@pytest.mark.parametrize("start", [32763, 32769])
def test_verify_shared_history(start):
    data = Inputs(
        1, 0, torch.bfloat16, dense=False, channelwise=True, rows=8, verify=True
    )
    set_long_shared_history(data, start)
    data.compare(groups=32, capture=True, split=True)


def check_independent_long_history(tp, rank, rows):
    data = Inputs(tp, rank, torch.bfloat16, False, True, rows=rows)
    # Different causal lengths on either side of the 32K page boundary.
    data.positions = 32764 + 3 * torch.arange(rows, device=data.device)
    data.lengths = (data.positions + 1).int()
    context = 32764 + 3 * (rows - 1) + 1
    pages = (context + 63) // 64
    data.blocks = torch.arange(
        1, rows * pages + 1, device=data.device, dtype=torch.int32
    ).view(rows, pages)
    row = torch.arange(rows, device=data.device)
    data.slots = (
        data.blocks[row, data.positions // 64].long() * 64 + data.positions % 64
    )
    data.keys = data.rand((rows * pages + 1, 64, data.kvheads, HEAD)) * 0.03
    data.values = data.rand(data.keys.shape) * 0.03
    phase = data.rand((context, HEAD // 2)).float() * 0.1
    data.rope = torch.cat((phase.cos(), phase.sin()), dim=-1).to(data.dtype)
    for index in (8, 10):
        data.weights[index] = (
            data.weights[index].transpose(-1, -2).contiguous().transpose(-1, -2)
        )
    data.compare(groups=32, capture=True)


@pytest.mark.parametrize("rows", [3, 7])
def test_decode_independent_long_history(rows):
    check_independent_long_history(1, 0, rows)


def test_partial_reduce_bounds():
    data = Inputs(1, 0, torch.bfloat16, dense=True, channelwise=True)
    if not mono.work_groups(data.device):
        pytest.skip("requires a supported Xe2 grid")
    table = mono.LayerTable([data.weights], [True], 1)
    workspace = mono.Workspace(device=data.device, dtype=data.dtype, layers=1, tp=1)
    output = data.hidden.new_zeros((2, 1, HIDDEN))
    residual = torch.zeros_like(output)
    guarded = torch.full((8, HIDDEN), -1.0, dtype=torch.float32, device=data.device)
    for part in (mono._Part.ATTENTION_PARTIAL, mono._Part.FEED_FORWARD_PARTIAL):
        torch.ops.sgl_kernel.minicpm5_decode_step(
            workspace.groups,
            1,
            0,
            1,
            table.descriptors,
            data.hidden,
            data.residual,
            data.positions,
            data.slots,
            data.blocks,
            data.lengths,
            data.rope,
            [data.keys],
            [data.values],
            workspace.scratch,
            workspace.counters,
            workspace.health,
            workspace.host_health,
            None,
            None,
            output,
            residual,
            part,
            guarded[:1],
        )
        torch.xpu.synchronize()
        workspace.check_health()
        assert torch.all(guarded[1:] == -1)


def test_tp2():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip(
            "run on two cards with torchrun --standalone --nproc-per-node=2 -m pytest"
        )
    rank = int(os.environ["LOCAL_RANK"])
    torch.xpu.set_device(rank)
    dist.init_process_group("gloo")
    try:
        supported = torch.tensor(
            mono.work_groups(torch.device("xpu", rank)), device="cpu"
        )
        dist.all_reduce(supported, op=dist.ReduceOp.MIN)
        max_groups = supported.item()
        if not max_groups:
            pytest.skip("requires a supported Xe2 grid on both ranks")
        for groups in (16, 32):
            if groups > max_groups:
                continue
            for dtype in (torch.float16, torch.bfloat16):
                for rows, dense, verify in (
                    (1, True, False),
                    (1, False, False),
                    (4, False, False),
                    (8, False, True),
                ):
                    Inputs(
                        2, rank, dtype, dense, True, verify=verify, rows=rows
                    ).compare(groups, split=True)

        if max_groups == 32:
            for start in (32763, 32769):
                data = Inputs(
                    2,
                    rank,
                    torch.bfloat16,
                    dense=False,
                    channelwise=True,
                    rows=8,
                    verify=True,
                )
                set_long_shared_history(data, start)
                data.compare(32, capture=True, split=True)
            for rows in (3, 4, 6, 8):
                check_independent_long_history(2, rank, rows)

        out, residual = Inputs(
            2, rank, torch.bfloat16, dense=False, channelwise=True, rows=4
        ).compare(max_groups)
        swapped_out, swapped_residual = Inputs(
            2,
            rank,
            torch.bfloat16,
            dense=False,
            channelwise=True,
            rows=4,
            shard_rank=1 - rank,
        ).compare(max_groups)
        assert torch.equal(out, swapped_out)
        assert torch.equal(residual, swapped_residual)

        check_mailbox_wrap(rank)

        # Both ranks report sticky health after invalid KV metadata on rank 1.
        for dtype in (torch.float16, torch.bfloat16):
            data = Inputs(2, rank, dtype, dense=True, channelwise=True)
            table = mono.LayerTable([data.weights], [True], 2)
            workspace = mono.Workspace(
                device=data.device,
                dtype=data.dtype,
                layers=1,
                tp=2,
                group=dist.group.WORLD,
                deadline_s=0.15,
            )
            if rank == 1:
                if dtype == torch.bfloat16:
                    data.positions.fill_(data.rope.size(0))
                else:
                    data.slots.fill_(data.keys.numel())
                keys, values = data.keys.clone(), data.values.clone()
            torch.xpu.synchronize()
            dist.barrier()
            data.step(workspace, table)
            torch.xpu.synchronize()
            with pytest.raises(RuntimeError, match="failed a device hand-off"):
                workspace.check_health()
            if rank == 1:
                assert torch.equal(data.keys, keys)
                assert torch.equal(data.values, values)
            dist.barrier()
    finally:
        dist.destroy_process_group()
