# SPDX-License-Identifier: Apache-2.0
"""Glue from SGLang's graph metadata and FP8 weights to MiniCPM5 decode."""

import logging
import os
import signal
import threading
import weakref

import torch
from sgl_kernel.minicpm5_decode import LayerTable, Workspace

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)

logger = logging.getLogger(__name__)


def _linear_weights(linear):
    return [linear.weight, linear.weight_scale]


def _layer_weights(layer):
    attn, mlp = layer.self_attn, layer.mlp
    weights = [layer.input_layernorm.weight, layer.post_attention_layernorm.weight]
    weights += _linear_weights(attn.qkv_proj) + _linear_weights(attn.o_proj)
    if not layer.is_layer_sparse:
        return (
            weights + _linear_weights(mlp.gate_up_proj) + _linear_weights(mlp.down_proj)
        )
    experts, shared = mlp.experts, mlp.shared_experts
    weights += [
        mlp.gate.weight,
        mlp.gate.e_score_correction_bias,
        experts.w13_weight,
        experts.w13_weight_scale,
        experts.w2_weight,
        experts.w2_weight_scale,
    ]
    return (
        weights
        + _linear_weights(shared.gate_up_proj)
        + _linear_weights(shared.down_proj)
    )


def _stop_on_failed_replay(workspace):
    # Replay executes without reentering Python. Watch the pinned health copy.
    stop = threading.Event()
    weakref.finalize(workspace, stop.set)
    health = workspace.host_health

    def watch():
        while not stop.wait(0.05):
            if health[0].item():
                logger.critical(
                    "MiniCPM5 mono decode failed a work-group hand-off or KV metadata check"
                )
                os.kill(os.getppid(), signal.SIGQUIT)

    threading.Thread(target=watch, daemon=True, name="minicpm5-mono-health").start()


class MiniCPM5Decode:
    def __init__(self, model, hf_config, context):
        self.model = model
        layers = tuple(model.model.layers)
        device, dtype = (
            layers[0].input_layernorm.weight.device,
            layers[0].input_layernorm.weight.dtype,
        )
        tp_group = get_tp_group()
        tp = tp_group.world_size
        self.workspace = Workspace(
            device=device,
            dtype=dtype,
            layers=len(layers),
            tp=tp,
            group=tp_group.cpu_group if tp == 2 else None,
            deadline_s=min(context.deadline_s, 2.0),
            groups=(
                int(groups)
                if (groups := os.getenv("SGLANG_XPU_MONO_WORK_GROUPS"))
                else None
            ),
        )
        if (
            self.workspace.groups == 32
            and dtype == torch.bfloat16
            and not get_moe_runner_backend().is_triton()
        ):
            for layer in layers:
                if not layer.is_layer_sparse:
                    continue
                experts = layer.mlp.experts
                for parameter in (experts.w13_weight, experts.w2_weight):
                    if (
                        parameter.dtype == torch.float8_e4m3fn
                        and parameter.is_contiguous()
                    ):
                        # Native W8A16 and mono read the same [E, N, K] view.
                        packed = parameter.detach().transpose(-1, -2).contiguous()
                        parameter.data = packed.transpose(-1, -2)
        self.table = LayerTable(
            [_layer_weights(layer) for layer in layers],
            [not layer.is_layer_sparse for layer in layers],
            tp,
        )
        self.rope = (
            layers[0]
            .self_attn.rotary_emb.cos_sin_cache.to(dtype=dtype, device=device)
            .contiguous()
        )
        self.cache = None
        logger.info(
            "MiniCPM5 XPU mono on %s, TP%d, %d work-groups",
            device,
            tp,
            self.workspace.groups,
        )
        _stop_on_failed_replay(self.workspace)

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        self.workspace.check_health()
        model = self.model
        if input_embeds is None:
            hidden = model.model.embed_tokens(input_ids)
        else:
            hidden = input_embeds
        backend = get_attn_backend()
        metadata = backend.forward_metadata
        width = plan.key.tokens_per_request
        blocks = metadata.page_table
        if width > 1 and plan.key.batch_size > 1:
            blocks = blocks.repeat_interleave(width, dim=0)
        blocks = blocks.to(dtype=torch.int32, memory_format=torch.contiguous_format)
        slots = forward_batch.out_cache_loc.to(torch.int64).contiguous()
        slots = torch.where(slots > 0, slots, -1)
        positions = positions.to(torch.int64).contiguous()
        lengths = (positions + 1).to(torch.int32)
        if self.cache is None:
            pool = get_token_to_kv_pool()
            page = backend.page_size
            keys, values = zip(
                *(pool.get_kv_buffer(i) for i in range(len(self.table.weights)))
            )
            shape = (-1, page, 2 // self.workspace.tp, 128)
            self.cache = (
                [key.view(shape) for key in keys],
                [value.view(shape) for value in values],
            )
        keys, values = self.cache
        captures = set(model.model.layers_to_capture)
        aux = AuxHiddenStateList()
        residual = None
        start = 0
        for end in sorted(captures | {len(self.table.weights)}):
            if end > start:
                hidden, residual = self.workspace.decode_step(
                    self.table,
                    start,
                    end,
                    hidden=hidden,
                    residual=residual,
                    positions=positions,
                    slots=slots,
                    blocks=blocks,
                    lengths=lengths,
                    rope=self.rope,
                    keys=keys,
                    values=values,
                )
            if end in captures:
                aux.capture(
                    hidden if residual is None else hidden + residual,
                    owned=residual is not None,
                )
            start = end
        hidden, _ = model.model.norm(hidden, residual)
        return model.logits_processor(
            input_ids, hidden, model.lm_head, forward_batch, aux if aux else None
        )
