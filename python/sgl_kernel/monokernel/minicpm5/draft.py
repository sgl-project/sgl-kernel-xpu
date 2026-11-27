# SPDX-License-Identifier: Apache-2.0
"""Seven-query DSpark draft with BF16 activations on a resident Xe2 grid."""

import os

import torch
from sgl_kernel.minicpm5_decode import LayerTable, Workspace, work_groups
from sgl_kernel.monokernel.minicpm5.runtime import _stop_on_failed_replay

from sglang.kernels.plugin_kernels.amd.spec import (
    MONOKERNEL_API_VERSION,
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.runtime_context import get_model, get_schedule, get_spec

_ROWS = 7


def _layer_weights(layer):
    attention = layer.self_attn
    norms = [
        layer.input_layernorm.weight,
        layer.post_attention_layernorm.weight,
        attention.q_norm.weight,
        attention.k_norm.weight,
    ]
    if any(w.dtype != torch.bfloat16 or not w.is_contiguous() for w in norms):
        return None
    linears = (
        attention.qkv_proj,
        attention.o_proj,
        layer.mlp.gate_up_proj,
        layer.mlp.down_proj,
    )
    if all(
        w.weight.dtype == torch.bfloat16 and w.weight.is_contiguous() for w in linears
    ):
        return norms + [w.weight for w in linears]

    weights = norms
    for linear in linears:
        weight = linear.weight
        scale = getattr(linear, "weight_scale", None)
        if (
            weight.dtype != torch.float8_e4m3fn
            or weight.ndim != 2
            or weight.stride() != (1, weight.shape[0])
            or scale is None
            or scale.dtype != torch.float32
            or scale.numel() not in (1, weight.shape[1])
            or not scale.is_contiguous()
        ):
            return None
        weights.extend((weight, scale))
    return weights


def _model_reasons(model, hf_config, copies):
    if not hasattr(torch.ops.sgl_kernel, "minicpm5_draft_layer_desc"):
        return ["the kernel build has no BF16 draft layer"]
    if model.gamma != _ROWS or not model.sample_from_anchor:
        return ["the BF16 draft kernel serves seven queries with the anchor included"]
    if get_spec().speculative_draft_attention_backend not in (None, "intel_xpu"):
        return ["the BF16 draft kernel requires the intel_xpu attention backend"]
    if get_schedule().page_size not in (64, 128):
        return ["the BF16 draft kernel requires a 64- or 128-token KV page"]
    kv_dtype = get_spec().speculative_draft_kv_cache_dtype or get_model().kv_cache_dtype
    if kv_dtype not in ("auto", "bf16", "bfloat16"):
        return ["the BF16 draft kernel requires a BF16 KV cache"]
    if get_tp_group().world_size not in (1, 2):
        return ["the BF16 draft kernel requires TP1 or TP2"]
    device = next(model.parameters()).device
    if work_groups(device) != 32 or os.getenv("SGLANG_XPU_MONO_WORK_GROUPS") == "16":
        return ["the BF16 draft layer requires 32 resident work-groups"]
    for layer in model.layers:
        attention = layer.self_attn
        if (
            layer.attention_conv is not None
            or layer.mlp_conv is not None
            or not attention.use_table_qk_norm_rope
            or attention.sliding_window_size >= 0
            or attention.attn_type != AttentionType.ENCODER_ONLY
            or not attention.save_kv_cache
            or attention.attention_sink_bias is not None
            or attention.v_scale is not None
        ):
            return [
                "the BF16 draft kernel requires full bidirectional DFlash attention"
            ]
        if _layer_weights(layer) is None:
            return [
                "the draft layer requires BF16 norms and either BF16 or scaled FP8 projections"
            ]
    return []


class MiniCPM5DraftDecode:
    def __init__(self, model, hf_config, context):
        self.model = model
        layers = tuple(model.layers)
        device = layers[0].input_layernorm.weight.device
        tp_group = get_tp_group()
        tp = tp_group.world_size
        self.workspace = Workspace(
            device=device,
            dtype=torch.bfloat16,
            layers=len(layers),
            tp=tp,
            group=tp_group.cpu_group if tp == 2 else None,
            deadline_s=min(context.deadline_s, 2.0),
        )
        self.table = LayerTable(
            [_layer_weights(layer) for layer in layers],
            [True] * len(layers),
            tp,
            draft=True,
        )
        self.rope = (
            layers[0]
            .self_attn.rotary_emb.cos_sin_cache.to(device=device, dtype=torch.bfloat16)
            .contiguous()
        )
        self.cache = None
        _stop_on_failed_replay(self.workspace)

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        self.workspace.check_health()
        hidden = (
            self.model.forward_embed(input_ids)
            if input_embeds is None
            else input_embeds
        )
        backend = get_attn_backend()
        metadata = backend.forward_metadata
        positions = positions.to(torch.int64).contiguous()
        slots = forward_batch.out_cache_loc.to(torch.int64).contiguous()
        slots = torch.where(slots > 0, slots, -1)
        lengths = metadata.cache_seqlens_int32[:1]
        blocks = metadata.page_table[:1].to(
            dtype=torch.int32, memory_format=torch.contiguous_format
        )
        if self.cache is None:
            pool = get_token_to_kv_pool()
            shape = (-1, backend.page_size, 2 // self.workspace.tp, 128)
            keys, values = zip(
                *(
                    pool.get_kv_buffer(layer.self_attn.attn.layer_id)
                    for layer in self.model.layers
                )
            )
            self.cache = (
                [key.view(shape) for key in keys],
                [value.view(shape) for value in values],
            )
        keys, values = self.cache
        output, residual = self.workspace.decode_step(
            self.table,
            0,
            len(self.table.weights),
            hidden=hidden,
            residual=None,
            positions=positions,
            slots=slots,
            blocks=blocks,
            lengths=lengths,
            rope=self.rope,
            keys=keys,
            values=values,
        )
        output, _ = self.model.norm(output, residual)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=output)


SPEC = MonoSpec(
    name="MiniCPM5 Xe2 DSpark draft",
    devices={"xe2": DeviceLimits((_ROWS,), (1, 2), 32)},
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"intel_xpu"}),
        hf_config={
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "attention_bias": False,
        },
    ),
    numerics=Numerics(NumericsKind.TOLERANCE, {"target_verify.hidden.cosine": 0.99}),
    model_disable_reasons=_model_reasons,
    build=MiniCPM5DraftDecode,
    api_version=MONOKERNEL_API_VERSION,
    tokens_per_request=lambda model: _ROWS,
    allow_model_fallback=True,
)
