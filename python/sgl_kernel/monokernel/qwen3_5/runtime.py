"""Runtime of the Qwen3.5 / Qwen3.6 MoE XPU monokernels: SGLang's loaded weights
and pools -> mono kernel launches. ``MonoQwen35.forward`` runs a routed decode
step in place of the model's forward: the embedding, one mono kernel per decoder
layer (mixer, MoE with the shared expert and both TP all-reduces in the kernel),
the final norm and the model's own logits processor."""

import logging
from typing import Any

import torch

from sgl_kernel import mono_decode

from ..peer import PeerRegion, tp_agree

logger = logging.getLogger(__name__)


def language_model(model: torch.nn.Module) -> torch.nn.Module:
    """The decoder stack (embed_tokens, layers, norm)."""
    return model.model


def layer_modules(model: torch.nn.Module) -> list:
    return list(language_model(model).layers)


def is_attention(layer: torch.nn.Module) -> bool:
    return hasattr(layer, "qkv_proj")


def layer_weights(layer: torch.nn.Module) -> list:
    """The kernel's weight list (mono_register_layer order): norms, router,
    shared gate, the mixer's output projection, routed and shared experts, then
    the mixer."""
    mlp = layer.mlp
    experts = mlp.experts
    attn = is_attention(layer)
    mixer = layer if attn else layer.linear_attn
    out = mixer.o_proj if attn else mixer.out_proj
    t = [
        layer.input_layernorm.weight,
        layer.post_attention_layernorm.weight,
        mlp.gate.weight,
        mlp.shared_expert_gate.weight,
        out.weight,
        out.weight_scale_inv,
        experts.w13_weight,
        experts.w13_weight_scale_inv,
        experts.w2_weight,
        experts.w2_weight_scale_inv,
        mlp.shared_expert.gate_up_proj.weight,
        mlp.shared_expert.gate_up_proj.weight_scale_inv,
        mlp.shared_expert.down_proj.weight,
        mlp.shared_expert.down_proj.weight_scale_inv,
    ]
    if attn:
        return t + [mixer.qkv_proj.weight, mixer.qkv_proj.weight_scale_inv, mixer.q_norm.weight, mixer.k_norm.weight]
    return t + [
        mixer.in_proj_qkvz.weight,
        mixer.in_proj_qkvz.weight_scale_inv,
        mixer.in_proj_ba.weight,
        mixer.conv1d.weight,
        mixer.A_log,
        mixer.dt_bias,
        mixer.norm.weight,
    ]


class MonoQwen35:
    """The built monokernels of one model on this rank (BuiltMonokernels)."""

    def __init__(self, model: torch.nn.Module, hf_config: Any, context: Any):
        from sglang.srt.runtime_context import get_parallel

        self.model = model
        self.lm = language_model(model)
        self.layers = layer_modules(model)
        self.attn = [is_attention(layer) for layer in self.layers]
        group = get_parallel().tp_group
        self.rank = group.rank_in_group
        mono_decode.set_deadline(context.deadline_s)
        names = [f"layer{i}" for i in range(len(self.layers))]
        self.peer = PeerRegion(names, self.rank, group.cpu_group)
        self.ids = [7000 + i for i in range(len(self.layers))]
        ok = True
        try:
            for i, layer in enumerate(self.layers):
                own, peer = self.peer.mailbox(names[i])
                torch.ops.sgl_kernel.mono_register_layer(self.ids[i], self.attn[i], own, peer, layer_weights(layer))
        except Exception:
            ok = False
            logger.exception("XPU monokernel: registering the layers failed")
        if not tp_agree(ok, group.cpu_group):
            raise RuntimeError("XPU monokernel: a rank could not register the layers (see its log)")
        self.bound = None  # (pool identities) the caches were bound for
        self.mrope = bool(getattr(model, "is_mrope_enabled", False))
        logger.info(
            "XPU monokernel: %d layers (%d full attention), %s",
            len(self.layers),
            sum(self.attn),
            next(model.parameters()).dtype,
        )

    def _bind(self, backend: Any) -> None:
        """Binds the pools' K / V buffers and GDN states (again if they moved:
        only pointers, no allocation, safe during capture)."""
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        pool = get_token_to_kv_pool()
        lin = backend.linear_attn_backend
        key = []
        for i, layer in enumerate(self.layers):
            if self.attn[i]:
                lid = layer.attn.layer_id
                caches = [pool.get_key_buffer(lid), pool.get_value_buffer(lid)]
            else:
                state = lin.state_layer_cache(layer.linear_attn.layer_id)
                caches = [state.conv[0], state.temporal]
            key.append(tuple(c.data_ptr() for c in caches))
            if self.bound is None or self.bound[i] != key[-1]:
                torch.ops.sgl_kernel.mono_bind_caches(self.ids[i], caches, int(pool.page_size))
        self.bound = key

    def forward(self, plan, input_ids, positions, forward_batch, *args, **kwargs):
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        self._bind(backend)
        rows = plan.key.rows
        fmd = backend.full_attn_backend.forward_metadata
        lmd = backend.linear_attn_backend.forward_metadata
        if self.mrope and getattr(forward_batch, "mrope_positions", None) is not None:
            pos = forward_batch.mrope_positions[0]  # decode: the three axes agree
        else:
            pos = positions
        m_attn = (fmd.page_table, fmd.cache_seqlens_int32, forward_batch.out_cache_loc)
        m_gdn = (lmd.mamba_cache_indices, None, None)
        hidden = self.lm.embed_tokens(input_ids[:rows])
        residual = torch.zeros_like(hidden)
        launch = torch.ops.sgl_kernel.mono_decode_layer
        for i in range(len(self.layers)):
            out, res = torch.empty_like(hidden), torch.empty_like(hidden)
            m = m_attn if self.attn[i] else m_gdn
            launch(self.ids[i], hidden, residual, out, res, pos, m[0], m[1], m[2])
            hidden, residual = out, res
        hidden, _ = self.lm.norm(hidden, residual)
        return self.model.logits_processor(input_ids, hidden, self.model.lm_head, forward_batch)
