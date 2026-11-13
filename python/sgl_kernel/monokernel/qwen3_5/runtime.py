"""Runtime of the Qwen3.5 / Qwen3.6 MoE XPU monokernels: SGLang's loaded weights
and pools -> mono kernel launches. ``MonoQwen35.forward`` runs a routed decode or
target-verify step in place of the model's forward: the embedding, one mono
kernel per decoder layer (mixer, MoE with the shared expert and both TP
all-reduces in the kernel), the final norm and the model's own logits
processor. A verify step leaves the committed GatedDeltaNet state alone and
writes the state after each token to SGLang's verify scratch, as the standard
target-verify path does."""

import logging
from typing import Any

import torch
from sgl_kernel import mono_decode

from .. import memory
from ..failure import FailureWatch, deadline_s, schedule_fault
from ..peer import PeerRegion, tp_agree

logger = logging.getLogger(__name__)


def _require(ok, message: str):
    """Raises (an assert would vanish under ``python -O``, leaving the outputs
    uninitialised) when a launch that the plan promised is refused."""
    if not ok:
        raise RuntimeError(message)


def _check_call(get_embedding, pp_proxy_tensors, extra: dict):
    """The model forward's other arguments: ``input_embeds`` is served, the
    rest is refused instead of ignored."""
    unsupported = [
        name
        for name, given in (
            ("get_embedding", bool(get_embedding)),
            ("pp_proxy_tensors", pp_proxy_tensors is not None),
            *((k, True) for k in extra),
        )
        if given
    ]
    _require(
        not unsupported,
        f"the monokernel forward does not serve {unsupported} (the model's own forward does)",
    )


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
        return t + [
            mixer.qkv_proj.weight,
            mixer.qkv_proj.weight_scale_inv,
            mixer.q_norm.weight,
            mixer.k_norm.weight,
        ]
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
        dev = next(model.parameters()).device
        self.budget = memory.apply(dev)  # before SGLang sizes its pools
        self.deadline = deadline_s(context.deadline_s)
        mono_decode.set_deadline(self.deadline)
        names = [f"layer{i}" for i in range(len(self.layers))]
        self.peer = PeerRegion(names, self.rank, group.cpu_group)
        self.ids = [7000 + i for i in range(len(self.layers))]
        ok = True
        try:
            for i, layer in enumerate(self.layers):
                own, peer = self.peer.mailbox(names[i])
                torch.ops.sgl_kernel.mono_register_layer(
                    self.ids[i], self.attn[i], own, peer, layer_weights(layer)
                )
        except Exception:
            ok = False
            logger.exception("XPU monokernel: registering the layers failed")
        if not tp_agree(ok, group.cpu_group):
            raise RuntimeError(
                "XPU monokernel: a rank could not register the layers (see its log)"
            )
        self.bound = None  # (pool identities) the caches were bound for
        self.mrope = bool(getattr(model, "is_mrope_enabled", False))
        self.pressure = memory.PressureWatch(dev, self.budget)
        addr = {}
        for i, lid in enumerate(self.ids):
            addr[
                torch.ops.sgl_kernel.mono_layer_counters(lid).data_ptr() & 0xFFFFFFFF
            ] = f"layer {i}"
        self.watch = FailureWatch(
            dev.index, self.rank, self.deadline, lambda a: addr.get(a & 0xFFFFFFFF)
        )
        schedule_fault(self.rank, dev.index, self.ids[0])
        logger.info(
            "XPU monokernel: %d layers (%d full attention), %s, wait deadline %g s",
            len(self.layers),
            sum(self.attn),
            next(model.parameters()).dtype,
            self.deadline,
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
                scratch = getattr(state, "intermediate_ssm", None)
                if scratch is not None:  # speculative decoding: the verify scratch
                    caches += [scratch, state.intermediate_conv_window[0]]
            key.append(tuple(c.data_ptr() for c in caches))
            if self.bound is None or self.bound[i] != key[-1]:
                torch.ops.sgl_kernel.mono_bind_caches(
                    self.ids[i], caches, int(pool.page_size)
                )
        self.bound = key

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
        **kwargs,
    ):
        from sglang.srt.model_executor.forward_context import get_attn_backend

        _check_call(get_embedding, pp_proxy_tensors, kwargs)
        backend = get_attn_backend()
        if not torch.xpu.is_current_stream_capturing():
            self.watch.check()  # a failed earlier step (eager; replayed graphs: the watch thread)
        self._bind(backend)
        rows = plan.key.rows
        tokens = plan.key.tokens_per_request
        verify = plan.key.forward_mode == "target_verify"
        fmd = backend.full_attn_backend.forward_metadata
        lmd = backend.linear_attn_backend.forward_metadata
        if self.mrope and getattr(forward_batch, "mrope_positions", None) is not None:
            pos = forward_batch.mrope_positions[0]  # decode: the three axes agree
        else:
            pos = positions
        m_attn = (fmd.page_table, fmd.cache_seqlens_int32, forward_batch.out_cache_loc)
        m_gdn = (lmd.mamba_cache_indices, None, None)
        if input_embeds is None:
            hidden = self.lm.embed_tokens(input_ids[:rows])
        else:
            hidden = input_embeds[:rows]
        residual = torch.zeros_like(hidden)
        ops = torch.ops.sgl_kernel
        for i in range(len(self.layers)):
            out, res = torch.empty_like(hidden), torch.empty_like(hidden)
            m = m_attn if self.attn[i] else m_gdn
            if verify:
                # the spec covers only the widths the verify kernels have
                ok = ops.mono_decode_layer_verify(
                    self.ids[i],
                    tokens,
                    hidden,
                    residual,
                    out,
                    res,
                    pos,
                    m[0],
                    m[1],
                    m[2],
                )
                _require(ok, f"no verify kernel for {rows} rows x {tokens} tokens")
            else:
                ops.mono_decode_layer(
                    self.ids[i], hidden, residual, out, res, pos, m[0], m[1], m[2]
                )
            hidden, residual = out, res
        hidden, _ = self.lm.norm(hidden, residual)
        return self.model.logits_processor(
            input_ids, hidden, self.model.lm_head, forward_batch
        )


class MonoQwen35MTP:
    """The built monokernels of the MTP drafter on this rank: a draft decode
    step (one token a request) runs the prologue kernel (the two Gemma norms
    and the fc), the draft decoder layer's attention kernel, then the model's
    final norm and logits processor. The draft extend step (the target's
    verified positions, tokens a request) runs the same prologue on all its rows
    and the layer's verify kernel (it writes the drafter's K / V of every row and
    attends causally). The target's runtime owns the fail-stop watch and the
    memory budget of the device."""

    PROLOGUE, LAYER = 7100, 7101  # kernel ids (the target's are 7000 + layer)

    def __init__(self, model: torch.nn.Module, hf_config: Any, context: Any):
        from sglang.srt.runtime_context import get_parallel

        self.model = model
        self.layer = layer_modules(model)[0]
        group = get_parallel().tp_group
        self.rank = group.rank_in_group
        mono_decode.set_deadline(deadline_s(context.deadline_s))
        names = ["mtp_prologue", "mtp_layer"]
        self.peer = PeerRegion(names, self.rank, group.cpu_group)
        ok = True
        try:
            own, peer = self.peer.mailbox(names[0])
            prologue = [
                model.fc.weight,
                model.pre_fc_norm_embedding.weight,
                model.pre_fc_norm_hidden.weight,
            ]
            torch.ops.sgl_kernel.mono_register_mtp(
                self.PROLOGUE, own, peer, prologue, self.rank
            )
            own, peer = self.peer.mailbox(names[1])
            torch.ops.sgl_kernel.mono_register_layer(
                self.LAYER, True, own, peer, layer_weights(self.layer)
            )
        except Exception:
            ok = False
            logger.exception("XPU monokernel: registering the MTP drafter failed")
        if not tp_agree(ok, group.cpu_group):
            raise RuntimeError(
                "XPU monokernel: a rank could not register the MTP drafter (see its log)"
            )
        self.bound = None
        self.seen = set()  # the (mode, rows) steps this forward has run, logged once
        logger.info(
            "XPU monokernel: MTP drafter (prologue + attention layer), %s",
            model.fc.weight.dtype,
        )

    def _bind(self, backend: Any) -> None:
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        pool = get_token_to_kv_pool()
        lid = self.layer.attn.layer_id
        caches = [pool.get_key_buffer(lid), pool.get_value_buffer(lid)]
        key = tuple(c.data_ptr() for c in caches)
        if key != self.bound:
            torch.ops.sgl_kernel.mono_bind_caches(
                self.LAYER, caches, int(pool.page_size)
            )
            self.bound = key

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
        **kwargs,
    ):
        from sglang.srt.model_executor.forward_context import get_attn_backend

        _check_call(get_embedding, pp_proxy_tensors, kwargs)
        backend = get_attn_backend()
        self._bind(backend)
        rows = plan.key.rows
        if (plan.key.forward_mode, rows) not in self.seen:  # captured or eager
            self.seen.add((plan.key.forward_mode, rows))
            logger.info(
                "XPU monokernel: MTP drafter runs a %s step of %d rows (%d x %d)",
                plan.key.forward_mode,
                rows,
                plan.key.batch_size,
                plan.key.tokens_per_request,
            )
        fmd = getattr(backend, "full_attn_backend", backend).forward_metadata
        lm = language_model(self.model)
        if input_embeds is None:
            embed = lm.embed_tokens(input_ids[:rows])
        else:
            embed = input_embeds[:rows]
        x = torch.empty_like(embed)
        ops = torch.ops.sgl_kernel
        ok = ops.mono_mtp_prologue(
            self.PROLOGUE, embed, forward_batch.spec_info.hidden_states, x
        )
        _require(ok, f"no MTP prologue kernel for {rows} rows")
        out, res = torch.empty_like(x), torch.empty_like(x)
        if plan.key.forward_mode == "draft_extend":
            # seq_lens already counts the extend rows, as a verify step's do
            ok = ops.mono_decode_layer_verify(
                self.LAYER,
                plan.key.tokens_per_request,
                x,
                torch.zeros_like(x),
                out,
                res,
                positions,
                fmd.page_table,
                fmd.cache_seqlens_int32,
                forward_batch.out_cache_loc,
            )
            _require(
                ok,
                f"no verify kernel for {rows} rows x "
                f"{plan.key.tokens_per_request} tokens (draft extend)",
            )
        else:
            ops.mono_decode_layer(
                self.LAYER,
                x,
                torch.zeros_like(x),
                out,
                res,
                positions,
                fmd.page_table,
                fmd.cache_seqlens_int32,
                forward_batch.out_cache_loc,
            )
        hidden, _ = lm.norm(out, res)
        return self.model.logits_processor(
            input_ids, hidden, self.model.lm_head, forward_batch
        )
