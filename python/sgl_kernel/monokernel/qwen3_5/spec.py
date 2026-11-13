"""The XPU monokernel spec of Qwen3.5 / Qwen3.6 MoE 35B-A3B (block-FP8, TP2 on
two Xe2 GPUs with 32 Xe cores): decode steps of 1, 2, 3, 4, 6 or 8 rows, and
MTP target-verify steps of those widths (EAGLE / NEXTN, top-1 chains of 2, 3,
4, 6 or 8 tokens a request), run one mono kernel per decoder layer, and the
MTP drafter's draft decode steps run a prologue kernel and an attention layer
kernel (MTP_SPEC); every other step runs the model's own forward."""

from typing import Any

import torch
from sgl_kernel import mono_decode

from sglang.kernels.plugin_kernels.xpu.spec import (
    MONOKERNEL_API_VERSION,
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

# The shapes the kernels are written for (the text config of the model; up to
# MAX_LAYERS layers, each a GatedDeltaNet or full-attention layer)
TEXT_CONFIG = {
    "hidden_size": 2048,
    "full_attention_interval": 4,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-06,
    "attn_output_gate": True,
    "hidden_act": "silu",
}
ROPE_THETA = 10_000_000
# the kernel ids of the target are 7000 + layer; the MTP drafter's are 7100 / 7101
MAX_LAYERS = 100
WIDTHS = (1, 2, 3, 4, 6, 8)
# tokens a request of a verify step (1 + draft tokens) the verify kernels have
VERIFY_TOKENS = (2, 3, 4, 6, 8)


def text_config(hf_config: Any) -> Any:
    return getattr(hf_config, "text_config", None) or hf_config


def _rope_dicts(tc: Any) -> list:
    """The RoPE dicts of the config (``rope_parameters`` in transformers 5,
    ``rope_scaling`` before), as SGLang's get_rope_config reads them."""
    return [
        d
        for d in (
            getattr(tc, "rope_parameters", None),
            getattr(tc, "rope_scaling", None),
        )
        if isinstance(d, dict) and d
    ]


def _rope_theta(tc: Any):
    params = (_rope_dicts(tc) or [{}])[0]
    return params.get("rope_theta", getattr(tc, "rope_theta", None))


def _rope_scaling_types(tc: Any) -> list:
    """The RoPE types other than 'default' (SGLang treats a dict without a
    type as unscaled). The kernels compute the plain frequencies of theta only."""
    types = []
    for d in _rope_dicts(tc):
        kind = d.get("rope_type", d.get("type"))
        if kind not in (None, "default"):
            types.append(str(kind))
    return types


_DTYPES = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "half": "float16",
    "fp16": "float16",
}


def _dtype_name(value: Any):
    """'bfloat16' / 'float16' for the spellings of a dtype (str or torch.dtype),
    else None."""
    return _DTYPES.get(str(value).removeprefix("torch.").lower())


def _model_dtype(cfg: Any, hf_config: Any) -> str:
    """The dtype the model runs in: the server argument, or for 'auto' the
    checkpoint's (SGLang's ModelConfig resolves it the same way); bfloat16 when
    neither is known."""
    named = _dtype_name(getattr(cfg, "dtype", "auto"))
    if named:
        return named
    tc = text_config(hf_config)
    for conf in (hf_config, tc):
        for attr in ("dtype", "torch_dtype"):
            named = _dtype_name(getattr(conf, attr, None))
            if named:
                return named
    return "bfloat16"


def config_disable_reasons(cfg: Any, hf_config: Any) -> list:
    """Checks on the resolved server arguments and the HF config that
    Requirements cannot state."""
    tc = text_config(hf_config)
    reasons = [
        f"config.{key} {getattr(tc, key, None)!r} is not {want!r}"
        for key, want in TEXT_CONFIG.items()
        if getattr(tc, key, None) != want
    ]
    layers = getattr(tc, "num_hidden_layers", None)
    if not layers or layers < 1 or layers > MAX_LAYERS:
        reasons.append(
            f"config.num_hidden_layers {layers!r} (one kernel per layer, 1 to {MAX_LAYERS})"
        )
    if _rope_theta(tc) != ROPE_THETA:
        reasons.append(f"rope_theta {_rope_theta(tc)!r} is not {ROPE_THETA}")
    scaled = _rope_scaling_types(tc)
    if scaled:
        reasons.append(
            f"rope_type {scaled} is not 'default' (the kernels compute unscaled RoPE only)"
        )
    quant = getattr(hf_config, "quantization_config", None) or {}
    if quant.get("quant_method") != "fp8" or list(
        quant.get("weight_block_size") or []
    ) != [128, 128]:
        reasons.append("the kernels need the block-FP8 (128x128) checkpoint")
    context = getattr(cfg, "context_length", None)
    if context and mono_decode.is_available() and context > mono_decode.max_context():
        reasons.append(
            f"context length {context} is above the kernels' {mono_decode.max_context()}"
        )
    algo = getattr(cfg, "speculative_algorithm", None)
    if algo is not None:
        if str(algo).upper() not in ("EAGLE", "NEXTN"):
            reasons.append(
                f"speculative algorithm {algo} (the verify kernels serve EAGLE / NEXTN MTP)"
            )
        if (getattr(cfg, "speculative_eagle_topk", None) or 1) != 1:
            reasons.append(
                "speculative_eagle_topk above 1 (the verify kernels serve top-1 chains)"
            )
        tokens = getattr(cfg, "speculative_num_draft_tokens", None)
        if tokens not in VERIFY_TOKENS:
            reasons.append(
                f"speculative_num_draft_tokens {tokens} is not one of {list(VERIFY_TOKENS)}"
            )
    dtype = _model_dtype(cfg, hf_config)
    conv_dtype = _conv_dtype()
    if dtype == "float16" and conv_dtype != "float16":
        reasons.append(
            "an fp16 model needs SGLANG_MAMBA_CONV_DTYPE=float16 (one dtype for activations and conv state)"
        )
    if dtype == "bfloat16" and conv_dtype not in ("bfloat16", None):
        reasons.append(
            f"SGLANG_MAMBA_CONV_DTYPE {conv_dtype!r} is not the model dtype bfloat16"
        )
    kv = str(getattr(cfg, "kv_cache_dtype", "auto"))
    if kv != "auto" and _dtype_name(kv) != dtype:
        reasons.append(
            f"kv_cache_dtype {kv!r} must be 'auto' or the model dtype {dtype!r} (one dtype for activations and the KV cache)"
        )
    return reasons


def _conv_dtype():
    try:
        from sglang.srt.environ import envs

        return envs.SGLANG_MAMBA_CONV_DTYPE.get()
    except Exception:
        return None


def model_disable_reasons(model: torch.nn.Module, hf_config: Any, copies: dict) -> list:
    """Per-rank checks of the loaded model (the kernels check every tensor again
    at registration)."""
    if not mono_decode.is_available():
        return [
            "this sgl-kernel-xpu build has no mono decode kernels (USE_MONO_DECODE)"
        ]
    device = next(model.parameters()).device
    if not mono_decode.supports(device):
        return [f"{device} is not an Xe2 GPU with at least 32 Xe cores"]
    from .runtime import layer_modules

    reasons = []
    try:
        layers = layer_modules(model)
    except Exception as e:  # noqa: BLE001
        return [f"unexpected model structure: {type(e).__name__}: {e}"]
    for i, layer in enumerate(layers):
        experts = layer.mlp.experts
        if experts.w13_weight.dtype != torch.float8_e4m3fn or not hasattr(
            experts, "w13_weight_scale_inv"
        ):
            reasons.append(f"layer {i}: the routed experts are not block-FP8")
            break
    return reasons


def build(model, hf_config, context):
    from .runtime import MonoQwen35

    return MonoQwen35(model, hf_config, context)


SPEC = MonoSpec(
    name="qwen3_5_moe_35b_a3b_fp8",
    devices={"xe2": DeviceLimits(widths=WIDTHS, tp_sizes=(2,), resident_blocks=32)},
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16", "float16"}),
        attention_backends=frozenset({"intel_xpu"}),
        server_args={
            "page_size": frozenset({64, 128}),
            "disable_radix_cache": frozenset({True}),
            "kv_cache_dtype": frozenset({"auto", "bfloat16", "float16"}),
        },
    ),
    numerics=Numerics(
        kind=NumericsKind.TOLERANCE,
        bounds={"layer.hidden.cosine": 0.999, "layer.residual.cosine": 0.9999},
    ),
    model_disable_reasons=model_disable_reasons,
    config_disable_reasons=config_disable_reasons,
    build=build,
    api_version=MONOKERNEL_API_VERSION,
)


def mtp_model_disable_reasons(
    model: torch.nn.Module, hf_config: Any, copies: dict
) -> list:
    """Per-rank checks of the loaded MTP drafter (the kernels check every
    tensor again at registration)."""
    if not mono_decode.is_available():
        return [
            "this sgl-kernel-xpu build has no mono decode kernels (USE_MONO_DECODE)"
        ]
    device = next(model.parameters()).device
    if not mono_decode.supports(device):
        return [f"{device} is not an Xe2 GPU with at least 32 Xe cores"]
    from .runtime import is_attention, layer_modules

    try:
        layers = layer_modules(model)
        fc = model.fc.weight
    except Exception as e:  # noqa: BLE001
        return [f"unexpected MTP drafter structure: {type(e).__name__}: {e}"]
    reasons = []
    if len(layers) != 1 or not is_attention(layers[0]):
        reasons.append("the MTP drafter is not one full-attention decoder layer")
    if tuple(fc.shape) != (2048, 4096) or fc.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        reasons.append(
            f"the MTP fc is {tuple(fc.shape)} {fc.dtype}, not the full 16-bit [2048, 4096]"
        )
    experts = layers[0].mlp.experts if layers else None
    if experts is not None and experts.w13_weight.dtype != torch.float8_e4m3fn:
        reasons.append("the MTP drafter's routed experts are not block-FP8")
    return reasons


def build_mtp(model, hf_config, context):
    from .runtime import MonoQwen35MTP

    return MonoQwen35MTP(model, hf_config, context)


# The MTP drafter of the same models: its draft decode steps (one token a
# request, EAGLE / NEXTN top-1 chains) run the prologue and attention kernels.
MTP_SPEC = MonoSpec(
    name="qwen3_5_moe_35b_a3b_fp8_mtp",
    devices={"xe2": DeviceLimits(widths=WIDTHS, tp_sizes=(2,), resident_blocks=32)},
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=SPEC.requires,
    numerics=SPEC.numerics,
    model_disable_reasons=mtp_model_disable_reasons,
    config_disable_reasons=config_disable_reasons,
    build=build_mtp,
    api_version=MONOKERNEL_API_VERSION,
    tokens_per_request=lambda model: 1,
)
