"""The XPU monokernel spec of Qwen3.5 / Qwen3.6 MoE 35B-A3B (block-FP8, TP2 on
two Xe2 GPUs with 32 Xe cores): decode steps of 1, 2, 4 or 8 rows run one mono
kernel per decoder layer; every other step runs the model's own forward."""

from typing import Any

import torch
from sglang.kernels.plugin_kernels.xpu.spec import (
    MONOKERNEL_API_VERSION,
    DeviceLimits,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_kernel import mono_decode

# The shapes the kernels are written for (the text config of the model)
TEXT_CONFIG = {
    "hidden_size": 2048,
    "num_hidden_layers": 40,
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
WIDTHS = (1, 2, 4, 8)


def text_config(hf_config: Any) -> Any:
    return getattr(hf_config, "text_config", None) or hf_config


def _rope_theta(tc: Any):
    params = getattr(tc, "rope_parameters", None) or getattr(tc, "rope_scaling", None) or {}
    return params.get("rope_theta", getattr(tc, "rope_theta", None))


def config_disable_reasons(cfg: Any, hf_config: Any) -> list:
    """Checks on the resolved server arguments and the HF config that
    Requirements cannot state."""
    tc = text_config(hf_config)
    reasons = [
        f"config.{key} {getattr(tc, key, None)!r} is not {want!r}"
        for key, want in TEXT_CONFIG.items()
        if getattr(tc, key, None) != want
    ]
    if _rope_theta(tc) != ROPE_THETA:
        reasons.append(f"rope_theta {_rope_theta(tc)!r} is not {ROPE_THETA}")
    quant = getattr(hf_config, "quantization_config", None) or {}
    if quant.get("quant_method") != "fp8" or list(quant.get("weight_block_size") or []) != [128, 128]:
        reasons.append("the kernels need the block-FP8 (128x128) checkpoint")
    context = getattr(cfg, "context_length", None)
    if context and mono_decode.is_available() and context > mono_decode.max_context():
        reasons.append(f"context length {context} is above the kernels' {mono_decode.max_context()}")
    dtype = str(getattr(cfg, "dtype", "auto"))
    conv_dtype = _conv_dtype()
    if dtype in ("float16", "half") and conv_dtype != "float16":
        reasons.append("an fp16 model needs SGLANG_MAMBA_CONV_DTYPE=float16 (one dtype for activations and conv state)")
    if dtype in ("auto", "bfloat16") and conv_dtype not in ("bfloat16", None):
        reasons.append(f"SGLANG_MAMBA_CONV_DTYPE {conv_dtype!r} is not the model dtype bfloat16")
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
        return ["this sgl-kernel-xpu build has no mono decode kernels (USE_MONO_DECODE)"]
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
        if experts.w13_weight.dtype != torch.float8_e4m3fn or not hasattr(experts, "w13_weight_scale_inv"):
            reasons.append(f"layer {i}: the routed experts are not block-FP8")
            break
    return reasons


def build(model, hf_config, context):
    from .runtime import MonoQwen35

    return MonoQwen35(model, hf_config, context)


SPEC = MonoSpec(
    name="qwen3_5_moe_35b_a3b_fp8",
    devices={"xe2": DeviceLimits(widths=WIDTHS, tp_sizes=(2,), resident_blocks=32)},
    features=frozenset(),
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
