# SPDX-License-Identifier: Apache-2.0
"""SGLang requirements for the MiniCPM5 Xe2 decode path."""

from pathlib import Path

from sgl_kernel import minicpm5_decode
from sgl_kernel.monokernel.minicpm5.runtime import MiniCPM5Decode

from sglang.kernels.plugin_kernels.amd.spec import (
    MONOKERNEL_API_VERSION,
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)


def _ptrace_available() -> bool:
    scope = Path("/proc/sys/kernel/yama/ptrace_scope")
    value = scope.read_text().strip() if scope.exists() else "0"
    if value == "0":
        return True
    if value == "3":
        return False
    cap = next(
        line
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("CapEff:")
    )
    return bool(int(cap.split()[1], 16) & (1 << 19))


def _config_reasons(cfg, hf_config):
    if not minicpm5_decode.is_available():
        return ["sglang-kernel-xpu was built without USE_MONO_DECODE"]
    if cfg.tp_size == 2 and not _ptrace_available():
        return [
            "TP2 symmetric memory needs SYS_PTRACE when kernel.yama.ptrace_scope is enabled"
        ]
    return []


def _model_reasons(model, hf_config, copies):
    device = next(model.parameters()).device
    if not minicpm5_decode.work_groups(device):
        return [f"{device} cannot run a resident MiniCPM5 Xe2 grid"]
    if any(
        layer.is_layer_sparse and layer.mlp.shared_experts is None
        for layer in model.model.layers
    ):
        return [
            "MiniCPM5 mono requires the shared expert to be separate from the routed expert bank"
        ]
    return []


SPEC = MonoSpec(
    name="MiniCPM5 Xe2 decode",
    devices={"xe2": DeviceLimits((1, 2, 3, 4, 5, 6, 7, 8), (1, 2), 20)},
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"intel_xpu"}),
        server_args={
            "quantization": frozenset({"fp8"}),
            "kv_cache_dtype": frozenset({"auto", "bf16", "bfloat16"}),
            "page_size": frozenset({64, 128}),
            "speculative_algorithm": frozenset({None, "DSPARK"}),
        },
        hf_config={
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "moe_intermediate_size": 512,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "n_routed_experts": 160,
            "n_shared_experts": 1,
            "num_experts_per_tok": 16,
            "topk_group": 1,
            "n_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 3.66,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "attention_bias": False,
            "use_gated_attention": True,
        },
    ),
    numerics=Numerics(NumericsKind.TOLERANCE, {"decode.hidden.cosine": 0.99}),
    model_disable_reasons=_model_reasons,
    config_disable_reasons=_config_reasons,
    build=MiniCPM5Decode,
    api_version=MONOKERNEL_API_VERSION,
)
