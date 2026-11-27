# SPDX-License-Identifier: Apache-2.0
"""Register the optional MiniCPM5 decode implementation."""


def register():
    try:
        from sglang.kernels.plugin_kernels.amd.spec import MonoSpec
        from sglang.kernels.plugin_kernels.xpu.spec import BACKEND
        from sglang.kernels.plugin_kernels.xpu.spec import register as register_spec
    except ImportError:
        return

    register_spec(
        backend=BACKEND,
        architecture="MiniCPM5ForCausalLM",
        model_type="minicpm5_moe",
        path="sgl_kernel.monokernel.minicpm5.spec:SPEC",
    )
    if "allow_model_fallback" in MonoSpec.__struct_fields__:
        register_spec(
            backend=BACKEND,
            architecture="Qwen3DSparkModel",
            model_type="qwen3",
            path="sgl_kernel.monokernel.minicpm5.draft:SPEC",
        )
