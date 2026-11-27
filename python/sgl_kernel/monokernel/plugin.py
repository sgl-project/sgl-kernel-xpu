"""The ``sglang.srt.plugins`` entry point: registers the XPU monokernel specs
with SGLang's monokernel registry. Importing a spec (and the kernels) is left
to SGLang, which does it only with --enable-xpu-monokernel."""

import logging

logger = logging.getLogger(__name__)

# HF architectures -> spec
SPECS = {
    "Qwen3_5MoeForConditionalGeneration": "sgl_kernel.monokernel.qwen3_5.spec:SPEC",
    "Qwen3_5MoeForCausalLM": "sgl_kernel.monokernel.qwen3_5.spec:SPEC",
    # the MTP drafter of either (SGLang's draft model architecture)
    "Qwen3_5ForCausalLMMTP": "sgl_kernel.monokernel.qwen3_5.spec:MTP_SPEC",
}


def register() -> None:
    try:
        from sglang.kernels.plugin_kernels.xpu.spec import BACKEND
        from sglang.kernels.plugin_kernels.xpu.spec import register as register_spec
    except ImportError:  # an SGLang without the XPU monokernel backend
        logger.debug("sgl_kernel.monokernel: SGLang has no XPU monokernel backend")
        return
    for architecture, path in SPECS.items():
        register_spec(backend=BACKEND, architecture=architecture, path=path)
