import ctypes
import os
import platform
from typing import Optional

import torch

SYSTEM_ARCH = platform.machine()


def _export_jit_toolchain_env() -> None:
    """Expose torch's include/lib dirs to the C++ runtime-JIT engine (src/jit).

    Only Python knows torch's install location; the C++ engine reads these env
    vars to build the icpx command line for on-demand FMHA kernel compilation.
    """
    try:
        from torch.utils.cpp_extension import include_paths

        os.environ.setdefault("SGL_JIT_TORCH_INCLUDE", os.pathsep.join(include_paths()))
        os.environ.setdefault(
            "SGL_JIT_TORCH_LIB", os.path.join(os.path.dirname(torch.__file__), "lib")
        )
    except Exception:
        pass


_export_jit_toolchain_env()

cuda_path = f"/usr/local/cuda/targets/{SYSTEM_ARCH}-linux/lib/libcudart.so.12"
if os.path.exists(cuda_path):
    ctypes.CDLL(cuda_path, mode=ctypes.RTLD_GLOBAL)

from sgl_kernel import common_ops
from sgl_kernel.allreduce import *
from sgl_kernel.attention import (
    flash_mla_decode,
    flash_mla_decode_get_workspace_size,
    flash_mla_get_workspace_size,
    flash_mla_prefill,
    flash_mla_prefill_get_workspace_size,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    lightning_attention_decode,
    merge_state,
    merge_state_v2,
)
from sgl_kernel.compress_plan import (
    plan_compress_decode,
    plan_compress_decode_legacy,
    plan_compress_prefill,
    plan_compress_prefill_legacy,
)
from sgl_kernel.elementwise import (
    apply_rope_with_cos_sin_cache_inplace,
    fused_add_rmsnorm,
    fused_inplace_qknorm_rope,
    fused_k_norm_rope_flashmla,
    fused_q_norm_rope,
    fused_qk_norm_rope,
    fused_qk_rope,
    fused_qk_rope_with_cos_sin_cache_inplace,
    gelu_and_mul,
    gelu_tanh_and_mul,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    multimodal_rotary_embedding,
    rmsnorm,
    silu_and_mul,
    silu_and_mul_clamp,
    store_cache_xpu,
)
from sgl_kernel.flash_compress_4 import flash_compress4_decode, flash_compress4_prefill
from sgl_kernel.flash_compress_128 import (
    flash_compress128_decode,
    flash_compress128_prefill,
)

try:
    from sgl_kernel.fp8_paged_mqa_logits import fp8_paged_mqa_logits_triton
except ImportError:
    # Triton (or its Intel backend) not importable in this environment.
    # fp8_paged_mqa_logits_triton will simply be unavailable.
    fp8_paged_mqa_logits_triton = None
from sgl_kernel.fused_norm_rope_v2 import compress_norm_rope_store
from sgl_kernel.gdn_attn import gdn_attention

hadamard_transform = torch.ops.sgl_kernel.hadamard_transform
fused_q_indexer_rope_hadamard_quant = (
    torch.ops.sgl_kernel.fused_q_indexer_rope_hadamard_quant
)
from sgl_kernel.gemm import (
    awq_dequantize,
    bmm_fp8,
    cutlass_scaled_fp4_mm,
    fp8_blockwise_scaled_mm,
    fp8_scaled_mm,
    int8_scaled_mm,
    qserve_w4a8_per_chn_gemm,
    qserve_w4a8_per_group_gemm,
    scaled_fp4_experts_quant,
    scaled_fp4_quant,
    sgl_per_tensor_quant_fp8,
    sgl_per_token_group_quant_8bit,
    sgl_per_token_group_quant_fp4,
    sgl_per_token_group_quant_fp8,
    sgl_per_token_group_quant_int8,
    sgl_per_token_quant_fp8,
)
from sgl_kernel.grammar import apply_token_bitmask_inplace_cuda
from sgl_kernel.hisparse import (
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
    transfer_cache_dsv4_mla,
)
from sgl_kernel.inkling_attn_prologue import (
    compile_inkling_attn_prologue,
    inkling_attn_prologue_decode,
    inkling_attn_prologue_extend,
    inkling_attn_prologue_verify,
)
from sgl_kernel.inkling_sconv import (
    causal_conv1d,
    fused_causal_conv1d_update_decode,
    fused_decode_sconv_metadata,
    fused_draft_extend_sconv_cache,
    fused_extend_sconv_metadata,
    fused_gather_scatter_to_sconv_cache,
    precompute_helion_decode_metadata,
    precompute_helion_extend_metadata,
    save_intermediate_conv_windows,
    track_conv_indices,
    update_sconv_cache,
)
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer,
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_all_layer_lf_pf,
    transfer_kv_all_layer_lf_ph,
    transfer_kv_all_layer_mla,
    transfer_kv_all_layer_mla_lf_pf,
    transfer_kv_direct,
    transfer_kv_per_layer,
    transfer_kv_per_layer_direct_pf_lf,
    transfer_kv_per_layer_mla,
    transfer_kv_per_layer_mla_pf_lf,
    transfer_kv_per_layer_pf_lf,
    transfer_kv_per_layer_ph_lf,
)
from sgl_kernel.lora import (
    embedding_lora_a_fwd,
    qkv_lora_b_fwd,
    sgemm_lora_a_fwd,
    sgemm_lora_b_fwd,
)
from sgl_kernel.mamba import causal_conv1d_fn_xpu, causal_conv1d_update_xpu
from sgl_kernel.memory import weak_ref_tensor
from sgl_kernel.mhc import (
    hc_post,
    hc_pre_big_fuse,
    hc_pre_gemm_sqr_sum,
    hc_split_sinkhorn,
    mhc_pre,
)

fused_hc_head = torch.ops.sgl_kernel.fused_hc_head.default
mhc_fused_post_pre = torch.ops.sgl_kernel.mhc_fused_post_pre.default
from sgl_kernel.minimax import minimax_decode_topk, minimax_decode_topk_page_table
from sgl_kernel.moe import (
    apply_shuffle_mul_sum,
    biased_topk,
    cutlass_fp4_group_mm,
    fp8_blockwise_scaled_grouped_mm,
    fused_experts,
    hash_topk,
    moe_align_block_size,
    moe_fused_gate,
    moe_sum,
    moe_sum_reduce,
    prepare_moe_input,
    scatter_tokens_to_experts,
    swiglu_gpt_oss_sigmoid_alpha,
    topk_sigmoid,
    topk_softmax,
)
from sgl_kernel.nsa import fp8_mqa_logits, fp8_paged_mqa_logits
from sgl_kernel.sampling import (
    min_p_sampling_from_probs,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
    top_p_sampling_from_probs,
)
from sgl_kernel.sparse_flash_attn import (
    convert_vertical_slash_indexes,
    convert_vertical_slash_indexes_mergehead,
    sparse_attn_func,
    sparse_attn_varlen_func,
)
from sgl_kernel.speculative import (
    build_tree_kernel_efficient,
    segment_packbits,
    tree_speculative_sampling_target_only,
    verify_tree_greedy,
)
from sgl_kernel.top_k import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
    topk_transform,
    topk_transform_paged,
    topk_transform_ragged,
)
from sgl_kernel.utils import get_device_capability, is_xe2_arch
from sgl_kernel.version import __version__

build_tree_kernel = (
    None  # TODO(ying): remove this after updating the sglang python code.
)
