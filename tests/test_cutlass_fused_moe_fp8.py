import argparse

import torch
from test_per_token_group_quant_8bit import *

# import triton  # Added import
##import triton.testing  # Added import
from transformers import AutoConfig

_is_hip = False
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
fp8_dtype = fp8_type_
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max

enable_sgl_per_token_group_quant_8bit = True

from typing import Optional, Tuple

from sgl_kernel import (
    apply_shuffle_mul_sum,
    fp8_blockwise_scaled_grouped_mm,
    prepare_moe_input,
    scatter_tokens_to_experts,
    silu_and_mul,
)

# from sglang.srt.layers.moe.cutlass_moe import cutlass_fused_experts_fp8
##from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
# from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
# from sglang.srt.layers.moe.topk import StandardTopKOutput


# Copy from: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/utils.py
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def get_model_config(tp_size: int):
    config = AutoConfig.from_pretrained(
        "deepseek-ai/Deepseek-R1", trust_remote_code=True
    )
    E = config.n_routed_experts
    topk = config.num_experts_per_tok
    intermediate_size = config.moe_intermediate_size
    shard_intermediate_size = 2 * intermediate_size // tp_size

    return {
        "num_experts": E,
        "topk": topk,
        "hidden_size": config.hidden_size,
        "shard_intermediate_size": shard_intermediate_size,
        "dtype": config.dtype,
        "block_shape": config.quantization_config["weight_block_size"],
    }


def to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    """Converts tensor to FP8 E4M3, scaling values to fit the range."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    # Calculate max absolute value safely
    max_val = torch.max(torch.abs(tensor))
    # Avoid division by zero if tensor is all zeros
    if max_val == 0:
        scale_factor = 1.0
    else:
        # Scale factor to bring the max value to finfo.max
        scale_factor = finfo.max / max_val

    # Apply scaling
    scaled_tensor = tensor * scale_factor

    # Clamp and convert
    fp8_tensor = scaled_tensor.clamp(min=finfo.min, max=finfo.max).to(
        dtype=torch.float8_e4m3fn
    )
    return fp8_tensor


##/home/kmshaik/sglang/python/sglang/srt/layers/quantization/fp8_kernel.py
# def sglang_per_token_group_quant_fp8(
#    x: torch.Tensor,
#    group_size: int,
#    eps: float = 1e-10,
#    column_major_scales: bool = False,
#    scale_tma_aligned: bool = False,
#    scale_ue8m0: bool = False,
#    fuse_silu_and_mul: bool = False,
#    masked_m: Optional[torch.Tensor] = None,
#    enable_v2: Optional[bool] = None,
# ):
#    assert (
#        x.shape[-1] % group_size == 0
#    ), "the last dimension of `x` cannot be divisible by `group_size`"
#    assert x.is_contiguous(), "`x` is not contiguous"

#    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))

#    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_dtype)
#    x_s = create_per_token_group_quant_fp8_output_scale(
#        x_shape=out_shape,
#        device=x.device,
#        group_size=group_size,
#        column_major_scales=column_major_scales,
#        scale_tma_aligned=scale_tma_aligned,
#        scale_ue8m0=scale_ue8m0,
#    )

#    if x.shape[0] > 0:
#        # Temporary
#        if enable_sgl_per_token_group_quant_8bit:
#            sgl_per_token_group_quant_8bit(
#                x,
#                x_q,
#                x_s,
#                group_size,
#                eps,
#                fp8_min,
#                fp8_max,
#                scale_ue8m0,
#                fuse_silu_and_mul,
#                masked_m,
#                enable_v2=enable_v2,
#            )
#        else:
#            assert not enable_v2
#            sgl_per_token_group_quant_fp8(
#                x, x_q, x_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
#            )

#    return x_q, x_s


def prepare_input_moe_ref(
    topk_ids,
    expert_offsets,
    blockscale_offsets,
    problem_sizes1,
    problem_sizes2,
    input_permutation,
    output_permutation,
    num_experts,
    hidden_dim,
    top_k,
):
    tokens, top_k = topk_ids.shape
    expert_cnt = torch.zeros(num_experts, dtype=torch.int32)
    for e in range(num_experts):
        expert_cnt[e] = (topk_ids == e).sum()
        expert_offsets[e] = expert_cnt[e]
    print(expert_cnt)
    print(expert_cnt.shape)
    print(expert_offsets)
    print(expert_offsets.shape)
    for e in range(num_experts):
        r = expert_cnt[e].item()
        c = hidden_dim
        problem_sizes1[e * 3 + 0] = r
        problem_sizes1[e * 3 + 1] = c * 2
        problem_sizes1[e * 3 + 2] = top_k
        problem_sizes2[e * 3 + 0] = r
        problem_sizes2[e * 3 + 1] = top_k
        problem_sizes2[e * 3 + 2] = c

    # compute offsets
    atomic_buffer = torch.zeros(num_experts, dtype=torch.int32)
    tot_offset = 0
    # expert_offsets[0] = 0
    for i in range(num_experts):
        atomic_buffer[i] = tot_offset
        tot_offset += problem_sizes1[i * 3].item()
        # expert_offsets[i + 1] = tot_offset

    # compute input/output permutes
    num_tokens = topk_ids.size(0)
    flat_topk = topk_ids.flatten()
    topk_length = num_tokens * top_k

    for i in range(topk_length):
        expert_id = int(flat_topk[i])
        start = int(atomic_buffer[expert_id].item())
        atomic_buffer[expert_id] += 1

        input_permutation[start] = i // top_k
        output_permutation[i] = start


# Pure PyTorch reference implementations
def per_token_group_quant_fp8_ref(
    x: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-10,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for per-token group FP8 quantization."""
    assert x.dim() == 2 and x.size(1) % group_size == 0
    num_tokens, hidden_dim = x.shape
    x_view = x.view(num_tokens, -1, group_size)
    x_amax = x_view.abs().float().amax(dim=2).clamp(min=eps)
    scales = x_amax / 448.0  # FP8 E4M3 max

    if scale_ue8m0:
        scales = ceil_to_ue8m0(scales)

    x_quantized = (x_view / scales.unsqueeze(2)).to(torch.float8_e4m3fn)
    return x_quantized.view(num_tokens, hidden_dim), scales


def scatter_tokens_to_experts_ref(input_tensor, src2dst_map, num_output_tokens, topk):
    """
    Reference implementation of scatter_tokens_to_experts on CPU.

    Args:
        input_tensor: [num_tokens, hidden_dim] - input tokens
        src2dst_map: [num_tokens * topk] - maps each (token, k) pair to destination row
        num_output_tokens: total output rows (num_tokens * topk)
        topk: number of experts per token

    Returns:
        output_tensor: [num_output_tokens, hidden_dim] - scattered tokens
    """
    num_tokens, hidden_dim = input_tensor.shape
    output_tensor = torch.zeros(
        (num_output_tokens, hidden_dim),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    for token_id in range(num_tokens):
        for k in range(topk):
            src_row = token_id
            dst_row = src2dst_map[token_id * topk + k].item()
            output_tensor[dst_row] = input_tensor[src_row]

    return output_tensor


def is_sm90_supported():
    return False


def run_get_group_gemm_starts(
    a_full_fp8, a_scales, b_full_fp8, b_scales, expert_offsets
):

    device = a_full.device

    # Each expert gets ex: 512 tokens (uniform distribution for simplicity)
    tokens_per_expert = total_tokens // num_experts
    if expert_offsets == None:
        expert_offsets = torch.tensor(
            [i * tokens_per_expert for i in range(num_experts)],
            dtype=torch.int32,
            device=device,
        )

    # Problem sizes: all experts have same dimensions
    if expert_offsets == None:
        problem_sizes = torch.tensor(
            [[tokens_per_expert, intermediate_size, hidden_size]] * num_experts,
            dtype=torch.int32,
            device=device,
        )
    else:
        # sizes = [2, 2]
        # expert_assignments = torch.tensor(sizes, dtype=torch.int32)
        # Compute problem_sizes and expert_offsets from assignments
        problem_sizes, expert_offsets_cum = (
            compute_problem_sizes_from_expert_assignments(
                expert_offsets,
                num_experts,
                intermediate_size,
                hidden_size,
                device=device,
            )
        )

    a_full_fp8, a_scales = quantize_activation_1d(a_full)
    b_full_fp8, b_scales = quantize_weight_2d(b_full)

    # Prepare inputs
    # inputs = prepare_xpu_mxfp8_gemm_inputs(
    #    a_full_fp8,
    #    b_full_fp8,
    #    a_scales,
    #    b_scales,
    #    expert_offsets,
    #    problem_sizes,
    #    device=device
    # )

    inputs = prepare_xpu_mxfp8_gemm_inputs_no_transpose(
        a_full_fp8,
        b_full_fp8,
        a_scales,
        b_scales,
        expert_offsets_cum,
        problem_sizes,
        device=device,
    )

    return inputs


def compute_problem_sizes_from_expert_assignments(
    expert_assignments: torch.Tensor,  # [num_tokens] - expert ID for each token
    num_experts: int,
    n: int,  # intermediate_size
    k: int,  # hidden_size
    device: str = "xpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute problem_sizes and cumulative expert_offsets from expert assignments.

    This is useful when you have a token-to-expert assignment array (e.g., from
    MoE routing) and need to compute the problem_sizes and expert_offsets for
    the grouped GEMM kernel.

    Args:
        expert_assignments: [num_tokens] tensor with expert ID for each token
                           Example: [1, 2, 0, 2, 4, 2, 0, 5]
                           Each value indicates which expert handles that token
        num_experts: Total number of experts in the MoE layer
        n: Output dimension (intermediate_size for up-projection)
        k: Input dimension (hidden_size)
        device: Target device for output tensors

    Returns:
        Tuple of:
        - problem_sizes: [num_experts, 3] as (M, N, K) per expert
                        M varies per expert (number of tokens assigned to it)
                        N and K are constant across experts
        - expert_offsets: [num_experts] cumulative token counts
                         Indicates where each expert's data starts in the reordered tensor

    Example:
        >>> # 8 tokens assigned to 8 experts
        >>> expert_assignments = torch.tensor([1, 2, 0, 2, 4, 2, 0, 5], dtype=torch.int32)
        >>> problem_sizes, offsets = compute_problem_sizes_from_expert_assignments(
        ...     expert_assignments, num_experts=8, n=11008, k=4096
        ... )
        >>> print(offsets)  # [0, 2, 3, 6, 6, 7, 8, 8]
        >>> print(problem_sizes[:, 0])  # M per expert: [2, 1, 3, 0, 1, 1, 0, 0]

        Interpretation:
        - Expert 0: 2 tokens (offsets 0-1)
        - Expert 1: 1 token (offset 2)
        - Expert 2: 3 tokens (offsets 3-5)
        - Expert 3: 0 tokens (no data)
        - Expert 4: 1 token (offset 6)
        - Expert 5: 1 token (offset 7)
        - Expert 6: 0 tokens (no data)
        - Expert 7: 0 tokens (no data)

    Note:
        The input activation tensor a_full must be REORDERED by expert before
        passing to prepare_xpu_mxfp8_gemm_inputs(). Tokens should be grouped
        by their assigned expert in the order: expert_0_tokens, expert_1_tokens, ...
    """

    # Cumulative offsets: where each expert's data starts in the reordered tensor
    # Example: counts [2, 1, 3, 0, 1, 1, 0, 0] -> offsets [0, 2, 3, 6, 6, 7, 8, 8]
    expert_offsets_cum = (
        torch.cumsum(
            torch.cat(
                [
                    torch.tensor([0], device=expert_assignments.device),
                    expert_assignments,
                ]
            ),
            dim=0,
        )
        .to(torch.int32)
        .to(device)
    )

    # Problem sizes: M varies per expert, N and K are constant
    problem_sizes = torch.zeros((num_experts, 3), dtype=torch.int32, device=device)
    problem_sizes[:, 0] = expert_assignments.to(
        device
    )  # M = number of tokens for this expert
    problem_sizes[:, 1] = n  # N = intermediate_size
    problem_sizes[:, 2] = k  # K = hidden_size

    return problem_sizes, expert_offsets_cum


def ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


FP8_BLOCK_SIZE = 128
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class XPUScaleConfig:
    """
    Configuration for MXFP8 blockwise scaling on Intel XPU.

    For XPU MXFP8 GEMM:
    - block_m: Block size in M dimension (typically 1 for per-row scaling)
    - block_n: Block size in N dimension (typically 128 for 2D blocking)
    - block_k: Block size in K dimension (always 128 for MXFP8)

    Layout conventions:
    - A: (M, K) RowMajor, scales (K//128, M) ColMajor
    - B: (N, K) ColumnMajor, scales (N//128, K//128) RowMajor
    - Output: (M, N) RowMajor, float32
    """

    block_m: int = 1  # Typically 1 for A (per-row scaling)
    block_n: int = FP8_BLOCK_SIZE  # Typically 128 for B (2D blocking)
    block_k: int = FP8_BLOCK_SIZE  # Fixed for MXFP8


# def fp8_blockwise_scaled_grouped_mm(
#     output,
#     a_ptrs,
#     b_ptrs,
#     out_ptrs,
#     a_scales_ptrs,
#     b_scales_ptrs,
#     a,
#     b,
#     scales_a,
#     scales_b,
#     stride_a,
#     stride_b,
#     stride_c,
#     layout_sfa,
#     layout_sfb,
#     problem_sizes,
#     expert_offsets,
#     workspace,
# ):
#     #inputs = run_get_group_gemm_starts(a, a1_scale, w1_q, w1_scale, expert_offsets, )

#     device = a.device
#     num_experts = expert_offsets.size(0) - 1
#     total_tokens = a.size(0)

#     # Get dimensions from problem_sizes
#     #m_first = problem_sizes[0, 0].item()        #Shape = (E, 3) → 2D tensor So indexing [row, col] works.
#     #n_first = problem_sizes[0, 1].item()
#     #k_first = problem_sizes[0, 2].item()

#     # if problem_sizes Shape = (E*3,) → 1D tensor
#     # You need to flatten indexing manually.
#     # OR problem_sizes1 = problem_sizes1.view(E, 3)
#     # here prepare_moe_input returns (#num occurrence, hidden dim, topk) kernel MODIFY
#     #m_first = problem_sizes[0 * 3 + 0].item()
#     #n_first = problem_sizes[0 * 3 + 1].item()
#     #k_first = problem_sizes[0 * 3 + 2].item()

#     # expert_offsets coming from XPU = [2,2,0]                                              Needed prefixsum like [0, 2, 4]
#     # Also problem_sizes prepare_moe_input returns (#num occurrence, hidden dim, topk)       Needed E * [m,n,k] for all three
#     hidden_size = a.shape[1]        # [m, k]
#     intermediate_size = b.size(1)   # E, n, k
#     problem_sizes, expert_offsets_cum = compute_problem_sizes_from_expert_assignments(
#         expert_offsets[:-1],
#         num_experts,
#         intermediate_size,
#         hidden_size,
#         device=device
#     )

#     # Ensure inputs are contiguous
#     a_full = ensure_contiguous(a)
#     b_full = ensure_contiguous(b)

#     # IMPORTANT: Do NOT transpose A scales - keep row-major [total_tokens, K//128]
#     a_scales_row_major = ensure_contiguous(scales_a)
#     b_scales_full = ensure_contiguous(scales_b)

#     # Get base addresses
#     a_base_ptr = a_full.data_ptr()
#     b_base_ptr = b_full.data_ptr()
#     out_base_ptr = output.data_ptr()
#     a_scales_base_ptr = a_scales_row_major.data_ptr()
#     b_scales_base_ptr = b_scales_full.data_ptr()

#     # Element sizes in bytes
#     fp8_size = 1   # float8_e4m3fn
#     fp32_size = 4  # float32

#     # Prepare pointer arrays    ALREADY PASSED CREATED IN FE
#     #a_ptrs = torch.zeros(num_experts, dtype=torch.int64, device=device)
#     #b_ptrs = torch.zeros(num_experts, dtype=torch.int64, device=device)
#     #out_ptrs = torch.zeros(num_experts, dtype=torch.int64, device=device)
#     #a_scales_ptrs = torch.zeros(num_experts, dtype=torch.int64, device=device)
#     #b_scales_ptrs = torch.zeros(num_experts, dtype=torch.int64, device=device)

#     #stride_a = torch.zeros(num_experts, dtype=torch.int64, device=device)
#     #stride_b = torch.zeros(num_experts, dtype=torch.int64, device=device)
#     #stride_c = torch.zeros(num_experts, dtype=torch.int64, device=device)

#     scale_config = XPUScaleConfig()                                                         # NEEDED
#     # Compute pointers for each expert
#     for expert_id in range(num_experts):
#         # Get problem dimensions for this expert
#         m = problem_sizes[expert_id, 0].item()
#         n = problem_sizes[expert_id, 1].item()
#         k = problem_sizes[expert_id, 2].item()

#         # Get cumulative token offset for this expert
#         token_offset = expert_offsets[expert_id].item()

#         # Compute offsets into the full tensors
#         # A: [total_tokens, K] -> slice starting at token_offset, size M x K
#         a_offset_elements = token_offset * k
#         a_offset_bytes = a_offset_elements * fp8_size

#         # B: [num_experts, N, K] -> expert_id-th slice, size N x K
#         b_offset_elements = expert_id * n * k
#         b_offset_bytes = b_offset_elements * fp8_size

#         # Output: [num_experts, M, N] -> expert_id-th slice
#         out_offset_elements = expert_id * m * n
#         out_offset_bytes = out_offset_elements * fp32_size

#         # A scales: [total_tokens, K//128] (row-major, NOT transposed)
#         # For row-major layout, offset is token_offset * (K//128)
#         a_scale_k_blocks = k // scale_config.block_k
#         a_scale_offset_elements = token_offset * a_scale_k_blocks
#         a_scale_offset_bytes = a_scale_offset_elements * fp32_size

#         # B scales: [num_experts, N//128, K//128] -> expert_id-th slice
#         b_scale_offset_elements = expert_id * (n // scale_config.block_n) * (k // scale_config.block_k)
#         b_scale_offset_bytes = b_scale_offset_elements * fp32_size

#         # Store pointers (base + offset)
#         a_ptrs[expert_id] = a_base_ptr + a_offset_bytes
#         b_ptrs[expert_id] = b_base_ptr + b_offset_bytes
#         out_ptrs[expert_id] = out_base_ptr + out_offset_bytes
#         a_scales_ptrs[expert_id] = a_scales_base_ptr + a_scale_offset_bytes
#         b_scales_ptrs[expert_id] = b_scales_base_ptr + b_scale_offset_bytes

#         # Store strides (leading dimension)
#         stride_a[expert_id] = k  # A is RowMajor [M, K], stride is K
#         stride_b[expert_id] = k  # B is ColumnMajor [N, K], stored as [K, N] so stride is K
#         stride_c[expert_id] = n  # C is RowMajor [M, N], stride is N

#     # Layout metadata (placeholder - actual layout is computed by kernel)
#     layout_sfa = torch.empty((num_experts, 5), dtype=torch.int32, device=device)
#     layout_sfb = torch.empty((num_experts, 5), dtype=torch.int32, device=device)

#     # Workspace buffer
#     #workspace_size_gb: int = 1,                                                                    # NEEDED
#     workspace_size_gb = 1
#     workspace_bytes = workspace_size_gb * 1024 * 1024 * 1024
#     workspace = torch.empty((workspace_bytes,), dtype=torch.uint8, device=device)

#     inputs = {
#         # Primary outputs for kernel
#         "output": output,
#         "a_ptrs": a_ptrs,
#         "b_ptrs": b_ptrs,
#         "out_ptrs": out_ptrs,
#         "a_scales_ptrs": a_scales_ptrs,
#         "b_scales_ptrs": b_scales_ptrs,
#         "stride_a": stride_a,
#         "stride_b": stride_b,
#         "stride_c": stride_c,
#         "layout_sfa": layout_sfa,
#         "layout_sfb": layout_sfb,
#         "problem_sizes": problem_sizes,
#         #"expert_offsets": expert_offsets,
#         "expert_offsets": torch.arange(num_experts, dtype=torch.int32, device=device),
#         "workspace": workspace,
#     }

#     fp8_blockwise_scaled_grouped_mm(
#         inputs["output"],
#         inputs["a_ptrs"],
#         inputs["b_ptrs"],
#         inputs["out_ptrs"],
#         inputs["a_scales_ptrs"],
#         inputs["b_scales_ptrs"],
#         inputs["a_stack"],
#         inputs["b_stack"],
#         inputs["scales_a_stack"],
#         inputs["scales_b_stack"],
#         inputs["stride_a"],
#         inputs["stride_b"],
#         inputs["stride_c"],
#         inputs["layout_sfa"],
#         inputs["layout_sfb"],
#         inputs["problem_sizes"],
#         inputs["expert_offsets"],
#         inputs["workspace"],
#     )


def sgl_per_token_group_quant_8bit_kernel(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
) -> None:
    if enable_v2 is None:
        from sglang.srt.utils import get_bool_env_var

        enable_v2 = get_bool_env_var("SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2")

    if enable_v2:
        return torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit_v2.default(
            input,
            output_q,
            output_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
            fuse_silu_and_mul,
            masked_m,
        )

    assert not fuse_silu_and_mul, "only v2 support fuse_silu_and_mul"
    assert masked_m is None, "only v2 support masked_m"
    torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit.default(
        input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
    )


# For legacy usage
sgl_per_token_group_quant_fp8_kernel = sgl_per_token_group_quant_8bit_kernel
sgl_per_token_group_quant_int8_kernel = sgl_per_token_group_quant_8bit_kernel


def create_per_token_group_quant_fp8_output_scale(
    x_shape,
    device,
    group_size,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
):
    if scale_ue8m0:
        assert column_major_scales and scale_tma_aligned
        *x_batch, x_q_mn, x_q_k = x_shape
        x_s_mn, x_s_k = x_q_mn, x_q_k // 128
        aligned_mn = align(x_s_mn, 4)
        aligned_k = align(x_s_k, 4)
        # TODO(FIXME): Fix cuda kernel and recover here to empty.
        return torch.empty(
            (*x_batch, aligned_k // 4, aligned_mn),
            device=device,
            dtype=torch.int,
        ).transpose(-1, -2)[..., :x_s_mn, :]
    elif column_major_scales:
        if scale_tma_aligned:
            # TODO extract "align" function
            # aligned to 4 * sizeof(float)
            aligned_size = (x_shape[-2] + 3) // 4 * 4
            return torch.empty(
                x_shape[:-2] + (x_shape[-1] // group_size, aligned_size),
                device=device,
                dtype=torch.float32,
            ).transpose(-1, -2)[: x_shape[-2], :]
        else:
            return torch.empty(
                (x_shape[-1] // group_size,) + x_shape[:-1],
                device=device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        return torch.empty(
            x_shape[:-1] + (x_shape[-1] // group_size,),
            device=device,
            dtype=torch.float32,
        )


def sglang_per_token_group_quant_fp8_layer(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))

    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )

    if x.shape[0] > 0:
        # Temporary
        if enable_sgl_per_token_group_quant_8bit:
            sgl_per_token_group_quant_8bit_kernel(
                x,
                x_q,
                x_s,
                group_size,
                eps,
                fp8_min,
                fp8_max,
                scale_ue8m0,
                fuse_silu_and_mul,
                masked_m,
                enable_v2=enable_v2,
            )
        else:
            assert not enable_v2
            sgl_per_token_group_quant_fp8_kernel(
                x, x_q, x_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
            )

    return x_q, x_s


def sglang_per_token_group_quant_8bit_layer(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
):

    if dst_dtype == torch.int8:
        assert not column_major_scales
        assert not scale_tma_aligned
        assert not fuse_silu_and_mul
        assert masked_m is None
        return sglang_per_token_group_quant_int8_layer(
            x=x,
            group_size=group_size,
            eps=eps,
            dtype=dst_dtype,
            enable_v2=enable_v2,
        )

    return sglang_per_token_group_quant_fp8_layer(
        x=x,
        group_size=group_size,
        eps=eps,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
        enable_v2=enable_v2,
    )


# /home/kmshaik/sglang/python/sglang/srt/layers/moe/cutlass_moe.py
def cutlass_fused_experts_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_strides: torch.Tensor,
    c1_strides: torch.Tensor,
    a2_strides: torch.Tensor,
    c2_strides: torch.Tensor,
    workspace: torch.Tensor,
    a_ptrs: torch.Tensor,
    b_ptrs: torch.Tensor,
    out_ptrs: torch.Tensor,
    a_scales_ptrs: torch.Tensor,
    b_scales_ptrs: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    use_fp8_blockscale: bool = True,
    use_mxfp8: bool = False,
    output: Optional[torch.Tensor] = None,
    enable_es: Tuple[bool, bool] = (False, False),
) -> torch.Tensor:
    """Performs Fused MoE computation using CUTLASS-like kernels with FP8 weights and activations.

    This function implements a Mixture of Experts (MoE) layer with a SwiGLU/SiLU
    activation, leveraging custom kernels likely derived from CUTLASS principles
    for grouped matrix multiplication (`fp8_blockwise_scaled_grouped_mm`) and
    data preparation (`prepare_moe_input`, `silu_and_mul`).

    It handles per-token routing, quantizes input activations to FP8 with
    per-token scales, performs the expert computations using FP8 GEMMs with
    pre-quantized FP8 weights (per-block scales), applies the SiLU activation,
    and combines the results weighted by the router scores.

    Args:
        a (torch.Tensor): Input activations. Shape: `(m, k)`, where `m` is the total
            number of tokens and `k` is the hidden size. Expected dtype: `torch.half`
            or `torch.bfloat16`.
        w1_q (torch.Tensor): Pre-quantized FP8 weight tensor for the first GEMM
            (up-projection part of SwiGLU). Expected shape: `(E, k, n*2)`, where
            `E` is the number of experts, `k` is the hidden size, and `n*2` is the
            intermediate size (`I`). Expected dtype: `torch.float8_e4m3fn`.
            Note: This shape implies weights are stored as (num_experts, hidden_size, intermediate_size).
        w2_q (torch.Tensor): Pre-quantized FP8 weight tensor for the second GEMM
            (down-projection). Expected shape: `(E, n, k)`, where `n` is half the
            intermediate size (`I // 2`). Expected dtype: `torch.float8_e4m3fn`.
            Note: This shape implies weights are stored as (num_experts, intermediate_size // 2, hidden_size).
        w1_scale (torch.Tensor): Scales corresponding to `w1_q` (per-block scales).
            Shape: `(E, num_blocks_n, num_blocks_k)`. Dtype: `torch.float32`.
        w2_scale (torch.Tensor): Scales corresponding to `w2_q` (per-block scales).
             Shape: `(E, num_blocks_k, num_blocks_n)`. Dtype: `torch.float32`.
        topk_weights (torch.Tensor): Router weights for the selected top-k experts
            for each token. Shape: `(m, topk)`. Dtype should ideally match `a`.
        topk_ids (torch.Tensor): Indices of the selected top-k experts for each token.
            Shape: `(m, topk)`. Dtype: `torch.int32`.
        a1_strides (torch.Tensor): Stride information for the first GEMM's 'a' input.
            Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
            Note: Its exact usage within `fp8_blockwise_scaled_grouped_mm` needs clarification
            as it's passed as both a_stride and b_stride in the first call.
        c1_strides (torch.Tensor): Stride information for the first GEMM's 'c' output.
            Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
        a2_strides (torch.Tensor): Stride information for the second GEMM's 'a' input.
            Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
            Note: Its exact usage within `fp8_blockwise_scaled_grouped_mm` needs clarification
            as it's passed as both a_stride and b_stride in the second call.
        c2_strides (torch.Tensor): Stride information for the second GEMM's 'c' output.
            Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
        workspace (torch.Tensor): Reusable workspace for the underlying kernel.
        a_ptrs (torch.Tensor): Pointers container for calculating offsets of the input activations for each expert.
        b_ptrs (torch.Tensor): Pointers container for calculating offsets of the input weights for each expert.
        out_ptrs (torch.Tensor): Pointers container for calculating offsets of the output activations for each expert.
        a_scales_ptrs (torch.Tensor): Pointers container for calculating offsets of the input scales for each expert.
        b_scales_ptrs (torch.Tensor): Pointers container for calculating offsets of the input scales for each expert.
        use_fp8_blockscale (bool, optional): Flag indicating usage of FP8 with
            block scaling. Currently, only `True` is supported. Defaults to `True`.
        use_mxfp8 (bool, optional): Flag indicating usage of MXFP8 (UE8M0 scales)
            with SM100 expert-specialization kernels. Defaults to `False`.
        output (torch.Tensor, optional): Output tensor. If not provided, a new tensor will be created.
        enable_es (tuple(bool, bool)): Flag indicating usage of expert specialization kernel for (up-projection, down-projection)
    Returns:
        torch.Tensor: The computed MoE layer output. Shape: `(m, k)`, dtype matches `a`.

    Raises:
        AssertionError: If input shapes, dtypes, or flags are inconsistent or unsupported.
        NotImplementedError: If CUDA is not available or `sgl_kernel` is not properly installed.
    """

    assert use_fp8_blockscale, "Only support fp8 blockscale for now"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert (
        a.shape[1] == w1_q.shape[2]
    ), "Hidden size mismatch w1"  # XPU galng API change we don;t send transpose
    assert (
        w1_q.shape[1] == w2_q.shape[1] * 2
    ), "Hidden size mismatch w2"  # w1q = e, I, H  -silu> e, i/2, H : w2q = e, I//2, H
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

    # if is_cuda:
    #    from sglang.srt.layers.quantization.fp8_kernel import (
    #        sglang_per_token_group_quant_fp8,
    #    )

    es_up, es_down = enable_es
    out_dtype = a.dtype
    num_experts = w1_q.size(0)
    m = a.size(0)
    # k = w1_q.size(1)        # API changed orig w1 (E, n*2, k) -> T -> (E, k, n*2)
    k = w1_q.size(2)
    # n = w2_q.size(1)        # API changed orig w1 (E, k, n) -> T -> (E, n, k)
    n = w2_q.size(2)

    topk = topk_ids.size(1)
    device = a.device

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    # prepare_moe_input(      # CPU
    #    topk_ids,
    #    expert_offsets,
    #    problem_sizes1,
    #    problem_sizes2,
    #    a_map,
    #    c_map,
    #    num_experts,
    #    n,
    #    k,
    # )

    if use_mxfp8:
        assert es_up and es_down, "MXFP8 requires expert-specialization for both GEMMs"
        assert is_sm100_supported(), "MXFP8 requires SM100"
        assert k % 32 == 0, "MXFP8 requires hidden size to be divisible by 32"
        assert n % 32 == 0, "MXFP8 requires intermediate size to be divisible by 32"
        assert w1_scale.dtype == torch.uint8, "MXFP8 w1_scale must be uint8"
        assert w2_scale.dtype == torch.uint8, "MXFP8 w2_scale must be uint8"
        expected_w1_scale_shape = (
            num_experts,
            w1_q.shape[1] // 32,
            w1_q.shape[2],
        )
        expected_w2_scale_shape = (
            num_experts,
            w2_q.shape[1] // 32,
            w2_q.shape[2],
        )
        assert (
            w1_scale.shape == expected_w1_scale_shape
        ), f"MXFP8 w1_scale must be {expected_w1_scale_shape}, got {w1_scale.shape}"
        assert (
            w2_scale.shape == expected_w2_scale_shape
        ), f"MXFP8 w2_scale must be {expected_w2_scale_shape}, got {w2_scale.shape}"

        mxfp8_blockscale_align = 128
        total_tokens = m * topk
        nonzero_experts = min(num_experts, total_tokens)
        max_total = total_tokens + (mxfp8_blockscale_align - 1) * nonzero_experts
        max_blockscale = (
            (max_total + mxfp8_blockscale_align - 1) // mxfp8_blockscale_align
        ) * mxfp8_blockscale_align

    blockscale_offsets = None
    if use_mxfp8 and (es_up or es_down):
        blockscale_offsets = torch.empty(
            (num_experts + 1,), dtype=torch.int32, device=device
        )

    prepare_moe_input(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
        blockscale_offsets,
    )

    # XPU adapter: prepare_moe_input writes per-expert M counts into
    # expert_offsets; the kernel's flat-2D path needs cumulative starts.
    expert_starts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    if num_experts > 1:
        expert_starts[1:] = torch.cumsum(expert_offsets[:-1], dim=0).to(torch.int32)
    # Empty int64 sentinels select the on-device prep mode in the XPU kernel
    # (kernel builds the {5, E} ptr table and transposes A-scales internally).
    _xpu_empty_ptrs = torch.empty((0,), dtype=torch.int64, device=device)

    if use_mxfp8 and es_up:
        rep_a = shuffle_rows(a, a_map, (m * topk, k))
        rep_a_q = torch.empty_like(rep_a, dtype=torch.float8_e4m3fn)
        rep_a1_scales = torch.empty(
            (max_blockscale, k // 32), dtype=torch.uint8, device=device
        )
        es_sm100_mxfp8_blockscaled_grouped_quant(
            rep_a,
            problem_sizes1,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
            rep_a_q,
            rep_a1_scales,
        )
    else:
        # #a_q, a1_scale = sglang_per_token_group_quant_fp8(a, 128) # CPU
        # a_q, a1_scale = per_token_group_quant_fp8_ref(a, 128)

        # call V2 because we need A_SCALE in transpose for XPU GEMM
        # a_q, a1_scale = sglang_per_token_group_quant_8bit_layer(
        #     x=a,
        #     group_size=128,
        #     dst_dtype=torch.float8_e4m3fn,
        #     column_major_scales=True,  # Transpose scales to match expected layout for XPU GEMM
        #     enable_v2=True,
        # )
        # callV2 No Transpose
        a_q, a1_scale = sglang_per_token_group_quant_8bit_layer(
            x=a, group_size=128, dst_dtype=torch.float8_e4m3fn, enable_v2=True
        )

        # rep_a_q = shuffle_rows(a_q, a_map, (m * topk, k))
        # rep_a_q = scatter_tokens_to_experts_ref(a_q, a_map, m*topk, topk)
        rep_a_q = torch.empty((m * topk, k), dtype=a_q.dtype, device=device)
        scatter_tokens_to_experts(a_q, c_map, rep_a_q)

        # rep_a1_scales = shuffle_rows(a1_scale, a_map, (m * topk, int(k / 128)))
        # rep_a1_scales = scatter_tokens_to_experts_ref(a1_scale, a_map, m*topk, topk)
        rep_a1_scales = torch.empty(
            (m * topk, int(k / 128)), dtype=a1_scale.dtype, device=device
        )
        scatter_tokens_to_experts(a1_scale, c_map, rep_a1_scales)

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.float32)
    c2 = torch.empty((m * topk, k), device=device, dtype=torch.float32)

    a_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)
    w_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)

    if is_sm90_supported() and es_up:
        es_fp8_blockwise_scaled_grouped_mm(
            c1,
            rep_a_q,
            w1_q,
            rep_a1_scales,
            w1_scale,
            a1_strides,
            a1_strides,
            c1_strides,
            problem_sizes1,
            expert_offsets[:-1],
            workspace,
        )
    elif use_mxfp8 and es_up:
        es_sm100_mxfp8_blockscaled_grouped_mm(
            c1,
            rep_a_q,
            w1_q,
            rep_a1_scales,
            w1_scale,
            problem_sizes1,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
    else:
        fp8_blockwise_scaled_grouped_mm(
            c1,
            _xpu_empty_ptrs,  # a_ptrs sentinel (XPU on-device prep)
            _xpu_empty_ptrs,
            _xpu_empty_ptrs,
            _xpu_empty_ptrs,
            _xpu_empty_ptrs,
            rep_a_q,
            w1_q,
            rep_a1_scales,
            w1_scale,
            a1_strides,
            a1_strides,
            c1_strides,
            a_sf_layout,
            w_sf_layout,
            problem_sizes1,
            expert_starts,  # cumulative starts, not per-expert M counts
            workspace,
        )
    c1_with_out_dtype = c1.to(out_dtype)
    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
    silu_and_mul(c1_with_out_dtype, intermediate)

    if use_mxfp8 and es_down:
        intemediate_q = torch.empty_like(intermediate, dtype=torch.float8_e4m3fn)
        a2_scale = torch.empty(
            (max_blockscale, n // 32), dtype=torch.uint8, device=device
        )
        es_sm100_mxfp8_blockscaled_grouped_quant(
            intermediate,
            problem_sizes2,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
            intemediate_q,
            a2_scale,
        )
    else:
        # intemediate_q, a2_scale = sglang_per_token_group_quant_fp8(intermediate, 128)
        intemediate_q, a2_scale = sglang_per_token_group_quant_8bit_layer(
            x=intermediate,
            group_size=128,
            dst_dtype=torch.float8_e4m3fn,
            enable_v2=True,
        )
    if is_sm90_supported() and es_down:
        es_fp8_blockwise_scaled_grouped_mm(
            c2,
            intemediate_q,
            w2_q,
            a2_scale,
            w2_scale,
            a2_strides,
            a2_strides,
            c2_strides,
            problem_sizes2,
            expert_offsets[:-1],
            workspace,
        )
    elif use_mxfp8 and es_down:
        es_sm100_mxfp8_blockscaled_grouped_mm(
            c2,
            intemediate_q,
            w2_q,
            a2_scale,
            w2_scale,
            problem_sizes2,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
    else:
        fp8_blockwise_scaled_grouped_mm(
            c2,
            _xpu_empty_ptrs,
            _xpu_empty_ptrs,
            _xpu_empty_ptrs,
            _xpu_empty_ptrs,
            _xpu_empty_ptrs,
            intemediate_q,
            w2_q,
            a2_scale,
            w2_scale,
            a2_strides,
            a2_strides,
            c2_strides,
            a_sf_layout,
            w_sf_layout,
            problem_sizes2,
            expert_starts,
            workspace,
        )

    if output is None:
        output = torch.empty((m, k), device=device, dtype=out_dtype)

    c2_with_out_dtype = c2.to(out_dtype)
    apply_shuffle_mul_sum(c2_with_out_dtype, output, c_map, topk_weights.to(out_dtype))
    return output


# Generate unique token
def generate_unique_topk_ids(tokens, top_k, num_experts):
    topk_ids = torch.empty((tokens, top_k), dtype=torch.int32)
    # avoid duplicate tokens
    for T in range(tokens):
        topk_ids[T] = torch.randperm(num_experts, dtype=torch.int32)[:top_k]
    return topk_ids


def quantize_weight_2d(
    tensor: torch.Tensor,
    block_n: int = FP8_BLOCK_SIZE,
    block_k: int = FP8_BLOCK_SIZE,
) -> tuple:
    """Quantize (N, K) weight matrix with 2D block-wise scales.

    Returns (quantized [float8_e4m3fn], scales [N//block_n, K//block_k]).
    """
    assert tensor.dim() == 2, "Input must be 2D (N, K)"
    N, K = tensor.shape
    assert (
        N % block_n == 0 and K % block_k == 0
    ), f"N ({N}) and K ({K}) must be divisible by block sizes ({block_n}, {block_k})"

    tensor_fp32 = tensor.float()
    n_blocks_n = N // block_n
    n_blocks_k = K // block_k

    # Reshape to (n_blocks_n, block_n, n_blocks_k, block_k)
    blocked = tensor_fp32.reshape(n_blocks_n, block_n, n_blocks_k, block_k)

    # Get max absolute value per block
    block_amax = blocked.abs().amax(dim=(1, 3))  # Shape: (n_blocks_n, n_blocks_k)
    block_amax = torch.clamp(block_amax, min=1e-12)
    scales = (block_amax / FP8_E4M3_MAX).float()

    # Scale and quantize
    scale_expanded = scales.unsqueeze(1).unsqueeze(3)  # (n_n, 1, n_k, 1)
    scaled_blocks = blocked / scale_expanded
    clamped = scaled_blocks.clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    quantized = clamped.reshape(N, K).to(torch.float8_e4m3fn)

    return quantized, scales


def create_random_fp8_weight(N: int, K: int, seed: int):
    """Create random FP8 quantized weight with 2D block scales (CPU).

    Returns (quantized, scales) on CPU.
    """
    torch.manual_seed(seed)
    original = torch.randn(N, K, dtype=torch.float32) * 2.0
    return quantize_weight_2d(original)


def run_test(tp_size, batch_size, model_config, check=False):
    print(f"\n--- Batch Size: {batch_size} ---")
    torch.set_default_device("cpu")
    torch.manual_seed(42)
    # torch.manual_seed_all(42)  # For reproducible random numbers

    E = model_config["num_experts"]
    topk = model_config["topk"]
    H = model_config["hidden_size"]
    I = model_config["shard_intermediate_size"]
    block_shape = model_config["block_shape"]  # Tuple (BLOCK_N, BLOCK_K)
    dtype = model_config["dtype"]  # e.g., torch.bfloat16

    print(
        f"Config: E={E}, topk={topk}, H={H}, I_shard={I}, dtype={dtype}, block_shape={block_shape}"
    )

    # --- Input Data ---
    # Use bf16/fp16 for input activation based on model config
    x = torch.randn((batch_size, H), device="cpu", dtype=dtype)
    # --- Weights (Generate in higher precision, then convert to FP8) ---
    # Generate weights suitable for FP8 conversion (e.g., scaled appropriately)
    w1_hp = torch.randn((E, I, H), device="cpu", dtype=torch.float32)
    w2_hp = torch.randn((E, H, I // 2), device="cpu", dtype=torch.float32)

    w1 = to_fp8(w1_hp)
    w2 = to_fp8(w2_hp)

    # --- Scales for FP8 Weights ---
    block_n, block_k = block_shape
    # Calculate number of blocks needed
    w1_blocks_dim1 = (I + block_n - 1) // block_n
    w1_blocks_dim2 = (H + block_k - 1) // block_k
    w2_blocks_dim1 = (H + block_n - 1) // block_n
    w2_blocks_dim2 = (I // 2 + block_k - 1) // block_k

    # Scales are typically float32 or float16/bfloat16
    scale_dtype = torch.float32  # Or dtype if scales match model dtype
    w1_scale = torch.full(
        (E, w1_blocks_dim1, w1_blocks_dim2), 1, device="cpu", dtype=scale_dtype
    )  # Avoid zero scales
    w2_scale = torch.full(
        (E, w2_blocks_dim1, w2_blocks_dim2), 1, device="cpu", dtype=scale_dtype
    )  # Avoid zero scales

    # quantize weights
    w1_hp_full = torch.randn((E * I, H), device="cpu", dtype=torch.float32)
    w2_hp_full = torch.randn((E * H, I // 2), device="cpu", dtype=torch.float32)
    w1_full_fp8_2d, w1_full_scales_2d = quantize_weight_2d(w1_hp_full)
    w2_full_fp8_2d, w2_full_scales_2d = quantize_weight_2d(w2_hp_full)

    w1_scale_dummy = torch.randn((I, H), device="cpu", dtype=torch.float32)
    w2_scale_dummy = torch.randn((H, I // 2), device="cpu", dtype=torch.float32)
    w1_q_dummy, w1_sc_dummy = quantize_weight_2d(w1_scale_dummy)
    w2_q_dummy, w2_sc_dummy = quantize_weight_2d(w2_scale_dummy)
    w1_sc_dymmu_shapes = w1_sc_dummy.shape
    w2_sc_dymmu_shapes = w2_sc_dummy.shape

    w1_full_fp8 = w1_full_fp8_2d.view(E, I, H)
    w1_full_scales = w1_full_scales_2d.view(
        E, w1_sc_dymmu_shapes[0], w1_sc_dymmu_shapes[1]
    )
    w2_full_fp8 = w2_full_fp8_2d.view(E, H, I // 2)
    w2_full_scales = w2_full_scales_2d.view(
        E, w2_sc_dymmu_shapes[0], w2_sc_dymmu_shapes[1]
    )

    assert (
        w1_full_fp8.shape == w1.shape
    ), f"Shape mismatch: {w1_full_fp8.shape} vs {w1.shape}"
    assert (
        w1_full_fp8.dtype == w1.dtype
    ), f"Dtype mismatch: {w1_full_fp8.dtype} vs {w1.dtype}"
    assert (
        w1_full_fp8.device == w1.device
    ), f"Device mismatch: {w1_full_fp8.device} vs {w1.device}"

    assert (
        w2_full_fp8.shape == w2.shape
    ), f"Shape mismatch: {w2_full_fp8.shape} vs {w2.shape}"
    assert (
        w2_full_fp8.dtype == w2.dtype
    ), f"Dtype mismatch: {w2_full_fp8.dtype} vs {w2.dtype}"
    assert (
        w2_full_fp8.device == w2.device
    ), f"Device mismatch: {w2_full_fp8.device} vs {w2.device}"

    assert (
        w1_full_scales.shape == w1_scale.shape
    ), f"Shape mismatch: {w1_full_scales.shape} vs {w1_scale.shape}"
    assert (
        w1_full_scales.dtype == w1_scale.dtype
    ), f"Dtype mismatch: {w1_full_scales.dtype} vs {w1_scale.dtype}"
    assert (
        w1_full_scales.device == w1_scale.device
    ), f"Device mismatch: {w1_full_scales.device} vs {w1_scale.device}"

    assert (
        w2_full_scales.shape == w2_scale.shape
    ), f"Shape mismatch: {w2_full_scales.shape} vs {w2_scale.shape}"
    assert (
        w2_full_scales.dtype == w2_scale.dtype
    ), f"Dtype mismatch: {w2_full_scales.dtype} vs {w2_scale.dtype}"
    assert (
        w2_full_scales.device == w2_scale.device
    ), f"Device mismatch: {w2_full_scales.device} vs {w2_scale.device}"

    # --- Routing Information ---
    topk_weights = torch.softmax(
        torch.rand(batch_size, topk, device="cpu", dtype=dtype), dim=-1
    )
    topk_ids = torch.randint(0, E, (batch_size, topk), dtype=torch.int32, device="cpu")
    topk_ids = generate_unique_topk_ids(batch_size, topk, E)
    print(topk_ids.shape)

    a1_strides = torch.full((E,), H, dtype=torch.int64, device="cpu")
    c1_strides = torch.full((E,), I, dtype=torch.int64, device="cpu")
    a2_strides = torch.full((E,), I // 2, dtype=torch.int64, device="cpu")
    c2_strides = torch.full((E,), H, dtype=torch.int64, device="cpu")

    workspace = torch.empty(
        (7182 * 1024), device="cpu", dtype=torch.uint8
    )  # Allocate sufficient workspace
    # Pointer arrays (often filled by the kernel or a prep step, but needed as args)
    a_ptrs = torch.empty((E,), dtype=torch.uint64, device="cpu")
    b_ptrs = torch.empty((E,), dtype=torch.uint64, device="cpu")
    out_ptrs = torch.empty((E,), dtype=torch.uint64, device="cpu")
    a_scales_ptrs = torch.empty((E,), dtype=torch.uint64, device="cpu")
    b_scales_ptrs = torch.empty((E,), dtype=torch.uint64, device="cpu")
    # XPU contract: prepare_moe_input writes into a size-E tensor (per-expert
    # M counts), and the kernel asserts problem_sizes is (E, 3).
    expert_offsets = torch.empty((E,), dtype=torch.int32, device="cpu")
    problem_sizes1 = torch.zeros((E, 3), dtype=torch.int32)
    problem_sizes2 = torch.zeros((E, 3), dtype=torch.int32)

    enable_es = (False, False)
    # if torch.cuda.get_device_name(torch.cuda.current_device()) == "NVIDIA H200":
    #    enable_es = (False, True)
    # elif torch.cuda.get_device_name(torch.cuda.current_device()) == "NVIDIA H20":
    #    enable_es = (True, True)

    x = x.clone().to("xpu")
    w1_full_fp8 = w1_full_fp8.clone().to("xpu")
    w2_full_fp8 = w2_full_fp8.clone().to("xpu")
    w1_full_scales = w1_full_scales.clone().to("xpu")
    w2_full_scales = w2_full_scales.clone().to("xpu")
    topk_weights = topk_weights.clone().to("xpu")
    topk_ids = topk_ids.clone().to("xpu")
    a1_strides = a1_strides.clone().to("xpu")
    c1_strides = c1_strides.clone().to("xpu")
    a2_strides = a2_strides.clone().to("xpu")
    c2_strides = c2_strides.clone().to("xpu")
    workspace = workspace.clone().to("xpu")
    a_ptrs = a_ptrs.clone().to("xpu")
    b_ptrs = b_ptrs.clone().to("xpu")
    out_ptrs = out_ptrs.clone().to("xpu")
    a_scales_ptrs = a_scales_ptrs.clone().to("xpu")
    b_scales_ptrs = b_scales_ptrs.clone().to("xpu")
    expert_offsets = expert_offsets.clone().to("xpu")
    problem_sizes1 = problem_sizes1.clone().to("xpu")
    problem_sizes2 = problem_sizes2.clone().to("xpu")

    # --- Lambdas for Benchmarking ---
    cutlass_lambda = lambda: cutlass_fused_experts_fp8(
        x,
        # w1.transpose(1, 2),  # Transposed                          API change send without transpose for XPU
        w1_full_fp8,
        # w2.transpose(1, 2),  # Transposed
        w2_full_fp8,
        # w1_scale.transpose(1, 2),
        w1_full_scales,
        # w2_scale.transpose(1, 2),
        w2_full_scales,
        topk_weights,
        topk_ids,
        a1_strides,
        c1_strides,
        a2_strides,
        c2_strides,
        workspace,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        enable_es=enable_es,
    )

    # topk_output = StandardTopKOutput(
    #    topk_weights=topk_weights,
    #    topk_ids=topk_ids,
    #    router_logits=torch.randn(
    #        (batch_size, topk), device=topk_weights.device, dtype=dtype
    #    ),
    # )

    # moe_runner_config = MoeRunnerConfig(
    #    num_experts=E,
    #    top_k=topk,
    #    hidden_size=H,
    #    intermediate_size_per_partition=I,
    #    params_dtype=dtype,
    #    activation="silu",
    #    inplace=False,
    # )

    # Note: Triton expects non-transposed weights
    # triton_lambda = lambda: fused_experts(
    #    x,
    #    w1,
    #    w2,
    #    topk_output,
    #    moe_runner_config,
    #    use_fp8_w8a8=True,
    #    w1_scale=w1_scale,
    #    w2_scale=w2_scale,
    #    block_shape=block_shape,
    # )

    # --- Warmup ---
    print("Warming up...")
    for _ in range(10):
        _ = cutlass_lambda()
        # _ = triton_lambda()
    # torch.cuda.synchronize()
    # --- Benchmarking ---
    quantiles = [0.5, 0.2, 0.8]
    print(f"Benchmarking Cutlass fused_experts...")
    cutlass_ms, cutlass_min, cutlass_max = triton.testing.do_bench_cudagraph(
        cutlass_lambda, rep=1000, quantiles=quantiles
    )

    print(f"Benchmarking Triton fused_experts...")
    triton_ms, triton_min, triton_max = triton.testing.do_bench_cudagraph(
        triton_lambda, rep=1000, quantiles=quantiles
    )
    print(
        f"Cutlass fused_experts time: {cutlass_ms:.3f} ms (median) [{cutlass_min:.3f} - {cutlass_max:.3f}]"
    )
    print(
        f"Triton  fused_experts time: {triton_ms:.3f} ms (median) [{triton_min:.3f} - {triton_max:.3f}]"
    )

    # --- Correctness Check ---
    if check:
        print("Running correctness check...")
        with torch.no_grad():
            # Run CUTLASS version (requires transposed weights)
            y_cutlass = cutlass_fused_experts_fp8(
                x,
                w1.transpose(1, 2),  # Transposed
                w2.transpose(1, 2),  # Transposed
                w1_scale.transpose(1, 2),
                w2_scale.transpose(1, 2),
                topk_weights,
                topk_ids,
                a1_strides,
                c1_strides,
                a2_strides,
                c2_strides,
                workspace,
                a_ptrs,
                b_ptrs,
                out_ptrs,
                a_scales_ptrs,
                b_scales_ptrs,
                expert_offsets,
                problem_sizes1,
                problem_sizes2,
                enable_es=enable_es,
            )

            # Run Triton version (requires original shape weights, use inplace=False)
            y_triton = fused_experts(
                x,
                w1,  # Original shape
                w2,  # Original shape
                topk_output,
                moe_runner_config,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                block_shape=block_shape,
            )

        diff = calc_diff(y_cutlass, y_triton)
        print(f"Diff: {diff:.6f}")

        # Tolerance might need adjustment based on FP8 specifics and kernel differences
        # FP8 comparisons often require higher tolerance than FP16/BF16
        assert diff < 1e-4, f"Diff too high! {diff}"
        print("Correctness check passed.")


def main(tp_size=8, batch_sizes=[5, 4, 8, 16, 32, 64, 128, 256, 512], check=False):
    # model_config = get_model_config(tp_size)       # this works in local network
    # model_config = {'num_experts': 256, 'topk': 8, 'hidden_size': 7168, 'shard_intermediate_size': 512, 'dtype': torch.bfloat16, 'block_shape': [128, 128]}
    model_config = {
        "num_experts": 4,
        "topk": 4,
        "hidden_size": 512,
        "shard_intermediate_size": 1024,
        "dtype": torch.bfloat16,
        "block_shape": [128, 128],
    }
    print("Model Config:", model_config)
    for batch_size in batch_sizes:
        run_test(tp_size, batch_size, model_config, check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=8, help="Tensor Parallel size")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
        ],  # Adjusted default
        help="List of batch sizes to test",
    )
    parser.add_argument("--check", action="store_true", help="Enable check mode")
    args = parser.parse_args()

    print(f"Running benchmarks with TP size: {args.tp_size}")
    print(f"Testing batch sizes: {args.batch_sizes}")

    main(tp_size=args.tp_size, batch_sizes=args.batch_sizes, check=args.check)
