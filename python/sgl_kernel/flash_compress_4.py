from __future__ import annotations

import torch


def flash_compress4_prefill(
    kv_buffer: torch.Tensor,  # [num_pages, 4, head_dim*4]
    kv_input: torch.Tensor,  # [num_q, head_dim*4]
    kv_output: torch.Tensor,  # [C, head_dim]
    ape: torch.Tensor,  # [8, head_dim]
    plan_c_u8: torch.Tensor,  # [C, 16]
    plan_w_u8: torch.Tensor,  # [W, 8]
) -> None:
    torch.ops.sgl_kernel.flash_compress4_prefill(
        kv_buffer, kv_input, kv_output, ape, plan_c_u8, plan_w_u8
    )


def flash_compress4_decode(
    kv_buffer: torch.Tensor,  # [num_pages, 4, head_dim*4]
    kv_input: torch.Tensor,  # [B, head_dim*4]
    kv_output: torch.Tensor,  # [B, head_dim]
    ape: torch.Tensor,  # [8, head_dim]
    plan_d_u8: torch.Tensor,  # [B, 16]
) -> None:
    torch.ops.sgl_kernel.flash_compress4_decode(
        kv_buffer, kv_input, kv_output, ape, plan_d_u8
    )
