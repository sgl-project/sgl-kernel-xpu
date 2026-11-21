"""Shared FP8 (E4M3) weight quantization helpers for tests and benchmarks.

Each helper returns ``(w_fp8, scale, w_dq)``: the FP8 weight, its scales in the
layout ``fp8_linear`` takes, and the dequantized weight in fp32 (the reference
the tests compare against; benchmarks ignore it). Scales are clamped away from
zero so an all-zero row, channel or block does not divide by zero.
"""

import torch

FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_BLOCK = 128
_MIN_SCALE = 1e-12


def quantize_fp8_per_tensor(w: torch.Tensor):
    """One scale for the whole weight; scale shape [1]."""
    scale = (w.abs().max().float() / FP8_E4M3_MAX).clamp(min=_MIN_SCALE)
    w_fp8 = (w.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = w_fp8.to(torch.float8_e4m3fn)
    return w_fp8, scale.reshape(1), w_fp8.float() * scale


def quantize_fp8_per_channel(w: torch.Tensor):
    """One scale per output channel (row of the [N, K] weight); scale shape [N]."""
    scale = (w.abs().amax(dim=1).float() / FP8_E4M3_MAX).clamp(min=_MIN_SCALE)
    w_fp8 = (w.float() / scale[:, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = w_fp8.to(torch.float8_e4m3fn)
    return w_fp8, scale, w_fp8.float() * scale[:, None]


def quantize_fp8_block(w: torch.Tensor, block: int = FP8_BLOCK):
    """128x128 block scales; scale shape [ceil(N / 128), K / 128].

    K must be a multiple of the block. A partial last block of N is padded with
    zeros for its scale and cropped from the returned weight.
    """
    n, k = w.shape
    n_blocks = (n + block - 1) // block
    padded = torch.zeros(n_blocks * block, k, dtype=torch.float32, device=w.device)
    padded[:n] = w.float()
    blocks = padded.view(n_blocks, block, k // block, block)
    scale = (blocks.abs().amax(dim=(1, 3)) / FP8_E4M3_MAX).clamp(min=_MIN_SCALE)
    expanded = scale[:, None, :, None]
    w_fp8 = (blocks / expanded).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = w_fp8.to(torch.float8_e4m3fn)
    w_dq = (w_fp8.float() * expanded).view(n_blocks * block, k)[:n]
    w_fp8 = w_fp8.view(n_blocks * block, k)[:n].contiguous()
    return w_fp8, scale.contiguous(), w_dq
