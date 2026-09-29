import functools

import torch

FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
HEAD_DIM = 128  # kernel contracts q_input/q_fp8 as [B, H, 128]
ROPE_DIM = 64  # kernel contracts rope_cache as [max_pos, 64]
NOPE_DIM = HEAD_DIM - ROPE_DIM


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    return device


@functools.lru_cache(maxsize=1)
def is_xe2_device() -> bool:
    if not torch.xpu.is_available():
        return False
    from sgl_kernel.utils import is_xe2_arch

    return is_xe2_arch()


def get_reference_device():
    """Reference device for eager PyTorch baselines."""
    return get_device() if is_xe2_device() else torch.device("cpu")


precision = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}
