import torch


def weak_ref_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if not tensor.is_xpu():
        return tensor

    return torch.ops.sgl_kernel.weak_ref_tensor(tensor)
