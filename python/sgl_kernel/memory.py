import torch


def weak_ref_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        return tensor

    return torch.ops.sgl_kernel.weak_ref_tensor(tensor)
