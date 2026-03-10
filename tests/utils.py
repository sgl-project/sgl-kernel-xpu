import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    return device


def is_sm10x():
    return torch.cuda.get_device_capability() >= (10, 0)


def is_hopper():
    return torch.cuda.get_device_capability() == (9, 0)
