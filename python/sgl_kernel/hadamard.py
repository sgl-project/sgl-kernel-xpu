import torch


def hadamard_transform(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Pure-torch FWHT fallback for backends without a fused kernel.

    Iterative Cooley-Tukey-style Walsh-Hadamard transform along the last
    dim. Hidden size must be a power of two; same contract as the fused
    ``hadamard_transform`` op.
    """
    n = x.size(-1)
    leading = x.shape[:-1]
    out = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        out = out.view(-1, n // (2 * h), 2, h)
        a = out[:, :, 0, :]
        b = out[:, :, 1, :]
        out = torch.stack((a + b, a - b), dim=2).view(-1, n)
        h *= 2
    return out.view(*leading, n) * scale
