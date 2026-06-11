"""
Row-wise reduction operations
"""

import torch


def row_wise_square_sum(input: torch.Tensor) -> torch.Tensor:
    """
    Compute row-wise square sum using CUTLASS-style kernel.

    Computes sum(x^2) for each row.

    Args:
        input: Input tensor of shape [M, N]

    Returns:
        Output tensor of shape [M] containing row square sums

    Example:
        >>> A = torch.randn(128, 512, device='xpu')
        >>> row_square_sums = row_wise_square_sum(A)
        >>> assert row_square_sums.shape == (128,)
        >>> # Equivalent to: (A ** 2).sum(dim=1)
    """
    assert input.ndim == 2, f"Input must be 2D, got {input.ndim}D"
    assert input.device.type == "xpu", f"Input must be on XPU, got {input.device.type}"

    M, N = input.shape
    output = torch.empty(M, dtype=input.dtype, device=input.device)

    torch.ops.sgl_kernel.row_wise_sum_cutlass.default(input, output)

    return output
