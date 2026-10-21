from typing import Optional

import torch


def _bf16_gemm_nt_use_cute_xe35(
    mat_a: torch.Tensor, m: int, n: int, k: int, packed_weight: bool
) -> bool:
    major, minor = torch.ops.sgl_kernel.query_device.default(mat_a.device.index or 0)
    if (major, minor) != (3, 5):
        return False
    if packed_weight:
        if (n, k) in ((4608, 8192), (4096, 8192), (8192, 4096), (8448, 4096)):
            return 0 < m <= 32
        if (n, k) in ((8192, 8192), (8448, 8192)):
            return 0 < m <= 16
        if (n, k) == (18432, 8192):
            return 0 < m <= 96
        if (n, k) == (202240, 8192):
            return 0 < m <= 16 or 33 <= m <= 79
    if not packed_weight:
        if (n, k) == (8192, 4096) and 65 <= m <= 128:
            return True
        if (n, k) == (4608, 8192) and 33 <= m <= 64:
            return True
        if (n, k) == (202048, 8192) and (65 <= m <= 128 or 193 <= m <= 256):
            return True
        # Paired native-B timings cross sharply at M=8/9. Around 4K, the
        # crossover also follows K tile alignment and weight size; use the
        # measured contiguous regions rather than individual model widths.
        if not 0 < m <= 8:
            return False
        weight_elems = n * k
        if k % 1024 == 0 and 3 * 1024 <= k <= 4 * 1024:
            min_elems = 20 * 1024 * 1024 if k >= 4 * 1024 else 30 * 1024 * 1024
            return min_elems <= weight_elems <= 100 * 1024 * 1024
        if (
            k % 16 != 0
            or not 4 * 1024 - 96 <= k <= 4 * 1024 + 96
            or weight_elems > 82 * 1024 * 1024
        ):
            return False
        # A power-of-two row count avoids the N crossover at this smaller
        # weight size. Neighboring non-power-of-two N favored torch.mm.
        if (
            (n & (n - 1)) == 0
            and abs(k - 4 * 1024) <= 16
            and weight_elems >= 31 * 1024 * 1024
        ):
            return True
        if abs(k - 4 * 1024) <= 16 and weight_elems >= 43 * 1024 * 1024:
            return True
        if k % 64 == 32 and weight_elems >= 39 * 1024 * 1024:
            return True
        # The edge K tiles have a short throughput dip just above 16K rows.
        min_large_elems = 65 * 1024 * 1024 if k % 64 == 48 else 66 * 1024 * 1024
        return (
            abs(k - 4 * 1024) <= 48
            and weight_elems >= 47 * 1024 * 1024
            and (n <= 16 * 1024 or weight_elems >= min_large_elems)
        )
    # On this Xe35 device, paired BF16 graph timings favor CuTe for packed
    # weights above roughly 200 MiB. Smaller weights and power-of-two row
    # strides favor torch.mm. The M bounds follow the measured kernel
    # crossover; larger M needs a smaller weight to stay ahead of torch.mm.
    if m <= 0 or n < 14336 or n * k < 100 * 1024 * 1024 or n & (n - 1) == 0:
        return False
    if m <= 16 or 33 <= m <= 80:
        return True
    return 81 <= m <= 128 and n >= 16384 and n * k <= 200 * 1024 * 1024


def _bf16_gemm_nt_torch_fallback_xe35(
    mat_a: torch.Tensor, mat_b: torch.Tensor, packed_weight: bool
) -> torch.Tensor:
    if packed_weight:
        return torch.mm(mat_a, mat_b)
    if (
        mat_a.device.type == "xpu"
        and mat_a.dtype == torch.bfloat16
        and mat_b.dtype == torch.bfloat16
    ):
        return torch.mm(mat_a, mat_b.T, out_dtype=torch.bfloat16)
    return torch.mm(mat_a, mat_b.T)


def bf16_gemm_nt_xe35(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    *,
    packed_weight: bool = False,
    output_n: Optional[int] = None,
) -> torch.Tensor:
    """Compute an XPU BF16 GEMM with a model-native or prepacked weight.

    ``mat_b`` is ``[N, K]`` by default. When ``packed_weight`` is true it is
    a contiguous ``[K, padded_N]`` tensor produced by
    :func:`pack_bf16_gemm_weight_xe35`. ``output_n`` selects the logical output
    width while retaining the packed weight's physical stride.
    """
    assert mat_a.ndim == 2 and mat_b.ndim == 2
    k = mat_b.shape[0] if packed_weight else mat_b.shape[1]
    n = mat_b.shape[1] if packed_weight else mat_b.shape[0]
    assert mat_a.shape[1] == k
    if output_n is not None and (not packed_weight or not 0 < output_n <= n):
        raise ValueError(
            "output_n requires a packed weight and must be within its width"
        )
    logical_n = output_n if output_n is not None else n
    fallback_b = mat_b[:, :logical_n] if packed_weight and logical_n != n else mat_b
    if (
        mat_a.device.type != "xpu"
        or mat_a.dtype != torch.bfloat16
        or mat_b.dtype != torch.bfloat16
        or not mat_a.is_contiguous()
        or not mat_b.is_contiguous()
        or k % 16 != 0
        or n % 16 != 0
        or logical_n % 16 != 0
    ):
        return _bf16_gemm_nt_torch_fallback_xe35(mat_a, fallback_b, packed_weight)
    if not _bf16_gemm_nt_use_cute_xe35(mat_a, mat_a.shape[0], n, k, packed_weight):
        return _bf16_gemm_nt_torch_fallback_xe35(mat_a, fallback_b, packed_weight)
    out = torch.empty(
        (mat_a.shape[0], logical_n), dtype=torch.bfloat16, device=mat_a.device
    )
    torch.ops.sgl_kernel.bf16_gemm_nt_xe35.default(
        out, mat_a, mat_b, 2 if packed_weight else 0
    )
    return out


def pack_bf16_gemm_weight_xe35(weight: torch.Tensor) -> torch.Tensor:
    """Pack a contiguous ``[N, K]`` BF16 weight as padded ``[K, N]``."""
    assert weight.ndim == 2
    assert weight.dtype == torch.bfloat16
    assert weight.is_contiguous()
    n, k = weight.shape
    padded_n = (n + 255) // 256 * 256
    # Power-of-two row strides from 16K onward show a repeatable packed GEMM
    # throughput cliff on this Xe35 device. One more block avoids it.
    power_of_two_stride = padded_n >= 16384 and padded_n & (padded_n - 1) == 0
    if power_of_two_stride:
        padded_n += 256
    # For very large weights with an N that is not 32-aligned, writing into a
    # sliced packed tensor is much slower than padding before the transpose.
    if power_of_two_stride or (n * k >= 1 << 30 and n % 32 != 0):
        return torch.nn.functional.pad(weight, (0, 0, 0, padded_n - n)).T.contiguous()
    # Materialize the transpose before writing into a padded slice. XPU
    # copy_ from a transposed source can leave isolated columns unwritten.
    transposed = weight.T.contiguous()
    if padded_n == n:
        return transposed
    packed = torch.zeros((k, padded_n), dtype=weight.dtype, device=weight.device)
    packed[:, :n].copy_(transposed)
    return packed
