import sys
from typing import List, Optional

import pytest
import torch
from sgl_kernel import chunked_sgmv_lora_expand_fwd

if not torch.xpu.is_available():
    pytest.skip(
        reason="chunked_sgmv_lora_expand_fwd requires XPU device.",
        allow_module_level=True,
    )


def _tolerances(dtype: torch.dtype):
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-5, 1e-5


# ----------------------------------------------------------------------------
# Helpers (mirror the three-kernel decomposition on CPU/XPU in fp32)
# ----------------------------------------------------------------------------


def _slice_offsets(slice_sizes: List[int]) -> torch.Tensor:
    """Build the [num_slices + 1] output-column boundary table."""
    offs = torch.zeros(len(slice_sizes) + 1, dtype=torch.int32)
    offs[1:] = torch.cumsum(torch.tensor(slice_sizes, dtype=torch.int32), dim=0)
    return offs


def _zero_weight_rank_tail(
    weights: torch.Tensor, lora_ranks: torch.Tensor
) -> torch.Tensor:
    """Zero LoRA-B weight columns (rank dim) beyond ``lora_ranks[lora]`` (kernel contract)."""
    num_loras = weights.size(0)
    out = weights.clone()
    ranks_cpu = lora_ranks.cpu()
    for l in range(num_loras):
        r = int(ranks_cpu[l].item())
        out[l, :, r:] = 0
    return out


def _reference_expand_sgemm(
    input_x: torch.Tensor,
    weights: torch.Tensor,
    slice_offsets: torch.Tensor,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    scalings: torch.Tensor,
    max_rank: int,
    base_output: Optional[torch.Tensor],
) -> torch.Tensor:
    """Per-slice segmented grouped GEMM (scaled + residual), computed on-device in fp32.

    Assumes rows are already in *logical* (adapter-grouped) order.
    """
    device = input_x.device
    num_tokens = input_x.size(0)
    n_total = weights.size(1)
    num_slices = slice_offsets.numel() - 1

    out = torch.zeros((num_tokens, n_total), dtype=torch.float32, device=device)

    seg_cpu = seg_indptr.cpu()
    wi_cpu = weight_indices.cpu()
    so_cpu = slice_offsets.cpu()
    scal_cpu = scalings.float().cpu()

    for s in range(seg_cpu.numel() - 1):
        start = int(seg_cpu[s].item())
        end = int(seg_cpu[s + 1].item())
        if end == start:
            continue
        lora = int(wi_cpu[s].item())
        alpha = float(scal_cpu[lora].item())
        for p in range(num_slices):
            col0 = int(so_cpu[p].item())
            col1 = int(so_cpu[p + 1].item())
            x = input_x[start:end, p * max_rank : (p + 1) * max_rank].float()
            w = weights[lora, col0:col1, :].float()
            out[start:end, col0:col1] = alpha * (x @ w.T)

    if base_output is not None:
        out = out + base_output.float()
    return out.to(weights.dtype)


def _reference_chunked_expand(
    input_x: torch.Tensor,
    weights: torch.Tensor,
    slice_offsets: torch.Tensor,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    scalings: torch.Tensor,
    max_rank: int,
    base_output: Optional[torch.Tensor],
    permutation: torch.Tensor,
) -> torch.Tensor:
    """gather (physical->logical) -> scaled sgemm (+residual) -> scatter (logical->physical)."""
    perm = permutation.to(torch.int64)
    x_sorted = input_x[perm]
    base_sorted = base_output[perm] if base_output is not None else None
    out_sorted = _reference_expand_sgemm(
        x_sorted,
        weights,
        slice_offsets,
        seg_indptr,
        weight_indices,
        scalings,
        max_rank,
        base_sorted,
    )
    out = torch.empty_like(out_sorted)
    out[perm] = out_sorted
    return out


def _build_logical_segments(row_adapters: List[int], permutation_mode: str = "sorted"):
    """Build (permutation, seg_indptr, weight_indices); mirrors the shrink test."""
    ra = torch.tensor(row_adapters, dtype=torch.int32)
    if permutation_mode == "sorted":
        permutation = torch.argsort(ra, stable=True).to(torch.int32)
        logical = ra[permutation.to(torch.int64)]
    elif permutation_mode == "identity":
        permutation = torch.arange(ra.numel(), dtype=torch.int32)
        logical = ra
    elif permutation_mode == "none":
        permutation = None
        logical = ra
    else:
        raise ValueError(f"unknown permutation_mode: {permutation_mode!r}")

    uniq, counts = torch.unique_consecutive(logical, return_counts=True)
    seg_indptr = torch.zeros(uniq.numel() + 1, dtype=torch.int32)
    seg_indptr[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    weight_indices = uniq.to(torch.int32)
    return permutation, seg_indptr, weight_indices


# ----------------------------------------------------------------------------
# chunked_sgmv_lora_expand_fwd (three-kernel orchestration)
# ----------------------------------------------------------------------------


def _run_and_compare_chunked(
    *,
    dtype: torch.dtype,
    row_adapters: List[int],
    slice_sizes: List[int],
    max_rank: int,
    num_loras: int,
    lora_ranks: torch.Tensor,
    scalings: torch.Tensor,
    permutation_mode: str = "sorted",
    with_base_output: bool = False,
) -> None:
    torch.manual_seed(0)
    num_tokens = len(row_adapters)
    num_slices = len(slice_sizes)
    n_total = sum(slice_sizes)
    max_slice_size = max(slice_sizes)

    permutation, seg_indptr, weight_indices = _build_logical_segments(
        row_adapters, permutation_mode
    )
    if permutation is not None:
        permutation = permutation.to("xpu")
    seg_indptr = seg_indptr.to("xpu")
    weight_indices = weight_indices.to("xpu")
    slice_offsets = _slice_offsets(slice_sizes).to("xpu")

    input_x = torch.randn(num_tokens, num_slices * max_rank, dtype=dtype, device="xpu")
    weights = torch.randn(num_loras, n_total, max_rank, dtype=dtype, device="xpu")
    weights = _zero_weight_rank_tail(weights, lora_ranks)
    base_output = (
        torch.randn(num_tokens, n_total, dtype=dtype, device="xpu")
        if with_base_output
        else None
    )

    out = chunked_sgmv_lora_expand_fwd(
        input_x=input_x,
        weights=weights,
        slice_offsets=slice_offsets,
        max_slice_size=max_slice_size,
        num_slices=num_slices,
        num_segments=weight_indices.numel(),
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        permutation=permutation,
        base_output=base_output,
    )

    if permutation is None:
        ref = _reference_expand_sgemm(
            input_x,
            weights,
            slice_offsets,
            seg_indptr,
            weight_indices,
            scalings,
            max_rank,
            base_output,
        )
    else:
        ref = _reference_chunked_expand(
            input_x,
            weights,
            slice_offsets,
            seg_indptr,
            weight_indices,
            scalings,
            max_rank,
            base_output,
            permutation,
        )

    assert out.shape == (num_tokens, n_total)
    assert out.dtype == dtype
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


# One data-driven test over named scenarios. Each scenario overrides only the
# params it exercises; everything else falls back to _DEFAULT_SCENARIO.
# ``lora_ranks=None`` means "full rank for every adapter" ([max_rank]*num_loras).
_DEFAULT_SCENARIO = {
    "slice_sizes": [256],
    "max_rank": 8,
    "num_loras": 2,
    "row_adapters": [0, 1, 0, 1, 0, 1, 0, 1],
    "lora_ranks": None,
    "scalings": None,
    "permutation_mode": "sorted",
    "with_base_output": False,
}

_MANY_ADAPTERS = torch.randint(
    0, 4, (512,), generator=torch.Generator().manual_seed(7)
).tolist()

_ZIGZAG = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]

_SCENARIOS = [
    # Interleaved (zigzag) decode adapters, swept over N x max_rank.
    pytest.param(
        {
            "num_loras": 3,
            "row_adapters": _ZIGZAG,
            "slice_sizes": [64],
            "max_rank": 8,
            "lora_ranks": [8, 4, 2],
        },
        id="zigzag-N64-r8",
    ),
    pytest.param(
        {
            "num_loras": 3,
            "row_adapters": _ZIGZAG,
            "slice_sizes": [4096],
            "max_rank": 64,
            "lora_ranks": [64, 32, 16],
        },
        id="zigzag-N4096-r64",
    ),
    # Stacked projections: o_proj=1, gate_up=2 (uniform), qkv=3 (uneven).
    pytest.param({"slice_sizes": [512], "lora_ranks": [8, 4]}, id="slice1"),
    pytest.param({"slice_sizes": [512, 512], "lora_ranks": [8, 4]}, id="gate_up"),
    pytest.param(
        {"slice_sizes": [512, 128, 128], "lora_ranks": [8, 4]}, id="qkv-uneven"
    ),
    # Fused residual add (base_output).
    pytest.param(
        {"slice_sizes": [256, 256], "lora_ranks": [8, 4], "with_base_output": True},
        id="base-output",
    ),
    # A rank-0 adapter must yield an all-zero LoRA term for its rows.
    pytest.param(
        {
            "row_adapters": [0, 1, 0, 1, 0, 1],
            "slice_sizes": [128],
            "lora_ranks": [0, 8],
        },
        id="zero-rank-adapter",
    ),
    # Many tokens across many adapters, with residual.
    pytest.param(
        {
            "num_loras": 4,
            "max_rank": 16,
            "slice_sizes": [256, 256],
            "row_adapters": _MANY_ADAPTERS,
            "lora_ranks": [1, 4, 8, 16],
            "with_base_output": True,
        },
        id="many-adapters",
    ),
    # permutation=None prefill fast path (rows pre-grouped, GEMM in place).
    pytest.param(
        {
            "num_loras": 3,
            "row_adapters": [0] * 16 + [2] * 32 + [1] * 16,
            "slice_sizes": [512, 512],
            "lora_ranks": [8, 4, 2],
            "permutation_mode": "none",
            "with_base_output": True,
        },
        id="permutation-none",
    ),
    # Identity permutation drives the gather/scatter path as a no-op remap.
    pytest.param(
        {
            "row_adapters": [0] * 16 + [1] * 16,
            "slice_sizes": [128],
            "lora_ranks": [8, 4],
            "permutation_mode": "identity",
        },
        id="identity-permutation",
    ),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_chunked_expand(dtype, scenario):
    cfg = {**_DEFAULT_SCENARIO, **scenario}
    num_loras = cfg["num_loras"]
    max_rank = cfg["max_rank"]
    ranks = (
        cfg["lora_ranks"] if cfg["lora_ranks"] is not None else [max_rank] * num_loras
    )
    lora_ranks = torch.tensor(ranks, dtype=torch.int32, device="xpu")
    scale_vals = (
        cfg["scalings"]
        if cfg["scalings"] is not None
        else [0.5 + 0.25 * i for i in range(num_loras)]
    )
    scalings = torch.tensor(scale_vals, dtype=torch.float32, device="xpu")
    _run_and_compare_chunked(
        dtype=dtype,
        row_adapters=cfg["row_adapters"],
        slice_sizes=cfg["slice_sizes"],
        max_rank=max_rank,
        num_loras=num_loras,
        lora_ranks=lora_ranks,
        scalings=scalings,
        permutation_mode=cfg["permutation_mode"],
        with_base_output=cfg["with_base_output"],
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
