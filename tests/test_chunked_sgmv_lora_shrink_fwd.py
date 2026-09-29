import sys
from typing import List

import pytest
import torch
from sgl_kernel import chunked_sgmv_lora_shrink_fwd

if not torch.xpu.is_available():
    pytest.skip(
        reason="chunked_sgmv_lora_shrink_fwd requires XPU device.",
        allow_module_level=True,
    )


def _tolerances(dtype: torch.dtype):
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-5, 1e-5


# ----------------------------------------------------------------------------
# Helpers (mirror the three-kernel decomposition on CPU in fp32)
# ----------------------------------------------------------------------------


def _zero_weight_rank_tail(
    weights: torch.Tensor, lora_ranks: torch.Tensor, stack_num: int
) -> torch.Tensor:
    """Zero-pad weight rows beyond ``lora_ranks[lora]`` per stack (kernel contract)."""
    num_loras, total_n, _ = weights.shape
    max_rank = total_n // stack_num
    out = weights.clone()
    ranks_cpu = lora_ranks.cpu()
    for l in range(num_loras):
        r = int(ranks_cpu[l].item())
        for k in range(stack_num):
            base = k * max_rank
            out[l, base + r : base + max_rank, :] = 0
    return out


def _reference_sgemm(
    input_x: torch.Tensor,
    weights: torch.Tensor,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
) -> torch.Tensor:
    """Segmented grouped GEMM computed on-device (XPU) in fp32"""
    device = input_x.device
    num_tokens = input_x.size(0)
    total_n = weights.size(1)
    out = torch.zeros((num_tokens, total_n), dtype=torch.float32, device=device)

    # seg_indptr / weight_indices are tiny; read them on the host to bound the
    # loop (the heavy matmuls stay on device).
    seg_cpu = seg_indptr.cpu()
    wi_cpu = weight_indices.cpu()

    for s in range(seg_cpu.numel() - 1):
        start = int(seg_cpu[s].item())
        end = int(seg_cpu[s + 1].item())
        if end == start:
            continue
        lora = int(wi_cpu[s].item())
        x = input_x[start:end].float()
        w = weights[lora].float()
        out[start:end] = x @ w.T
    return out.to(weights.dtype)


def _reference_chunked_shrink(
    input_x: torch.Tensor,
    weights: torch.Tensor,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """gather (physical->logical) -> sgemm -> scatter (logical->physical), on-device (XPU)."""
    perm = permutation.to(torch.int64)
    x_sorted = input_x[perm]  # x_sorted[i] = x[perm[i]]
    out_sorted = _reference_sgemm(x_sorted, weights, seg_indptr, weight_indices)
    out = torch.empty_like(out_sorted)
    out[perm] = out_sorted  # out[perm[i]] = out_sorted[i]
    return out


def _build_logical_segments(row_adapters: List[int], permutation_mode: str = "sorted"):
    """Build (permutation, seg_indptr, weight_indices) for a chunked-shrink call.

    ``permutation_mode`` selects one of the op's three permutation regimes; the
    seg_indptr / weight_indices always run-length encode the adapter of each
    *logical* row.

    - ``"sorted"`` (decode): physical rows interleave adapters ("zigzag"). The
      permutation is the stable argsort (logical -> physical) that groups rows
      by adapter -- mirrors ChunkedLoraBackend._get_permutation /
      _get_segments_info. Segments describe the sorted (logical) layout.
    - ``"identity"``: rows are already grouped by adapter (physical == logical),
      but the op is still driven through gather/scatter with an identity
      permutation (``arange``) -- a no-op remap.
    - ``"none"`` (prefill fast path): rows already grouped; ``permutation`` is
      ``None`` so the op skips gather/scatter and runs the GEMM in place.

    For ``"identity"`` / ``"none"`` the run-length encoding just follows the
    given order, so any ``row_adapters`` is valid
    """
    ra = torch.tensor(row_adapters, dtype=torch.int32)
    if permutation_mode == "sorted":
        permutation = torch.argsort(ra, stable=True).to(
            torch.int32
        )  # logical -> physical
        logical = ra[permutation.to(torch.int64)]
    elif permutation_mode == "identity":
        permutation = torch.arange(ra.numel(), dtype=torch.int32)  # no-op remap
        logical = ra
    elif permutation_mode == "none":
        permutation = None  # prefill fast path: GEMM in place, no gather/scatter
        logical = ra
    else:
        raise ValueError(f"unknown permutation_mode: {permutation_mode!r}")

    uniq, counts = torch.unique_consecutive(logical, return_counts=True)
    seg_indptr = torch.zeros(uniq.numel() + 1, dtype=torch.int32)
    seg_indptr[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    weight_indices = uniq.to(torch.int32)
    return permutation, seg_indptr, weight_indices


# ----------------------------------------------------------------------------
# chunked_sgmv_lora_shrink_fwd (three-kernel orchestration)
# ----------------------------------------------------------------------------


def _run_and_compare_chunked(
    *,
    dtype: torch.dtype,
    row_adapters: List[int],
    input_dim: int,
    max_rank: int,
    stack_num: int,
    num_loras: int,
    lora_ranks: torch.Tensor,
    permutation_mode: str = "sorted",
) -> None:
    torch.manual_seed(0)
    num_tokens = len(row_adapters)
    total_n = stack_num * max_rank

    permutation, seg_indptr, weight_indices = _build_logical_segments(
        row_adapters, permutation_mode
    )
    if permutation is not None:
        permutation = permutation.to("xpu")
    seg_indptr = seg_indptr.to("xpu")
    weight_indices = weight_indices.to("xpu")

    input_x = torch.randn(num_tokens, input_dim, dtype=dtype, device="xpu")
    weights = torch.randn(num_loras, total_n, input_dim, dtype=dtype, device="xpu")
    weights = _zero_weight_rank_tail(weights, lora_ranks, stack_num)

    out = chunked_sgmv_lora_shrink_fwd(
        input_x=input_x,
        weights=weights,
        stack_num=stack_num,
        num_segments=weight_indices.numel(),
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        permutation=permutation,
    )

    if permutation is None:
        # Prefill fast path: rows are already logical, so the reference is a
        # plain segmented grouped GEMM (no gather/scatter).
        ref = _reference_sgemm(input_x, weights, seg_indptr, weight_indices)
    else:
        ref = _reference_chunked_shrink(
            input_x, weights, seg_indptr, weight_indices, permutation
        )

    assert out.shape == (num_tokens, total_n)
    assert out.dtype == dtype
    # Compare on device: both `out` and `ref` live on XPU, so torch.testing
    # validates them without a host round-trip.
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


# One data-driven test over named scenarios. Each scenario overrides only the
# params it exercises; everything else falls back to _DEFAULT_SCENARIO.
# ``lora_ranks=None`` means "full rank for every adapter" ([max_rank]*num_loras).
_DEFAULT_SCENARIO = {
    "input_dim": 256,
    "max_rank": 8,
    "stack_num": 1,
    "num_loras": 2,
    "row_adapters": [0, 1, 0, 1, 0, 1, 0, 1],
    "lora_ranks": None,
    "permutation_mode": "sorted",
}

# Deterministic 512-token random adapter assignment (isolated from global RNG).
_MANY_ADAPTERS = torch.randint(
    0, 4, (512,), generator=torch.Generator().manual_seed(7)
).tolist()

_ZIGZAG = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]

_SCENARIOS = [
    # Interleaved (zigzag) decode adapters, swept over input_dim x max_rank.
    pytest.param(
        {
            "num_loras": 3,
            "row_adapters": _ZIGZAG,
            "input_dim": 64,
            "max_rank": 8,
            "lora_ranks": [8, 4, 2],
        },
        id="zigzag-K64-r8",
    ),
    pytest.param(
        {
            "num_loras": 3,
            "row_adapters": _ZIGZAG,
            "input_dim": 64,
            "max_rank": 64,
            "lora_ranks": [64, 32, 16],
        },
        id="zigzag-K64-r64",
    ),
    pytest.param(
        {
            "num_loras": 3,
            "row_adapters": _ZIGZAG,
            "input_dim": 4096,
            "max_rank": 8,
            "lora_ranks": [8, 4, 2],
        },
        id="zigzag-K4096-r8",
    ),
    pytest.param(
        {
            "num_loras": 3,
            "row_adapters": _ZIGZAG,
            "input_dim": 4096,
            "max_rank": 64,
            "lora_ranks": [64, 32, 16],
        },
        id="zigzag-K4096-r64",
    ),
    # Stacked projections: o_proj=1, gate_up=2, qkv=3.
    pytest.param({"stack_num": 1, "lora_ranks": [8, 4]}, id="stack1"),
    pytest.param({"stack_num": 2, "lora_ranks": [8, 4]}, id="stack2"),
    pytest.param({"stack_num": 3, "lora_ranks": [8, 4]}, id="stack3"),
    # A rank-0 adapter must yield an all-zero output for its rows.
    pytest.param(
        {"row_adapters": [0, 1, 0, 1, 0, 1], "input_dim": 128, "lora_ranks": [0, 8]},
        id="zero-rank-adapter",
    ),
    # Many tokens across many adapters.
    pytest.param(
        {
            "num_loras": 4,
            "max_rank": 16,
            "input_dim": 512,
            "row_adapters": _MANY_ADAPTERS,
            "lora_ranks": [1, 4, 8, 16],
        },
        id="many-adapters",
    ),
    # permutation=None prefill fast path (rows pre-grouped, GEMM in place).
    pytest.param(
        {
            "num_loras": 3,
            "row_adapters": [0] * 16 + [2] * 32 + [1] * 16,
            "lora_ranks": [8, 4, 2],
            "permutation_mode": "none",
        },
        id="permutation-none",
    ),
    # Identity permutation drives the gather/scatter path as a no-op remap.
    pytest.param(
        {
            "row_adapters": [0] * 16 + [1] * 16,
            "input_dim": 128,
            "lora_ranks": [8, 4],
            "permutation_mode": "identity",
        },
        id="identity-permutation",
    ),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_chunked_shrink(dtype, scenario):
    cfg = {**_DEFAULT_SCENARIO, **scenario}
    num_loras = cfg["num_loras"]
    max_rank = cfg["max_rank"]
    ranks = (
        cfg["lora_ranks"] if cfg["lora_ranks"] is not None else [max_rank] * num_loras
    )
    lora_ranks = torch.tensor(ranks, dtype=torch.int32, device="xpu")
    _run_and_compare_chunked(
        dtype=dtype,
        row_adapters=cfg["row_adapters"],
        input_dim=cfg["input_dim"],
        max_rank=max_rank,
        stack_num=cfg["stack_num"],
        num_loras=num_loras,
        lora_ranks=lora_ranks,
        permutation_mode=cfg["permutation_mode"],
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
