import sys
import types
from pathlib import Path

import pytest
import torch
from sconv_reference import (
    ar_fused_decode_ref,
    ar_scattered_sconv_ref,
    assert_close,
    comm_all_reduce_ref,
    rand,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PKG = _REPO_ROOT / "python" / "sgl_kernel"
_LOCAL_EXT = _REPO_ROOT / "build" / "src" / "inkling_sconv_ops.abi3.so"

if _LOCAL_PKG.is_dir() and _LOCAL_EXT.is_file() and "sgl_kernel" not in sys.modules:
    pkg = types.ModuleType("sgl_kernel")
    pkg.__path__ = [str(_LOCAL_PKG)]
    sys.modules["sgl_kernel"] = pkg
    sys.modules["sgl_kernel.common_ops"] = types.ModuleType("sgl_kernel.common_ops")
    torch.ops.load_library(str(_LOCAL_EXT))

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Inkling comm/AR sconv ops are XPU-only",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "variant", ["direct", "two_shot", "full_oneshot", "push_oneshot"]
)
@pytest.mark.parametrize("use_shared", [False, True])
def test_comm_all_reduce_variants_match_reference(dtype, variant, use_shared):
    from sgl_kernel.inkling_comm_ar_sconv import comm_all_reduce

    torch.manual_seed(11)
    partials = rand((4, 5, 8), dtype, scale=0.3).contiguous()
    shared = rand((4, 5, 8), dtype, scale=0.1).contiguous() if use_shared else None

    actual = comm_all_reduce(partials, shared=shared, variant=variant)
    expected = comm_all_reduce_ref(partials, shared=shared)
    assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("use_residual", [False, True])
@pytest.mark.parametrize("use_shared", [False, True])
def test_ar_fused_decode_matches_reference(
    dtype, activation, use_residual, use_shared
):
    from sgl_kernel.inkling_comm_ar_sconv import ar_fused_decode

    torch.manual_seed(12)
    world, T, D, W = 4, 5, 8, 4
    partials = rand((world, T, D), dtype, scale=0.25).contiguous()
    shared = rand((world, T, D), dtype, scale=0.1).contiguous() if use_shared else None
    residual = rand((T, D), dtype, scale=0.2).contiguous()
    cache = rand((9, W - 1, D), dtype, scale=0.1)
    weight = rand((D, W), dtype, scale=0.2)
    norm_weight = rand((D,), dtype, scale=0.15) + 1
    cache_indices = torch.tensor([0, 1, -1, 3, 4], dtype=torch.int32, device="xpu")
    cache_mask = torch.tensor(
        [True, False, False, True, True], dtype=torch.bool, device="xpu"
    )
    expected_hs, expected_residual, expected_cache, _ = ar_fused_decode_ref(
        partials,
        residual,
        cache,
        cache_indices,
        cache_mask,
        weight,
        norm_weight,
        activation=activation,
        use_residual=use_residual,
        shared=shared,
    )

    actual_hs, actual_residual = ar_fused_decode(
        partials,
        residual,
        cache,
        cache_indices,
        cache_mask,
        weight,
        norm_weight,
        activation=activation,
        use_residual=use_residual,
        shared=shared,
    )
    assert_close(actual_hs, expected_hs, dtype)
    assert_close(actual_residual, expected_residual, dtype)
    torch.testing.assert_close(
        cache.detach().cpu(), expected_cache, atol=0, rtol=0, check_dtype=False
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("use_residual", [False, True])
@pytest.mark.parametrize("use_shared", [False, True])
def test_ar_scattered_sconv_matches_reference(
    dtype, activation, use_residual, use_shared
):
    from sgl_kernel.inkling_comm_ar_sconv import ar_scattered_sconv

    torch.manual_seed(13)
    world, B, T, D, W = 4, 3, 5, 8, 4
    partials = rand((world, T, D), dtype, scale=0.25).contiguous()
    shared = rand((world, T, D), dtype, scale=0.1).contiguous() if use_shared else None
    cache = rand((7, W - 1, D), dtype, scale=0.1)
    weight = rand((D, W), dtype, scale=0.2)
    cache_indices = torch.tensor([0, -1, 2], dtype=torch.int32, device="xpu")
    cache_mask = torch.tensor([True, False, True], dtype=torch.bool, device="xpu")
    cu = torch.tensor([0, 2, 2, 5], dtype=torch.int64, device="xpu")
    si = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int32, device="xpu")
    has_initial_state = torch.tensor(
        [True, False, False], dtype=torch.bool, device="xpu"
    )
    expected_out, expected_scratch, expected_cache = ar_scattered_sconv_ref(
        partials,
        cache,
        cache_indices,
        cache_mask,
        cu,
        si,
        weight,
        has_initial_state,
        activation=activation,
        use_residual=use_residual,
        shared=shared,
    )

    actual_out, actual_scratch = ar_scattered_sconv(
        partials,
        cache,
        cache_indices,
        cache_mask,
        cu,
        si,
        weight,
        has_initial_state,
        activation=activation,
        use_residual=use_residual,
        shared=shared,
    )
    assert B == cache_indices.numel()
    assert_close(actual_out, expected_out, dtype)
    assert_close(actual_scratch, expected_scratch, dtype)
    torch.testing.assert_close(
        cache.detach().cpu(), expected_cache, atol=0, rtol=0, check_dtype=False
    )
