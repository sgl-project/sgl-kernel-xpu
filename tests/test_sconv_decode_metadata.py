"""fused_decode_sconv_metadata must be bit-identical to the unfused prep.

The unfused reference is the exact op sequence `_prepare_decode_sconv_metadata`
used to launch: two arange calls + ones + precompute_helion_decode_metadata
(!= PAD, &, clamp, long, arange x2).
"""

import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PKG = _REPO_ROOT / "python" / "sgl_kernel"
_LOCAL_EXT = _REPO_ROOT / "build" / "src" / "common_ops.abi3.so"

if _LOCAL_PKG.is_dir() and _LOCAL_EXT.is_file() and "sgl_kernel" not in sys.modules:
    pkg = types.ModuleType("sgl_kernel")
    pkg.__path__ = [str(_LOCAL_PKG)]
    sys.modules["sgl_kernel"] = pkg
    torch.ops.load_library(str(_LOCAL_EXT))

from sgl_kernel.inkling_sconv import (
    PAD_SLOT_ID,
    fused_decode_sconv_metadata,
    precompute_helion_decode_metadata,
)

requires_cuda = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="XPU only"
)

# cross the BLOCK=1024 grid boundary and hit odd sizes
BATCH_SIZES = [1, 2, 3, 17, 64, 160, 257, 1023, 1024, 1025]


def _reference(B: int, cache_indices: torch.Tensor):
    device = cache_indices.device
    query_start_loc = torch.arange(B + 1, dtype=torch.int32, device=device)
    has_initial_state = torch.ones(B, dtype=torch.bool, device=device)
    precomputed = precompute_helion_decode_metadata(
        B=B, W=4, cache_indices=cache_indices, has_initial_state=has_initial_state
    )
    return query_start_loc, has_initial_state, precomputed


def _metadata_out(B: int, T: int):
    return {
        "query_start_loc": torch.empty(B + 1, dtype=torch.int32, device="xpu"),
        "has_initial_state": torch.empty(B, dtype=torch.bool, device="xpu"),
        "cache_mask": torch.empty((B, 1, 1), dtype=torch.bool, device="xpu"),
        "safe_idx": torch.empty(B, dtype=torch.int64, device="xpu"),
        "cu": torch.empty(B + 1, dtype=torch.int64, device="xpu"),
        "si": torch.empty(T, dtype=torch.int32, device="xpu"),
    }


def _assert_uses_out(got, out):
    for got_tensor, out_tensor in (
        (got[0], out["query_start_loc"]),
        (got[1], out["has_initial_state"]),
        (got[2]["cache_mask"], out["cache_mask"]),
        (got[2]["safe_idx"], out["safe_idx"]),
        (got[2]["cu"], out["cu"]),
        (got[2]["si"], out["si"]),
    ):
        assert got_tensor.data_ptr() == out_tensor.data_ptr()


@requires_cuda
@pytest.mark.parametrize("b", BATCH_SIZES)
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_matches_unfused(b: int, idx_dtype: torch.dtype):
    torch.manual_seed(b)
    cache_indices = torch.randint(0, 4096, (b,), dtype=idx_dtype, device="xpu")
    # sprinkle PAD slots (cudagraph padding lanes)
    pad = torch.rand(b, device="xpu") < 0.25
    cache_indices[pad] = PAD_SLOT_ID

    ref_qsl, ref_his, ref_meta = _reference(b, cache_indices)
    qsl, his, meta = fused_decode_sconv_metadata(B=b, cache_indices=cache_indices)

    for tag, got, ref in (
        ("query_start_loc", qsl, ref_qsl),
        ("has_initial_state", his, ref_his),
        ("cache_mask", meta["cache_mask"], ref_meta["cache_mask"]),
        ("safe_idx", meta["safe_idx"], ref_meta["safe_idx"]),
        ("cu", meta["cu"], ref_meta["cu"]),
        ("si", meta["si"], ref_meta["si"]),
    ):
        assert got.dtype == ref.dtype, (tag, got.dtype, ref.dtype)
        assert got.shape == ref.shape, (tag, got.shape, ref.shape)
        assert torch.equal(got, ref), tag


@requires_cuda
def test_writes_graph_static_outputs():
    b = 17
    cache_indices = torch.tensor(
        [0, 1, PAD_SLOT_ID, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        dtype=torch.int32,
        device="xpu",
    )
    out = _metadata_out(b, b)
    got = fused_decode_sconv_metadata(B=b, cache_indices=cache_indices, out=out)

    _assert_uses_out(got, out)
    ref = _reference(b, cache_indices)
    for got_tensor, ref_tensor in (
        (got[0], ref[0]),
        (got[1], ref[1]),
        (got[2]["cache_mask"], ref[2]["cache_mask"]),
        (got[2]["safe_idx"], ref[2]["safe_idx"]),
        (got[2]["cu"], ref[2]["cu"]),
        (got[2]["si"], ref[2]["si"]),
    ):
        assert torch.equal(got_tensor, ref_tensor)


@requires_cuda
def test_all_pad():
    cache_indices = torch.full((8,), PAD_SLOT_ID, dtype=torch.int32, device="xpu")
    _, _, meta = fused_decode_sconv_metadata(B=8, cache_indices=cache_indices)
    assert not meta["cache_mask"].any()
    assert (meta["safe_idx"] == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
