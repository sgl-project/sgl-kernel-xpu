import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PKG = _REPO_ROOT / "python" / "sgl_kernel"
_LOCAL_EXT = _REPO_ROOT / "build" / "src" / "inkling_hmlp_fold_ops.abi3.so"

if _LOCAL_PKG.is_dir() and _LOCAL_EXT.is_file() and "sgl_kernel" not in sys.modules:
    pkg = types.ModuleType("sgl_kernel")
    pkg.__path__ = [str(_LOCAL_PKG), str(_LOCAL_EXT.parent)]
    sys.modules["sgl_kernel"] = pkg
    torch.ops.load_library(str(_LOCAL_EXT))
else:
    import sgl_kernel  # noqa: F401
    try:
        importlib.import_module("sgl_kernel.inkling_hmlp_fold_ops")
    except ImportError:
        pass

pytestmark = pytest.mark.skipif(
    not (
        hasattr(torch, "xpu")
        and torch.xpu.is_available()
        and hasattr(torch.ops.sgl_kernel, "inkling_hmlp_fold_timespace_to_depth")
    ),
    reason="Inkling hMLP fold op is XPU-only",
)


def _reference(x: torch.Tensor, t_fold: int, hw_fold: int) -> torch.Tensor:
    B, T, H, W, C = x.shape
    t_new = T // t_fold
    h_new = H // hw_fold
    w_new = W // hw_fold
    return (
        x.reshape(B, t_new, t_fold, h_new, hw_fold, w_new, hw_fold, C)
        .permute(0, 1, 3, 5, 2, 4, 6, 7)
        .reshape(B, t_new, h_new, w_new, t_fold * hw_fold * hw_fold * C)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape,t_fold,hw_fold",
    [
        ((2, 4, 6, 6, 3), 2, 3),
        ((1, 6, 4, 8, 2), 3, 2),
        ((2, 2, 2, 2, 5), 1, 1),
        ((3, 3, 9, 6, 17), 3, 3),
        ((128, 1, 8, 8, 64), 1, 2),
        ((32, 2, 16, 16, 64), 2, 1),
        ((16, 1, 14, 14, 3), 1, 7),
    ],
)
def test_hmlp_fold_timespace_to_depth_matches_reference(dtype, shape, t_fold, hw_fold):
    numel = 1
    for dim in shape:
        numel *= dim
    values = torch.arange(numel, dtype=torch.float32, device="xpu").reshape(shape)
    x = ((values % 4096) / 257.0 - 8.0).to(dtype)

    actual = torch.ops.sgl_kernel.inkling_hmlp_fold_timespace_to_depth(
        x,
        t_fold,
        hw_fold,
    )
    torch.xpu.synchronize()
    expected = _reference(x, t_fold, hw_fold)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0, check_dtype=True)


def test_hmlp_fold_timespace_to_depth_rejects_bad_fold():
    x = torch.zeros((1, 3, 4, 4, 8), dtype=torch.bfloat16, device="xpu")
    with pytest.raises(RuntimeError, match="T must be divisible"):
        torch.ops.sgl_kernel.inkling_hmlp_fold_timespace_to_depth(x, 2, 2)
