import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PKG = _REPO_ROOT / "python" / "sgl_kernel"
_LOCAL_EXT = _REPO_ROOT / "build" / "src" / "inkling_mel_embedding_ops.abi3.so"

if _LOCAL_PKG.is_dir() and _LOCAL_EXT.is_file() and "sgl_kernel" not in sys.modules:
    pkg = types.ModuleType("sgl_kernel")
    pkg.__path__ = [str(_LOCAL_PKG), str(_LOCAL_EXT.parent)]
    sys.modules["sgl_kernel"] = pkg
    torch.ops.load_library(str(_LOCAL_EXT))
else:
    import sgl_kernel  # noqa: F401
    try:
        importlib.import_module("sgl_kernel.inkling_mel_embedding_ops")
    except ImportError:
        pass

pytestmark = pytest.mark.skipif(
    not (
        hasattr(torch, "xpu")
        and torch.xpu.is_available()
        and hasattr(torch.ops.sgl_kernel, "inkling_mel_embedding_sum")
    ),
    reason="Inkling mel_embedding_sum op is XPU-only",
)


def _make_features(tokens: int, n_mel_bins: int, mel_vocab_size: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(17)
    features = torch.randint(
        0,
        mel_vocab_size,
        (tokens, n_mel_bins),
        dtype=torch.int32,
        generator=gen,
    )
    if tokens > 0:
        features[0] = torch.arange(n_mel_bins, dtype=torch.int32) % mel_vocab_size
        features[-1] = mel_vocab_size - 1 - (
            torch.arange(n_mel_bins, dtype=torch.int32) % mel_vocab_size
        )
    return features.to("xpu")


def _make_weight(n_mel_bins: int, mel_vocab_size: int, hidden: int, dtype) -> torch.Tensor:
    rows = n_mel_bins * mel_vocab_size
    gen = torch.Generator(device="cpu").manual_seed(29)
    weight = torch.empty((rows, hidden), dtype=torch.float32)
    weight.uniform_(-0.125, 0.125, generator=gen)
    row = torch.arange(rows, dtype=torch.int64)[:, None]
    channel = torch.arange(hidden, dtype=torch.int64)[None, :]
    pattern = (((row * 131 + channel * 17) % 29).to(torch.float32) - 14) * 0.001
    return (weight + pattern).to(dtype=dtype, device="xpu")


def _reference(features: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    features_cpu = features.cpu().to(torch.long)
    weight_cpu = weight.cpu()
    tokens, n_mel_bins = features_cpu.shape
    mel_vocab_size = weight_cpu.shape[0] // n_mel_bins
    hidden = weight_cpu.shape[1]
    table = weight_cpu.float().view(n_mel_bins, mel_vocab_size, hidden)
    out = torch.zeros((tokens, hidden), dtype=torch.float32)
    for mel in range(n_mel_bins):
        out += table[mel, features_cpu[:, mel], :]
    return out.to(weight_cpu.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens,n_mel_bins,mel_vocab_size,hidden,chunk_size,channels_per_item",
    [
        (4, 3, 5, 6, 16, 1),
        (17, 80, 16, 259, 8, 0),
        (65, 80, 16, 384, 32, 2),
        (513, 80, 16, 384, 512, 8),
    ],
)
def test_mel_embedding_sum_matches_reference(
    dtype,
    tokens,
    n_mel_bins,
    mel_vocab_size,
    hidden,
    chunk_size,
    channels_per_item,
):
    features = _make_features(tokens, n_mel_bins, mel_vocab_size)
    weight = _make_weight(n_mel_bins, mel_vocab_size, hidden, dtype)
    expected = _reference(features, weight)

    actual = torch.ops.sgl_kernel.inkling_mel_embedding_sum(
        features,
        weight,
        chunk_size,
        channels_per_item,
    )
    torch.xpu.synchronize()

    assert actual.shape == (tokens, hidden)
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        atol=0 if dtype is not torch.float32 else 1.0e-5,
        rtol=0 if dtype is not torch.float32 else 1.0e-5,
        check_dtype=True,
    )


def test_mel_embedding_sum_rejects_bad_feature_dtype():
    features = torch.zeros((1, 3), dtype=torch.int64, device="xpu")
    weight = torch.zeros((15, 8), dtype=torch.bfloat16, device="xpu")
    with pytest.raises(RuntimeError, match="features must be int32"):
        torch.ops.sgl_kernel.inkling_mel_embedding_sum(features, weight)
