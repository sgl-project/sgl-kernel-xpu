"""Debug-only value checks of the LoRA kernels (USE_DEBUG_CHECKS=ON builds).

Value-level validation (index ranges, monotonic seg_indptr, permutation
bijectivity, offset tables) forces device -> host syncs, so it is compiled only
in builds configured with USE_DEBUG_CHECKS=ON. In a default build these inputs
are NOT validated, and feeding them to a kernel is undefined behaviour (e.g. an
out-of-range adapter index reads out of bounds), so this file is skipped unless
the build under test was configured with the flag:

    USE_DEBUG_CHECKS=ON pip install -e . --no-build-isolation -v
    SGL_KERNEL_DEBUG_CHECKS=1 pytest tests/test_lora_debug_checks.py
"""

import sys

import pytest
import torch
from sgl_kernel import (
    chunked_sgmv_lora_expand_fwd,
    chunked_sgmv_lora_shrink_fwd,
    embedding_lora_a_fwd,
    gate_up_lora_b_fwd,
    qkv_lora_b_fwd,
    sgemm_lora_a_fwd,
    sgemm_lora_b_fwd,
)

pytestmark = pytest.mark.debug_checks  # see tests/conftest.py

DEV = "xpu"
DT = torch.bfloat16
T, L, R, K = 8, 2, 16, 64  # tokens, adapters, rank, input dim


def _i32(values):
    return torch.tensor(values, dtype=torch.int32, device=DEV)


def _segments():
    # Two segments of 4 rows each, adapters 0 and 1.
    return {"seg_indptr": _i32([0, 4, T]), "weight_indices": _i32([0, 1])}


# Each corruption overrides one input and names the check that must fire.
_BAD_SEGMENTS = [
    pytest.param(
        {"weight_indices": _i32([0, L])},
        "weight_indices values",
        id="adapter-out-of-range",
    ),
    pytest.param(
        {"weight_indices": _i32([-1, 0])},
        "weight_indices values",
        id="adapter-negative",
    ),
    pytest.param(
        {"seg_indptr": _i32([1, 4, T])}, r"seg_indptr\[0\]", id="indptr-not-zero-based"
    ),
    pytest.param(
        {"seg_indptr": _i32([0, 4, T - 1])}, r"seg_indptr\[-1\]", id="indptr-wrong-end"
    ),
    pytest.param(
        {"seg_indptr": _i32([0, T + 2, T])}, "non-decreasing", id="indptr-decreasing"
    ),
    pytest.param({"lora_ranks": _i32([R, R + 1])}, "lora_ranks", id="rank-too-large"),
]


def _ranks():
    return {"lora_ranks": _i32([R] * L)}


def _sgemm_a(**over):
    args = {
        "input_x": torch.randn(T, K, dtype=DT, device=DEV),
        "weights": torch.randn(L, R, K, dtype=DT, device=DEV),
        "stack_num": 1,
        **_segments(),
        **_ranks(),
    }
    args.update(over)
    return sgemm_lora_a_fwd(**args)


def _sgemm_b(**over):
    args = {
        "input_x": torch.randn(T, R, dtype=DT, device=DEV),
        "weights": torch.randn(L, 128, R, dtype=DT, device=DEV),
        "scalings": torch.ones(L, device=DEV),
        **_segments(),
        **_ranks(),
    }
    args.update(over)
    return sgemm_lora_b_fwd(**args)


def _qkv(**over):
    args = {
        "input_x": torch.randn(T, 3 * R, dtype=DT, device=DEV),
        "qkv_lora_b": torch.randn(L, 128 + 2 * 64, R, dtype=DT, device=DEV),
        "output_offset": _i32([0, 128, 192, 256]),
        "max_qkv_out_dim": 128,
        "scalings": torch.ones(L, device=DEV),
        **_segments(),
        **_ranks(),
    }
    args.update(over)
    return qkv_lora_b_fwd(**args)


def _gate_up(**over):
    args = {
        "input_x": torch.randn(T, 2 * R, dtype=DT, device=DEV),
        "gate_up_lora_b": torch.randn(L, 2 * 128, R, dtype=DT, device=DEV),
        "output_dim": 128,
        "scalings": torch.ones(L, device=DEV),
        **_segments(),
        **_ranks(),
    }
    args.update(over)
    return gate_up_lora_b_fwd(**args)


def _embedding(**over):
    vocab = 32
    args = {
        "input_ids": torch.randint(0, vocab, (T,), device=DEV),
        "weights": torch.randn(L, R, vocab, dtype=DT, device=DEV),
        "vocab_size": vocab,
        **_segments(),
        **_ranks(),
    }
    args.update(over)
    return embedding_lora_a_fwd(**args)


def _shrink(**over):
    args = {
        "input_x": torch.randn(T, K, dtype=DT, device=DEV),
        "weights": torch.randn(L, R, K, dtype=DT, device=DEV),
        "stack_num": 1,
        "num_segments": 2,
        "permutation": torch.randperm(T, device=DEV),
        **_segments(),
        **_ranks(),
    }
    args.update(over)
    return chunked_sgmv_lora_shrink_fwd(**args)


def _expand(**over):
    args = {
        "input_x": torch.randn(T, 2 * R, dtype=DT, device=DEV),
        "weights": torch.randn(L, 256, R, dtype=DT, device=DEV),
        "slice_offsets": _i32([0, 128, 256]),
        "max_slice_size": 128,
        "num_slices": 2,
        "num_segments": 2,
        "scalings": torch.ones(L, device=DEV),
        "permutation": torch.randperm(T, device=DEV),
        **_segments(),
        **_ranks(),
    }
    args.update(over)
    return chunked_sgmv_lora_expand_fwd(**args)


_OPS = [
    pytest.param(_sgemm_a, id="sgemm_lora_a"),
    pytest.param(_sgemm_b, id="sgemm_lora_b"),
    pytest.param(_qkv, id="qkv_lora_b"),
    pytest.param(_gate_up, id="gate_up_lora_b"),
    pytest.param(_embedding, id="embedding_lora_a"),
    pytest.param(_shrink, id="chunked_shrink"),
    pytest.param(_expand, id="chunked_expand"),
]


@pytest.mark.parametrize("op", _OPS)
def test_valid_inputs_pass(op):
    op()
    torch.xpu.synchronize()


@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("bad, match", _BAD_SEGMENTS)
def test_bad_segments_raise(op, bad, match):
    with pytest.raises(RuntimeError, match=match):
        op(**bad)


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(_shrink, id="chunked_shrink"),
        pytest.param(_expand, id="chunked_expand"),
    ],
)
@pytest.mark.parametrize(
    "perm, match",
    [
        pytest.param([0, 1, 2, 3, 4, 5, 6, T], "permutation values", id="out-of-range"),
        pytest.param([0, 0, 2, 3, 4, 5, 6, 7], "bijection", id="repeated-row"),
    ],
)
def test_bad_permutation_raises(op, perm, match):
    with pytest.raises(RuntimeError, match=match):
        op(permutation=torch.tensor(perm, dtype=torch.int64, device=DEV))


@pytest.mark.parametrize(
    "op, bad, match",
    [
        pytest.param(
            _qkv,
            {"output_offset": _i32([1, 128, 192, 256])},
            r"output_offset\[0\]",
            id="qkv-not-zero-based",
        ),
        pytest.param(
            _qkv,
            {"output_offset": _i32([0, 128, 192, 250])},
            r"output_offset\[-1\]",
            id="qkv-wrong-end",
        ),
        pytest.param(
            _qkv,
            {"output_offset": _i32([0, 200, 192, 256])},
            "non-decreasing",
            id="qkv-decreasing",
        ),
        pytest.param(
            _qkv, {"max_qkv_out_dim": 64}, "max_qkv_out_dim", id="qkv-wrong-max"
        ),
        pytest.param(
            _expand,
            {"slice_offsets": _i32([1, 128, 256])},
            r"slice_offsets\[0\]",
            id="expand-not-zero-based",
        ),
        pytest.param(
            _expand,
            {"slice_offsets": _i32([0, 128, 250])},
            r"slice_offsets\[-1\]",
            id="expand-wrong-end",
        ),
        pytest.param(
            _expand,
            {"slice_offsets": _i32([0, 300, 256])},
            "non-decreasing",
            id="expand-decreasing",
        ),
        pytest.param(
            _expand, {"max_slice_size": 100}, "max_slice_size", id="expand-wrong-max"
        ),
    ],
)
def test_bad_offsets_raise(op, bad, match):
    with pytest.raises(RuntimeError, match=match):
        op(**bad)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
