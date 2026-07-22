import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Inkling sconv ops are XPU-only",
)


def _reference_root() -> Path:
    candidates = []
    if "INKLING_SCONV_REFERENCE_DIR" in os.environ:
        candidates.append(Path(os.environ["INKLING_SCONV_REFERENCE_DIR"]))
    candidates.extend(
        [
            Path("/workspace/modeltune/inkling/01_sconv"),
            Path("/data2/syk/modeltune/inkling/01_sconv"),
        ]
    )
    for candidate in candidates:
        if (candidate / "01_01_causal_conv1d/reference_case.py").is_file():
            modeltune_root = candidate.parents[1]
            if str(modeltune_root) not in sys.path:
                sys.path.insert(0, str(modeltune_root))
            return candidate
    pytest.skip("modeltune Inkling sconv reference cases are unavailable")


def _load_reference(root: Path, rel: str):
    path = root / rel / "reference_case.py"
    spec = importlib.util.spec_from_file_location(rel, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _xpu(data, dtype=torch.float32):
    return torch.as_tensor(data, dtype=dtype, device="xpu")


def _assert_close(actual, expected, *, atol=1.0e-5, rtol=1.0e-5):
    torch.testing.assert_close(
        actual.detach().cpu(), torch.as_tensor(expected), atol=atol, rtol=rtol, check_dtype=False
    )


def _tol(dtype: torch.dtype):
    if dtype is torch.bfloat16:
        return 2.0e-2, 2.0e-2
    if dtype is torch.float16:
        return 2.0e-3, 2.0e-3
    return 1.0e-4, 1.0e-4


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_inkling_causal_conv1d_against_modeltune_reference(dtype):
    root = _reference_root()
    ref = _load_reference(root, "01_01_causal_conv1d")
    from inkling.reference_common import make_rng

    data = ref.inputs(make_rng())
    atol, rtol = _tol(dtype)

    from sgl_kernel.inkling_sconv import causal_conv1d

    starts = data["starts"]
    si = torch.empty(data["x"].shape[0], dtype=torch.int32, device="xpu")
    for b, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        si[int(start) : int(end)] = b

    x = _xpu(data["x"], dtype=dtype)
    weight_wd = _xpu(data["weight"], dtype=dtype)
    weight_dw = weight_wd.t().contiguous()
    cache = _xpu(data["cache"], dtype=dtype)
    cache_mask = torch.ones((len(starts) - 1, 1, 1), dtype=torch.bool, device="xpu")
    safe_idx = torch.arange(len(starts) - 1, dtype=torch.int64, device="xpu")
    cu = torch.as_tensor(starts, dtype=torch.int64, device="xpu")
    expected = ref.numpy_causal_conv1d(
        data["x"], data["weight"], data["cache"], starts, use_silu=True, residual=None
    )
    expected_residual = ref.numpy_causal_conv1d(
        data["x"], data["weight"], data["cache"], starts, use_silu=True, residual=data["x"]
    )

    out_wd = causal_conv1d(
        x, weight_wd, cache, cache_mask, safe_idx, cu, si, activation="silu", use_residual=False
    )
    out_dw = causal_conv1d(
        x, weight_dw, cache, cache_mask, safe_idx, cu, si, activation="silu", use_residual=False
    )
    out_residual_wd = causal_conv1d(
        x, weight_wd, cache, cache_mask, safe_idx, cu, si, activation="silu", use_residual=True
    )
    out_residual_dw = causal_conv1d(
        x, weight_dw, cache, cache_mask, safe_idx, cu, si, activation="silu", use_residual=True
    )
    _assert_close(out_wd, expected, atol=atol, rtol=rtol)
    _assert_close(out_dw, expected, atol=atol, rtol=rtol)
    _assert_close(out_residual_wd, expected_residual, atol=atol, rtol=rtol)
    _assert_close(out_residual_dw, expected_residual, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_silu", [False, True])
@pytest.mark.parametrize("use_residual", [False, True])
def test_inkling_causal_conv1d_w4_block_path_against_modeltune_reference(
    dtype, use_silu, use_residual
):
    root = _reference_root()
    ref = _load_reference(root, "01_01_causal_conv1d")
    from inkling.reference_common import make_rng, rand

    gen = make_rng()
    x_np = rand(gen, (9, 7))
    weight_np = rand(gen, (4, 7), 0.2)
    cache_np = rand(gen, (3, 3, 7), 0.1)
    starts = torch.tensor([0, 1, 5, 9], dtype=torch.int64)
    residual_np = x_np if use_residual else None
    atol, rtol = _tol(dtype)

    from sgl_kernel.inkling_sconv import causal_conv1d

    si = torch.empty(x_np.shape[0], dtype=torch.int32, device="xpu")
    for b, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        si[int(start) : int(end)] = b

    x = _xpu(x_np, dtype=dtype)
    weight_wd = _xpu(weight_np, dtype=dtype)
    weight_dw = weight_wd.t().contiguous()
    cache = _xpu(cache_np, dtype=dtype)
    cache_mask = torch.ones((len(starts) - 1, 1, 1), dtype=torch.bool, device="xpu")
    safe_idx = torch.arange(len(starts) - 1, dtype=torch.int64, device="xpu")
    cu = starts.to(device="xpu")
    expected = ref.numpy_causal_conv1d(
        x_np,
        weight_np,
        cache_np,
        starts.numpy(),
        use_silu=use_silu,
        residual=residual_np,
    )

    out_wd = causal_conv1d(
        x,
        weight_wd,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation="silu" if use_silu else None,
        use_residual=use_residual,
    )
    out_dw = causal_conv1d(
        x,
        weight_dw,
        cache,
        cache_mask,
        safe_idx,
        cu,
        si,
        activation="silu" if use_silu else None,
        use_residual=use_residual,
    )
    _assert_close(out_wd, expected, atol=atol, rtol=rtol)
    _assert_close(out_dw, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_inkling_update_sconv_cache_against_modeltune_reference(dtype):
    root = _reference_root()
    ref = _load_reference(root, "01_02_update_sconv_cache")
    from inkling.reference_common import make_rng

    data = ref.inputs(make_rng())
    atol, rtol = _tol(dtype)

    from sgl_kernel.inkling_sconv import update_sconv_cache

    cache = _xpu(data["cache"], dtype=dtype)
    update_sconv_cache(
        _xpu(data["x"], dtype=dtype),
        cache,
        torch.arange(cache.shape[0], dtype=torch.int32, device="xpu"),
        torch.ones(cache.shape[0], dtype=torch.bool, device="xpu"),
        torch.as_tensor(data["starts"], dtype=torch.int32, device="xpu"),
    )
    expected = ref.numpy_update_sconv_cache(data["x"], data["cache"], data["starts"])
    _assert_close(cache, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_inkling_fused_decode_update_against_modeltune_reference(dtype):
    _reference_root()

    from inkling.reference_common import make_rng, rand, np_silu
    from sgl_kernel.inkling_sconv import fused_causal_conv1d_update_decode

    gen = make_rng()
    x_np = rand(gen, (3, 4))
    cache_np = rand(gen, (3, 2, 4), 0.1)
    weight_np = rand(gen, (3, 4), 0.2)
    atol, rtol = _tol(dtype)

    def decode_expected(cache_input, *, use_residual=False):
        y_expected = torch.zeros((x_np.shape[0], x_np.shape[1]), dtype=torch.float32)
        for b in range(x_np.shape[0]):
            history = torch.cat([torch.as_tensor(cache_input[b]), torch.as_tensor(x_np[b : b + 1])], dim=0)
            acc = torch.zeros((x_np.shape[1],), dtype=torch.float32)
            for tap in range(weight_np.shape[0]):
                acc = acc + history[cache_input.shape[1] - tap] * torch.as_tensor(weight_np[tap])
            y_expected[b] = torch.as_tensor(np_silu(acc.numpy()))
            if use_residual:
                y_expected[b] = y_expected[b] + torch.as_tensor(x_np[b])
        cache_expected = torch.as_tensor(cache_input.copy())
        cache_expected[: x_np.shape[0]] = torch.cat(
            [torch.as_tensor(cache_input[: x_np.shape[0], 1:, :]), torch.as_tensor(x_np[:, None, :])], dim=1
        )
        return y_expected, cache_expected

    y_expected, cache_expected = decode_expected(cache_np)

    x = _xpu(x_np, dtype=dtype)
    weight_wd = _xpu(weight_np, dtype=dtype)
    weight_dw = weight_wd.flip(0).t().contiguous()

    cache = _xpu(cache_np, dtype=dtype)
    y = fused_causal_conv1d_update_decode(
        x,
        weight_wd,
        cache,
        torch.arange(x.shape[0], dtype=torch.int32, device="xpu"),
        torch.ones(x.shape[0], dtype=torch.bool, device="xpu"),
        activation="silu",
        use_residual=False,
    )
    _assert_close(y, y_expected, atol=atol, rtol=rtol)
    _assert_close(cache, cache_expected, atol=atol, rtol=rtol)

    cache = _xpu(cache_np, dtype=dtype)
    y = fused_causal_conv1d_update_decode(
        x,
        weight_dw,
        cache,
        torch.arange(x.shape[0], dtype=torch.int32, device="xpu"),
        torch.ones(x.shape[0], dtype=torch.bool, device="xpu"),
        activation="silu",
        use_residual=False,
    )
    _assert_close(y, y_expected, atol=atol, rtol=rtol)
    _assert_close(cache, cache_expected, atol=atol, rtol=rtol)

    y_expected_residual, cache_expected_residual = decode_expected(cache_np, use_residual=True)
    cache = _xpu(cache_np, dtype=dtype)
    y = fused_causal_conv1d_update_decode(
        x,
        weight_wd,
        cache,
        torch.arange(x.shape[0], dtype=torch.int32, device="xpu"),
        torch.ones(x.shape[0], dtype=torch.bool, device="xpu"),
        activation="silu",
        use_residual=True,
    )
    _assert_close(y, y_expected_residual, atol=atol, rtol=rtol)
    _assert_close(cache, cache_expected_residual, atol=atol, rtol=rtol)

    cache_track_np = rand(gen, (6, 2, 4), 0.1)
    y_expected_track, cache_expected_track = decode_expected(cache_track_np)
    post_update = cache_expected_track[: x_np.shape[0]].clone()
    cache_expected_track[5] = post_update[0]
    cache_expected_track[3] = post_update[2]
    cache = _xpu(cache_track_np, dtype=dtype)
    y = fused_causal_conv1d_update_decode(
        x,
        weight_wd,
        cache,
        torch.arange(x.shape[0], dtype=torch.int32, device="xpu"),
        torch.ones(x.shape[0], dtype=torch.bool, device="xpu"),
        activation="silu",
        use_residual=False,
        track_mask=torch.tensor([True, False, True], dtype=torch.bool, device="xpu"),
        track_indices=torch.tensor([5, 4, 3], dtype=torch.int64, device="xpu"),
    )
    _assert_close(y, y_expected_track, atol=atol, rtol=rtol)
    _assert_close(cache, cache_expected_track, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_silu", [False, True])
@pytest.mark.parametrize("use_residual", [False, True])
def test_inkling_fused_decode_update_w4_packed_path(dtype, use_silu, use_residual):
    _reference_root()

    from inkling.reference_common import make_rng, rand, np_silu
    from sgl_kernel.inkling_sconv import fused_causal_conv1d_update_decode

    gen = make_rng()
    x_np = rand(gen, (5, 8))
    cache_np = rand(gen, (9, 3, 8), 0.1)
    weight_np = rand(gen, (8, 4), 0.2)
    cache_indices_np = torch.tensor([0, 1, -1, 3, 4], dtype=torch.int32)
    cache_mask_np = torch.tensor([True, False, False, True, True], dtype=torch.bool)
    track_mask_np = torch.tensor([True, True, False, False, True], dtype=torch.bool)
    track_indices_np = torch.tensor([6, 7, 8, 8, 5], dtype=torch.int64)
    atol, rtol = _tol(dtype)

    y_expected = torch.zeros((x_np.shape[0], x_np.shape[1]), dtype=torch.float32)
    cache_expected = torch.as_tensor(cache_np.copy())
    for t in range(x_np.shape[0]):
        slot = int(cache_indices_np[t])
        valid = slot != -1
        safe_slot = slot if valid else 0
        mask = bool(cache_mask_np[t])
        acc = torch.zeros((x_np.shape[1],), dtype=torch.float32)
        for w in range(3):
            if mask:
                acc = acc + torch.as_tensor(cache_np[safe_slot, w]) * torch.as_tensor(weight_np[:, w])
        acc = acc + torch.as_tensor(x_np[t]) * torch.as_tensor(weight_np[:, 3])
        if use_silu:
            acc = torch.as_tensor(np_silu(acc.numpy()))
        if use_residual:
            acc = acc + torch.as_tensor(x_np[t])
        y_expected[t] = acc

        if valid:
            updated = torch.empty((3, x_np.shape[1]), dtype=torch.float32)
            updated[0] = torch.as_tensor(cache_np[slot, 1]) if mask else 0
            updated[1] = torch.as_tensor(cache_np[slot, 2]) if mask else 0
            updated[2] = torch.as_tensor(x_np[t])
            cache_expected[slot] = updated
            if bool(track_mask_np[t]):
                cache_expected[int(track_indices_np[t])] = updated

    cache = _xpu(cache_np, dtype=dtype)
    y = fused_causal_conv1d_update_decode(
        _xpu(x_np, dtype=dtype),
        _xpu(weight_np, dtype=dtype),
        cache,
        cache_indices_np.to(device="xpu"),
        cache_mask_np.to(device="xpu"),
        activation="silu" if use_silu else None,
        use_residual=use_residual,
        track_mask=track_mask_np.to(device="xpu"),
        track_indices=track_indices_np.to(device="xpu"),
    )
    _assert_close(y, y_expected, atol=atol, rtol=rtol)
    _assert_close(cache, cache_expected, atol=atol, rtol=rtol)


def test_inkling_gather_scatter_and_draft_extend_against_modeltune_reference():
    root = _reference_root()
    ref = _load_reference(root, "01_04_gather_scatter_and_draft_extend")

    from inkling.reference_common import make_rng, rand
    from sgl_kernel.inkling_sconv import (
        fused_draft_extend_sconv_cache,
        fused_gather_scatter_to_sconv_cache,
    )

    gen = make_rng()
    cache_np = rand(gen, (5, 3, 4))
    updates_np = rand(gen, (2, 3, 4))
    gathered_expected = cache_np.copy()
    gathered_expected[[3, 4]] = updates_np

    cache = _xpu(cache_np)
    fused_gather_scatter_to_sconv_cache(
        _xpu(updates_np.reshape(-1, 4)),
        cache,
        torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device="xpu"),
        torch.ones(2, dtype=torch.bool, device="xpu"),
        torch.tensor([3, 4], dtype=torch.int64, device="xpu"),
    )
    _assert_close(cache, gathered_expected)

    x_np = rand(gen, (5, 4))
    starts = torch.tensor([0, 2, 5], dtype=torch.int64)
    drafted_expected = torch.as_tensor(gathered_expected.copy())
    for local_b, cache_b in enumerate([1, 3]):
        start, end = int(starts[local_b]), int(starts[local_b + 1])
        history = torch.cat([drafted_expected[cache_b], torch.as_tensor(x_np[start:end])], dim=0)
        drafted_expected[cache_b] = history[-drafted_expected.shape[1] :]

    hidden = torch.zeros((2, 3, 4), dtype=torch.float32, device="xpu")
    hidden[0, :2] = _xpu(x_np[:2])
    hidden[1, :3] = _xpu(x_np[2:5])
    fused_draft_extend_sconv_cache(
        hidden,
        cache,
        torch.tensor([1, 3], dtype=torch.int32, device="xpu"),
        num_accepted_tokens=torch.tensor([2, 3], dtype=torch.int32, device="xpu"),
        draft_token_num=3,
    )
    _assert_close(cache, drafted_expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_inkling_gather_scatter_and_draft_extend_packed_paths(dtype):
    _reference_root()

    from inkling.reference_common import make_rng, rand
    from sgl_kernel.inkling_sconv import (
        fused_draft_extend_sconv_cache,
        fused_gather_scatter_to_sconv_cache,
    )

    gen = make_rng()
    cache_np = rand(gen, (8, 3, 8), 0.1)
    hidden_np = rand(gen, (12, 8))
    track_idx = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=torch.int32
    )
    mask = torch.tensor([True, False, True, True], dtype=torch.bool)
    dst = torch.tensor([5, 6, -1, 2], dtype=torch.int64)
    gathered_expected = torch.as_tensor(cache_np.copy())
    gathered_expected[5] = torch.as_tensor(hidden_np[[0, 1, 2]])
    gathered_expected[2] = torch.as_tensor(hidden_np[[9, 10, 11]])

    cache = _xpu(cache_np, dtype=dtype)
    fused_gather_scatter_to_sconv_cache(
        _xpu(hidden_np, dtype=dtype),
        cache,
        track_idx.to(device="xpu"),
        mask.to(device="xpu"),
        dst.to(device="xpu"),
    )
    atol, rtol = _tol(dtype)
    _assert_close(cache, gathered_expected, atol=atol, rtol=rtol)

    draft_hidden_np = rand(gen, (4, 5, 8))
    cache_indices = torch.tensor([0, 1, -1, 4], dtype=torch.int32)
    accepted = torch.tensor([0, 2, 3, 5], dtype=torch.int32)
    crossed = torch.tensor([True, True, True, False], dtype=torch.bool)
    track_step = torch.tensor([1, 0, 2, 2], dtype=torch.int32)
    track_dst = torch.tensor([3, 1, 2, 7], dtype=torch.int64)

    drafted_expected = gathered_expected.clone()
    before = drafted_expected.clone()
    for b in range(cache_indices.numel()):
        slot = int(cache_indices[b])
        if slot == -1 or int(accepted[b]) < 0:
            continue
        virtual = torch.cat([before[slot], torch.as_tensor(draft_hidden_np[b])], dim=0)
        if bool(crossed[b]) and int(track_dst[b]) != -1:
            drafted_expected[int(track_dst[b])] = virtual[int(track_step[b]) : int(track_step[b]) + 3]
        drafted_expected[slot] = virtual[int(accepted[b]) : int(accepted[b]) + 3]

    fused_draft_extend_sconv_cache(
        _xpu(draft_hidden_np, dtype=dtype),
        cache,
        cache_indices.to(device="xpu"),
        num_accepted_tokens=accepted.to(device="xpu"),
        draft_token_num=5,
        do_tracking=True,
        crossed=crossed.to(device="xpu"),
        track_step=track_step.to(device="xpu"),
        mamba_track_indices=track_dst.to(device="xpu"),
    )
    _assert_close(cache, drafted_expected, atol=atol, rtol=rtol)


def test_inkling_metadata_and_windows_smoke():
    _reference_root()

    from sgl_kernel.inkling_sconv import (
        HIS_PREFIX,
        fused_decode_sconv_metadata,
        fused_extend_sconv_metadata,
        save_intermediate_conv_windows,
        track_conv_indices,
    )

    decode_cache_indices = torch.tensor([7, -1, 2], dtype=torch.int32, device="xpu")
    decode_query_start_loc, decode_has_initial_state, decode_meta = fused_decode_sconv_metadata(
        3, decode_cache_indices
    )
    _assert_close(decode_query_start_loc, torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    _assert_close(decode_has_initial_state, torch.tensor([True, True, True]))
    _assert_close(decode_meta["cache_mask"], torch.tensor([[[True]], [[False]], [[True]]]))
    _assert_close(decode_meta["safe_idx"], torch.tensor([7, 0, 2], dtype=torch.int64))
    _assert_close(decode_meta["cu"], torch.tensor([0, 1, 2, 3], dtype=torch.int64))
    _assert_close(decode_meta["si"], torch.tensor([0, 1, 2], dtype=torch.int32))

    cache_indices = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    query_start_loc, has_initial_state, meta = fused_extend_sconv_metadata(
        B=2,
        T=5,
        cache_indices=cache_indices,
        his_mode=HIS_PREFIX,
        extend_seq_lens=torch.tensor([2, 3], dtype=torch.int32, device="xpu"),
        his_src=torch.tensor([1, 1], dtype=torch.int32, device="xpu"),
    )
    _assert_close(query_start_loc, torch.tensor([0, 2, 5], dtype=torch.int32))
    _assert_close(has_initial_state, torch.tensor([True, True]))
    _assert_close(meta["safe_idx"], torch.tensor([0, 1], dtype=torch.int64))
    _assert_close(meta["cu"], torch.tensor([0, 2, 5], dtype=torch.int64))
    _assert_close(meta["si"], torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32))

    track = track_conv_indices(
        query_start_loc,
        torch.tensor([5, 7], dtype=torch.int32, device="xpu"),
        torch.tensor([1, 3], dtype=torch.int32, device="xpu"),
        width_minus_one=3,
        chunk_size=2,
        total_tokens=5,
    )
    _assert_close(track, torch.tensor([[1, 2, 3], [3, 4, 4]], dtype=torch.int32))

    cache = torch.arange(2 * 2 * 3, dtype=torch.float32, device="xpu").reshape(2, 2, 3)
    hidden = torch.arange(2 * 3 * 3, dtype=torch.float32, device="xpu").reshape(2, 3, 3) + 100
    out = torch.empty((2, 2, 2, 3), dtype=torch.float32, device="xpu")
    save_intermediate_conv_windows(cache, hidden, cache_indices, out, batch_size=2, draft_token_num=2)

    expected = torch.empty((2, 2, 2, 3), dtype=torch.float32)
    for b in range(2):
        virtual = torch.cat([cache.detach().cpu()[b], hidden.detach().cpu()[b]], dim=0)
        for t in range(2):
            for w in range(2):
                expected[b, t, w] = virtual[t + 1 + w]
    _assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D", [16, 17])
@pytest.mark.parametrize("hidden_layout", ["btd", "flat"])
def test_inkling_save_intermediate_conv_windows_packed_paths(dtype, D, hidden_layout):
    _reference_root()

    from inkling.reference_common import make_rng, rand
    from sgl_kernel.inkling_sconv import save_intermediate_conv_windows

    gen = make_rng()
    B = 4
    draft_token_num = 5
    W1 = 3
    cache_np = rand(gen, (8, W1, D), 0.1)
    hidden_np = rand(gen, (B, draft_token_num, D))
    cache_indices = torch.tensor([0, 2, -1, 5], dtype=torch.int32, device="xpu")

    cache = _xpu(cache_np, dtype=dtype)
    hidden_input_np = hidden_np if hidden_layout == "btd" else hidden_np.reshape(B * draft_token_num, D)
    hidden = _xpu(hidden_input_np, dtype=dtype)
    out = torch.full((B, draft_token_num, W1, D), -7.0, dtype=dtype, device="xpu")
    expected = out.detach().cpu().clone()
    cache_cpu = cache.detach().cpu()
    hidden_cpu = hidden.detach().cpu().reshape(B, draft_token_num, D)
    for b, slot in enumerate(cache_indices.cpu().tolist()):
        if slot == -1:
            continue
        virtual = torch.cat([cache_cpu[slot], hidden_cpu[b]], dim=0)
        for t in range(draft_token_num):
            for w in range(W1):
                expected[b, t, w] = virtual[t + 1 + w]

    save_intermediate_conv_windows(
        cache,
        hidden,
        cache_indices,
        out,
        batch_size=B,
        draft_token_num=draft_token_num,
    )
    atol, rtol = _tol(dtype)
    _assert_close(out, expected, atol=atol, rtol=rtol)
