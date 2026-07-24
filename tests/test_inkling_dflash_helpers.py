import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PKG = _REPO_ROOT / "python" / "sgl_kernel"
_LOCAL_EXT = _REPO_ROOT / "build" / "src" / "inkling_dflash_helpers_ops.abi3.so"

if _LOCAL_PKG.is_dir() and _LOCAL_EXT.is_file() and "sgl_kernel" not in sys.modules:
    pkg = types.ModuleType("sgl_kernel")
    pkg.__path__ = [str(_LOCAL_PKG), str(_LOCAL_EXT.parent)]
    sys.modules["sgl_kernel"] = pkg
    torch.ops.load_library(str(_LOCAL_EXT))
else:
    import sgl_kernel  # noqa: F401

    try:
        importlib.import_module("sgl_kernel.inkling_dflash_helpers_ops")
    except ImportError:
        pass

pytestmark = pytest.mark.skipif(
    not (
        hasattr(torch, "xpu")
        and torch.xpu.is_available()
        and hasattr(torch.ops.sgl_kernel, "inkling_dflash_cache_path")
        and hasattr(
            torch.ops.sgl_kernel,
            "inkling_scatter_mamba_states_after_mtp_verify",
        )
    ),
    reason="Inkling DFLASH helpers are XPU-only",
)


def _offsets_from_mask(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    flat = mask.flatten().to(torch.int32)
    cumsum = torch.cumsum(flat, dim=0)
    offsets = torch.where(flat != 0, cumsum - 1, torch.full_like(cumsum, -1))
    return offsets.to(torch.int32).reshape_as(mask).contiguous(), int(cumsum[-1].item())


def test_dflash_cache_path_matches_reference():
    req_to_token = torch.arange(3 * 11, dtype=torch.int64, device="xpu").reshape(3, 11)
    req_pool_indices = torch.tensor([2, 0, 1], dtype=torch.int64, device="xpu")
    pos2d = torch.tensor(
        [[1, 3, 0, 8], [2, 4, 6, 1], [9, 5, 7, 10]],
        dtype=torch.int64,
        device="xpu",
    )
    mask = torch.tensor(
        [[1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 1]],
        dtype=torch.uint8,
        device="xpu",
    )
    out_offsets, gather_count = _offsets_from_mask(mask)
    logits = torch.tensor(
        [
            [1.0, 2.0, 2.0, -1.0],
            [4.0, -1.0, 8.0, 7.0],
            [3.0, 3.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
        device="xpu",
    )

    gathered, greedy = torch.ops.sgl_kernel.inkling_dflash_cache_path(
        req_to_token,
        req_pool_indices,
        pos2d,
        mask,
        out_offsets,
        gather_count,
        logits,
    )
    torch.xpu.synchronize()

    expected_gather = []
    for b in range(pos2d.size(0)):
        req = int(req_pool_indices[b].item())
        for t in range(pos2d.size(1)):
            if int(mask[b, t].item()) != 0:
                expected_gather.append(int(req_to_token[req, pos2d[b, t]].item()))
    expected_gather = torch.tensor(expected_gather, dtype=torch.int64)
    expected_greedy = torch.tensor([1, 2, 0], dtype=torch.int64)

    torch.testing.assert_close(gathered.cpu(), expected_gather, atol=0, rtol=0)
    torch.testing.assert_close(greedy.cpu(), expected_greedy, atol=0, rtol=0)


def _names_tensor(names: list[str], stride: int = 32) -> torch.Tensor:
    rows = torch.zeros((len(names), stride), dtype=torch.uint8)
    for i, name in enumerate(names):
        encoded = name.encode("ascii")
        rows[i, : len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8)
    return rows.to("xpu")


def test_dflash_device_guard_matches_prefix_rules():
    names = ["cuda:0", "xpu", "cuda", "cpu", "level_zero:gpu", "hip"]
    legacy, supported = torch.ops.sgl_kernel.inkling_dflash_device_guard(
        _names_tensor(names)
    )
    torch.xpu.synchronize()

    expected_legacy = torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.uint8)
    expected_supported = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.uint8)
    torch.testing.assert_close(legacy.cpu(), expected_legacy, atol=0, rtol=0)
    torch.testing.assert_close(supported.cpu(), expected_supported, atol=0, rtol=0)


def _values(shape, dtype, salt):
    n = 1
    for dim in shape:
        n *= dim
    values = torch.arange(n, dtype=torch.float32, device="xpu").reshape(shape)
    return ((values % 251) / 17.0 + salt).to(dtype).contiguous()


def _scatter_ref(dst, intermediate, slots, steps):
    out = dst.clone()
    slots_cpu = slots.cpu().tolist()
    steps_cpu = steps.cpu().tolist()
    for slot, step in zip(slots_cpu, steps_cpu):
        if step >= 0:
            out[slot].copy_(intermediate[slot, step])
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_tracking", [False, True])
def test_scatter_mamba_states_after_mtp_verify_matches_reference(dtype, use_tracking):
    slots_n, t_max, d_ssm, width_a, width_b, d_conv = 7, 5, 19, 3, 4, 13
    ssm = _values((slots_n, d_ssm), dtype, 0.1)
    ssm_inter = _values((slots_n, t_max, d_ssm), dtype, 0.2)
    conv_a = _values((slots_n, width_a, d_conv), dtype, 0.3)
    conv_a_inter = _values((slots_n, t_max, width_a, d_conv), dtype, 0.4)
    conv_b = _values((slots_n, width_b, d_conv), dtype, 0.5)
    conv_b_inter = _values((slots_n, t_max, width_b, d_conv), dtype, 0.6)

    slots = torch.tensor([2, 0, 3, 5, 1], dtype=torch.int64, device="xpu")
    steps = torch.tensor([1, -1, 2, 4, 0], dtype=torch.int64, device="xpu")
    if use_tracking:
        track_slots = torch.tensor([1, 3, 6], dtype=torch.int64, device="xpu")
        track_steps = torch.tensor([3, 0, -1], dtype=torch.int64, device="xpu")
    else:
        track_slots = None
        track_steps = None

    exp_ssm = _scatter_ref(ssm, ssm_inter, slots, steps)
    exp_a = _scatter_ref(conv_a, conv_a_inter, slots, steps)
    exp_b = _scatter_ref(conv_b, conv_b_inter, slots, steps)
    if use_tracking:
        exp_ssm = _scatter_ref(exp_ssm, ssm_inter, track_slots, track_steps)
        exp_a = _scatter_ref(exp_a, conv_a_inter, track_slots, track_steps)
        exp_b = _scatter_ref(exp_b, conv_b_inter, track_slots, track_steps)

    torch.ops.sgl_kernel.inkling_scatter_mamba_states_after_mtp_verify(
        ssm,
        ssm_inter,
        conv_a,
        conv_a_inter,
        conv_b,
        conv_b_inter,
        slots,
        steps,
        track_slots,
        track_steps,
        t_max,
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(ssm.cpu(), exp_ssm.cpu(), atol=0, rtol=0, check_dtype=True)
    torch.testing.assert_close(conv_a.cpu(), exp_a.cpu(), atol=0, rtol=0, check_dtype=True)
    torch.testing.assert_close(conv_b.cpu(), exp_b.cpu(), atol=0, rtol=0, check_dtype=True)


def test_scatter_rejects_mismatched_tracking_args():
    ssm = torch.zeros((2, 4), dtype=torch.bfloat16, device="xpu")
    inter = torch.zeros((2, 3, 4), dtype=torch.bfloat16, device="xpu")
    slots = torch.zeros((1,), dtype=torch.int64, device="xpu")
    steps = torch.zeros((1,), dtype=torch.int64, device="xpu")
    with pytest.raises(RuntimeError, match="both set or both None"):
        torch.ops.sgl_kernel.inkling_scatter_mamba_states_after_mtp_verify(
            ssm,
            inter,
            ssm.clone(),
            inter.clone(),
            ssm.clone(),
            inter.clone(),
            slots,
            steps,
            slots,
            None,
            3,
        )
