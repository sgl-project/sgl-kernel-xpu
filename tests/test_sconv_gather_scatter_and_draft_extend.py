import pytest
import torch
from sconv_reference import draft_extend_ref, gather_scatter_ref, rand

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Inkling sconv ops are XPU-only",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gather_scatter_sconv_cache_matches_inkling_pr_semantics(dtype):
    from sgl_kernel.inkling_sconv import fused_gather_scatter_to_sconv_cache

    torch.manual_seed(4)
    hidden = rand((12, 6), dtype)
    cache = rand((8, 3, 6), dtype, scale=0.1)
    track_idx = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=torch.int32, device="xpu"
    )
    mask = torch.tensor([True, False, True, True], dtype=torch.bool, device="xpu")
    dst = torch.tensor([5, 6, -1, 2], dtype=torch.int64, device="xpu")
    expected = gather_scatter_ref(hidden, cache, track_idx, mask, dst)

    fused_gather_scatter_to_sconv_cache(hidden, cache, track_idx, mask, dst)
    torch.testing.assert_close(
        cache.detach().cpu(), expected, atol=0, rtol=0, check_dtype=False
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden_layout", ["btd", "flat"])
def test_draft_extend_sconv_cache_matches_inkling_pr_semantics(dtype, hidden_layout):
    from sgl_kernel.inkling_sconv import fused_draft_extend_sconv_cache

    torch.manual_seed(5)
    B, draft_token_num, D, W1 = 4, 5, 8, 3
    hidden_btd = rand((B, draft_token_num, D), dtype)
    hidden = (
        hidden_btd
        if hidden_layout == "btd"
        else hidden_btd.reshape(B * draft_token_num, D)
    )
    cache = rand((8, W1, D), dtype, scale=0.1)
    cache_indices = torch.tensor([0, 1, -1, 4], dtype=torch.int32, device="xpu")
    accepted = torch.tensor([0, 2, 3, 5], dtype=torch.int32, device="xpu")
    crossed = torch.tensor([True, True, True, False], dtype=torch.bool, device="xpu")
    track_step = torch.tensor([1, 0, 2, 2], dtype=torch.int32, device="xpu")
    track_dst = torch.tensor([3, 6, 2, 7], dtype=torch.int64, device="xpu")
    expected = draft_extend_ref(
        hidden,
        cache,
        cache_indices,
        accepted,
        draft_token_num=draft_token_num,
        do_tracking=True,
        crossed=crossed,
        track_step=track_step,
        mamba_track_indices=track_dst,
    )

    fused_draft_extend_sconv_cache(
        hidden,
        cache,
        cache_indices,
        num_accepted_tokens=accepted,
        draft_token_num=draft_token_num,
        do_tracking=True,
        crossed=crossed,
        track_step=track_step,
        mamba_track_indices=track_dst,
    )
    torch.testing.assert_close(
        cache.detach().cpu(), expected, atol=0, rtol=0, check_dtype=False
    )
