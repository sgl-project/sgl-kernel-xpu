import pytest
import torch
from sconv_reference import rand, windows_ref

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Inkling sconv ops are XPU-only",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden_layout", ["btd", "flat"])
def test_save_intermediate_conv_windows_matches_inkling_pr_semantics(
    dtype, hidden_layout
):
    from sgl_kernel.inkling_sconv import save_intermediate_conv_windows

    torch.manual_seed(6)
    B, draft_token_num, D, W1 = 4, 5, 8, 3
    cache = rand((8, W1, D), dtype, scale=0.1)
    hidden_btd = rand((B, draft_token_num, D), dtype)
    hidden = (
        hidden_btd
        if hidden_layout == "btd"
        else hidden_btd.reshape(B * draft_token_num, D)
    )
    cache_indices = torch.tensor([0, 2, -1, 5], dtype=torch.int32, device="xpu")
    out = torch.zeros((B, draft_token_num, W1, D), dtype=dtype, device="xpu")
    expected = windows_ref(
        cache,
        hidden,
        cache_indices,
        out.shape,
        batch_size=B,
        draft_token_num=draft_token_num,
    )

    save_intermediate_conv_windows(
        cache,
        hidden,
        cache_indices,
        out,
        batch_size=B,
        draft_token_num=draft_token_num,
    )
    torch.testing.assert_close(
        out.detach().cpu(), expected, atol=0, rtol=0, check_dtype=False
    )
