import pytest
import torch
from sconv_reference import rand, update_cache_ref

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Inkling sconv ops are XPU-only",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_update_sconv_cache_matches_inkling_pr_semantics(dtype):
    from sgl_kernel.inkling_sconv import update_sconv_cache

    torch.manual_seed(2)
    x = rand((7, 6), dtype)
    cache = rand((5, 3, 6), dtype, scale=0.1)
    cache_indices = torch.tensor([0, 1, -1, 3], dtype=torch.int32, device="xpu")
    has_initial_state = torch.tensor(
        [True, False, True, True], dtype=torch.bool, device="xpu"
    )
    query_start_loc = torch.tensor([0, 1, 4, 4, 7], dtype=torch.int32, device="xpu")
    expected = update_cache_ref(
        x, cache, cache_indices, has_initial_state, query_start_loc
    )

    update_sconv_cache(x, cache, cache_indices, has_initial_state, query_start_loc)
    torch.testing.assert_close(
        cache.detach().cpu(), expected, atol=0, rtol=0, check_dtype=False
    )
