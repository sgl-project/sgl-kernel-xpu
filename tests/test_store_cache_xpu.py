"""Tests for store_cache_xpu kernel."""

import pytest
import torch


@pytest.fixture(autouse=True)
def skip_if_no_xpu():
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")


def reference_store_cache(k, v, k_cache, v_cache, indices):
    """Reference implementation using index_put."""
    k_cache[indices] = k
    v_cache[indices] = v


class TestStoreCacheXPU:
    @pytest.mark.parametrize("num_tokens", [1, 4, 32, 128])
    @pytest.mark.parametrize("row_dim", [128, 256, 512, 1024])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_parity(self, num_tokens, row_dim, dtype):
        from sgl_kernel import store_cache_xpu

        torch.manual_seed(42)
        cache_size = 2048

        k = torch.randn(num_tokens, row_dim, dtype=dtype, device="xpu")
        v = torch.randn(num_tokens, row_dim, dtype=dtype, device="xpu")
        indices = torch.randperm(cache_size, device="xpu")[:num_tokens].to(torch.int64)

        k_cache_ref = torch.zeros(cache_size, row_dim, dtype=dtype, device="xpu")
        v_cache_ref = torch.zeros_like(k_cache_ref)
        k_cache_test = torch.zeros_like(k_cache_ref)
        v_cache_test = torch.zeros_like(k_cache_ref)

        reference_store_cache(k, v, k_cache_ref, v_cache_ref, indices)
        store_cache_xpu(k, v, k_cache_test, v_cache_test, indices)

        torch.testing.assert_close(k_cache_test, k_cache_ref)
        torch.testing.assert_close(v_cache_test, v_cache_ref)

    def test_negative_indices_skip(self):
        """Tokens with index=-1 should not write to cache."""
        from sgl_kernel import store_cache_xpu

        torch.manual_seed(42)
        num_tokens, row_dim, cache_size = 4, 256, 1024

        k = torch.randn(num_tokens, row_dim, dtype=torch.bfloat16, device="xpu")
        v = torch.randn(num_tokens, row_dim, dtype=torch.bfloat16, device="xpu")
        indices = torch.tensor([10, -1, 20, -1], dtype=torch.int64, device="xpu")

        k_cache = torch.zeros(cache_size, row_dim, dtype=torch.bfloat16, device="xpu")
        v_cache = torch.zeros_like(k_cache)

        store_cache_xpu(k, v, k_cache, v_cache, indices)

        zero_row = torch.zeros(row_dim, dtype=torch.bfloat16, device="xpu")

        # Written slots should have data
        assert not torch.all(k_cache[10] == 0).item()
        assert not torch.all(v_cache[10] == 0).item()
        assert not torch.all(k_cache[20] == 0).item()
        assert not torch.all(v_cache[20] == 0).item()

        # Skipped slots should remain zero
        for slot in [0, 1, 5, 15, 100]:
            torch.testing.assert_close(k_cache[slot], zero_row)
            torch.testing.assert_close(v_cache[slot], zero_row)

    def test_empty_input(self):
        """Zero tokens should not crash."""
        from sgl_kernel import store_cache_xpu

        k = torch.empty(0, 256, dtype=torch.bfloat16, device="xpu")
        v = torch.empty(0, 256, dtype=torch.bfloat16, device="xpu")
        k_cache = torch.zeros(1024, 256, dtype=torch.bfloat16, device="xpu")
        v_cache = torch.zeros_like(k_cache)
        indices = torch.empty(0, dtype=torch.int64, device="xpu")

        store_cache_xpu(k, v, k_cache, v_cache, indices)
        assert torch.all(k_cache == 0).item()

    def test_single_token(self):
        """Single token decode (most common case)."""
        from sgl_kernel import store_cache_xpu

        torch.manual_seed(0)
        row_dim = 512

        k = torch.randn(1, row_dim, dtype=torch.bfloat16, device="xpu")
        v = torch.randn(1, row_dim, dtype=torch.bfloat16, device="xpu")
        k_cache = torch.zeros(4096, row_dim, dtype=torch.bfloat16, device="xpu")
        v_cache = torch.zeros_like(k_cache)
        indices = torch.tensor([42], dtype=torch.int64, device="xpu")

        store_cache_xpu(k, v, k_cache, v_cache, indices)

        torch.testing.assert_close(k_cache[42], k[0])
        torch.testing.assert_close(v_cache[42], v[0])

    @pytest.mark.parametrize("num_heads", [2, 10])
    @pytest.mark.parametrize("head", [0, 1])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_strided_head_slice(self, num_heads, head, dtype):
        """Non-contiguous K/V: a per-head slice of a [tokens, heads, dim]
        tensor has row stride heads*dim (not dim). The kernel must address
        source rows by their real stride, matching index_put.

        This is the layout that sliding-window-attention (SWA) layers pass on
        Gemma-style models, e.g. k.stride == (2560, 1) for heads=10, dim=256.
        """
        from sgl_kernel import store_cache_xpu

        torch.manual_seed(123)
        num_tokens, row_dim, cache_size = 271, 256, 2048

        kw = torch.randn(num_tokens, num_heads, row_dim, dtype=dtype, device="xpu")
        vw = torch.randn(num_tokens, num_heads, row_dim, dtype=dtype, device="xpu")
        k = kw[:, head, :]  # shape (num_tokens, row_dim), stride (num_heads*row_dim, 1)
        v = vw[:, head, :]
        assert not k.is_contiguous()
        assert k.stride() == (num_heads * row_dim, 1)

        indices = torch.randperm(cache_size, device="xpu")[:num_tokens].to(torch.int64)

        k_cache_ref = torch.zeros(cache_size, row_dim, dtype=dtype, device="xpu")
        v_cache_ref = torch.zeros_like(k_cache_ref)
        k_cache_test = torch.zeros_like(k_cache_ref)
        v_cache_test = torch.zeros_like(k_cache_ref)

        reference_store_cache(k, v, k_cache_ref, v_cache_ref, indices)
        store_cache_xpu(k, v, k_cache_test, v_cache_test, indices)

        torch.testing.assert_close(k_cache_test, k_cache_ref)
        torch.testing.assert_close(v_cache_test, v_cache_ref)
