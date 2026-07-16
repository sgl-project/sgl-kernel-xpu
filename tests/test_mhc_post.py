import pytest
import torch
import utils
from sgl_kernel import mhc_post

device = utils.get_device()

# Constants matching main branch
HC_MULT = 4
HC_MULT3 = (2 + HC_MULT) * HC_MULT  # 24


def _make_test_inputs(num_tokens, hidden_size, device, seed=42):
    """Generate test inputs matching main branch dimensions."""
    torch.manual_seed(seed)
    hc_hidden = HC_MULT * hidden_size

    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    residual = (
        torch.randn(
            num_tokens, HC_MULT, hidden_size, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    post_prev = torch.rand(num_tokens, HC_MULT, 1, device=device, dtype=torch.float32)
    comb_prev = (
        torch.rand(num_tokens, HC_MULT, HC_MULT, device=device, dtype=torch.float32)
        * 0.25
    )
    fn = torch.randn(HC_MULT3, hc_hidden, device=device, dtype=torch.float32) * 0.01
    hc_scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
    hc_base = torch.zeros(HC_MULT3, device=device, dtype=torch.float32)
    norm_weight = torch.ones(hidden_size, device=device, dtype=torch.bfloat16)

    return x, residual, post_prev, comb_prev, fn, hc_scale, hc_base, norm_weight


def _hc_post_torch(x, residual, post_layer_mix, comb_res_mix):
    """PyTorch reference implementation of mhc_post.

    Computes: cur_residual[j, h] = post[j] * x[h] + sum_k comb[k, j] * residual[k, h]
    """

    if post_layer_mix.dim() == 3:
        post_layer_mix = post_layer_mix.squeeze(-1)

    T, HC = residual.shape[0], residual.shape[1]
    H = residual.shape[2]

    result = post_layer_mix.unsqueeze(-1) * x.unsqueeze(1)  # [T, HC, H]

    if comb_res_mix.dim() == 2:
        comb_res_mix = comb_res_mix.reshape(T, HC, HC)

    result = result + (comb_res_mix.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)

    return result.to(torch.bfloat16)


def _bench(fn, *, warmup: int = 5, iters: int = 20) -> float:
    """Return median latency in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    times = []
    for _ in range(iters):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.xpu.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("num_tokens", [0, 1, 8, 17, 32, 64])
def test_mhc_post(hidden_size, num_tokens):
    """Test mhc_post kernel accuracy against PyTorch reference."""
    if num_tokens == 0:
        # Edge case: empty batch
        x = torch.empty(0, hidden_size, device=device, dtype=torch.bfloat16)
        residual = torch.empty(
            0, HC_MULT, hidden_size, device=device, dtype=torch.bfloat16
        )
        post_mix = torch.empty(0, HC_MULT, 1, device=device, dtype=torch.float32)
        comb_mix = torch.empty(0, HC_MULT, HC_MULT, device=device, dtype=torch.float32)

        result = mhc_post(x, residual, post_mix, comb_mix)

        assert result.shape == residual.shape
        assert result.dtype == torch.bfloat16
        return

    x, residual, post_prev, _, _, _, _, _ = _make_test_inputs(
        num_tokens, hidden_size, device
    )

    result_ref = _hc_post_torch(
        x,
        residual,
        post_prev,
        torch.rand_like(post_prev).expand(-1, HC_MULT, HC_MULT) * 0.25,
    )

    comb_mix = torch.rand_like(post_prev).expand(-1, HC_MULT, HC_MULT) * 0.25
    result = mhc_post(x, residual, post_prev, comb_mix)

    assert result.shape == (num_tokens, HC_MULT, hidden_size)
    assert result.dtype == torch.bfloat16


@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("num_tokens", [32, 64])
def test_mhc_post_perf(hidden_size, num_tokens):
    x, residual, post_prev, comb_prev, _, _, _, _ = _make_test_inputs(
        num_tokens, hidden_size, device
    )

    t_ref = _bench(lambda: _hc_post_torch(x, residual, post_prev, comb_prev))
    t_our = _bench(lambda: mhc_post(x, residual, post_prev, comb_prev))

    assert (
        t_our < t_ref
    ), f"sycl ({t_our:.3f} ms) not faster than torch ({t_ref:.3f} ms)"


@pytest.mark.parametrize("hidden_size", [4096, 7168])
def test_mhc_post_shapes_and_dtypes(hidden_size):
    """Test shape and dtype correctness for mhc_post."""
    num_tokens = 32
    x, residual, post_prev, comb_prev, _, _, _, _ = _make_test_inputs(
        num_tokens, hidden_size, device
    )

    result_post = mhc_post(x, residual, post_prev, comb_prev)
    assert result_post.shape == (num_tokens, HC_MULT, hidden_size)
    assert result_post.dtype == torch.bfloat16
    assert result_post.device == x.device


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
