# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/tests/test_norm.py

import sys

import pytest
import sgl_kernel
import torch
import utils

device = utils.get_device()


def llama_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def gemma_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x


def gemma_fused_add_rms_norm(x, residual, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x + residual
    residual = x
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x, residual


def fused_add_rms_norm(x, residual, weight, eps):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = (x * weight.float()).to(orig_dtype)
    return x, residual


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_norm(batch_size, hidden_size, dtype, specify_out):
    x = torch.randn(batch_size, hidden_size).to(device).to(dtype)
    w = torch.randn(hidden_size).to(device).to(dtype)

    y_ref = llama_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        sgl_kernel.rmsnorm(x, w, out=y)
    else:
        y = sgl_kernel.rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_add_rmsnorm(batch_size, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    sgl_kernel.fused_add_rmsnorm(x_fused, residual_fused, weight, eps)

    torch.testing.assert_close(x_fused, x_native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_gemma_norm(batch_size, hidden_size, dtype, specify_out):
    x = torch.randn(batch_size, hidden_size).to(device).to(dtype)
    w = torch.randn(hidden_size).to(device).to(dtype)

    y_ref = gemma_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        sgl_kernel.gemma_rmsnorm(x, w, out=y)
    else:
        y = sgl_kernel.gemma_rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_gemma_fused_add_rmsnorm(batch_size, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    x_native, residual_native = gemma_fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    sgl_kernel.gemma_fused_add_rmsnorm(x_fused, residual_fused, weight, eps)

    torch.testing.assert_close(x_fused, x_native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


###############################################################################
# Non-contiguous input tests (DeepSeek split pattern: stride[0] != hidden_size)
###############################################################################


def _make_non_contiguous(batch_size, hidden_size, dtype, extra=64):
    """Create a non-contiguous tensor by slicing a larger tensor,
    mimicking latent_cache.split([hidden_size, extra], dim=-1)[0]."""
    full = torch.randn(batch_size, hidden_size + extra, device=device, dtype=dtype)
    view = full[:, :hidden_size]  # stride = (hidden_size + extra, 1)
    # assert not view.is_contiguous()
    assert view.stride(0) == hidden_size + extra
    return view


@pytest.mark.parametrize("batch_size", [1, 19, 99])
@pytest.mark.parametrize("hidden_size", [512, 1024, 3072])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_norm_non_contiguous(batch_size, hidden_size, dtype):
    x_nc = _make_non_contiguous(batch_size, hidden_size, dtype)
    w = torch.randn(hidden_size, device=device, dtype=dtype)

    y_ref = llama_rms_norm(x_nc.clone(), w)
    y = sgl_kernel.rmsnorm(x_nc, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99])
@pytest.mark.parametrize("hidden_size", [512, 1024, 3072])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_gemma_norm_non_contiguous(batch_size, hidden_size, dtype):
    x_nc = _make_non_contiguous(batch_size, hidden_size, dtype)
    w = torch.randn(hidden_size, device=device, dtype=dtype)

    y_ref = gemma_rms_norm(x_nc.clone(), w)
    y = sgl_kernel.gemma_rmsnorm(x_nc, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


###############################################################################
# 3D tensor input tests (batch_size, seq_len, hidden_size)
###############################################################################


@pytest.mark.parametrize("batch_size", [1, 4, 19])
@pytest.mark.parametrize("seq_len", [1, 7, 32])
@pytest.mark.parametrize("hidden_size", [111, 1024, 4096])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_norm_3d(batch_size, seq_len, hidden_size, dtype, specify_out):
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    w = torch.randn(hidden_size, device=device, dtype=dtype)

    y_ref = llama_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        sgl_kernel.rmsnorm(x, w, out=y)
    else:
        y = sgl_kernel.rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4, 19])
@pytest.mark.parametrize("seq_len", [1, 7, 32])
@pytest.mark.parametrize("hidden_size", [111, 1024, 4096])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_fused_add_rmsnorm_3d(batch_size, seq_len, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    sgl_kernel.fused_add_rmsnorm(x_fused, residual_fused, weight, eps)

    torch.testing.assert_close(x_fused, x_native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4, 19])
@pytest.mark.parametrize("seq_len", [1, 7, 32])
@pytest.mark.parametrize("hidden_size", [111, 1024, 4096])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_gemma_norm_3d(batch_size, seq_len, hidden_size, dtype, specify_out):
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    w = torch.randn(hidden_size, device=device, dtype=dtype)

    y_ref = gemma_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        sgl_kernel.gemma_rmsnorm(x, w, out=y)
    else:
        y = sgl_kernel.gemma_rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4, 19])
@pytest.mark.parametrize("seq_len", [1, 7, 32])
@pytest.mark.parametrize("hidden_size", [111, 1024, 4096])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_gemma_fused_add_rmsnorm_3d(batch_size, seq_len, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    x_native, residual_native = gemma_fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    sgl_kernel.gemma_fused_add_rmsnorm(x_fused, residual_fused, weight, eps)

    torch.testing.assert_close(x_fused, x_native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


###############################################################################
# Non-contiguous 3D tensor tests (sliced last-dim, flattenable leading dims)
###############################################################################


def _make_non_contiguous_3d(batch_size, seq_len, hidden_size, dtype, extra=64):
    """Create a last-dim-contiguous but not fully contiguous 3D tensor by
    slicing a larger tensor along the last dimension."""
    full = torch.randn(
        batch_size, seq_len, hidden_size + extra, device=device, dtype=dtype
    )
    view = full[
        :, :, :hidden_size
    ]  # stride = (seq_len*(hidden_size+extra), hidden_size+extra, 1)
    assert view.stride(-1) == 1
    assert view.stride(-2) == hidden_size + extra
    return view


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 7])
@pytest.mark.parametrize("hidden_size", [512, 1024])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_norm_3d_non_contiguous(batch_size, seq_len, hidden_size, dtype):
    x_nc = _make_non_contiguous_3d(batch_size, seq_len, hidden_size, dtype)
    w = torch.randn(hidden_size, device=device, dtype=dtype)

    y_ref = llama_rms_norm(x_nc.clone(), w)
    y = sgl_kernel.rmsnorm(x_nc, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 7])
@pytest.mark.parametrize("hidden_size", [512, 1024])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_gemma_norm_3d_non_contiguous(batch_size, seq_len, hidden_size, dtype):
    x_nc = _make_non_contiguous_3d(batch_size, seq_len, hidden_size, dtype)
    w = torch.randn(hidden_size, device=device, dtype=dtype)

    y_ref = gemma_rms_norm(x_nc.clone(), w)
    y = sgl_kernel.gemma_rmsnorm(x_nc, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


###############################################################################
# Non-flattenable 3D tensor tests (QKV slice pattern: stride(0) != size(1)*stride(1))
###############################################################################


def _make_non_flattenable_3d(num_tokens, num_heads, head_dim, dtype, extra_heads=4):
    """Create a 3D tensor whose leading dimensions are NOT flattenable.

    This mimics the pattern where q is obtained by slicing a packed QKV
    buffer along dim-1 and then unflattening to (tokens, heads, head_dim).
    The resulting tensor has stride(0) = (num_heads + extra_heads) * head_dim,
    which differs from size(1) * stride(1) = num_heads * head_dim.

    Note: parametrizations should use num_tokens > 1 so the flattenability
    check (which short-circuits when size(0) == 1) is actually exercised.
    """
    assert num_tokens > 1, "use num_tokens > 1 to exercise the non-flattenable path"
    total_heads = num_heads + extra_heads
    full = torch.randn(num_tokens, total_heads * head_dim, device=device, dtype=dtype)
    # Slice the first num_heads*head_dim columns (non-contiguous in dim-0)
    q_flat = full[:, : num_heads * head_dim]
    # Unflatten to 3D – strides become (total_heads*head_dim, head_dim, 1)
    q_3d = q_flat.unflatten(-1, (num_heads, head_dim))
    assert q_3d.stride(0) == total_heads * head_dim
    assert q_3d.stride(0) != q_3d.size(1) * q_3d.stride(1)
    return q_3d


@pytest.mark.parametrize("num_tokens", [7, 32])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_norm_3d_non_flattenable(num_tokens, num_heads, head_dim, dtype, specify_out):
    x = _make_non_flattenable_3d(num_tokens, num_heads, head_dim, dtype)
    w = torch.randn(head_dim, device=device, dtype=dtype)

    y_ref = llama_rms_norm(x.clone(), w)
    if specify_out:
        y = torch.empty_like(x)
        sgl_kernel.rmsnorm(x, w, out=y)
    else:
        y = sgl_kernel.rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("num_tokens", [7, 32])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_gemma_norm_3d_non_flattenable(
    num_tokens, num_heads, head_dim, dtype, specify_out
):
    x = _make_non_flattenable_3d(num_tokens, num_heads, head_dim, dtype)
    w = torch.randn(head_dim, device=device, dtype=dtype)

    y_ref = gemma_rms_norm(x.clone(), w)
    if specify_out:
        y = torch.empty_like(x)
        sgl_kernel.gemma_rmsnorm(x, w, out=y)
    else:
        y = sgl_kernel.gemma_rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
