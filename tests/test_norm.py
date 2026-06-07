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


def norm_tolerances(dtype):
    if dtype == torch.float32:
        return dict(rtol=1e-4, atol=1e-4)
    if dtype == torch.bfloat16:
        return dict(rtol=1e-2, atol=1e-2)
    return dict(rtol=1e-3, atol=1e-3)


def fused_add_rms_norm(x, residual, weight, eps):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = (x * weight.float()).to(orig_dtype)
    return x, residual


def qk_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    return x.to(orig_dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
def test_fused_inplace_qknorm(dtype, weight_dtype, head_dim):
    eps = 1e-6
    num_tokens, num_q_heads, num_k_heads = 17, 8, 2
    q = torch.randn(num_tokens, num_q_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(num_tokens, num_k_heads, head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=weight_dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=weight_dtype, device=device)

    q_ref = qk_rms_norm(q, q_weight, eps)
    k_ref = qk_rms_norm(k, k_weight, eps)

    sgl_kernel.fused_inplace_qknorm(q, k, q_weight, k_weight, eps)

    torch.testing.assert_close(q, q_ref, **norm_tolerances(dtype))
    torch.testing.assert_close(k, k_ref, **norm_tolerances(dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_fused_inplace_qknorm_qwen3_split_view(dtype, num_tokens, head_dim):
    eps = 1e-6
    num_q_heads, num_k_heads = 40, 8
    q_size = num_q_heads * head_dim
    kv_size = num_k_heads * head_dim
    qkv = torch.randn(
        num_tokens, q_size + kv_size + kv_size, dtype=dtype, device=device
    )
    q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(num_tokens, num_q_heads, head_dim)
    k = k.view(num_tokens, num_k_heads, head_dim)

    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert q.stride(0) == q_size + kv_size + kv_size
    assert k.stride(0) == q_size + kv_size + kv_size

    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    q_ref = qk_rms_norm(q.clone(), q_weight, eps)
    k_ref = qk_rms_norm(k.clone(), k_weight, eps)

    sgl_kernel.fused_inplace_qknorm(q, k, q_weight, k_weight, eps)

    torch.testing.assert_close(q, q_ref, **norm_tolerances(dtype))
    torch.testing.assert_close(k, k_ref, **norm_tolerances(dtype))


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


@pytest.mark.parametrize(
    "batch_size, hidden_size, dtype",
    [
        *[
            (batch_size, hidden_size, torch.float16)
            for batch_size in [1, 19, 99, 989]
            for hidden_size in [111, 500, 1024, 3072, 3584, 4096, 8192, 16384]
        ],
        (19, 1024, torch.bfloat16),
        (19, 1024, torch.float32),
        (2, 32768, torch.float16),
    ],
)
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

    torch.testing.assert_close(y_ref, y, **norm_tolerances(dtype))


@pytest.mark.parametrize(
    "batch_size, hidden_size, dtype",
    [
        *[
            (batch_size, hidden_size, torch.float16)
            for batch_size in [1, 19, 99, 989]
            for hidden_size in [111, 500, 1024, 3072, 3584, 4096, 8192, 16384]
        ],
        (19, 1024, torch.bfloat16),
        (19, 1024, torch.float32),
        (2, 32768, torch.float16),
    ],
)
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

    torch.testing.assert_close(x_fused, x_native, **norm_tolerances(dtype))
    torch.testing.assert_close(
        residual_fused, residual_native, **norm_tolerances(dtype)
    )


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


def test_gemma_norm_3d_non_flattenable_row_strides():
    x = _make_non_flattenable_3d(7, 4, 128, torch.float16)
    w = torch.randn(x.size(-1), device=device, dtype=x.dtype)

    assert x.size(1) > 1
    assert x.stride(1) != 0
    assert x.stride(0) != x.size(1) * x.stride(1)

    y = torch.empty_strided(x.shape, x.stride(), device=device, dtype=x.dtype)
    y_ref = gemma_rms_norm(x.clone(), w)
    sgl_kernel.gemma_rmsnorm(x, w, out=y)

    torch.testing.assert_close(y_ref, y, **norm_tolerances(x.dtype))


def test_gemma_norm_3d_non_flattenable_unaligned_row_strides():
    full = torch.randn(7, 8, 130, device=device, dtype=torch.float16)
    x = full[:, :4, :128]
    w = torch.randn(x.size(-1), device=device, dtype=x.dtype)

    assert x.stride(0) != x.size(1) * x.stride(1)
    assert x.size(-1) % 8 == 0
    assert x.stride(1) % 8 != 0

    y = torch.empty_strided(x.shape, x.stride(), device=device, dtype=x.dtype)
    y_ref = gemma_rms_norm(x.clone(), w)
    sgl_kernel.gemma_rmsnorm(x, w, out=y)

    torch.testing.assert_close(y_ref, y, **norm_tolerances(x.dtype))


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


###############################################################################
# Mixed input/weight dtype tests
###############################################################################


@pytest.mark.parametrize("batch_size", [1, 19])
@pytest.mark.parametrize("hidden_size", [512, 1024, 4096])
@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ],
)
def test_norm_mixed_dtype(batch_size, hidden_size, input_dtype, weight_dtype):
    x = torch.randn(batch_size, hidden_size, device=device, dtype=input_dtype)
    w = torch.randn(hidden_size, device=device, dtype=weight_dtype)

    y_ref = llama_rms_norm(x, w)
    y = sgl_kernel.rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19])
@pytest.mark.parametrize("hidden_size", [512, 1024, 4096])
@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ],
)
def test_fused_add_rmsnorm_mixed_dtype(
    batch_size, hidden_size, input_dtype, weight_dtype
):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=input_dtype, device=device)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=weight_dtype, device=device)

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    sgl_kernel.fused_add_rmsnorm(x_fused, residual_fused, weight, eps)

    tol = 1e-2 if input_dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(x_fused, x_native, rtol=tol, atol=tol)
    torch.testing.assert_close(residual_fused, residual_native, rtol=tol, atol=tol)


@pytest.mark.parametrize("batch_size", [1, 19])
@pytest.mark.parametrize("hidden_size", [512, 1024, 4096])
@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ],
)
def test_gemma_norm_mixed_dtype(batch_size, hidden_size, input_dtype, weight_dtype):
    x = torch.randn(batch_size, hidden_size, device=device, dtype=input_dtype)
    w = torch.randn(hidden_size, device=device, dtype=weight_dtype)

    y_ref = gemma_rms_norm(x, w)
    y = sgl_kernel.gemma_rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19])
@pytest.mark.parametrize("hidden_size", [512, 1024, 4096])
@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ],
)
def test_gemma_fused_add_rmsnorm_mixed_dtype(
    batch_size, hidden_size, input_dtype, weight_dtype
):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=input_dtype, device=device)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=weight_dtype, device=device)

    x_native, residual_native = gemma_fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    sgl_kernel.gemma_fused_add_rmsnorm(x_fused, residual_fused, weight, eps)

    tol = 1e-2 if input_dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(x_fused, x_native, rtol=tol, atol=tol)
    torch.testing.assert_close(residual_fused, residual_native, rtol=tol, atol=tol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
