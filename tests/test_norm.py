# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/tests/test_norm.py

import gc
import sys
import time

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


@pytest.mark.parametrize(
    "batch_size, hidden_size, dtype",
    [
        *[
            (batch_size, hidden_size, dtype)
            for batch_size in [1, 19, 99, 989]
            for hidden_size in [111, 500, 1024, 2880, 3072, 3584, 4096, 8192, 16384]
            for dtype in [torch.float16, torch.bfloat16]
        ],
        # gpt-oss: bf16 input + bf16 weight, H = 2880, prefill batch.
        (4096, 2880, torch.bfloat16),
        # fp32 correctness smoke test for rmsnorm. `SYCL_DISPATCH_FLOATING_TYPES`
        # instantiates the kernel for Float, so fp32 is a distinct instantiation
        # that the fp16/bf16 sweep above never covers.
        (19, 1024, torch.float32),
    ],
)
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

    torch.testing.assert_close(y_ref, y, **norm_tolerances(dtype))


@pytest.mark.parametrize(
    "batch_size, hidden_size, dtype",
    [
        *[
            (batch_size, hidden_size, dtype)
            for batch_size in [1, 19, 99, 989]
            for hidden_size in [111, 500, 1024, 2880, 3072, 3584, 4096, 8192, 16384]
            for dtype in [torch.float16, torch.bfloat16, torch.float32]
        ],
        # gpt-oss: bf16 input + bf16 weight, H = 2880, prefill batch.
        (4096, 2880, torch.bfloat16),
    ],
)
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

    torch.testing.assert_close(x_fused, x_native, **norm_tolerances(dtype))
    torch.testing.assert_close(
        residual_fused, residual_native, **norm_tolerances(dtype)
    )


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


# Hidden sizes that reach the *non-cached* `update()` fallback.
#
# `dispatch_rmsnorm_no_rstd_iters` keeps the row in registers only while
#
#     iters = ceil(N / (workgroup_size * update_vec_size)) <= 8
#
# `workgroup_size` saturates at 512, and `update_vec_size` is halved until it
# divides N, so it collapses to 1 on odd N. That gives two ways past the bound:
#
#   * odd N > 4096            -> vec 1, iters > 8
#   * N % 8 == 0, N > 32768   -> vec 8 (bf16/fp16) or vec 4 (fp32), iters > 8
#
# Nothing else in this file reaches it: the largest swept hidden size is 16384,
# and the (2, 32768) gemma cases land on iters == 8 exactly -- still cached.
#
# The fallback matters because it is the only path where `AddNoRstdForward`
# re-reads the residual sum from `add_data`; `reduce_combine` writes the sum
# there and leaves `X_data` holding the un-summed x, so sourcing from `X_data`
# would be wrong here and invisible everywhere else.
NON_CACHED_SHAPES = [
    (2, 4097),  # vec 1, iters = 9   -- first shape past the bound
    (2, 6143),  # vec 1, iters = 12
    (2, 40960),  # vec 8 / vec 4, iters = 10 / 20 -- full-width vector fallback
]

# Controls: same code shape, but iters <= 8, so these stay on the cached path.
# They also cover the vec-1 cached instantiations (ITERS >= 2), which the
# power-of-two sweeps above can never reach.
CACHED_CONTROL_SHAPES = [
    (2, 4095),  # vec 1, iters = 8 -- immediately below the bound
    (2, 2881),  # vec 1, iters = 6 -- odd, production-shaped (gpt-oss 2880 + 1)
]


@pytest.mark.parametrize("batch_size, hidden_size", NON_CACHED_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gemma", [False, True])
def test_fused_add_rmsnorm_non_cached_fallback(batch_size, hidden_size, dtype, gemma):
    """Residual-add norm on shapes that bypass the register cache.

    Both entry points share `AddNoRstdForward`, so both are checked: a wrong
    residual source in the shared fallback would otherwise show up in only one.
    """
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    ref = gemma_fused_add_rms_norm if gemma else fused_add_rms_norm
    kernel = (
        sgl_kernel.gemma_fused_add_rmsnorm if gemma else sgl_kernel.fused_add_rmsnorm
    )

    x_native, residual_native = ref(x.clone(), residual.clone(), weight, eps)

    x_fused = x.clone()
    residual_fused = residual.clone()
    kernel(x_fused, residual_fused, weight, eps)

    torch.testing.assert_close(x_fused, x_native, **norm_tolerances(dtype))
    torch.testing.assert_close(
        residual_fused, residual_native, **norm_tolerances(dtype)
    )


@pytest.mark.parametrize("batch_size, hidden_size", CACHED_CONTROL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_add_rmsnorm_odd_hidden_cached(batch_size, hidden_size, dtype):
    """Control for the test above: odd N that still fits the register cache."""
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

    torch.testing.assert_close(x_fused, x_native, **norm_tolerances(dtype))
    torch.testing.assert_close(
        residual_fused, residual_native, **norm_tolerances(dtype)
    )


@pytest.mark.parametrize("hidden_size", [4097, 40960])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rmsnorm_non_cached_fallback(hidden_size, dtype):
    """Plain rmsnorm on the same fallback shapes (no residual source to get
    wrong, but it is a distinct `update()` instantiation)."""
    x = torch.randn(2, hidden_size).to(device).to(dtype)
    w = torch.randn(hidden_size).to(device).to(dtype)

    torch.testing.assert_close(
        sgl_kernel.rmsnorm(x, w), llama_rms_norm(x, w), **norm_tolerances(dtype)
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


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("seq_len", [1, 7])
@pytest.mark.parametrize("num_groups", [2, 4])
@pytest.mark.parametrize("hidden_size", [1024, 2048, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_4d(batch_size, seq_len, num_groups, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(
        batch_size, seq_len, num_groups, hidden_size, dtype=dtype, device=device
    )
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    sgl_kernel.fused_add_rmsnorm(x_fused, residual_fused, weight, eps)

    torch.testing.assert_close(x_fused, x_native, **norm_tolerances(dtype))
    torch.testing.assert_close(
        residual_fused, residual_native, **norm_tolerances(dtype)
    )


@pytest.mark.parametrize("batch_size", [1, 4, 19])
@pytest.mark.parametrize("seq_len", [1, 7, 32])
# hidden_size=1 exercises the "other dim == 1" shape (excluding the leading
# batch dim) that bypasses the flattenable fast-path check but still yields
# a correct result.
@pytest.mark.parametrize("hidden_size", [1, 111, 1024, 4096])
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
# 4D tensor tests for gemma_rmsnorm
###############################################################################


def _make_non_flattenable_4d(
    batch_size,
    num_tokens,
    num_heads,
    head_dim,
    dtype,
    break_d0_d1,
    extra_tokens=3,
    extra_heads=4,
):
    """Build a 4D tensor [batch, tokens, heads, head_dim] that is not flattenable:

      break_d0_d1=False  -> dim0 can be flattened
      break_d0_d1=True  -> fully independent; needs the full 3-level form

    break_d0_d1 allocates extra token rows and slices down to num_tokens, so
    stride(0) != size(1)*stride(1) (requires batch_size > 1, since dim0 of
    size 1 always folds regardless of stride).
    """
    if break_d0_d1:
        assert batch_size > 1, "batch_size must be > 1 to exercise an independent dim0"
        tokens_alloc = num_tokens + extra_tokens
    else:
        tokens_alloc = num_tokens

    full = torch.randn(
        batch_size,
        tokens_alloc,
        (num_heads + extra_heads) * head_dim,
        device=device,
        dtype=dtype,
    )
    x = full[:, :num_tokens, : num_heads * head_dim].unflatten(
        -1, (num_heads, head_dim)
    )

    if break_d0_d1:
        assert x.stride(0) != x.size(1) * x.stride(1)
    else:
        assert x.stride(0) == x.size(1) * x.stride(1)

    return x


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("num_tokens", [1, 7])
# num_heads=1 and head_dim=1 exercise the "other dim == 1" shapes (excluding
# the leading batch/token dims) that bypass the flattenable fast-path check
# but still yield a correct result.
@pytest.mark.parametrize("num_heads", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [1, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_gemma_norm_4d(batch_size, num_tokens, num_heads, head_dim, dtype, specify_out):
    x = torch.randn(
        batch_size, num_tokens, num_heads, head_dim, device=device, dtype=dtype
    )
    w = torch.randn(head_dim, device=device, dtype=dtype)

    y_ref = gemma_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        sgl_kernel.gemma_rmsnorm(x, w, out=y)
    else:
        y = sgl_kernel.gemma_rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, **norm_tolerances(dtype))


###############################################################################
# Non-contiguous 4D tensor tests (sliced last-dim, flattenable leading dims)
###############################################################################


def _make_non_contiguous_4d(
    batch_size, num_tokens, num_heads, head_dim, dtype, extra=64
):
    """Create a last-dim-non-contiguous 4D tensor [batch_size, tokens, heads, head_dim]
    by slicing a larger tensor along the last dimension."""
    full = torch.randn(
        batch_size, num_tokens, num_heads, head_dim + extra, device=device, dtype=dtype
    )
    view = full[..., :head_dim]
    assert view.stride(-1) == 1
    assert view.stride(-2) == head_dim + extra
    return view


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("num_tokens", [1, 7])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_gemma_norm_4d_non_contiguous(
    batch_size, num_tokens, num_heads, head_dim, dtype
):
    x_nc = _make_non_contiguous_4d(batch_size, num_tokens, num_heads, head_dim, dtype)
    w = torch.randn(head_dim, device=device, dtype=dtype)

    y_ref = gemma_rms_norm(x_nc.clone(), w)
    y = sgl_kernel.gemma_rmsnorm(x_nc, w)

    torch.testing.assert_close(y_ref, y, **norm_tolerances(dtype))


def test_gemma_norm_4d_non_flattenable_unaligned_row_strides():
    full = torch.randn(1, 7, 8, 130, device=device, dtype=torch.float16)
    x = full[:, :, :4, :128]
    w = torch.randn(x.size(-1), device=device, dtype=x.dtype)

    assert x.stride(-3) != x.size(-2) * x.stride(-2)
    assert x.size(-1) % 8 == 0
    assert x.stride(-2) % 8 != 0

    y = torch.empty_strided(x.shape, x.stride(), device=device, dtype=x.dtype)
    y_ref = gemma_rms_norm(x.clone(), w)
    sgl_kernel.gemma_rmsnorm(x, w, out=y)

    torch.testing.assert_close(y_ref, y, **norm_tolerances(x.dtype))


@pytest.mark.parametrize("break_d0_d1", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("num_tokens", [7, 32])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_gemma_norm_4d_non_flattenable(
    break_d0_d1,
    batch_size,
    num_tokens,
    num_heads,
    head_dim,
    dtype,
    specify_out,
):
    if break_d0_d1 and batch_size == 1:
        pytest.skip("break_d0_d1 requires batch_size > 1")

    x = _make_non_flattenable_4d(
        batch_size,
        num_tokens,
        num_heads,
        head_dim,
        dtype,
        break_d0_d1=break_d0_d1,
    )
    w = torch.randn(head_dim, device=device, dtype=dtype)

    y_ref = gemma_rms_norm(x.clone(), w)
    if specify_out:
        y = torch.empty_strided(x.shape, x.stride(), device=device, dtype=x.dtype)
        sgl_kernel.gemma_rmsnorm(x, w, out=y)
    else:
        y = sgl_kernel.gemma_rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, **norm_tolerances(dtype))


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

    tol = 1e-2 if input_dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(y_ref, y, rtol=tol, atol=tol)


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

    tol = 1e-2 if input_dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(y_ref, y, rtol=tol, atol=tol)


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


POST_EPS = 1e-8
PRE_EPS = 1e-5


def is_residual_norm_available_xe35():
    if not torch.xpu.is_available():
        return False
    from sgl_kernel.utils import is_xe3_arch

    return is_xe3_arch() and hasattr(
        torch.ops.sgl_kernel, "fused_post_norm_residual_pre_norm_xe35"
    )


def round_bfloat16_xe35(values: torch.Tensor) -> torch.Tensor:
    """Round an independent FP64 oracle directly to BF16, without an FP32 double-round."""
    high = values.float()
    midpoint = ((high.view(torch.int32) & 0xFFFF) == 0x8000) & (values != high.double())
    if midpoint.any():
        direction = torch.where(
            values[midpoint] > high[midpoint].double(), float("inf"), float("-inf")
        ).float()
        high[midpoint] = torch.nextafter(high[midpoint], direction)
    return high.bfloat16()


def reference_xe35(
    branch: torch.Tensor,
    stream: torch.Tensor,
    post_weight: torch.Tensor,
    pre_weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    post_norm_input_scale: bool,
    round_pre_norm_input: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent FP64 mathematics with the model's BF16 boundaries and FP32 stream store."""
    post_input = branch.double()
    if post_weight.numel() and post_norm_input_scale:
        post_input = round_bfloat16_xe35(post_input * post_weight.double()).double()
    post_eps = torch.tensor(POST_EPS, dtype=torch.float32, device="cpu").double()
    pre_eps = torch.tensor(PRE_EPS, dtype=torch.float32, device="cpu").double()
    post_rstd = torch.rsqrt(post_input.square().mean(dim=-1, keepdim=True) + post_eps)
    normed_branch = post_input * post_rstd
    if post_weight.numel() and not post_norm_input_scale:
        normed_branch = normed_branch * post_weight.double()
    normed_branch = round_bfloat16_xe35(normed_branch)
    rounded_post_norm = normed_branch.float()
    new_stream = (
        alpha.double() * stream.double() + beta.double() * rounded_post_norm.double()
    ).float()
    pre_norm_input = (
        new_stream.bfloat16().double() if round_pre_norm_input else new_stream.double()
    )
    pre_rstd = torch.rsqrt(pre_norm_input.square().mean(dim=-1, keepdim=True) + pre_eps)
    output = (pre_norm_input * pre_rstd * pre_weight.double()).float()
    return output, new_stream, normed_branch


def run_op_xe35(
    branch: torch.Tensor,
    stream: torch.Tensor,
    post_weight: torch.Tensor,
    pre_weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    post_norm_input_scale: bool = False,
    round_pre_norm_input: bool = False,
    output_fp32: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(branch)
    new_stream = torch.empty_like(stream)
    sgl_kernel.fused_post_norm_residual_pre_norm_xe35(
        branch,
        stream,
        post_weight,
        pre_weight,
        alpha,
        beta,
        output,
        new_stream,
        POST_EPS,
        PRE_EPS,
        post_norm_input_scale,
        round_pre_norm_input,
        output_fp32,
    )
    return output, new_stream


@pytest.mark.skipif(
    not is_residual_norm_available_xe35(),
    reason="Xe35 compute capability 3.5 and dedicated op required",
)
def test_round_bfloat16_reference_midpoints_xe35():
    midpoint = 1.30859375
    values = torch.tensor(
        [midpoint - 2**-40, midpoint, midpoint + 2**-40], dtype=torch.float64
    )
    expected = torch.tensor([1.3046875, 1.3125, 1.3125], dtype=torch.bfloat16)
    torch.testing.assert_close(round_bfloat16_xe35(values), expected, rtol=0, atol=0)
    torch.testing.assert_close(round_bfloat16_xe35(-values), -expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not is_residual_norm_available_xe35(),
    reason="Xe35 compute capability 3.5 and dedicated op required",
)
def test_fused_norm_residual_fp32_fma_boundary_xe35():
    device = torch.device("xpu")
    branch = -torch.ones(1, 8, dtype=torch.bfloat16, device=device)
    stream = torch.full((1, 8), 1 - 2**-23, dtype=torch.float32, device=device)
    alpha = torch.full((8,), 1 + 2**-23, dtype=torch.float32, device=device)
    beta = torch.ones_like(alpha)
    post = torch.empty(0, dtype=torch.bfloat16, device=device)
    pre = torch.ones(8, dtype=torch.bfloat16, device=device)
    _, updated = run_op_xe35(branch, stream, post, pre, alpha, beta)
    torch.testing.assert_close(
        updated.cpu(), torch.full((1, 8), -(2**-46)), rtol=0, atol=0
    )


@pytest.mark.parametrize("rows", [1, 37, 257])
@pytest.mark.parametrize("hidden_size", [4096, 8192, 16384])
@pytest.mark.parametrize("post_mode", ["none", "weight", "input_scale"])
@pytest.mark.parametrize("round_pre_norm_input", [False, True])
@pytest.mark.skipif(
    not is_residual_norm_available_xe35(),
    reason="Xe35 compute capability 3.5 and dedicated op required",
)
def test_fused_norm_residual_xe35(
    rows: int, hidden_size: int, post_mode: str, round_pre_norm_input: bool
) -> None:
    torch.manual_seed(1234 + rows)
    device = torch.device("xpu")
    branch = torch.randn(rows, hidden_size, device=device, dtype=torch.bfloat16)
    stream = torch.randn(rows, hidden_size, device=device, dtype=torch.float32)
    post_weight = (
        torch.empty(0, device=device, dtype=torch.bfloat16)
        if post_mode == "none"
        else torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    )
    pre_weight = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    alpha = torch.rand(hidden_size, device=device, dtype=torch.float32)
    beta = torch.rand(hidden_size, device=device, dtype=torch.float32)

    post_weight_cpu, pre_weight_cpu = post_weight.cpu(), pre_weight.cpu()
    alpha_cpu, beta_cpu = alpha.cpu(), beta.cpu()
    actual_output, actual_stream = run_op_xe35(
        branch,
        stream,
        post_weight,
        pre_weight,
        alpha,
        beta,
        post_mode == "input_scale",
        round_pre_norm_input,
    )

    for start in range(0, rows, 512):
        expected_output, expected_stream, _ = reference_xe35(
            branch[start : start + 512].cpu(),
            stream[start : start + 512].cpu(),
            post_weight_cpu,
            pre_weight_cpu,
            alpha_cpu,
            beta_cpu,
            post_mode == "input_scale",
            round_pre_norm_input,
        )
        torch.testing.assert_close(
            actual_output[start : start + 512].cpu().float(),
            expected_output,
            rtol=5e-3,
            atol=5e-3,
        )
        torch.testing.assert_close(
            actual_stream[start : start + 512].cpu(),
            expected_stream,
            rtol=1e-5,
            atol=1e-5,
        )

    output_fp32 = torch.empty_like(stream)
    widened_output, widened_stream = run_op_xe35(
        branch,
        stream,
        post_weight,
        pre_weight,
        alpha,
        beta,
        post_mode == "input_scale",
        round_pre_norm_input,
        output_fp32,
    )
    for start in range(0, rows, 512):
        torch.testing.assert_close(
            widened_output[start : start + 512].cpu(),
            actual_output[start : start + 512].cpu(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            widened_stream[start : start + 512].cpu(),
            actual_stream[start : start + 512].cpu(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output_fp32[start : start + 512].cpu(),
            actual_output[start : start + 512].cpu().float(),
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(
    not is_residual_norm_available_xe35(),
    reason="Xe35 compute capability 3.5 and dedicated op required",
)
def test_fused_norm_residual_hidden_tail_rows_boundary_xe35() -> None:
    test_fused_norm_residual_xe35(385, 8200, "none", True)


@pytest.mark.parametrize("hidden_size", [8, 32768])
@pytest.mark.skipif(
    not is_residual_norm_available_xe35(),
    reason="Xe35 compute capability 3.5 and dedicated op required",
)
def test_fused_norm_residual_hidden_limits_xe35(hidden_size):
    test_fused_norm_residual_xe35(1, hidden_size, "weight", False)


@pytest.mark.skipif(
    not is_residual_norm_available_xe35(),
    reason="Xe35 compute capability 3.5 and dedicated op required",
)
def test_fused_norm_residual_empty_rows_xe35():
    branch = torch.empty(0, 8, device="xpu", dtype=torch.bfloat16)
    stream = torch.empty(0, 8, device="xpu", dtype=torch.float32)
    post = torch.empty(0, device="xpu", dtype=torch.bfloat16)
    pre = torch.ones(8, device="xpu", dtype=torch.bfloat16)
    alpha = torch.ones(8, device="xpu", dtype=torch.float32)
    output, updated = run_op_xe35(branch, stream, post, pre, alpha, alpha)
    assert output.shape == updated.shape == (0, 8)


@pytest.mark.skipif(
    not is_residual_norm_available_xe35(),
    reason="Xe35 compute capability 3.5 and dedicated op required",
)
def test_fused_norm_residual_reject_fp16_xe35():
    branch = torch.ones(1, 8, device="xpu", dtype=torch.float16)
    stream = torch.ones(1, 8, device="xpu", dtype=torch.float32)
    post = torch.empty(0, device="xpu", dtype=torch.bfloat16)
    pre = torch.ones(8, device="xpu", dtype=torch.bfloat16)
    alpha = torch.ones(8, device="xpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="branch must be bfloat16"):
        run_op_xe35(branch, stream, post, pre, alpha, alpha)


@pytest.mark.parametrize("rows", [8, 32768])
@pytest.mark.parametrize("post_mode", ["none", "weight", "input_scale"])
@pytest.mark.parametrize("round_pre_norm_input", [False, True])
@pytest.mark.skipif(
    not is_residual_norm_available_xe35(),
    reason="Xe35 compute capability 3.5 and dedicated op required",
)
def test_fused_norm_residual_production_shapes_xe35(rows, post_mode, round_pre_norm_input):
    test_fused_norm_residual_xe35(rows, 8192, post_mode, round_pre_norm_input)


@pytest.fixture(autouse=True)
def cool_residual_norm_case_xe35(request):
    yield
    if request.node.function.__name__.endswith("_xe35"):
        torch.xpu.synchronize()
        gc.collect()
        torch.xpu.empty_cache()
        time.sleep(2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
