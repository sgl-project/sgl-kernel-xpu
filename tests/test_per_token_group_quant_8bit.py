import itertools
import sys
from typing import List, Optional, Tuple

import pytest
import sgl_kernel
import torch
import triton
import triton.language as tl

try:
    HAS_XPU = torch.xpu.is_available()
except (ImportError, AttributeError):
    HAS_XPU = False


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round scale to nearest power of 2 for UE8M0 format."""
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs().clamp(min=1e-10))))


# Pure PyTorch reference implementations
def per_token_group_quant_fp8_ref(
    x: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-10,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for per-token group FP8 quantization."""
    assert x.dim() == 2 and x.size(1) % group_size == 0
    num_tokens, hidden_dim = x.shape
    x_view = x.view(num_tokens, -1, group_size)
    x_amax = x_view.abs().float().amax(dim=2).clamp(min=eps)
    scales = x_amax / 448.0  # FP8 E4M3 max

    if scale_ue8m0:
        scales = ceil_to_ue8m0(scales)

    x_quantized = (x_view / scales.unsqueeze(2)).to(torch.float8_e4m3fn)
    return x_quantized.view(num_tokens, hidden_dim), scales


def per_token_group_quant_int8_ref(
    x: torch.Tensor, group_size: int = 128, eps: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for per-token group INT8 quantization."""
    assert x.dim() == 2 and x.size(1) % group_size == 0
    num_tokens, hidden_dim = x.shape
    x_view = x.view(num_tokens, -1, group_size)
    x_amax = x_view.abs().float().amax(dim=2).clamp(min=eps)
    scales = x_amax / 127.0  # INT8 max
    x_scaled = (x_view / scales.unsqueeze(2)).clamp(-127.0, 127.0)
    x_quantized = x_scaled.to(torch.int8)
    return x_quantized.view(num_tokens, hidden_dim), scales


def sglapi_per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
    enable_v2: Optional[bool] = None,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit.default(
        x, x_q, x_s, group_size, eps, int8_min, int8_max, scale_ue8m0=False
    )
    return x_q, x_s


def create_fp8_output_scale(
    x_shape,
    device,
    group_size,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
):
    if scale_ue8m0:
        assert column_major_scales and scale_tma_aligned
        *x_batch, x_q_mn, x_q_k = x_shape
        x_s_mn, x_s_k = x_q_mn, x_q_k // 128
        aligned_mn = ceil_align(x_s_mn, 4)
        aligned_k = ceil_align(x_s_k, 4)
        return torch.empty(
            (*x_batch, aligned_k // 4, aligned_mn),
            device=device,
            dtype=torch.int,
        ).transpose(-1, -2)[..., :x_s_mn, :]
    elif column_major_scales:
        if scale_tma_aligned:
            # TODO extract "align" function
            # aligned to 4 * sizeof(float)
            aligned_size = (x_shape[-2] + 3) // 4 * 4
            return torch.empty(
                x_shape[:-2] + (x_shape[-1] // group_size, aligned_size),
                device=device,
                dtype=torch.float32,
            ).transpose(-1, -2)[: x_shape[-2], :]
        else:
            return torch.empty(
                (x_shape[-1] // group_size,) + x_shape[:-1],
                device=device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        return torch.empty(
            x_shape[:-1] + (x_shape[-1] // group_size,),
            device=device,
            dtype=torch.float32,
        )


def sglapi_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
    dtype: torch.dtype = torch.float8_e4m3fn,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))

    x_q = torch.empty(out_shape, device=x.device, dtype=dtype)
    x_s = create_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )

    fp8_max = torch.finfo(dtype).max
    fp8_min = -fp8_max
    if x.shape[0] > 0:
        torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit.default(
            x, x_q, x_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
        )
    return x_q, x_s


def sglapi_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
):

    if dst_dtype == torch.int8:
        assert not column_major_scales
        assert not scale_tma_aligned
        assert not fuse_silu_and_mul
        assert masked_m is None
        return sglapi_per_token_group_quant_int8(
            x=x,
            group_size=group_size,
            eps=eps,
            dtype=dst_dtype,
            enable_v2=enable_v2,
        )

    return sglapi_per_token_group_quant_fp8(
        x=x,
        group_size=group_size,
        eps=eps,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
        enable_v2=enable_v2,
        dtype=dst_dtype,
    )


@pytest.mark.skipif(not HAS_XPU, reason="XPU not available")
class TestPerTokenGroupQuantXPU:
    @pytest.fixture(autouse=True)
    def setup(self):
        torch.xpu.set_device(0)
        self.device = torch.device("xpu")
        self.eps = 1e-10

    def _test_against_reference(
        self,
        num_tokens: int,
        hidden_dim: int,
        group_size: int,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype = torch.bfloat16,
        column_major_scales: bool = False,
        scale_ue8m0: bool = False,
        seed: int = 42,
    ):
        """Test XPU implementation against PyTorch reference."""
        torch.manual_seed(seed)
        x_cpu = torch.randn(num_tokens, hidden_dim, dtype=src_dtype)
        x_xpu = x_cpu.to(self.device)

        # Get reference output
        if dst_dtype == torch.float8_e4m3fn:
            x_q_ref, scales_ref = per_token_group_quant_fp8_ref(
                x_cpu, group_size, self.eps, scale_ue8m0
            )
        else:
            x_q_ref, scales_ref = per_token_group_quant_int8_ref(
                x_cpu, group_size, self.eps
            )

        # Run XPU implementation
        x_q_xpu, scales_xpu = sglapi_per_token_group_quant_8bit(
            x=x_xpu,
            masked_m=None,
            group_size=group_size,
            eps=self.eps,
            dst_dtype=dst_dtype,
            column_major_scales=column_major_scales,
            scale_ue8m0=scale_ue8m0,
            enable_v2=False,
        )

        # Compare
        x_q_xpu_cpu = x_q_xpu.cpu()
        scales_xpu_cpu = scales_xpu.cpu()

        assert x_q_xpu_cpu.shape == x_q_ref.shape
        assert x_q_xpu_cpu.dtype == dst_dtype

        torch.testing.assert_close(
            scales_xpu_cpu, scales_ref, rtol=1e-3, atol=1e-5, msg=f"Scales mismatch"
        )

        # Dequantize and compare
        num_groups = hidden_dim // group_size
        x_dq_ref = (
            x_q_ref.view(num_tokens, num_groups, group_size).to(torch.float32)
            * scales_ref.unsqueeze(2)
        ).view(num_tokens, hidden_dim)
        x_dq_xpu = (
            x_q_xpu_cpu.view(num_tokens, num_groups, group_size).to(torch.float32)
            * scales_xpu_cpu.unsqueeze(2)
        ).view(num_tokens, hidden_dim)

        rtol, atol = (1e-1, 1e-1) if dst_dtype == torch.float8_e4m3fn else (1e-2, 1e-2)
        torch.testing.assert_close(x_dq_xpu, x_dq_ref, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "num_tokens,hidden_dim,group_size,dst_dtype,src_dtype,column_major_scales,scale_ue8m0",
        [
            # Basic FP8 quantization
            (128, 1024, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            # Basic INT8 quantization
            (128, 1024, 128, torch.int8, torch.bfloat16, False, False),
            # Various sizes
            (64, 512, 64, torch.float8_e4m3fn, torch.bfloat16, False, False),
            (128, 2048, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            (256, 4096, 64, torch.int8, torch.bfloat16, False, False),
            # Column-major scale layout
            (128, 1024, 128, torch.float8_e4m3fn, torch.bfloat16, True, False),
            # Float16 source dtype tests
            (128, 1024, 128, torch.float8_e4m3fn, torch.float16, False, False),
            (128, 1024, 128, torch.int8, torch.float16, False, False),
            (64, 512, 64, torch.float8_e4m3fn, torch.float16, False, False),
            # Float32 source dtype tests
            (128, 1024, 128, torch.float8_e4m3fn, torch.float32, False, False),
            (128, 1024, 128, torch.int8, torch.float32, False, False),
            (64, 512, 64, torch.float8_e4m3fn, torch.float32, False, False),
            # Prefill shapes
            (5120, 7168, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            (5120, 1536, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            (5120, 512, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            (5120, 16384, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            (5120, 18432, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            (5120, 2048, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            (40960, 2048, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            # Decode shapes (batch size 5)
            (5, 7168, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            (5, 1536, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            (5, 16384, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            (5, 18432, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            (5, 2048, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
            (40, 2048, 128, torch.float8_e4m3fn, torch.bfloat16, False, False),
        ],
    )
    def test_quantization(
        self,
        num_tokens,
        hidden_dim,
        group_size,
        dst_dtype,
        src_dtype,
        column_major_scales,
        scale_ue8m0,
    ):
        """Test per-token group quantization with various configurations.
        NOTE: This test doesn't enable any ue8m0 scales because it fails a check in fp8_kernel.py for
        scale_tma_aligned = True"""
        self._test_against_reference(
            num_tokens,
            hidden_dim,
            group_size,
            dst_dtype,
            src_dtype,
            column_major_scales,
            scale_ue8m0,
        )

    @pytest.mark.parametrize("scale_factor", [1e-3, 100.0])
    def test_edge_cases(self, scale_factor):
        torch.manual_seed(42)
        x = torch.randn(64, 512, dtype=torch.bfloat16) * scale_factor
        x_xpu = x.to(self.device)
        x_q, scales = sglapi_per_token_group_quant_8bit(
            x=x_xpu,
            masked_m=None,
            group_size=64,
            eps=self.eps,
            dst_dtype=torch.float8_e4m3fn,
            column_major_scales=False,
            scale_ue8m0=False,
            enable_v2=False,
        )
        assert x_q.shape == x.shape
        assert (scales > 0).all() and torch.isfinite(scales).all()


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * group_size
    y_q_ptr += g_id * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def triton_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tenosr with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    if dtype == torch.int8:
        finfo = torch.iinfo(dtype)
    else:
        finfo = torch.finfo(dtype)

    fp8_max = finfo.max
    fp8_min = -fp8_max

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    M = x.numel() // group_size
    N = group_size
    if column_major_scales:
        if scale_tma_aligned:
            # aligned to 4 * sizeof(float)
            aligned_size = (x.shape[-2] + 3) // 4 * 4
            x_s = torch.empty(
                x.shape[:-2] + (x.shape[-1] // group_size, aligned_size),
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)[: x.shape[-2], :]
        else:
            x_s = torch.empty(
                (x.shape[-1] // group_size,) + x.shape[:-1],
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        x_s = torch.empty(
            x.shape[:-1] + (x.shape[-1] // group_size,),
            device=x.device,
            dtype=torch.float32,
        )

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _per_token_group_quant_fp8[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            N,
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s


@pytest.mark.parametrize(
    "num_tokens, hidden_dim, group_size, dst_dtype, column_major_scales, scale_tma_aligned",
    [
        (128, 1024, 128, torch.float8_e4m3fn, False, False),
        (256, 4096, 64, torch.float8_e4m3fn, False, False),
        (64, 512, 64, torch.float8_e4m3fn, False, False),
        (128, 2048, 128, torch.float8_e4m3fn, False, False),
        (128, 1024, 128, torch.float8_e4m3fn, True, False),
        # Prefill shapes
        (5120, 7168, 128, torch.float8_e4m3fn, False, False),
        (5120, 1536, 128, torch.float8_e4m3fn, False, False),
        (5120, 512, 128, torch.float8_e4m3fn, False, False),
        (5120, 16384, 128, torch.float8_e4m3fn, False, False),
        (5120, 18432, 128, torch.float8_e4m3fn, False, False),
        (5120, 2048, 128, torch.float8_e4m3fn, False, False),
        (40960, 2048, 128, torch.float8_e4m3fn, False, False),
        # Decode shapes (batch size 5)
        (5, 7168, 128, torch.float8_e4m3fn, False, False),
        (5, 1536, 128, torch.float8_e4m3fn, False, False),
        (5, 16384, 128, torch.float8_e4m3fn, False, False),
        (5, 18432, 128, torch.float8_e4m3fn, False, False),
        (5, 2048, 128, torch.float8_e4m3fn, False, False),
        (40, 2048, 128, torch.float8_e4m3fn, False, False),
    ],
)
def test_per_token_group_quant_with_column_major_fp8(
    num_tokens,
    hidden_dim,
    group_size,
    dst_dtype,
    column_major_scales,
    scale_tma_aligned,
):
    if not column_major_scales and scale_tma_aligned:
        return

    x = torch.randn(num_tokens, hidden_dim, device="xpu", dtype=torch.bfloat16)

    x_q_triton, x_s_triton = triton_per_token_group_quant_8bit(
        x,
        group_size,
        eps=1e-10,
        dtype=dst_dtype,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
    )

    x_q_xpu, x_s_xpu = sglapi_per_token_group_quant_8bit(
        x,
        group_size,
        eps=1e-10,
        dst_dtype=dst_dtype,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
    )

    torch.testing.assert_close(
        x_s_triton.contiguous(), x_s_xpu.contiguous(), rtol=1e-3, atol=1e-5
    )
    # Dequantize and compare values (convert to float for comparison)
    x_dq_triton = (
        x_q_triton.view(num_tokens, -1, group_size).to(torch.float32)
        * x_s_triton.unsqueeze(2)
    ).view(num_tokens, hidden_dim)
    x_dq_xpu = (
        x_q_xpu.cpu().view(num_tokens, -1, group_size).to(torch.float32)
        * x_s_xpu.cpu().unsqueeze(2)
    ).view(num_tokens, hidden_dim)

    rtol, atol = (1e-1, 1e-1) if dst_dtype == torch.float8_e4m3fn else (1e-2, 1e-2)
    torch.testing.assert_close(x_dq_xpu.cpu(), x_dq_triton.cpu(), rtol=rtol, atol=atol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
