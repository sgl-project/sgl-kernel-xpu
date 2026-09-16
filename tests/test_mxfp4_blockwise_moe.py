# SPDX-License-Identifier: Apache-2.0
"""
Tests for MXFP4 (E2M1) Block-Scaled Grouped GEMM for MoE on Intel XPU

MXFP4 follows the OpenCompute MX (Microscaling) format specification:
- Data type: E2M1 (4-bit float with 2-bit exponent, 1-bit mantissa)
- Block size: 32 elements per scale factor
- Scale format: UE8M0 (unsigned 8-bit exponent-only, no mantissa)

Matrix Layout Requirements:
- Matrix A: (M, K) RowMajor, quantized along K, scales (M, K//32)
- Matrix B: (N, K) ColumnMajor (CUTLASS convention), quantized along K, scales (N, K//32)

Usage:
    pytest test_mxfp4_blockwise_moe.py -v
"""

import pytest
import torch
from utils import get_device

MXFP4_BLOCK_SIZE = 32
FLOAT4_E2M1_MAX = 6.0

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

MNK_FACTORS = [
    (64, 64, 64),
    (64, 128, 128),
    (128, 256, 256),
    (256, 512, 512),
    (512, 512, 512),
]


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_cri_device() -> bool:
    """Check whether the current XPU device is CRI (Xe3P)."""
    if not is_xpu_available():
        return False
    try:
        from sgl_kernel import is_xe3_arch

        return is_xe3_arch()
    except ImportError:
        return False


def skip_if_no_xpu():
    if not is_xpu_available():
        pytest.skip("Intel XPU not available")


def skip_if_kernel_unavailable():
    try:
        from sgl_kernel import mxfp4_blockwise_scaled_grouped_mm
    except ImportError:
        pytest.skip("mxfp4_blockwise_scaled_grouped_mm kernel not available")


def quantize_to_e2m1(tensor: torch.Tensor) -> torch.Tensor:
    e2m1_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    sign = (tensor < 0).to(torch.uint8)
    abs_val = torch.clamp(tensor.abs(), max=6.0)
    abs_val_expanded = abs_val.unsqueeze(-1)
    e2m1_expanded = e2m1_values.view(*([1] * abs_val.dim()), -1)
    distances = (abs_val_expanded - e2m1_expanded).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)
    quantized = (sign << 3) | indices
    return quantized


def pack_fp4(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape[-1] % 2 == 0
    shape = tensor.shape[:-1] + (tensor.shape[-1] // 2, 2)
    paired = tensor.reshape(shape)
    packed = (paired[..., 0] & 0x0F) | ((paired[..., 1] & 0x0F) << 4)
    return packed.to(torch.uint8)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    return unpacked


def dequantize_e2m1(
    quantized: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    sign = ((quantized >> 3) & 1).to(torch.bool)
    magnitude_idx = (quantized & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=quantized.device)
    magnitude = kE2M1[magnitude_idx]
    result = torch.where(sign, -magnitude, magnitude)
    return result.to(dtype)


def quantize_to_mxfp4(
    tensor: torch.Tensor, block_size: int = MXFP4_BLOCK_SIZE
) -> tuple:
    assert tensor.dim() == 2
    m, k = tensor.shape
    assert k % block_size == 0
    assert k % 2 == 0

    tensor_fp32 = tensor.float()
    num_blocks = k // block_size
    tensor_blocks = tensor_fp32.reshape(m, num_blocks, block_size)

    block_max = tensor_blocks.abs().max(dim=-1, keepdim=True).values
    block_max = torch.clamp(block_max, min=1e-12)

    log2_max = torch.log2(block_max / FLOAT4_E2M1_MAX)
    exponent = torch.ceil(log2_max).clamp(min=-127, max=127).to(torch.int32)
    scales_ue8m0 = (exponent + 127).to(torch.uint8).squeeze(-1)

    scale_values = torch.pow(2.0, exponent.float())
    scaled_tensor = tensor_blocks / scale_values
    quantized_blocks = quantize_to_e2m1(scaled_tensor)
    quantized = quantized_blocks.reshape(m, k)
    packed = pack_fp4(quantized)

    return packed, scales_ue8m0


def dequantize_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> torch.Tensor:
    m, packed_k = packed.shape
    k = packed_k * 2

    unpacked = unpack_fp4(packed)
    dequantized = dequantize_e2m1(unpacked, dtype)

    num_blocks = k // block_size
    dequantized_blocks = dequantized.reshape(m, num_blocks, block_size)

    scale_exp = scales.to(torch.int32) - 127
    scale_values = torch.pow(2.0, scale_exp.float()).unsqueeze(-1)
    scaled = dequantized_blocks * scale_values

    return scaled.reshape(m, k).to(dtype)


def reference_grouped_gemm(
    a_list: list,
    b_list: list,
    scales_a_list: list,
    scales_b_list: list,
    target_device: str = "cpu",
) -> list:
    outputs = []
    for a_packed, b_packed, scales_a, scales_b in zip(
        a_list, b_list, scales_a_list, scales_b_list
    ):
        a_packed_cpu = a_packed.cpu()
        b_packed_cpu = b_packed.cpu()
        scales_a_cpu = scales_a.cpu()
        scales_b_cpu = scales_b.cpu()

        a_dq = dequantize_mxfp4(a_packed_cpu, scales_a_cpu, torch.float32)
        b_dq_nk = dequantize_mxfp4(b_packed_cpu, scales_b_cpu, torch.float32)
        b_dq = b_dq_nk.t()

        out = torch.matmul(a_dq, b_dq)
        outputs.append(out.to(target_device))

    return outputs


def create_random_mxfp4_data(m: int, k: int, device: str, seed: int = 42):
    torch.manual_seed(seed)
    original = torch.randn(m, k, dtype=torch.float32, device=device) * 2.0
    packed, scales = quantize_to_mxfp4(original)
    packed = packed.to(device)
    scales = scales.to(device)
    return packed, scales, original


def ensure_contiguous_layout(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def prepare_kernel_inputs(
    a_list: list,
    b_list: list,
    scales_a_list: list,
    scales_b_list: list,
    device: str,
):
    """Flat-2D inputs + empty-ptr sentinels for on-device prep. Uniform m."""
    num_experts = len(a_list)
    m, packed_k = a_list[0].shape
    k = packed_k * 2
    n_b, packed_k_b = b_list[0].shape
    assert k == packed_k_b * 2
    n = n_b
    total_m = num_experts * m

    a_flat = torch.cat(
        [ensure_contiguous_layout(a) for a in a_list], dim=0
    ).contiguous()
    sa_flat = torch.cat(
        [ensure_contiguous_layout(s) for s in scales_a_list], dim=0
    ).contiguous()
    assert a_flat.shape == (total_m, packed_k)
    assert sa_flat.shape == (total_m, k // MXFP4_BLOCK_SIZE)

    b_stack = torch.stack(
        [ensure_contiguous_layout(b) for b in b_list], dim=0
    ).contiguous()
    sb_stack = torch.stack(
        [ensure_contiguous_layout(s) for s in scales_b_list], dim=0
    ).contiguous()

    output = torch.zeros((total_m, n), dtype=torch.float32, device=device)
    expert_offsets = torch.arange(0, total_m, m, dtype=torch.int32, device=device)
    empty_ptrs = torch.empty((0,), dtype=torch.int64, device=device)

    return {
        "output": output,
        "a_ptrs": empty_ptrs,
        "b_ptrs": empty_ptrs,
        "out_ptrs": empty_ptrs,
        "a_scales_ptrs": empty_ptrs,
        "b_scales_ptrs": empty_ptrs,
        "a_stack": a_flat,
        "b_stack": b_stack,
        "scales_a_stack": sa_flat,
        "scales_b_stack": sb_stack,
        "problem_sizes": torch.tensor(
            [[m, n, k]] * num_experts, dtype=torch.int32, device=device
        ),
        "expert_offsets": expert_offsets,
        # 64 MiB sized for these tests; 1 GiB legacy can fragment device pool.
        "workspace": torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=device),
        "_per_expert_m": m,
        "_num_experts": num_experts,
    }


def _expert_slice(inputs, i):
    """Slice expert i's (m, n) output from flat (sum_m_i, n) buffer."""
    m = inputs["_per_expert_m"]
    return inputs["output"][i * m : (i + 1) * m]


@pytest.mark.skipif(not is_xpu_available(), reason="Intel XPU not available")
@pytest.mark.skipif(
    is_xpu_available() and not is_cri_device(),
    reason="MXFP4 blockwise scaled grouped GEMM requires a CRI (Xe3P) device",
)
class TestMXFP4BlockwiseScaledGroupedMM:
    """Tests for the MXFP4 MoE CUTLASS kernel on Intel XPU."""

    @pytest.fixture(autouse=True)
    def check_kernel_available(self):
        skip_if_no_xpu()
        if not is_cri_device():
            pytest.skip(
                "MXFP4 blockwise scaled grouped GEMM requires a CRI (Xe3P) device"
            )

    @pytest.mark.parametrize("m,n,k", MNK_FACTORS)
    @pytest.mark.parametrize("num_experts", [2, 4, 8])
    @torch.inference_mode()
    def test_kernel_vs_reference(self, m: int, n: int, k: int, num_experts: int):
        from sgl_kernel import mxfp4_blockwise_scaled_grouped_mm

        device = "xpu"

        a_list = []
        b_list = []
        scales_a_list = []
        scales_b_list = []

        for i in range(num_experts):
            a_packed, scales_a, _ = create_random_mxfp4_data(m, k, "cpu", seed=42 + i)
            b_packed, scales_b, _ = create_random_mxfp4_data(n, k, "cpu", seed=100 + i)

            a_list.append(ensure_contiguous_layout(a_packed))
            b_list.append(ensure_contiguous_layout(b_packed))
            scales_a_list.append(ensure_contiguous_layout(scales_a))
            scales_b_list.append(ensure_contiguous_layout(scales_b))

        ref_outputs = reference_grouped_gemm(
            a_list, b_list, scales_a_list, scales_b_list, target_device="cpu"
        )

        a_list = [x.to(device) for x in a_list]
        b_list = [x.to(device) for x in b_list]
        scales_a_list = [x.to(device) for x in scales_a_list]
        scales_b_list = [x.to(device) for x in scales_b_list]

        inputs = prepare_kernel_inputs(
            a_list, b_list, scales_a_list, scales_b_list, device
        )

        mxfp4_blockwise_scaled_grouped_mm(
            inputs["output"],
            inputs["a_ptrs"],
            inputs["b_ptrs"],
            inputs["out_ptrs"],
            inputs["a_scales_ptrs"],
            inputs["b_scales_ptrs"],
            inputs["a_stack"],
            inputs["b_stack"],
            inputs["scales_a_stack"],
            inputs["scales_b_stack"],
            inputs["problem_sizes"],
            inputs["expert_offsets"],
            inputs["workspace"],
        )

        for i in range(num_experts):
            kernel_out = _expert_slice(inputs, i).to("cpu")
            ref_out = ref_outputs[i].to("cpu")

            assert not torch.isnan(kernel_out).any()
            assert not torch.isinf(kernel_out).any()

            torch.testing.assert_close(kernel_out, ref_out, atol=1e-1, rtol=1e-1)

            kernel_magnitude = kernel_out.abs().mean()
            ref_magnitude = ref_out.abs().mean()
            magnitude_ratio = kernel_magnitude / (ref_magnitude + 1e-6)
            assert 0.8 < magnitude_ratio < 1.2

            correlation = torch.corrcoef(
                torch.stack([kernel_out.flatten(), ref_out.flatten()])
            )[0, 1]
            assert correlation > 0.99

    def test_sanity_check_small_values(self):
        from sgl_kernel import mxfp4_blockwise_scaled_grouped_mm

        device = "xpu"
        m, n, k = 64, 64, 64

        a_data = torch.ones(m, k, dtype=torch.float32) * 2.0
        b_data = torch.eye(k, n, dtype=torch.float32) * 3.0

        a_packed, scales_a = quantize_to_mxfp4(a_data)
        b_packed, scales_b = quantize_to_mxfp4(b_data.t().contiguous())

        a_dq = dequantize_mxfp4(a_packed, scales_a, torch.float32)
        b_dq_nk = dequantize_mxfp4(b_packed, scales_b, torch.float32)
        b_dq = b_dq_nk.t()

        actual_ref = torch.matmul(a_dq, b_dq)

        a_list = [ensure_contiguous_layout(a_packed)]
        b_list = [ensure_contiguous_layout(b_packed)]
        scales_a_list = [ensure_contiguous_layout(scales_a)]
        scales_b_list = [ensure_contiguous_layout(scales_b)]

        a_list = [x.to(device) for x in a_list]
        b_list = [x.to(device) for x in b_list]
        scales_a_list = [x.to(device) for x in scales_a_list]
        scales_b_list = [x.to(device) for x in scales_b_list]

        inputs = prepare_kernel_inputs(
            a_list, b_list, scales_a_list, scales_b_list, device
        )

        mxfp4_blockwise_scaled_grouped_mm(
            inputs["output"],
            inputs["a_ptrs"],
            inputs["b_ptrs"],
            inputs["out_ptrs"],
            inputs["a_scales_ptrs"],
            inputs["b_scales_ptrs"],
            inputs["a_stack"],
            inputs["b_stack"],
            inputs["scales_a_stack"],
            inputs["scales_b_stack"],
            inputs["problem_sizes"],
            inputs["expert_offsets"],
            inputs["workspace"],
        )

        kernel_out = _expert_slice(inputs, 0).to("cpu")
        kernel_mean = kernel_out.mean()
        ref_mean = actual_ref.mean()

        assert 4.0 < kernel_mean < 8.0
        assert 4.0 < ref_mean < 8.0
        assert abs(kernel_mean - ref_mean) < 1.0


# Ragged-M (varying m_i per expert) — flat-2D layout with expert_offsets.
# Catches per-expert A-scale M-stride bug when scale_cols > 1.


@pytest.mark.skipif(not is_xpu_available(), reason="Intel XPU not available")
@pytest.mark.skipif(
    is_xpu_available() and not is_cri_device(),
    reason="MXFP4 blockwise scaled grouped GEMM requires a CRI (Xe3P) device",
)
class TestMXFP4RaggedM:
    """Ragged-M tests: per-expert m_i varies, flat-2D layout."""

    @pytest.fixture(autouse=True)
    def check_kernel_available(self):
        skip_if_no_xpu()
        if not is_cri_device():
            pytest.skip(
                "MXFP4 blockwise scaled grouped GEMM requires a CRI (Xe3P) device"
            )

    @torch.inference_mode()
    def test_flat_2d_ragged_distribution(self):
        """k=64 (scale_cols=2) exercises per-expert A-scale M-stride."""
        from sgl_kernel import mxfp4_blockwise_scaled_grouped_mm

        device = "xpu"
        n, k = 64, 64
        m_per_expert = [32, 64, 96, 64]
        num_experts = len(m_per_expert)
        total_m = sum(m_per_expert)
        packed_k = k // 2
        scale_cols = k // MXFP4_BLOCK_SIZE

        a_flat = torch.zeros((total_m, packed_k), dtype=torch.uint8, device=device)
        sa_flat = torch.zeros((total_m, scale_cols), dtype=torch.uint8, device=device)
        b_list, sb_list, ref_outputs = [], [], []

        expert_offsets_h = []
        row = 0
        for e, m_i in enumerate(m_per_expert):
            expert_offsets_h.append(row)
            a_q, sa, _ = create_random_mxfp4_data(m_i, k, "cpu", seed=42 + e)
            b_q, sb, _ = create_random_mxfp4_data(n, k, "cpu", seed=100 + e)

            a_flat[row : row + m_i] = a_q.to(device)
            sa_flat[row : row + m_i] = sa.to(device)
            b_list.append(b_q.to(device).contiguous())
            sb_list.append(sb.to(device).contiguous())

            a_dq = dequantize_mxfp4(a_q.cpu(), sa.cpu(), torch.float32)
            b_dq = dequantize_mxfp4(b_q.cpu(), sb.cpu(), torch.float32)
            ref_outputs.append(torch.matmul(a_dq, b_dq.t()))
            row += m_i

        b_stack = torch.stack(b_list).contiguous()
        sb_stack = torch.stack(sb_list).contiguous()
        output = torch.zeros((total_m, n), dtype=torch.float32, device=device)
        empty_ptrs = torch.empty((0,), dtype=torch.int64, device=device)
        problem_sizes = torch.tensor(
            [[m_i, n, k] for m_i in m_per_expert],
            dtype=torch.int32,
            device=device,
        )
        expert_offsets = torch.tensor(
            expert_offsets_h, dtype=torch.int32, device=device
        )
        workspace = torch.zeros((64 * 1024 * 1024,), dtype=torch.uint8, device=device)

        mxfp4_blockwise_scaled_grouped_mm(
            output,
            empty_ptrs,
            empty_ptrs,
            empty_ptrs,
            empty_ptrs,
            empty_ptrs,
            a_flat,
            b_stack,
            sa_flat,
            sb_stack,
            problem_sizes,
            expert_offsets,
            workspace,
        )

        row = 0
        for e, m_i in enumerate(m_per_expert):
            kernel_out = output[row : row + m_i].cpu()
            ref_out = ref_outputs[e]
            assert not torch.isnan(kernel_out).any(), f"Expert {e}: NaN"
            assert kernel_out.shape == ref_out.shape
            torch.testing.assert_close(kernel_out, ref_out, atol=1e-1, rtol=1e-1)
            row += m_i
