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
    # (64, 128, 128),
    # (128, 256, 256),
    # (256, 512, 512),
    # (512, 512, 512),
]


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


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
    num_experts = len(a_list)

    m, packed_k = a_list[0].shape
    k = packed_k * 2
    n_b, packed_k_b = b_list[0].shape
    k_b = packed_k_b * 2
    n = n_b
    assert k == k_b

    a_stack = torch.stack(
        [ensure_contiguous_layout(a) for a in a_list], dim=0
    ).contiguous()
    b_stack = torch.stack(
        [ensure_contiguous_layout(b) for b in b_list], dim=0
    ).contiguous()
    scales_a_stack = torch.stack(
        [ensure_contiguous_layout(s.t().contiguous()) for s in scales_a_list], dim=0
    ).contiguous()
    scales_b_stack = torch.stack(
        [ensure_contiguous_layout(s.t().contiguous()) for s in scales_b_list], dim=0
    ).contiguous()

    output = torch.zeros((num_experts, m, n), dtype=torch.float32, device=device)

    a_ptrs = torch.tensor(
        [a_stack[i].data_ptr() for i in range(num_experts)],
        dtype=torch.uint64,
        device=device,
    )
    b_ptrs = torch.tensor(
        [b_stack[i].data_ptr() for i in range(num_experts)],
        dtype=torch.uint64,
        device=device,
    )
    out_ptrs = torch.tensor(
        [output[i].data_ptr() for i in range(num_experts)],
        dtype=torch.uint64,
        device=device,
    )
    a_scales_ptrs = torch.tensor(
        [scales_a_stack[i].data_ptr() for i in range(num_experts)],
        dtype=torch.uint64,
        device=device,
    )
    b_scales_ptrs = torch.tensor(
        [scales_b_stack[i].data_ptr() for i in range(num_experts)],
        dtype=torch.uint64,
        device=device,
    )

    stride_a = torch.tensor(
        [[k, 1, 0] for _ in range(num_experts)], dtype=torch.int64, device=device
    )
    stride_b = torch.tensor(
        [[k, 1, 0] for _ in range(num_experts)], dtype=torch.int64, device=device
    )
    stride_c = torch.tensor(
        [[n, 1, 0] for _ in range(num_experts)], dtype=torch.int64, device=device
    )

    layout_sfa = torch.tensor(
        [[1, m, 1] for _ in range(num_experts)], dtype=torch.int64, device=device
    )
    layout_sfb = torch.tensor(
        [[1, n, 1] for _ in range(num_experts)], dtype=torch.int64, device=device
    )

    problem_sizes = torch.tensor(
        [[m, n, k] for _ in range(num_experts)], dtype=torch.int32, device=device
    )
    expert_offsets = torch.arange(num_experts, dtype=torch.int32, device=device)

    workspace = torch.empty((1024 * 1024 * 1024,), dtype=torch.uint8, device=device)

    return {
        "output": output,
        "a_ptrs": a_ptrs,
        "b_ptrs": b_ptrs,
        "out_ptrs": out_ptrs,
        "a_scales_ptrs": a_scales_ptrs,
        "b_scales_ptrs": b_scales_ptrs,
        "a_stack": a_stack,
        "b_stack": b_stack,
        "scales_a_stack": scales_a_stack,
        "scales_b_stack": scales_b_stack,
        "stride_a": stride_a,
        "stride_b": stride_b,
        "stride_c": stride_c,
        "layout_sfa": layout_sfa,
        "layout_sfb": layout_sfb,
        "problem_sizes": problem_sizes,
        "expert_offsets": expert_offsets,
        "workspace": workspace,
    }


class TestMXFP4Quantization:
    """Tests for MXFP4 quantization and dequantization utilities."""

    def test_e2m1_roundtrip(self):
        device = torch.device("cpu")
        test_values = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        quantized = quantize_to_e2m1(test_values)
        dequantized = dequantize_e2m1(quantized)
        torch.testing.assert_close(dequantized, test_values, atol=0.0, rtol=0.0)

    def test_pack_unpack_roundtrip(self):
        device = torch.device("cpu")
        m, k = 16, 64
        original = torch.randint(0, 16, (m, k), dtype=torch.uint8, device=device)
        packed = pack_fp4(original)
        unpacked = unpack_fp4(packed)
        torch.testing.assert_close(unpacked, original)

    def test_mxfp4_quantization_shape(self):
        device = torch.device("cpu")
        m, k = 32, 128
        original = torch.randn(m, k, dtype=torch.float32, device=device)
        packed, scales = quantize_to_mxfp4(original)
        assert packed.shape == (m, k // 2)
        assert scales.shape == (m, k // MXFP4_BLOCK_SIZE)
        assert packed.dtype == torch.uint8
        assert scales.dtype == torch.uint8

    def test_mxfp4_dequantization_accuracy(self):
        device = torch.device("cpu")
        m, k = 32, 128
        original = torch.randn(m, k, dtype=torch.float32, device=device) * 3.0
        packed, scales = quantize_to_mxfp4(original)
        dequantized = dequantize_mxfp4(packed, scales, torch.float32)
        assert dequantized.shape == original.shape
        relative_error = (dequantized - original).abs() / (original.abs() + 1e-6)
        mean_error = relative_error.mean().item()
        assert mean_error < 0.5


@pytest.mark.skipif(not is_xpu_available(), reason="Intel XPU not available")
class TestMXFP4BlockwiseScaledGroupedMM:
    """Tests for the MXFP4 MoE CUTLASS kernel on Intel XPU."""

    @pytest.fixture(autouse=True)
    def check_kernel_available(self):
        skip_if_no_xpu()

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
            inputs["stride_a"],
            inputs["stride_b"],
            inputs["stride_c"],
            inputs["layout_sfa"],
            inputs["layout_sfb"],
            inputs["problem_sizes"],
            inputs["expert_offsets"],
            inputs["workspace"],
        )

        for i in range(num_experts):
            kernel_out = inputs["output"][i].to("cpu")
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
            inputs["stride_a"],
            inputs["stride_b"],
            inputs["stride_c"],
            inputs["layout_sfa"],
            inputs["layout_sfb"],
            inputs["problem_sizes"],
            inputs["expert_offsets"],
            inputs["workspace"],
        )

        kernel_out = inputs["output"][0].to("cpu")
        kernel_mean = kernel_out.mean()
        ref_mean = actual_ref.mean()

        assert 4.0 < kernel_mean < 8.0
        assert 4.0 < ref_mean < 8.0
        assert abs(kernel_mean - ref_mean) < 1.0
