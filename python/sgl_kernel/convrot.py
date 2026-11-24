import torch


def convrot_rotate_quantize(
    input: torch.Tensor, group_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate BF16 [M, K] by regular H256 and return INT8 data and FP32 [M] scales.

    BMG only; contiguous inputs, K in [256, 8192] and divisible by 256.
    FP32 butterfly summation is rounded to BF16 before row quantization; it
    is not bit-exact with a dense-matmul rotation at rounding boundaries.
    """
    return torch.ops.sgl_kernel.convrot_rotate_quantize(input, group_size)


def convrot_int8_linear_prequant(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """INT8 [M,K] @ weight[N,K].T, with FP32 row/channel scales and BF16 output.

    Scaling is evaluated as acc * (input_scale * weight_scale), rounded to BF16
    before adding optional BF16 bias. No full INT32 output is materialized.
    """
    return torch.ops.sgl_kernel.convrot_int8_linear_prequant(
        input, weight, input_scale, weight_scale, bias
    )
