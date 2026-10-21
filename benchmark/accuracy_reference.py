import json

import torch


def dequantize_nvfp4_prepared_xe35(packed, multipliers, scales, mode):
    packed = packed.cpu()
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
    )
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    values = lookup[codes.long()]
    multipliers = multipliers.float().cpu()
    exponents = torch.exp2(scales.float().cpu().T - 127)
    if mode == "bf16":
        factors = (multipliers * exponents.repeat_interleave(2, -1)).to(torch.bfloat16)
        return (
            (values * factors.float().repeat_interleave(16, -1))
            .to(torch.bfloat16)
            .float()
        )
    normalized = values * multipliers.repeat_interleave(16, -1)
    return normalized.to(torch.float8_e4m3fn).float() * exponents.repeat_interleave(
        32, -1
    )


def verify_grouped_output_xe35(
    output,
    activations,
    packed_weights,
    multipliers,
    scales,
    counts,
    mode,
    activation_scales=None,
    max_rows=8,
):
    counts = counts.cpu().tolist() if isinstance(counts, torch.Tensor) else counts
    tolerance = 1e-3 if output.dtype == torch.float16 else 5e-3
    offset = checked = 0
    maximum_error = 0.0
    for expert, rows in enumerate(counts):
        if not rows:
            continue
        selected = (
            torch.arange(rows)
            if rows <= max_rows
            else torch.linspace(0, rows - 1, max_rows).long().unique()
        )
        indices = offset + selected.to(activations.device)
        inputs = activations[indices].float().cpu()
        if activation_scales is not None:
            factors = torch.exp2(activation_scales[indices].float().cpu() - 127)
            inputs *= factors.repeat_interleave(32, -1)
        if mode == "plain_bf16":
            weights = packed_weights[expert].float().cpu()
        elif mode == "plain_mxfp8":
            factors = torch.exp2(scales[expert].float().cpu() - 127)
            weights = packed_weights[expert].float().cpu() * factors.repeat_interleave(
                32, -1
            )
        else:
            weights = dequantize_nvfp4_prepared_xe35(
                packed_weights[expert], multipliers[expert], scales[expert], mode
            )
        expected = inputs @ weights.T
        actual = output[indices].float().cpu()
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        maximum_error = max(maximum_error, (actual - expected).abs().max().item())
        checked += actual.numel()
        offset += rows
    report = {
        "reference": "CPU FP32 matmul of independently decoded production operands",
        "mode": mode,
        "checked": checked,
        "max_abs": maximum_error,
        "atol": tolerance,
        "rtol": tolerance,
    }
    print(json.dumps(report), flush=True)
    return report
