import json

import torch


def dequantize_packed_xe35(packed, scales):
    packed = packed.cpu()
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
    )
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    values = lookup[codes.long()]
    factors = torch.exp2(scales.cpu().float() - 127).repeat_interleave(32, -1)
    return values * factors


def verify_grouped_output_xe35(
    output,
    packed_activations,
    a_scale,
    packed_weights,
    b_scale,
    counts,
    max_rows=8,
    *,
    atol=5e-3,
    rtol=5e-3,
):
    """Production uses 0.005/0.005; wide synthetic uses explicit 0.005/0.05.

    The analogous NVFP4 grouped E2M1 FP32-accumulator peer in SGLang 75a7879
    test_nvfp4_blockwise_moe.py uses 0.1/0.1, not an identical MXFP4 scale
    contract: NVFP4 has per-16 E4M3 scales; MXFP4 has per-32 E8M0 scales.
    """
    counts = counts.cpu().tolist() if isinstance(counts, torch.Tensor) else counts
    if output.dtype == torch.float16:
        atol = rtol = 1e-3
    groups = packed_activations.shape[1] * 2 // 32
    row_start = scale_start = checked = 0
    maximum_error = 0.0
    for expert, rows in enumerate(counts):
        if not rows:
            continue
        padded_rows = (rows + 3) // 4 * 4
        if max_rows is None or rows <= max_rows:
            selected = torch.arange(rows)
        else:
            selected = torch.linspace(0, rows - 1, max_rows).long().unique()
        scales = a_scale.flatten()[
            scale_start * groups : (scale_start + padded_rows) * groups
        ]
        scales = scales.cpu().view(groups, padded_rows).T[selected]
        activations = dequantize_packed_xe35(
            packed_activations[row_start + selected.to(packed_activations.device)],
            scales,
        )
        weights = dequantize_packed_xe35(packed_weights[expert], b_scale[expert].T)
        expected = activations.double() @ weights.double().T
        actual = output[row_start + selected.to(output.device)].double().cpu()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        maximum_error = max(maximum_error, (actual - expected).abs().max().item())
        checked += actual.numel()
        row_start += rows
        scale_start += padded_rows
    report = {
        "reference": "CPU FP64 dequantized matmul, no output rounding",
        "checked": checked,
        "max_abs": maximum_error,
        "atol": atol,
        "rtol": rtol,
        "max_rows_per_expert": max_rows,
    }
    print(json.dumps(report), flush=True)
    return report
