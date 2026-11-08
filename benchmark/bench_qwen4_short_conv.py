import argparse

import sgl_kernel
import torch
import torch.nn.functional as F
import triton

# Qwen3.8-Flash-Next PLE short conv: 10240 channels, kernel 4, dilation 3 (state length 9).
CHANNELS = 10240
KERNEL_SIZE = 4
DILATION = 3
STATE_LEN = (KERNEL_SIZE - 1) * DILATION
VERIFY_WIDTH = 4
DECODE_TOKENS = [1, 8, 32, 64, 128, 256]
VERIFY_REQUESTS = [1, 4, 16, 32, 64]
PROVIDERS = ["sgl_kernel", "torch"]
BF16_BYTES = 2
FP32_BYTES = 4

times_us = {}


# Mirrors the decode branch of Qwen4ExpPLELayer._short_conv.
def short_conv_state_torch(state, state_indices, x):
    conv_input = torch.cat(
        [state.index_select(0, state_indices), x.unsqueeze(-1)], dim=-1
    )
    real = state_indices != 0
    state[state_indices[real]] = conv_input[real][:, :, 1:]
    return conv_input


# Mirrors the target-verify branch of Qwen4ExpPLELayer._short_conv and forward.
def verify_conv_torch(x, residual, weight, state, state_indices, valid, intermediate):
    requests = state_indices.numel()
    selected = state.index_select(0, state_indices).to(x.dtype)
    seq = x.view(requests, VERIFY_WIDTH, CHANNELS).transpose(1, 2)
    conv_input = torch.cat([selected, seq], dim=-1)
    conv = F.conv1d(conv_input, weight, dilation=DILATION, groups=CHANNELS)
    activated = F.silu(conv.transpose(1, 2).reshape(-1, CHANNELS))
    windows = conv_input.unfold(2, STATE_LEN, 1)[:, :, 1 : VERIFY_WIDTH + 1]
    windows = windows.permute(0, 2, 1, 3)
    intermediate[:requests, :VERIFY_WIDTH] = torch.where(
        valid.view(requests, VERIFY_WIDTH, 1, 1), windows, torch.zeros_like(windows)
    ).to(intermediate.dtype)
    return torch.where(
        valid.unsqueeze(-1), residual + activated, torch.zeros_like(residual)
    )


def state_bytes(tokens):
    # Reads state rows and x, writes conv_input and the advanced state.
    per_token = STATE_LEN + 1 + (STATE_LEN + 1) + STATE_LEN
    return tokens * CHANNELS * per_token * BF16_BYTES


def verify_bytes(requests):
    # Reads x, residual and state rows; writes the output and the fp32 intermediate windows.
    tokens = requests * VERIFY_WIDTH
    return (
        3 * tokens * CHANNELS * BF16_BYTES
        + requests * CHANNELS * STATE_LEN * FP32_BYTES
        + tokens * CHANNELS * STATE_LEN * FP32_BYTES
    )


def report(label, size, nbytes):
    t = times_us
    gbps = nbytes / (t["sgl_kernel"] * 1e-6) / 1e9
    print(
        f"{label:>8}{size:>8}{t['sgl_kernel']:>14.2f}{t['torch']:>14.2f}"
        f"{gbps:>14.2f}{t['torch'] / t['sgl_kernel']:>14.2f}",
        flush=True,
    )


def bench(fn, provider):
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    times_us[provider] = ms * 1000
    return ms * 1000, min_ms * 1000, max_ms * 1000


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["tokens"],
        x_vals=DECODE_TOKENS,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=["sgl_kernel", "torch (eager)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="qwen4-short-conv-state-performance",
        args={},
    )
)
def benchmark_state(tokens, provider):
    device = "xpu"
    torch.manual_seed(0)
    slots = 2 * tokens + 1
    state = torch.randn(slots, CHANNELS, STATE_LEN, dtype=torch.bfloat16, device=device)
    x = torch.randn(tokens, CHANNELS, dtype=torch.bfloat16, device=device)
    state_indices = torch.randperm(slots - 1, device=device)[:tokens] + 1
    if provider == "sgl_kernel":

        def fn():
            sgl_kernel.qwen4_short_conv_state(state, state_indices, x)

    else:

        def fn():
            short_conv_state_torch(state, state_indices, x)

    result = bench(fn, provider)
    if provider == PROVIDERS[-1]:
        report("decode", tokens, state_bytes(tokens))
    return result


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["requests"],
        x_vals=VERIFY_REQUESTS,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=["sgl_kernel", "torch (eager)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="qwen4-verify-conv-performance",
        args={},
    )
)
def benchmark_verify(requests, provider):
    device = "xpu"
    torch.manual_seed(0)
    tokens = requests * VERIFY_WIDTH
    slots = 2 * requests + 1
    x = torch.randn(tokens, CHANNELS, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(x)
    weight = torch.randn(CHANNELS, 1, KERNEL_SIZE, dtype=torch.bfloat16, device=device)
    state = torch.randn(slots, CHANNELS, STATE_LEN, dtype=torch.float32, device=device)
    state_indices = torch.randperm(slots - 1, device=device)[:requests] + 1
    valid = torch.rand(tokens, device=device) < 0.9
    intermediate = torch.empty(
        requests, VERIFY_WIDTH, CHANNELS, STATE_LEN, dtype=torch.float32, device=device
    )

    if provider == "sgl_kernel":

        def fn():
            sgl_kernel.qwen4_verify_conv(
                x,
                residual,
                weight,
                state,
                state_indices,
                valid,
                VERIFY_WIDTH,
                DILATION,
                intermediate,
            )

    else:

        def fn():
            verify_conv_torch(
                x, residual, weight, state, state_indices, valid, intermediate
            )

    result = bench(fn, provider)
    if provider == PROVIDERS[-1]:
        report("verify", requests, verify_bytes(requests))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-tokens", nargs="+", type=int, default=DECODE_TOKENS)
    parser.add_argument(
        "--verify-requests", nargs="+", type=int, default=VERIFY_REQUESTS
    )
    args = parser.parse_args()
    benchmark_state.benchmarks.x_vals = args.decode_tokens
    benchmark_verify.benchmarks.x_vals = args.verify_requests

    print(
        f"qwen4 PLE short conv (C={CHANNELS}, L={STATE_LEN}, verify W={VERIFY_WIDTH}): "
        "median time (triton do_bench), us\n"
    )
    print(
        f"{'op':>8}{'size':>8}{'sgl_kernel':>14}{'torch':>14}"
        f"{'sgl_GB/s':>14}{'vs_torch':>14}",
        flush=True,
    )
    benchmark_state.run(print_data=False)
    benchmark_verify.run(print_data=False)
