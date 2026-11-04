import argparse

import sgl_kernel
import torch
import torch.nn.functional as F
import triton

# Qwen3.8-Flash-Next: hc_count 4, hidden 2560, mix low-rank 320.
HC, HIDDEN, LOWRANK = 4, 2560, 320
WIDTH = HC * HIDDEN
EPS = 1e-6
TOKENS = [1, 4, 16, 64, 256, 1024, 8192, 32768]
BYTES = 2  # bf16
DEVICE, DTYPE = "xpu", torch.bfloat16


def bench(fn):
    return triton.testing.do_bench(fn) * 1000  # mean time, us


def copy_bandwidth_gbps():
    # Empirical streaming ceiling: a device-to-device copy of 512 MB.
    src = torch.empty(256 * 2**20, device=DEVICE, dtype=DTYPE)
    dst = torch.empty_like(src)
    us = bench(lambda: dst.copy_(src))
    return 2 * src.numel() * BYTES / (us * 1e-6) / 1e9


def randn(*shape, scale=1.0):
    return torch.randn(*shape, device=DEVICE, dtype=DTYPE) * scale


# Each op builder returns (providers, bytes, flops) for `t` tokens. The last provider is the kernel under test and
# its bandwidth/TFLOPS use the DRAM bytes it must move once and the arithmetic ops per element (exp, reciprocal,
# divide, multiply and add each count as one).


def silu_eager(x):
    return F.silu(x / HC)


def build_silu(t):
    x = randn(t, LOWRANK)
    compiled = torch.compile(silu_eager)
    return (
        {
            "eager": lambda: silu_eager(x),
            "compile": lambda: compiled(x),
            "sgl": lambda: sgl_kernel.hc_silu(x, HC),
        },
        2 * x.numel() * BYTES,
        5 * x.numel(),
    )


def gate_mix_eager(x, gate):
    w = torch.sigmoid(gate).unflatten(-1, (HC, HIDDEN))
    return (w * x.unflatten(-1, (HC, HIDDEN))).mean(dim=-2)


def build_gate_mix(t):
    x, gate = randn(t, WIDTH), randn(t, WIDTH)
    compiled = torch.compile(gate_mix_eager)
    return (
        {
            "eager": lambda: gate_mix_eager(x, gate),
            "compile": lambda: compiled(x, gate),
            "sgl": lambda: sgl_kernel.hc_gate_mix(x, gate, HC),
        },
        (2 * t * WIDTH + t * HIDDEN) * BYTES,
        t * HIDDEN * (5 * HC + 1),
    )


def combine_eager(residual, block, inj):
    w = 2 * torch.sigmoid(inj / HC)
    return (
        residual.unflatten(-1, (HC, HIDDEN)) + block.unsqueeze(-2) * w.unsqueeze(-1)
    ).flatten(-2)


def build_combine(t):
    residual, block, inj = randn(t, WIDTH), randn(t, HIDDEN), randn(t, HC)
    compiled = torch.compile(combine_eager)
    return (
        {
            "eager": lambda: combine_eager(residual, block, inj),
            "compile": lambda: compiled(residual, block, inj),
            "sgl": lambda: sgl_kernel.hc_combine(residual, block, inj, HC),
        },
        (2 * t * WIDTH + t * HIDDEN + t * HC) * BYTES,
        2 * t * WIDTH,
    )


def combine_norm_eager(residual, block, inj, weight):
    combined = combine_eager(residual, block, inj)
    g = combined.float().unflatten(-1, (HC, HIDDEN))
    out = g * torch.rsqrt(g.square().mean(-1, keepdim=True) + EPS)
    out = out * (1 + weight.float().view(HC, HIDDEN))
    return combined, out.flatten(-2).to(DTYPE)


def build_combine_norm(t):
    residual, block, inj = randn(t, WIDTH), randn(t, HIDDEN), randn(t, HC)
    weight = randn(WIDTH, scale=0.1)
    compiled = torch.compile(combine_norm_eager)
    return (
        {
            "eager": lambda: combine_norm_eager(residual, block, inj, weight),
            "compile": lambda: compiled(residual, block, inj, weight),
            "sgl_2kernel": lambda: sgl_kernel.grouped_gemma_rmsnorm(
                sgl_kernel.hc_combine(residual, block, inj, HC), weight, HIDDEN
            ),
            "sgl_fused": lambda: sgl_kernel.hc_combine_norm(
                residual, block, inj, weight, EPS, HC
            ),
        },
        (3 * t * WIDTH + t * HIDDEN + t * HC + WIDTH) * BYTES,
        7 * t * WIDTH,
    )


def run_table(title, tokens, build, copy_gbps):
    print(f"\n{title}")
    header = None
    for t in tokens:
        torch.manual_seed(0)
        providers, nbytes, flops = build(t)
        us = {name: bench(fn) for name, fn in providers.items()}
        names = list(providers)
        mine = us[names[-1]]
        if header is None:
            header = (
                f"{'tokens':>8}"
                + "".join(f"{n + '_us':>16}" for n in names)
                + (f"{'GB/s':>9}{'%copy':>7}{'TFLOPS':>9}" if nbytes else "")
                + f"{'vs_eager':>10}{'vs_compile':>12}"
            )
            print(header, flush=True)
        rates = ""
        if nbytes:
            gbps = nbytes / (mine * 1e-6) / 1e9
            rates = f"{gbps:>9.1f}{100 * gbps / copy_gbps:>7.0f}{flops / (mine * 1e-6) / 1e12:>9.3f}"
        print(
            f"{t:>8}"
            + "".join(f"{us[n]:>16.1f}" for n in names)
            + rates
            + f"{us['eager'] / mine:>10.2f}{us['compile'] / mine:>12.2f}",
            flush=True,
        )


def combine_layer(block, residual, normed, weight):
    w = 2 * torch.sigmoid(F.linear(normed, weight) / HC)
    return (
        residual.unflatten(-1, (HC, HIDDEN)) + block.unsqueeze(-2) * w.unsqueeze(-1)
    ).flatten(-2)


def combine_layer_sgl(block, residual, normed, weight):
    return sgl_kernel.hc_combine(residual, block, F.linear(normed, weight), HC)


def mix_layer(normed, down, up):
    w = F.silu(F.linear(normed, down) / HC)
    w = torch.sigmoid(F.linear(w, up)).unflatten(-1, (HC, HIDDEN))
    return (w * normed.unflatten(-1, (HC, HIDDEN))).mean(dim=-2)


def mix_layer_sgl(normed, down, up):
    gate = F.linear(sgl_kernel.hc_silu(F.linear(normed, down), HC), up)
    return sgl_kernel.hc_gate_mix(normed, gate, HC)


def build_layers(weights):
    # Linear layers included, so only times are reported (the GEMMs dominate the bytes and flops).
    compiled_combine = torch.compile(combine_layer)
    compiled_mix = torch.compile(mix_layer)

    def build_combine_layer(t):
        args = (randn(t, HIDDEN), randn(t, WIDTH), randn(t, WIDTH), weights["inject"])
        return (
            {
                "eager": lambda: combine_layer(*args),
                "compile": lambda: compiled_combine(*args),
                "sgl": lambda: combine_layer_sgl(*args),
            },
            None,
            None,
        )

    def build_mix_layer(t):
        args = (randn(t, WIDTH), weights["down"], weights["up"])
        return (
            {
                "eager": lambda: mix_layer(*args),
                "compile": lambda: compiled_mix(*args),
                "sgl": lambda: mix_layer_sgl(*args),
            },
            None,
            None,
        )

    return build_combine_layer, build_mix_layer


def check_correctness(weights):
    t = 64
    c_args = (randn(t, HIDDEN), randn(t, WIDTH), randn(t, WIDTH), weights["inject"])
    m_args = (randn(t, WIDTH), weights["down"], weights["up"])
    torch.testing.assert_close(
        combine_layer_sgl(*c_args).float(),
        combine_layer(*c_args).float(),
        atol=0.07,
        rtol=2**-6,
    )
    torch.testing.assert_close(
        mix_layer_sgl(*m_args).float(),
        mix_layer(*m_args).float(),
        atol=0.05,
        rtol=2**-6,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=TOKENS)
    args = parser.parse_args()
    weights = dict(
        inject=randn(HC, WIDTH, scale=0.02),
        down=randn(LOWRANK, WIDTH, scale=0.02),
        up=randn(WIDTH, LOWRANK, scale=0.02),
    )
    check_correctness(weights)
    copy_gbps = copy_bandwidth_gbps()
    print(
        f"HC={HC} hidden={HIDDEN} bf16, mean time (triton do_bench). "
        f"Measured device copy bandwidth: {copy_gbps:.0f} GB/s (%copy = kernel GB/s / this).\n"
        "GB/s counts the DRAM bytes each op must move once; TFLOPS counts exp, reciprocal, divide, multiply and add "
        "as one op each."
    )
    run_table(
        "hc_silu (low-rank mix activation, 320 wide)",
        args.tokens,
        build_silu,
        copy_gbps,
    )
    run_table("hc_gate_mix", args.tokens, build_gate_mix, copy_gbps)
    run_table(
        "hc_combine (precomputed injection)", args.tokens, build_combine, copy_gbps
    )
    run_table(
        "hc_combine_norm (combine + grouped Gemma RMSNorm; sgl_fused is the kernel under test)",
        args.tokens,
        build_combine_norm,
        copy_gbps,
    )
    combine_layer_builder, mix_layer_builder = build_layers(weights)
    run_table(
        "combine layer (inject Linear + hc_combine)",
        args.tokens,
        combine_layer_builder,
        copy_gbps,
    )
    run_table(
        "mix layer (down/up Linear + hc_silu + hc_gate_mix)",
        args.tokens,
        mix_layer_builder,
        copy_gbps,
    )


if __name__ == "__main__":
    main()
