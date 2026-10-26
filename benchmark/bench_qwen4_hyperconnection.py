import argparse

import sgl_kernel
import torch
import torch.nn.functional as F
import triton

# Qwen3.8-Flash-Next: hc_count 4, hidden 2560, mix low-rank 320.
HC, HIDDEN, LOWRANK = 4, 2560, 320
TOKENS = [1, 4, 16, 64, 256, 1024, 8192, 32768]


def combine_compute(block, residual, normed, weight, hc, hs):
    w = 2 * torch.sigmoid(F.linear(normed, weight) / hc)
    return (
        residual.unflatten(-1, (hc, hs)) + block.unsqueeze(-2) * w.unsqueeze(-1)
    ).flatten(-2)


def mix_compute(normed, down, up, hc, hs):
    w = F.silu(F.linear(normed, down) / hc)
    w = torch.sigmoid(F.linear(w, up)).unflatten(-1, (hc, hs))
    return (w * normed.unflatten(-1, (hc, hs))).mean(dim=-2)


def combine_sgl(block, residual, normed, weight, hc, hs):
    return sgl_kernel.hc_combine(residual, block, F.linear(normed, weight), hc)


def mix_sgl(normed, down, up, hc, hs):
    gate = F.linear(sgl_kernel.hc_silu(F.linear(normed, down), hc), up)
    return sgl_kernel.hc_gate_mix(normed, gate, hc)


def bench(fn):
    return triton.testing.do_bench(fn) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=TOKENS)
    args = parser.parse_args()
    dev, dt = "xpu", torch.bfloat16
    width = HC * HIDDEN
    weights = dict(
        inject=torch.randn(HC, width, device=dev, dtype=dt) * 0.02,
        down=torch.randn(LOWRANK, width, device=dev, dtype=dt) * 0.02,
        up=torch.randn(width, LOWRANK, device=dev, dtype=dt) * 0.02,
    )
    compiled_combine = torch.compile(combine_compute)
    compiled_mix = torch.compile(mix_compute)

    print("median time, us (Linear layers included)\n")
    print(
        f"{'tokens':>8}{'comb_eager':>12}{'comb_compile':>14}{'comb_sgl':>10}{'mix_eager':>12}"
        f"{'mix_compile':>13}{'mix_sgl':>9}{'tail_eager':>12}{'tail_sgl':>10}"
    )
    for t in args.tokens:
        torch.manual_seed(0)
        residual = torch.randn(t, width, device=dev, dtype=dt)
        normed = torch.randn(t, width, device=dev, dtype=dt)
        block = torch.randn(t, HIDDEN, device=dev, dtype=dt)
        inj = F.linear(normed, weights["inject"])
        c_args = (block, residual, normed, weights["inject"], HC, HIDDEN)
        m_args = (normed, weights["down"], weights["up"], HC, HIDDEN)
        gate = torch.randn(t, width, device=dev, dtype=dt)

        def tail_eager():
            w = 2 * torch.sigmoid(inj / HC)
            return (
                residual.unflatten(-1, (HC, HIDDEN))
                + block.unsqueeze(-2) * w.unsqueeze(-1)
            ).flatten(-2)

        row = [
            bench(lambda: combine_compute(*c_args)),
            bench(lambda: compiled_combine(*c_args)),
            bench(lambda: combine_sgl(*c_args)),
            bench(lambda: mix_compute(*m_args)),
            bench(lambda: compiled_mix(*m_args)),
            bench(lambda: mix_sgl(*m_args)),
            bench(tail_eager),
            bench(lambda: sgl_kernel.hc_combine(residual, block, inj, HC)),
        ]
        print(
            f"{t:>8}{row[0]:>12.1f}{row[1]:>14.1f}{row[2]:>10.1f}{row[3]:>12.1f}{row[4]:>13.1f}{row[5]:>9.1f}"
            f"{row[6]:>12.1f}{row[7]:>10.1f}",
            flush=True,
        )
        del gate


if __name__ == "__main__":
    main()
