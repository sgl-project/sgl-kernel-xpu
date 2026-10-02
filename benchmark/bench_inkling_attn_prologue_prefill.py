"""Production-shape Inkling TP4 attention-prologue extend benchmark."""

from __future__ import annotations

import argparse

import torch
import triton
from sgl_kernel import inkling_attn_prologue_extend


def make_inputs() -> dict[str, object]:
    torch.manual_seed(20260923)
    device = "xpu"
    dtype = torch.bfloat16
    t, dq, dkv, width = 4096, 768, 128, 4
    row = dq + 2 * dkv + 64
    qkvr = (torch.randn((t, row), device=device) * 0.2).to(dtype)
    return {
        "qkvr": qkvr,
        "q": qkvr[:, :dq],
        "k_cache": (torch.randn((64, width - 1, dkv), device=device) * 0.1).to(dtype),
        "v_cache": (torch.randn((64, width - 1, dkv), device=device) * 0.1).to(dtype),
        "cache_indices": torch.tensor([3], dtype=torch.int32, device=device),
        "cache_mask": torch.ones(1, dtype=torch.bool, device=device),
        "has_initial_state": torch.ones(1, dtype=torch.bool, device=device),
        "cu": torch.tensor([0, t], dtype=torch.int64, device=device),
        "si": torch.zeros(t, dtype=torch.int32, device=device),
        "k_weight": (torch.randn((dkv, width), device=device) * 0.1).to(dtype),
        "v_weight": (torch.randn((dkv, width), device=device) * 0.1).to(dtype),
        "track_rows": torch.tensor(
            [[t - 3, t - 2, t - 1]], dtype=torch.int64, device=device
        ),
        "track_mask": torch.zeros(1, dtype=torch.bool, device=device),
        "track_dst": torch.zeros(1, dtype=torch.int64, device=device),
        "q_gamma": (1 + torch.randn(128, device=device) * 0.1).to(dtype),
        "k_gamma": (1 + torch.randn(128, device=device) * 0.1).to(dtype),
        "loc": torch.arange(t, dtype=torch.int64, device=device),
        "k_buf": torch.empty((t, 1, 128), dtype=dtype, device=device),
        "v_buf": torch.empty((t, 1, 128), dtype=dtype, device=device),
        "dq": dq,
        "dkv": dkv,
    }


def invoke(x: dict[str, object]):
    return inkling_attn_prologue_extend(
        x["q"],
        x["k_cache"],
        x["v_cache"],
        x["cache_indices"],
        x["cache_mask"],
        x["has_initial_state"],
        x["cu"],
        x["si"],
        x["k_weight"],
        x["v_weight"],
        x["track_rows"],
        x["track_mask"],
        x["track_dst"],
        x["q_gamma"],
        x["k_gamma"],
        1.0e-5,
        x["loc"],
        x["k_buf"],
        x["v_buf"],
        0,
        x["dq"] + 64,
        x["dq"] + 64 + x["dkv"],
        x["dq"],
        x["dkv"],
        activation="silu",
        use_residual=True,
        do_store=True,
        do_cache_update=True,
    )[:3]


def reference(x: dict[str, object]):
    qkvr = x["qkvr"].float()
    dq, dkv = x["dq"], x["dkv"]
    k_off, v_off = dq + 64, dq + 64 + dkv
    q = qkvr[:, :dq]
    q_heads = q.view(-1, dq // 128, 128)
    q_out = q_heads * torch.rsqrt(q_heads.square().mean(-1, keepdim=True) + 1.0e-5)
    q_out = (q_out * x["q_gamma"].float()).to(torch.bfloat16).view(-1, dq)

    def conv(off, weight):
        z = qkvr[:, off : off + dkv]
        padded = torch.cat(
            [
                x["k_cache" if off == k_off else "v_cache"][3].float(),
                z,
            ]
        )
        y = sum(padded[i : i + z.shape[0]] * weight[:, i].float() for i in range(4))
        return torch.nn.functional.silu(y) + z

    k_conv = conv(k_off, x["k_weight"]).to(torch.bfloat16).float()
    k_heads = k_conv.view(-1, dkv // 128, 128)
    k_out = k_heads * torch.rsqrt(k_heads.square().mean(-1, keepdim=True) + 1.0e-5)
    k_out = (k_out * x["k_gamma"].float()).to(torch.bfloat16).view(-1, dkv)
    v_out = conv(v_off, x["v_weight"]).to(torch.bfloat16)
    return q_out, k_out, v_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--correctness", action="store_true")
    args = parser.parse_args()
    x = make_inputs()
    if args.correctness:
        expected = reference(x)
        actual = invoke(x)
        for name, got, ref in zip(("q", "k", "v"), actual, expected):
            diff = (got.float() - ref.float()).abs()
            torch.testing.assert_close(
                got.float(), ref.float(), atol=3.0e-2, rtol=3.0e-2
            )
            print(
                f"{name}: max_abs={diff.max().item():.8f} "
                f"mean_abs={diff.mean().item():.8f}"
            )
        torch.testing.assert_close(
            x["k_buf"].view(-1, x["dkv"]).float(),
            expected[1].float(),
            atol=3.0e-2,
            rtol=3.0e-2,
        )
        torch.testing.assert_close(
            x["v_buf"].view(-1, x["dkv"]).float(),
            expected[2].float(),
            atol=3.0e-2,
            rtol=3.0e-2,
        )
        print("kv_store: matches non-zero reference")
        torch.testing.assert_close(
            x["k_cache"][3].float(),
            x["qkvr"][-3:, x["dq"] + 64 : x["dq"] + 64 + x["dkv"]].float(),
        )
        torch.testing.assert_close(
            x["v_cache"][3].float(),
            x["qkvr"][
                -3:, x["dq"] + 64 + x["dkv"] : x["dq"] + 64 + 2 * x["dkv"]
            ].float(),
        )
        print("conv_cache: matches trailing input rows")

    median_ms, p20_ms, p80_ms = triton.testing.do_bench(
        lambda: invoke(x),
        warmup=args.warmup,
        rep=args.rep,
        quantiles=[0.5, 0.2, 0.8],
    )
    print(
        f"T=4096 dq=768 dkv=128 W=4 BF16 GPU "
        f"p50={median_ms:.6f} ms "
        f"p20={p20_ms:.6f} ms "
        f"p80={p80_ms:.6f} ms"
    )


if __name__ == "__main__":
    main()
