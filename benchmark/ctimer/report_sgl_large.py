"""Report the large E=2 SGL grouped-GEMM sweep as a compact TSV table."""

import argparse
import json


ROWS = [
    ("bf16", "gate_up_proj", "gemm1_gate_up", 1024, 1536, 2048, 2),
    ("bf16", "down_proj", "gemm2_down", 1024, 2048, 768, 2),
    ("bf16", "gate_up_proj", "gemm1_gate_up", 3500, 1536, 2048, 2),
    ("bf16", "down_proj", "gemm2_down", 3500, 2048, 768, 2),
]

PEAK_TFLOPS_32_XECORE = 512.0


def tflops(total_m, n, k, elapsed_s):
    return (2.0 * total_m * n * k) * 1e-12 / elapsed_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="out/sgl_large_E2.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    lines = [
        "\t".join(
            [
                "dtype",
                "proj",
                "M_per_group",
                "N",
                "K",
                "E",
                "total_M",
                "iterations",
                "avg_ms",
                "TFLOPS",
                "MFU(32 xecore)",
            ]
        )
    ]

    for dtype, proj, label, m_per_group, n, k, experts in ROWS:
        total_m = m_per_group * experts
        result = data["results"][label][str(m_per_group)]
        avg_s = result["avg_s"] if isinstance(result, dict) else result
        avg_ms = avg_s * 1000.0
        tf = tflops(total_m, n, k, avg_s)
        mfu = tf / PEAK_TFLOPS_32_XECORE
        lines.append(
            "\t".join(
                [
                    dtype,
                    proj,
                    str(m_per_group),
                    str(n),
                    str(k),
                    str(experts),
                    str(total_m),
                    "1",
                    f"{avg_ms:.6f}",
                    f"{tf:.7f}",
                    f"{mfu:.2%}",
                ]
            )
        )

    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
