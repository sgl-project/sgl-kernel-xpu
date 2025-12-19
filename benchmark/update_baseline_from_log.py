import json
import re


def parse_benchmark_log(log_text: str) -> dict:
    lines = log_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if "Benchmark finished!" in line:
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("Benchmark finished! not found")

    result = {}

    for line in lines[start_idx + 1 :]:
        line = line.strip()

        if not line.startswith("|"):
            continue
        if re.match(r"\|\s*-+", line):
            continue
        if "num_tokens" in line:
            continue

        cols = [c.strip() for c in line.strip("|").split("|")]

        num_tokens = cols[1]
        num_experts = cols[2]
        topk = cols[3]
        hidden_size = cols[4]
        shard_intermediate_size = cols[5]
        ms = float(cols[-1])

        key = (
            f"{num_tokens}-{num_experts}-{topk}-{hidden_size}-{shard_intermediate_size}"
        )
        result[key] = ms

    return result


def format_section(title, data):
    if not data:
        return f"### {title}\n\nNone\n"

    lines = [
        f"### {title}",
        "",
        "| num_tokens - num_experts - topk - hidden_size - shard_intermediate_size | log | baseline | ratio |",
        "|---|---:|---:|---:|",
    ]
    for k, (l, b) in sorted(data.items()):
        ratio = l / b
        delta_pct = (ratio - 1.0) * 100.0
        lines.append(f"| `{k}` | {l:.3f} | {b} | {delta_pct:+.2f}% |")
    lines.append("")
    return "\n".join(lines)


def compare(log_data: dict, baseline: dict):
    lower = {}
    higher = {}
    equal = {}

    for k, log_ms in log_data.items():
        if k not in baseline:
            continue

        log_us = log_ms
        base_us = baseline[k]

        if log_us < base_us:
            lower[k] = (log_us, base_us)
        elif log_us > base_us:
            higher[k] = (log_us, base_us)
        else:
            equal[k] = (log_us, base_us)

    return lower, higher, equal


def main():

    with open("fused_moe.log") as f:
        log_text = f.read()

    data = parse_benchmark_log(log_text)

    with open("benchmark/baseline.json") as f:
        baseline = json.load(f)

    lower, higher, equal = compare(data, baseline)

    print("=== LOWER (log < baseline) ===")
    for k, (l, b) in lower.items():
        ratio = l / b
        delta_pct = (ratio - 1.0) * 100.0
        print(f"{k}: log={l:.3f}, baseline={b}", ratio={delta_pct})

    print("\n=== HIGHER (log > baseline) ===")
    for k, (l, b) in higher.items():
        ratio = l / b
        delta_pct = (ratio - 1.0) * 100.0
        print(f"{k}: log={l:.3f}, baseline={b}", ratio={delta_pct})

    print("\n=== EQUAL (log == baseline) ===")
    for k, (l, b) in equal.items():
        print(f"{k}: log={l:.3f}, baseline={b}")

    print("data")
    print(data)

    pr_body = "\n".join(
        [
            "## Benchmark Comparison",
            "",
            "_Ratio = log / baseline (lower is better)_",
            "",
            format_section("LOWER (log < baseline)", lower),
            format_section("HIGHER (log > baseline)", higher),
            format_section("EQUAL", equal),
        ]
    )

    if lower:
        for k, (l, _) in lower.items():
            baseline[k] = l
        with open("benchmark/baseline.json", "w") as f:
            json.dump(baseline, f, indent=4)
            f.write("\n")

        open("ci_has_lower.txt", "w").write("1")
    else:
        open("ci_has_lower.txt", "w").write("0")

    open("ci_pr_body.md", "w").write(pr_body)


if __name__ == "__main__":
    main()
