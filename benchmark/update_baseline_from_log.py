import json
import os
import re


def parse_fused_moe_log(log_text: str) -> dict:
    lines = log_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if "Benchmark finished!" in line:
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("Benchmark finished! not found in fused_moe log")

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
        dtype = cols[6]
        with_bias = cols[7]
        act_type = cols[8]
        ms = float(cols[-1])

        key = f"fused_moe:{num_tokens}-{num_experts}-{topk}-{hidden_size}-{shard_intermediate_size}-{dtype}-{with_bias}-{act_type}"
        result[key] = ms

    return result


def parse_flash_attn_log(log_text: str) -> dict:
    lines = log_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if "Benchmark finished!" in line:
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("Benchmark finished! not found in flash_attn log")

    result = {}

    for line in lines[start_idx + 1 :]:
        line = line.strip()

        if not line.startswith("|"):
            continue
        if re.match(r"\|\s*-+", line):
            continue
        if "batch" in line:
            continue

        cols = [c.strip() for c in line.strip("|").split("|")]

        batch = cols[1]
        q_seq_length = cols[2]
        kv_seq_length = cols[3]
        num_heads_q = cols[4]
        num_heads_kv = cols[5]
        head_dim = cols[6]
        causal = cols[7]
        local = cols[8]
        use_sinks = cols[9]
        page_size = cols[10]
        ms = float(cols[-1])

        key = (
            f"flash_attn:{batch}-{q_seq_length}-{kv_seq_length}"
            f"-{num_heads_q}-{num_heads_kv}-{head_dim}"
            f"-{causal}-{local}-{use_sinks}-{page_size}"
        )
        result[key] = ms

    return result


def format_section(title, data, benchmark_type="fused_moe"):
    if not data:
        return f"### {title}\n\nNone\n"

    if benchmark_type == "flash_attn":
        header = "| config | log | baseline | ratio |"
    else:
        header = "| num_tokens - num_experts - topk - hidden_size - shard_intermediate_size | log | baseline | ratio |"

    lines = [
        f"### {title}",
        "",
        header,
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

        base_ms = baseline[k]

        if log_ms < base_ms:
            lower[k] = (log_ms, base_ms)
        elif log_ms > base_ms:
            higher[k] = (log_ms, base_ms)
        else:
            equal[k] = (log_ms, base_ms)

    return lower, higher, equal


def process_log(log_file, parser, benchmark_type, baseline):
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found, skipping {benchmark_type} benchmark")
        return {}, {}, {}

    with open(log_file) as f:
        log_text = f.read()

    data = parser(log_text)
    lower, higher, equal = compare(data, baseline)

    print(f"\n=== {benchmark_type} ===")
    print("=== LOWER (log < baseline) ===")
    for k, (l, b) in lower.items():
        ratio = l / b
        delta_pct = (ratio - 1.0) * 100.0
        print(f"{k}: log={l:.3f}, baseline={b}, ratio={delta_pct}")

    print("\n=== HIGHER (log > baseline) ===")
    for k, (l, b) in higher.items():
        ratio = l / b
        delta_pct = (ratio - 1.0) * 100.0
        print(f"{k}: log={l:.3f}, baseline={b}, ratio={delta_pct}")

    print("\n=== EQUAL (log == baseline) ===")
    for k, (l, b) in equal.items():
        ratio = l / b
        delta_pct = (ratio - 1.0) * 100.0
        print(f"{k}: log={l:.3f}, baseline={b}, ratio={delta_pct}")

    print("Collected benchmark data:")
    print(data)

    return lower, higher, equal


def main():
    with open("benchmark/baseline.json") as f:
        baseline = json.load(f)

    benchmarks = [
        ("fused_moe.log", parse_fused_moe_log, "fused_moe"),
        ("flash.log", parse_flash_attn_log, "flash_attn"),
    ]

    all_lower = {}
    all_higher = {}
    all_equal = {}

    for log_file, parser, benchmark_type in benchmarks:
        lower, higher, equal = process_log(log_file, parser, benchmark_type, baseline)
        all_lower.update(lower)
        all_higher.update(higher)
        all_equal.update(equal)

    # Separate results by type for formatting
    fused_moe_lower = {
        k: v for k, v in all_lower.items() if not k.startswith("flash_attn:")
    }
    fused_moe_higher = {
        k: v for k, v in all_higher.items() if not k.startswith("flash_attn:")
    }
    fused_moe_equal = {
        k: v for k, v in all_equal.items() if not k.startswith("flash_attn:")
    }
    flash_attn_lower = {
        k: v for k, v in all_lower.items() if k.startswith("flash_attn:")
    }
    flash_attn_higher = {
        k: v for k, v in all_higher.items() if k.startswith("flash_attn:")
    }
    flash_attn_equal = {
        k: v for k, v in all_equal.items() if k.startswith("flash_attn:")
    }

    sections = []
    if fused_moe_lower or fused_moe_higher or fused_moe_equal:
        sections.append("## Fused MoE Benchmark Comparison\n")
        sections.append("_Ratio = log / baseline (lower is better)_\n")
        sections.append(
            format_section("LOWER (log < baseline)", fused_moe_lower, "fused_moe")
        )
        sections.append(
            format_section("HIGHER (log > baseline)", fused_moe_higher, "fused_moe")
        )
        sections.append(format_section("EQUAL", fused_moe_equal, "fused_moe"))

    if flash_attn_lower or flash_attn_higher or flash_attn_equal:
        sections.append("## Flash Attention Benchmark Comparison\n")
        sections.append("_Ratio = log / baseline (lower is better)_\n")
        sections.append(
            format_section("LOWER (log < baseline)", flash_attn_lower, "flash_attn")
        )
        sections.append(
            format_section("HIGHER (log > baseline)", flash_attn_higher, "flash_attn")
        )
        sections.append(format_section("EQUAL", flash_attn_equal, "flash_attn"))

    pr_body = "\n".join(sections) if sections else "## Benchmark Comparison\n\nNo data."

    if all_lower:
        for k, (l, _) in all_lower.items():
            baseline[k] = l
        with open("benchmark/baseline.json", "w") as f:
            json.dump(baseline, f, indent=4)
            f.write("\n")

        open("ci_has_lower.txt", "w").write("1")
    else:
        open("ci_has_lower.txt", "w").write("0")
    print("ci_has_lower.txt content:", open("ci_has_lower.txt").read().strip())
    open("ci_pr_body.md", "w").write(pr_body)


if __name__ == "__main__":
    main()
