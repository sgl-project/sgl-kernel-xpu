"""Check paired normal-package Muse Spark B8 permutation measurements."""

import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error("At least three alternating-order rounds are required")
    comparisons = []
    for index_dtype in ("int32", "int64"):
        for routing in ("balanced", "random", "colliding"):
            samples = {"candidate": [], "legacy": []}
            reference = None
            for round_index in range(args.rounds):
                for implementation in samples:
                    result_path = args.results_dir / (
                        f"r{round_index}-{index_dtype}-{routing}-{implementation}.json"
                    )
                    result = json.loads(result_path.read_text())
                    if (
                        result["implementation"] != implementation
                        or result["shape"] != [8, 8, 256, 4]
                        or result["index_dtype"] != index_dtype
                        or result["routing"] != routing
                        or not result["libraries"]
                        or len(result["samples_us"]) < 31
                        or not all(
                            math.isfinite(sample) and sample > 0
                            for sample in result["samples_us"]
                        )
                    ):
                        raise RuntimeError(f"Invalid measurement provenance: {result_path}")
                    actual = (result["input_ids"], result["outputs"])
                    if reference is None:
                        reference = actual
                    elif actual != reference:
                        raise RuntimeError(f"Input or exact four-output mismatch: {result_path}")
                    median = statistics.median(result["samples_us"])
                    if median != result["median_us"]:
                        raise RuntimeError(f"Invalid recorded median: {result_path}")
                    samples[implementation].append(median)
            candidate = statistics.median(samples["candidate"])
            legacy = statistics.median(samples["legacy"])
            ratio = candidate / legacy
            comparison = {
                "index_dtype": index_dtype,
                "routing": routing,
                "candidate_us": candidate,
                "legacy_us": legacy,
                "ratio": ratio,
                "round_medians_us": samples,
                "pass": ratio <= 1,
            }
            comparisons.append(comparison)
            print(json.dumps(comparison), flush=True)
    (args.results_dir / "comparison.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )
    if not all(comparison["pass"] for comparison in comparisons):
        raise RuntimeError("Performance regression against the old repository operator")


if __name__ == "__main__":
    main()
