This folder is inherited from https://github.com/sgl-project/sglang/tree/1de4db9bef4521c4779769e4fcac87e68607e109/sgl-kernel. Currently **no test cases** here has been verified on XPU devices.

## Running the benchmark suite

`run_suite.py` owns the list of benchmarks per-commit CI runs, mirroring
`tests/run_suite.py`:

```bash
cd benchmark
python3 run_suite.py --suite per-commit       # run the CI set
python3 run_suite.py --suite per-commit --list # show what would run, then exit
```

Each benchmark's combined stdout/stderr is written to its own log under
`benchmark/`, and a timing summary is printed at the end. Unlike the previous
`&&`-chained shell command, every benchmark runs even if an earlier one fails,
and all failures are reported together.

To add a benchmark, append a `BenchFile(...)` to the `per-commit` list. To stop
running one, move it to the `SKIPPED` dict with a reason rather than deleting
the line.

### Sweep sizes

`BenchFile.args` narrows a sweep using flags the benchmark already exposes
(e.g. `--block-sizes 16 128`). Prefer that over editing sweep constants, so the
full sweep stays available for local runs and the nightly suite.

Two logs are load-bearing: **`flash.log`** and **`fused_moe.log`** are parsed by
`update_baseline_from_log.py` and compared against `baseline.json`. Do not
rename them, and do not trim those two benchmarks' configs — dropping a config
silently removes the matching baseline keys from the perf comparison instead of
failing loudly.
