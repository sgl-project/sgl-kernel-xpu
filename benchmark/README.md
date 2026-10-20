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

## Muse-Spark memory operators

`bench_memory_compare.py` compares the registered Xe35 paged allocation,
request-to-token write, and fused allocation/write operators against SGLang's
Triton memory path. The earlier `sgl-kernel-xpu` package has no corresponding
native operators. The report records the actual reference source paths.

Validate the native package registration, not a diagnostic registration bridge.
For a memory-only validation build, the repository's existing feature switches
can omit unrelated attention and MoE families without changing any memory
operator, its launch configuration, or its registered API:

```bash
cmake -S . -B build -DDPCPP_SYCL_TARGET=cri \
  -DUSE_MOE=OFF -DUSE_FMHA=OFF -DUSE_MLA=OFF -DUSE_GDN=OFF -DUSE_QSA=OFF
cmake --build build --target common_ops --parallel 2
cmake --install build --prefix "$PWD/installed-memory"
cp -a python/sgl_kernel installed-memory/
export PYTHONPATH="$PWD/installed-memory${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from sgl_kernel import common_ops; print(common_ops.__file__)'
```

Check that the printed extension path belongs to `installed-memory` before
running the tests and benchmark below. Keep this validation package isolated;
it is not a full-feature serving installation or proof that unrelated kernel
families build successfully. If `USE_*` environment variables are set, unset
them or set them consistently because they override the CMake switches.

For B8, ISL4096, OSL1024, TP4/EP1, run on an idle, locked GPU:

```bash
python benchmark/bench_memory_compare.py --mode profiler --samples 101 \
  --lengths 1 1024 4096 --prefix-lengths 0 4096 --require-no-regression
```

Each paired case checks integer outputs exactly (`rtol=atol=0`) and alternates
baseline/candidate launch order. Timings are XPU kernel event durations from
`torch.profiler`, not host wall time. Fused cases compare against the sum of
the two baseline kernel durations. Decode covers sequence lengths 4097,
4098, 4160, 4161, and 5120; writes cover prefill and the first/last decode
token. The regression gate fails if any candidate median exceeds its
baseline median. The sweep synchronizes and cools for two seconds between
cases. Allocation/setup and profiler overhead are outside the device timings.

These small working sets are launch/cache-sensitive, so their effective
bandwidth is not a sustained-memory-bandwidth measurement. Integer metadata
and locations use int64, last locations support int32/int64, and pool entries
use int32; floating-point precision does not apply to these operators.

Accuracy coverage is directly runnable with
`python -m pytest -q tests/test_paged_alloc.py tests/test_write_req_to_token_pool.py`.
It includes empty requests, irregular prefixes and extensions, non-power-of-two
page sizes, untouched output tails, both last-location dtypes, and all 1024
decode steps after the production prefill.

### Validated memory package (2026-10-21)

The isolated native package built successfully with the feature switches above
on CRI Xe35 (`0x674f`), using PyTorch `2.15.0a0+git76ba663`. Its own
`common_ops` and memory libraries supplied all four registered operators; no
diagnostic bridge was loaded. The accuracy suite passed all 77 tests with no
skips. The B8/ISL4096/OSL1024 command above passed all 20 paired cases, with 101
samples per case and zero median regressions against the original Triton path:

| Operator | Cases | Median speedup range |
| --- | ---: | ---: |
| Decode allocation | 5 | 2.70x |
| Request-to-token write | 3 | 2.07–2.80x |
| Extend allocation | 6 | 6.08–6.42x |
| Fused allocation/write | 6 | 5.75–8.64x |

This validates the complete memory family, not a full-feature serving package.
