# Xe35 BF16 GEMM validation

Run the following payloads in a compatible XPU container through its GPU-locked
entry point. Use the same device, compiler, driver, and PyTorch for both branches.
Build the candidate's default feature set and import its isolated installed
package before final acceptance; testing a standalone GEMM library does not
verify `common_ops` integration.

## Accuracy

```bash
python -m pytest -q tests/test_bf16_gemm_nt_xpu.py tests/test_hc_head.py
```

GEMM tests compare against CPU FP32 accumulation with `rtol=5e-3`, `atol=5e-3`.
They cover production projections, vocabulary weights, prefill, tile boundaries,
logical output tails, and FP16 fallback. The custom kernel itself is BF16-only.
Copy weights to CPU **before** FP32 conversion to avoid unnecessary GPU storage.
The FP32 reference transfers weights in at most 64 MiB row chunks, preserving
the full inner-dimension accumulation and every output column without requiring
a multi-GiB transfer allocation. Accuracy tolerances are unchanged.
The B8 × ISL4096 prefill projection compares all 32768 output rows, not a sample.

## Baseline bridge

Keep a frozen copy of the baseline checkout, including its original build flags
and CUTLASS dependency. Build its `sgl-ops-sycl-bf16_gemm_nt` target. Give that
target a distinct library name in the validation copy's CMake configuration:

```cmake
set_target_properties(sgl-ops-sycl-bf16_gemm_nt PROPERTIES OUTPUT_NAME sgl-bf16-baseline)
```

Compile `bf16_gemm_baseline.cpp` as a shared library using PyTorch's include paths,
C++ ABI setting, and libraries. Link it against `torch`, `torch_cpu`, `c10`,
and `dl`. The bridge binds the supplied kernel library using `dlopen`/`dlsym`
and checks the symbol's image with `dladdr`, so a preloaded compatibility shim
cannot intercept the baseline function. The benchmark supplies the absolute
kernel path through `SGL_BF16_BASELINE_KERNEL_LIBRARY`.
The distinct SONAME prevents the loader from substituting the candidate library.
It does not isolate SYCL kernel identifiers. In the validation copy of the old
GEMM source, define `cutlass` as `sgl_bf16_baseline_cutlass` immediately before
the first CuTe/CUTLASS include, after the PyTorch headers. Rebuild without changing
kernel arithmetic or launch parameters. This renames CUTLASS kernel types so the
SYCL runtime cannot coalesce the old and new device images. Paired timing rejects
shared GEMM kernel identifiers and requires both specified libraries to be loaded.
It also rejects an installed older BF16 GEMM library sharing candidate identifiers;
use the isolated candidate package, not an older package plus a kernel bridge.
Earlier measurements made with shared identifiers are not acceptance evidence.
Record the baseline revision and source/library hashes with the results.

## Paired timing

```bash
python benchmark/bench_bf16_gemm_nt.py \
  --compare-baseline --baseline-source /path/to/frozen-baseline \
  --baseline-library /path/to/bf16_gemm_baseline.so \
  --baseline-kernel-library /path/to/libsgl-bf16-baseline.so \
  --candidate-kernel-library /path/to/libsgl-ops-sycl-bf16_gemm_nt.so \
  --shape 4608 8192 --shape 4096 8192 --shape 8192 4096 \
  --shape 18432 8192 --shape 8192 8192 \
  --m 8 9 16 17 32 33 --warmup 5 --repeats 30 --calls 16 \
  --cache-mode cold --output paired.csv
```

Also run hot-cache timing and the `202048 8192` vocabulary projection, including
dispatch boundaries. Timing uses XPU graph replay and device events, excludes
weight packing, and alternates execution order. Graph resources and private
allocator pools are released after each case, including partially failed
captures, outside the timing interval. Same-layout packed operands are checked
exactly and share storage to avoid address-dependent comparison bias.
Cold timing rotates nontrivial weights totaling at least 512 MiB per layout;
`--calls` must cover the entire ring. CSV includes medians, sample spans, and
working-set metadata. Serving batch is fixed at eight; `--m` specifies GEMM rows.
Packing and other setup costs must be accounted for separately in end-to-end work.
For B8 × ISL4096 prefill, compare the TP4 projection shapes at `--m 32768` in
dispatch mode with `--calls 8`, in both hot and cold modes. The vocabulary
projection remains at the selected-token row counts rather than all prompt rows.

The isolated hot/cold measurements for the packed `18432 × 8192` projection
favor CuTe at rows 9–32 as well as the neighboring regions: approximately
0.34–0.38 ms versus 0.42–0.44 ms for the old dispatcher. These cases read a
302 MB weight at approximately 790–895 GB/s and have 9–32 FLOP/byte, so the
small-row region remains dominated by weight traffic. The candidate selects
CuTe through row 96; accuracy covers the new region and the 96/97 boundary.
For native `8192 × 4096` weights, CuTe wins at rows 65–128 (approximately
0.072 ms versus 0.091 ms at rows 96/97 on GPU 2). Boundary sweeps retain the
torch path through row 64 and from row 129; precision covers both transitions.
