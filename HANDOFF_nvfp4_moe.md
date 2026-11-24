# HANDOFF — NVFP4 MoE optimization on Xe3P (CRI)

Status as of 2026-11-09. Branch `nvfp4-moe`, HEAD `c96ffba1`. **Nothing is committed.**
All work lives as uncommitted changes in the main checkout. The three sub-agent worktrees
were removed on 2026-11-09; their changes are preserved in the handoff bundle
`/root/hengyume/sgl-deps/handoff_bundle/` (`wt-bf16/`, `wt-mxfp8/`, `wt-decode/` plus `.patch` files) and in
`/root/hengyume/sgl-deps/nvfp4_moe_handoff_bundle.tar.gz`. See §4 and the bundle's README.

**Goal (user):** NVFP4 MoE faster than BF16 / MXFP8 MoE and comparable with MXFP4 MoE,
on either the BF16-activation path or the MXFP8-activation path.

**Short answer:**
- **MXFP8 path: done and merged in the main checkout.**
  - GEMM level: faster than the MXFP8 GEMM on every measured shape except DSv3 T=4096
    down, which is at parity (1.03x).
  - It is 7–30x faster than MXFP4 W4A4.
  - MoE-layer level on Qwen3: 0.02–0.50x the MXFP8 layer time, and below the BF16 layer
    too.
- **BF16 path: mostly done but NOT merged.**
  - The new kernel is in bundle `wt-bf16/` (its worktree was removed).
  - It beats BF16 everywhere on Qwen3.
  - It is still slower than BF16 at large M with long K (DSv3 T>=1024: 1.6–2.4x) and at
    parity on gpt-oss T=4096.

---

## 1. Environment (non-obvious, needed to build/run)

- The machine has 4 *real* Intel CRI GPUs (PCI 8086:674F, `xe` driver). This is NOT the
  simulator that the older `benchmark/PERF_*.md` docs used.
  - 256 EUs / 32 Xe-cores, 160 GB, 384 KB SLM per core, 32 MB LLC.
  - About 1.3 TB/s read bandwidth (measured); bf16 dense GEMM about 273 TFLOPS.
  - MXFP8 grouped GEMM reaches about 435 TFLOPS.
- **The GPUs are shared with other sessions.** Ask before running anything on them, and
  re-measure if numbers look off: the last GEMM sweeps may already include contention.
- Run `source /opt/gfx-deps/env.sh` first (oneAPI + venv, torch 2.15, python 3.12).
- `sgl_kernel` in the venv site-packages is the **pristine baseline build**. Modified
  builds live in private package dirs and are selected with `PYTHONPATH`:
  - `/root/hengyume/sgl-deps/pkg_main` = main checkout
  - (the agent worktrees' `pkg_bf16` / `pkg_mxfp8` / `pkg_decode` were deleted together
    with the worktrees; rebuild from the bundle with `setup_agent_env.sh` if needed)
- CUTLASS source: `/root/hengyume/sgl-deps/cutlass-cri`, a clone of
  `/root/hengyume/libraries.ai.cutlass.internal` at the pinned `08168df2`. The internal
  GitHub repo isn't fetchable from here.
- Full build (about 10 min):
  ```
  CMAKE_BUILD_PARALLEL_LEVEL=200 pip install -v --no-build-isolation . -Cbuild-dir=build \
    -Ccmake.define.DPCPP_SYCL_TARGET=cri \
    -Ccmake.define.FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS-SYCL=/root/hengyume/sgl-deps/cutlass-cri
  ```
  Incremental build in the main tree:
  `cd build && cmake . && ninja sgl-ops-sycl-blockwise_moe_nvfp4_woq common_ops`.
  Then copy `build/src/*.so` (changed ones) plus `python/sgl_kernel/*.py` into
  `pkg_main/sgl_kernel/`.
- Two local build fixes were **required in this environment** (uncommitted):
  - `cmake/BuildFlags.cmake`: add `-std=c++20` to `SYCL_KERNEL_OPTIONS`. The torch 2.15
    headers need C++20 in the device compile.
  - `src/BuildOnLinux.cmake`: add `+SPV_INTEL_float4` to `-spirv-ext`. The NVFP4 kernels
    use `__builtin_spirv_ConvertE2M1To*INTEL`. Using `__bf16` types in device code would
    also need `+SPV_KHR_bfloat16`.
- Helper scripts in `/root/hengyume/sgl-deps/scripts/`:
  - `setup_agent_env.sh <name>`: worktree + build dir + private package. Its
    `rebuild_nvfp4.sh` rebuilds only the NVFP4 .so files, in about 2 min.
  - `dump_isa.sh <worktree> <outdir> [target]` and `loop_stats.py`: IGC ISA dump plus the
    per-loop instruction mix of the (b)dpas mainloop. This is the main tool for this work.
  - `layer_breakdown.py`: per-kernel device time of a fused MoE layer.
  - `test_mxfp8_quant_rows.py`.
- Shared analysis doc given to sub-agents: `/root/hengyume/sgl-deps/AGENT_CONTEXT.md`.
  Logs: `/root/hengyume/sgl-deps/logs/`.

## 2. Benchmarks (new, uncommitted)

- `benchmark/bench_nvfp4_moe_compare.py` times the grouped GEMMs. It reports device time
  from the torch profiler, as "GEMM kernel" and "whole op" columns. Providers:
  - BF16 activations: `bf16`, `mxfp4_w4a16`, `nvfp4_bf16`
  - MXFP8 activations: `mxfp8`, `mxfp4` (W4A4), `nvfp4_mxfp8`
  - Models: qwen3-30b (E128 top8 H2048 I768), gpt-oss-120b, dsv3-ep8 (E32 H7168 I2048),
    qwen3-235b.
  - Per-expert rows are rounded up to 4. The `mxfp4` provider pads to 64 because its
    kernel's device asserts abort the process otherwise; redirect its stderr.
- `benchmark/bench_nvfp4_fused_moe_compare.py` times the whole fused MoE layer: routing
  prep, scatter, quant, both GEMMs, activation and combine. Providers: `bf16`,
  `mxfp4_w4a16`, `nvfp4_bf16`, `mxfp8`, `nvfp4_mxfp8`. The MXFP4 W4A4 fused path is
  excluded because it asserts on ragged M.

## 3. Root cause of the original slowness (key finding)

Baseline, Qwen3 gemm-up (N=1536, K=2048), GEMM device time in µs:

| T | bf16 | MXFP4-W4A16 | NVFP4-BF16 (orig) | MXFP8 | MXFP4 W4A4 | NVFP4-MXFP8 (orig) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 63 | 47 | 226 | 143 | 221 | 1236 |
| 64 | 843 | 552 | 2228 | 849 | 5401 | 14011 |
| 4096 | 1241 | 8426 | 6906 | 883 | 8656 | 20998 |

**Register layout.**
- CuTe's 8/16-bit register fragments are SoA: logical (work-item t, value v) is physical
  byte/half `16*v + t`.
- The DPAS hardware gives B column n to lane n, with K in order (VNNI).
- So any SIMT code that writes the B operand byte-by-byte through CuTe fragments forces
  cross-lane movement. The ISA showed:
  - MXFP8 path: about 6100 instructions per k-tile (4047 `mov`, 64 `goto`/`join`) for
    64 `bdpas`;
  - BF16 path: about 1277 instructions for 32 `dpas`.

**Fix used by both new kernels.**
- A *transposed d32* block-2D load of the packed (N, K/2) weights puts row n in lane n,
  with dword j = nibbles k = 8j..8j+7.
- Building the DPAS/BDPAS B operand as **dword** arrays (work-item == hardware lane) and
  calling (b)dpas on raw registers via inline vISA makes the conversion fully lane-local.

**Other measured facts.**
- `__builtin_spirv_ConvertE2M1To{FP16,BF16}INTEL` = one `shfl.idx4` per packed byte
  (16-entry LUT, gives 2 values).
- `ClampConvertFP16ToE4M3INTEL` adds a cmp/bfn/add inf fixup per 16 values; raw
  `mov (sat) :hf8` / `fcvt` avoids it.
- C++ `half2` math is not packed; use vISA `mul (M1_NM, 32)`.
- `mov :hf <- :bf` / `:bf <- :hf` assembles but **produces zeros** on Xe3P.
- The CUTLASS group tile scheduler costs about 70 µs per launch at E=128 decode.
- The budget: conversion must stay under about 3.5 ALU ops per weight element even in
  the memory-bound regime (1.3 TB/s ≈ 34 elements/cycle/Xe-core versus 128 lane-ops per
  cycle).

## 4. Where everything is

### 4a. Main checkout `/root/hengyume/sgl-kernel-xpu` (package `pkg_main`)

127 tests pass (`LONG_TESTS=1`): `test_nvfp4_woq_moe.py`, `test_mxfp8_quant_rows.py`,
`test_mxfp8_blockwise_moe.py`. `test_es_mxfp8_blockscaled_moe.py` hits an ImportError at
collection, so that file ran nothing; it isn't related to these changes.

Contents:
- **Build fixes** (§1).
- **MXFP8-path NVFP4 kernel, v1 from the sub-agent, merged.** Files:
  `src/sycl/xe35/nvfp4_woq_mxfp8_mainloop.hpp`, `xe_subbyte_upconvert.hpp`, the MXFP8
  entry in `blockwise_moe_nvfp4_woq.cpp`, the `python/sgl_kernel/nvfp4.py` docstring, and
  26 new tests in `tests/test_nvfp4_woq_moe.py`. A copy of v1 is in
  `/root/hengyume/sgl-deps/merge_base/mxfp8_v1/` as the base for 3-way merges.
  - Standalone persistent SYCL kernel; it no longer uses CUTLASS
    CollectiveMma/GemmUniversal/group scheduler. The op is one kernel launch: the prep
    kernel and the A-scale transpose are gone, and the expert is selected by surface y.
  - Conversion per input dword (8 nibbles), inline vISA: 4 `shfl.idx4` + 4 packed fp16
    `mul (32)` + 4 `mov (32) :hf8` with dst stride 4. That's about 300 instructions per
    64-k step for 64 bdpas, with no spills and no branches.
  - c and sfmx come via block-2D loads (OOB rows zero-filled); A scales come via a 2-byte
    gather.
  - Tile menu picked host-side from rows/E and K, with no D2H sync. Override with
    `SGLANG_NVFP4_WOQ_MXFP8_TILE`:
    - m8/m16/m32: 4 subgroups, subgroup tile SG_M x 16, 128-k steps.
    - m128/m256: 4x4 subgroups sharing converted B through double-buffered SLM, with
      split barriers.
  - Bit-exact against host requantization: w = e4m3_rne(bf16 c * e2m1), bdpas B
    scale = sfmx.
  - `workspace` is now unused. `SGLANG_NVFP4_WOQ_QUANTIZE_IN_K_LOOP` is parsed but selects
    the same kernel.
- **New op `mxfp8_quant_rows(input, out_q, out_s, fuse_silu_and_mul, src2dst=None)`.**
  Files: `src/sycl/xe35/mxfp8_quant_rows.cpp`, declaration in `include/sgl_kernel_ops.h`,
  registration in `src/torch_extension_sycl.cc` (Xe35 only), tests in
  `tests/test_mxfp8_quant_rows.py`.
  - Row-major MXFP8 quant: group 32, UE8M0, 4 lanes per group.
  - Plain mode is **bit-identical** to `sgl_per_token_group_quant_8bit` and 4–5x faster.
    The old kernel idled 12 of 16 lanes for group 32. Single-byte scale stores were the
    remaining bottleneck, fixed by packing them per subgroup.
  - `fuse_silu_and_mul` reads the fp32 or bf16 (M, 2N) GEMM output directly. Its SiLU is
    more accurate than `silu_and_mul`, which is off by ≥1 bf16 ulp on 27% of outputs
    versus float64.
  - `src2dst` quantizes each token once and scatters it to its top-k rows.
- **`python/sgl_kernel/cutlass_moe.py` (fused paths):**
  - `_per_token_group_quant_mxfp8` uses the new op.
  - New helpers `_scatter_quant_mxfp8` and `_silu_and_mul_quant_mxfp8`.
  - The MXFP8 baseline (`cutlass_fused_experts_fp8`, `use_mxfp8=True`) and the NVFP4 path
    both use them, so the comparison stays fair.
  - `c1`/`c2` are allocated with `torch.empty` (no zero-fill; beta = 0 and every row
    belongs to one expert) on the MXFP8 and NVFP4 paths.
  - Effect: the MXFP8 layer at T=4096 went from 6473 to 4234 µs.
- **The BF16-path NVFP4 kernel in the main checkout is still the original slow one.**

### 4b. BF16 kernel — bundle `wt-bf16/` (worktree and its build removed), NOT merged

The agent's own `test_nvfp4_woq_moe.py` (37 tests) passed before the worktree was removed. The agent was adding more
tile-path tests when it stopped on an API auth error.

Files: `src/sycl/xe35/nvfp4_woq_bf16_mainloop.hpp` (rewritten), the BF16 entry/tile menu
in `blockwise_moe_nvfp4_woq.cpp`, and `tests/test_nvfp4_woq_moe.py`.

Design (see the file header):
- Plain SYCL kernel with raw registers and the transposed d32 B load.
- Per B dword: `shfl.idx4` E2M1→FP16 pair, then `mul (32)` by c as fp16 (exact), then an
  integer FP16→BF16 bit trick. The bit trick yields value·2^-112; that factor and 2^kexp
  are applied for free by **block-scaled `bdpas.bf.bf`** (B scale = sfmx, A scale = 2^112).
- About 40 SIMD instructions per 16x16 atom, shared by SG_M/8 DPAS.
- A small metadata kernel builds an m-tile → expert table.
- Tiles: M8N128, M8N256, M16N256, M64N256, M128N256. Override with
  `SGLANG_NVFP4_BF16_TILE=<idx>`.
- fp32 output via block-2D stores.

GEMM device time, NVFP4-BF16 / bf16 ratio (and / MXFP4-W4A16 ratio):

| T | qwen3 up | qwen3 down | gpt-oss up | gpt-oss down | dsv3 up | dsv3 down |
|---:|---|---|---|---|---|---|
| 1 | 0.23 (0.31) | 0.30 (0.24) | 0.23 (0.32) | 0.24 (0.34) | 0.87 (1.67) | 0.33 (0.48) |
| 16 | 0.30 (0.48) | 0.33 (0.44) | 0.81 (1.45) | 0.72 (1.28) | 0.53 (1.02) | 0.60 (1.01) |
| 64 | 0.30 (0.46) | 0.31 (0.43) | 0.76 (1.37) | 0.77 (1.35) | 0.77 (0.14) | 0.85 (0.28) |
| 256 | 0.28 (0.09) | 0.35 (0.10) | 0.78 (0.27) | 0.80 (0.27) | 0.75 (0.14) | 0.73 (0.25) |
| 1024 | 0.37 (0.13) | 0.47 (0.15) | 0.74 (0.26) | 0.72 (0.25) | **1.56** (0.28) | **2.43** (0.38) |
| 4096 | 0.70 (0.11) | 0.75 (0.12) | 1.05 (0.19) | 1.01 (0.18) | **1.79** (0.27) | **2.02** (0.29) |

Absolute examples: qwen3 up T=4096 is 882 µs (bf16 1256); dsv3 up T=4096 is 15219 µs
(bf16 8484). Raw data: `logs/bf16wt_*.txt`.

Weak spots:
- Large M with long K (dsv3). The tile selection at `rows_per_group > 96` picks M64N256,
  and `k <= 4096` matters too. That code is the first place to look.
- gpt-oss T=16–64, where MXFP4-W4A16 is faster.
- dsv3 decode up (K=7168).

### 4c. MXFP8 follow-up — bundle `wt-mxfp8/` (worktree and its build removed): v1 plus unfinished follow-up

The worktree's own `test_nvfp4_woq_moe.py` (76 tests) passed before the worktree was removed.

- The follow-up was asked to (1) let `nvfp4_woq_mxfp8_scaled_grouped_mm` emit **bf16
  output** and (2) find a margin on the large-M down GEMMs.
- It stopped on the API auth error while evaluating extra tiles m512 (512x256, 8x4 of
  64x64) and m256s (256x256, 8x4 of 32x64).
- bf16 output is **not** done.
- Diff versus v1: about 219 changed lines in the mainloop, 23 in upconvert, 8 in the entry,
  41 in tests.
- Its T=1024/4096 numbers vs v1:
  - qwen3 down T=4096: 474 µs (v1 551, MXFP8 564);
  - qwen3 up T=4096: 693 µs (v1 716);
  - dsv3 down T=4096: 2490 µs (MXFP8 2471), still parity.
  - These came from a possibly contended run.
- Review before merging. Merge = 3-way against `merge_base/mxfp8_v1` (main has not
  changed those files since).

### 4d. Decode kernel — bundle `wt-decode/` (worktree and its build removed; last build was broken): incomplete

- New `src/sycl/xe35/nvfp4_woq_bf16_small_m.hpp` (661 lines), a dedicated decode kernel:
  - unscaled B, scale applied on the accumulator side;
  - persistent, with SLM expert compaction and in-WG K-split reduced through SLM (no
    atomics);
  - software-pipelined 2D loads.
- Plus a dispatch hook at the top of `nvfp4_woq_bf16_grouped_mm`.
- The current package fails with "The program was built for 1 devices". The agent died
  mid-rewrite.
- The BF16 kernel in 4b already covers decode well except for long K (dsv3 T=1 up 0.87x
  bf16, 1.67x MXFP4-W4A16), so this is optional. Either finish it for long-K decode or
  drop it.

## 5. Verified results: MXFP8 path (main checkout)

GEMM device time in µs: NVFP4-MXFP8 / MXFP8 GEMM / MXFP4 W4A4 GEMM. The MXFP8 *op* adds
about 0.5–4.5 ms of `TransposeScalesU8B` per call; the NVFP4 op = its GEMM.

| model, gemm | T=1 | T=16 | T=64 | T=256 | T=1024 | T=4096 |
|---|---|---|---|---|---|---|
| qwen3 up | 12.8/144/221 | 166/603/2452 | 255/849/5462 | 270/847/5327 | 354/812/5648 | 716/864/8579 |
| qwen3 down | 8.3/104/148 | 98/357/876 | 151/491/1311 | 164/494/1333 | 248/488/1515 | 551/567/3596 |
| gpt-oss up | 40/231/478 | 569/1806/10967 | 1198/3779/36834 | 1519/4255/46525 | 1705/4507/48372 | 2458/4136/57113 |
| gpt-oss down | 17/195/272 | 283/1137/2730 | 611/2217/13281 | 730/2574/18692 | 847/2588/18788 | 1268/2454/23777 |
| dsv3 up | 164/468/1289 | 717/1796/18867 | 617/1782/18661 | 764/1708/19629 | 1360/1753/27083 | 3780/4613/82414 |
| dsv3 down | 78/252/767 | 285/939/7199 | 299/921/6985 | 403/901/7960 | 821/937/8914 | 2467/2397/38361 |

Fused MoE layer (qwen3-30b, all kernels, µs): NVFP4-MXFP8 versus MXFP8 layer
(BF16 layer for reference):

| T | 1 | 16 | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|---|---|
| nvfp4_mxfp8 | 38.8 | 283 | 435 | 543 | 826 | 2100 |
| mxfp8 | 1793 | 2564 | 2978 | 3016 | 3214 | 4187 |
| bf16 (earlier run) | 110 | 840 | 1278 | 1286 | 1481 | 2317 |

## 6. Next steps (priority order)

1. **Merge the BF16 kernel (4b) into main.**
   - Both agents edited `blockwise_moe_nvfp4_woq.cpp` and `tests/test_nvfp4_woq_moe.py`,
     in disjoint sections (MXFP8 entry vs BF16 entry/types). Use
     `git apply -3 /root/hengyume/sgl-deps/handoff_bundle/wt-bf16.patch` on the main tree, or a manual merge. Then rebuild
     `pkg_main` and run all tests.
2. **Fix BF16-path large-M/long-K** (dsv3 T≥1024, gpt-oss T=4096). Options:
   - re-tune tile selection; M64N256 is chosen above 96 rows/expert;
   - share converted B across subgroups through SLM, as the MXFP8 kernel does for
     m128/m256;
   - larger SG_M at 512 GRF.
3. **bf16 GEMM outputs for both NVFP4 ops**, then switch `cutlass_fused_experts_nvfp4_woq`
   to bf16 `c1`/`c2`. This removes the fp32→bf16 `.to()` copy (about 380 µs per layer at
   T=4096 qwen3) and halves output writes.
   - `mxfp8_quant_rows(..., fuse=True)` already accepts bf16 input.
   - The BF16 path could also use the fused-silu idea: gate/up interleaving at prep, plus
     a SiLU epilogue.
4. Finish or discard the MXFP8 follow-up (4c) and the decode kernel (4d).
5. Run the layer benchmark for the BF16 path after the merge, for all models.
6. Optional, from the agents' notes:
   - read raw E4M3 block scales instead of bf16 `c` (0.0625 vs 0.125 B/elem);
   - store `c` as fp16 at prep (saves about 6 instructions per atom);
   - k-tile-major repack of B/c for long-K decode (gpt-oss/dsv3 decode only reaches about
     0.9–1.0 TB/s);
   - compile small tiles in a 256-GRF TU (more threads);
   - the 64-byte `sfmx` stride check is now only kept for an existing test.
7. Commit when the user asks; that hasn't happened yet. Keep the two build fixes separate
   from the kernel work.
