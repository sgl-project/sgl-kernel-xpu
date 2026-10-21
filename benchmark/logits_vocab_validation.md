# Xe35 logits/vocab validation (2026-10-21)

## Status

- Native `SoftcapLogits` and `VocabMask` CMake targets compile successfully.
- Accuracy: **130 passed**, no skips, including FP32/BF16/FP16, int32/int64
  IDs, noncontiguous logits, saturation, shard boundaries, vector tails,
  output-buffer canaries, the `[50512, 8192]` embedding shard, both launch
  layouts, three-tile/four-tile indexing, and the strict performance gate.
- The earlier GPU 1 native A/B is **superseded**: baseline and candidate had
  identical SYCL kernel IDs. ID-isolated GPU 5 measurements exposed embedding
  regressions, addressed by specializing away the vector-tail check for full tiles.
  Vector tails still retain the check; scalar paths retain per-element bounds.
- The final native GPU 5 A/B crosses launch order and output-buffer assignment.
  Softcap and mask medians match the rebuilt baseline. Embedding improves
  0.57%–28.60% for production dimensions and 18.16%–19.98% for historical
  dimensions. Every measured median passes the strict no-regression gate.
  Installed-package confirmation remains required.
- **Full default-feature package build and isolated installation passed.**
  `USE_MOE`, `USE_FMHA`, `USE_GDN`, `USE_MLA`, and `USE_QSA` are all ON;
  `USE_SYCL_JIT` is OFF. Installed `common_ops`, `SoftcapLogits`, and `VocabMask`
  hashes match the build/install log. CPU-only normal package import and all
  three operator schemas pass (exit 0), without a candidate registration bridge.
- **Installed-package GPU accuracy/performance are not yet complete.**
  The GPU 5 preflight is stuck inside `query_device(0)` during GPU VM creation,
  not package import or a logits/vocab kernel launch. The Python process is in
  uninterruptible `D` state at `drm_pagemap_acquire_owner`. Other unrelated
  VLLM/Python tasks and an `xe-destroy-wq` worker are also blocked in the same
  driver's acquire/release-owner paths. No shared GPU reset or host restart
  has been attempted. A user-authorized GPU 0 retry reproduced the same
  ownership wait before any accuracy or performance test could start.
  See the driver-blocker evidence below.
- Build history:
  The GCC-host parallel build hit OOM at 12 and then 6 jobs. The single-job
  GCC-host build was intentionally interrupted (exit 130) after a successful
  IntelLLVM-host probe compiled RMSNorm and both new targets. The successful
  full build finished using sixteen jobs and a 12 GiB/job estimate.
- The bounded 48-hour runner completed source verification, incremental build,
  and isolated installation; it remains live at the blocked GPU preflight.
  Its final status is not yet available, so the pipeline as a whole is not passed.
- Concurrency adjustment: the four-job process tree measured 6.35 GiB RSS,
  with 40.88 GB total container usage against a 256 GiB limit and 689 GiB
  host memory available. The owned supervisor and its children were cleanly
  stopped; completed objects and all feature settings were retained. The new
  supervisor still caps concurrency by current headroom and retries once after
  OOM. No compiler flags, kernel source, or benchmark settings changed here.
- A follow-up measurement during high-dimensional FMHA compilation found
  15.64 GiB RSS across eight jobs, with an observed live compiler peak of
  4.94 GiB. The final build used sixteen jobs at 12 GiB/job (over twice that
  observed peak), preserving the prior 192 GiB estimated compile budget.
  This is an incremental resource adjustment, not a feature reduction or clean.

## Driver blocker

CPU-only installed-package verification is recorded in
`/tmp/muse-logits-installed-cpu-registration-20261021.log` (exit 0).
The selected process environment is `ZE_AFFINITY_MASK=5`,
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, with `PYTHONPATH` pointing at the
isolated installed prefix and a single loaded `libsycl.so.9` runtime.

The stuck preflight is host PID 1651883 / container PID 153943. Its Python
stack is `torch/_ops.py:1350` called from stdin line 6 (`query_device(0)`).
The kernel stack is `drm_pagemap_acquire_owner -> xe_svm_init -> xe_vm_create
-> xe_vm_create_ioctl`. An ownership-checked, nonblocking host-side diagnostic
is recorded in `/tmp/muse-logits-import-host-diagnostic-20261021.log` (exit 0).
Container-only stack inspection lacks ptrace permission; those failed attempts
are diagnostic failures, not package compilation or numerical failures.

Several unrelated processes and an `xe-destroy-wq` worker are blocked in
`drm_pagemap_release_owner`. Host logs also contain an
earlier OOM-killed VLLM worker, but causality between that OOM and the driver
blockage has not been established. GPU-driver recovery is required before the
remaining installed-package gates can be verified safely.

At the user's request, a separate installed-package validation selected GPU 0
(PCI `0000:59:00.0`) through `enter-sglang-syk.sh --gpu-lock --gpu 0`, leaving
the original GPU 5 pipeline unchanged. GPU 0 had no test-lock owner; its device
file holders were diagnostic utilities, not observed compute workloads.
Source and installed-binary SHA256 manifests passed before the retry. The
600-second bounded command logs to
`/tmp/muse-logits-installed-gpu0-20261021.log`. It acquired GPU 0's lock, imported
the isolated installed package, printed `GPU_AFFINITY 0`, and then blocked at
`query_device(0)` with the same `drm_pagemap_acquire_owner -> xe_svm_init ->
xe_vm_create -> xe_vm_create_ioctl` kernel stack (host PID 1675929). No accuracy
cases or benchmark shapes ran. The runner expired with exit 124. The surviving
container workload was checked; SIGTERM and then SIGKILL were sent only to this
owned preflight, and SIGTERM to its validation shell to prevent later tests.
The Python process remains in uninterruptible `D` state: signal handling cannot
finish until the driver wait returns.
This retry does not constitute a numerical failure or a performance regression,
and further GPU retries are not useful while this shared driver wait persists.

Installed binary hashes are retained in
`logits-installed-artifacts-final-20261021.sha256`. The existing live pipeline
must not be duplicated while the blocked workload persists.

After driver recovery, if the original pipeline no longer exists, use
`~/sgl-kernel-xpu-up/.validation/resume_logits_gpu_validation.sh` for GPU-only
acceptance without recompilation. It refuses an existing original runner or
preflight (PID and process-start identity), refuses remaining driver ownership
waits, verifies source and installed-binary hashes, and uses the GPU-lock wrapper
under a 900-second long-command deadline. Its duplicate-run guard was exercised
and returned exit 2 without launching a GPU workload. Accuracy must contain
exactly 130 cases with no skips/errors/failures; both A/B suites retain their
strict no-latency-regression gates. Do not use this helper to bypass live work.

## Baseline and method

The old implementation is preserved on `muse_logits_vocab`, revision
`448f0c0cf6cd13de7c2d6ec28de1d392039fdaca`. The current default checkout of
`~/sgl-kernel-xpu` no longer contains these operators. Baseline kernel sources
were extracted from that revision and rebuilt with the same current DPC++
device compiler/runtime as the candidate. Baseline kernel bodies are unchanged;
the sole source edit places them in `at::native::xpu::logits_vocab_baseline` to
prevent SYCL kernel-ID collisions. Baseline library names and the registration
namespace also differ. Baseline host compilation uses GCC; the current candidate
uses IntelLLVM host compilation. DSO kernel-ID sets were checked to be disjoint
(the initial namespace-isolation audit predates the new layout variants).

The benchmark additionally validates each profiled kernel name against its
expected baseline/candidate position and refuses shared IDs or wrong ordering.
Regression tests cover rejection of shared IDs and both alternating orders.

Device: CRI PCI `8086:674f`, driver `1.18.40430`. GPU 1 belongs to the
superseded measurements. Final native rerun: physical GPU 5, PCI `0000:71:00.0`,
confirmed free of GPU processes in the second GPU inventory before selection.
Compiler: DPC++ `20260300`, CRI AOT. PyTorch: `2.15.0a0+git76ba663`.
Current GPU execution uses the CRI `enter-sglang-syk.sh --gpu-lock --gpu 5`, initial five-second
cooldown, and synchronization plus two-second cooldown between shapes.

`bench_logits_vocab_ab.py` calls both registered native operators directly,
validates against CPU math/index mapping, interleaves their calls on one stream,
and reverses order every round. Timing is XPU kernel duration from
`torch.profiler`, not host wall time; each call must produce exactly one kernel.
There are sixteen rounds, 100 samples per implementation per round, and ten
warmup calls per round. Raw samples and per-round ratios are retained in logs.
Embedding output buffers swap every two rounds, independently of alternating
launch order; each of the four order/buffer combinations appears four times. This
keeps the two-output working set while avoiding a fixed allocation-address bias.
Every shape reports `performance_gate`; a candidate median greater than its
baseline median fails the command after recording all shapes in that suite.
There is no allowed latency-regression margin.
In-place softcap repeatedly transforms the same tensors, matching the historical
benchmark. These are operator timings, not an E2E throughput claim.

## Historical tensor shapes

The rebuilt-old and candidate columns below are ID-isolated, buffer-crossover
GPU 5 native measurements, not yet installed-package completion evidence.
Archived-old measurements remain historical evidence.

Embedding uses the original 528-row BF16 table, hidden size 4096, token pattern,
original/added windows, and masking. Serving batch remains 8; token counts are
kernel dimensions, not alternate serving batches.

| Operator | Shape | Archived old median (us) | Rebuilt old median (us) | Candidate median (us) |
| --- | --- | ---: | ---: | ---: |
| Softcap FP32 | `[8,196608]` | 7.500 | 7.291 | 7.291 |
| Softcap FP32 | `[8,202048]` | 7.916 | 7.812 | 7.812 |
| Softcap FP32 | `[8,208896]` | 8.229 | 7.916 | 7.916 |
| Vocab mask int64 | `[8]` | 0.625 | 0.625 | 0.625 |
| Vocab mask int64 | `[32]` | 0.625 | 0.625 | 0.625 |
| Vocab mask int64 | `[32768]` | 0.937 | 0.833 | 0.833 |
| Embedding BF16 | `[8,4096]` | 1.250 | 1.041 | 0.833 |
| Embedding BF16 | `[32,4096]` | 1.354 | 1.145 | 0.937 |
| Embedding BF16 | `[32768,4096]` | 370.937 | 363.645 | 295.729 |

Archived results reside under
`~/worktrees/sgl-kernel-xpu/review-validation-artifacts/logits-vocab-20261015/`:
`gpu7-softcap-rename-perf.jsonl`, `paired-vocab-before.jsonl`, and
`paired-embedding-before.jsonl`. Archived measurements used PyTorch 2.14 on
GPU 7, so the rebuilt same-device A/B is the stronger comparison.

The large historical embedding latency improves 18.68%; the round-level 95% t
interval for the latency ratio is `[0.812144, 0.814495]`, entirely below equality.

## Current production dimensions

B8/ISL4096/OSL1024/TP4/EP1; embedding hidden size 8192, local table
`[50512,8192]`. Production-layout IDs span the full vocabulary, including
out-of-shard tokens. The large case has a 789 MiB weight table and 512 MiB output
per implementation, unlike the small historical table. No sustained-bandwidth
claim is made for the cache-resident decode cases.

| Operator | Shape | Rebuilt old median (us) | Candidate median (us) |
| --- | --- | ---: | ---: |
| Softcap FP32 | `[8,202048]` | 7.812 | 7.812 |
| Softcap FP32 | `[8,50512]` | 2.812 | 2.812 |
| Vocab mask int64 | `[8]` | 0.625 | 0.625 |
| Vocab mask int64 | `[32]` | 0.625 | 0.625 |
| Vocab mask int64 | `[32768]` | 0.833 | 0.833 |
| Embedding BF16 | `[8,8192]` | 1.041 | 0.885 |
| Embedding BF16 | `[32,8192]` | 1.458 | 1.041 |
| Embedding BF16 | `[32768,8192]` | 852.708 | 847.864 |

The large production embedding latency improves 0.57%; the round-level 95% t
interval is `[0.993633, 0.995228]`, entirely below equality. The previous
buffer-crossover and intermediate optimization measurements are superseded.

## Embedding layout tuning

Embedding is memory-dominated: a BF16 vector transfers 16–32 bytes per load
and store while performing only integer row/mask calculations (well below one
operation per byte). The large production case has 669 MB of logical traffic
and measures approximately 789 GB/s; the small historical weight table is
cache-resident, so its logical throughput is not sustained external bandwidth.
These are measurements on this CRI/driver stack, not B60 or older CRI ceilings.

The shared kernel now parameterizes launch dimensionality and vector byte width.
Two-dimensional groups eliminate per-thread quotient/remainder index mapping,
and 32-byte vectors reduce group count for small/cache-heavy cases. Measured
decode improvements are 10%–29%. For more than 256 IDs and a table larger than
64 MiB, the kernel retains the original one-dimensional, 16-byte BF16/FP16
mapping; FP32 retains 32-byte vectors. This preserves the production large-table
bandwidth behavior. All dimensions remain runtime parameters; this is not a
hard-coded Muse shape lookup or a duplicate kernel body.

Rejected intermediate changes: applying 2-D mapping to all cases regressed the
large production embedding by 1.6%; using 32-byte vectors there regressed it by
8.9%; replacing legacy quotient/remainder with shifts regressed it by 1.2%.
The final large-table path restores the baseline mapping and row guard.
Accuracy tests cover both layouts, paired dtypes, non-power-of-two group counts,
partial vector tiles, scalar fallback, and output canaries.

## Local evidence and rerun

The container-visible validation root is
`/workspace/sgl-kernel-xpu-up/.validation/`; the host root is
`~/sgl-kernel-xpu-up/.validation/`. Candidate snapshot:
`muse-logits-vocab-xe35`; baseline snapshot: `logits-baseline-448f0c0`.
Temporary registration bridges allow targeted testing while the full package
build is incomplete. The final native measurements use `muse-logits-vocab-tuning`
and `logits_vocab_tuning_bridge.so`, whose source matches the final worktree kernel.
They are not substitutes for the pending package gate.

Logs:

- `/tmp/muse-logits-targets-20261021.log`: target compilation.
- `/tmp/muse-logits-accuracy-final-gpu1-20261021.log`: 85 accuracy/scope tests.
- `/tmp/muse-logits-ab-gpu1-historical-20261021.log`: historical-shape A/B.
- `/tmp/muse-logits-ab-gpu1-8192-20261021.log`: production-dimension A/B.
- `/tmp/muse-logits-full-build-20261021.log`: parallel-build OOM.
- `/tmp/muse-logits-full-build-serial-20261021.log`: single-job package build.
- `/tmp/muse-logits-gcc-stop-20261021.log`: intentional GCC-build interruption.
- `/tmp/muse-logits-intelllvm-probe-20261021.log`: successful host-compiler probe.
- `/tmp/muse-logits-full-build-intelllvm-20261021.log`: current all-features build.
- `/tmp/muse-logits-intelllvm-accuracy-gpu4-20261021.log`: 85 passing native tests.
- `/tmp/muse-logits-gpu-inventory-20261021.log`: GPU process inventory.
- `/tmp/muse-logits-baseline-isolated-kernel-id-20261021.log`: isolated baseline build.
- `/tmp/muse-logits-isolated-ab-gpu4-20261021.log`: lock timeout (exit 124).
- `/tmp/muse-logits-isolated-ab-gpu5-20261021.log`: isolated-ID A/B before tail fix.
- `/tmp/muse-logits-tail-specialization-recheck-20261021.log`: 88 passing tests
  and tail-fixed A/B with fixed output-buffer assignment (exit 0).
- `/tmp/muse-logits-buffer-crossover-20261021.log`: 11 benchmark-scope tests
  and both buffer-crossover A/B suites (exit 0); superseded pre-tuning results.
- `/tmp/muse-logits-adaptive-legacy-validation-20261021.log`: 119 passing tests
  and two sixteen-round A/B suites, confirming the final kernel (exit 0).
- `/tmp/muse-logits-adaptive-gated-validation-20261021.log`: 130 passing tests
  and both strict no-regression A/B suites (exit 0); source of the updated tables.
- `/tmp/muse-logits-installed-final-20261021.log`: original waiter intentionally
  stopped (exit 143) to fix redundant source copying; compilation was not stopped.
- `/tmp/muse-logits-installed-gated-final-20261021.log`: waiter intentionally
  stopped (exit 143) before the measured concurrency adjustment.
- `/tmp/muse-logits-parallelism-transition-20261021.log`: ownership-checked
  stop of the four-job supervisor and Ninja, retaining completed objects (exit 0).
- `/tmp/muse-logits-installed-eight-job-final-20261021.log`: eight-job build
  intentionally interrupted (runner exit 143) for the measured resource adjustment.
- `/tmp/muse-logits-sixteen-job-transition-20261021.log`: ownership-checked
  stop of the eight-job supervisor and Ninja, retaining completed objects (exit 0).
- `/tmp/muse-logits-installed-sixteen-job-final-20261021.log`: current bounded
  incremental full build, isolated installation, and normal-import validation
  with the strict performance gate.

The final package runner uses
`~/sgl-kernel-xpu-up/.validation/finish_logits_validation.sh`. Its expected
worktree hashes are in `logits-final-source-20261021.sha256`; any intervening
source change fails the manifest check instead of validating stale code.
Unchanged files are not copied, preserving dependency timestamps and avoiding
an unnecessary all-target rebuild. Native DSO and bridge hashes are retained
in `logits-native-artifacts-final-20261021.sha256`.
The installed prefix will be `.validation/muse-logits-vocab-xe35/installed`.

After building/installing the candidate package, run the A/B script through
the container wrapper and mandatory long-command runner, with the preserved
baseline bridge:

```bash
python benchmark/bench_logits_vocab_ab.py \
  --baseline-library /workspace/sgl-kernel-xpu-up/.validation/logits_vocab_baseline_bridge.so

python benchmark/bench_logits_vocab_ab.py \
  --baseline-library /workspace/sgl-kernel-xpu-up/.validation/logits_vocab_baseline_bridge.so \
  --hidden 4096 --embedding-layout historical \
  --softcap-widths 196608 202048 208896
```

Pending: recover the shared driver blockage, then complete accuracy and both
strict A/B suites using the already installed package's normal `sgl_kernel`
import and registrations. Compilation, isolated installation, CPU registration,
and the earlier native-target accuracy/performance gates are already verified.
