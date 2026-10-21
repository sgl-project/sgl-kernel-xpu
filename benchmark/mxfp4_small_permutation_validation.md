# Xe35 small permutation validation (2026-10-21)

## Scope and provenance

- Candidate: this worktree's `mxfp4_small_permutation_xe35`, called through its
  registered XPU op and Python wrapper. The initial checks use an isolated
  registration bridge linked against the CMake-built candidate library, not a
  replacement kernel. Complete-package validation is still pending.
- Device: Intel Graphics PCI 8086:674f; GPU 2 for accuracy, GPU 3 for timings,
  selected with the CRI container wrapper and exclusive per-GPU locks.
- Compiler: Intel DPC++ 2026.3.0 (2026.x.0.20260929).
- PyTorch: 2.15.0a0+git76ba663.
- Production operator shape: tokens=8, topk=8, experts=256, alignment=4.
  Serving context: B8/ISL4096/OSL1024/TP4/EP1. No model E2E claim is made.
- Baseline: the unchanged SGLang
  `_get_xpu_w4a4_small_permutation_kernel` factory in
  `/workspace/sglang/python/sglang/srt/layers/quantization/mxfp4.py`, SHA256
  `de247d5625e3f1d022fef697d2fa7445721a517fc8593d9da7be6b2c639b4e38`.
  These initial measurements compare the existing production Triton algorithm,
  not the requested old sgl-kernel operator. Although the older checkout's
  default branch lacks that operator, its `origin/muse_moe_permutation` branch
  does contain `mxfp4_small_permutation`. The archived complete installed
  package described below provides the corresponding baseline artifact.

## Completed checks

- CMake configuration succeeds after correcting `Python3_EXECUTABLE` usage.
- CMake targets `sgl-ops-sycl-mxfp4_small_permutation`, `sgl-ops-sycl-Device`,
  and `sgl-ops-sycl-HCHead` compile and link successfully.
- Accuracy: **258 passed**, 518.05 seconds. Integer outputs are checked
  exactly against a CPU reference. Coverage includes int32/int64 indices,
  0/1/7/8/9/16/32 tokens, 1/2/128/256/512/1024 experts, alignment
  1/3/4/16/64, low/high/colliding expert IDs, padding, slack, and invalid
  dimensions/dtypes. None of these tests was skipped.
- Performance sweep: all 18 comparisons pass a zero-regression gate and all
  four output tensors match the baseline bit-for-bit. The sweep covers
  E128/alignment4, E256/alignment4, and E512/alignment16, both index dtypes,
  and balanced/random/colliding routing.

## Production-shape timing

Median of five per-round p50 XPU-event measurements, milliseconds. Both paths
use preallocated outputs, identical inputs, 100 ms warmup, 500 ms repetition,
alternating measurement order, synchronization and a two-second cooldown.
These are event-bracketed operator latencies, not profiler-isolated kernel
execution durations; queue idle time inside the event interval can contribute.

| Index dtype | Routing | Candidate | Baseline | Candidate / baseline |
| --- | --- | ---: | ---: | ---: |
| int32 | balanced | 0.007865 | 0.008698 | 0.9042 |
| int32 | random | 0.007865 | 0.008802 | 0.8935 |
| int32 | colliding | 0.007812 | 0.008802 | 0.8875 |
| int64 | balanced | 0.007917 | 0.008958 | 0.8838 |
| int64 | random | 0.007917 | 0.008959 | 0.8837 |
| int64 | colliding | 0.007865 | 0.009010 | 0.8729 |

Some E512 baseline rounds have substantial variability. Their unusually large
speedups must not be interpreted as stable kernel-only improvements; repeat
those measurements under quieter host conditions before making such claims.

## Logs and remaining gates

- Target build: `/tmp/muse-permutation-target-build-20261021-02.log`, exit 0.
- Accuracy: `/tmp/muse-permutation-accuracy-gpu2-20261021.log`, exit 0.
- Timing sweep: `/tmp/muse-permutation-benchmark-sweep-gpu3-20261021.log`, exit 0.
- Initial AOT package dependency build:
  `/tmp/muse-permutation-target-build-20261021.log`, failed after OOM kills
  during the large GDN translation unit and a host compile, including the
  supervisor's reduced-parallelism retry. This does not prove an AOT build.
- Full `common_ops` build, all feature families enabled and
  `USE_SYCL_JIT=ON`: `/tmp/muse-permutation-full-jit-build-20261021.log`.
  This attempt exited 1: the host memory guard stopped `KVCacheIO.cpp`
  compilation when host memory usage reached 99% (9.77 GiB available).
  This was not a C++ diagnostic and the memory protection remains enabled.
- Resumed full JIT build:
  `/tmp/muse-permutation-full-jit-build-20261021-02.log`, exit 1. Host available memory
  recovered to approximately 428 GiB and container usage to approximately
  3.3 GiB before the restart. The supervisor now requests eight jobs, retaining
  its resource checks and one reduced-parallelism OOM retry. The prior
  AOT-derived job limit was reset for the changed JIT workload and recovered
  memory conditions.
  This attempt also hit the host 99% memory guard, including MXFP8, grouped
  GEMM and FP8 compilation. It never reached package installation.
- Normal-package validation pipeline:
  `/tmp/muse-permutation-package-validation-20261021-02.log`. After a successful
  full build it verifies staged source equality, builds a wheel, installs it
  into an isolated system-site-packages virtual environment, imports the
  installed package normally, and runs accuracy and production-shape timing
  checks without library loading or Python module injection.
  This JIT pipeline exited 1 because its prerequisite full build failed.
- Compiler-internal device-link parallelism previously defaulted to the host
  CPU count (256 here), independently of the supervisor's outer job budget.
  `cmake/BuildFlags.cmake` now defaults it to one with an explicit positive
  integer CMake cache override. Runtime kernel options are unchanged. This
  limits one source of nested build concurrency; it does not prove that all
  observed host memory pressure came from this build.
- Default-feature AOT build:
  `/tmp/muse-permutation-full-aot-build-20261021.log`. The current configuration
  has MoE/FMHA/GDN/MLA/QSA enabled, runtime JIT disabled, and compiler-internal
  device-link parallelism set to one. The outer supervisor requests four
  resource-budgeted jobs; memory protections stay enabled.
- Default AOT package validation pipeline:
  `/tmp/muse-permutation-package-validation-aot-20261021.log`. Completion of
  the full build, normal wheel installation, expanded accuracy suite and
  normal-import performance checks remains unproven until this pipeline exits
  successfully and its output is inspected.
- The CLI observation of that pipeline was interrupted. Its original host
  runner disappeared, but container build supervisor PID 881552 and Ninja PID
  885513 were subsequently confirmed alive with active FMHA compiler children.
  Missing original status files are not evidence of success or compile failure.
- Recovery logs:
  `/tmp/muse-permutation-full-aot-recovered-20261021.log` and
  `/tmp/muse-permutation-package-validation-aot-recovered-20261021.log`.
  The bounded recovery runner waits for those existing processes to terminate
  before verifying/resuming the same persistent default AOT target. It does
  not launch a duplicate concurrent build. The normal wheel installation and
  verification stages run only after that target command succeeds.
- The recovery pipeline exited 1. Its resumed default AOT build hit the same
  host memory guard in the page-2/head-dim-96 FMHA translation units
  (9.77–9.91 GiB available, 99% host memory usage). Subsequent inspection
  found only approximately 2.1 GiB available on the 1 TiB host. Limiting
  compiler-internal link concurrency did not resolve this external pressure.
- At that point the Ninja log retained 200 FMHA device-object output names.
  The legacy custom commands log both absolute and relative aliases for each
  physical object, so this must not be interpreted as 200 distinct completed
  translation units. The generated FMHA source count is 474. Build-log output
  counts require path normalization before being used as a progress measure;
  none of these partial counts proves a full-build pass.
- Resource-admitted pipeline:
  `/tmp/muse-permutation-package-validation-aot-admitted-20261021.log`.
  Its admission check waits at most 600 seconds for 24 GiB host MemAvailable,
  with `/tmp/muse-permutation-memory-admission-20261021.log` recording the
  result. Only then does the same persistent default AOT build resume, with
  eight requested outer jobs, eight GiB per job and a sixteen-GiB reserve;
  the existing supervisor may cap the actual job count further. The host and
  per-file memory protections remain enabled. Admission timeout is not a
  compiler failure and does not trigger a duplicate build.
- Admission succeeded with approximately 172 GiB host MemAvailable; the
  resource-admitted AOT build started. Later inspection found approximately
  385 GiB available, so the previous low-memory condition is not being treated
  as a continuing blocker without revalidation.
- The runtime validation stage now checks idle utilization, less than 256 MiB
  device memory usage, an empty process list and an available shared GPU lock
  immediately before selecting from the six container-visible GPUs. It then
  uses the normal wrapper's exclusive GPU lock and cooldown. GPU 3 was also
  independently checked: zero utilization, approximately 79.5 MiB used and no
  reported GPU processes. These checks do not replace validation-stage checks
  after the long CPU build.
- The current test source adds topk=1, topk=7 and topk=256 coverage and now
  contains 366 cases. The earlier 258-case result does not cover these added
  cases; the normal-package pipeline must prove the expanded suite passes.

The goal is not complete: full package compilation/runtime validation and a
normal-package comparison against the old repository's same operator remain
explicit completion gates. Passing an isolated bridge is not a substitute for
loading the newly built package.

The current source additionally uses a 64-bit induction variable for allocation
slack initialization. The advertised int32 row-count limit alone did not
prevent `2 * row` or the final stride increment from overflowing a 32-bit
intermediate at large output capacities. Stored slack row IDs remain int32
under the existing host dimension check. The normal wheel build must rebuild
this source change and repeat the accuracy and performance gates; the earlier
bridge measurements do not validate the current binary. No multi-gigabyte
capacity stress test is claimed by the existing production-shape suite.

The resource-admitted pipeline also exited 1 before package installation.
The compiler's host-memory guard stopped page-8/head-dim-128 FMHA translation
units at 99% usage (approximately 9.77–10.05 GiB available). No C++ diagnostic
matching `error:` or `fatal error:` was found. Its build supervisor and Ninja
processes have terminated; partial objects remain reusable. A subsequent
resource check found approximately 133 GiB available. The next incremental
attempt requests only two outer jobs and requires at least 64 GiB available
for sixty seconds before starting, without disabling any memory protection.
Its logs are `/tmp/muse-permutation-full-aot-lowjobs-20261021.log` and
`/tmp/muse-permutation-package-validation-aot-lowjobs-20261021.log`.

## Old repository same-operator baseline discovery

The old repository's `origin/muse_moe_permutation` currently resolves to
`04dce6561dfb85397ee06c2d6ded1ebe677f133c`, also checked out at
`/home/syk/worktrees/sgl-kernel-xpu/muse-permutation-only-main-20261015`.
Its `src/sycl/Mxfp4SmallPermutation.cpp` SHA256 is
`7b6173fdee8f51a8faa18e13e444ed8b4df9c9e15cd04f1bfcbd82cb80de61cd`.
That file matches the archived full-build source snapshot byte-for-byte.
The archived package is at
`/home/syk/worktrees/sgl-kernel-xpu/review-validation-artifacts/accuracy-20261016/permutation-scope-correction/install`.
Its common extension SHA256 is
`683e32c6f4130387f0385f186c3118305b0843266eb64a11bfed1789d3453c2d`;
its permutation library SHA256 is
`546d8c719e9067ba9c3dc7733516ced41b165225d4845989336241da232ded7e`.
Both match the retained full-build hash manifest. The original source-freeze
revision is `4b0132c3ea067610d4ffacdd3483ad1c4b003ec8`, with a documented
unrelated activation rollback overlay; this is not claimed to be a fresh
full build of the later branch revision.

The complete package and provenance files are staged at
`/workspace/sgl-kernel-xpu-up/.validation/muse-permutation-baseline-04dce656`.
Direct import against the current PyTorch/SYCL fails with unresolved SYCL
symbols. After restoring the archived compiler runtime, MPI/profiling libraries
and PyTorch from its original stopped container into this isolated environment,
normal package import succeeds with PyTorch `2.14.0a0+gitd35bb04` and exposes
`torch.ops.sgl_kernel.mxfp4_small_permutation.default`.
The successful probe log is
`/tmp/muse-permutation-old-torch-import-20261021-02.log`, exit 0.
The shared runtime and candidate sources are unchanged. No baseline GPU runtime
or performance pass is claimed yet. The missing `ensurepip` was also detected: the
candidate installation script now uses the already installed `virtualenv`
and an ordinary site-packages `.pth` dependency path, rather than a failing
`python -m venv` call or runtime module injection.

The benchmark now also supports normal-package workers for the old and new
registered operators in separate processes. Five alternating-order rounds
cover both index dtypes and balanced/random/colliding B8 routing, using the
same inputs and preallocated output capacities. Workers preserve the older
benchmark's XPU-profiler kernel-duration methodology and record 201 samples,
package/library hashes, inputs and all four outputs. The paired comparator
checks exact input/output equality across all rounds and uses a zero-regression
median-kernel-time gate. It does not compare the earlier event-bracketed timings
against profiler kernel timings. These new paths are wired into the validation
pipeline but have not yet been exercised on GPU.

A management query for GPU 3 became stuck in uninterruptible kernel sleep.
Independent `xpu-smi` processes also show long-lived D states. Only this task's
own management query received a termination signal; no foreign workload or
driver reset was attempted. Idle-GPU selection now rejects a long-blocked
management state and bounds new queries instead of hanging indefinitely or
assuming that an unmeasurable device is idle. This remains a runtime-selection
concern, not a reason to stop the live CPU compilation.

The task-owned GPU-3 query's wait channel is `drm_pagemap_acquire_owner`.
A separate bounded GPU-2 health query also did not return; its 15-second
deadline and two-second forced-termination grace ended with exit 137.
The log is `/tmp/muse-permutation-gpu2-health-query-20261021.log`.
Thus the runtime issue is not inferred only from a stale GPU-3 observation.
No successful idle-device selection, GPU test or performance result follows
from these failed management queries. The package pipeline also now verifies
that normal import resolves to the candidate environment's own site-packages,
and compares staged test/benchmark bytes against the working tree before use.

The stalled host queries hold descriptors for all sixteen render nodes, even
when a specific device is requested. Container inventory confirms only six
render nodes (`renderD128` through `renderD133`) are present. To test whether
restricting visibility could avoid the problem, the host management binary
was copied to the shared validation directory and run through the approved
container wrapper against device 2, with the same bounded deadline. That
container-visible-only query also exited 137 without metrics:
`/tmp/muse-permutation-container-gpu-health-20261021.log`.
This rules out treating reduced device visibility alone as a demonstrated
workaround; it does not prove that every card is individually unhealthy.
