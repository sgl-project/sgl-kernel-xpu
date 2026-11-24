# Contributing to sgl-kernel-xpu

Welcome to sgl-kernel-xpu! We are building high performance kernels for Intel
GPUs, and we appreciate your interest in contributing. This page is the
contract between authors and reviewers: what a pull request must contain, and
how it is reviewed. Coding rules for agents and humans are in
[`.github/copilot-instructions.md`](.github/copilot-instructions.md).

We welcome pull requests of four kinds:

- **Feature enabling**: enable a new feature for XPU, e.g. `fused_add_rmsnorm`.
- **Performance optimization**: speed up critical operators, data types or
  quantization recipes, e.g. FP8 flash attention.
- **Module refactoring**: improve maintainability and readability, e.g.
  refactor the scratchpad memory allocation in kernels.
- **Bug fixing**: fix wrong behavior and improve test coverage and robustness.

## Principle: AI explores, the author decides

You may use AI in any step: writing code, analysis, tests, profiling. The bar
for the engineer does not change. Understanding the problem, choosing the
design and weighing the trade-offs (performance, code size, maintainability,
fit with the repository design) stay with the author, no matter how much of
the code AI produced.

- **The author does the first review.** Understanding, judging and validating
  the change is the author's job. Review is an independent second line of
  defense, not a place to finish that work.
- **Disclose AI use** in the PR template (none / assisted / substantial, and
  which parts).
- **Own the change, not just the code.** You must be able to explain, without
  the AI:
  - what problem it solves and why this approach (e.g. how the optimization
    works and where the gain comes from);
  - its limitations and the conditions it relies on;
  - whether it helps only the target case, and whether it regresses or
    changes behavior for other shapes, dtypes, devices or callers;
  - why each part of the diff is needed for the change.

  "The model wrote it" is not an answer in review.

## Before you start

- Sync with the latest `main`.
- Use `main` for device-agnostic features and optimizations.
- **Do not expose hardware or software features covered by Intel(R) Protect
  IP on `main`.**
- New public ops, API changes and new kernel families should be discussed
  first in an issue (see [Design discussion](#design-discussion)).
- Set up the environment as described in [`README.md`](README.md) and
  `Dockerfile.xpu_kernel`.

## Scope and size

- **One logical change per PR.** Do not mix refactors, formatting or unrelated
  fixes with behavior changes.
- **Size guideline:** aim for at most ~500 added + deleted lines of source
  (`src/`, `include/`, `python/`, `CMakeLists.txt`, `cmake/`) per PR. Tests
  and benchmarks are not counted but must stay focused. Where possible, split
  larger work into PRs that are each complete and reviewable on their own
  (e.g. kernel + tests, then Python wrapper and dispatch). This is not a hard
  limit: a PR above it is fine if its description explains why it cannot
  reasonably be split (e.g. a new kernel that is only testable end to end, or
  a port of an existing kernel family).

  ```bash
  git diff --numstat origin/main... -- src include python CMakeLists.txt cmake \
    | awk '{a+=$1; d+=$2} END {print a+d}'
  ```

- **Only necessary code.** Cleaning up is the author's job, whoever wrote the
  code. Reviewers may ask to remove any of these on sight, without proving
  harm:
  - drive-by refactors or formatting outside the change;
  - comments that restate the code, or carry ticket ids, measured data or
    history;
  - re-implementations of helpers that already exist (in `include/`,
    `src/`, sibling kernels, sycl-tla APIs or `python/sgl_kernel`);
  - defensive checks on internal call paths whose invariants already hold
    (validate at the public op entry, not at every layer);
  - abstraction layers, templates or options with a single user;
  - dead code, commented-out code, debug prints, local logs and CSV/profile
    artifacts.

## Core modules and major changes

These PRs need a written self-justification (next section):

| Category | Examples |
|---|---|
| Kernel or launcher behavior | any functional change under `src/` or `include/` |
| Dispatch and specialization | new template instance, tile config, split-KV heuristic, head-dim / dtype / GQA / device dispatch |
| Synchronization or memory lifetime | barriers, atomics, cross-work-group reductions, side queues/events, workspace reuse, async paths |
| Public API and registration | `include/sgl_kernel_ops.h`, `src/torch_extension_sycl.cc`, `python/sgl_kernel/`, op schema, defaults, dtypes |
| Build | `CMakeLists.txt`, `cmake/`, build flags, wheel scripts |
| Third-party bump | sycl-tla or other dependency versions |
| Performance claim | any PR that states a speedup |

Docs-only, test-only (no tolerance changes) and CI-script fixes need only the
short PR description.

## Self-justification

Write it yourself, from your own understanding and measurements. AI may polish
the wording; the key decisions and evidence must be ones you can defend
without it. Conclusions without evidence are not enough.

The PR description (see [the template](.github/pull_request_template.md))
must state:

1. **Intent**: the problem and the user-visible effect.
2. **Core logic**: how it works, in a few sentences, and the files /
   functions where the core change is.
3. **Assumptions and preconditions**: shapes, dtypes, alignment, layout, head
   dims, devices the code relies on. Specialized paths also declare them at the
   code entry as a short comment plus a `TORCH_CHECK` or dispatch condition
   (see [Boundary conditions](#boundary-conditions)).
4. **Approach and trade-offs**: alternatives considered, why this one; what it
   costs (code size, build time, wheel size, maintenance).
5. **Risks**: where it helps, where it can hurt, what it does not cover.
6. **Validation**: correctness and performance: baseline commit, device and
   driver, shapes/dtypes, exact commands to reproduce, results, and what is
   **not** covered.
7. **Review focus**: the parts you are unsure about. Leave inline comments on
   the diff for tricky spots.

If the self-justification is missing or insufficient, reviewers return the PR
without a line-by-line review.

## Tests

Every bug fix or feature change includes or updates tests in `tests/`. Tests
are part of the self-justification: state what scenarios they cover, which
failure mode each group of cases targets, and what is not covered. Test count
or coverage numbers do not replace this.

Insufficient tests:

- **Same-source reference**: the expected value copies the kernel's algorithm
  (same tiling, same split, same accumulation order) instead of an independent
  reference (plain PyTorch math, a trusted library, or closed-form values).
- **Happy path only**: no tails, unaligned sizes, extreme shapes or rejected
  inputs.
- **Loosened checks**: tolerances raised or asserts weakened to make a test
  pass, without explaining the numerical reason (dtype, accumulation order,
  quantization error) in the PR.
- **Unexplained tolerances**: a new test (especially for a new op or feature)
  whose tolerance has no stated derivation (see below).

### Tolerances for new tests

The author shows the logic of every new tolerance, so a reviewer can tell it
would catch a real bug.

- **Reference precision.** Compute the reference in FP32 (FP64 for FP32
  kernels) from the same inputs the kernel sees; for quantized operands,
  dequantize them with the same scales first. The tolerance then measures the
  kernel's own error, not input quantization.
- **Derive from the dtype path.** Base the tolerance on the lowest precision on
  the compute path (input dtype, FP8/FP4 format, quantized intermediates such
  as P in attention) and on the accumulation length (head dim, K, seqlen) when
  error grows with it. Write the formula, not only the number.
- **Define it once.** Use a per-dtype / per-mode helper shared by the tests of
  the op, not magic numbers per test case.
- **Exact where exact.** Integer outputs, indices, top-k / routing results,
  copies, layout and packing transforms, and deterministic paths compare with
  `atol=0, rtol=0` (or `torch.equal`).
- **Tight, with evidence.** In the PR, report the maximum observed error over
  the test sweep next to the chosen tolerance, and keep a small margin. Show
  sensitivity at least once: the test fails when a typical bug is injected
  (dropped tail tile, wrong scale, narrower accumulator).
- **Right metric.** Elementwise `atol`/`rtol` by default; `atol` must be small
  relative to the output magnitude. Aggregate metrics (cosine similarity,
  relative L2) are allowed for quantized paths only together with a max-abs
  bound or a stated reason.

Also:

- Register new test files in `tests/run_suite.py` (`per-commit` for a fast
  representative set; the full sweep can run nightly).
- Ops whose grid or kernel choice depends on shapes, and which serve inside
  graph capture in SGLang, get a capture/replay test.
- Stride / offset fixes ship with a test whose offset crosses 2^31 elements.
- Skip with `pytest.mark.skipif(..., reason=...)` and a concrete reason (e.g.
  the device does not support the feature), never to hide a failure.

## Boundary conditions

Every op states where it is valid and what happens outside.

- **Declare** preconditions at the public op entry: a one-line comment and a
  `TORCH_CHECK` (or an explicit dispatch condition) for each one.
- **Handle or reject**: tail blocks and non-divisible sizes; unaligned
  pointers, strides and sequence lengths; extreme shapes (empty batch, seqlen
  0/1, very long sequences, large block ids whose offsets pass 2^31); every
  supported head dim, GQA ratio, page size, dtype and device.
- **Never silently wrong.** Inputs outside the preconditions are rejected with
  a clear error, or routed to a general path. A device-specific path states
  which devices it supports, and other devices get a clear error or a general
  path.
- **Unit tests cover** the boundary cases and at least one rejected input per
  precondition (`pytest.raises`).

## Data types and precision

Precision is the user's decision. The kernel follows the input dtypes;
unsupported combinations are rejected; inputs are never quantized or
downcast unless the API opts in; FP8 operands need user-provided scales; the
output dtype follows the API contract (and the CUDA `sgl-kernel` op it
mirrors, if any).

Inside the op, optimizations often lower precision without saying so. Avoid
implicit downcasts: a narrowing conversion must be explicit in code and
justified in the PR.

- **What counts:** fp32 accumulators, softmax max/sum, LSE, split-KV partials,
  reductions or recurrent states stored or carried in bf16/fp16/fp8; a
  narrower type introduced by `auto`, template deduction, a register fragment
  or a smaller workspace dtype; a store converting before a final reduction;
  int64 sizes, strides or offsets narrowed to int32.
- **Rule:** keep accumulation and cross-tile / cross-split reductions in fp32
  unless the interface declares otherwise. Write each intended narrowing as an
  explicit conversion (`static_cast`, a named convert helper) at a single
  point.
- **In the PR:** for each new narrowing, state where it happens, why it is
  safe (value range, error bound, what upstream or the reference does), and
  show a test that would catch the error (long sequences, large magnitudes,
  many splits) at the existing tolerance. A user-visible output dtype or
  default never changes silently.

## Synchronization and concurrency

Synchronization and resource lifetimes are where most hard bugs come from. Any
change that adds, removes or moves synchronization (host↔device, queue↔queue,
work-group↔work-group, sub-group↔sub-group) needs its own paragraph in the PR
answering:

1. Who produces and who consumes each shared buffer or value.
2. What establishes execution order and memory visibility (event, barrier and
   its fence scope, atomic with which memory order and scope, kernel
   boundary).
3. Which participants the synchronization covers (all work-items of the
   work-group? all work-groups of the grid? both queues?).
4. When a buffer can be reused or freed, and what guarantees nobody still
   reads it.
5. Whether correctness depends on scheduling (e.g. all work-groups resident at
   once, launch order), sub-group size, or a hardware feature.

Removing synchronization needs the same proof as adding it. Async conversions,
workspace reuse and lifetime changes alter the contract even when no sync API
appears in the diff. Avoid host-device synchronization inside an op: it breaks
graph capture and adds latency to every call in serving.

## Performance, binary size and build time

Optimizations often speed up the target case and slow down others. A PR that
affects kernel code, launch configuration or dispatch must show it does not
regress the cases it did not target.

- Update or add benchmarks under `benchmark/` (not `benchmarks/`) when
  enabling a new feature or optimizing a new scenario, especially for critical
  operators. Register CI benchmarks in `benchmark/run_suite.py`.
- Report before/after numbers on the same device, driver and shapes, for the
  target case **and** representative other cases.
- PR CI compares the flash attention and fused MoE benchmark results
  (`flash.log`, `fused_moe.log`) against `benchmark/baseline.json` (see
  [`benchmark/README.md`](benchmark/README.md)). Explain every case that gets
  slower. Add the `perf` label to performance-sensitive PRs so CI opens the
  baseline update PR.
- **New specializations** (template instances, tile configs, dispatch
  entries) state the cases they cover, the gain, and the cost: change in wheel
  / `.so` size and in build time against `main`.
- **Benchmark changes** (new or edited shapes, thresholds, baselines) explain
  why the cases are representative; do not tune the suite to the change.

## Lint and formatting

Run `pre-commit run --files <changed files>` (or `--all-files`) before every
push. The hooks cover YAML/TOML checks, whitespace, isort, ruff, black,
codespell and clang-format. CI runs the same checks.

## Pull request title and description

- Title: specific and in the imperative, with area tags as used in history,
  e.g. `[Feat] Make bias optional for biased_topk`,
  `[Perf] topk: optimize performance of fast_topk`, `[FMHA] ...`, `[MoE] ...`,
  `[CI] ...`, `[Fix] ...`. Avoid `fix bug`, `update`, `WIP`, `misc`.
- Description: fill in [the PR template](.github/pull_request_template.md).
  Call out new build flags, environment variables and requirements.
- Labels: `run-ci` to run XPU CI, `perf` for performance-sensitive changes.
  Reviewers add `need-author-clarification` (see [Review](#review)).
- Keep generated logs, CSVs, dumps and local scripts out of the PR.

## Design discussion

Agree on the problem, interface and design before writing the code when a
change adds or changes a public op or API, adds a kernel family, introduces a
blocked layout or prepack API, or changes shared build, CI or benchmark
infrastructure. Open an issue with: the problem and evidence, the proposed API
and semantics, preconditions, design, alternatives, costs, validation plan and
rollout split. Implementation PRs link the issue.

## Review

### Authors

- Mark the PR as `draft` and ask maintainers for review (mingfeima, airMeng,
  chunyuan-w, sunjiweiswift). Check `.github/CODEOWNERS` for owned paths.
- For public pull requests on the SGLang repo, conduct an internal review
  first before requesting reviews from the community.
- Before requesting review, run the self-check in the
  [pr-review skill](.claude/skills/pr-review/SKILL.md) (scanners, size,
  clutter, boundaries, tests, description).
- Answer comments with code or reasoning, not just "done". Re-run the relevant
  validation after substantial updates and update the numbers in the
  description.
- Mark the PR `ready for review` once comments are addressed.

### Reviewers

Review in this order and stop at the first step that fails:

1. **Self-justification**: present and sufficient for the change category. If
   not, return the PR with the list of missing items.
2. **Scope**: size guideline (or a reason for exceeding it), one logical
   change, no clutter.
3. **Repository rules**: `.github/copilot-instructions.md` (op coding rules
   OP-1..OP-7, TLA-1, int64 strides SO-1..SO-6) on new and changed lines,
   the `scan_op_rules.py --diff` and `scan_stride_overflow.py --diff`
   scanners, pre-commit, no Protect IP exposure on `main`.
4. **Correctness**: synchronization, boundaries, numerics (including implicit
   downcasts), test quality.
5. **Performance evidence**: before/after data, untargeted cases, size and
   build cost.
6. **Details.**

If you suspect the author has not digested a piece of complex logic, ask them
to explain it. If they cannot explain the key logic, assumptions or evidence,
or the explanation does not match the code, pause the review and add the
`need-author-clarification` label. Review resumes after the author updates the
PR.

AI review (e.g. the pr-review skill) helps reviewers find issues; a human
reviewer makes the decision.

### Merge readiness

- At least one approval for simple PRs; at least two for complex PRs.
- Specific title; description complete for its change category, with AI use
  disclosed.
- Focused scope; within the size guideline, or the description explains why
  it cannot reasonably be split.
- All CI tests pass.
- Tests cover boundaries and rejected inputs against an independent reference,
  with tolerances derived as in [Tolerances for new tests](#tolerances-for-new-tests).
- Performance data for performance-relevant changes; regressions explained.
- All review threads resolved; no artifacts or unrelated edits.

## Rerunning PR CI

Maintainers with write, maintain, or admin repository permission can comment on
an open PR targeting `main` after adding the `run-ci` label:

* `/rerun-ci` starts fresh XPU CI for the current PR head and reruns the latest
	lint run for that head. This also handles XPU runs skipped before `run-ci`
	was added; there is no need to close and reopen the PR.
* `/rerun-failed-ci` reruns only failed jobs in the latest XPU and lint runs for
	the current head. Successful and skipped runs are left alone.

Post the command by itself in a new PR comment. Active runs are not duplicated,
including retries of older matching runs, and the bot reports its actions in a
reply. Runs explicitly associated with another PR are not retried, even when
the branch and head SHA match. XPU manual dispatch also accepts a PR number and
its expected head SHA. Dispatched runs execute tests and benchmarks;
host-side benchmark processing and container tests use the PR merge ref.
Automatic performance-baseline PR creation remains limited to normal PR events.
