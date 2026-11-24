---
name: pr-review
description: End-to-end sgl-kernel-xpu pull-request workflow for authors and reviewers, implementing CONTRIBUTING.md. Author mode - pre-submit self-check (size guideline, clutter, rule scanners, boundary preconditions, test independence, sync/lifetime notes, performance evidence) and drafting a PR description that contains only verified facts and leaves the self-justification to the author. Reviewer mode - gate on self-justification first, then scope, op coding and stride rules, Protect IP, synchronization, boundaries, numerics, test quality and performance evidence; ask understanding questions; report findings as [TAG] path:line. Use when preparing, describing, self-reviewing or reviewing a PR or a branch diff.
---

# PR self-check and review (sgl-kernel-xpu)

The policy is [CONTRIBUTING.md](../../../CONTRIBUTING.md); this skill is the
procedure. Coding rules are in
[`.github/copilot-instructions.md`](../../../.github/copilot-instructions.md)
and the rule skills it links (op-coding-rules, stride-overflow-safety).
Base defaults to `origin/main`; use `git diff <base>...` (three dots)
everywhere.

## Finding tags

Report every finding as `[TAG] path:line — what is wrong — required fix`.

| Tag | Covers | Blocking |
|---|---|---|
| JUSTIFY | Self-justification missing or insufficient | Return PR |
| CLARIFY | Author cannot explain logic, or description contradicts code | Pause, `need-author-clarification` |
| SCOPE | Over the ~500-line size guideline with no reason given; mixed unrelated changes | Yes |
| CLUTTER | Drive-by refactor, restating comment, duplicate helper, defensive internal check, single-use abstraction, artifact | Yes (removal, no proof needed) |
| OP-n / TLA-1 / SO-n | Op coding and stride rules (new and changed lines only) | Yes |
| IP | Hardware or software features covered by Intel(R) Protect IP exposed on `main` | Yes |
| SYNC | Ordering, visibility, participants, reuse/free, scheduling dependence | Yes |
| BOUNDARY | Undeclared precondition, silent wrong result, untested tail/unaligned/extreme/rejected input | Yes |
| TEST | Same-source reference, happy path only, unexplained tolerance change, new tolerance without derivation/evidence, test not registered in `tests/run_suite.py`, no graph test where needed | Yes |
| PERF | No before/after data, unexplained regression, untargeted cases missing, size/build cost missing | Yes |
| NUM | dtype path, accumulation precision, overflow/underflow, implicit downcast | Yes |
| HINT | inline `asm` that could be a sycl-tla API; code another kernel already has; pre-existing rule violation in a touched file | No |

## Author mode: before requesting review

1. **Scope.** Count source lines:
   `git diff --numstat origin/main... -- src include python CMakeLists.txt cmake | awk '{a+=$1; d+=$2} END {print a+d}'`.
   Over ~500: propose a split into self-contained PRs; if it cannot
   reasonably be split, ask the author for the reason and leave an
   `<!-- AUTHOR: why not split? -->` placeholder in the Summary.
   List files that do not serve the stated intent.
2. **Clutter.** Read every hunk of `git diff origin/main...` and remove the
   items listed under "Only necessary code" in CONTRIBUTING.md. Search for an
   existing helper before keeping a new one (`include/`, sibling kernels in
   `src/sycl/`, sycl-tla, `python/sgl_kernel/`).
3. **Scanners and format.** Fix or justify every hit:
   ```bash
   python3 .claude/skills/op-coding-rules/scan_op_rules.py --diff
   python3 .claude/skills/stride-overflow-safety/scan_stride_overflow.py --diff
   pre-commit run --files $(git diff --name-only origin/main...)
   ```
   Check that the diff exposes no feature covered by Intel(R) Protect IP
   (unreleased hardware names, features or performance data).
4. **Wiring.** A new op is wired through all layers listed in
   `.github/copilot-instructions.md` ("Where to place changes"): kernel in
   `src/`, interface in `include/sgl_kernel_ops.h`, registration in
   `src/torch_extension_sycl.cc`, `CMakeLists.txt`, Python in
   `python/sgl_kernel/`. No partially wired feature.
5. **Preconditions.** For each public op touched, list its preconditions and
   confirm each has a `TORCH_CHECK` or dispatch condition at the entry and a
   test, including one rejected input.
6. **Tests.** Confirm the reference is independent of the kernel, tails /
   unaligned / extreme shapes are covered, new test files are registered in
   `tests/run_suite.py`, and any tolerance change has a numerical reason.
   Build and run the affected tests on an XPU device.
   For each new tolerance, check it follows "Tolerances for new tests" in
   CONTRIBUTING.md; record the maximum observed error from the run you did
   and leave `<!-- AUTHOR: derivation of the tolerance -->` for the reason.
   Check that the change introduces no implicit downcast (CONTRIBUTING.md
   "Data types and precision"); make each intended narrowing explicit and
   note it in the description.
7. **Sync and lifetimes.** If the diff touches barriers, atomics, events,
   queues, workspaces or async paths, draft the five answers from
   "Synchronization and concurrency" in CONTRIBUTING.md.
8. **Performance.** For kernel / launch / dispatch changes, run the matching
   benchmark under `benchmark/` on the base and on the branch, on the same
   device, for the target case and representative other cases. If the
   change touches flash attention or fused MoE, also compare against
   `benchmark/baseline.json` (see `benchmark/README.md`). For new
   specializations, measure wheel/`.so` size and build time against main.
9. **Description.** Fill `.github/pull_request_template.md`:
   - Write only facts you verified in this session: changed files, commands
     you ran and their real output, diff stats, scanner results, benchmark
     numbers. Never invent numbers, devices or test results; mark anything
     not run as "not run".
   - Do **not** write Intent, Trade-offs, Risks or Review focus as finished
     prose. Leave `<!-- AUTHOR: ... -->` placeholders, each with 2–4
     concrete questions the author must answer (e.g. "Why split-KV 4 rather
     than a heuristic on seqlen?"). The author owns these sections.
   - Fill the AI-assistance section truthfully for the work done here.
   - Labels: `run-ci`; `perf` if performance-sensitive.

## Reviewer mode

Work through the steps in order. Stop and report at the first blocking
step; do not start a line-by-line review of a PR that fails step 1 or 2.

1. **Gate: self-justification.** Classify the change with the "Core modules
   and major changes" table in CONTRIBUTING.md. For core/major changes check
   that the description has intent, core logic and key locations,
   preconditions, trade-offs, risks, validation (baseline, device, driver,
   reproducible commands, uncovered scope), review focus, and AI
   disclosure. Missing or conclusion-only items → `[JUSTIFY]` listing each
   gap; verdict **Return**.
2. **Scope and clutter.** Over the size guideline is acceptable only with a
   stated reason why it cannot reasonably be split; check the reason holds
   (could the kernel, the Python wrapper and the dispatch land separately?).
   One logical change; `[CLUTTER]` items.
3. **Repository rules.** Run the scanners from author step 3, then the
   review steps in the op-coding-rules and stride-overflow-safety skills.
   Confirm each hit in context. The rules cover new and changed lines;
   pre-existing violations in touched files are `[HINT]` unless the change
   makes them worse. Check for Protect IP exposure (`[IP]`) and partial
   wiring (author step 4).
4. **Consistency with the description.** Check that the code does what the
   description claims, the stated preconditions are the ones the code
   checks, and the "key locations" are where the logic is. If the PR
   implements a design agreed in an issue, check the API and preconditions
   against it. Mismatches are `[CLARIFY]`.
5. **Synchronization.** For every sync primitive, atomic, event, side queue,
   workspace reuse or async path added, removed or moved: identify producer
   and consumer, the ordering/visibility mechanism and its scope (work-group
   barrier vs global fence, atomic memory order and scope), the participants
   covered, the reuse/free point, and any dependence on all work-groups being
   resident, launch order or sub-group size. Unanswered → `[SYNC]`.
6. **Boundaries.** From the kernel's tiling, derive the boundary cases:
   tails and non-divisible sizes, unaligned pointers/strides/seqlens, empty
   batch, seqlen 0/1, max head dim, GQA ratios, page sizes, large block
   ids, unsupported devices. Each must be handled or rejected at the entry
   and tested. `[BOUNDARY]` otherwise.
7. **Tests and numerics.** Reference independence, failure modes targeted,
   tolerance changes, dtype/accumulation paths, graph test when grid or
   kernel choice depends on shapes, registration in `tests/run_suite.py`.
   `[TEST]`, `[NUM]`.
   Tolerances in new tests: reference in FP32 on the same (dequantized)
   inputs; value derived from the dtype path and accumulation length and
   defined once per dtype/mode; exact comparison for integer, index, routing
   and layout outputs; max observed error and a sensitivity check in the
   description. A bare number, a per-case magic constant, or an aggregate
   metric without a max-abs bound is `[TEST]`.
   Downcasts: for each changed accumulator, reduction, workspace, state or
   register fragment, compare its type with the base. A narrower type that
   is implicit (`auto`, template deduction, assignment, a smaller workspace
   dtype) or not justified in the description is `[NUM]`. Show the old and
   new type in the finding.
8. **Performance.** Before/after data for the target and representative
   untargeted cases on the same device; `baseline.json` comparison explained
   when it reports slower cases; size/build cost for new specializations;
   benchmark changes justified. `[PERF]`.
9. **Understanding probes.** Pick the 1–3 most complex hunks (index math,
   sync, dispatch heuristics) and write one concrete question each for the
   author. If earlier answers in the thread contradict the code, raise
   `[CLARIFY]` and recommend the `need-author-clarification` label.

## Output format

```
Verdict: Return | Need author clarification | Request changes | Comment
Category: <core/major category or "minor">
Size: <source lines> (within guideline | over, reason given | over, no reason)

Blocking
[TAG] path:line — what — fix

Questions for the author
1. ...

Non-blocking
[HINT] path:line — ...
```

Only evidence-backed findings, each with `path:line`. No style comments
beyond pre-commit. The agent never approves; a human reviewer decides.
