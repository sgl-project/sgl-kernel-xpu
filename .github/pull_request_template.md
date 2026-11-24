<!--
See CONTRIBUTING.md. Write the sections yourself; AI may polish wording,
but you must be able to defend every statement. Delete hints in comments.
Sections marked (core/major) are required for the change categories listed
under "Core modules and major changes"; write "N/A" otherwise.
-->

## Summary

<!-- What changed and the user-visible effect. One short paragraph. -->

## Motivation

<!-- Why it is needed. Link the issue / design discussion if any. -->

## Core logic and key locations (core/major)

<!-- How it works in a few sentences. List the files/functions holding the
core change, in the order a reviewer should read them. -->

## Assumptions and preconditions (core/major)

<!-- Shapes, dtypes, alignment, layouts, head dims, page sizes and devices the
code relies on, and where each one is checked in code (TORCH_CHECK / dispatch
condition). What happens to inputs outside them (rejected / routed to which
path). -->

## Trade-offs and risks (core/major)

<!-- Alternatives considered and why this one. Costs: code size, wheel size,
build time, maintenance. Where it helps, where it can hurt, what it does not
cover. -->

## Synchronization and lifetimes

<!-- Required if the PR adds/removes/moves any sync, atomics, barriers,
events, side queues, async paths or workspace reuse; otherwise "N/A".
Producer/consumer, ordering + visibility mechanism, participants covered,
reuse/free point, scheduling/sub-group/hardware dependencies. -->

## Validation

**Correctness**
<!-- Device + driver, base commit, exact commands, results. For each test
group: which failure mode it targets. Reference used (must be independent of
the kernel). For each new tolerance: how it is derived (dtype path,
accumulation length), max observed error, and a bug it would catch.
Tolerance changes and why. What is NOT covered. -->

**Performance** (kernel, launch or dispatch changes)
<!-- Before/after numbers on the same device and shapes, for the target case
and representative other cases. Explain every regression. New
specializations: wheel/.so size and build time vs main. -->

## Review focus

<!-- Parts you are unsure about or want reviewed closely. Add inline comments
on the diff for tricky spots. -->

## Checklist

- [ ] One logical change; source diff within ~500 lines, or the Summary explains why it cannot reasonably be split
- [ ] No drive-by refactors, restating comments, duplicated helpers, single-use abstractions or artifacts
- [ ] `pre-commit run --files <changed files>` is clean
- [ ] `scan_op_rules.py --diff` and `scan_stride_overflow.py --diff` hits are fixed or explained
- [ ] Preconditions declared at the op entry; tests cover boundaries and rejected inputs
- [ ] New tests registered in `tests/run_suite.py`; benchmarks updated under `benchmark/` if perf-relevant
- [ ] No hardware or software features covered by Intel(R) Protect IP exposed on `main`
- [ ] Labels: `run-ci`; `perf` if performance-sensitive
- [ ] New build flags / env vars / requirements are documented

## AI assistance

- [ ] None
- [ ] Assisted (completion, small edits, wording)
- [ ] Substantial (AI generated significant code, tests or analysis)

<!-- If assisted/substantial: which parts, and how you verified them. -->

- [ ] I can explain, without AI, what this change solves and how, its limitations, and its effect on cases it does not target
