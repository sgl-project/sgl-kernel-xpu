# GitHub Copilot Instructions for sgl-kernel

These rules apply to code generated or modified in this repository.

## 1) Source of truth

- Environment and dependencies: `Dockerfile.xpu_kernel`
- Build and development workflow: `README.md`
- Contribution expectations: `CONTRIBUTING.md`
- Formatting/lint hooks: `.pre-commit-config.yaml`

If guidance conflicts, prefer repository files over generic defaults.

## 2) Environment and build expectations

- Assume Intel XPU development (SYCL + oneAPI).
- Use oneAPI environment before build/test commands:

```bash
source /PATH/TO/ONEAPI/setvars.sh
```

- Default build command should follow repository guidance:

```bash
pip install -v .
```

- For wheel builds, use the documented path (uv/scikit-build) and keep `build/` as build dir.
- Do not introduce a new build system.

## 3) Code change rules

- Keep changes minimal and scoped to the requested fix.
- Reuse existing utilities and patterns in `src/`, `include/`, and `python/sgl_kernel`.
- Avoid adding dependencies unless clearly required.
- Preserve API compatibility unless the task explicitly requires a breaking change.

### Kernel-specific rules

- Follow existing SYCL kernel style and launch patterns.
- Prefer pure SYCL/C++ implementation; keep torch binding concerns in extension/shim layers.
- If torch types are needed in C++, prefer `torch/all.h` over `torch/extension.h` in SABI-sensitive code paths.
- When integrating third-party kernels, use existing shim patterns in `include/` for type conversion.

### Op coding rules (new and changed code)

Apply when writing, optimizing or reviewing any op under `src/`, `include/`
or `python/sgl_kernel/`. An op must be stateless and graph-capture
compatible. Existing code predates these rules; do not copy violating
patterns. Details, examples and scanners are in the skills
`.claude/skills/op-coding-rules/` and `.claude/skills/stride-overflow-safety/`.

- **OP-1** No host-device sync (`.item()`, `.cpu()`, `queue.wait()`, ...).
- **OP-2** Grid size and kernel choice from host-known info only; all work on
  the current queue; side queues from `c10::xpu::getStreamFromPool()`,
  joined back by events.
- **OP-3** CuTe preferred, Collective and Kernel API allowed; a SYCL-TLA
  Device API (`*Adapter`) needs a reason and must run on the current queue.
- **OP-4** Plain layouts unless the interface declares a blocked layout;
  weights via a prepack API.
- **OP-5** Outputs and workspaces from the torch allocator; no raw
  `sycl::malloc_*`.
- **OP-6** No device tensors or pointers kept across calls.
- **OP-7** Respect input dtypes: no silent downcast, FP8 operands need user
  scales, FP32 accumulation.
- **TLA-1** sycl-tla public headers only.
- **SO-1..SO-6** Strides, sizes and global offsets are `int64_t` end to end;
  widen before the multiply; narrow only with a guarded bound.

Before finishing, run on your changes and fix or justify every hit:

```bash
python3 .claude/skills/op-coding-rules/scan_op_rules.py --diff
python3 .claude/skills/stride-overflow-safety/scan_stride_overflow.py --diff
```

## 4) Where to place changes

When adding a new kernel or op, update all relevant layers:

1. Kernel implementation in `src/`
2. Public C++ interface in `include/sgl_kernel_ops.h`
3. Torch extension registration in `src/torch_extension_sycl.cc`
4. Build integration in `CMakeLists.txt` or relevant CMake files
5. Python exposure in `python/sgl_kernel/`

Do not leave partially wired features.

## 5) Testing requirements

- Every bug fix or feature change must include or update tests in `tests/`.
- Follow existing test naming and patterns (pytest-based).
- Run targeted tests for modified areas, then run broader validation when feasible.
- For skips, use `pytest.mark.skipif(..., reason=...)` with a concrete reason.

Example targeted test runs:

```bash
pytest -q tests/test_awq_dequant.py
pytest -q tests/test_flash_attention.py
```

## 6) Benchmark requirements

- Performance-sensitive changes require benchmark updates or evidence.
- Put benchmark changes under `benchmark/` (not `benchmarks/`).
- Report before/after numbers and workload shape when proposing perf changes.

## 7) Lint and formatting gates

Before commit or PR, run:

```bash
pre-commit run --all-files
```

The configured hooks include YAML/TOML checks, whitespace checks, isort, ruff, codespell, clang-format, and notebook cleanup. Fix all issues introduced by the change.

## 8) Pull request checklist

PRs fill in `.github/pull_request_template.md`. At minimum:

- What changed and why
- Build/test commands run locally and outcomes
- Benchmark evidence for performance changes

Apply labels:

- Always: `run-ci`
- If perf-sensitive: `perf`

## 9) Contributing with AI

PR policy is in `CONTRIBUTING.md`: AI explores, the author decides and must
understand what the change solves, how, its limitations, and its effect on
untargeted cases. For agents this means:

- Write only necessary code: no drive-by refactors, restating comments,
  re-implemented helpers, defensive checks on internal paths, or single-use
  abstractions.
- Declare each op precondition at the entry (comment + `TORCH_CHECK` or
  dispatch condition); reject inputs outside it, never return wrong results
  silently. Test boundaries and rejected inputs against an independent
  reference, with tolerances derived from the dtype path.
- Keep accumulation and reductions in fp32; make every narrowing conversion
  explicit and tell the user where it happens.
- When a change adds, removes or moves synchronization, workspace reuse or
  async work, tell the user who produces/consumes, what orders it, and when
  buffers are reused.
- Follow the pr-review skill (`.claude/skills/pr-review/SKILL.md`) for
  self-checks, PR descriptions and reviews; as a reviewer, check the
  self-justification first and return the PR without a detailed review if
  it is insufficient.
- When drafting a PR description, follow `.github/pull_request_template.md`
  and state only facts verified in the session (commands run, real output).
  Leave intent, trade-offs, risks and review focus to the author as
  questions; never invent results.

## 10) Safety and quality bar

- Do not claim tests or benchmarks were run if they were not run.
- Do not commit if pre-commit fails.
- Avoid speculative refactors unrelated to the request.
- Prefer correctness and maintainability first, then optimize.
