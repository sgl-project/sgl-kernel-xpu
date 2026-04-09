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

PRs should include:

- What changed and why
- Build/test commands run locally and outcomes
- Benchmark evidence for performance changes

Apply labels:

- Always: `run-ci`
- If perf-sensitive: `perf`

## 9) Safety and quality bar

- Do not claim tests or benchmarks were run if they were not run.
- Do not commit if pre-commit fails.
- Avoid speculative refactors unrelated to the request.
- Prefer correctness and maintainability first, then optimize.
