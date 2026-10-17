---
description: "Task list for INT4 GPTQ on Intel GPU"
---

# Tasks: INT4 GPTQ on Intel GPU

**Input**: Design documents from `/specs/001-int4-gptq-xpu/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: The spec does not request a test-first cycle. Each story ends with a `scripts/test-loop.sh` run whose evidence is the Job log from node `hal`. `tests/test_gptq_kernel.py` is corrected because it already calls `gptq_gemm` with a bad fixture. Any CPU-only check is marked `non_device` and is not B70 proof.

**Organization**: Tasks are grouped by user story so each story can be implemented and checked on its own.

**Spec clarifications (2026-10-16)**: Honor a per-channel group index. A missing or v1 zero is stored+1. A declared v2 zero is used as stored. Acceptance generation is greedy (temperature 0) and must return the same 32 tokens on two runs. These override any older v1-only wording in `research.md` or `contracts/gptq-gemm.md` until T003 refreshes those files.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Setup and foundational tasks have no story label

## Path Conventions

- SYCL sources: `src/sycl/` (already globbed by `src/CMakeLists.txt`)
- Declarations: `include/sgl_kernel_ops.h`
- Registration: `src/torch_extension_sycl.cc` (`m.def` + `m.impl` for `torch::kXPU` only)
- Python: `python/sgl_kernel/gemm.py`, `python/sgl_kernel/__init__.py`
- Remote loop: `Dockerfile.xpu`, `deploy/k8s/test-job.yaml`, `scripts/test-loop.sh`, `scripts/run_gptq_mode.py`
- Device probe: `tests/test_gptq_kernel.py`. Timings: `benchmark/bench_gptq_gemm.py`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Label non-device checks, land the mode script, and align design docs with the clarified spec

- [X] T001 [P] Register pytest marker `non_device` in `tests/conftest.py` with the description "CPU or schema check; not Intel Arc Pro B70 evidence"
- [X] T002 [P] Create `scripts/run_gptq_mode.py` that reads `MODE` (`smoke`, `numeric`, `generate`, `bench`) and defines `MARKER = "SGL_KERNEL_XPU_GPTQ_ENTRY"`, `PROMPT = "Write a one-sentence hardware note."`, `MAX_NEW_TOKENS = 32`, `TEMPERATURE = 0`, `MODEL_DIR = "/tank/nas/models/Qwen3.8-27B-TURBO-DF735-882-NM-DAW-GPTQ-Int4-g128-tuned"`, and record prefix `SGL_GPTQ_GENERATION_RECORD`. Unknown or not-yet-implemented modes exit non-zero with the mode name in the message
- [X] T003 [P] Update `specs/001-int4-gptq-xpu/contracts/gptq-gemm.md`, `specs/001-int4-gptq-xpu/data-model.md`, and the numeric decision in `specs/001-int4-gptq-xpu/research.md` so they match spec FR-021, FR-022, and FR-018: honor `b_g_idx`, apply a missing or v1 zero as stored+1, apply a declared v2 zero as stored, reject any other format, and require greedy temperature 0 for the 32-token check

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: XPU-only op skeletons the image can compile before any story body is filled in

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Declare `gptq_entry_probe()` and `gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, Tensor b_gptq_scales, Tensor b_g_idx, bool use_shuffle, int bit) -> Tensor` in `include/sgl_kernel_ops.h` using the argument rules in `specs/001-int4-gptq-xpu/contracts/gptq-gemm.md` after T003
- [X] T005 [P] Add `src/sycl/gptq_entry_probe.cpp` including `<torch/all.h>` (never `<torch/extension.h>`) whose exported `gptq_entry_probe` raises a clear not-implemented error. Do not edit `src/CMakeLists.txt`; `src/sycl/*.cpp` is already globbed
- [X] T006 [P] Add `src/sycl/gptq_gemm.cpp` including `<torch/all.h>` whose exported `gptq_gemm` raises a clear not-implemented error, with the same CMake rule as T005
- [X] T007 [P] Export `gptq_gemm` and `gptq_entry_probe` from `python/sgl_kernel/gemm.py` through `torch.ops.sgl_kernel`, and re-export both from `python/sgl_kernel/__init__.py`
- [X] T008 Register both ops in `src/torch_extension_sycl.cc` with `m.def` schemas and `m.impl(..., torch::kXPU, ...)` only. Do not add a `torch::kCUDA` impl. Depends on T004, T005, and T006

**Checkpoint**: The extension exposes both symbols on XPU and no other device key

---

## Phase 3: User Story 1 - Prove the math path on the GPU node (Priority: P1) 🎯 MVP

**Goal**: One command builds an image, publishes it, starts a Job only on `hal`, and streams a log that contains the entry marker

**Independent Test**: `MODE=smoke ./scripts/test-loop.sh` with `REGISTRY` set. The streamed log contains `SGL_KERNEL_XPU_GPTQ_ENTRY`, and a second run replaces the previous Job

### Implementation for User Story 1

- [X] T009 [US1] Implement `gptq_entry_probe` in `src/sycl/gptq_entry_probe.cpp` so a call on XPU logs one line containing `SGL_KERNEL_XPU_GPTQ_ENTRY` and returns without launching the GEMM
- [X] T010 [P] [US1] Implement the `smoke` branch in `scripts/run_gptq_mode.py` to call `sgl_kernel.gptq_entry_probe` after selecting the XPU device, and exit 0 only when that call returns
- [X] T011 [P] [US1] Write `Dockerfile.xpu` from `intel/deep-learning-essentials:2026.1.3-devel-ubuntu24.04`, install `torch==2.14.0` from the official XPU wheel index, `COPY` this repository (do not `git clone` sgl-kernel-xpu), and `pip install` it with `DPCPP_SYCL_TARGET=bmg`. Install SGLang from source and fail the build if the installed version is below 0.5.22. Do not copy `/tank/nas/models` into the image. Default command runs `scripts/run_gptq_mode.py`
- [X] T012 [P] [US1] Write `deploy/k8s/test-job.yaml` as Job `sgl-gptq-xpu-test` with `backoffLimit: 0`, `restartPolicy: Never`, `nodeName: hal`, resource request and limit `gpu.intel.com/i915: 1`, and hostPath `/tank/nas/models` mounted at `/tank/nas/models`. Image, tag, and `MODE` come from placeholders the loop script substitutes
- [X] T013 [US1] Write `scripts/test-loop.sh` (`set -euo pipefail`) per `specs/001-int4-gptq-xpu/contracts/test-loop.md`: require `REGISTRY` (no default host), default `IMAGE` to `$REGISTRY/sgl-kernel-xpu-gptq` and `TAG` to `dev`, `docker build -f Dockerfile.xpu`, `docker push`, `kubectl delete job sgl-gptq-xpu-test --ignore-not-found`, `kubectl apply` the substituted `deploy/k8s/test-job.yaml`, then `kubectl logs -f job/sgl-gptq-xpu-test`. Use the default kubectl context. Exit non-zero if the Job fails. A failed build or push must not apply a Job. Depends on T011 and T012
- [X] T014 [US1] Run `MODE=smoke ./scripts/test-loop.sh` and confirm the streamed log from node `hal` contains `SGL_KERNEL_XPU_GPTQ_ENTRY`. Run it again and confirm a single new Job is streamed. Depends on T009, T010, and T013

**Checkpoint**: Story 1 is done when the smoke log from `hal` shows the marker. A local image build is not enough

---

## Phase 4: User Story 2 - Match grouped INT4 weight math (Priority: P2)

**Goal**: Packed INT4 group-128 weights produce a single-row and a many-row product that matches the unpack reference for FP16 and BF16, including a non-contiguous group index and both zero formats

**Independent Test**: `MODE=numeric ./scripts/test-loop.sh`. The `hal` log reports both dtypes, both shapes, a non-contiguous group index, and both v1 and v2 zeros with zero tolerance failures

### Implementation for User Story 2

- [X] T015 [US2] Replace the stub in `src/sycl/gptq_gemm.cpp` with a fused GEMV (`M == 1`) and GEMM (`M > 1`). Dequantize inside the kernel from packed weights: group size 128, `bit == 4`, scale and zero row chosen by `b_g_idx` (a contiguous index `i -> i/128` must match contiguous groups). A missing or v1 zero uses `stored_nibble + 1`. A declared v2 zero uses `stored_nibble` as-is. `result = (q - effective_zero) * scale`. Do not write an unpacked weight tensor. Accept FP16 and BF16. `TORCH_CHECK` with a visible message and no output tensor when `bit != 4`, `use_shuffle` is true, the format is neither missing/v1 nor declared v2, `K` is not divisible by 128, `N` is not divisible by 8, `b_g_idx` length is not `K`, dtypes differ, or zero/scale shapes do not match `K/128`. Include `<torch/all.h>`
- [X] T016 [P] [US2] Fix the fixture in `tests/test_gptq_kernel.py` so it quantizes the original `[K, N]` tile (do not index that tile by group id), keeps group size 128, covers `M == 1` and `M > 1`, adds `torch.bfloat16` beside `torch.float16`, covers one non-contiguous `b_g_idx` and both v1 (`stored+1`) and declared v2 (stored as-is) zeros, and compares with `max(0.01 * abs(ref), 0.02)`. Skip the module when XPU is unavailable. Leave it unlabeled as `non_device`
- [X] T017 [P] [US2] Add `tests/test_gptq_reference.py` marked `non_device` that checks the FP32 formula on CPU for v1 (`effective_zero = stored_nibble + 1`) and declared v2 (`effective_zero = stored_nibble`), both as `(q - effective_zero) * scale`, including one non-contiguous group index. The file must not claim a B70 result
- [X] T018 [US2] Implement the `numeric` branch in `scripts/run_gptq_mode.py` to run the FP16 and BF16 probes for `M == 1` and `M > 1` on XPU, once with a non-contiguous group index and once for v1 and declared v2, print a pass count, and exit non-zero if any element exceeds the tolerance, if a mismatched zero fixture fails to raise, or if an unknown format fails to raise. Depends on T015
- [X] T019 [US2] Run `MODE=numeric ./scripts/test-loop.sh` and confirm the streamed `hal` log shows zero probe failures for both dtypes, both shapes, the non-contiguous index, and both zero formats. Depends on T018

**Checkpoint**: Story 2 is done when the numeric Job log on `hal` is clean. `tests/test_gptq_reference.py` does not count

---

## Phase 5: User Story 3 - Generate text from the acceptance model (Priority: P3)

**Goal**: The named checkpoint returns the same 32 greedy tokens on two `hal` runs, and only an XPU activation uses the new op

**Independent Test**: `MODE=generate ./scripts/test-loop.sh` twice at temperature 0. Both records contain the same 32 tokens. A missing model directory fails without a success record. A non-zero temperature is rejected

### Implementation for User Story 3

- [X] T020 [P] [US3] Write `deploy/sglang-gptq-xpu.patch` against `python/sglang/srt/hardware_backend/xpu/quantization/gptq_kernels.py` so `GPTQXPULinearKernel.process_weights_after_loading` keeps GPTQ `qweight`, `qzeros`, `scales`, and `g_idx` (no `aten._convert_weight_to_int4pack`, no stored full-precision weight), records whether the checkpoint is missing/v1 or declared v2, and `apply` calls `sgl_kernel.gptq_gemm` only when `x.device.type == "xpu"`, raising otherwise. Do not modify `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py`
- [X] T021 [US3] Update `Dockerfile.xpu` to apply `deploy/sglang-gptq-xpu.patch` and fail the image build if the patch does not apply or if SGLang is below 0.5.22. Depends on T020
- [X] T022 [P] [US3] Implement the `generate` branch in `scripts/run_gptq_mode.py`: reject an empty prompt, and reject sampling or any temperature other than 0, each with a non-zero exit and no record; if `MODEL_DIR` is missing or unreadable, log that path and exit non-zero with no `SGL_GPTQ_GENERATION_RECORD`; otherwise load the checkpoint through the patched SGLang engine, generate 32 tokens for `PROMPT` at temperature 0, and print one JSON record with `stage`, `model`, `prompt`, `max_new_tokens` 32, `temperature` 0, `text`, `tokens_per_second`, and `node` `hal`
- [X] T023 [US3] Run `MODE=generate ./scripts/test-loop.sh` twice. Confirm both `hal` logs show a record with `temperature` 0 and the same 32 tokens, the Job stayed on `hal`, a missing model path prints that path and no success record, and a non-zero temperature is rejected. Depends on T021 and T022

**Checkpoint**: Story 3 is done when two greedy `hal` generation logs agree and the CUDA GPTQ file is untouched

---

## Phase 6: User Story 4 - Record and raise generation speed (Priority: P4)

**Goal**: After a correct greedy generation, a memory-access pass on the same SYCL op records a higher tokens-per-second rate

**Independent Test**: `MODE=bench BASELINE_TPS=<first correct rate> ./scripts/test-loop.sh` exits 0 only when the tuned rate is higher, both records use temperature 0, and both remain in the Job logs

### Implementation for User Story 4

- [X] T024 [US4] Retune loads in `src/sycl/gptq_gemm.cpp` and use Xe2 XMX DPAS for tiles the subgroup can issue. Keep one op, no Triton file, and no global unpacked weight tensor. Numeric results must still meet the T015 tolerance for both zero formats and a non-contiguous group index
- [X] T025 [P] [US4] Add `benchmark/bench_gptq_gemm.py` that times `gptq_gemm` on XPU for an `M == 1` shape and an `M > 1` shape and prints the rates. Do not use the Triton benchmark harness. Skip with a message when XPU is unavailable so a laptop run is not a B70 result
- [X] T026 [US4] Implement the `bench` branch in `scripts/run_gptq_mode.py` to print `SGL_GPTQ_GENERATION_RECORD` with `temperature` 0, `max_new_tokens` 32, and `stage` `baseline` or `tuned` (from env `STAGE`, default `baseline`), and exit non-zero when `BASELINE_TPS` is set and `tokens_per_second` is not strictly greater. Reject a non-zero temperature the same way as T022. Depends on T022's record shape
- [X] T027 [US4] Run `MODE=bench ./scripts/test-loop.sh` to capture the baseline record, then run `MODE=bench STAGE=tuned BASELINE_TPS=<that rate> ./scripts/test-loop.sh` after T024. Confirm the script exits 0, both records use temperature 0, and both are in the `hal` logs. Depends on T024 and T026

**Checkpoint**: Story 4 is done when the tuned greedy `hal` log beats the stored baseline on the same prompt

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Keep the operator guide and the image contract aligned with the scripts

- [X] T028 [P] Update `specs/001-int4-gptq-xpu/quickstart.md` so every command matches `scripts/test-loop.sh` (`REGISTRY`, `MODE`, `BASELINE_TPS`, `STAGE`) and the generate and bench sections state temperature 0
- [X] T029 [P] Check `Dockerfile.xpu` contains no `COPY` of `/tank/nas/models` and `scripts/test-loop.sh` exits before `docker build` when `REGISTRY` is unset, with no default registry host

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies. T003 must finish before T004 quotes the contract
- **Foundational (Phase 2)**: Depends on Setup. Blocks every user story
- **User Story 1 (Phase 3)**: Depends on Foundational. No dependency on US2, US3, or US4
- **User Story 2 (Phase 4)**: Depends on Foundational and on the US1 loop (`scripts/test-loop.sh`, `Dockerfile.xpu`). The GEMM source does not depend on the smoke probe body
- **User Story 3 (Phase 5)**: Depends on US2 (`gptq_gemm` correct for both zero formats and `b_g_idx`) and on the US1 image and Job
- **User Story 4 (Phase 6)**: Depends on a correct greedy US3 generation record to supply `BASELINE_TPS`
- **Polish (Phase 7)**: Depends on the scripts existing (US1 onward)

### User Story Dependencies

- **User Story 1 (P1)**: First runnable slice. Later stories call the same script with a different `MODE`
- **User Story 2 (P2)**: Reuses the loop. Does not require the SGLang patch
- **User Story 3 (P3)**: Requires the fused op and the patch applied inside `Dockerfile.xpu`
- **User Story 4 (P4)**: Requires one correct greedy 32-token rate from US3

### Within Each User Story

- US1: probe body and smoke branch before the hal run. Dockerfile and Job manifest before `test-loop.sh`
- US2: fused kernel before the numeric hal run. The CPU reference file can be written beside the kernel
- US3: patch file before the Dockerfile applies it. Generate branch can be written beside the patch
- US4: DPAS retune and the bench script before the comparison hal run

### Parallel Opportunities

- T001, T002, and T003
- T005, T006, and T007 after T004
- T010, T011, and T012 after the foundational checkpoint (T009 can join them; different files)
- T016 and T017 beside T015
- T020 and T022
- T024 and T025
- T028 and T029

---

## Parallel Example: User Story 1

```bash
# After T008, different files:
Task: "T009 implement gptq_entry_probe in src/sycl/gptq_entry_probe.cpp"
Task: "T010 smoke branch in scripts/run_gptq_mode.py"
Task: "T011 write Dockerfile.xpu"
Task: "T012 write deploy/k8s/test-job.yaml"

# Then, in order:
Task: "T013 write scripts/test-loop.sh"
Task: "T014 MODE=smoke ./scripts/test-loop.sh on hal"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 and Phase 2
2. Complete Phase 3 (T009 through T014)
3. Stop. The smoke log from `hal` is the MVP evidence

### Incremental Delivery

1. Setup + Foundational → design docs match the clarified spec, and XPU symbols link
2. User Story 1 → one-command smoke on `hal`
3. User Story 2 → numeric Job on `hal` for both zero formats and a non-contiguous group index
4. User Story 3 → two greedy 32-token generations from the NAS checkpoint agree
5. User Story 4 → tuned tokens/sec higher than the baseline record

### Parallel Team Strategy

1. Together: Phase 1 and Phase 2
2. After that, one person can own `Dockerfile.xpu` and `scripts/test-loop.sh` while another owns `src/sycl/gptq_entry_probe.cpp`
3. After US1, the kernel file (`src/sycl/gptq_gemm.cpp`) and the SGLang patch (`deploy/sglang-gptq-xpu.patch`) are different files, but the patch must not merge until the GEMM contract in T015 holds

---

## Notes

- [P] tasks use different files and do not depend on an unfinished task
- Story labels map to spec.md priorities P1 through P4
- Acceptance evidence for US1–US4 is `kubectl logs` from node `hal`, not a local compile
- `tests/test_gptq_reference.py` is `non_device`. Do not cite it as B70 proof
- Commit after each task or logical group
