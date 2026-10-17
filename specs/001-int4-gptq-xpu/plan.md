# Implementation Plan: INT4 GPTQ on Intel GPU

**Branch**: `001-int4-gptq-xpu` | **Date**: 2026-10-16 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-int4-gptq-xpu/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

Operators need the named INT4 GPTQ checkpoint to generate text on the Arc
Pro B70, with grouped weight math that stays packed, and one command that
builds an image, publishes it, and streams a Job locked to node `hal`.

The math path is a new SYCL op `gptq_gemm` in this repository, registered
only for `device="xpu"`. The test image patches SGLang's existing
`GPTQXPULinearKernel` so that class calls this op and stops unpacking
weights into PyTorch's int4pack layout. `scripts/test-loop.sh` drives
`Dockerfile.xpu` and `deploy/k8s/test-job.yaml`. Research decisions are in
[research.md](research.md).

## Technical Context

**Language/Version**: C++/SYCL (C++20, DPC++ from the oneAPI image) and
Python 3.12. PyTorch == 2.14.0 from the XPU wheel index, which satisfies
>= 2.14.0+xpu.

**Primary Dependencies**: SGLang >= 0.5.22 installed from source in the
image. Intel oneAPI via `intel/deep-learning-essentials:2026.1.3-devel-ubuntu24.04`.
No Triton kernel.

**Storage**: NAS host path `/tank/nas/models/` mounted into the Job.
Acceptance weights stay on that volume.

**Testing**: `scripts/test-loop.sh` runs a kubectl Job on node `hal`.
CPU-only checks of the unpack formula, if added under `tests/`, are labeled
non-device and are not acceptance evidence.

**Target Platform**: Intel Arc Pro B70 (32 GB) on k3s node `hal`. AOT
target `bmg` (Xe2).

**Project Type**: XPU kernel library plus a device-scoped SGLang dispatch
patch applied at image build.

**Performance Goals**: Fused INT4 dequant-GEMV (`M == 1`) and dequant-GEMM
(`M > 1`) with no steady-state unpacked weight tensor. After that kernel is
correct, a SYCL memory-access and Xe2 XMX DPAS pass must raise tokens per
second on the fixed 32-token prompt.

**Constraints**: No Intel GPU on the development host. One script performs
build, push, `kubectl apply`, and `kubectl logs -f` on the default context.
The Job sets `nodeName: hal` and `gpu.intel.com/i915: 1`. `REGISTRY` is
required and is not hard-coded. Group size is 128 and `bit` is 4 only.

**Scale/Scope**: One checkpoint,
`Qwen3.8-27B-TURBO-DF735-882-NM-DAW-GPTQ-Int4-g128-tuned`. Dense GPTQ
linear only. No other quant formats, group sizes, or nodes.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Pre-research result: PASS. The technical context above commits to every
gate. No Complexity Tracking row is required.

- **I. Hardware-Truth Testing**: PASS. Acceptance is the `hal` Job log.
  Local builds and CPU formula checks are labeled non-evidence.
- **II. Bandwidth-Efficient INT4 Kernels**: PASS. One SYCL fused GEMV/GEMM.
  The FP32 unpack oracle stays in tests. Upstream int4pack unpack is not
  the steady-state path.
- **III. Device-Scoped SGLang Routing**: PASS. `m.impl` uses `torch::kXPU`
  only. The patch touches `GPTQXPULinearKernel` and checks
  `x.device.type == "xpu"`. The CUDA GPTQ file is not edited.
- **IV. Automated Remote Test Loop**: PASS. `scripts/test-loop.sh` is the
  single command. `Dockerfile.xpu` uses the Intel oneAPI image and sets
  `DPCPP_SYCL_TARGET=bmg` so SYCL AOT does not need a local GPU.
- **V. Pinned Model and Node Contract**: PASS. Job `nodeName: hal` and
  host path `/tank/nas/models/` loading the named checkpoint.
- **Stack floor**: PASS. torch==2.14.0+xpu. Image build fails if SGLang is
  below 0.5.22. The `v0.5.22` GitHub release URL 404'd on 2026-10-16, so
  the pin is a version check rather than that tag.
- **Extension contract**: PASS. `m.def` plus `m.impl` in
  `src/torch_extension_sycl.cc`. SYCL sources include `<torch/all.h>`.

### Post-design re-check

PASS. Phase 1 contracts do not add a second kernel technology, a CUDA
registration, an unpinned Job, or a weight copy inside the image. The
DPAS work is a later revision of the same SYCL op, gated by the bench
mode, not a parallel steady-state implementation.

## Project Structure

### Documentation (this feature)

```text
specs/001-int4-gptq-xpu/
├── plan.md              # This file
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/           # Phase 1
│   ├── gptq-gemm.md
│   ├── test-loop.md
│   ├── test-job.md
│   └── sglang-dispatch.md
└── tasks.md             # Phase 2 (/speckit-tasks, not this command)
```

### Source Code (repository root)

```text
Dockerfile.xpu                         # oneAPI image, local kernel, SGLang floor
deploy/k8s/test-job.yaml               # Job pinned to hal
deploy/sglang-gptq-xpu.patch           # XPU GPTQ class only
scripts/test-loop.sh                   # build, push, apply, logs
scripts/run_gptq_mode.py               # smoke, numeric, generate, bench

include/sgl_kernel_ops.h               # declare gptq_gemm, gptq_entry_probe
src/torch_extension_sycl.cc            # m.def + m.impl for kXPU
src/sycl/gptq_gemm.cpp                 # fused GEMV and GEMM

python/sgl_kernel/gemm.py              # Python wrappers
python/sgl_kernel/__init__.py          # export wrappers

tests/test_gptq_kernel.py              # device probe, skip without XPU
benchmark/bench_gptq_gemm.py           # hal timings for the SYCL op
```

**Structure Decision**: This feature extends the existing XPU kernel
package. New SYCL sources land in `src/sycl/` so the current CMake glob
picks them up. The remote loop is `Dockerfile.xpu`, `deploy/k8s/`, and
`scripts/`. SGLang is not vendored. The image build checks out SGLang
>= 0.5.22 and applies `deploy/sglang-gptq-xpu.patch`.

## Complexity Tracking

No constitution violations. No second kernel technology and no unfused
steady-state path are introduced, so this table is empty.
