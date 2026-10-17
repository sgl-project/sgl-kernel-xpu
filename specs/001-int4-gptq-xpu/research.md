# Research: INT4 GPTQ on Intel GPU

## 1. Kernel technology

**Decision:** Implement fused INT4 GPTQ GEMV and GEMM as a SYCL/C++ PyTorch
extension in this repository. Do not add a Triton twin.

**Rationale:** `src/CMakeLists.txt` already globs `src/sycl/*.cpp` into the
XPU extension. Ops are declared in `include/sgl_kernel_ops.h` and registered
with `m.def` plus `m.impl(..., torch::kXPU, ...)` in
`src/torch_extension_sycl.cc`. That is the constitution's extension contract.
`tests/test_gptq_kernel.py` already calls `sgl_kernel.gptq_gemm` with the
CUDA SGLang schema, and that symbol is not defined anywhere in this tree.
A second technology would need a Complexity Tracking justification. None is
warranted.

**Alternatives considered:**

- Triton XPU. Allowed by the constitution, but it would bypass the existing
  extension path and duplicate the op. Rejected.
- Reuse `awq_dequantize` (`src/sycl/awq_dequantize.cpp`). That kernel writes
  a full unpacked weight tensor and uses AWQ nibble order, not GPTQ
  `pack_rows` / `pack_cols`. Rejected as the steady-state path.
- Keep upstream SGLang's XPU path (`GPTQXPULinearKernel` on `main`), which
  unpacks GPTQ codes to `[K, N]`, repacks them with
  `aten._convert_weight_to_int4pack`, and calls `xpu_int4pack_mm`. The
  forward GEMM is fused, but load time materializes a full unpacked code
  tensor and the math does not live in this kernel library. Rejected as the
  steady-state path. The new `apply` replaces that call when the activation
  device is `xpu`.

## 2. Numeric contract

**Decision:** Match the reference in `tests/test_gptq_kernel.py`
`torch_dequantize`:

- `bit == 4`, `group_size == 128`.
- `b_q_weight` is int32 `[K/8, N]`, eight consecutive K nibbles per word
  (`pack_rows`).
- `b_gptq_qzeros` is int32 `[K/128, N/8]`, nibbles packed along N
  (`pack_cols`).
- `dequant = q * scale - effective_zero * scale`.
- Missing or v1 checkpoints (`zero_format=1`) use `effective_zero = stored + 1`.
  Declared v2 (`zero_format=2`) uses `effective_zero = stored`. Any other
  format is an error. The op argument defaults to 1 so a 7-argument call
  stays v1.
- Activations and scales are FP16 or BF16. The product is the same dtype.
- `b_g_idx` is int32 `[K]`. The kernel selects the scale and zero row from
  `g_idx` inside the fused loop. A contiguous index `i -> i/128` matches
  contiguous groups. No global unpacked weight is written.
- Acceptance generation is greedy (temperature 0) for 32 tokens. Sampling
  or any other temperature is rejected.
- Probe tolerance is the spec bound: the larger of 1% relative and 0.02
  absolute, per element.

**Rationale:** This is the packing SGLang's GPTQ checkpoint loader and the
existing test already assume. The current test's non-shuffle setup indexes
the float weight by group id (`b_fp[g_idx]`), which is not a permutation.
Implementation must correct that probe so it quantizes the original tile and
compares against `torch_dequantize`. The formula itself stays.

**Alternatives considered:** PyTorch int4pack layout as the public op.
Rejected because the acceptance checkpoint and `tests/test_gptq_kernel.py`
are GPTQ-packed, and a load-time unpack violates the steady-state rule.

## 3. SGLang dispatch

**Decision:** Register `gptq_gemm` only for `torch::kXPU`. In the test image,
patch `python/sglang/srt/hardware_backend/xpu/quantization/gptq_kernels.py`
so `GPTQXPULinearKernel` keeps the GPTQ packed parameters and its `apply`
calls `sgl_kernel.gptq_gemm` only when `x.device.type == "xpu"`. Do not edit
`python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py`.

**Rationale:** Upstream CUDA registers `gptq_gemm` on `torch::kCUDA` only.
Upstream `main` already isolates XPU in `GPTQXPULinearKernel`. Patching that
class is additive. A global monkeypatch is forbidden. The GitHub release
page for tag `v0.5.22` returned 404 on 2026-10-16, so the image must install
SGLang from source and fail the build unless the installed version is
>= 0.5.22. The patch must apply cleanly or the image build fails.

**Alternatives considered:** Call `_weight_int4pack_mm` from this repo.
Rejected in decision 1. Forking all of SGLang into this repository.
Rejected. The engine stays an image dependency. Only the XPU GPTQ class is
patched.

## 4. Image and ahead-of-time target

**Decision:** New `Dockerfile.xpu` based on
`intel/deep-learning-essentials:2026.1.3-devel-ubuntu24.04`, the base
already used by `Dockerfile.xpu_kernel`. Install `torch==2.14.0` from the
official XPU wheel index. Copy this repository into the image and build it
with `DPCPP_SYCL_TARGET=bmg`. Install SGLang >= 0.5.22 from source and apply
the XPU dispatch patch. Do not clone `sgl-kernel-xpu` from GitHub.

**Rationale:** The development host has no B70. `cmake/BuildFlags.cmake`
auto-detects the SYCL target from a local device and fails closed when none
is present. Setting `-DDPCPP_SYCL_TARGET=bmg` (Xe2 / Battlemage, which
covers the Arc Pro B70) compiles the AOT image inside the container. That
is the required cross-build. Ubuntu 24.10 is not required: the existing
oneAPI 2026.1.3 image already builds the `bmg` target.

**Alternatives considered:** A separate cross-compilation SDK stage on the
host. Unnecessary, because `icpx` inside the oneAPI image can AOT for `bmg`
without the GPU. Floating `SG_LANG_BRANCH=main` as in
`Dockerfile.xpu_kernel`. Rejected. The version floor must be checked.

## 5. Remote loop

**Decision:** `scripts/test-loop.sh` is the only operator entry. It requires
`REGISTRY`, defaults `IMAGE` to `$REGISTRY/sgl-kernel-xpu-gptq`, and accepts
`MODE` of `smoke`, `numeric`, `generate`, or `bench`. Stages, each aborting
the script on failure: `docker build -f Dockerfile.xpu`, `docker push`,
`kubectl delete job` (ignore missing), `kubectl apply -f
deploy/k8s/test-job.yaml`, `kubectl logs -f`. kubectl uses the default
context. No registry host is hard-coded.

The Job `sgl-gptq-xpu-test` sets `nodeName: hal`, requests
`gpu.intel.com/i915: 1`, and mounts host path `/tank/nas/models/` at the
same path. `generate` and `bench` load
`/tank/nas/models/Qwen3.8-27B-TURBO-DF735-882-NM-DAW-GPTQ-Int4-g128-tuned`.
`smoke` and `numeric` do not need the checkpoint. Weights are never copied
into the image.

**Rationale:** Matches constitution principles I, IV, and V and the spec's
single-command requirement. `MODE` selects the story under test without a
second script.

**Alternatives considered:** A Deployment or an unpinned Pod. Rejected. A
Job is one-shot and `nodeName: hal` cannot land on the other k3s node.

## 6. Entry marker and speed records

**Decision:** Smoke mode calls `gptq_entry_probe`, which logs the fixed
line `SGL_KERNEL_XPU_GPTQ_ENTRY` once. The fused GEMM does not print per
call. Generation and bench modes print one JSON line
`SGL_GPTQ_GENERATION_RECORD` with `stage` (`baseline` or `tuned`), the
prompt, `max_new_tokens` (32), the returned text, and `tokens_per_second`.
The tuned run receives `BASELINE_TPS` and exits non-zero unless its rate is
strictly higher. The fixed prompt is `Write a one-sentence hardware note.`

**Rationale:** SC-002 asks for a log marker on the path check, not on every
math call. SC-005 asks for two retained rates. The job log is the record.

**Alternatives considered:** Printing from inside the GEMM. Rejected. It
would dominate the bandwidth measurement.

## 7. Bandwidth pass

**Decision:** The first fused kernel dequantizes each group of 128 in
registers (or subgroup scratch) and accumulates the product without a global
unpacked tensor. GEMV is the `M == 1` specialization. GEMM covers `M > 1`.
After numeric and generation correctness, a follow-up change in the same
SYCL sources retunes vector loads and uses Xe2 XMX DPAS for tiles the
subgroup can issue. The bench mode on `hal` is the proof that tokens per
second rose. No second library is introduced.

**Rationale:** Fusion is required for correctness of the steady-state rule.
DPAS is a tuning of that kernel, which is what story 4 measures. The
constitution requires the Triton benchmark harness only when the kernel
itself is Triton. This kernel is SYCL, so `benchmark/bench_gptq_gemm.py`
times the op on `hal` in the repo's existing benchmark style.

**Alternatives considered:** Shipping the scalar kernel and a DPAS kernel as
two selectable steady-state paths. Rejected unless Complexity Tracking is
filled. The scalar body is the first revision of the same op. DPAS replaces
the inner accumulate once tiles match. The test reference stays off the
steady-state path.
