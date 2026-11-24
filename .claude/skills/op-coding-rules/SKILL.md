---
name: op-coding-rules
description: sgl-kernel-xpu op coding rules for new and changed code (no host-device sync; graph-capture compatible, current queue, pool side queues joined by events; CuTe/Collective preferred, Device API needs a reason; plain layouts unless declared; torch-allocated memory; no persistent device tensors; respect the user's input data types with no silent precision reduction; sycl-tla public headers only) plus a diff scanner. Use when generating, modifying, optimizing or reviewing any kernel, host launcher or Python op wrapper under src/, include/ or python/sgl_kernel/, when choosing a SYCL-TLA API layer, adding a workspace/split-KV buffer, deciding grid size or kernel selection, adding dtype dispatch or FP8 scales, or adding a prepack API.
---

# Op coding rules (sgl-kernel-xpu)

An **op** is a host launcher plus its kernel(s) and its Python wrapper.
Principle: an op is a pure function of its input arguments, stateless and
self-contained, so SGLang can call it inside graph capture and in a
high-rate serving loop. Host caches that never change results (device
properties, config lookup) are not op state.

These rules apply to every **new or changed line**. Existing code predates
them and is not rewritten for them; do not copy those patterns into new
code and do not expand them. Pre-existing violations in a touched file are
non-blocking notes unless the change makes them worse.

## Rules

| ID | Rule | Status | Reference |
|---|---|---|---|
| OP-1 | No host-device sync | Disallowed | [stateless-op.md](references/stateless-op.md#op-1-no-host-device-sync) |
| OP-2 | Graph-capture compatible: grid/kernel choice from host-known info only; all work on the user's current queue; side queues from the torch pool, joined back by events | Disallowed | [stateless-op.md](references/stateless-op.md#op-2-graph-capture-compatible) |
| OP-3 | SYCL-TLA API layer: CuTe preferred, Collective and Kernel allowed, Device API (`*Adapter`, `device::`) needs a written reason and must run on the current queue | Discouraged (Device) | [stateless-op.md](references/stateless-op.md#op-3-sycl-tla-api-layer) |
| OP-4 | Plain layout for all inputs/outputs unless the interface declares a blocked layout; weights via a dedicated prepack API | Disallowed | [stateless-op.md](references/stateless-op.md#op-4-plain-layout-unless-declared) |
| OP-5 | All global memory (outputs, workspace) through the torch allocator | Disallowed | [stateless-op.md](references/stateless-op.md#op-5-global-memory-through-the-torch-allocator) |
| OP-6 | No device tensors or device pointers kept across calls | Disallowed | [stateless-op.md](references/stateless-op.md#op-6-no-persistent-device-tensors) |
| OP-7 | Respect the user's input data types: kernel choice from input dtypes; no silent downcast or common-dtype cast; FP8 operands need user scales (never default 1.0); output dtype follows the API contract; FP32 accumulation | Disallowed | [precision.md](references/precision.md) |
| TLA-1 | Use SYCL-TLA public headers (`include/`, `tools/util/include/`) only; never copy-include kernel code from sycl-tla `applications/` or `examples/` | Disallowed | [stateless-op.md](references/stateless-op.md#tla-1-sycl-tla-public-headers-only) |

Int64 strides and offsets are a separate rule set: see the
[stride-overflow-safety](../stride-overflow-safety/SKILL.md) skill.

## When generating or modifying code

1. Pick the SYCL-TLA layer first (OP-3). Default to CuTe; Collective is fine
   for plain GEMM-like mainloops. Reuse an existing kernel in `src/sycl/`
   before writing a new one.
2. Host launcher: shapes, dtypes and host tensors only decide grid size and
   kernel variant. Use host-known upper bounds (`max_seqlen`, `num_tokens`)
   and let the kernel read per-call values (`cu_seqlens`, counts) on device.
3. Allocate outputs and every workspace with `at::empty` / `at::zeros` (or
   `torch.empty` in the Python wrapper) per call. The caching allocator makes
   this cheap; do not cache it.
4. Launch on `at::xpu::getCurrentXPUStream().queue()`. A side queue comes
   from `c10::xpu::getStreamFromPool()` and joins back via events.
5. Pick the kernel from the input dtypes (OP-7): reject unsupported dtype
   combinations, require and validate every FP8 scale, allocate the output
   in the documented dtype, accumulate in FP32.
6. Before finishing, run the scanner on your changes and fix or justify
   every hit:
   `python3 .claude/skills/op-coding-rules/scan_op_rules.py --diff`.
7. Ops whose grid or kernel choice depends on shapes, and that SGLang calls
   inside graph capture, need a capture/replay test (see the graph tests in
   `tests/test_flash_attention.py`).

## When optimizing

A speedup never justifies breaking a rule. Common traps:

- Caching a workspace across calls to save allocation time (OP-6).
- Reading `cu_seqlens` / token counts to the host to pick splits or grid
  (OP-1, OP-2). Pick from host-known shapes, or let the kernel early-exit.
- Requiring callers to repack activations into a kernel-private layout
  (OP-4). Pack inside the kernel, or add a declared prepack API for weights.
- Wrapping a Device API adapter to reuse a tuned path (OP-3). Launch the
  kernel functor yourself on the current queue.
- Quantizing a BF16/FP16 input to FP8 inside the op, or defaulting a missing
  scale to 1.0, to reach a faster kernel (OP-7). Add an explicit opt-in API
  argument that takes the scales instead.

If an optimization seems to require a violation, stop and ask the user.

## When reviewing

1. Run `python3 .claude/skills/op-coding-rules/scan_op_rules.py --diff <base>`
   (default base `origin/main`). It checks only lines the change adds. Hits
   are candidates: confirm each one in context (e.g. `.cpu()` in a test
   helper or a documented host-side planning API is fine).
2. Check what the scanner cannot see:
   - grid size / kernel selection derived from device data (OP-2);
   - side queues not from the torch pool, or not joined back to the current
     queue via events (OP-2);
   - new interface args with a non-plain layout that is not declared in the
     docstring/API name (OP-4);
   - workspace fallbacks such as `ws ? ws : malloc` (OP-5);
   - raw pointers or tensors stored in static/global/member state (OP-6);
   - a new Device API use without a written reason (OP-3);
   - dtype dispatch that casts mismatched inputs instead of rejecting them,
     a scale that is optional for an FP8 operand, or an output dtype that
     does not follow the API contract (OP-7).
3. Report each confirmed finding as
   `[RULE-ID] path:line — what is wrong — required fix`.
   Disallowed items block the change; a Device API use without a reason is a
   request for one.

`--all` audits every file under `src/`, `include/` and `python/`. Existing
code has many hits; use it to find patterns, not as a PR gate.
