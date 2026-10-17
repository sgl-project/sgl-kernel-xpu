<!--
Sync Impact Report
- Version change: 1.0.0 → 1.1.0
- Modified principles: none (I–V unchanged)
- Added sections:
  - VI. Untracked Assistant Artifacts
  - VII. Sparse Practical Comments
  - VIII. Human Authorship in Version Control
- Removed sections: none
- Templates:
  - .specify/templates/plan-template.md ✅ updated (Constitution Check gates VI–VIII)
  - .specify/templates/spec-template.md ✅ updated (Project Constraints)
  - .specify/templates/tasks-template.md ✅ updated (always-on artifact, comment, and authorship tasks)
  - .specify/templates/checklist-template.md ✅ no change required
  - .cursor/skills/speckit-*/SKILL.md ✅ no outdated agent-specific references
  - README.md ✅ no change required (existing pointer still accurate)
  - .gitignore ✅ updated (assistant scratch; specs/ and .specify/ remain tracked)
- Follow-up TODOs: none
-->

# sgl-kernel-xpu Constitution

## Core Principles

### I. Hardware-Truth Testing (NON-NEGOTIABLE)

XPU correctness and performance claims MUST be validated on the Intel Arc Pro
B70 (32 GB) attached to Kubernetes node `hal`. The local development machine
has no Intel GPU. A change that only configures, compiles, type-checks, or
runs on CPU or CUDA MUST NOT be treated as validation.

Rationale: SYCL and Triton-XPU kernels do not execute on the development host.
Accepting a local build as proof ships an untested INT4 path.

### II. Bandwidth-Efficient INT4 Kernels

INT4 GPTQ weight-only dequantization, and the GEMV or GEMM that consumes those
weights, MUST be implemented as Triton-XPU kernels or as raw SYCL/C++ PyTorch
extensions. Kernels MUST keep the B70 memory bus busy: weights stay packed in
device memory, and dequantization MUST be fused into the math path whenever
that fusion is feasible. A steady-state path that first materializes a full
unpacked weight tensor is forbidden when a fused dequant-GEMM or dequant-GEMV
path can produce the same numeric result.

Rationale: INT4 GPTQ on a 32 GB Arc Pro B70 is memory-bandwidth bound.
Unpacking weights before the GEMM spends the bandwidth budget twice.

### III. Device-Scoped SGLang Routing

SGLang GPTQ INT4 calls MUST dispatch to this repository's XPU kernel when
`device="xpu"` is detected. Registration MUST be additive. CUDA and every
other device implementation MUST keep their existing dispatch. The device
check MUST read the tensor or module device. A global patch that forces the
XPU kernel onto every device is forbidden.

Rationale: This tree is an SGLang XPU fork. A routing change that hijacks
non-XPU calls breaks the upstream contract the fork still has to carry.

### IV. Automated Remote Test Loop

Every change to kernel behavior or SGLang dispatch MUST be runnable through
one shell script that performs these stages with no manual step between them:

1. Build the container image locally.
2. Push that image to the container registry the k3s cluster can pull.
3. `kubectl apply` a Pod or Job with the default kubectl context.
4. Follow the workload with `kubectl logs -f` until it finishes or fails.

The Dockerfile MUST use an Intel oneAPI base image, a PyTorch XPU base image,
or both. When the development host cannot compile SYCL for the B70, that
compilation MUST happen inside the image build, including a cross-compilation
stage when the base image requires one. The script MUST fail if any stage
fails. The registry host and image tag MUST be parameters of the script, not
hard-coded secrets.

Rationale: Hardware lives on `hal`, not on the laptop. An unscripted deploy
loop hides the only environment that can falsify a kernel change.

### V. Pinned Model and Node Contract

Any Pod or Job that loads the acceptance model MUST:

- Schedule only on node `hal`, using `nodeName: hal` or a `nodeSelector` that
  matches no other node.
- Mount host path `/tank/nas/models/` into the container.
- Load weights from
  `/tank/nas/models/Qwen3.8-27B-TURBO-DF735-882-NM-DAW-GPTQ-Int4-g128-tuned`.

The workload MUST NOT schedule onto the other node of the 2-node k3s cluster.
Model weights MUST NOT be copied into the image.

Rationale: The named INT4 GPTQ checkpoint is the acceptance artifact, and it
already resides on the NAS attached to `hal`. Embedding weights in the image
makes every kernel iteration a multi-gigabyte push.

### VI. Untracked Assistant Artifacts

AI-specific tracking files, agent state logs, `.cursor/` directories, and
Spec Kit generation scratch outside `specs/` and `.specify/` MUST NOT be
committed. `.gitignore` MUST list those paths, including file extensions and
hidden folders that coding assistants write during background execution.
Required ignore entries include `.cursor/`, `*.cursor-*`, agent transcripts
and agent state, and Claude Code local scratch under `.claude/` that is not
a tracked project skill. `specs/` and `.specify/` are this repository's Spec
Kit workflow and MUST stay tracked. Ignoring either tree, or any path that
would drop the constitution or a feature spec from version control, is
forbidden.

Rationale: Assistant indexes, transcripts, and generation scratch are local
tool state. Feature specs and this constitution are the project's own
documents and stay in version control.

### VII. Sparse Practical Comments

Comments MUST be sparse and practical. A comment MUST explain why complex
logic exists. A comment MUST NOT restate what ordinary code does. Robotic,
overly verbose, or declarative comments are forbidden, including "This
function calculates the total", "Generated by AI", and "Optimized for
performance". Comments MUST NOT announce that they were machine-written.

Rationale: Comments that narrate the next line, or advertise their author,
hide the few notes a reader actually needs.

### VIII. Human Authorship in Version Control

Git commits, commit messages, and pull requests MUST read as the author's
own work. Prefixes and tags that advertise AI generation are forbidden in
commit summaries and pull request descriptions, including "AI:",
"Generated:", and "Cursor agent updated". Messages MUST follow ordinary
human conventions, such as `fix: resolve rendering bug in data panel` or
`feat: add risk layer endpoints`, and MUST NOT include meta-commentary
about a prompt. Satisfying this principle MUST NOT change git config and
MUST NOT rewrite history.

Rationale: The history of this repository is the contributor's record. Tool
attribution and prompt narration do not belong in that record, and rewriting
history to remove them is itself a breach.

## Target Stack

- GPU: Intel Arc Pro B70, 32 GB VRAM, Kubernetes node `hal`.
- Cluster: 2-node k3s. kubectl MUST use the default context.
- Framework: PyTorch >= 2.14.0+xpu.
- Inference engine: SGLang >= 0.5.22.
- Acceptance model: `Qwen3.8-27B-TURBO-DF735-882-NM-DAW-GPTQ-Int4-g128-tuned`
  (INT4 GPTQ, group size 128).
- Images: built on the development host from Intel oneAPI and/or PyTorch XPU
  bases, pushed to the registry the cluster can pull.
- Kernel sources follow the repository extension path: implementation under
  the SYCL/C++ tree, declaration in `include/sgl_kernel_ops.h`, torch
  registration, `CMakeLists.txt` source list, and a Python binding under
  `python/sgl_kernel`.

A plan that lowers any version floor, swaps the acceptance model, or targets
a GPU other than the Arc Pro B70 MUST record that deviation in Complexity
Tracking and MUST NOT present it as the default contract.

## Development Workflow and Quality Gates

Plans and pull requests MUST pass every gate below before a kernel or
dispatch change is called done:

1. **Constitution Check.** `/speckit-plan` records pass or fail for
   Principles I–VIII. A fail is allowed only with a row in Complexity Tracking.
2. **Image contract.** The Dockerfile bases on Intel oneAPI and/or
   PyTorch XPU, and the build compiles SYCL for the B70 without requiring
   an Intel GPU on the development host.
3. **Schedule contract.** The test manifest sets `nodeName: hal` or an
   equivalent selector, and mounts `/tank/nas/models/` when the model is
   loaded.
4. **Loop contract.** `build`, registry push, `kubectl apply`, and
   `kubectl logs -f` run as one script against the default kubectl context.
5. **Routing contract.** New GPTQ INT4 ops register for `device="xpu"` only.
6. **Extension contract.** Torch bindings use `m.def` plus `m.impl`. SYCL
   sources that touch `Torch::tensor` include `<torch/all.h>`, never
   `<torch/extension.h>`.
7. **Evidence contract.** The completion claim cites logs from the `hal`
   Pod or Job. Build success on the development host is not evidence.
8. **Artifact contract.** `.gitignore` blocks `.cursor/`, agent transcripts
   and state, `*.cursor-*`, and other assistant scratch. Those paths MUST
   NOT be committed. `specs/` and `.specify/` stay tracked.
9. **Comment contract.** Comments explain why complex logic exists. They
   MUST NOT narrate ordinary code or announce machine authorship.
10. **Authorship contract.** Commit messages and pull requests use ordinary
    human conventions. AI-generation prefixes and prompt meta-commentary
    are forbidden. Git config MUST NOT be changed and history MUST NOT be
    rewritten to satisfy this gate.

Gates 8–10 apply to every change, including changes that do not touch
kernels or dispatch.

Tests that can run without the B70 (schema checks, manifest lint, CPU
reference numerics) MAY land in `tests/` and MUST be labeled as non-device
tests. Benchmarks that claim B70 bandwidth MUST run on `hal` and MUST live
under `benchmark/` using the Triton benchmark harness when the kernel is
Triton.

## Governance

This constitution is the compliance authority for `sgl-kernel-xpu`. Where it
conflicts with an informal habit, the constitution wins. `README.md` remains
the build and contribution guide. A conflict between this file and
`README.md` MUST be resolved in the same change that introduces it.

Amendments MUST edit `.specify/memory/constitution.md`, propagate dependent
templates, and bump **Version** by semantic version:

- MAJOR: a principle is removed or redefined so existing plans would no
  longer comply.
- MINOR: a principle or section is added, or guidance is materially expanded.
- PATCH: wording, clarification, or a non-semantic fix.

**Ratified** stays 2026-10-16. **Last Amended** is the date of the amendment.
Reviewers MUST reject kernel or dispatch changes that skip the `hal` test
loop. `/speckit-analyze` treats a breach of a MUST principle as CRITICAL.
A second implementation strategy (for example both a Triton kernel and a
SYCL kernel, or an unfused reference beside the fused path) MUST be justified
in the plan Complexity Tracking table.

Runtime development guidance: `README.md`.

**Version**: 1.1.0 | **Ratified**: 2026-10-16 | **Last Amended**: 2026-10-17
