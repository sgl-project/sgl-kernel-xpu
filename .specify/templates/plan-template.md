# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]

**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

[Extract from feature spec: primary requirement + technical approach from research]

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: [e.g., C++/SYCL + Python, PyTorch >= 2.14.0+xpu or NEEDS CLARIFICATION]

**Primary Dependencies**: [e.g., SGLang >= 0.5.22, Intel oneAPI, Triton XPU or NEEDS CLARIFICATION]

**Storage**: [e.g., NAS host path `/tank/nas/models/` or N/A]

**Testing**: [e.g., kubectl Job on node `hal`; pytest in `tests/` for non-device checks or NEEDS CLARIFICATION]

**Target Platform**: [e.g., Intel Arc Pro B70 32 GB on k3s node `hal` or NEEDS CLARIFICATION]

**Project Type**: [e.g., XPU kernel library + SGLang custom op or NEEDS CLARIFICATION]

**Performance Goals**: [e.g., fused INT4 dequant-GEMM/GEMV bandwidth on the B70 or NEEDS CLARIFICATION]

**Constraints**: [e.g., no local Intel GPU; build → push → kubectl apply → logs; `nodeName: hal` or NEEDS CLARIFICATION]

**Scale/Scope**: [e.g., Qwen3.8-27B INT4 GPTQ g128 acceptance model or NEEDS CLARIFICATION]

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Gates from `.specify/memory/constitution.md`. Mark each PASS, FAIL, or N/A.
A FAIL is allowed only with a Complexity Tracking row.

- **I. Hardware-Truth Testing**: Validation of XPU behavior is a Pod or Job
  on node `hal` (Intel Arc Pro B70). Local compile or CPU/CUDA runs are not
  the acceptance evidence.
- **II. Bandwidth-Efficient INT4 Kernels**: INT4 GPTQ dequant and GEMV/GEMM
  are Triton-XPU or SYCL/C++ extensions, fused when fusion is feasible.
  N/A only when the feature does not add or change a math kernel.
- **III. Device-Scoped SGLang Routing**: GPTQ INT4 dispatch selects the XPU
  kernel only when `device="xpu"`. Other devices keep their current path.
- **IV. Automated Remote Test Loop**: One script runs local image build,
  push to the cluster registry, `kubectl apply` (default context), and
  `kubectl logs -f`. The image bases on Intel oneAPI and/or PyTorch XPU.
- **V. Pinned Model and Node Contract**: Inference manifests set
  `nodeName: hal` (or an equivalent selector) and mount
  `/tank/nas/models/`, loading
  `Qwen3.8-27B-TURBO-DF735-882-NM-DAW-GPTQ-Int4-g128-tuned`.
- **VI. Untracked Assistant Artifacts**: `.cursor/`, agent transcripts and
  state, `*.cursor-*`, and other assistant scratch stay gitignored and
  uncommitted. `specs/` and `.specify/` stay tracked.
- **VII. Sparse Practical Comments**: Comments explain why complex logic
  exists. They do not narrate ordinary code or announce machine authorship.
- **VIII. Human Authorship in Version Control**: Commits and pull requests
  use ordinary human messages. No AI-generation prefixes, prompt
  meta-commentary, git-config edits, or history rewrites.
- **Stack floor**: PyTorch >= 2.14.0+xpu and SGLang >= 0.5.22, unless a
  deviation is justified.
- **Extension contract**: `m.def` + `m.impl`; `<torch/all.h>` for
  `Torch::tensor` in SYCL sources.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
# [REMOVE IF UNUSED] Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# [REMOVE IF UNUSED] Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# [REMOVE IF UNUSED] Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure: feature modules, UI flows, platform tests]
```

**Structure Decision**: [Document the selected structure and reference the real
directories captured above]

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
