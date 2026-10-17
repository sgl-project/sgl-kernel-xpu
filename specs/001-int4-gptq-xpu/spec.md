# Feature Specification: INT4 GPTQ on Intel GPU

**Feature Branch**: `001-int4-gptq-xpu`

**Created**: 2026-10-16

**Status**: Draft

**Input**: User description: "Implement INT4 GPTQ inference for the named Qwen checkpoint on the Intel Arc Pro B70, route that math only when the workload is on the Intel GPU, and provide one operator command that builds, publishes, runs, and streams a test on node `hal`."

## Clarifications

### Session 2026-10-16

- Q: When the acceptance checkpoint supplies a per-channel group index (activation order), what must the multiply do? → A: Honor the per-channel group index during the multiply. Contiguous groups of 128 stay valid.
- Q: How should the stored zero-point be applied? → A: Follow the checkpoint. A missing or v1 format uses stored+1. A declared v2 format uses the stored zero as-is.
- Q: How must the 32-token continuation be decoded? → A: Greedy only (temperature 0). Two runs with the same prompt and settings must return the same 32 tokens.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prove the math path on the GPU node (Priority: P1)

An operator on the development machine, which has no Intel GPU, runs one command. The command publishes a fresh test image, starts a single run on the machine that has the Intel GPU, and streams that run's output until it ends. The log shows that the new weight-math entry point was invoked on that GPU.

**Why this priority**: Later numeric and generation work is unknowable until a run actually executes on the GPU node. This slice is the smallest demonstration that the remote loop and the new entry point are alive.

**Independent Test**: Run the single operator command with the stand-in entry point. The streamed log contains the entry-point marker, and the run is recorded on the designated GPU node.

**Acceptance Scenarios**:

1. **Given** a development machine with no Intel GPU and a reachable GPU node, **When** the operator runs the single test command, **Then** the command builds an image, publishes it, starts one run, and streams the log until the run finishes or fails, with no extra commands.
2. **Given** the stand-in entry point is included in that image, **When** the run invokes it on the Intel GPU, **Then** the streamed log contains a distinguishable marker that the entry point ran.
3. **Given** a previous test run is still present, **When** the operator runs the command again, **Then** the previous run is removed and exactly one new run is streamed.
4. **Given** image build or publish fails, **When** the operator runs the command, **Then** the command stops at that stage, reports the failure, and does not start a GPU-node run.

---

### User Story 2 - Match grouped INT4 weight math (Priority: P2)

A reviewer checks a known weight tile and a known activation tile on the GPU node. Packed 4-bit weights, one scale per group of 128, and one zero per group produce a matrix product that agrees with a high-precision unpack-and-multiply reference. A missing or v1 format uses stored zero plus one. A declared v2 format uses the stored zero as-is. Each input channel uses the scale and zero named by the checkpoint's per-channel group index. A contiguous index, channel i mapping to group i/128, matches contiguous groups. The same check covers a single row (token-by-token generation) and many rows (prompt processing), for both supported 16-bit activation formats.

**Why this priority**: Generation quality depends on this numeric contract. The check does not need the full checkpoint, so it can pass or fail on its own.

**Independent Test**: Submit one fixed weight tile and activation tile on the GPU node for each activation format and for both single-row and many-row shapes. Compare every reported output element to the reference unpack-and-multiply result.

**Acceptance Scenarios**:

1. **Given** packed INT4 weights with group size 128 and matching per-group scales and zeros, **When** a single-row activation in the standard half format is multiplied, **Then** every compared output element is within the agreed tolerance of the reference.
2. **Given** the same weight tile, **When** a many-row activation in the brain-float format is multiplied, **Then** every compared output element meets the same tolerance.
3. **Given** scales or zeros whose length does not match the weight groups, **When** the math path is called, **Then** the call fails with a visible error and does not return a numeric result.
4. **Given** an activation format outside the two supported 16-bit formats, **When** the math path is called, **Then** the call fails with a visible error.
5. **Given** a per-channel group index that is not the contiguous map, **When** the tile is multiplied, **Then** each channel uses the scale and zero named by that index and every compared element meets the tolerance.
6. **Given** the same packed tile stored once as v1 and once as declared v2, **When** each is multiplied, **Then** the v1 product uses stored+1 and the v2 product uses the stored zero as-is, and both meet the tolerance against their own reference.

---

### User Story 3 - Generate text from the acceptance model (Priority: P3)

An operator loads the named INT4 checkpoint from the shared model store and asks for a short continuation. The run executes only on the designated GPU node, uses one Intel GPU, and returns text. Workloads that are not on the Intel GPU keep the math path they already had.

**Why this priority**: Text from this checkpoint is the user-visible goal. It depends on the remote loop and on numeric agreement, and it is still demonstrable as its own generation run.

**Independent Test**: Start one greedy generation of a fixed non-empty prompt for 32 new tokens. Confirm the text comes back, the run used the GPU node and the shared model store, and a non-Intel-GPU request does not select the new path.

**Acceptance Scenarios**:

1. **Given** the acceptance model is readable from the shared model store, **When** the operator requests a greedy 32-token continuation of a fixed prompt, **Then** the run returns the generated text and reports tokens per second.
2. **Given** the same prompt and greedy settings (temperature 0), **When** generation succeeds twice in a row, **Then** both runs return the same 32 tokens.
3. **Given** a workload that is not on the Intel GPU, **When** INT4 grouped weight math is requested, **Then** the existing path for that device is the one that runs.
4. **Given** the shared model store is missing or unreadable, **When** generation starts, **Then** the run fails, the streamed log names the failure, and the run is not reported as a success.

---

### User Story 4 - Record and raise generation speed (Priority: P4)

After a correct generation exists, the operator runs the same prompt again once memory access has been tuned. Both rates are kept, and the later rate is higher.

**Why this priority**: Speed matters only after the text is correct. This slice is a measured improvement on a fixed prompt, not a new product behavior.

**Independent Test**: Record tokens per second for the first correct 32-token generation, tune memory access, repeat the identical prompt and settings on the same GPU node, and compare the two recorded rates.

**Acceptance Scenarios**:

1. **Given** a first correct generation of the fixed prompt, **When** the run finishes, **Then** its tokens-per-second rate is stored with the prompt and settings.
2. **Given** that stored baseline, **When** the tuned run repeats the same prompt and settings on the same GPU node, **Then** the tuned tokens-per-second rate is stored and is higher than the baseline.

---

### Edge Cases

- A previous test run is still present. The operator command removes it before starting the new run, and the stream shows one new run.
- Image build or publish fails. The command stops and does not start a GPU-node run.
- The designated node cannot grant one Intel GPU. The run is not placed on the other node, and the failure is visible in the command output.
- The shared model directory is missing or unreadable. Generation fails with that location named in the log.
- Scales or zeros are missing or do not match the number of groups of 128. The math call fails with a visible error and returns no numeric result.
- A per-channel group index is present. Each input channel uses the scale and zero named by that index. A contiguous index (channel i maps to group i/128) matches the contiguous-group product. An index whose length is not the channel count fails with a visible error and returns no numeric result.
- The checkpoint declares no format or GPTQ v1. The multiply uses stored zero plus one. A declared v2 checkpoint uses the stored zero as-is. Any other format fails with a visible error and returns no numeric result.
- The activation format is outside the two supported 16-bit formats. The math call fails with a visible error.
- The prompt is empty. Generation is rejected with a visible error before any tokens are produced.
- A generation request uses sampling or a temperature other than 0. The run is rejected with a visible error and is not a successful generation. Acceptance runs use greedy decoding only.
- The prompt is a single token. The single-row multiply is used, and a successful call still returns continuation text.
- The prompt is long enough to require the many-row multiply. Prompt processing finishes, then token-by-token generation continues.
- A caller requests grouped INT4 math on a device other than the Intel GPU. The existing path for that device remains the one that runs.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: An operator MUST be able to launch the remote test with one command and no further manual steps.
- **FR-002**: That command MUST build the test image, publish it to the cluster-local registry, clear any previous test run, start one new run, and stream that run's output until the run finishes or fails.
- **FR-003**: The command MUST stop at the first failed stage and MUST show that failure in its output.
- **FR-004**: A successful test run MUST execute only on the designated Intel GPU node.
- **FR-005**: The test run MUST request one Intel GPU on that node.
- **FR-006**: Generation MUST read the acceptance model from the shared model store. Model weights MUST stay out of the test image.
- **FR-007**: A stand-in entry point MUST load on the Intel GPU and MUST write one distinguishable marker to the run log when it is called.
- **FR-008**: The math path MUST accept packed INT4 weights with group size 128, plus one scale and one zero for each group.
- **FR-021**: The math path MUST select each channel's scale and zero using the checkpoint's per-channel group index when that index is supplied. A contiguous index (channel i maps to group i/128) MUST match the contiguous-group product. An index whose length is not the channel count MUST fail visibly and MUST return no numeric result.
- **FR-022**: The math path MUST apply a missing or v1 zero as stored+1, and a declared v2 zero as the stored value. Any other declared format MUST fail visibly and MUST return no numeric result.
- **FR-009**: The math path MUST accept activations in the standard 16-bit half format and in the 16-bit brain-float format.
- **FR-010**: The math path MUST provide a single-row product for token-by-token generation and a many-row product for prompt processing.
- **FR-011**: One math call MUST produce the product from packed weights and per-group scale and zero. The steady-state path MUST NOT require a stored full-precision copy of the weights.
- **FR-012**: Every compared probe output MUST match the high-precision unpack-and-multiply reference within the tolerance in SC-003.
- **FR-013**: Intel GPU workloads that request INT4 grouped weight math MUST use this new path.
- **FR-014**: Workloads on any other device MUST keep the math path they already use.
- **FR-015**: A generation request MUST load the acceptance model and MUST return continuation text for a non-empty prompt.
- **FR-016**: A successful generation MUST report its speed in tokens per second.
- **FR-017**: The first correct generation's tokens-per-second rate MUST be retained with the prompt and settings. The tuned repeat of that same generation MUST be retained and MUST be faster.
- **FR-018**: Acceptance generation MUST use greedy decoding (temperature 0). Two consecutive successful generations with the same prompt and those settings MUST return the same 32 tokens. A sampled or non-zero-temperature request MUST be rejected and MUST NOT count as a successful generation.
- **FR-019**: An empty prompt MUST be rejected with a visible error before tokens are produced.
- **FR-020**: A failed or rejected run MUST NOT be reported as a successful generation.

### Key Entities

- **Acceptance model**: The named INT4 grouped-weight checkpoint used for generation. Attributes: storage location on the shared model store, group size 128, packed 4-bit weights, per-group scale, per-group zero, zero format (missing or v1, or declared v2), and a per-channel group index when activation order is set. It is read at run time and is not copied into the test image.
- **Weight group**: 128 packed 4-bit weight values with one scale and one stored zero. A missing or v1 format uses that zero plus one. A declared v2 format uses the stored zero as-is. A per-channel group index names which group each input channel uses. The math path consumes groups; it does not persist an unpacked copy as its steady-state input.
- **Activation tile**: The input matrix for a probe or a generation step. Format is either standard 16-bit half or 16-bit brain-float. Shape is one row for token-by-token generation or many rows for prompt processing.
- **Remote test run**: One published image executed on the designated GPU node, requesting one Intel GPU, with a streamed log. A later command replaces any previous run.
- **Generation record**: The prompt, the settings (including greedy decoding at temperature 0 and 32 new tokens), the returned text, and the tokens-per-second rate. The baseline record and the tuned record are both kept so the speed comparison is auditable.

## Project Constraints *(mandatory)*

- **Validation**: Intel GPU behavior is proven only by a run pinned to node `hal` (Intel Arc Pro B70, 32 GB). The operator command streams that run's log. A local build, a CPU run, or a CUDA run is not acceptance evidence.
- **Kernel contract**: INT4 GPTQ, group size 128, with per-group scales and zeros. A missing or v1 zero is applied as stored+1. A declared v2 zero is applied as stored. A per-channel group index selects the scale and zero for each input channel. Contiguous indexes stay valid. Dequantization is fused into the single-row product (GEMV, decode) and the many-row product (GEMM, prefill) whenever fusion is feasible. Weights stay packed. Activations are FP16 or BF16. The steady-state path does not materialize a full unpacked weight tensor.
- **Dispatch**: GPTQ INT4 math selects the new kernel only when `device="xpu"`. CUDA and every other device keep their current dispatch. The check reads the workload device.
- **Stack**: PyTorch >= 2.14.0+xpu and SGLang >= 0.5.22. The test image is based on the official Intel PyTorch XPU image and installs that SGLang floor. No version deviation.
- **Model path**: Host path `/tank/nas/models/` is mounted into the run. Weights load from `/tank/nas/models/Qwen3.8-27B-TURBO-DF735-882-NM-DAW-GPTQ-Int4-g128-tuned`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An operator starts a remote test with one command and sees the run finish or fail in the streamed log without issuing any further command. Re-running that command while an older run exists yields exactly one new streamed run.
- **SC-002**: On a successful path check, the streamed log from the GPU node contains the entry-point marker.
- **SC-003**: For each supported 16-bit activation format, and for both a single-row and a many-row probe, 100% of compared output elements are within 1% relative error or 0.02 absolute error of the high-precision unpack-and-multiply reference, whichever bound is larger.
- **SC-004**: The acceptance model returns 32 new tokens for a fixed non-empty prompt under greedy decoding (temperature 0), and a second run of that prompt and those settings returns the same 32 tokens.
- **SC-005**: Each of those generation runs reports tokens per second. The rate after memory-access tuning is higher than the first correct rate for the same prompt, settings, and GPU node, and both rates are retained.
- **SC-006**: 100% of runs reported as successful generation execute on the designated GPU node and read the acceptance model from the shared store. A request that is not on the Intel GPU keeps its previous math path.
- **SC-007**: When the model store is unavailable, when the prompt is empty, or when scales and zeros do not match the groups, the outcome is a visible failure in 100% of those trials, and none of them are counted as successful generations.

## Assumptions

- The development machine has no Intel GPU. GPU proof comes from the designated node on the existing 2-node cluster, using the operator's default cluster context.
- The cluster-local registry is already reachable from that node. Its address is an operator setting supplied to the test command.
- The cluster's Intel GPU resource is `gpu.intel.com/i915`. Each test run requests one such device.
- The acceptance checkpoint is already present on the shared store, is INT4 GPTQ, and uses group size 128. This feature does not download or re-quantize it.
- Numeric agreement is judged against an on-node high-precision unpack-and-multiply reference. A second vendor's GPU is not required.
- The demonstration prompt is fixed, non-empty, and requests 32 new tokens under greedy decoding (temperature 0). "First correct generation" means a run that already meets SC-003 and SC-004.
- The entry-point marker is one fixed phrase documented with the test command so a log search can find it.
- This specification does not choose a compiler, a binding mechanism, or an instruction set. Those choices belong to the implementation plan, provided the outcomes above hold.
- Other quantization formats, other group sizes, training, multi-GPU runs, and nodes other than the designated GPU node are out of scope.
