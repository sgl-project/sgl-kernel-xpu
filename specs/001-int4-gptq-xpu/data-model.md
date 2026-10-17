# Data Model: INT4 GPTQ on Intel GPU

Entities from `spec.md`. Fields are the values the ops, the Job, and the
generation record actually carry. No new stored database.

## Acceptance model

| Field | Value |
| --- | --- |
| id | `Qwen3.8-27B-TURBO-DF735-882-NM-DAW-GPTQ-Int4-g128-tuned` |
| root | `/tank/nas/models/<id>` |
| host mount | `/tank/nas/models/` mounted at the same path in the Job |
| format | INT4 GPTQ, group size 128 |
| packaged in image | no |

**Validation:** `generate` and `bench` fail if `root` is missing or
unreadable, name that path in the log, and exit non-zero. `smoke` and
`numeric` do not open the checkpoint.

**Relationships:** A generation record names this id. A remote test run in
`generate` or `bench` mode mounts the store that holds it.

## Weight group

| Field | Value |
| --- | --- |
| group_size | 128 |
| bit | 4 |
| qweight | int32 `[K/8, N]`, eight K nibbles per word |
| qzeros | int32 `[K/128, N/8]`, GPTQ v1 stored zeros |
| effective zero | `stored_nibble + 1` |
| scales | FP16 or BF16 `[K/128, N]` |
| g_idx | int32 `[K]`, scale-row selector. Contiguous `i -> i/128` matches contiguous groups |
| zero format | `1` missing or v1 (`stored+1`), `2` declared v2 (stored as-is) |
| steady-state unpacked copy | none |

**Validation:** `K` divisible by 128 and by 8. `N` divisible by 8. Scale and
zero leading dimension equals `K/128`. `bit` other than 4, any other group
size, a mismatched zero tensor, or a dtype other than FP16 or BF16 is a
visible error and returns no product.

**Relationship:** Many groups form one weight matrix consumed by one
`gptq_gemm` call together with an activation tile.

## Activation tile

| Field | Value |
| --- | --- |
| tensor | `a`, shape `[M, K]` |
| dtype | FP16 or BF16, same as scales |
| device | `xpu` for a successful math call |
| single-row | `M == 1`, token-by-token generation |
| many-row | `M > 1`, prompt processing |
| product | `[M, N]`, same dtype and device as `a` |

**Validation:** Empty prompt never builds a tile. A non-`xpu` device does
not enter this op through SGLang. A direct call with a non-`xpu` tensor
fails the XPU dispatch key rather than running a CUDA body from this
package.

## Remote test run

| Field | Value |
| --- | --- |
| name | Job `sgl-gptq-xpu-test` |
| node | `nodeName: hal` |
| gpu | `gpu.intel.com/i915: 1` |
| image | `$REGISTRY/sgl-kernel-xpu-gptq:$TAG` |
| mode | `smoke`, `numeric`, `generate`, or `bench` |
| context | kubectl default context |

**States:**

1. `absent` — no Job of that name.
2. `building` — `docker build` in progress. Failure stops here.
3. `publishing` — `docker push` in progress. Failure stops here.
4. `starting` — previous Job deleted, new Job applied.
5. `running` — log stream open.
6. `succeeded` or `failed` — stream ends. The script's exit code matches
   the Job.

A new operator command deletes a Job in any state before `starting`, so two
commands never leave two Jobs.

## Generation record

One JSON object printed as a single log line prefixed
`SGL_GPTQ_GENERATION_RECORD`.

| Field | Value |
| --- | --- |
| stage | `baseline` or `tuned` |
| model | acceptance model id |
| prompt | `Write a one-sentence hardware note.` |
| max_new_tokens | 32 |
| temperature | 0 (greedy). Any other temperature is rejected |
| text | decoded continuation |
| tokens_per_second | measured rate |
| node | `hal` |

**Validation:** Two successful runs with the same prompt and settings must
report the same `text`. `bench` with `BASELINE_TPS` set fails unless
`tokens_per_second` is greater than that value. An empty prompt produces no
record and a non-zero exit.

**Relationships:** Each record belongs to one succeeded `generate` or
`bench` run and to the acceptance model.
