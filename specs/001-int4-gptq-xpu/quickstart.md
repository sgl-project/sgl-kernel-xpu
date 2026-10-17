# Quickstart: INT4 GPTQ on node `hal`

Validation guide for `specs/001-int4-gptq-xpu`. Operator commands only.
Kernel source and the SGLang patch are implemented in later tasks.

## Prerequisites

- Docker on the development machine. That machine has no Intel GPU.
- `kubectl` aimed at the 2-node k3s cluster. Leave the context at its
  default. Do not pass `--context`.
- Node `hal` has the Arc Pro B70 and host path `/tank/nas/models/`.
- The acceptance checkpoint is already at
  `/tank/nas/models/Qwen3.8-27B-TURBO-DF735-882-NM-DAW-GPTQ-Int4-g128-tuned`.
- `REGISTRY` is the host:port the cluster can pull. Export it in the shell.
  The scripts do not embed a registry address.

Contracts: [test loop](contracts/test-loop.md), [Job](contracts/test-job.md),
[GEMM op](contracts/gptq-gemm.md), [dispatch](contracts/sglang-dispatch.md).
Records and tensors: [data model](data-model.md).

## Smoke (story 1)

```bash
export REGISTRY=<cluster-registry>
MODE=smoke ./scripts/test-loop.sh
```

Expected: the command builds `Dockerfile.xpu`, pushes
`$REGISTRY/sgl-kernel-xpu-gptq:dev`, replaces any Job named
`sgl-gptq-xpu-test`, and streams logs. A successful stream contains
`SGL_KERNEL_XPU_GPTQ_ENTRY` and the script exits 0. Run the same command
again. Exactly one new Job is streamed.

A forced build failure (unset `REGISTRY`, or a broken Dockerfile) exits
before `kubectl apply`.

## Numeric probe (story 2)

```bash
MODE=numeric ./scripts/test-loop.sh
```

Expected: the log reports FP16 and BF16 probes for `M=1` and for `M>1`,
group size 128, a non-contiguous group index, and both v1 (`stored+1`)
and declared v2 (stored as-is) zeros, with every compared element inside
the tolerance in [the GEMM contract](contracts/gptq-gemm.md). A mismatched
zero fixture and an unknown zero format fail visibly and are not counted
as a pass.

Local CPU checks of the unpack formula may exist under `tests/` and are
labeled non-device. They do not replace this Job.

## Generation (story 3)

```bash
MODE=generate ./scripts/test-loop.sh
```

Expected: one `SGL_GPTQ_GENERATION_RECORD` line for the prompt
`Write a one-sentence hardware note.`, greedy decoding at temperature 0,
`max_new_tokens` 32, non-empty `text`, and `tokens_per_second`. Run twice.
Both records have `temperature` 0 and the same 32 tokens. The Job stays
on `hal`.

If the model directory is not mounted, the log names that path and the
script exits non-zero. No success record is printed. A non-zero
`TEMPERATURE`, or `SAMPLING` set to anything other than greedy, exits
non-zero with no record. An empty `SGL_GPTQ_PROMPT` is rejected the same
way.

## Speed (story 4)

Keep the baseline rate from the first correct greedy generate record
(temperature 0), then rerun after the memory-access change:

```bash
MODE=bench ./scripts/test-loop.sh
MODE=bench STAGE=tuned BASELINE_TPS=<baseline rate> ./scripts/test-loop.sh
```

`STAGE` defaults to `baseline`. Expected: both records use temperature 0
and `max_new_tokens` 32. The tuned record's `tokens_per_second` is greater
than `BASELINE_TPS`. The script exits non-zero when it is not. Both
records remain in their job logs. A non-zero temperature is rejected with
no record, the same as generation.
