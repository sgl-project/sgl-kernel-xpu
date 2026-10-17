# Contract: `scripts/test-loop.sh`

One command runs build, push, replace, apply, and log follow. The script
uses `set -euo pipefail`. The first failed stage is the script's non-zero
exit. Later stages do not run.

## Environment

| Name | Required | Meaning |
| --- | --- | --- |
| `REGISTRY` | yes | Registry the k3s nodes can pull. No default host |
| `IMAGE` | no | Default `$REGISTRY/sgl-kernel-xpu-gptq` |
| `TAG` | no | Default `dev` |
| `MODE` | no | `smoke` (default), `numeric`, `generate`, `bench` |
| `BASELINE_TPS` | bench only | First correct tokens/sec. Tuned run must exceed it |
| `KUBE_CONTEXT` | no | Unset. kubectl uses the default context |

## Stages

1. `docker build -f Dockerfile.xpu -t "$IMAGE:$TAG" .`
2. `docker push "$IMAGE:$TAG"`
3. `kubectl delete job sgl-gptq-xpu-test --ignore-not-found`
4. `kubectl apply -f deploy/k8s/test-job.yaml` with `MODE` and the image
   passed into the Job
5. `kubectl logs -f job/sgl-gptq-xpu-test`

The script then exits 0 only if the Job succeeded.

## Mode outcomes

| Mode | Success in the streamed log |
| --- | --- |
| `smoke` | Line `SGL_KERNEL_XPU_GPTQ_ENTRY`. No model load |
| `numeric` | Probe summary for FP16 and BF16, `M=1` and `M>1`, zero failures |
| `generate` | `SGL_GPTQ_GENERATION_RECORD` with 32 tokens and a tokens/sec field |
| `bench` | Same record with `stage` `baseline` or `tuned`. Non-zero exit if `BASELINE_TPS` is set and the new rate is not higher |

## Failure behavior

- Unset `REGISTRY`: exit before `docker build`, message names `REGISTRY`.
- Build or push failure: no `kubectl apply`.
- Re-invoke while a Job exists: the delete stage removes it, then one new
  Job is applied.
- GPU unschedulable or wrong node: the Job does not run on a node other
  than `hal`. The stream or kubectl events show the schedule failure, and
  the script exits non-zero.
