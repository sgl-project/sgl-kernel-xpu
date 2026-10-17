# Contract: `deploy/k8s/test-job.yaml`

Kind `Job`, name `sgl-gptq-xpu-test`, `backoffLimit: 0`,
`restartPolicy: Never`.

## Placement and device

- `nodeName: hal`
- Container resource request and limit: `gpu.intel.com/xe: 1` (the resource the Arc plugin on `hal` advertises)
- No affinity that can select the other k3s node

## Volumes

| Mount | Host | When |
| --- | --- | --- |
| `/tank/nas/models` | `/tank/nas/models` | Always mounted. `generate` and `bench` read it. `smoke` and `numeric` may ignore it |

The container image does not contain the checkpoint.

## Container command

The image entrypoint reads `MODE` and runs the matching script shipped in
the image:

| `MODE` | Command |
| --- | --- |
| `smoke` | call `gptq_entry_probe` on XPU |
| `numeric` | run the grouped INT4 probe |
| `generate` | load the acceptance model, 32 new tokens |
| `bench` | same generation, print the rate, compare `BASELINE_TPS` when set |

Model path inside the container:

```text
/tank/nas/models/Qwen3.8-27B-TURBO-DF735-882-NM-DAW-GPTQ-Int4-g128-tuned
```

An empty prompt is rejected by the generate script before the engine runs.
A missing model directory is a non-zero exit whose log includes that path.
