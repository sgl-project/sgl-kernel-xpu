# Contract: `sgl_kernel.gptq_gemm`

Torch library `sgl_kernel`. Dispatch key `XPU` only. This package does not
register a CUDA implementation.

## Schema

```text
gptq_gemm(
  Tensor a,
  Tensor b_q_weight,
  Tensor b_gptq_qzeros,
  Tensor b_gptq_scales,
  Tensor b_g_idx,
  bool use_shuffle,
  int bit,
  int zero_format=1
) -> Tensor
```

Python: `sgl_kernel.gptq_gemm(...)` forwards to
`torch.ops.sgl_kernel.gptq_gemm`.

## Arguments

| Name | Shape and type | Rule |
| --- | --- | --- |
| `a` | `[M, K]`, FP16 or BF16, XPU | Activations |
| `b_q_weight` | `[K/8, N]`, int32, XPU | GPTQ row-packed nibbles |
| `b_gptq_qzeros` | `[K/128, N/8]`, int32, XPU | GPTQ v1 stored zeros |
| `b_gptq_scales` | `[K/128, N]`, same dtype as `a`, XPU | Per-group scales |
| `b_g_idx` | `[K]`, int32, XPU | Scale-row index per K. Contiguous `i -> i/128` matches contiguous groups |
| `use_shuffle` | bool | `false` for this feature. `true` raises |
| `bit` | int | Must be 4 |
| `zero_format` | int, default 1 | `1` = missing or GPTQ v1 (`stored+1`). `2` = declared v2 (stored as-is). Any other value raises |

## Result

`[M, N]`, same dtype and device as `a`.

`M == 1` uses the GEMV specialization. `M > 1` uses the GEMM
specialization. Both read packed weights only. Neither writes an unpacked
weight tensor.

## Errors

The call raises, returns no tensor, and the message states which check
failed when:

- any math tensor is not XPU, not contiguous, or has the wrong rank
- `bit != 4`, `use_shuffle` is true, or `zero_format` is not 1 or 2
- `K` is not divisible by 128, or `N` is not divisible by 8
- `b_g_idx` length is not `K`
- scale or zero shapes do not match `K/128` and `N`
- dtype is not FP16 or BF16

## Probe

```text
gptq_entry_probe() -> ()
```

XPU only. Logs exactly one line containing `SGL_KERNEL_XPU_GPTQ_ENTRY`.
Does not run the GEMM. Smoke mode is the only caller.

## Reference

For a packed fixture, every output element is within `max(0.01 * |ref|, 0.02)`
of the FP32 unpack-and-multiply:

```text
effective_zero = stored_nibble + 1    # zero_format 1 (missing or v1)
effective_zero = stored_nibble        # zero_format 2 (declared v2)
ref = (q - effective_zero) * scale
```

The scale and zero row is `b_g_idx[k]`, not `k / 128`, unless that index is the contiguous map. Acceptance generation is greedy (temperature 0) and requests 32 tokens.

The reference is a test oracle. It is not a runtime fallback.
