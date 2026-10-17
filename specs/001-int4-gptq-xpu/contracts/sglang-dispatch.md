# Contract: SGLang XPU GPTQ dispatch

## Where

Image build patches only:

```text
python/sglang/srt/hardware_backend/xpu/quantization/gptq_kernels.py
```

`GPTQXPULinearKernel.process_weights_after_loading` keeps GPTQ `qweight`,
`qzeros`, `scales`, and `g_idx` in their checkpoint packing. It does not
call `aten._convert_weight_to_int4pack` and does not store a full-precision
weight.

`GPTQXPULinearKernel.apply` calls `sgl_kernel.gptq_gemm` when
`x.device.type == "xpu"`. Any other device type raises before the kernel
runs.

## What stays unchanged

- `python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py`
- CUDA and HIP registration of `gptq_gemm` on `torch::kCUDA`
- This package's `m.impl` line, which binds `gptq_gemm` to `torch::kXPU`
  only

## Version gate

The image build installs SGLang from source and fails if the installed
version is below 0.5.22. The patch fails the image build if it does not
apply. There is no runtime fallback to `xpu_int4pack_mm`.

## Observable result

An XPU linear module whose quantization is INT4 GPTQ group 128 executes
`gptq_gemm` for both the prefill (`M > 1`) and decode (`M == 1`) calls.
A CUDA module never enters this XPU `apply`.
