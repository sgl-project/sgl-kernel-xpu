# Sparse MLA Prefill (mla_sparse_xattn) — Build / UT / Benchmark Results

- Date: 2026-07-10 15:53
- Repo commit: ed12014 (sgl-kernel-xpu)
- GPU: Intel(R) Arc(TM) B580 Graphics (BMG / Xe2)
- Python 3.10.16, pytest 9.0.0, torch (XPU)
- Kernel source: `src/sycl/kernels/mla_sparse_xattn/` (flattened, MHA/decode code pruned)

## Build

- `cmake build && ninja -C build` — SUCCESS (exit 0), no errors/warnings.
- Installed libs to `$CONDA_PREFIX/lib/python3.10/site-packages/sgl_kernel/`:
  - `common_ops.abi3.so`
  - `libsgl-ops-sycl-sparse_mla_prefill_fwd_k512.so`
  - `libsgl-ops-sycl-sparse_mla_prefill_fwd_k576.so`
  - `libsgl-ops-sycl-sparse_mla_prefill_fwd_topklen_k512.so`
  - `libsgl-ops-sycl-sparse_mla_prefill_fwd_topklen_k576.so`

## Unit Tests

Command: `python -m pytest tests/test_flash_mla_sparse_prefill.py -v`

Result: **96 passed in 12.23s** (0 failed).

Parameter grid covered: h_q ∈ {6, 22, 64, 96}, s_q ∈ {1, 32, 128}, d_qk = 512,
topk ∈ {6, 512}, has_attn_sink ∈ {False, True}, has_topk_length ∈ {False, True}.
Tolerances: max_logits/lse atol=rtol=1e-3, out atol=rtol=1e-2.

## Benchmark

Command: `python benchmark/bench_flash_mla_sparse_prefill.py`

DeepSeek-V4 configs (D_QK=576, D_V=512, bf16). `ok` = allclose(atol=rtol=1e-2) vs reference.
xattention comparison unavailable in this env (`flash_attn_2_xpu` has no `sparse_prefill_fwd`), so xatt columns are nan.

| config                      | MB/call | sgl ms | sgl GB/s | max_abs  | ok   |
|-----------------------------|---------|--------|----------|----------|------|
| TP=8  (16 heads) topk=2048  | 1225.8  | 17.084 | 71.75    | 9.77e-04 | True |
| TP=4  (32 heads) topk=2048  | 1243.6  | 22.233 | 55.94    | 1.95e-03 | True |
| TP=1 (128 heads) topk=2048  | 1350.6  | 31.951 | 42.27    | 1.95e-03 | True |
| TP=8  (16 heads) topk=512   | 1279.3  | 19.843 | 64.47    | 1.95e-03 | True |
| TP=1 (128 heads) topk=512   | 1778.4  | 28.637 | 62.10    | 3.91e-03 | True |

All configs numerically correct (ok=True); bandwidth 42–72 GB/s on B580.
