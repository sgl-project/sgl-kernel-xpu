---
name: stride-overflow-safety
description: Keep tensor strides and global offsets in sgl-kernel-xpu kernels and host launchers 64-bit so they never overflow int32 (paged KV block id * stride, batch/head/token/slot offsets, varlen cu_seqlens * stride, params struct stride fields), plus a diff scanner. Use when writing, modifying or reviewing a kernel, launcher or params struct that reads t.stride()/t.size()/numel(), computes pointer offsets, indexes a paged KV cache through a block table, or when a large-shape or long-running serving workload hangs the GPU, reads out of bounds, or returns wrong results only for high block ids.
---

# Stride / offset overflow safety

A global offset is `index * stride` in units of the pointer's element type.
If any operand or any storage hop on the way is 32-bit, the product wraps at
2^31 - 1 elements. Small-shape tests never catch it; it appears in serving
when high-numbered KV cache blocks get used.

**Typical case**: a paged FP4 MQA-logits launcher stored `kv_stride0` and
`scale_stride0` as `int32_t` in launcher locals and in the params struct.
With a hybrid KV page stride of 191488 B, `physical_block * kv_stride0`
wrapped once block ids passed ~11214 (2^31 / 191488), giving out-of-bounds
reads and a GPU hang under prefix caching. The FP8 sibling path already used
`int64_t`; the fix widened every stride in the launcher and params struct.

These rules apply to every new or changed line. Existing code predates them;
do not copy narrow-stride patterns into new code.

## Rules

| ID | Rule |
|---|---|
| SO-1 | Carry strides, sizes and `numel()` from torch as `int64_t` end to end: launcher locals, params struct fields, kernel arguments. `t.stride(i)` is already int64; never store it in `int` / `int32_t` / `uint32_t`. |
| SO-2 | Widen **before** the multiply: `static_cast<int64_t>(idx) * stride`, or make `idx` / `stride` `int64_t`. `int64_t off = a * b;` with two 32-bit operands still wraps. |
| SO-3 | Treat outer indices as unbounded: physical block ids from block tables, slot mappings, `cu_seqlens` values, token / batch / expert offsets, head index of large pools. Every offset built from them is int64. |
| SO-4 | Narrow to int32 only with a proven bound: inner strides and indices bounded by block size, head dim, head count or `max_model_len` may be int32 in hot loops, with an int64 `TORCH_CHECK` at the entry (`stride * extent <= INT32_MAX`). |
| SO-5 | Fields that must stay narrow (2D block load surface width / height / pitch; 32-bit CuTe strides) get an int64 host guard, or the surface is rebased. |
| SO-6 | A stride fix ships with a test whose offset crosses 2^31 elements of the pointer type. |

The limit counts elements of the pointer type: `uint8_t*` and packed FP4/FP8
pointers hit it at 2 GiB. Largest safe outer index:
`(2^31 - 1) / stride0`.

## When writing or modifying code

1. Declare every stride / size taken from a tensor as `int64_t`, and give the
   params struct `int64_t` fields for them (SO-1).
2. For each pointer offset, check that the first multiply already has an
   int64 operand (SO-2, SO-3).

   ```cpp
   // launcher
   const int64_t kv_stride0 = kv.stride(0);           // not int32_t
   // params struct
   int64_t kv_stride0;                                 // not int32_t
   // kernel
   const int32_t physical_block = block_table(b, i);   // ids may stay int32 ...
   const auto* kv_base =
       p.kv_ptr + static_cast<int64_t>(physical_block) * p.kv_stride0;  // ... product is int64
   ```

3. If you downcast for performance, write the bound next to it and guard it
   at the entry (SO-4):

   ```cpp
   TORCH_CHECK(static_cast<int64_t>(block_size) * kv.stride(1) <= INT32_MAX,
               "foo: per-block extent exceeds int32");
   ```

4. Compare with sibling variants of the same op (FP8 vs FP4, paged vs dense,
   decode vs prefill, Xe2 vs Xe3 path). Make all variants use the same
   stride width.
5. Add a test (SO-6), then run the scanner:
   `python3 .claude/skills/stride-overflow-safety/scan_stride_overflow.py --diff`.

## Tests that cross 2^31

- **Unused wide stride** (cheap): give a dimension that is never indexed
  past 0 a stride of `2**31 + 128` with `torch.as_strided`, and compare
  against the normal tensor. This catches launchers that truncate strides.
- **Real crossing**: build a KV pool or input where the highest used
  block / batch / head starts beyond 2^31 elements, and point a block table
  entry at that block. This needs > 2 GiB for byte-typed pointers. Keep it
  out of the `per-commit` suite in `tests/run_suite.py` (nightly, or an
  env-var `skipif`).

## When reviewing

1. Run `python3 .claude/skills/stride-overflow-safety/scan_stride_overflow.py --diff`
   (default base `origin/main`; it checks only added lines). Hits are
   candidates. `--all` audits all of `src/` and `include/`; existing code has
   many hits, so do not use it as a PR gate.
2. Trace each global pointer offset along the whole chain:
   `t.stride()` → launcher local → params field → kernel expression. Any
   32-bit hop is a finding, even if the final expression looks 64-bit.
3. For each offset, ask what bounds the index. If nothing in the op bounds it
   (block table, slot mapping, cu_seqlens, batch), it must be int64. If a
   downcast relies on a bound, check that the bound is written down and
   guarded.
4. Check sibling variants of the op for the same stride width.
5. Check that a test crosses 2^31, or ask for one.
6. Report as `[SO-n] path:line — what can overflow and when — fix`, e.g.
   `[SO-1] src/sycl/foo.cpp:120 — kv_stride0 stored as int32_t; block * stride wraps past block 11214 at 191488 B pages — make it int64_t in the launcher and in Params`.
