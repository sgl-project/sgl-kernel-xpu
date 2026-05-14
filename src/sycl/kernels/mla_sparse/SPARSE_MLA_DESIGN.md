# Sparse MLA Decode Kernel — Implementation Design

## Overview

This kernel implements DeepSeek V4's sparse decode attention on Intel Xe (BMG) GPUs.
It extends the standard MLA decode kernel with:
- **Two KV cache pools** (SWA + Extra compressed)
- **Token-level scattered gather** (indices point to individual tokens across different pages)
- **attn_sink merge** in the epilogue
- **Dual output**: `out [B,1,H,512]` + `lse [B,H,1]`

---

## Directory Structure

```
sparse_mla/
├── collective/
│   ├── xe_sparse_mla_mainloop.hpp   — Scattered gather + two-phase online softmax
│   └── xe_sparse_mla_epilogue.hpp   — LSE output + attn_sink merge
├── device/
│   ├── sparse_mla_decode_types.hpp  — Problem shape, config, args, runSparseMla()
│   ├── sparse_mla_decode_dispatch.hpp — Forward declarations per (dtype, page_size)
│   └── sparse_mla_runner.hpp        — SparseMLA device class (initialize, run)
├── kernel/
│   ├── xe_sparse_mla_kernel.hpp     — XeSparseMlaFwdKernel operator()
│   └── sparse_mla_tile_scheduler.hpp — Work distribution (batch × heads × V-tiles)
└── SPARSE_MLA_DESIGN.md            — This file
```

---

## Key Constants (DeepSeek V4)

```
D_NOPE      = 448   (latent nope dimension, 7 tiles × 64)
D_ROPE      = 64    (RoPE positional dimension)
D_QK        = 512   (D_NOPE + D_ROPE)
D_V         = 512   (output value dimension)
D_SLICE     = 64    (= QK_BLK_K, DPAS inner tile width)
H_Q         = 64    (query attention heads)
H_KV        = 1     (MLA: single shared KV head)
PAGE_SIZE   = 256   (tokens per page for k_cache; extra_k_cache uses 2 or 64)
TILE_N      = 64    (GEMM tile width — fixed regardless of page_size)
SWA_WINDOW  = 128   (sliding window token count)
C4_TOPK     = 512   (max compressed tokens selected; also 64, 8256 observed)
```

---

## Mainloop Design: Slice-by-Slice Scattered Gather

### Problem
Standard MLA reads one full page (contiguous tokens) per iteration.
Sparse MLA reads individual tokens from **scattered pages** via index arrays.
Full-page SLM staging `[TILE_N × 512]` = 64 KB — too large for bigger page sizes.

### Solution
Gather ONE d-slice `[TILE_N × 64]` = 8 KB at a time.
Reuse the same 8 KB SLM buffer for each of the 17 DPAS iterations.

### SLM Usage
```
SharedStorage:
  sg_max_data [NumSubgroups]         ~16 bytes   (cross-SG softmax)
  tile_slice  [TILE_N × D_SLICE]     8 KB        (one d-slice at a time)
                                     ────────
  Total:                             ~8 KB       (fits ANY page_size)
```

### Per-Tile Flow

```
process_token_tile(indices, K_cache, Kpe_cache, Q_nope, Q_pe):

  ┌─ Step 1: compute_tile_addresses (ONCE) ─────────────────────┐
  │  For i in 0..TILE_N:                                         │
  │    idx = indices[tile_start + i]                             │
  │    valid[i] = (idx >= 0)                                     │
  │    page_ids[i] = idx / page_size                             │
  │    token_offsets[i] = idx % page_size                        │
  │  Result: TileAddresses struct (reused 17 times)              │
  └──────────────────────────────────────────────────────────────┘
              │
              ▼
  ┌─ Step 2: GEMM1a — 8 iterations for K_c ─────────────────────┐
  │  for D in 0..7:                                              │
  │    gather_d_slice(K_base, addrs, d_offset=D*64) → SLM       │
  │    barrier()                                                  │
  │    S += Q_nope[:, D*64:(D+1)*64] × SLM_slice^T   (DPAS)    │
  └──────────────────────────────────────────────────────────────┘
              │
              ▼
  ┌─ Step 3: GEMM1b — 1 iteration for K_pe ─────────────────────┐
  │    gather_d_slice(Kpe_base, addrs, d_offset=0) → SLM        │
  │    barrier()                                                  │
  │    S += Q_pe × SLM_slice^T                        (DPAS)    │
  └──────────────────────────────────────────────────────────────┘
              │
              ▼  S = [1, TILE_N] raw scores
  ┌─ Step 4: Mask invalid tokens ───────────────────────────────┐
  │  S[i] = -inf where addrs.valid[i] == false                  │
  └──────────────────────────────────────────────────────────────┘
              │
              ▼
  ┌─ Step 5: Online softmax (identical to standard MLA) ────────┐
  │  m_new = max(m, scale * max(S))                              │
  │  correction = exp2(m_old - m_new)                            │
  │  P = exp2(scale * S - m_new)                                 │
  │  ℓ = ℓ * correction + sum(P)                                 │
  │  O_acc = O_acc * correction                                  │
  └──────────────────────────────────────────────────────────────┘
              │
              ▼  P = [1, TILE_N] probabilities
  ┌─ Step 6: GEMM2 — 8 iterations for V (V = K_c) ─────────────┐
  │  for VV in 0..7:                                             │
  │    gather_d_slice(K_base, addrs, d_offset=VV*64) → SLM      │
  │    barrier()                                                  │
  │    O_acc[VV] += P × SLM_slice                     (DPAS)    │
  └──────────────────────────────────────────────────────────────┘
```

### FP8 Page Stride Formula (stride(0))

```
total_page_bytes = page_size * 576 + ceil_to_576(page_size * 8)

Verified: page_size=256 → 149760, page_size=64 → 37440, page_size=2 → 1728
```

### Two-Phase Outer Loop

```cpp
operator()(...) {
  // Initialize: O_acc=0, m=-inf, ℓ=0
  bool is_first_tile = true;

  // Phase 1: SWA tokens from primary K cache
  for (tile in 0..ceil(swa_topk / TILE_N)):
    process_token_tile(K_3D, Kpe_3D, V_3D, swa_indices, ...)
    is_first_tile = false

  // Phase 2: Extra tokens from extra K cache (continues same accumulators)
  for (tile in 0..ceil(extra_topk / TILE_N)):
    process_token_tile(ExtraK_3D, ExtraKpe_3D, ExtraV_3D, extra_indices, ...)
    is_first_tile = false

  // O_acc, m, ℓ now reflect ALL tokens from both phases
  // → passed to epilogue
}
```

---

## gather_d_slice() — Cooperative Scattered Gather

```cpp
gather_d_slice(kv_cache_ptr, addrs, stride_page, stride_token, d_offset, thr_id):
  // Fills shared.tile_slice [TILE_N × D_SLICE] cooperatively
  total_elements = TILE_N * D_SLICE  // 4096
  for elem = thr_id; elem < total_elements; elem += num_work_items:
    token = elem / D_SLICE
    col = elem % D_SLICE
    if !addrs.valid[token]:
      SLM[elem] = 0
    else:
      page = addrs.page_ids[token]
      offset = addrs.token_offsets[token]
      SLM[elem] = cache[page * stride_page + offset * stride_token + d_offset + col]
  barrier()  // all work-items must finish before DPAS reads
```

---

## SLM Tensor Views (CuTe Rank-3)

CuTe's `local_tile` / `TiledCopy` require matching ranks with TileShape.
SLM tensors are shaped rank-3 to match `K_3D`'s `(k, d, num_blocks)`:

```cpp
// For GEMM1 (K reads): shape (TILE_N, 64, 1), stride (64, 1, TILE_N*64)
auto slm_slice_3D = make_tensor(
    make_smem_ptr(shared.tile_slice),
    make_layout(make_shape(Int<TILE_N>{}, Int<D_SLICE>{}, _1{}),
                make_stride(Int<D_SLICE>{}, _1{}, Int<TILE_N * D_SLICE>{})));

// For GEMM2 (V reads, transposed): shape (64, TILE_N, 1), stride (1, 64, TILE_N*64)
auto slm_slice_V_3D = make_tensor(
    make_smem_ptr(shared.tile_slice),
    make_layout(make_shape(Int<D_SLICE>{}, Int<TILE_N>{}, _1{}),
                make_stride(_1{}, Int<D_SLICE>{}, Int<TILE_N * D_SLICE>{})));
```

---

## Epilogue Design

After the mainloop produces `(O_acc, m, ℓ)`:

```
1. Cross-SG reduction (reduce_A): merge partial results if ReduceK > 1
2. Normalization: sparse_out = O_acc / ℓ
3. LSE computation: lse = m / log2(e) + ln(ℓ)
4. Attn_sink merge: w = exp(lse) / (exp(lse) + exp(attn_sink[h]))
                    out = w × sparse_out
5. Store: out → [B, 1, H, 512], lse → [B, H, 1]
```

---

## Kernel Arguments (KernelArguments struct)

```
From deepseek_v4_backend_radix → runSparseMla → kernel:

  shape:        FSparseMlAProblemShape (batch, heads, topk, page_size, etc.)
  Q_nope:       const ElementQ* → q[:,:,:,:512]
  Q_pe:         const ElementQ* → q[:,:,:,512:]  (same tensor, offset pointer)
  K:            const ElementK* → k_cache base
  K_pe:         const ElementK* → k_cache + D_NOPE offset
  extra_K:      const ElementK* → extra_k_cache base
  extra_K_pe:   const ElementK* → extra_k_cache + D_NOPE offset
  O:            ElementO* → output buffer
  lse_out:      float* → LSE output buffer
  attn_sink:    const float* → [H] per-head sink values
```

---

## Mainloop Arguments

```
  scale:            float (sm_scale × log2(e), pre-multiplied)
  page_size:        int
  total_page:       int (primary cache)
  ptr_swa_indices:  const int* → [B, 1, swa_topk] flattened
  swa_topk:         int (128)
  ptr_extra_indices: const int* → [B, 1, extra_topk] flattened
  extra_topk:       int (512)
```

---

## Grid Dimensions

```
grid.x = ceil(D_V / 64)    = 512/64 = 8       (V output tiles)
grid.y = ceil(seq_len_qo)  = 1                 (Q tiles, always 1 for decode)
grid.z = B × H             = batch × 64        (batch × heads)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Slice-by-slice (8 KB SLM) | Fits any PAGE_SIZE within 128 KB SLM budget |
| Addresses computed once per tile | 1 div + 1 mod per token, reused 17 times |
| No separate V gather | V = K_c in MLA (same addresses, same d-offsets) |
| Rank-3 SLM tensors | CuTe `local_tile` / TiledCopy require rank match |
| Single accumulator across phases | Online softmax correction handles phase boundary |
| attn_sink in epilogue, not mainloop | mainloop softmax() stays identical to standard MLA |
| Q passed as one tensor | Q_pe = Q_nope + D_NOPE (pointer offset, no copy) |

---

## Files Modified Outside sparse_mla/

| File | Change |
|------|--------|
| `include/sgl_flash_kernel_ops.h` | `flash_mla_sparse_decode` declaration |
| `src/torch_extension_sycl.cc` | Op registration (def + impl) |
| `src/CMakeLists.txt` | `include(SparseMlaDecodeXe20.cmake)` |
| `src/SparseMlaDecodeXe20.cmake` | Generates 4 compilation units |
| `src/sycl/sparse_mla_decode.cpp` | C++ dispatch (dtype → page_size) |
| `src/sycl/sparse_mla_decode_kernel.cpp.in` | Template for generated .cpp files |
| `python/sgl_kernel/attention.py` | `flash_mla_sparse_decode()` Python wrapper |
| `python/sgl_kernel/__init__.py` | Export |

---

## Current Status

- [x] Directory structure created
- [x] All 7 kernel header files
- [x] C++ dispatch + torch binding + Python wrapper
- [x] CMake build integration (4 compilation units)
- [x] Test suite (test_flash_mla_sparse_decode.py) with FP8 KV cache
- [x] Design doc (deepseek_v4_sparse_mla_decode_design.md)
- [ ] Compilation passes (fixing CuTe rank/type issues)
- [ ] Correctness validation against _sm120_sparse_decode_fwd reference
- [ ] FP8 dequantization in gather_d_slice (currently bf16 passthrough)
- [ ] Performance optimization (prefetch, larger TILE_N for sequential indices)
