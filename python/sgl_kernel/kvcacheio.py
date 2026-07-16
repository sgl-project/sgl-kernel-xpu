# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# KV-cache scatter/gather transfer ops for Intel XPU.
#
# Layout naming conventions:
#   lf  = layer-first  [num_layers, num_tokens, item_size]
#   pf  = page-first   [num_pages, num_layers, page_size, item_size]
#   ph  = page-head    [num_pages, head_num, page_size, num_layers, head_dim]

from typing import List

import torch

# Tuned for B580 (Xe2, 20 Xe-cores).
# sgs_per_wg=32: work-group size of 512 threads (32 sub-groups × 16 lanes).
# block_quota=16: fixed pool of 16*32=512 sub-groups.  num_wgs scales with the
#   token count up to that pool, filling all Xe-cores across the whole range
#   (measured 5-8x faster than an N-derived quota at N<=1024, which starves the
#   GPU to 1-2 work-groups).
_DEFAULT_SGS_PER_WG = 32
_DEFAULT_BLOCK_QUOTA = 16


# ---------------------------------------------------------------------------
# Group A: SYCL kernel-backed ops (device-side layout conversion)
# ---------------------------------------------------------------------------


def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = _DEFAULT_BLOCK_QUOTA,
    sgs_per_wg: int = _DEFAULT_SGS_PER_WG,
) -> None:
    """Single-layer lf→lf transfer for K and V."""
    torch.ops.sgl_kernel.transfer_kv_per_layer.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        sgs_per_wg,
    )


def transfer_kv_per_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = _DEFAULT_BLOCK_QUOTA,
    sgs_per_wg: int = _DEFAULT_SGS_PER_WG,
) -> None:
    """Single-layer lf→lf transfer, MLA (K only)."""
    torch.ops.sgl_kernel.transfer_kv_per_layer_mla.default(
        src,
        dst,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        sgs_per_wg,
    )


def transfer_kv_all_layer(
    src_k_layers: torch.Tensor,
    dst_k_layers: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = _DEFAULT_BLOCK_QUOTA,
    sgs_per_wg: int = _DEFAULT_SGS_PER_WG,
) -> None:
    """All-layer lf_tbl→lf_tbl transfer for K and V.

    src_k_layers / dst_k_layers are uint64 tensors of data_ptr() values,
    one per layer.
    """
    torch.ops.sgl_kernel.transfer_kv_all_layer.default(
        src_k_layers,
        dst_k_layers,
        src_v_layers,
        dst_v_layers,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        block_quota,
        sgs_per_wg,
    )


def transfer_kv_all_layer_mla(
    src_layers: torch.Tensor,
    dst_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = _DEFAULT_BLOCK_QUOTA,
    sgs_per_wg: int = _DEFAULT_SGS_PER_WG,
) -> None:
    """All-layer lf_tbl→lf_tbl transfer, MLA (K only)."""
    torch.ops.sgl_kernel.transfer_kv_all_layer_mla.default(
        src_layers,
        dst_layers,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        block_quota,
        sgs_per_wg,
    )


def transfer_kv_all_layer_lf_ph(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    page_size: int,
    head_num: int,
    block_quota: int = _DEFAULT_BLOCK_QUOTA,
    sgs_per_wg: int = _DEFAULT_SGS_PER_WG,
) -> None:
    """All-layer lf_tbl → page-head transfer for K and V."""
    torch.ops.sgl_kernel.transfer_kv_all_layer_lf_ph.default(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        page_size,
        head_num,
        block_quota,
        sgs_per_wg,
    )


def transfer_kv_per_layer_ph_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    page_size: int,
    head_num: int,
    block_quota: int = _DEFAULT_BLOCK_QUOTA,
    sgs_per_wg: int = _DEFAULT_SGS_PER_WG,
) -> None:
    """Single-layer page-head → lf transfer for K and V."""
    torch.ops.sgl_kernel.transfer_kv_per_layer_ph_lf.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        page_size,
        head_num,
        block_quota,
        sgs_per_wg,
    )


def transfer_kv_per_layer_pf_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = _DEFAULT_BLOCK_QUOTA,
    sgs_per_wg: int = _DEFAULT_SGS_PER_WG,
) -> None:
    """Single-layer page-first → lf transfer for K and V."""
    torch.ops.sgl_kernel.transfer_kv_per_layer_pf_lf.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        block_quota,
        sgs_per_wg,
    )


def transfer_kv_all_layer_lf_pf(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = _DEFAULT_BLOCK_QUOTA,
    sgs_per_wg: int = _DEFAULT_SGS_PER_WG,
) -> None:
    """All-layers lf → page-first transfer for K and V."""
    torch.ops.sgl_kernel.transfer_kv_all_layer_lf_pf.default(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        sgs_per_wg,
    )


def transfer_kv_per_layer_mla_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = _DEFAULT_BLOCK_QUOTA,
    sgs_per_wg: int = _DEFAULT_SGS_PER_WG,
) -> None:
    """Single-layer page-first → lf transfer for K only (MLA)."""
    torch.ops.sgl_kernel.transfer_kv_per_layer_mla_pf_lf.default(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        block_quota,
        sgs_per_wg,
    )


def transfer_kv_all_layer_mla_lf_pf(
    src_layers: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = _DEFAULT_BLOCK_QUOTA,
    sgs_per_wg: int = _DEFAULT_SGS_PER_WG,
) -> None:
    """All-layers lf → page-first transfer for K only (MLA)."""
    torch.ops.sgl_kernel.transfer_kv_all_layer_mla_lf_pf.default(
        src_layers,
        dst,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        sgs_per_wg,
    )


# ---------------------------------------------------------------------------
# Group B: Python/PyTorch fallbacks (host↔device; no equivalent of
# cudaMemcpyBatchAsync on XPU — use PyTorch copy_ page-by-page).
# ---------------------------------------------------------------------------


def _transfer_page_direct(
    src_buf: torch.Tensor,
    dst_buf: torch.Tensor,
    src_start: int,
    dst_start: int,
    count: int,
) -> None:
    dst_buf[dst_start : dst_start + count].copy_(
        src_buf[src_start : src_start + count], non_blocking=True
    )


def transfer_kv_direct(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> None:
    """Direct (PyTorch copy_) transfer for an arbitrary list of tensor pools.

    Contiguous runs of indices are coalesced into a single copy_ call.
    """
    assert len(src_layers) == len(dst_layers)
    assert src_indices.numel() == dst_indices.numel()
    assert src_indices.numel() % page_size == 0

    src_cpu = src_indices.cpu()
    dst_cpu = dst_indices.cpu()
    n = src_cpu.numel()
    src_ptr = src_cpu.tolist()
    dst_ptr = dst_cpu.tolist()

    num_layers = len(src_layers)
    start = 0
    for i in range(n):
        if i < n - 1:
            if src_ptr[i + 1] - src_ptr[i] == 1 and dst_ptr[i + 1] - dst_ptr[i] == 1:
                continue
            end = i + 1
        else:
            end = n
        count = end - start
        for j in range(num_layers):
            _transfer_page_direct(
                src_layers[j], dst_layers[j], src_ptr[start], dst_ptr[start], count
            )
        start = end


def transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> None:
    """Page-first (pinned host) → layer-first (device) for one layer.

    src_ptrs: list of per-layer tensors from the pf pool (or [kv_pool] for MLA).
    dst_ptrs: list of per-layer destination tensors.
    """
    assert src_indices.numel() == dst_indices.numel()
    assert src_indices.numel() % page_size == 0

    src_cpu = src_indices.cpu()
    dst_cpu = dst_indices.cpu()
    num_pages = src_cpu.numel() // page_size
    src_ptr = src_cpu.tolist()
    dst_ptr = dst_cpu.tolist()

    is_mla = len(src_ptrs) == 1
    num_layers = len(dst_ptrs) if is_mla else len(dst_ptrs) // 2

    for i in range(num_pages):
        s_page = src_ptr[i * page_size] // page_size
        d_start = dst_ptr[i * page_size]
        for j in range(num_layers):
            # src is pf: [num_pages, num_layers, page_size, item_size]
            # copy page_size rows from src_ptrs[0 or 1][s_page][layer_id+j]
            # to dst_ptrs[j] starting at d_start
            src_kv = src_ptrs[
                0 if is_mla else j
            ]  # per-layer tensor already sliced by caller
            # src_ptrs holds per-layer slices: shape [total_tokens, item_size]
            # The pf tensor layout means the caller passes the pf pool and we index it.
            # Here we mirror the CUDA fallback: src_ptrs[0].select(0, s_page).select(0, layer_id+j)
            # gives a [page_size, item_size] slice.
            src_slice = (
                src_ptrs[0 if is_mla else j].select(0, s_page).select(0, layer_id + j)
            )
            _transfer_page_direct(src_slice, dst_ptrs[j], 0, d_start, page_size)
            if not is_mla:
                src_slice_v = src_ptrs[1].select(0, s_page).select(0, layer_id + j)
                _transfer_page_direct(
                    src_slice_v, dst_ptrs[j + num_layers], 0, d_start, page_size
                )


def transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> None:
    """Layer-first (device) → page-first (pinned host) for all layers.

    src_ptrs: per-layer tensors [k0, k1, ..., v0, v1, ...] or [kv0, kv1, ...] for MLA.
    dst_ptrs: [dst_k_pool] or [dst_k_pool, dst_v_pool] where pools are
              shaped [num_pages, num_layers, page_size, item_size].
    """
    assert src_indices.numel() == dst_indices.numel()
    assert src_indices.numel() % page_size == 0

    src_cpu = src_indices.cpu()
    dst_cpu = dst_indices.cpu()
    num_pages = src_cpu.numel() // page_size
    src_ptr = src_cpu.tolist()
    dst_ptr = dst_cpu.tolist()

    is_mla = len(dst_ptrs) == 1
    num_layers = len(src_ptrs) if is_mla else len(src_ptrs) // 2

    for i in range(num_pages):
        s_start = src_ptr[i * page_size]
        d_page = dst_ptr[i * page_size] // page_size
        for j in range(num_layers):
            # dst is pf: dst_ptrs[0][d_page][j] is shape [page_size, item_size]
            dst_slice = dst_ptrs[0].select(0, d_page).select(0, j)
            _transfer_page_direct(src_ptrs[j], dst_slice, s_start, 0, page_size)
            if not is_mla:
                dst_slice_v = dst_ptrs[1].select(0, d_page).select(0, j)
                _transfer_page_direct(
                    src_ptrs[j + num_layers], dst_slice_v, s_start, 0, page_size
                )
