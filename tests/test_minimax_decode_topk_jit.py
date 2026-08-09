"""Accuracy tests for the XPU/SYCL MiniMax decode block-top-k kernels.

PyTorch reference is inlined so the test is self-contained. Block-id selections
are compared as sets rather than element-wise because tie-breaking among
exactly-equal scores is unspecified in the kernel's radix regimes.
"""

from __future__ import annotations

import pytest
import torch

HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()

try:
    from sgl_kernel.jit.minimax import (
        minimax_decode_topk,
        minimax_decode_topk_page_table,
    )

    HAS_SGL_JIT = True
except ImportError:
    HAS_SGL_JIT = False

DEVICE = "xpu"


def _ref_topk_block_ids(
    score: torch.Tensor,       # [H, B, S] fp32
    seq_lens: torch.Tensor,    # [B]
    block_size: int,
    topk: int,
) -> torch.Tensor:
    """Reference block-id output: front-packed, ``-1`` padded. Breaks ties by
    lower block id winning; the kernel only matches that in the small regime,
    so callers compare selections as sets (see ``_assert_topk_matches``)."""
    num_heads, batch, max_seqblock = score.shape
    out = torch.full(
        (num_heads, batch, topk),
        fill_value=-1,
        dtype=torch.int32,
        device=score.device,
    )
    seq_lens_l = seq_lens.to(torch.int64).tolist()
    for b in range(batch):
        num_blocks_raw = (seq_lens_l[b] + block_size - 1) // block_size
        num_blocks = min(num_blocks_raw, max_seqblock)
        if num_blocks == 0:
            continue
        for h in range(num_heads):
            row = score[h, b, :num_blocks]
            k = min(topk, num_blocks)
            # Stable sort by -score preserves ascending block-id order among
            # ties -> lower id wins, matching the CUDA is_greater comparator.
            order = torch.argsort(-row, stable=True)
            out[h, b, :k] = order[:k].to(torch.int32)
    return out


def _ref_page_table(
    score: torch.Tensor,        # [num_kv_heads, B, S] fp32
    seq_lens: torch.Tensor,     # [B]
    req_to_token: torch.Tensor, # [max_reqs, max_kv_len] int32
    slot_ids: torch.Tensor,     # [B] int64
    block_size: int,
    topk: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_heads, batch, max_seqblock = score.shape
    max_kv_len = req_to_token.shape[1]
    ppb = block_size // page_size
    max_sparse_pages = topk * ppb

    page_table = torch.empty(
        (batch * num_heads, max_sparse_pages),
        dtype=torch.int32,
        device=score.device,
    )
    real_seq_lens = torch.empty(
        (batch * num_heads,), dtype=torch.int32, device=score.device
    )

    seq_lens_l = seq_lens.to(torch.int64).tolist()
    slot_ids_l = slot_ids.tolist()

    for b in range(batch):
        seq_len = seq_lens_l[b]
        num_blocks_raw = (seq_len + block_size - 1) // block_size
        num_blocks = min(num_blocks_raw, max_seqblock)
        slot = slot_ids_l[b]
        r2t_row = req_to_token[slot]
        for h in range(num_heads):
            out_row = b * num_heads + h
            if num_blocks <= topk:
                selected = torch.arange(
                    num_blocks, dtype=torch.int64, device=score.device
                )
                eff_kv = seq_len
            else:
                row = score[h, b, :num_blocks]
                order = torch.argsort(-row, stable=True)[:topk]
                selected, _ = torch.sort(order.to(torch.int64))
                eff_kv = 0
                for bid in selected.tolist():
                    rem = seq_len - bid * block_size
                    eff_kv += rem if rem < block_size else block_size
            real_seq_lens[out_row] = eff_kv
            k_eff = selected.shape[0]
            offsets = (
                torch.arange(ppb, dtype=torch.int64, device=score.device)
                * page_size
            )
            tok = (
                selected[:, None] * block_size + offsets[None, :]
            ).clamp(max=max_kv_len - 1)
            pages = r2t_row[tok] // page_size * num_heads + h
            pages_flat = pages.reshape(-1).to(torch.int32)
            total = k_eff * ppb
            page_table[out_row, :total] = pages_flat
    return page_table, real_seq_lens


def _build_topk_inputs(
    num_heads: int,
    batch: int,
    max_seqblock: int,
    seq_lens_list: list[int],
    *,
    seq_dtype: torch.dtype = torch.int32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    score = torch.randn(
        (num_heads, batch, max_seqblock), dtype=torch.float32, device=DEVICE
    )
    seq_lens = torch.tensor(seq_lens_list, dtype=seq_dtype, device=DEVICE)
    return score, seq_lens


def _assert_topk_matches(
    kernel_out: torch.Tensor,
    ref_out: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    topk: int,
    max_seqblock: int,
) -> None:
    """Block-id top-k is compared as SETS per (head, batch) because the CUDA
    tie-break vs the reference tie-break may differ under identical scores.
    The invalid-position padding (``-1``) is checked element-wise."""
    num_heads, batch, _ = kernel_out.shape
    seq_lens_l = seq_lens.to(torch.int64).tolist()
    for b in range(batch):
        # Clamp to max_seqblock: kernel and reference both stop at the
        # materialized score columns, so k_eff must too.
        num_blocks = min(
            (seq_lens_l[b] + block_size - 1) // block_size, max_seqblock
        )
        k_eff = min(topk, num_blocks)
        for h in range(num_heads):
            kernel_set = set(kernel_out[h, b, :k_eff].tolist())
            ref_set = set(ref_out[h, b, :k_eff].tolist())
            assert kernel_set == ref_set, (
                f"topk mismatch at (h={h}, b={b}): "
                f"kernel={sorted(kernel_set)} ref={sorted(ref_set)}"
            )
            if k_eff < topk:
                assert (
                    (kernel_out[h, b, k_eff:] == -1).all().item()
                ), f"padding not -1 at (h={h}, b={b}): {kernel_out[h, b, k_eff:].tolist()}"



@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
@pytest.mark.skipif(not HAS_SGL_JIT, reason="Requires sgl_kernel.jit.minimax")
class TestMinimaxDecodeTopKBlockId:
    """Block-id output kernel across all three size regimes + edge cases."""

    def test_trivial_num_blocks_leq_topk(self) -> None:
        """num_blocks <= topk: identity block ids, -1 padded."""
        num_heads, batch, max_seqblock = 4, 3, 8
        block_size, topk = 64, 32
        # 1-2 blocks per row -> hits trivial fast-path.
        seq_lens_list = [block_size, 2 * block_size, block_size]
        score, seq_lens = _build_topk_inputs(
            num_heads, batch, max_seqblock, seq_lens_list, seed=1
        )
        kernel_out = minimax_decode_topk(score, seq_lens, block_size, topk)
        ref_out = _ref_topk_block_ids(score, seq_lens, block_size, topk)
        _assert_topk_matches(
            kernel_out, ref_out, seq_lens, block_size, topk, max_seqblock
        )

    def test_small_regime(self) -> None:
        """num_blocks in (topk, kSmallThreshold=128] -> O(n^2) rank-by-compare."""
        num_heads, batch, max_seqblock = 2, 4, 128
        block_size, topk = 64, 16
        seq_lens_list = [64 * 100, 64 * 128, 64 * 90, 64 * 80]
        score, seq_lens = _build_topk_inputs(
            num_heads, batch, max_seqblock, seq_lens_list, seed=2
        )
        kernel_out = minimax_decode_topk(score, seq_lens, block_size, topk)
        ref_out = _ref_topk_block_ids(score, seq_lens, block_size, topk)
        _assert_topk_matches(
            kernel_out, ref_out, seq_lens, block_size, topk, max_seqblock
        )

    def test_register_1_regime(self) -> None:
        """num_blocks in (kSmallThreshold, kCTASize=512] -> one-elem-per-thread radix."""
        num_heads, batch, max_seqblock = 2, 2, 512
        block_size, topk = 64, 32
        seq_lens_list = [64 * 400, 64 * 512]
        score, seq_lens = _build_topk_inputs(
            num_heads, batch, max_seqblock, seq_lens_list, seed=3
        )
        kernel_out = minimax_decode_topk(score, seq_lens, block_size, topk)
        ref_out = _ref_topk_block_ids(score, seq_lens, block_size, topk)
        _assert_topk_matches(
            kernel_out, ref_out, seq_lens, block_size, topk, max_seqblock
        )

    def test_register_M_regime(self) -> None:
        """num_blocks in (kCTASize, kMaxNumBlocks=4096] -> multi-elem-per-thread radix."""
        num_heads, batch, max_seqblock = 2, 2, 4096
        block_size, topk = 64, 32
        seq_lens_list = [64 * 2048, 64 * 4096]
        score, seq_lens = _build_topk_inputs(
            num_heads, batch, max_seqblock, seq_lens_list, seed=4
        )
        kernel_out = minimax_decode_topk(score, seq_lens, block_size, topk)
        ref_out = _ref_topk_block_ids(score, seq_lens, block_size, topk)
        _assert_topk_matches(
            kernel_out, ref_out, seq_lens, block_size, topk, max_seqblock
        )

    def test_seqlens_i64(self) -> None:
        """int64 seq_lens dispatches to the ``_i64`` exported symbol."""
        num_heads, batch, max_seqblock = 2, 2, 256
        block_size, topk = 64, 16
        seq_lens_list = [64 * 200, 64 * 240]
        score, seq_lens = _build_topk_inputs(
            num_heads,
            batch,
            max_seqblock,
            seq_lens_list,
            seq_dtype=torch.int64,
            seed=5,
        )
        kernel_out = minimax_decode_topk(score, seq_lens, block_size, topk)
        ref_out = _ref_topk_block_ids(score, seq_lens, block_size, topk)
        _assert_topk_matches(
            kernel_out, ref_out, seq_lens, block_size, topk, max_seqblock
        )

    def test_topk_1(self) -> None:
        """topk=1 exercises the single-selection path."""
        num_heads, batch, max_seqblock = 2, 3, 256
        block_size, topk = 64, 1
        seq_lens_list = [64 * 100, 64 * 200, 64 * 256]
        score, seq_lens = _build_topk_inputs(
            num_heads, batch, max_seqblock, seq_lens_list, seed=6
        )
        kernel_out = minimax_decode_topk(score, seq_lens, block_size, topk)
        ref_out = _ref_topk_block_ids(score, seq_lens, block_size, topk)
        _assert_topk_matches(
            kernel_out, ref_out, seq_lens, block_size, topk, max_seqblock
        )

    def test_out_param(self) -> None:
        """Caller-supplied ``out`` tensor is written in-place and returned."""
        num_heads, batch, max_seqblock = 2, 2, 256
        block_size, topk = 64, 16
        seq_lens_list = [64 * 200, 64 * 240]
        score, seq_lens = _build_topk_inputs(
            num_heads, batch, max_seqblock, seq_lens_list, seed=7
        )
        out = torch.empty(
            (num_heads, batch, topk), dtype=torch.int32, device=DEVICE
        )
        kernel_out = minimax_decode_topk(
            score, seq_lens, block_size, topk, out=out
        )
        assert kernel_out.data_ptr() == out.data_ptr()
        ref_out = _ref_topk_block_ids(score, seq_lens, block_size, topk)
        _assert_topk_matches(
            kernel_out, ref_out, seq_lens, block_size, topk, max_seqblock
        )



def _build_page_table_inputs(
    num_heads: int,
    batch: int,
    max_seqblock: int,
    seq_lens_list: list[int],
    max_kv_len: int,
    *,
    contiguous_r2t: bool = True,
    seq_dtype: torch.dtype = torch.int32,
    seed: int = 0,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    torch.manual_seed(seed)
    score = torch.randn(
        (num_heads, batch, max_seqblock), dtype=torch.float32, device=DEVICE
    )
    seq_lens = torch.tensor(seq_lens_list, dtype=seq_dtype, device=DEVICE)

    max_reqs = batch
    slot_ids = torch.arange(batch, dtype=torch.int64, device=DEVICE)
    if contiguous_r2t:
        base = (
            torch.arange(batch, dtype=torch.int32, device=DEVICE) * max_kv_len
        )
        req_to_token = (
            base[:, None]
            + torch.arange(max_kv_len, dtype=torch.int32, device=DEVICE)[None, :]
        )
    else:
        req_to_token = torch.empty(
            (max_reqs, max_kv_len), dtype=torch.int32, device=DEVICE
        )
        for i in range(max_reqs):
            perm = torch.randperm(max_kv_len, device=DEVICE) + i * max_kv_len
            req_to_token[i] = perm.to(torch.int32)
    return score, seq_lens, req_to_token, slot_ids


@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
@pytest.mark.skipif(not HAS_SGL_JIT, reason="Requires sgl_kernel.jit.minimax")
class TestMinimaxDecodeTopKPageTable:
    """Page-table output kernel: trivial + non-trivial + DP-attention cases."""

    def test_trivial_rows(self) -> None:
        """num_blocks <= topk: every block selected, all tokens valid."""
        num_heads, batch = 4, 3
        block_size, topk, page_size = 64, 32, 1
        max_seqblock = 8
        seq_lens_list = [block_size, 2 * block_size, block_size]
        max_kv_len = max_seqblock * block_size
        score, seq_lens, req_to_token, slot_ids = _build_page_table_inputs(
            num_heads, batch, max_seqblock, seq_lens_list, max_kv_len, seed=11
        )
        pt_kernel, sl_kernel = minimax_decode_topk_page_table(
            score, seq_lens, req_to_token, slot_ids,
            block_size, topk, page_size,
        )
        pt_ref, sl_ref = _ref_page_table(
            score, seq_lens, req_to_token, slot_ids,
            block_size, topk, page_size,
        )
        # Trivial-row real_seq_lens is deterministic: block-set is identity so
        # ordering matches -> compare exactly.
        assert torch.equal(sl_kernel, sl_ref), (
            f"real_seq_lens mismatch: kernel={sl_kernel.tolist()} "
            f"ref={sl_ref.tolist()}"
        )
        # Only the first ``num_blocks * ppb`` entries per row are written by
        # the kernel; compare only those.
        seq_lens_l = seq_lens.to(torch.int64).tolist()
        for b in range(batch):
            num_blocks = (seq_lens_l[b] + block_size - 1) // block_size
            valid = num_blocks * (block_size // page_size)
            for h in range(num_heads):
                row = b * num_heads + h
                assert torch.equal(
                    pt_kernel[row, :valid], pt_ref[row, :valid]
                ), f"page_table mismatch at row {row}"

    def test_nontrivial_selection(self) -> None:
        """num_blocks > topk: TopK -> sort -> page emit, checked as sets per row."""
        num_heads, batch = 1, 2
        block_size, topk, page_size = 64, 16, 1
        max_seqblock = 512
        seq_lens_list = [block_size * 400, block_size * 512]
        max_kv_len = max_seqblock * block_size
        score, seq_lens, req_to_token, slot_ids = _build_page_table_inputs(
            num_heads, batch, max_seqblock, seq_lens_list, max_kv_len, seed=12
        )
        pt_kernel, sl_kernel = minimax_decode_topk_page_table(
            score, seq_lens, req_to_token, slot_ids,
            block_size, topk, page_size,
        )
        pt_ref, sl_ref = _ref_page_table(
            score, seq_lens, req_to_token, slot_ids,
            block_size, topk, page_size,
        )
        # real_seq_lens depends only on the *set* of selected blocks: same set
        # -> same total. Kernel tie-break may differ from reference tie-break
        # among identical scores, but tie-breaks aren't expected on random
        # fp32 inputs, so an exact match is fine here.
        assert torch.equal(sl_kernel, sl_ref)
        # Page-table rows depend on selected block ids sorted ascending: with
        # the same block set we get the same rows, so element-wise compare.
        assert torch.equal(pt_kernel, pt_ref)

    def test_page_size_gt_1(self) -> None:
        """block_size=64, page_size=8 -> ppb=8 pages per selected block."""
        num_heads, batch = 2, 2
        block_size, topk, page_size = 64, 16, 8
        max_seqblock = 256
        seq_lens_list = [block_size * 200, block_size * 256]
        max_kv_len = max_seqblock * block_size
        score, seq_lens, req_to_token, slot_ids = _build_page_table_inputs(
            num_heads, batch, max_seqblock, seq_lens_list, max_kv_len, seed=13
        )
        pt_kernel, sl_kernel = minimax_decode_topk_page_table(
            score, seq_lens, req_to_token, slot_ids,
            block_size, topk, page_size,
        )
        pt_ref, sl_ref = _ref_page_table(
            score, seq_lens, req_to_token, slot_ids,
            block_size, topk, page_size,
        )
        assert torch.equal(sl_kernel, sl_ref)
        assert torch.equal(pt_kernel, pt_ref)

    def test_paged_r2t_randperm(self) -> None:
        """Shuffled req_to_token: kernel must follow the r2t indirection."""
        num_heads, batch = 1, 3
        block_size, topk, page_size = 64, 16, 1
        max_seqblock = 128
        seq_lens_list = [block_size * 100, block_size * 128, block_size * 90]
        max_kv_len = max_seqblock * block_size
        score, seq_lens, req_to_token, slot_ids = _build_page_table_inputs(
            num_heads, batch, max_seqblock, seq_lens_list, max_kv_len,
            contiguous_r2t=False, seed=14,
        )
        pt_kernel, sl_kernel = minimax_decode_topk_page_table(
            score, seq_lens, req_to_token, slot_ids,
            block_size, topk, page_size,
        )
        pt_ref, sl_ref = _ref_page_table(
            score, seq_lens, req_to_token, slot_ids,
            block_size, topk, page_size,
        )
        assert torch.equal(sl_kernel, sl_ref)
        assert torch.equal(pt_kernel, pt_ref)

    def test_multi_kv_head_head_encoding(self) -> None:
        """DP attention: page indices head-encoded as base_page*num_heads + h."""
        num_heads, batch = 4, 2
        block_size, topk, page_size = 64, 16, 1
        max_seqblock = 128
        seq_lens_list = [block_size * 100, block_size * 128]
        max_kv_len = max_seqblock * block_size
        score, seq_lens, req_to_token, slot_ids = _build_page_table_inputs(
            num_heads, batch, max_seqblock, seq_lens_list, max_kv_len, seed=15
        )
        pt_kernel, sl_kernel = minimax_decode_topk_page_table(
            score, seq_lens, req_to_token, slot_ids,
            block_size, topk, page_size,
        )
        pt_ref, sl_ref = _ref_page_table(
            score, seq_lens, req_to_token, slot_ids,
            block_size, topk, page_size,
        )
        assert torch.equal(sl_kernel, sl_ref)
        assert torch.equal(pt_kernel, pt_ref)

    def test_seqlens_i64(self) -> None:
        """int64 seq_lens dispatches to the ``_i64`` page-table symbol."""
        num_heads, batch = 2, 2
        block_size, topk, page_size = 64, 16, 1
        max_seqblock = 256
        seq_lens_list = [block_size * 200, block_size * 240]
        max_kv_len = max_seqblock * block_size
        score, seq_lens, req_to_token, slot_ids = _build_page_table_inputs(
            num_heads, batch, max_seqblock, seq_lens_list, max_kv_len,
            seq_dtype=torch.int64, seed=16,
        )
        pt_kernel, sl_kernel = minimax_decode_topk_page_table(
            score, seq_lens, req_to_token, slot_ids,
            block_size, topk, page_size,
        )
        pt_ref, sl_ref = _ref_page_table(
            score, seq_lens, req_to_token, slot_ids,
            block_size, topk, page_size,
        )
        assert torch.equal(sl_kernel, sl_ref)
        assert torch.equal(pt_kernel, pt_ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
