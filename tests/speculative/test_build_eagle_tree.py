"""Tests for build_tree_kernel_efficient (EAGLE draft-tree metadata) on XPU.

The python reference `_ref_build_tree` and the random tree generator mirror
test/registered/cpu/test_spec_kernels.py upstream; the golden vectors in
test_build_tree_upstream_golden come from the CUDA UT
test/registered/spec/utils/test_build_eagle_tree.py.
"""

import pytest
import torch
import utils
from sgl_kernel import TreeMaskMode, build_tree_kernel_efficient

device = utils.get_device()


def _topk1_chain_inputs(bs, num_steps):
    """MTP-style chain (topk=1): token i descends from i-1, index 0 is the root."""
    parent_width = num_steps if num_steps > 1 else 0
    parent_list = torch.arange(-1, parent_width - 1, dtype=torch.int64).repeat(bs, 1)
    selected_index = torch.arange(num_steps, dtype=torch.int64).repeat(bs, 1)
    return parent_list, selected_index


def _gen_draft_tree(bs, topk, num_steps, draft_token_num):
    """Simulate the EAGLE draft loop (select_top_k_tokens + organize_draft_results)
    with random probabilities to obtain a valid random (parent_list, selected_index).
    """
    scores = torch.rand(bs, topk, dtype=torch.float32)
    score_chunks = [scores]
    parents_chunks = [torch.arange(-1, topk, dtype=torch.int64).expand(bs, -1)]
    cum_scores = scores
    for i in range(1, num_steps):
        # Probabilities in (0, 1): a child cumulative score is strictly smaller
        # than its parent's, so the global topk always keeps full ancestor chains.
        step_p = torch.rand(bs, topk, topk, dtype=torch.float32)
        expand_scores = cum_scores.unsqueeze(2) * step_p
        cum_scores, topk_cs_index = torch.topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )
        score_chunks.append(expand_scores.flatten(start_dim=1))
        parents_chunks.append(topk_cs_index + (topk * topk * (i - 1) + topk))
    score_flat = torch.cat(score_chunks, dim=1)
    selected_index = torch.sort(
        torch.topk(score_flat, draft_token_num - 1, dim=-1).indices, dim=-1
    ).values
    parent_list = torch.cat(parents_chunks[:-1], dim=1)
    return parent_list, selected_index


def _organize_draft_results(score_list, token_list, parents_list, num_draft_token):
    """Port of sglang.srt.speculative.eagle_utils.organize_draft_results."""
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = torch.sort(top_scores.indices).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(
            batch_size, 0, dtype=torch.long, device=parents_list[0].device
        )
    return parent_list, top_scores_index, draft_tokens


def _ref_build_tree(parent_list, selected_index, seq_lens, topk, draft_token_num):
    """Serial CPU reference for the tree metadata (QLEN_ONLY mask block)."""
    bs = selected_index.shape[0]
    retrieve_index = torch.full((bs, draft_token_num), -1, dtype=torch.int64)
    retrieve_next_token = torch.full((bs, draft_token_num), -1, dtype=torch.int64)
    retrieve_next_sibling = torch.full((bs, draft_token_num), -1, dtype=torch.int64)
    positions = torch.zeros(bs * draft_token_num, dtype=torch.int64)
    tree_mask = torch.zeros(bs, draft_token_num, draft_token_num, dtype=torch.bool)

    for bid in range(bs):
        off = bid * draft_token_num
        sel = selected_index[bid].tolist()
        parents = parent_list[bid].tolist()
        retrieve_index[bid] = torch.arange(off, off + draft_token_num)

        # parent position (in tree-node numbering) of each node i >= 1
        parent_pos = [0] * draft_token_num
        for i in range(1, draft_token_num):
            parent_tb_idx = sel[i - 1] // topk
            if parent_tb_idx == 0:
                parent_pos[i] = 0
            else:
                parent_token_idx = parents[parent_tb_idx]
                parent_pos[i] = sel.index(parent_token_idx) + 1

        # head-insertion linking, iterating from the last node (kernel order)
        next_token = [-1] * draft_token_num
        next_sibling = [-1] * draft_token_num
        for i in range(draft_token_num - 1, 0, -1):
            p = parent_pos[i]
            if next_token[p] == -1:
                next_token[p] = i
            else:
                next_sibling[i] = next_token[p]
                next_token[p] = i
        retrieve_next_token[bid] = torch.tensor(next_token, dtype=torch.int64)
        retrieve_next_sibling[bid] = torch.tensor(next_sibling, dtype=torch.int64)

        seq_len = int(seq_lens[bid])
        positions[off] = seq_len
        tree_mask[bid, :, 0] = True
        for i in range(1, draft_token_num):
            ancestors = []
            j = i
            while j != 0:
                ancestors.append(j)
                j = parent_pos[j]
            positions[off + i] = seq_len + len(ancestors)
            for j in ancestors:
                tree_mask[bid, i, j] = True

    return (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        positions,
        tree_mask,
    )


def _run_kernel(
    parent_list,
    selected_index,
    seq_lens,
    topk,
    num_steps,
    draft_token_num,
    mode,
    tree_mask=None,
):
    """Allocate the output buffers the way eagle_utils does and call the op."""
    bs = seq_lens.numel()
    if tree_mask is None:
        if mode == TreeMaskMode.QLEN_ONLY:
            numel = bs * draft_token_num * draft_token_num
        else:  # FULL_MASK also carries the (all-true) committed-prefix columns
            numel = (
                int(seq_lens.sum()) * draft_token_num
                + bs * draft_token_num * draft_token_num
            )
        tree_mask = torch.full((numel,), True, dtype=torch.bool, device=device)

    positions = torch.zeros(bs * draft_token_num, dtype=torch.int64, device=device)
    retrieve_buf = torch.full(
        (3, bs, draft_token_num), -1, dtype=torch.int64, device=device
    )
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf

    build_tree_kernel_efficient(
        parent_list.to(device),
        selected_index.to(device),
        seq_lens.to(device),
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        topk,
        num_steps,
        draft_token_num,
        mode,
    )
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
    )


def _split_full_mask(full_mask, seq_lens, draft_token_num):
    """FULL_MASK -> (per-request prefix columns, per-request qlen x qlen block)."""
    prefixes, blocks = [], []
    offset = 0
    for seq_len in seq_lens.tolist():
        row_len = seq_len + draft_token_num
        chunk = full_mask[offset : offset + row_len * draft_token_num].view(
            draft_token_num, row_len
        )
        prefixes.append(chunk[:, :seq_len])
        blocks.append(chunk[:, seq_len:])
        offset += row_len * draft_token_num
    return prefixes, blocks


def _check_against_reference(
    parent_list, selected_index, seq_lens, topk, num_steps, draft_token_num
):
    """Both mask modes must match the reference; returns (tree_mask_ref, positions_ref)."""
    bs = seq_lens.numel()
    (
        retrieve_index_ref,
        retrieve_next_token_ref,
        retrieve_next_sibling_ref,
        positions_ref,
        tree_mask_ref,
    ) = _ref_build_tree(parent_list, selected_index, seq_lens, topk, draft_token_num)

    for mode in (TreeMaskMode.QLEN_ONLY, TreeMaskMode.FULL_MASK):
        (
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
        ) = _run_kernel(
            parent_list,
            selected_index,
            seq_lens,
            topk,
            num_steps,
            draft_token_num,
            mode,
        )
        torch.testing.assert_close(
            retrieve_index.cpu(), retrieve_index_ref, atol=0, rtol=0
        )
        torch.testing.assert_close(
            retrieve_next_token.cpu(), retrieve_next_token_ref, atol=0, rtol=0
        )
        torch.testing.assert_close(
            retrieve_next_sibling.cpu(), retrieve_next_sibling_ref, atol=0, rtol=0
        )
        torch.testing.assert_close(positions.cpu(), positions_ref, atol=0, rtol=0)

        if mode == TreeMaskMode.QLEN_ONLY:
            torch.testing.assert_close(
                tree_mask.cpu().view(bs, draft_token_num, draft_token_num),
                tree_mask_ref,
                atol=0,
                rtol=0,
            )
        else:
            prefixes, blocks = _split_full_mask(
                tree_mask.cpu(), seq_lens, draft_token_num
            )
            for bid in range(bs):
                assert (
                    prefixes[bid].all().item()
                ), f"prefix columns clobbered (bid={bid})"
                torch.testing.assert_close(
                    blocks[bid], tree_mask_ref[bid], atol=0, rtol=0
                )

    return tree_mask_ref, positions_ref


def test_build_tree_chain_topk1():
    # MTP config: topk=1, steps=3, draft_token_num=4
    num_steps, draft_token_num = 3, 4
    bs = 2
    seq_lens = torch.tensor([7, 12], dtype=torch.int64)
    parent_list, selected_index = _topk1_chain_inputs(bs, num_steps)
    tree_mask_ref, positions_ref = _check_against_reference(
        parent_list, selected_index, seq_lens, 1, num_steps, draft_token_num
    )
    # A chain must yield a causal (lower-triangular) mask and consecutive positions.
    tril = torch.tril(torch.ones(draft_token_num, draft_token_num, dtype=torch.bool))
    for bid in range(bs):
        torch.testing.assert_close(tree_mask_ref[bid], tril, atol=0, rtol=0)
        assert positions_ref[
            bid * draft_token_num : (bid + 1) * draft_token_num
        ].tolist() == [int(seq_lens[bid]) + i for i in range(draft_token_num)]


def test_build_tree_chain_mtp_step1():
    # MTP steps=1: parent_list is the empty (bs, 0) tensor organize_draft_results
    # emits when there are no non-root parents; the kernel must accept it.
    num_steps, draft_token_num = 1, 2
    bs = 2
    seq_lens = torch.tensor([7, 12], dtype=torch.int64)
    parent_list, selected_index = _topk1_chain_inputs(bs, num_steps)
    assert parent_list.shape == (bs, 0)
    tree_mask_ref, _ = _check_against_reference(
        parent_list, selected_index, seq_lens, 1, num_steps, draft_token_num
    )
    tril = torch.tril(torch.ones(draft_token_num, draft_token_num, dtype=torch.bool))
    for bid in range(bs):
        torch.testing.assert_close(tree_mask_ref[bid], tril, atol=0, rtol=0)


def test_build_tree_topk2_hand_case():
    # topk=2, steps=2, draft_token_num=4. Nodes 1 and 2 are children of the
    # root; node 3 is a child of node 1.
    topk, num_steps, draft_token_num = 2, 2, 4
    parent_list = torch.tensor([[-1, 0, 1]], dtype=torch.int64)
    selected_index = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    seq_lens = torch.tensor([5], dtype=torch.int64)

    (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
    ) = _run_kernel(
        parent_list,
        selected_index,
        seq_lens,
        topk,
        num_steps,
        draft_token_num,
        TreeMaskMode.QLEN_ONLY,
    )
    assert retrieve_index.tolist() == [[0, 1, 2, 3]]
    assert retrieve_next_token.tolist() == [[1, 3, -1, -1]]
    assert retrieve_next_sibling.tolist() == [[-1, 2, -1, -1]]
    assert positions.tolist() == [5, 6, 6, 7]
    assert tree_mask.cpu().view(draft_token_num, draft_token_num).int().tolist() == [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
    ]
    _check_against_reference(
        parent_list, selected_index, seq_lens, topk, num_steps, draft_token_num
    )


@pytest.mark.parametrize("seq_len_dtype", [torch.int32, torch.int64])
def test_build_tree_seq_len_dtypes(seq_len_dtype):
    """verified_seq_len is int64 in sglang but int32 elsewhere; both must work."""
    topk, num_steps, draft_token_num = 4, 3, 16
    bs = 3
    torch.manual_seed(1234)
    seq_lens = torch.tensor([9, 3, 21], dtype=seq_len_dtype)
    parent_list, selected_index = _gen_draft_tree(bs, topk, num_steps, draft_token_num)
    _check_against_reference(
        parent_list, selected_index, seq_lens, topk, num_steps, draft_token_num
    )


@pytest.mark.parametrize(
    "topk,num_steps,draft_token_num,bs",
    [
        (1, 5, 6, 4),
        (2, 3, 8, 2),
        (4, 3, 16, 3),
        (8, 4, 32, 2),
        (4, 6, 64, 1),
    ],
)
def test_build_tree_random_matches_reference(topk, num_steps, draft_token_num, bs):
    torch.manual_seed(1234)
    seq_lens = torch.randint(1, 64, (bs,), dtype=torch.int64)
    parent_list, selected_index = _gen_draft_tree(bs, topk, num_steps, draft_token_num)
    tree_mask_ref, positions_ref = _check_against_reference(
        parent_list, selected_index, seq_lens, topk, num_steps, draft_token_num
    )

    # Structural invariants: causality, self-attention, depth bounded by the
    # number of speculative steps, and depth == #visible draft tokens - 1.
    for bid in range(bs):
        mask = tree_mask_ref[bid]
        assert mask.diagonal().all().item()
        assert torch.triu(mask.int(), diagonal=1).sum().item() == 0
        depths = positions_ref[
            bid * draft_token_num : (bid + 1) * draft_token_num
        ] - int(seq_lens[bid])
        assert int(depths[0]) == 0
        assert bool((depths[1:] >= 1).all())
        assert bool((depths <= num_steps).all())
        torch.testing.assert_close(
            depths, mask.sum(dim=1).to(torch.int64) - 1, atol=0, rtol=0
        )


def test_skip_prefix_fill_preserves_tree_blocks():
    """The kernel must write every tree cell regardless of the caller's prefix fill.

    eagle_utils skips the (up to 100s of MB) `tree_mask.fill_(True)` when nothing
    reads the prefix columns, so the qlen x qlen block must be identical whether
    the buffer arrived all-True or all-False.
    """
    topk, num_steps, draft_token_num = 1, 3, 4
    bs = 2
    seq_lens = torch.tensor([5, 10], dtype=torch.int64)
    parent_list, selected_index = _topk1_chain_inputs(bs, num_steps)
    numel = (
        int(seq_lens.sum()) * draft_token_num + bs * draft_token_num * draft_token_num
    )

    def build(fill_prefix_mask):
        buf = torch.full((numel,), fill_prefix_mask, dtype=torch.bool, device=device)
        return _run_kernel(
            parent_list,
            selected_index,
            seq_lens,
            topk,
            num_steps,
            draft_token_num,
            TreeMaskMode.FULL_MASK,
            tree_mask=buf,
        )

    filled = build(True)
    skipped = build(False)

    filled_prefix, filled_blocks = _split_full_mask(
        filled[0].cpu(), seq_lens, draft_token_num
    )
    skipped_prefix, skipped_blocks = _split_full_mask(
        skipped[0].cpu(), seq_lens, draft_token_num
    )

    for bid in range(bs):
        assert torch.equal(filled_blocks[bid], skipped_blocks[bid]), (
            "tree blocks diverged: the kernel must write every tree cell "
            "regardless of the prefix fill"
        )
        # Anti-vacuous: proves the two runs really differ on the prefix columns.
        assert filled_prefix[bid].all(), "fill did not mark the prefix columns"
        assert not skipped_prefix[bid].any(), "kernel wrote into the prefix columns"

    for idx, name in enumerate(
        ("positions", "retrieve_index", "retrieve_next_token", "retrieve_next_sibling"),
        start=1,
    ):
        assert torch.equal(filled[idx], skipped[idx]), f"{name} diverged"


def test_build_tree_upstream_golden():
    """Golden vectors captured from the CUDA UT (test_build_eagle_tree.py).

    The raw EAGLE draft trace goes through organize_draft_results, exactly as in
    the sglang runtime, and the expected positions / retrieve_* are the CUDA
    kernel's outputs.
    """
    bonus_tokens = torch.tensor([29974, 13], device=device, dtype=torch.int32)
    score_list = [
        torch.tensor(
            [
                [[7.1127e-01, 2.8292e-01, 2.2995e-03, 1.7357e-03]],
                [[9.7476e-01, 2.2219e-02, 6.5031e-04, 1.3212e-04]],
            ],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            [
                [
                    [6.9142e-01, 1.2863e-02, 1.6873e-03, 1.1871e-03],
                    [2.4787e-01, 1.8818e-02, 1.4204e-02, 9.2235e-04],
                    [2.2971e-03, 1.6700e-06, 1.8737e-07, 8.3146e-08],
                    [1.2771e-03, 2.4374e-04, 1.7832e-04, 1.1947e-05],
                ],
                [
                    [8.4832e-02, 6.6068e-02, 5.8304e-02, 5.7851e-02],
                    [2.3616e-03, 1.1243e-03, 5.4368e-04, 2.7768e-04],
                    [2.5286e-04, 1.5578e-04, 2.8817e-05, 1.2888e-05],
                    [1.2834e-04, 2.5417e-06, 1.1279e-06, 1.6088e-08],
                ],
            ],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            [
                [
                    [6.6438e-01, 2.6997e-02, 2.4236e-05, 4.0821e-06],
                    [2.4402e-01, 2.8409e-03, 5.0935e-04, 2.9022e-04],
                    [1.6178e-02, 2.0567e-03, 4.5892e-04, 3.0034e-05],
                    [1.3023e-02, 5.0497e-04, 3.6371e-04, 8.7750e-05],
                ],
                [
                    [2.3263e-02, 2.0054e-02, 9.3990e-03, 2.7783e-03],
                    [6.4156e-02, 5.5506e-04, 1.0429e-04, 9.7211e-05],
                    [4.9950e-02, 5.0630e-03, 9.0068e-04, 3.3656e-04],
                    [7.5817e-03, 8.5731e-04, 6.9972e-04, 6.0793e-04],
                ],
            ],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            [
                [
                    [6.6420e-01, 1.0525e-04, 6.5864e-05, 1.2253e-06],
                    [1.3019e-01, 1.0461e-01, 5.2083e-03, 1.6777e-03],
                    [2.0103e-02, 6.7335e-03, 1.2625e-04, 1.0364e-05],
                    [1.5142e-02, 7.0819e-04, 9.6595e-05, 8.7951e-05],
                ],
                [
                    [5.8608e-02, 1.8840e-03, 7.8535e-04, 4.4400e-04],
                    [1.2185e-02, 2.0684e-03, 1.7418e-03, 1.4327e-03],
                    [6.2455e-03, 6.1487e-03, 2.6862e-03, 1.8034e-03],
                    [1.8590e-03, 1.6151e-03, 1.2481e-03, 3.6038e-04],
                ],
            ],
            dtype=torch.float32,
            device=device,
        ),
    ]
    token_list = [
        torch.tensor(
            [[29896, 29906, 29900, 29945], [13, 2, 29871, 28956]],
            dtype=torch.int64,
            device=device,
        ),
        torch.tensor(
            [
                [
                    29889,
                    29974,
                    29945,
                    29900,
                    29974,
                    29922,
                    29930,
                    29958,
                    29889,
                    29974,
                    29930,
                    29945,
                    29974,
                    29922,
                    29930,
                    29958,
                ],
                [
                    22550,
                    4136,
                    16492,
                    8439,
                    29871,
                    2,
                    3001,
                    13,
                    2,
                    13,
                    29906,
                    29946,
                    2,
                    13,
                    29871,
                    259,
                ],
            ],
            dtype=torch.int64,
            device=device,
        ),
        torch.tensor(
            [
                [
                    29946,
                    29945,
                    29953,
                    29906,
                    29896,
                    29945,
                    29900,
                    29906,
                    29896,
                    29945,
                    29906,
                    29953,
                    29896,
                    29945,
                    29906,
                    29946,
                ],
                [
                    29871,
                    2,
                    29901,
                    29889,
                    29871,
                    2,
                    395,
                    259,
                    29901,
                    29871,
                    2,
                    29889,
                    3001,
                    1234,
                    7146,
                    2186,
                ],
            ],
            dtype=torch.int64,
            device=device,
        ),
        torch.tensor(
            [
                [
                    29946,
                    29974,
                    29945,
                    29930,
                    29889,
                    29922,
                    29974,
                    29930,
                    29974,
                    29946,
                    29930,
                    29922,
                    29889,
                    29974,
                    29945,
                    29922,
                ],
                [
                    29941,
                    29906,
                    2,
                    29946,
                    29871,
                    450,
                    319,
                    14990,
                    29946,
                    29941,
                    2,
                    29906,
                    29871,
                    2,
                    3001,
                    13,
                ],
            ],
            dtype=torch.int64,
            device=device,
        ),
    ]
    parents_list = [
        torch.tensor(
            [[-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]], dtype=torch.int64, device=device
        ),
        torch.tensor([[4, 8, 9, 10], [4, 5, 6, 7]], dtype=torch.int64, device=device),
        torch.tensor(
            [[20, 24, 21, 28], [24, 28, 20, 21]], dtype=torch.int64, device=device
        ),
        torch.tensor(
            [[36, 40, 41, 44], [36, 40, 44, 45]], dtype=torch.int64, device=device
        ),
    ]
    seq_lens = torch.tensor([5, 10], dtype=torch.int64, device=device)
    topk, depth, num_draft_token = 4, 4, 8

    parent_list, top_scores_index, draft_tokens = _organize_draft_results(
        score_list, token_list, parents_list, num_draft_token
    )
    draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()

    (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
    ) = _run_kernel(
        parent_list,
        top_scores_index,
        seq_lens,
        topk,
        depth,
        num_draft_token,
        TreeMaskMode.FULL_MASK,
    )

    assert positions.tolist() == [
        5,
        6,
        6,
        7,
        7,
        8,
        8,
        9,
        10,
        11,
        12,
        12,
        12,
        12,
        13,
        14,
    ]
    assert retrieve_index.tolist() == [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
    ]
    assert retrieve_next_token.tolist() == [
        [1, 3, 4, 5, 6, 7, -1, -1],
        [1, 2, -1, 6, -1, -1, 7, -1],
    ]
    assert retrieve_next_sibling.tolist() == [
        [-1, 2, -1, -1, -1, -1, -1, -1],
        [-1, -1, 3, 4, 5, -1, -1, -1],
    ]
    assert draft_tokens.tolist() == [
        29974,
        29896,
        29906,
        29889,
        29974,
        29946,
        29896,
        29946,
        13,
        13,
        22550,
        4136,
        16492,
        8439,
        29871,
        29941,
    ]

    # Cross-check the whole trace against the python reference too.
    _check_against_reference(
        parent_list.cpu(),
        top_scores_index.cpu(),
        seq_lens.cpu(),
        topk,
        depth,
        num_draft_token,
    )


def test_bitpacking_mode_rejected():
    parent_list, selected_index = _topk1_chain_inputs(1, 3)
    with pytest.raises(RuntimeError, match="QLEN_ONLY_BITPACKING"):
        _ = _run_kernel(
            parent_list,
            selected_index,
            torch.tensor([5], dtype=torch.int64),
            1,
            3,
            4,
            TreeMaskMode.QLEN_ONLY_BITPACKING,
            tree_mask=torch.zeros(64, dtype=torch.bool, device=device),
        )


if __name__ == "__main__":
    pytest.main([__file__])
