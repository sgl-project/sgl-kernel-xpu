import pytest
import torch


def _ref_build_tree(parent_list, selected_index, seq_lens, topk, draft_token_num):
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


def test_build_tree_topk2_hand_case():
    # topk=2, steps=2, draft_token_num=4. Nodes: 1 and 2 are children of the
    # root; node 3 is a child of node 1.
    topk, num_steps, draft_token_num = 2, 2, 4
    parent_list = torch.tensor([[-1, 0, 1]], dtype=torch.int64)
    selected_index = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    seq_lens = torch.tensor([5], dtype=torch.int64)
    x = _ref_build_tree(parent_list, selected_index, seq_lens, topk, draft_token_num)
