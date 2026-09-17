import pytest
import torch
import torch.nn.functional as F
import utils
from sgl_kernel import reconstruct_indices_from_tree_mask

device = utils.get_device()


def test_reconstruct_indices_from_tree_mask():
    bs = 1
    num_branch_token = 4
    seq_lens = torch.tensor([12], device=device, dtype=torch.int64)

    retrive_index = torch.full(
        (bs, num_branch_token), -1, device=device, dtype=torch.int64
    )
    retrive_next_token = torch.full(
        (bs, num_branch_token), -1, device=device, dtype=torch.int64
    )
    retrive_next_sibling = torch.full(
        (bs, num_branch_token), -1, device=device, dtype=torch.int64
    )
    positions = torch.empty((bs * num_branch_token), device=device, dtype=torch.int64)

    tree_mask = torch.tensor(
        [
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
        ],
        device=device,
        dtype=torch.int32,
    ).to(torch.bool)

    reconstruct_indices_from_tree_mask(
        tree_mask,
        seq_lens,
        positions,  # mutable
        retrive_index,  # mutable
        retrive_next_token,  # mutable
        retrive_next_sibling,  # mutable
        bs,
        num_branch_token,
    )
    # print(f"debug: \n\n{tree_mask=}, {retrive_index=}, {retrive_next_token=}, {retrive_next_sibling=}, {positions=}\n\n")
    assert retrive_index.tolist() == [
        [0, 1, 2, 3],
    ], f"{retrive_index=}"
    assert retrive_next_token.tolist() == [
        [1, -1, 3, -1],
    ], f"{retrive_next_token=}"
    assert retrive_next_sibling.tolist() == [
        [-1, 2, -1, -1],
    ], f"{retrive_next_sibling=}"
    assert positions.tolist() == [
        12,
        13,
        13,
        14,
    ], f"{positions=}"


def _make_valid_tree_mask(bs: int, n: int, seed: int):
    """Random *valid* tree mask [bs, n, n]: mask[b, i, j] == node j is an ancestor
    of node i (transitive closure, diagonal set). Each node either roots (~30%) or
    attaches under a uniformly random earlier node, so multi-root batches -- the
    case where roots must not link to each other as siblings -- are covered."""
    import numpy as np

    rng = np.random.default_rng(seed)
    mask = np.zeros((bs, n, n), dtype=bool)
    for b in range(bs):
        ancestors = [set() for _ in range(n)]
        for i in range(n):
            ancestors[i].add(i)
            if i > 0 and rng.random() >= 0.3:
                parent = int(rng.integers(0, i))
                ancestors[i] |= ancestors[parent]
            for j in ancestors[i]:
                mask[b, i, j] = True
    return mask


def _reference(mask, seq_lens, bs: int, n: int):
    """Plain-python oracle for the documented contract -- no vectorization shared
    with the kernel."""
    import numpy as np

    positions = np.empty(bs * n, dtype=np.int64)
    retrive_index = np.empty(bs * n, dtype=np.int64)
    next_token = np.full(bs * n, -1, dtype=np.int64)
    next_sibling = np.full(bs * n, -1, dtype=np.int64)

    for b in range(bs):
        parent = [-1] * n
        for tid in range(n):
            ancestors = [j for j in range(tid) if mask[b, tid, j]]
            positions[b * n + tid] = len(ancestors) + int(seq_lens[b])
            retrive_index[b * n + tid] = b * n + tid
            parent[tid] = max(ancestors) if ancestors else -1

        for tid in range(n):
            children = [k for k in range(tid + 1, n) if mask[b, k, tid]]
            if children:
                next_token[b * n + tid] = min(children)
            if parent[tid] >= 0:
                siblings = [k for k in range(tid + 1, n) if parent[k] == parent[tid]]
                if siblings:
                    next_sibling[b * n + tid] = min(siblings)

    return positions, retrive_index, next_token, next_sibling


# n sweeps power-of-two and non-power-of-two widths, straddles the 64-bit word
# boundary of the packed mask rows (63/64/65), and bs sweeps below, at, and above
# the point where the launch starts packing several requests per work-group.
@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 16, 17, 32, 63, 64, 65, 130])
@pytest.mark.parametrize("bs", [1, 3, 20, 64, 256])
def test_reconstruct_indices_matches_reference(n, bs):
    import numpy as np

    seed = bs * 1000 + n
    mask = _make_valid_tree_mask(bs, n, seed)
    seq_lens_cpu = torch.from_numpy(
        np.random.default_rng(seed).integers(1, 200, size=bs)
    ).to(torch.int64)

    tree_mask = torch.from_numpy(mask).reshape(-1).contiguous().to(device)
    seq_lens = seq_lens_cpu.to(device)
    positions = torch.empty(bs * n, dtype=torch.int64, device=device)
    retrive_index = torch.full((bs, n), -1, dtype=torch.int64, device=device)
    next_token = torch.full((bs, n), -1, dtype=torch.int64, device=device)
    next_sibling = torch.full((bs, n), -1, dtype=torch.int64, device=device)

    reconstruct_indices_from_tree_mask(
        tree_mask,
        seq_lens,
        positions,
        retrive_index,
        next_token,
        next_sibling,
        bs,
        n,
    )
    torch.xpu.synchronize()

    got = (
        positions.cpu().numpy(),
        retrive_index.reshape(-1).cpu().numpy(),
        next_token.reshape(-1).cpu().numpy(),
        next_sibling.reshape(-1).cpu().numpy(),
    )
    ref = _reference(mask, seq_lens_cpu.numpy(), bs, n)
    names = ("positions", "retrive_index", "retrive_next_token", "retrive_next_sibling")
    for name, g, r in zip(names, got, ref):
        np.testing.assert_array_equal(g, r, err_msg=f"{name} mismatch bs={bs} n={n}")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_reconstruct_indices_seq_len_dtypes(dtype):
    """verified_seq_len is dispatched over index types; both must work."""
    bs, n = 4, 8
    mask = _make_valid_tree_mask(bs, n, seed=7)
    seq_lens_cpu = torch.arange(1, bs + 1, dtype=dtype) * 13

    tree_mask = torch.from_numpy(mask).reshape(-1).contiguous().to(device)
    positions = torch.empty(bs * n, dtype=torch.int64, device=device)
    retrive_index = torch.full((bs, n), -1, dtype=torch.int64, device=device)
    next_token = torch.full((bs, n), -1, dtype=torch.int64, device=device)
    next_sibling = torch.full((bs, n), -1, dtype=torch.int64, device=device)

    reconstruct_indices_from_tree_mask(
        tree_mask,
        seq_lens_cpu.to(device),
        positions,
        retrive_index,
        next_token,
        next_sibling,
        bs,
        n,
    )
    torch.xpu.synchronize()

    import numpy as np

    ref = _reference(mask, seq_lens_cpu.numpy(), bs, n)
    np.testing.assert_array_equal(positions.cpu().numpy(), ref[0])
    np.testing.assert_array_equal(next_token.reshape(-1).cpu().numpy(), ref[2])


if __name__ == "__main__":
    test_reconstruct_indices_from_tree_mask()
    pytest.main([__file__])
