import itertools

import pytest
import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel.*
import torch
import utils

device = utils.get_device()

ne_n = 4
ne_k = 2
num_configs = (ne_n - 1) * ne_k
eos_token_id = 2
req_lens = [4, 3, 5]

g = torch.Generator().manual_seed(0)
ne_weights = torch.randint(1, 7, (ne_n - 1, ne_k, ne_n), generator=g, dtype=torch.int32)
ne_mods = torch.randint(50, 120, (ne_n - 1, ne_k), generator=g, dtype=torch.int32)
size_sums = torch.arange(num_configs, dtype=torch.int32) * 1000
row_indices = torch.tensor([0, 2, 3], dtype=torch.int64)
column_starts = torch.tensor([5, 7, 4], dtype=torch.int32)
token_table = torch.randint(1, 40, (5, 16), generator=g, dtype=torch.int32)
# Exercise both early-exit paths: an ignored (negative) token and an eos.
token_table[0, 3] = -1
token_table[2, 6] = eos_token_id


def to_dev(*tensors):
    return [t.to(device) for t in tensors]


def ref_n_gram_ids(req, pos):
    """Ids of every (n, k) config ending at column `pos` of request `req`."""
    row = token_table[row_indices[req]].tolist()
    ids = []
    for c in range(num_configs):
        n, k = divmod(c, ne_k)
        mod = ne_mods[n, k].item()
        ng = 0
        for j in range(n + 2):
            tok = row[pos - j] if pos - j >= 0 else -1
            if tok < 0 or (tok == eos_token_id and j > 0):
                break
            ng += tok * ne_weights[n, k, j].item() % mod
        ids.append(ng % mod + size_sums[c].item())
    return ids


@pytest.mark.parametrize("decode", [False, True])
def test_compute_n_gram_ids(decode):
    lens = [1] * len(req_lens) if decode else req_lens
    out = torch.zeros(sum(lens), num_configs, dtype=torch.int32, device=device)
    params = to_dev(ne_weights, ne_mods, size_sums)
    table = to_dev(token_table, row_indices, column_starts)
    if decode:
        torch.ops.sgl_kernel.compute_n_gram_ids_decode(
            ne_n, ne_k, *params, *table, out, eos_token_id
        )
    else:
        tokens = torch.zeros(sum(lens), dtype=torch.int32)  # unused by the kernel
        len_sums = torch.tensor([0, *itertools.accumulate(lens)], dtype=torch.int32)
        torch.ops.sgl_kernel.compute_n_gram_ids(
            ne_n, ne_k, *params, *to_dev(tokens, len_sums), *table, out, eos_token_id
        )
    expected = [
        ref_n_gram_ids(r, column_starts[r].item() + off)
        for r in range(len(lens))
        for off in range(lens[r])
    ]
    assert out.cpu().tolist() == expected


@pytest.mark.parametrize("decode,num_ignore", [(True, 0), (False, 0), (False, 1)])
def test_update_token_table(decode, num_ignore):
    lens = [1] * len(req_lens) if decode else req_lens
    tokens = torch.randint(1, 40, (sum(lens),), generator=g, dtype=torch.int32)
    ignore_tokens = tokens[1 : 1 + num_ignore].clone()
    table = token_table.to(device)
    if decode:
        torch.ops.sgl_kernel.update_token_table_decode(
            *to_dev(tokens), table, *to_dev(row_indices, column_starts)
        )
    else:
        lens_t = torch.tensor(lens, dtype=torch.int32)
        torch.ops.sgl_kernel.update_token_table(
            *to_dev(tokens),
            table,
            *to_dev(row_indices, column_starts, lens_t, ignore_tokens),
        )
    # Ignored tokens are stored negated so the n-gram lookup stops at them.
    expected = token_table.clone()
    tok = iter(tokens.tolist())
    for r in range(len(lens)):
        for off in range(lens[r]):
            t = next(tok)
            expected[row_indices[r], column_starts[r] + off] = (
                -t if t in ignore_tokens.tolist() else t
            )
    assert torch.equal(table.cpu(), expected)


if __name__ == "__main__":
    pytest.main([__file__])
