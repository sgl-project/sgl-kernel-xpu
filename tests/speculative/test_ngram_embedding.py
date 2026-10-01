import pytest
import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel.*
import torch
import utils

device = utils.get_device()

batch_size = 3
ne_n = 4
ne_k = 2
num_configs = (ne_n - 1) * ne_k
max_running_reqs = 5
max_context_len = 16
eos_token_id = 2

g = torch.Generator().manual_seed(0)
ne_weights = torch.randint(1, 7, (ne_n - 1, ne_k, ne_n), generator=g, dtype=torch.int32)
ne_mods = torch.randint(50, 120, (ne_n - 1, ne_k), generator=g, dtype=torch.int32)
exclusive_ne_embedder_size_sums = torch.arange(num_configs, dtype=torch.int32) * 1000
row_indices = torch.tensor([0, 2, 3], dtype=torch.int64)
column_starts = torch.tensor([5, 7, 4], dtype=torch.int32)
token_table = torch.randint(
    1, 40, (max_running_reqs, max_context_len), generator=g, dtype=torch.int32
)
# Exercise both early-exit paths: an ignored (negative) token and an eos.
token_table[0, 3] = -1
token_table[2, 6] = eos_token_id


def ref_n_gram_id(req, pos, n, k):
    """n-gram id ending at column `pos` of request `req`, as ngram_embedding.cuh."""
    weights = ne_weights[n, k].tolist()
    mod = ne_mods[n, k].item()
    row = token_table[row_indices[req]].tolist()
    ng = 0
    for j in range(n + 2):
        if pos - j < 0:
            break
        tok = row[pos - j]
        if tok < 0 or (tok == eos_token_id and j > 0):
            break
        ng += (tok * weights[j]) % mod
    return ng % mod + exclusive_ne_embedder_size_sums[n * ne_k + k].item()


def ref_n_gram_ids(req, pos):
    return [ref_n_gram_id(req, pos, c // ne_k, c % ne_k) for c in range(num_configs)]


def test_compute_n_gram_ids_decode():
    out = torch.zeros(batch_size, num_configs, dtype=torch.int32, device=device)
    torch.ops.sgl_kernel.compute_n_gram_ids_decode(
        ne_n,
        ne_k,
        ne_weights.to(device),
        ne_mods.to(device),
        exclusive_ne_embedder_size_sums.to(device),
        token_table.to(device),
        row_indices.to(device),
        column_starts.to(device),
        out,
        eos_token_id,
    )
    expected = [ref_n_gram_ids(r, column_starts[r].item()) for r in range(batch_size)]
    assert out.cpu().tolist() == expected


def test_compute_n_gram_ids():
    req_lens = [4, 3, 5]
    exclusive_req_len_sums = torch.tensor([0, 4, 7, 12], dtype=torch.int32)
    num_tokens = sum(req_lens)
    tokens = torch.randint(1, 40, (num_tokens,), generator=g, dtype=torch.int32)
    out = torch.zeros(num_tokens, num_configs, dtype=torch.int32, device=device)
    torch.ops.sgl_kernel.compute_n_gram_ids(
        ne_n,
        ne_k,
        ne_weights.to(device),
        ne_mods.to(device),
        exclusive_ne_embedder_size_sums.to(device),
        tokens.to(device),
        exclusive_req_len_sums.to(device),
        token_table.to(device),
        row_indices.to(device),
        column_starts.to(device),
        out,
        eos_token_id,
    )
    expected = [
        ref_n_gram_ids(r, column_starts[r].item() + off)
        for r in range(batch_size)
        for off in range(req_lens[r])
    ]
    assert out.cpu().tolist() == expected


def test_update_token_table_decode():
    table = token_table.to(device)
    tokens = torch.tensor([11, 22, 33], dtype=torch.int32)
    torch.ops.sgl_kernel.update_token_table_decode(
        tokens.to(device), table, row_indices.to(device), column_starts.to(device)
    )
    expected = token_table.clone()
    expected[row_indices, column_starts.long()] = tokens
    assert torch.equal(table.cpu(), expected)


def test_update_token_table():
    req_lens = torch.tensor([4, 3, 5], dtype=torch.int32)
    tokens = torch.randint(
        1, 40, (int(req_lens.sum()),), generator=g, dtype=torch.int32
    )
    ignore_tokens = tokens[1:2].clone()
    table = token_table.to(device)
    torch.ops.sgl_kernel.update_token_table(
        tokens.to(device),
        table,
        row_indices.to(device),
        column_starts.to(device),
        req_lens.to(device),
        ignore_tokens.to(device),
    )
    # Ignored tokens are stored negated so the n-gram lookup stops at them.
    expected = token_table.clone()
    i = 0
    for r in range(batch_size):
        for off in range(req_lens[r]):
            tok = tokens[i].item()
            expected[row_indices[r], column_starts[r] + off] = (
                -tok if tok == ignore_tokens.item() else tok
            )
            i += 1
    assert torch.equal(table.cpu(), expected)


if __name__ == "__main__":
    pytest.main([__file__])
