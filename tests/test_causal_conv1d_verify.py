# Correctness tests for the GDN target-verify (speculative decoding) conv1d
# op, which mirrors SGLang's Triton tree-mode `causal_conv1d_update`
# contract (_causal_conv1d_update_kernel w/ HAS_EAGLE_TREE_CUSTOM_ATTN_MASK).
import pytest
import torch
import utils
from sgl_kernel import causal_conv1d_update_verify_xpu

device = utils.get_device()

PAD_SLOT_ID = -1


def ref_causal_conv1d_update_verify(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: bool,
    cache_indices: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
):
    """Sequential reference mirroring the Triton tree-mode kernel exactly:
    decode next_token/next_sibling into per-token parent indices (token t's
    children/sibling are only resolved once t itself has been visited, so
    this scan must run in increasing token order), then build each token's
    conv window by walking its ancestor chain, falling back to conv_state
    once the chain is exhausted.
    """
    batch, dim, cache_steps = x.shape
    width = weight.shape[-1]
    out = torch.zeros_like(x)
    parent_token = torch.zeros(batch, cache_steps, dtype=torch.int32)
    window = torch.zeros(batch, cache_steps, dim, width - 1, dtype=x.dtype)

    for b in range(batch):
        cache_idx = cache_indices[b].item()
        if cache_idx == PAD_SLOT_ID:
            continue
        parent = [0] * cache_steps
        for t in range(cache_steps):
            child = retrieve_next_token[b, t].item()
            if child != -1:
                parent[child] = t
            sibling = retrieve_next_sibling[b, t].item()
            if sibling != -1:
                parent[sibling] = parent[t]
        parent_token[b] = torch.tensor(parent, dtype=torch.int32)

        for t in range(cache_steps):
            acc = bias.clone().float() if bias is not None else torch.zeros(dim)
            cur = t
            hist = []
            which_col = 0
            cur_val = x[b, :, cur].float()
            for j in range(width):
                hist.append(cur_val)
                acc += cur_val * weight[:, width - 1 - j].float()
                if cur > 0:
                    cur = parent[cur]
                    cur_val = x[b, :, cur].float()
                else:
                    col_idx = max(0, (width - 2) + which_col)
                    cur_val = conv_state[cache_idx, :, col_idx].float()
                    which_col -= 1
            for j in range(width - 1):
                window[b, t, :, width - 2 - j] = hist[j].to(x.dtype)
            if activation:
                acc = acc / (1.0 + torch.exp(-acc))
            out[b, :, t] = acc.to(x.dtype)
    return out, window, parent_token


def build_chain_tree(batch, cache_steps):
    # Pure linear chain: token t's only child is t+1, no siblings.
    next_token = torch.full((batch, cache_steps), -1, dtype=torch.int32)
    next_sibling = torch.full((batch, cache_steps), -1, dtype=torch.int32)
    for t in range(cache_steps - 1):
        next_token[:, t] = t + 1
    return next_token, next_sibling


def build_binary_tree(batch, cache_steps):
    # Complete binary tree over token indices (token t's children are
    # 2t+1, 2t+2 when in range); first-child/next-sibling encoding.
    next_token = torch.full((batch, cache_steps), -1, dtype=torch.int32)
    next_sibling = torch.full((batch, cache_steps), -1, dtype=torch.int32)
    for t in range(cache_steps):
        c0, c1 = 2 * t + 1, 2 * t + 2
        if c0 < cache_steps:
            next_token[:, t] = c0
        if c1 < cache_steps:
            next_sibling[:, c0] = c1
    return next_token, next_sibling


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("dim", [64, 160])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("cache_steps", [1, 4, 13])
@pytest.mark.parametrize("tree", ["chain", "binary"])
@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("input_layout", ["contiguous", "transposed"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_causal_conv1d_update_verify(
    width, dim, batch, cache_steps, tree, itype, input_layout, index_dtype
):
    torch.manual_seed(0)
    num_cache_lines = batch + 2  # exercise non-identity cache_indices too

    x = (
        torch.randn(batch, cache_steps, dim, dtype=itype, device=device).transpose(1, 2)
        if input_layout == "transposed"
        else torch.randn(batch, dim, cache_steps, dtype=itype, device=device)
    )
    conv_state = torch.randn(
        num_cache_lines, dim, width - 1, dtype=itype, device=device
    )
    weight = torch.randn(dim, width, dtype=itype, device=device)
    bias = torch.randn(dim, dtype=itype, device=device)
    cache_indices = torch.randperm(num_cache_lines, dtype=torch.int32)[:batch].to(
        device
    )
    intermediate_state_indices = torch.arange(batch, dtype=torch.int32, device=device)
    scratch_rows = batch
    intermediate_conv_window = torch.zeros(
        scratch_rows, cache_steps, dim, width - 1, dtype=itype, device=device
    )
    if tree == "chain":
        next_token, next_sibling = build_chain_tree(batch, cache_steps)
    else:
        next_token, next_sibling = build_binary_tree(batch, cache_steps)
    next_token = next_token.to(device=device, dtype=index_dtype)
    next_sibling = next_sibling.to(device=device, dtype=index_dtype)
    retrieve_parent_token = torch.empty_like(next_token)

    x_ref, window_ref, parent_ref = ref_causal_conv1d_update_verify(
        x.cpu(),
        conv_state.cpu(),
        weight.cpu(),
        bias.cpu(),
        activation=True,
        cache_indices=cache_indices.cpu(),
        retrieve_next_token=next_token.cpu(),
        retrieve_next_sibling=next_sibling.cpu(),
    )

    x_before = x.clone()
    x_out = causal_conv1d_update_verify_xpu(
        x,
        conv_state,
        weight,
        intermediate_conv_window,
        intermediate_state_indices,
        next_token,
        next_sibling,
        retrieve_parent_token,
        bias=bias,
        activation="silu",
        cache_indices=cache_indices,
    )

    assert x_out.data_ptr() != x.data_ptr()
    assert x_out.stride() == x.stride()
    torch.testing.assert_close(x, x_before)
    tol = 2e-2 if itype != torch.float32 else 1e-3
    torch.testing.assert_close(x_out.float().cpu(), x_ref.float(), atol=tol, rtol=tol)
    torch.testing.assert_close(
        intermediate_conv_window.float().cpu(), window_ref.float(), atol=tol, rtol=tol
    )
    torch.testing.assert_close(retrieve_parent_token.cpu(), parent_ref.to(index_dtype))


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("input_layout", ["contiguous", "transposed"])
def test_causal_conv1d_update_verify_pad_slot(input_layout, index_dtype):
    width, dim, cache_steps = 4, 32, 5
    batch = 3
    num_cache_lines = batch
    itype = torch.float32

    x = (
        torch.randn(batch, cache_steps, dim, dtype=itype, device=device).transpose(1, 2)
        if input_layout == "transposed"
        else torch.randn(batch, dim, cache_steps, dtype=itype, device=device)
    )
    x_before = x.clone()
    conv_state = torch.randn(
        num_cache_lines, dim, width - 1, dtype=itype, device=device
    )
    weight = torch.randn(dim, width, dtype=itype, device=device)
    bias = torch.randn(dim, dtype=itype, device=device)
    cache_indices = torch.tensor([0, PAD_SLOT_ID, 2], dtype=torch.int32, device=device)
    intermediate_state_indices = torch.arange(batch, dtype=torch.int32, device=device)
    intermediate_conv_window = torch.zeros(
        batch, cache_steps, dim, width - 1, dtype=itype, device=device
    )
    next_token, next_sibling = build_chain_tree(batch, cache_steps)
    next_token = next_token.to(device=device, dtype=index_dtype)
    next_sibling = next_sibling.to(device=device, dtype=index_dtype)
    retrieve_parent_token = torch.zeros_like(next_token)

    x_out = causal_conv1d_update_verify_xpu(
        x,
        conv_state,
        weight,
        intermediate_conv_window,
        intermediate_state_indices,
        next_token,
        next_sibling,
        retrieve_parent_token,
        bias=bias,
        activation="silu",
        cache_indices=cache_indices,
    )
    assert x_out.data_ptr() != x.data_ptr()
    assert x_out.stride() == x.stride()
    torch.testing.assert_close(x, x_before)
    # Padded entries retain their input values in the new output.
    torch.testing.assert_close(x_out[1].cpu(), x_before[1].cpu())


@pytest.mark.parametrize(
    "sibling_dtype,parent_dtype",
    [
        (torch.int64, torch.int32),
        (torch.int32, torch.int64),
        (torch.float32, torch.float32),
    ],
)
def test_causal_conv1d_update_verify_rejects_invalid_tree_dtypes(
    sibling_dtype, parent_dtype
):
    x = torch.zeros(1, 32, 3, device=device)
    conv_state = torch.zeros(1, 32, 2, device=device)
    weight = torch.zeros(32, 3, device=device)
    next_token = torch.tensor([[1, 2, -1]], dtype=torch.int32, device=device)
    next_sibling = torch.full((1, 3), -1, dtype=sibling_dtype, device=device)
    parent = torch.empty((1, 3), dtype=parent_dtype, device=device)
    window = torch.empty(1, 3, 32, 2, device=device)
    indices = torch.zeros(1, dtype=torch.int32, device=device)

    with pytest.raises(RuntimeError, match="int32 or int64|match retrieve_next_token"):
        causal_conv1d_update_verify_xpu(
            x, conv_state, weight, window, indices, next_token, next_sibling, parent
        )
