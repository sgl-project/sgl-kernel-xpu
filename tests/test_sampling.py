# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/93e1a2634e22355b0856246b032b285ad1d1da6b/tests/test_sampling.py

import sys

import pytest
import sgl_kernel
import torch
import utils

device = utils.get_device()


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256, 151936])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_joint_sampling_from_probs(batch_size, vocab_size, p):
    torch.manual_seed(42)
    if p == 0.1:
        k = int(vocab_size * 0.5)
    elif p == 0.5:
        k = int(vocab_size * 0.1)
    else:
        raise ValueError("p not recognized")
    eps = 1e-4
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=f"{device}:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    # top-p mask
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask_top_p = torch.zeros(
        batch_size, vocab_size, dtype=torch.int32, device=f"{device}:0"
    )
    mask_top_p.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
    # top-k mask
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask_top_k = (normalized_prob >= pivot.unsqueeze(-1)).int()
    # overall mask
    mask = torch.minimum(mask_top_p, mask_top_k)
    top_p_tensor = torch.full((batch_size,), p, device=f"{device}:0")
    top_k_tensor = torch.full((batch_size,), k, device=f"{device}:0")

    num_trails = 1000
    for _ in range(num_trails):
        samples = sgl_kernel.top_k_top_p_sampling_from_probs(
            normalized_prob,
            top_k_tensor,
            top_p_tensor,
            filter_apply_order="joint",
        )
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


def torch_top_p_renorm_probs(normalized_prob, p):
    """Compute ground truth for top-p renormalization.

    Args:
        normalized_prob: [batch_size, vocab_size] probabilities
        p: float or [batch_size] tensor of per-row p values
    """
    batch_size, vocab_size = normalized_prob.size()
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    if isinstance(p, torch.Tensor):
        # Per-row p array: [batch_size] -> [batch_size, 1] for broadcasting
        threshold = (1 - p).unsqueeze(-1)
    else:
        threshold = 1 - p
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device=f"{device}:0")
    mask.scatter_add_(1, indices, (cdf >= threshold).int())
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )
    return renorm_prob_ground_truth

@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256, 151936])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_renorm_probs(batch_size, vocab_size, p):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=f"{device}:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    renorm_prob_ground_truth = torch_top_p_renorm_probs(normalized_prob, p)

    renorm_prob = sgl_kernel.top_p_renorm_prob(normalized_prob, p)
    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )



@pytest.mark.parametrize("batch_size", [1, 16, 128])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p_range", [(0.1, 0.5), (0.5, 0.9)])
def test_top_p_renorm_probs_tensor(batch_size, vocab_size, p_range):
    p_min, p_max = p_range
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=f"{device}:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    # Create per-row top-p array with varied values
    top_p_arr = (
        torch.rand(batch_size, device=f"{device}:0") * (p_max - p_min) + p_min
    )

    # Compute ground truth using unified function
    renorm_prob_ground_truth = torch_top_p_renorm_probs(normalized_prob, top_p_arr)

    # Test with per-row p array
    renorm_prob = sgl_kernel.top_p_renorm_prob(normalized_prob, top_p_arr)

    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )


def torch_top_k_renorm_probs(normalized_prob, k):
    """Compute ground truth for top-k renormalization.

    Args:
        normalized_prob: [batch_size, vocab_size] probabilities
        k: int or [batch_size] tensor of per-row k values
    """
    if isinstance(k, torch.Tensor):
        # Per-row k array
        batch_size = normalized_prob.size(0)
        k_cpu = k.to("cpu").tolist()
        renorm_prob_ground_truth = torch.zeros_like(normalized_prob)
        for i in range(batch_size):
            k_i = k_cpu[i]
            sorted_prob, _ = torch.sort(normalized_prob[i : i + 1], descending=True)
            pivot = sorted_prob[:, k_i - 1].unsqueeze(-1)
            mask = (normalized_prob[i : i + 1] >= pivot).int()
            row_result = normalized_prob[i : i + 1].clone()
            row_result[mask == 0] = 0
            row_result = row_result / row_result.sum(dim=-1, keepdim=True)
            renorm_prob_ground_truth[i] = row_result[0]
        return renorm_prob_ground_truth
    else:
        # Scalar k
        sorted_prob, indices = torch.sort(normalized_prob, descending=True)
        pivot = sorted_prob[:, k - 1].unsqueeze(-1)
        mask = (normalized_prob >= pivot).int()
        renorm_prob_ground_truth = normalized_prob.clone()
        renorm_prob_ground_truth[mask == 0] = 0
        renorm_prob_ground_truth = (
            renorm_prob_ground_truth
            / renorm_prob_ground_truth.sum(dim=-1, keepdim=True)
        )
        return renorm_prob_ground_truth


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_renorm_probs(batch_size, vocab_size, k):
    # Note: SYCL kernel clamps k > vocab_size to vocab_size
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=f"{device}:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    renorm_prob_ground_truth = torch_top_k_renorm_probs(normalized_prob, k)

    renorm_prob = sgl_kernel.top_k_renorm_prob(normalized_prob, k)

    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("batch_size", [1, 16, 128])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k_range", [(10, 50), (50, 200)])
def test_top_k_renorm_probs_array(batch_size, vocab_size, k_range):
    # Note: SYCL kernel clamps k > vocab_size to vocab_size
    k_min, k_max = k_range
    if k_max > vocab_size:
        pytest.skip("k_max should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=f"{device}:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    # Create per-row top-k array with varied values
    top_k_arr = torch.randint(
        k_min, k_max, (batch_size,), dtype=torch.int64, device=f"{device}:0"
    )
    if torch.any(top_k_arr <= 0):
        pytest.skip("top_k_arr values should be greater than 0")

    # Compute ground truth using unified function
    renorm_prob_ground_truth = torch_top_k_renorm_probs(normalized_prob, top_k_arr)

    # Test with per-row k array
    renorm_prob = sgl_kernel.top_k_renorm_prob(normalized_prob, top_k_arr)

    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skip(reason="not implemented")
@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.7, 1])
def test_min_p_sampling(batch_size, vocab_size, p):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device=f"{device}:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    # scale min-p
    top_probs = sorted_prob[:, -1].unsqueeze(-1)
    scaled_p = p * top_probs
    # min-p mask
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device=f"{device}:0")
    mask.scatter_add_(1, indices, (sorted_prob >= scaled_p).int())
    min_p_tensor = torch.full((batch_size,), p, device=f"{device}:0")

    num_trails = 1000
    for _ in range(num_trails):
        samples = sgl_kernel.min_p_sampling_from_probs(
            normalized_prob,
            min_p_tensor,
        )

        assert torch.all(mask[torch.arange(batch_size), samples] == 1), samples[
            torch.nonzero(mask[torch.arange(batch_size), samples] == 0)
        ]

        assert torch.all(mask[torch.arange(batch_size), samples] == 1), samples[
            torch.nonzero(mask[torch.arange(batch_size), samples] == 0)
        ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
