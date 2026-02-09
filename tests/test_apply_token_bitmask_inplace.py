import sys

import pytest
import torch
import utils
from sgl_kernel import apply_token_bitmask_inplace_cuda

device = utils.get_device()


def test_apply_token_bitmask_inplace_kernel():
    neginf = float("-inf")
    bool_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
    logits = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32
    )
    expected = torch.where(bool_mask, logits, neginf)

    logits_gpu = logits.to(device)
    bitmask = torch.tensor([0b1010101010], dtype=torch.int32).to(device)
    apply_token_bitmask_inplace_cuda(logits_gpu, bitmask)
    torch.accelerator.synchronize()
    torch.testing.assert_close(logits_gpu, expected.to(device))


if __name__ == "__main__":
    test_apply_token_bitmask_inplace_kernel()
    sys.exit(pytest.main([__file__]))
