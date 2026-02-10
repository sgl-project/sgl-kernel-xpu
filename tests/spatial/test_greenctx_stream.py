import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import create_greenctx_stream_by_value, get_sm_available
import utils

device = utils.get_device()

def test_green_ctx():
    A = torch.randn(5120, 5120).to(device)
    B = torch.randn(5120, 5120).to(device)
    C = torch.matmul(A, B)
    sm_counts = get_sm_available(0)
    stream_group = create_greenctx_stream_by_value(sm_counts // 2, sm_counts // 2, 0)
    with torch.Stream(stream_group[0]):
        for _ in range(100):
            result_0 = torch.matmul(A, B)
    with torch.Stream(stream_group[1]):
        for _ in range(100):
            result_1 = torch.matmul(A, B)
    torch.accelerator.synchronize()
    assert torch.allclose(result_0, C)
    assert torch.allclose(result_1, C)


if __name__ == "__main__":
    pytest.main([__file__])