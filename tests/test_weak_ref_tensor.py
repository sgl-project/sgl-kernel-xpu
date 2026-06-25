import pytest
import torch
import utils

device = utils.get_device()


@pytest.mark.skipif(device.type != "xpu", reason="XPU not available")
def test_weak_ref_tensor_aliasing():
    """Test that weak_ref_tensor output shares storage with the input."""
    from sgl_kernel.memory import weak_ref_tensor

    x = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
    y = weak_ref_tensor(x)

    # Same data pointer: output aliases input storage
    assert (
        y.data_ptr() == x.data_ptr()
    ), "weak_ref_tensor output should share storage with input"

    # Sizes and strides are preserved
    assert list(y.shape) == list(
        x.shape
    ), "weak_ref_tensor output shape must match input"
    assert list(y.stride()) == list(
        x.stride()
    ), "weak_ref_tensor output strides must match input"

    # Non-XPU tensors are returned as-is (no op)
    cpu_x = torch.randn(4, 8)
    cpu_y = weak_ref_tensor(cpu_x)
    assert cpu_y is cpu_x, "weak_ref_tensor should return CPU tensor unchanged"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
