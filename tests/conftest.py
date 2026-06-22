import pytest
import torch


# This fixture ensures the torch defaults don't get left in modified states between
# tests (e.g., when a test fails before restoring the original value), which
# can cause subsequent tests to fail.
@pytest.fixture(autouse=True)
def reset_torch_defaults():
    orig_default_device = torch.get_default_device()
    orig_default_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(orig_default_dtype)
    torch.set_default_device(orig_default_device)


# This fixture ensures XPU memory is released after every test, even when the
# test raises an exception before its own torch.xpu.empty_cache() call.
# Without this, an OOM in one test leaves the cache full and causes all
# subsequent tests to fail with UR_RESULT_ERROR_OUT_OF_RESOURCES.
@pytest.fixture(autouse=True)
def clear_xpu_cache():
    yield
    if torch.xpu.is_available():
        torch.xpu.empty_cache()
