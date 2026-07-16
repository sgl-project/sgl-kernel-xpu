import pytest
import torch

# Disable tensor content sampling in reprs. pytest's saferepr calls
# `torch/_tensor_str.py:get_summarized_data` when formatting a failed assertion,
# which reads the tensor's data — for an XPU tensor whose backing buffer has
# been left in a bad state by a kernel failure, that read SIGSEGVs the process
# and hides the real assertion. threshold=0 makes torch print the tensor's
# metadata only, keeping the traceback intact.
torch.set_printoptions(threshold=0)


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
    if torch.xpu.is_available():
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    yield
    if torch.xpu.is_available():
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
