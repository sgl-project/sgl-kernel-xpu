import pytest
import torch
from _xpu import KNOWN_ARCHS, current_arch


# Arch gate (arch-aware tests). An op author declares which
# arches it supports with one marker:
#
#     @pytest.mark.arch("xe20", "xe35")     # skipped on any other device
#     def test_per_tensor_quant_fp8(...):
#         ...
#
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "arch(*names): XPU arches this op supports (xe20, xe35, xe40). The test "
        "is skipped unless the live device is one of them. No mark = "
        "arch-agnostic (runs on any device).",
    )


def pytest_runtest_setup(item):
    marker = item.get_closest_marker("arch")
    if marker is None:
        return  # arch-agnostic — runs everywhere

    supported = set(marker.args)
    unknown = supported - KNOWN_ARCHS
    if unknown:
        raise pytest.UsageError(
            f"{item.nodeid}: unknown arch(s) {sorted(unknown)} in "
            f"@pytest.mark.arch; valid tags are {sorted(KNOWN_ARCHS)}."
        )

    running = current_arch()
    if running not in supported:
        pytest.skip(f"op supports {sorted(supported)}; running device is {running}")


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
