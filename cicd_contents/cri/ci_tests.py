CI_SGL_KERNEL_XPU_TESTS = [
    "tests/test_activation.py::test_fused_silu_mul[1-1-128]",
    "tests/test_mxfp4_blockwise_moe.py::TestMXFP4BlockwiseScaledGroupedMM::test_kernel_vs_reference[2-64-64-64]",
    "tests/test_mxfp4_blockwise_moe.py::TestMXFP4BlockwiseScaledGroupedMM::test_sanity_check_small_values",
    "tests/test_mxfp8_blockwise_moe.py::TestMXFP8BlockwiseScaledGroupedMM::test_kernel_vs_reference[2-128-128-128]",
    "tests/test_mxfp8_blockwise_moe.py::TestMXFP8BlockwiseScaledGroupedMM::test_single_expert",
]
