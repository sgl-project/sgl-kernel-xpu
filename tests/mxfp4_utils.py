"""Shared MXFP4 quantisation/dequantisation helpers for tests.

Thin wrappers around the reference implementations in
test_per_token_group_quant_mxfp4.py, importable without sys.path hacks.
"""

from test_per_token_group_quant_mxfp4 import MXFP4_BLOCK_SIZE  # noqa: F401
from test_per_token_group_quant_mxfp4 import dequantize_mxfp4 as dequantize_mxfp4_2d  # noqa: F401
from test_per_token_group_quant_mxfp4 import quantize_to_mxfp4 as quantize_mxfp4_2d  # noqa: F401
