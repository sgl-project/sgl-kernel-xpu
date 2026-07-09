"""
Utility functions for JIT kernel compilation.

Provides type conversion and caching utilities used by both CUDA and SYCL JIT paths.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeAlias, TypeVar, Union

import torch

F = TypeVar("F", bound=Callable[..., Any])
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, str, bool, torch.dtype]


class CPPArgList(list[str]):
    """List of C++ template arguments that can be converted to a string."""

    def __str__(self) -> str:
        return ", ".join(self)


CPP_DTYPE_MAP = {
    torch.float: "fp32_t",
    torch.float16: "fp16_t",
    torch.float8_e4m3fn: "fp8_e4m3_t",
    torch.bfloat16: "bf16_t",
    torch.int8: "int8_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
}


def cache_once(fn: F) -> F:
    """
    Simple cache decorator compatible with torch.compile.

    NOTE: `functools.lru_cache` is not compatible with `torch.compile`,
    so we manually implement a simple cache_once decorator to replace it.
    """
    result_map = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]

    return wrapper  # type: ignore


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CPPArgList:
    """Convert Python values to C++ template argument strings.

    Args:
        *args: Values to convert (int, float, str, bool, or torch.dtype)

    Returns:
        CPPArgList that can be used in C++ template instantiation

    Example:
        >>> make_cpp_args(128, True, torch.float16)
        CPPArgList(['128', 'true', 'fp16_t'])
    """

    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, str, float)):
            return str(arg)
        if isinstance(arg, torch.dtype):
            return CPP_DTYPE_MAP[arg]
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CPPArgList(_convert(arg) for arg in args)


__all__ = [
    "CPPArgList",
    "cache_once",
    "make_cpp_args",
]
