"""
Utility functions for JIT kernel compilation.

Provides caching utilities used by both CUDA and SYCL JIT paths.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


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


__all__ = [
    "cache_once",
]
