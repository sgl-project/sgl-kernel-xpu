"""KV-cache-family JIT kernels for Intel XPU.

Groups SYCL JIT wrappers whose CUDA-side counterparts live under
``sglang.kernels.ops.kvcache.*``. Mirrors that layout so consumers can write
the same import shape across backends:

    from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla  # CUDA
    from sgl_kernel.jit.kvcache.hisparse    import load_cache_to_device_buffer_mla  # XPU

Modules exported here:

- ``hisparse``: DSA/DSv4 hisparse swap-in + evict/backup kernels
  (``load_cache_to_device_buffer_{mla,dsv4_mla}``, ``transfer_cache_dsv4_mla``).
"""

from .hisparse import (
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
    transfer_cache_dsv4_mla,
)

__all__ = [
    "load_cache_to_device_buffer_dsv4_mla",
    "load_cache_to_device_buffer_mla",
    "transfer_cache_dsv4_mla",
]
