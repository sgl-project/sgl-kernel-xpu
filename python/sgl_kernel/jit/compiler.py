"""
XPU/SYCL JIT kernel compilation utilities for Intel GPUs.

This module provides JIT compilation support for SYCL kernels using the icpx compiler.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import pathlib
import re
import subprocess
import tempfile
from collections import OrderedDict
from typing import Any, List, Tuple

import torch

from .utils import cache_once

logger = logging.getLogger(__name__)

# Default SYCL compilation flags matching sgl-kernel-xpu build
DEFAULT_SYCL_CFLAGS = [
    "-fsycl",
    "-sycl-std=2020",
    "-std=c++20",
    "-O3",
    "-fPIC",
    "-shared",
    "-ftemplate-backtrace-limit=0",
    "-fno-sycl-unnamed-lambda",
    "-fhonor-nans",
    "-fhonor-infinities",
    "-fno-associative-math",
    "-fno-approx-func",
    "-no-ftz",
    "-fno-sycl-instrument-device-code",
]


def _get_sycl_aot_flags() -> List[str]:
    """Detect GPU device and return AOT compilation flags.

    The AOT targets can be overridden via the ``SGLANG_SYCL_AOT_TARGETS``
    environment variable (comparable to CUDA's ``TVM_FFI_CUDA_ARCH_LIST``).

    Otherwise uses the numeric device capability reported by the sgl_kernel
    runtime (equivalent to CUDA's (major, minor)).

    Falls back to generic spir64 if the device can't be detected.
    """
    override = os.environ.get("SGLANG_SYCL_AOT_TARGETS")
    if override:
        return [f"-fsycl-targets={override}"]

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.xpu.current_device()
            major, _ = torch.ops.sgl_kernel.query_device.default(device)
            # Xe2 (BMG-G21, Arc B-series/Battlemage) has compute capability 20
            if major == 2:
                return [
                    "-fsycl-targets=intel_gpu_bmg_g21",
                ]
    except Exception as e:
        logger.warning("Failed to detect XPU device capability for AOT flags: %s", e)
    # Fallback: generic SPIR-V (runtime JIT to native ISA)
    logger.warning(
        "Falling back to generic SPIR-V AOT target (-fsycl-targets=spir64); "
        "AOT-optimized kernels are unavailable, which may cause a performance "
        "regression. Set SGLANG_SYCL_AOT_TARGETS to override the target explicitly."
    )
    return ["-fsycl-targets=spir64"]


def _resolve_kernel_include_path() -> pathlib.Path:
    """
    Resolve the path to sgl-kernel's include directory.

    Returns the include directory containing sgl_kernel/jit_kernel headers.
    """
    # We're in sgl_kernel/jit/compiler.py, so go up to sgl_kernel package root
    current_file = pathlib.Path(__file__).resolve()
    sgl_kernel_package = current_file.parent.parent  # sgl_kernel/

    # Check for include directory in installed package
    include_path = sgl_kernel_package / "include"
    if include_path.exists() and (include_path / "sgl_kernel" / "jit_kernel").exists():
        return include_path

    # Check for development environment (one level up from python/)
    dev_include_path = sgl_kernel_package.parent.parent / "include"
    if (
        dev_include_path.exists()
        and (dev_include_path / "sgl_kernel" / "jit_kernel").exists()
    ):
        return dev_include_path

    raise RuntimeError(
        f"Cannot find sgl_kernel include directory. "
        f"Checked: {include_path}, {dev_include_path}"
    )


# Resolve paths at module load time
_KERNEL_INCLUDE_PATH = _resolve_kernel_include_path()
_SYCL_KERNEL_PATH = _KERNEL_INCLUDE_PATH / "sgl_kernel" / "jit_kernel"

DEFAULT_SYCL_INCLUDE = [str(_KERNEL_INCLUDE_PATH)]


def _get_pytorch_sycl_lib_path() -> str:
    """Get the directory containing PyTorch's libsycl.so.

    Matches any ``libsycl.so*`` soname (e.g. ``libsycl.so.8``, ``libsycl.so.9``)
    so detection stays version-agnostic across PyTorch builds compiled against
    different oneAPI releases.

    Returns an empty string if PyTorch's SYCL runtime cannot be located.
    """
    try:
        torch_lib_path = pathlib.Path(torch.__file__).parent / "lib"
        # Match libsycl.so, libsycl.so.8, libsycl.so.9, ... regardless of version
        if any(torch_lib_path.glob("libsycl.so*")):
            return str(torch_lib_path)
    except Exception as e:
        logger.warning("Failed to locate PyTorch SYCL runtime: %s", e)

    logger.warning(
        "Could not find PyTorch's libsycl.so under torch/lib; linking against the "
        "SYCL runtime will be left to the system loader, which may cause an ABI "
        "mismatch. Ensure PyTorch is installed with SYCL/XPU support."
    )
    return ""


_PYTORCH_SYCL_LIB_PATH = _get_pytorch_sycl_lib_path()

# Link against PyTorch's SYCL runtime to avoid ABI mismatch
DEFAULT_LDFLAGS = (
    [f"-L{_PYTORCH_SYCL_LIB_PATH}", f"-Wl,-rpath,{_PYTORCH_SYCL_LIB_PATH}"]
    if _PYTORCH_SYCL_LIB_PATH
    else []
)


def _compute_cache_key(cache_inputs: List[Any]) -> str:
    """Compute SHA256 hash for caching compiled kernels."""
    hasher = hashlib.sha256()
    for item in cache_inputs:
        hasher.update(str(item).encode())
        hasher.update(b"\0")
    return hasher.hexdigest()[:16]


def _get_cache_dir() -> pathlib.Path:
    """Get or create cache directory for compiled SYCL kernels."""
    cache_dir = pathlib.Path.home() / ".cache" / "sgl_kernel" / "jit_sycl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _sanitize_module_name(*args: str) -> str:
    """Create a safe module name from arguments."""
    safe_parts = []
    for arg in args:
        part = re.sub(r"[^A-Za-z0-9_.-]", "_", str(arg)).strip("._-")
        if part:
            safe_parts.append(part)
    if not safe_parts:
        return "sgl_kernel_jit_sycl"
    return "sgl_kernel_jit_sycl_" + "_".join(safe_parts)


@functools.lru_cache(maxsize=1)
def _get_icpx_version() -> str:
    """Get icpx compiler version string."""
    try:
        result = subprocess.run(
            ["icpx", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip()
    except Exception:
        return ""


@cache_once
def is_icpx_available() -> bool:
    """Check if icpx compiler is available."""
    try:
        result = subprocess.run(
            ["icpx", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class SYCLModule:
    """Wrapper for dynamically loaded SYCL shared library."""

    def __init__(self, so_path: pathlib.Path, module_name: str):
        import ctypes

        self.so_path = so_path
        self.module_name = module_name
        try:
            self._lib = ctypes.CDLL(str(so_path))
        except OSError as e:
            raise RuntimeError(
                f"Failed to load SYCL module '{module_name}' from {so_path}: {e}"
            ) from e
        self._functions = {}
        self._configured_funcs = (
            {}
        )  # Cache for functions with configured argtypes/restype

    def __getattr__(self, name: str):
        """Dynamically load function from shared library."""
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        if name not in self._functions:
            # Try bare name first, then with _wrapper suffix
            for candidate in (name, f"{name}_wrapper"):
                try:
                    func = getattr(self._lib, candidate)
                    self._functions[name] = func
                    break
                except AttributeError:
                    continue
            else:
                raise AttributeError(
                    f"SYCL module '{self.module_name}' has no function '{name}'"
                )

        return self._functions[name]

    def get_function(self, func_name: str, argtypes: list, restype=None):
        """Get a ctypes function with configured argtypes and restype, with caching.

        Args:
            func_name: Name of the exported C function
            argtypes: List of ctypes types for function arguments
            restype: Return type (defaults to None for void functions)

        Returns:
            Configured ctypes function object

        Example:
            func = module.get_function(
                "my_kernel_fp32",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
            )
            func(queue_ptr, 10, 3.14)
        """
        # Use tuple of argtypes for cache key (lists aren't hashable)
        cache_key = (func_name, tuple(argtypes), restype)

        if cache_key not in self._configured_funcs:
            func = getattr(self._lib, func_name)
            func.argtypes = argtypes
            func.restype = restype
            self._configured_funcs[cache_key] = func

        return self._configured_funcs[cache_key]


class _LRUModuleCache:
    """LRU cache for loaded SYCL modules to prevent unbounded memory growth."""

    def __init__(self, maxsize: int = 128):
        self._cache: OrderedDict[str, SYCLModule] = OrderedDict()
        self._maxsize = maxsize

    def _close_module(self, module: SYCLModule) -> None:
        """No-op to prevent unloading modules that are still referenced by cached wrappers."""
        pass

    def get(self, key: str) -> SYCLModule | None:
        """Get module from cache, moving it to end (most recently used)."""
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, module: SYCLModule) -> None:
        """Add module to cache, evicting oldest if at capacity."""
        if key in self._cache:
            old_module = self._cache[key]
            self._cache[key] = module
            self._cache.move_to_end(key)
            if old_module is not module:
                self._close_module(old_module)
        else:
            # Evict oldest if at capacity
            if len(self._cache) >= self._maxsize:
                _, evicted_module = self._cache.popitem(last=False)
                self._close_module(evicted_module)
            self._cache[key] = module

    def clear(self) -> None:
        """Clear all cached modules."""
        for module in self._cache.values():
            self._close_module(module)
        self._cache.clear()


# In-memory cache for loaded SYCL modules (avoids repeated dlopen overhead)
_LOADED_MODULES_CACHE = _LRUModuleCache(maxsize=128)


def clear_module_cache() -> None:
    """Clear the in-memory SYCL module cache.

    Useful for testing or to free memory in long-lived processes.
    """
    _LOADED_MODULES_CACHE.clear()


def load_jit_sycl(
    *args: str,
    sycl_files: List[str] | None = None,
    cpp_files: List[str] | None = None,
    sycl_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_sycl_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> SYCLModule:
    """
    Load a JIT-compiled SYCL module from source files.

    :param args: Unique marker of the JIT module. Must be distinct for different kernels.
    :type args: str
    :param sycl_files: A list of SYCL source files (relative to jit_kernel directory).
    :type sycl_files: List[str] | None
    :param cpp_files: A list of C++ source files.
    :type cpp_files: List[str] | None
    :param sycl_wrappers: A list of SYCL wrappers (export_name, kernel_class).
    :type sycl_wrappers: List[Tuple[str, str]] | None
    :param extra_cflags: Extra C++ compiler flags.
    :type extra_cflags: List[str] | None
    :param extra_sycl_cflags: Extra SYCL compiler flags.
    :type extra_sycl_cflags: List[str] | None
    :param extra_ldflags: Extra linker flags.
    :type extra_ldflags: List[str] | None
    :param extra_include_paths: Extra include paths.
    :type extra_include_paths: List[str] | None
    :param build_directory: The build directory for JIT compilation.
    :type build_directory: str | None
    :return: A JIT-compiled SYCL module.
    :rtype: SYCLModule
    """

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("XPU is not available. Cannot compile SYCL kernels.")

    if not is_icpx_available():
        raise RuntimeError(
            "icpx compiler not found. Please install Intel oneAPI toolkit and source setvars.sh"
        )

    sycl_files = sycl_files or []
    cpp_files = cpp_files or []
    sycl_wrappers = sycl_wrappers or []
    extra_cflags = extra_cflags or []
    extra_sycl_cflags = extra_sycl_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    if cpp_files:
        raise NotImplementedError(
            "load_jit_sycl does not currently support 'cpp_files'; "
            "pass only SYCL sources via 'sycl_files'."
        )
    if extra_cflags:
        raise NotImplementedError(
            "load_jit_sycl does not currently support 'extra_cflags'; "
            "use 'extra_sycl_cflags' for SYCL compilation flags."
        )
    if sycl_wrappers:
        raise NotImplementedError(
            "Automatic SYCL wrapper generation is not implemented. "
            "Use exported C APIs from SYCL source files directly."
        )

    # Generate module name
    module_name = _sanitize_module_name(*args)

    # Read source files
    sycl_sources = []
    for f in sycl_files:
        sycl_path = (_SYCL_KERNEL_PATH / f).resolve()
        if not sycl_path.exists():
            raise FileNotFoundError(f"SYCL source file not found: {sycl_path}")
        sycl_sources.append(sycl_path.read_text())

    # Check cache
    if build_directory:
        cache_dir = pathlib.Path(build_directory)
    else:
        cache_dir = _get_cache_dir()

    aot_flags = _get_sycl_aot_flags()
    effective_include_paths = DEFAULT_SYCL_INCLUDE + extra_include_paths

    env_cache_vars = [
        os.environ.get("SYCL_INCLUDE_DIR", ""),
        os.environ.get("LIBRARY_PATH", ""),
        os.environ.get("LD_LIBRARY_PATH", ""),
    ]

    # Compute cache key from compilation-affecting inputs. Use the relative
    # source filenames (not absolute paths) plus the source contents so the
    # cache key stays portable across machines and install locations.
    cache_key = _compute_cache_key(
        [
            args,
            sycl_files,
            sycl_sources,
            DEFAULT_SYCL_CFLAGS,
            aot_flags,
            extra_sycl_cflags,
            DEFAULT_LDFLAGS,
            extra_ldflags,
            effective_include_paths,
            _get_icpx_version(),
            env_cache_vars,
        ]
    )

    cache_dir.mkdir(parents=True, exist_ok=True)

    so_path = cache_dir / f"{module_name}_{cache_key}.so"

    # Check in-memory cache first (avoids dlopen overhead)
    memory_cache_key = f"{module_name}_{cache_key}"
    cached_module = _LOADED_MODULES_CACHE.get(memory_cache_key)
    if cached_module is not None:
        return cached_module

    # Compile if not cached on disk
    if not so_path.exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)

            # Write main source file that includes everything
            main_source = tmpdir_path / "kernel.cpp"
            includes = []
            for f in sycl_files:
                sycl_path = (_SYCL_KERNEL_PATH / f).resolve()
                includes.append(f'#include "{sycl_path}"')

            main_source.write_text("\n".join(includes) + "\n")

            # Create the output file in the cache directory
            with tempfile.NamedTemporaryFile(
                dir=so_path.parent,
                prefix=f"{so_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp_so_file:
                tmp_so_path = pathlib.Path(tmp_so_file.name)

            try:
                # Build icpx command
                cmd = ["icpx"]
                cmd += DEFAULT_SYCL_CFLAGS
                cmd += aot_flags
                cmd += extra_sycl_cflags or []

                # Add include paths
                for inc_dir in effective_include_paths:
                    cmd += ["-I", inc_dir]

                # Add source file
                cmd.append(str(main_source))

                # Add output
                cmd += ["-o", str(tmp_so_path)]

                # Add linker flags
                cmd += DEFAULT_LDFLAGS + extra_ldflags

                # Compile
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    error_msg = f"ICPX compilation failed:\n"
                    error_msg += f"Command: {' '.join(cmd)}\n"
                    error_msg += f"STDOUT: {result.stdout}\n"
                    error_msg += f"STDERR: {result.stderr}\n"
                    raise RuntimeError(error_msg)

                # Publish compiled library atomically
                os.replace(tmp_so_path, so_path)
            finally:
                # Clean up temp file if compilation failed
                if tmp_so_path.exists():
                    tmp_so_path.unlink()

    # Load module from disk
    module = SYCLModule(so_path, module_name)

    # Store in in-memory cache
    _LOADED_MODULES_CACHE.put(memory_cache_key, module)

    return module


__all__ = [
    "load_jit_sycl",
    "is_icpx_available",
    "SYCLModule",
    "clear_module_cache",
]
