/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

// The per-kernel SYCL shared libraries are compiled with -fvisibility=hidden to
// keep their dynamic symbol tables (.dynsym/.dynstr) small. Public entry points
// that are called from common_ops (torch_extension_sycl.cc) must therefore be
// re-exported with default visibility so they remain dynamically linkable.
//
// This header is intentionally dependency-free (no Python.h / torch / cutlass),
// so it is safe to include in any host or SYCL device translation unit and is
// insensitive to include ordering.
#if defined(__GNUC__) || defined(__clang__)
#define SGL_KERNEL_EXPORT __attribute__((visibility("default")))
#else
#define SGL_KERNEL_EXPORT
#endif
