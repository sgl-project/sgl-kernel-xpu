/* Copyright 2026 SGLang Team. All Rights Reserved.

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

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("inkling_mxfp4_mapping(Tensor x, bool column_major_scales=False, float eps=1e-10) -> (Tensor, Tensor)");
  m.impl("inkling_mxfp4_mapping", torch::kXPU, &inkling_mxfp4_mapping);

  m.def("inkling_nvfp4_layout(Tensor x, float global_scale) -> (Tensor, Tensor)");
  m.impl("inkling_nvfp4_layout", torch::kXPU, &inkling_nvfp4_layout);
}

REGISTER_EXTENSION(inkling_quantization_ops)
