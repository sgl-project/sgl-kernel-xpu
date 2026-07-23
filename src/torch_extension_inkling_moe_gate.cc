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
  m.def(
      "inkling_moe_gate_topk_renorm(Tensor logits, Tensor bias, Tensor global_scale, float route_scale, "
      "bool return_packed, int rows_per_workgroup=0) -> Tensor[]");
  m.impl("inkling_moe_gate_topk_renorm", torch::kXPU, &inkling_moe_gate_topk_renorm);

  m.def(
      "inkling_moe_gate_gemv(Tensor x, Tensor weight, int experts_per_workgroup=0, int subgroup_size=0) -> Tensor");
  m.impl("inkling_moe_gate_gemv", torch::kXPU, &inkling_moe_gate_gemv);

  m.def(
      "inkling_moe_gate_gemv_fused(Tensor x, Tensor weight, Tensor bias, Tensor global_scale, Tensor(a!) workspace, "
      "Tensor(b!) ticket, float route_scale, bool return_packed, int experts_per_workgroup=0, int subgroup_size=0) "
      "-> Tensor[]");
  m.impl("inkling_moe_gate_gemv_fused", torch::kXPU, &inkling_moe_gate_gemv_fused);
}

REGISTER_EXTENSION(inkling_moe_gate_ops)
