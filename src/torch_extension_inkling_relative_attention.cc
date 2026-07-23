/* Copyright 2026 SGLang Team. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "inkling_relative_attention(Tensor q, Tensor k, Tensor v, Tensor q_to_seq, Tensor q_pos, Tensor cu_k, "
      "Tensor? rel_bias, float softmax_scale, bool causal, int window_size_left, int window_size_right, "
      "float softcap, int local_size=0, Tensor(a!)? out=None) -> (Tensor(a!), Tensor)");
  m.impl("inkling_relative_attention", torch::kXPU, &inkling_relative_attention);
}

REGISTER_EXTENSION(inkling_relative_attention_ops)
