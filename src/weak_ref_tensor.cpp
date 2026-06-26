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

#include <ATen/ATen.h>
#include <ATen/Tensor.h>

#include <vector>

// [NOTE] weak_ref_tensor returns a no-ownership alias over the XPU tensor via `at::from_blob`.
// Use weak references for XPU Graph outputs to decouple object lifetime from XPU graph.
// This allows the memory to be efficiently reused.
at::Tensor weak_ref_tensor(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_xpu(), "weak_ref_tensor expects an XPU tensor");
  TORCH_CHECK(!tensor.requires_grad(), "weak_ref_tensor does not support autograd tensors");
  // The returned tensor does not own the underlying storage; ensure the backing allocation outlives it.
  void* data_ptr = tensor.data_ptr();
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();

  auto options = tensor.options();

  auto new_tensor = at::from_blob(data_ptr, sizes, strides, options);

  return new_tensor;
}
