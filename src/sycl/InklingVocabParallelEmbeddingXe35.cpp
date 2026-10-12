/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "sgl_kernel_export.h"

namespace {

constexpr int kWorkgroupSize = 256;

template <typename Index, typename Word>
class InklingVocabParallelEmbeddingKernel {
 public:
  InklingVocabParallelEmbeddingKernel(
      Word* out, const Index* ids, const Word* weight, int64_t width, int64_t vocab_start, int64_t vocab_end)
      : out_(out), ids_(ids), weight_(weight), width_(width), vocab_start_(vocab_start), vocab_end_(vocab_end) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t token = static_cast<int64_t>(item.get_group(0));
    const int64_t id = static_cast<int64_t>(ids_[token]);
    const bool valid = id >= vocab_start_ && id < vocab_end_;
    const Word* source = valid ? weight_ + (id - vocab_start_) * width_ : nullptr;
    Word* destination = out_ + token * width_;

    for (int64_t column = item.get_local_id(0); column < width_; column += kWorkgroupSize) {
      destination[column] = valid ? source[column] : Word{0};
    }
  }

 private:
  Word* out_;
  const Index* ids_;
  const Word* weight_;
  int64_t width_;
  int64_t vocab_start_;
  int64_t vocab_end_;
};

template <typename Index, typename Word>
void launch_embedding(
    at::Tensor& out,
    const at::Tensor& ids,
    const at::Tensor& weight,
    int64_t width,
    int64_t vocab_start,
    int64_t vocab_end,
    sycl::queue& queue) {
  auto* out_words = reinterpret_cast<Word*>(out.data_ptr());
  const auto* id_data = ids.data_ptr<Index>();
  const auto* weight_words = reinterpret_cast<const Word*>(weight.data_ptr());
  sycl_kernel_submit(
      ids.size(0) * kWorkgroupSize,
      kWorkgroupSize,
      queue,
      InklingVocabParallelEmbeddingKernel<Index, Word>(
          out_words, id_data, weight_words, width, vocab_start, vocab_end));
}

template <typename Index>
void dispatch_words(
    at::Tensor& out,
    const at::Tensor& ids,
    const at::Tensor& weight,
    int64_t vocab_start,
    int64_t vocab_end,
    sycl::queue& queue) {
  const int64_t hidden = weight.size(1);
  if (hidden % 2 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % alignof(uint32_t) == 0 &&
      reinterpret_cast<uintptr_t>(weight.data_ptr()) % alignof(uint32_t) == 0) {
    // A 32-bit word carries two BF16 or FP16 values without changing either bit pattern.
    launch_embedding<Index, uint32_t>(out, ids, weight, hidden / 2, vocab_start, vocab_end, queue);
  } else {
    launch_embedding<Index, uint16_t>(out, ids, weight, hidden, vocab_start, vocab_end, queue);
  }
}

}  // namespace

SGL_KERNEL_EXPORT void inkling_vocab_parallel_embedding(
    at::Tensor& out, const at::Tensor& ids, const at::Tensor& weight, int64_t vocab_start, int64_t vocab_end) {
  TORCH_CHECK(ids.is_xpu() && weight.is_xpu() && out.is_xpu(), "embedding tensors must be XPU tensors");
  TORCH_CHECK(
      ids.device() == weight.device() && ids.device() == out.device(), "embedding tensors must be on the same device");
  TORCH_CHECK(ids.dim() == 1, "ids must have shape [tokens]");
  TORCH_CHECK(weight.dim() == 2 && out.dim() == 2, "weight and out must be matrices");
  TORCH_CHECK(ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong, "ids must be int32 or int64");
  TORCH_CHECK(
      (weight.scalar_type() == at::kBFloat16 || weight.scalar_type() == at::kHalf) &&
          out.scalar_type() == weight.scalar_type(),
      "weight and out must have the same bfloat16 or float16 dtype");
  TORCH_CHECK(ids.is_contiguous() && weight.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(out.size(0) == ids.size(0) && out.size(1) == weight.size(1), "out must have shape [tokens, hidden]");
  TORCH_CHECK(
      vocab_end >= vocab_start && static_cast<uint64_t>(vocab_end) - static_cast<uint64_t>(vocab_start) ==
                                      static_cast<uint64_t>(weight.size(0)),
      "weight rows must equal vocab_end - vocab_start");

  if (out.numel() == 0) {
    return;
  }

  auto& queue = at::xpu::getCurrentXPUStream().queue();
  if (ids.scalar_type() == at::kInt) {
    dispatch_words<int32_t>(out, ids, weight, vocab_start, vocab_end, queue);
  } else {
    dispatch_words<int64_t>(out, ids, weight, vocab_start, vocab_end, queue);
  }
}
