#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include "sgl_kernel/xe35/logits_vocab_ops.h"

#include <sycl/sycl.hpp>

#include "../MemoryAccess.h"
#include "../SYCLHelpers.h"

namespace at::native::xpu {
namespace {

template <typename Scalar>
struct VocabMaskKernel {
  const Scalar* __restrict__ input;
  int64_t* __restrict__ mapped;
  bool* __restrict__ invalid;
  int64_t count;
  int64_t org_start;
  int64_t org_end;
  int64_t added_start;
  int64_t added_end;
  int64_t added_offset;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t i = item.get_global_id(0);
    if (i >= count) {
      return;
    }
    const int64_t value = static_cast<int64_t>(input[i]);
    const bool org_valid = value >= org_start && value < org_end;
    const bool added_valid = value >= added_start && value < added_end;
    const bool valid = org_valid || added_valid;
    const int64_t offset = (org_valid ? org_start : 0) + (added_valid ? added_offset : 0);
    mapped[i] = valid ? value - offset : 0;
    invalid[i] = !valid;
  }
};

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> vocab_mask_xe35(
    torch::Tensor input,
    int64_t org_start,
    int64_t org_end,
    int64_t num_org_padding,
    int64_t added_start,
    int64_t added_end) {
  TORCH_CHECK(input.is_xpu(), "input must be on XPU");
  TORCH_CHECK(input.scalar_type() == at::kInt || input.scalar_type() == at::kLong, "input must be int32 or int64");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  auto mapped = torch::empty(input.sizes(), input.options().dtype(at::kLong));
  auto invalid = torch::empty(input.sizes(), input.options().dtype(at::kBool));
  const int64_t count = input.numel();
  if (count == 0) {
    return {mapped, invalid};
  }
  constexpr int Workgroup = 256;
  const auto global_size = ((count + Workgroup - 1) / Workgroup) * Workgroup;
  const auto added_offset = added_start - (org_end - org_start) - num_org_padding;
  auto queue = at::xpu::getCurrentXPUStream().queue();
  if (input.scalar_type() == at::kInt) {
    VocabMaskKernel<int32_t> kernel{
        input.data_ptr<int32_t>(),
        mapped.data_ptr<int64_t>(),
        invalid.data_ptr<bool>(),
        count,
        org_start,
        org_end,
        added_start,
        added_end,
        added_offset};
    sycl_kernel_submit(sycl::range<1>(global_size), sycl::range<1>(Workgroup), queue, kernel);
  } else {
    VocabMaskKernel<int64_t> kernel{
        input.data_ptr<int64_t>(),
        mapped.data_ptr<int64_t>(),
        invalid.data_ptr<bool>(),
        count,
        org_start,
        org_end,
        added_start,
        added_end,
        added_offset};
    sycl_kernel_submit(sycl::range<1>(global_size), sycl::range<1>(Workgroup), queue, kernel);
  }
  return {mapped, invalid};
}

namespace {

constexpr int kVocabEmbeddingWorkgroup = 256;
constexpr int kVocabEmbeddingVectorBytes = 32;

template <typename Id, typename Scalar, bool Vectorized, bool HasTail, int Dimensions, int VectorBytes>
struct MuseVocabEmbeddingKernel {
  const Id* __restrict__ input;
  const Scalar* __restrict__ weight;
  Scalar* __restrict__ output;
  int64_t n_tokens;
  int64_t hidden;
  int64_t row_stride;
  int64_t groups_per_row;
  int64_t org_start;
  int64_t org_end;
  int64_t added_start;
  int64_t added_end;
  int64_t added_offset;

  void operator()(sycl::nd_item<Dimensions> item) const {
    constexpr int Workgroup = kVocabEmbeddingWorkgroup;
    constexpr int ValuesPerThread = Vectorized ? VectorBytes / sizeof(Scalar) : 4;
    int64_t row;
    int64_t tile;
    if constexpr (Dimensions == 2) {
      row = item.get_group(0);
      tile = item.get_group(1);
    } else {
      const int64_t group = item.get_group(0);
      row = group / groups_per_row;
      tile = group % groups_per_row;
      if (row >= n_tokens) return;
    }
    const int64_t col = (tile * Workgroup + item.get_local_id(Dimensions - 1)) * ValuesPerThread;
    if constexpr (Vectorized && HasTail) {
      if (col >= hidden) return;
    }
    const int64_t token = static_cast<int64_t>(input[row]);
    const bool original = token >= org_start && token < org_end;
    const bool added = token >= added_start && token < added_end;
    const bool valid = original || added;
    const int64_t local = original ? token - org_start : token - added_offset;
    if constexpr (Vectorized) {
      using Vec = aligned_vector_loop<Scalar, ValuesPerThread>;
      Vec result{};
      if (valid) {
        result = *reinterpret_cast<const Vec*>(weight + local * row_stride + col);
      }
      *reinterpret_cast<Vec*>(output + row * hidden + col) = result;
    } else {
      for (int j = 0; j < ValuesPerThread; ++j) {
        if (col + j < hidden) {
          output[row * hidden + col + j] = valid ? weight[local * row_stride + col + j] : static_cast<Scalar>(0.0f);
        }
      }
    }
  }
};

template <typename Id, typename Scalar, bool Vectorized, bool HasTail, int Dimensions, int VectorBytes>
void submit_vocab_embedding(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    int64_t org_start,
    int64_t org_end,
    int64_t added_start,
    int64_t added_end,
    int64_t added_offset,
    sycl::queue& queue) {
  constexpr int Workgroup = kVocabEmbeddingWorkgroup;
  constexpr int ValuesPerThread = Vectorized ? VectorBytes / sizeof(Scalar) : 4;
  const int64_t hidden = weight.size(1);
  const int64_t groups_per_row = (hidden + Workgroup * ValuesPerThread - 1) / (Workgroup * ValuesPerThread);
  MuseVocabEmbeddingKernel<Id, Scalar, Vectorized, HasTail, Dimensions, VectorBytes> kernel{
      input.data_ptr<Id>(),
      weight.data_ptr<Scalar>(),
      output.data_ptr<Scalar>(),
      input.numel(),
      hidden,
      weight.stride(0),
      groups_per_row,
      org_start,
      org_end,
      added_start,
      added_end,
      added_offset};
  if constexpr (Dimensions == 2) {
    sycl_kernel_submit(
        sycl::range<2>(input.numel(), groups_per_row * Workgroup), sycl::range<2>(1, Workgroup), queue, kernel);
  } else {
    sycl_kernel_submit(
        sycl::range<1>(input.numel() * groups_per_row * Workgroup), sycl::range<1>(Workgroup), queue, kernel);
  }
}

template <typename Id, typename Scalar, bool Vectorized, int Dimensions, int VectorBytes>
void launch_vocab_embedding_layout(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    int64_t org_start,
    int64_t org_end,
    int64_t added_start,
    int64_t added_end,
    int64_t added_offset,
    sycl::queue& queue) {
  if constexpr (Vectorized) {
    if (weight.size(1) % (kVocabEmbeddingWorkgroup * VectorBytes / sizeof(Scalar)) == 0) {
      submit_vocab_embedding<Id, Scalar, Vectorized, false, Dimensions, VectorBytes>(
          input, weight, output, org_start, org_end, added_start, added_end, added_offset, queue);
      return;
    }
  }
  submit_vocab_embedding<Id, Scalar, Vectorized, true, Dimensions, VectorBytes>(
      input, weight, output, org_start, org_end, added_start, added_end, added_offset, queue);
}

template <typename Id, typename Scalar, bool Vectorized>
void launch_vocab_embedding(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    int64_t org_start,
    int64_t org_end,
    int64_t added_start,
    int64_t added_end,
    int64_t added_offset,
    sycl::queue& queue) {
  if (input.numel() > kVocabEmbeddingWorkgroup && weight.nbytes() > 64 * 1024 * 1024) {
    constexpr int VectorBytes = sizeof(Scalar) == 2 ? 16 : kVocabEmbeddingVectorBytes;
    launch_vocab_embedding_layout<Id, Scalar, Vectorized, 1, VectorBytes>(
        input, weight, output, org_start, org_end, added_start, added_end, added_offset, queue);
  } else {
    launch_vocab_embedding_layout<Id, Scalar, Vectorized, 2, kVocabEmbeddingVectorBytes>(
        input, weight, output, org_start, org_end, added_start, added_end, added_offset, queue);
  }
}

template <typename Id>
void dispatch_vocab_embedding(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    int64_t org_start,
    int64_t org_end,
    int64_t added_start,
    int64_t added_end,
    int64_t added_offset,
    sycl::queue& queue) {
  const auto vector_aligned = [&]() {
    const int64_t elements = kVocabEmbeddingVectorBytes / weight.element_size();
    return weight.size(1) % elements == 0 && weight.stride(0) % elements == 0 &&
           reinterpret_cast<uintptr_t>(weight.data_ptr()) % kVocabEmbeddingVectorBytes == 0 &&
           reinterpret_cast<uintptr_t>(output.data_ptr()) % kVocabEmbeddingVectorBytes == 0;
  };
  if (weight.scalar_type() == at::kBFloat16) {
    if (vector_aligned()) {
      launch_vocab_embedding<Id, at::BFloat16, true>(
          input, weight, output, org_start, org_end, added_start, added_end, added_offset, queue);
    } else {
      launch_vocab_embedding<Id, at::BFloat16, false>(
          input, weight, output, org_start, org_end, added_start, added_end, added_offset, queue);
    }
  } else if (weight.scalar_type() == at::kHalf) {
    if (vector_aligned()) {
      launch_vocab_embedding<Id, at::Half, true>(
          input, weight, output, org_start, org_end, added_start, added_end, added_offset, queue);
    } else {
      launch_vocab_embedding<Id, at::Half, false>(
          input, weight, output, org_start, org_end, added_start, added_end, added_offset, queue);
    }
  } else {
    if (vector_aligned()) {
      launch_vocab_embedding<Id, float, true>(
          input, weight, output, org_start, org_end, added_start, added_end, added_offset, queue);
    } else {
      launch_vocab_embedding<Id, float, false>(
          input, weight, output, org_start, org_end, added_start, added_end, added_offset, queue);
    }
  }
}

}  // namespace

void vocab_parallel_embedding_xe35(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int64_t org_start,
    int64_t org_end,
    int64_t num_org_padding,
    int64_t added_start,
    int64_t added_end) {
  TORCH_CHECK(input.is_xpu() && weight.is_xpu() && output.is_xpu(), "vocab embedding requires XPU tensors");
  TORCH_CHECK(
      (input.scalar_type() == at::kInt || input.scalar_type() == at::kLong) && input.is_contiguous(),
      "vocab embedding requires contiguous int32/int64 IDs");
  TORCH_CHECK(
      weight.dim() == 2 && weight.stride(1) == 1 &&
          (weight.scalar_type() == at::kBFloat16 || weight.scalar_type() == at::kHalf ||
           weight.scalar_type() == at::kFloat),
      "invalid vocab embedding weights");
  TORCH_CHECK(
      output.scalar_type() == weight.scalar_type() && output.is_contiguous() && output.dim() == input.dim() + 1 &&
          output.size(-1) == weight.size(1),
      "invalid vocab embedding output");
  for (int64_t i = 0; i < input.dim(); ++i) {
    TORCH_CHECK(output.size(i) == input.size(i), "vocab embedding leading dimensions must match");
  }
  TORCH_CHECK(
      org_start >= 0 && org_start <= org_end && added_start <= added_end && num_org_padding >= 0 &&
          org_end - org_start + num_org_padding + added_end - added_start <= weight.size(0),
      "vocab embedding windows exceed the local shard");
  if (input.numel() == 0 || weight.size(1) == 0) return;
  auto queue = at::xpu::getCurrentXPUStream().queue();
  const int64_t added_offset = added_start - (org_end - org_start) - num_org_padding;
  if (input.scalar_type() == at::kInt) {
    dispatch_vocab_embedding<int32_t>(
        input, weight, output, org_start, org_end, added_start, added_end, added_offset, queue);
  } else {
    dispatch_vocab_embedding<int64_t>(
        input, weight, output, org_start, org_end, added_start, added_end, added_offset, queue);
  }
}

}  // namespace at::native::xpu
