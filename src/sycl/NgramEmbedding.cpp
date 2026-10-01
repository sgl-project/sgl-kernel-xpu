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

/*
SYCL port of the ngram-embedding compute kernels (speculative/ngram_embedding.cuh).
Four pure index/gather kernels, no reductions/atomics/shared-memory:
  - compute_n_gram_ids / _decode: hash the last (n+2) tokens of each request's
    context into an n-gram id per (n, k, token) combination.
  - update_token_table / _decode: write accepted tokens into the per-request
    ring of the ngram token table (negated when the token is in ignore_tokens).
Index map mirrors CUDA: blockIdx.x -> work-group, threadIdx.x -> local id.
*/

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

constexpr int kBlockThreads = 256;
constexpr int kDecodeBlockSize = 256;
constexpr int kMaxComputeNGramIdsDecodeBlocks = 65535;
constexpr int kMaxUpdateTokenTableDecodeBlocks = 1024;

// Prefill: grid is (num_configs * batch_size) work-groups, request-major within
// each config; every group walks its request's tokens with a stride loop.
struct ComputeNGramIdsKernel {
  int batch_size_;
  int ne_n_;
  int ne_k_;
  const int* ne_weights_;                      // [ne_n-1, ne_k, ne_n]
  const int* ne_mods_;                         // [ne_n-1, ne_k]
  const int* exclusive_ne_embeder_size_sums_;  // [(ne_n-1)*ne_k]
  const int* exclusive_req_len_sums_;          // [batch_size+1]
  const int* ne_token_table_;                  // [max_running_reqs, max_context_len]
  int max_context_len_;
  const int64_t* row_indices_;  // [batch_size]
  const int* column_starts_;    // [batch_size]
  int* n_gram_ids_;             // [token_num, (ne_n-1)*ne_k]
  int eos_token_id_;

  void operator()(sycl::nd_item<1> item) const {
    const int group_id = static_cast<int>(item.get_group(0));
    const int req_id = group_id % batch_size_;
    const int config_id = (group_id - req_id) / batch_size_;
    // n and k are offset from their physical meaning (real_n-2, real_k-1): they
    // index ne_weights [ne_n-1, ne_k, ne_n] and ne_mods [ne_n-1, ne_k].
    const int k = config_id % ne_k_;
    const int n = (config_id - k) / ne_k_;
    const int ne_weight_base_idx = n * ne_k_ * ne_n_ + k * ne_n_;
    const int ne_mod = ne_mods_[n * ne_k_ + k];

    const int tid = static_cast<int>(item.get_local_id(0));
    const int lrange = static_cast<int>(item.get_local_range(0));
    for (int i = exclusive_req_len_sums_[req_id] + tid; i < exclusive_req_len_sums_[req_id + 1]; i += lrange) {
      uint64_t n_gram_id = 0;
      const int64_t current_token_offset = i - exclusive_req_len_sums_[req_id];
      const int64_t req_token_table_index = row_indices_[req_id] * static_cast<int64_t>(max_context_len_);
      const int64_t current_token_table_index = req_token_table_index + column_starts_[req_id] + current_token_offset;
      for (int j = 0; j < n + 2; j++) {
        if (current_token_table_index - j < req_token_table_index) {
          break;  // out of this request's range
        }
        const int table_token = ne_token_table_[current_token_table_index - j];
        if (table_token < 0) {
          break;  // marked ignored during write
        }
        if (table_token == eos_token_id_ && j > 0) {
          // j==0 (the current token) is allowed; only break when looking back,
          // so the n-gram context never crosses an eos boundary.
          break;
        }
        const uint64_t term =
            static_cast<uint64_t>(table_token) * static_cast<uint64_t>(ne_weights_[ne_weight_base_idx + j]);
        n_gram_id += term % ne_mod;
      }
      n_gram_id %= ne_mod;
      n_gram_id += exclusive_ne_embeder_size_sums_[n * ne_k_ + k];
      n_gram_ids_[i * (ne_n_ - 1) * ne_k_ + n * ne_k_ + k] = static_cast<int>(n_gram_id);
    }
  }
};

// Decode: one output per (request, config); flat grid-stride loop.
struct ComputeNGramIdsDecodeKernel {
  int batch_size_;
  int ne_n_;
  int ne_k_;
  const int* ne_weights_;
  const int* ne_mods_;
  const int* exclusive_ne_embeder_size_sums_;
  const int* ne_token_table_;
  int max_context_len_;
  const int64_t* row_indices_;
  const int* column_starts_;
  int* n_gram_ids_;  // [batch_size, (ne_n-1)*ne_k]
  int eos_token_id_;

  void operator()(sycl::nd_item<1> item) const {
    const int num_configs = (ne_n_ - 1) * ne_k_;
    const int total_outputs = batch_size_ * num_configs;
    const int stride = static_cast<int>(item.get_global_range(0));
    for (int output_idx = static_cast<int>(item.get_global_id(0)); output_idx < total_outputs; output_idx += stride) {
      const int req_id = output_idx / num_configs;
      const int config_idx = output_idx - req_id * num_configs;
      const int k_idx = config_idx % ne_k_;
      const int n_idx = config_idx / ne_k_;
      const int weight_offset = n_idx * ne_k_ * ne_n_ + k_idx * ne_n_;
      const int ne_mod = ne_mods_[n_idx * ne_k_ + k_idx];

      uint64_t n_gram_id = 0;
      const int64_t req_token_table_offset = row_indices_[req_id] * static_cast<int64_t>(max_context_len_);
      const int64_t current_token_table_offset = req_token_table_offset + column_starts_[req_id];
      for (int j = 0; j < n_idx + 2; j++) {
        if (current_token_table_offset - j < req_token_table_offset) {
          break;
        }
        const int token = ne_token_table_[current_token_table_offset - j];
        if (token < 0) {
          break;
        }
        if (token == eos_token_id_ && j > 0) {
          break;
        }
        const uint64_t term = static_cast<uint64_t>(token) * static_cast<uint64_t>(ne_weights_[weight_offset + j]);
        n_gram_id += term % ne_mod;
      }
      n_gram_id %= ne_mod;
      n_gram_id += exclusive_ne_embeder_size_sums_[n_idx * ne_k_ + k_idx];
      n_gram_ids_[output_idx] = static_cast<int>(n_gram_id);
    }
  }
};

// Prefill table write: one work-group per request; req start is the prefix sum
// of req_lens (computed per-group, matching the CUDA kernel).
struct UpdateTokenTableKernel {
  int batch_size_;
  const int* tokens_;    // [token_num]
  int* ne_token_table_;  // [max_running_reqs, max_context_len]
  int max_context_len_;
  const int64_t* row_indices_;
  const int* column_starts_;
  const int* req_lens_;  // [batch_size]
  int ignore_token_num_;
  const int* ignore_tokens_;  // [ignore_token_num] or nullptr

  void operator()(sycl::nd_item<1> item) const {
    const int req_id = static_cast<int>(item.get_group(0)) % batch_size_;
    int start = 0;
    for (int i = 0; i < req_id; i++) {
      start += req_lens_[i];
    }
    const int end = start + req_lens_[req_id];

    const int tid = static_cast<int>(item.get_local_id(0));
    const int lrange = static_cast<int>(item.get_local_range(0));
    for (int i = start + tid; i < end; i += lrange) {
      const int64_t current_token_offset = i - start;
      const int64_t req_token_table_index = row_indices_[req_id] * static_cast<int64_t>(max_context_len_);
      const int64_t current_token_table_index = req_token_table_index + column_starts_[req_id] + current_token_offset;
      ne_token_table_[current_token_table_index] = tokens_[i];
      for (int j = 0; j < ignore_token_num_; j++) {
        if (ignore_tokens_[j] == tokens_[i]) {
          ne_token_table_[current_token_table_index] = -tokens_[i];
          break;
        }
      }
    }
  }
};

// Decode table write: one accepted token per request; flat grid-stride loop.
struct UpdateTokenTableDecodeKernel {
  int batch_size_;
  const int* tokens_;    // [batch_size]
  int* ne_token_table_;  // [max_running_reqs, max_context_len]
  int max_context_len_;
  const int64_t* row_indices_;
  const int* column_starts_;

  void operator()(sycl::nd_item<1> item) const {
    const int stride = static_cast<int>(item.get_global_range(0));
    for (int req_id = static_cast<int>(item.get_global_id(0)); req_id < batch_size_; req_id += stride) {
      const int64_t token_table_offset =
          row_indices_[req_id] * static_cast<int64_t>(max_context_len_) + column_starts_[req_id];
      ne_token_table_[token_table_offset] = tokens_[req_id];
    }
  }
};

}  // namespace

SGL_KERNEL_EXPORT void compute_n_gram_ids(
    int64_t ne_n,
    int64_t ne_k,
    const at::Tensor& ne_weights,
    const at::Tensor& ne_mods,
    const at::Tensor& exclusive_ne_embedder_size_sums,
    const at::Tensor& tokens,
    const at::Tensor& exclusive_req_len_sums,
    const at::Tensor& ne_token_table,
    const at::Tensor& row_indices,
    const at::Tensor& column_starts,
    at::Tensor& n_gram_ids,
    int64_t eos_token_id) {
  CHECK_INPUT(ne_weights);
  CHECK_INPUT(ne_mods);
  CHECK_INPUT(exclusive_ne_embedder_size_sums);
  CHECK_INPUT(exclusive_req_len_sums);
  CHECK_INPUT(ne_token_table);
  CHECK_INPUT(row_indices);
  CHECK_INPUT(column_starts);
  CHECK_INPUT(n_gram_ids);

  const int batch_size = static_cast<int>(exclusive_req_len_sums.numel() - 1);
  if (batch_size <= 0) {
    return;
  }
  const int max_context_len = static_cast<int>(ne_token_table.size(1));
  const int num_configs = (static_cast<int>(ne_n) - 1) * static_cast<int>(ne_k);
  const int grid_size = num_configs * batch_size;
  if (grid_size <= 0) {
    return;
  }

  ComputeNGramIdsKernel kernel{
      batch_size,
      static_cast<int>(ne_n),
      static_cast<int>(ne_k),
      ne_weights.data_ptr<int32_t>(),
      ne_mods.data_ptr<int32_t>(),
      exclusive_ne_embedder_size_sums.data_ptr<int32_t>(),
      exclusive_req_len_sums.data_ptr<int32_t>(),
      ne_token_table.data_ptr<int32_t>(),
      max_context_len,
      row_indices.data_ptr<int64_t>(),
      column_starts.data_ptr<int32_t>(),
      n_gram_ids.data_ptr<int32_t>(),
      static_cast<int>(eos_token_id)};
  (void)tokens;  // present for CUDA-signature parity; the kernel reads ne_token_table
  auto& queue = dpcppGetCurrentQueue();
  sycl_kernel_submit(
      static_cast<int64_t>(grid_size) * kBlockThreads, static_cast<int64_t>(kBlockThreads), queue, kernel);
}

SGL_KERNEL_EXPORT void compute_n_gram_ids_decode(
    int64_t ne_n,
    int64_t ne_k,
    const at::Tensor& ne_weights,
    const at::Tensor& ne_mods,
    const at::Tensor& exclusive_ne_embedder_size_sums,
    const at::Tensor& ne_token_table,
    const at::Tensor& row_indices,
    const at::Tensor& column_starts,
    at::Tensor& n_gram_ids,
    int64_t eos_token_id) {
  CHECK_INPUT(ne_weights);
  CHECK_INPUT(ne_mods);
  CHECK_INPUT(exclusive_ne_embedder_size_sums);
  CHECK_INPUT(ne_token_table);
  CHECK_INPUT(row_indices);
  CHECK_INPUT(column_starts);
  CHECK_INPUT(n_gram_ids);

  const int batch_size = static_cast<int>(row_indices.numel());
  if (batch_size <= 0) {
    return;
  }
  const int max_context_len = static_cast<int>(ne_token_table.size(1));
  const int num_configs = (static_cast<int>(ne_n) - 1) * static_cast<int>(ne_k);
  const int total_outputs = batch_size * num_configs;
  if (total_outputs <= 0) {
    return;
  }
  const int grid_size =
      std::min(kMaxComputeNGramIdsDecodeBlocks, (total_outputs + kDecodeBlockSize - 1) / kDecodeBlockSize);

  ComputeNGramIdsDecodeKernel kernel{
      batch_size,
      static_cast<int>(ne_n),
      static_cast<int>(ne_k),
      ne_weights.data_ptr<int32_t>(),
      ne_mods.data_ptr<int32_t>(),
      exclusive_ne_embedder_size_sums.data_ptr<int32_t>(),
      ne_token_table.data_ptr<int32_t>(),
      max_context_len,
      row_indices.data_ptr<int64_t>(),
      column_starts.data_ptr<int32_t>(),
      n_gram_ids.data_ptr<int32_t>(),
      static_cast<int>(eos_token_id)};
  auto& queue = dpcppGetCurrentQueue();
  sycl_kernel_submit(
      static_cast<int64_t>(grid_size) * kDecodeBlockSize, static_cast<int64_t>(kDecodeBlockSize), queue, kernel);
}

SGL_KERNEL_EXPORT void update_token_table(
    const at::Tensor& tokens,
    const at::Tensor& ne_token_table,
    const at::Tensor& row_indices,
    const at::Tensor& column_starts,
    const at::Tensor& req_lens,
    const at::Tensor& ignore_tokens) {
  CHECK_INPUT(tokens);
  CHECK_INPUT(ne_token_table);
  CHECK_INPUT(row_indices);
  CHECK_INPUT(column_starts);
  CHECK_INPUT(req_lens);

  const int batch_size = static_cast<int>(req_lens.numel());
  if (batch_size <= 0) {
    return;
  }
  const int max_context_len = static_cast<int>(ne_token_table.size(1));

  int ignore_token_num = 0;
  const int32_t* ignore_tokens_ptr = nullptr;
  if (ignore_tokens.defined() && ignore_tokens.numel() > 0) {
    CHECK_INPUT(ignore_tokens);
    ignore_token_num = static_cast<int>(ignore_tokens.numel());
    ignore_tokens_ptr = ignore_tokens.data_ptr<int32_t>();
  }

  UpdateTokenTableKernel kernel{
      batch_size,
      tokens.data_ptr<int32_t>(),
      ne_token_table.data_ptr<int32_t>(),
      max_context_len,
      row_indices.data_ptr<int64_t>(),
      column_starts.data_ptr<int32_t>(),
      req_lens.data_ptr<int32_t>(),
      ignore_token_num,
      ignore_tokens_ptr};
  auto& queue = dpcppGetCurrentQueue();
  sycl_kernel_submit(
      static_cast<int64_t>(batch_size) * kBlockThreads, static_cast<int64_t>(kBlockThreads), queue, kernel);
}

SGL_KERNEL_EXPORT void update_token_table_decode(
    const at::Tensor& tokens,
    const at::Tensor& ne_token_table,
    const at::Tensor& row_indices,
    const at::Tensor& column_starts) {
  CHECK_INPUT(tokens);
  CHECK_INPUT(ne_token_table);
  CHECK_INPUT(row_indices);
  CHECK_INPUT(column_starts);

  const int batch_size = static_cast<int>(row_indices.numel());
  if (batch_size <= 0) {
    return;
  }
  const int max_context_len = static_cast<int>(ne_token_table.size(1));
  const int grid_size =
      std::min(kMaxUpdateTokenTableDecodeBlocks, (batch_size + kDecodeBlockSize - 1) / kDecodeBlockSize);

  UpdateTokenTableDecodeKernel kernel{
      batch_size,
      tokens.data_ptr<int32_t>(),
      ne_token_table.data_ptr<int32_t>(),
      max_context_len,
      row_indices.data_ptr<int64_t>(),
      column_starts.data_ptr<int32_t>()};
  auto& queue = dpcppGetCurrentQueue();
  sycl_kernel_submit(
      static_cast<int64_t>(grid_size) * kDecodeBlockSize, static_cast<int64_t>(kDecodeBlockSize), queue, kernel);
}
