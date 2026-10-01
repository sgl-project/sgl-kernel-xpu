// SYCL port of sglang's speculative/ngram_embedding.cuh (LongCat-Flash ngram embedding).

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

constexpr int kBlockThreads = 256;

// Inputs shared by the prefill and decode n-gram id kernels.
struct NGramIdParams {
  int ne_n;
  int ne_k;
  const int* ne_weights;                       // [ne_n-1, ne_k, ne_n]
  const int* ne_mods;                          // [ne_n-1, ne_k]
  const int* exclusive_ne_embedder_size_sums;  // [(ne_n-1)*ne_k]
  const int* ne_token_table;                   // [max_running_reqs, max_context_len]
  int max_context_len;
  const int64_t* row_indices;  // [batch_size]
  const int* column_starts;    // [batch_size]
  int* n_gram_ids;             // [token_num, (ne_n-1)*ne_k]
  int eos_token_id;

  // Id of config (n, k) = (config / ne_k, config % ne_k) for the token `offset` past column_starts[req].
  int id(int req, int64_t offset, int config) const {
    const int64_t row_start = row_indices[req] * static_cast<int64_t>(max_context_len);
    const int64_t cur = row_start + column_starts[req] + offset;
    const int* weights = ne_weights + config * ne_n;
    const int mod = ne_mods[config];
    uint64_t n_gram_id = 0;
    for (int j = 0; j < config / ne_k + 2 && cur - j >= row_start; j++) {
      const int token = ne_token_table[cur - j];
      if (token < 0 || (token == eos_token_id && j > 0)) {
        break;  // ignored token, or an eos before the current token
      }
      n_gram_id += static_cast<uint64_t>(token) * static_cast<uint64_t>(weights[j]) % mod;
    }
    return static_cast<int>(n_gram_id % mod) + exclusive_ne_embedder_size_sums[config];
  }
};

// One work-group per (config, request); each walks the request's tokens.
struct ComputeNGramIdsKernel {
  NGramIdParams p_;
  int batch_size_;
  const int* exclusive_req_len_sums_;  // [batch_size+1]

  void operator()(sycl::nd_item<1> item) const {
    const int req = static_cast<int>(item.get_group(0)) % batch_size_;
    const int config = static_cast<int>(item.get_group(0)) / batch_size_;
    const int num_configs = (p_.ne_n - 1) * p_.ne_k;
    const int start = exclusive_req_len_sums_[req];
    for (int i = start + static_cast<int>(item.get_local_id(0)); i < exclusive_req_len_sums_[req + 1];
         i += static_cast<int>(item.get_local_range(0))) {
      p_.n_gram_ids[i * num_configs + config] = p_.id(req, i - start, config);
    }
  }
};

// One work-item per (request, config).
struct ComputeNGramIdsDecodeKernel {
  NGramIdParams p_;
  int total_outputs_;

  void operator()(sycl::nd_item<1> item) const {
    const int idx = static_cast<int>(item.get_global_id(0));
    if (idx < total_outputs_) {
      const int num_configs = (p_.ne_n - 1) * p_.ne_k;
      p_.n_gram_ids[idx] = p_.id(idx / num_configs, 0, idx % num_configs);
    }
  }
};

// One work-group per request; ignored tokens are stored negated.
struct UpdateTokenTableKernel {
  const int* tokens_;
  int* ne_token_table_;
  int max_context_len_;
  const int64_t* row_indices_;
  const int* column_starts_;
  const int* req_lens_;
  int ignore_token_num_;
  const int* ignore_tokens_;

  void operator()(sycl::nd_item<1> item) const {
    const int req = static_cast<int>(item.get_group(0));
    int start = 0;
    for (int i = 0; i < req; i++) {
      start += req_lens_[i];
    }
    int* row = ne_token_table_ + row_indices_[req] * static_cast<int64_t>(max_context_len_) + column_starts_[req];
    for (int i = static_cast<int>(item.get_local_id(0)); i < req_lens_[req];
         i += static_cast<int>(item.get_local_range(0))) {
      const int token = tokens_[start + i];
      bool ignored = false;
      for (int j = 0; j < ignore_token_num_ && !ignored; j++) {
        ignored = ignore_tokens_[j] == token;
      }
      row[i] = ignored ? -token : token;
    }
  }
};

// One work-item per request.
struct UpdateTokenTableDecodeKernel {
  int batch_size_;
  const int* tokens_;
  int* ne_token_table_;
  int max_context_len_;
  const int64_t* row_indices_;
  const int* column_starts_;

  void operator()(sycl::nd_item<1> item) const {
    const int req = static_cast<int>(item.get_global_id(0));
    if (req < batch_size_) {
      ne_token_table_[row_indices_[req] * static_cast<int64_t>(max_context_len_) + column_starts_[req]] = tokens_[req];
    }
  }
};

int64_t round_up(int64_t n) {
  return (n + kBlockThreads - 1) / kBlockThreads * kBlockThreads;
}

NGramIdParams make_params(
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
  return {
      static_cast<int>(ne_n),
      static_cast<int>(ne_k),
      ne_weights.data_ptr<int32_t>(),
      ne_mods.data_ptr<int32_t>(),
      exclusive_ne_embedder_size_sums.data_ptr<int32_t>(),
      ne_token_table.data_ptr<int32_t>(),
      static_cast<int>(ne_token_table.size(1)),
      row_indices.data_ptr<int64_t>(),
      column_starts.data_ptr<int32_t>(),
      n_gram_ids.data_ptr<int32_t>(),
      static_cast<int>(eos_token_id)};
}

}  // namespace

SGL_KERNEL_EXPORT void compute_n_gram_ids(
    int64_t ne_n,
    int64_t ne_k,
    const at::Tensor& ne_weights,
    const at::Tensor& ne_mods,
    const at::Tensor& exclusive_ne_embedder_size_sums,
    const at::Tensor& tokens,  // unused, kept for parity with the CUDA signature
    const at::Tensor& exclusive_req_len_sums,
    const at::Tensor& ne_token_table,
    const at::Tensor& row_indices,
    const at::Tensor& column_starts,
    at::Tensor& n_gram_ids,
    int64_t eos_token_id) {
  CHECK_INPUT(exclusive_req_len_sums);
  const int batch_size = static_cast<int>(exclusive_req_len_sums.numel() - 1);
  if (batch_size <= 0) {
    return;
  }
  ComputeNGramIdsKernel kernel{
      make_params(
          ne_n,
          ne_k,
          ne_weights,
          ne_mods,
          exclusive_ne_embedder_size_sums,
          ne_token_table,
          row_indices,
          column_starts,
          n_gram_ids,
          eos_token_id),
      batch_size,
      exclusive_req_len_sums.data_ptr<int32_t>()};
  const int64_t num_groups = (ne_n - 1) * ne_k * batch_size;
  sycl_kernel_submit(num_groups * kBlockThreads, kBlockThreads, dpcppGetCurrentQueue(), kernel);
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
  const int total_outputs = static_cast<int>((ne_n - 1) * ne_k * row_indices.numel());
  if (total_outputs <= 0) {
    return;
  }
  ComputeNGramIdsDecodeKernel kernel{
      make_params(
          ne_n,
          ne_k,
          ne_weights,
          ne_mods,
          exclusive_ne_embedder_size_sums,
          ne_token_table,
          row_indices,
          column_starts,
          n_gram_ids,
          eos_token_id),
      total_outputs};
  sycl_kernel_submit(round_up(total_outputs), kBlockThreads, dpcppGetCurrentQueue(), kernel);
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
  const int64_t batch_size = req_lens.numel();
  if (batch_size <= 0) {
    return;
  }
  const bool has_ignore = ignore_tokens.defined() && ignore_tokens.numel() > 0;
  if (has_ignore) {
    CHECK_INPUT(ignore_tokens);
  }
  UpdateTokenTableKernel kernel{
      tokens.data_ptr<int32_t>(),
      ne_token_table.data_ptr<int32_t>(),
      static_cast<int>(ne_token_table.size(1)),
      row_indices.data_ptr<int64_t>(),
      column_starts.data_ptr<int32_t>(),
      req_lens.data_ptr<int32_t>(),
      has_ignore ? static_cast<int>(ignore_tokens.numel()) : 0,
      has_ignore ? ignore_tokens.data_ptr<int32_t>() : nullptr};
  sycl_kernel_submit(batch_size * kBlockThreads, kBlockThreads, dpcppGetCurrentQueue(), kernel);
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
  UpdateTokenTableDecodeKernel kernel{
      batch_size,
      tokens.data_ptr<int32_t>(),
      ne_token_table.data_ptr<int32_t>(),
      static_cast<int>(ne_token_table.size(1)),
      row_indices.data_ptr<int64_t>(),
      column_starts.data_ptr<int32_t>()};
  sycl_kernel_submit(round_up(batch_size), kBlockThreads, dpcppGetCurrentQueue(), kernel);
}
