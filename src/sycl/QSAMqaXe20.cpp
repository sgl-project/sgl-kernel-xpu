#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>

#include <cmath>
#include <cstdint>
#include <cute/tensor.hpp>
#include <limits>
#include <sycl/sycl.hpp>

#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;
using Dpas = cute::XE_DPAS_TT<4, float, cute::bfloat16_t>;
using DpasTraits = cute::MMA_Traits<Dpas>;
// These parameters are for Qwen3.8-Flash-Next only
constexpr int kHeads = 4;
constexpr int kDim = 128;
constexpr int kSubgroup = 16;
constexpr int kThreads = 128;
constexpr int kRowsPerTile = 4;
constexpr int kColumnsPerTile = kThreads;

struct QSAMqaPrefillKernel {
  const bf16* q;
  const bf16* k;
  const int32_t* starts;
  const int32_t* ends;
  float* logits;
  int rows;
  int columns;
  int tiles;
  float inv_scale;

  [[sycl::reqd_sub_group_size(kSubgroup)]]
  void operator()(sycl::nd_item<1> item) const {
    const int64_t group = item.get_group(0);
    const int64_t first_row = (group / tiles) * kRowsPerTile;
    const int64_t first_col = (group % tiles) * kColumnsPerTile + (item.get_local_id(0) / kSubgroup) * kSubgroup;
    const int64_t col = first_col + item.get_local_id(0) % kSubgroup;
    const int lane = item.get_local_id(0) % kSubgroup;
    float totals[kRowsPerTile] = {};
    const auto mma = cute::make_tiled_mma(cute::MMA_Atom<Dpas>{});
    const auto thread_mma = mma.get_thread_slice(lane);
    const auto a_shape = cute::make_shape(cute::Int<kRowsPerTile>{}, cute::Int<kSubgroup>{});
    const auto b_shape = cute::make_shape(cute::Int<kSubgroup>{}, cute::Int<kSubgroup>{});
    const auto a_coords = thread_mma.partition_A(cute::make_identity_tensor(a_shape));
    const auto b_coords = thread_mma.partition_B(cute::make_identity_tensor(b_shape));
    cute::bfloat16_t keys[kDim / kSubgroup][kSubgroup];
    for (int step = 0; step < kDim / kSubgroup; ++step) {
      const int64_t key_start = first_col < columns ? first_col : 0;
      const auto g_keys = cute::make_tensor(
          cute::make_gmem_ptr(k + key_start * kDim + step * kSubgroup),
          cute::make_layout(b_shape, cute::make_stride(cute::Int<kDim>{}, cute::Int<1>{})));
      const auto thread_keys = thread_mma.partition_B(g_keys);
      for (int j = 0; j < kSubgroup; ++j) {
        const int64_t column = first_col + cute::get<0>(b_coords(j));
        keys[step][j] = column < columns ? sycl::bit_cast<cute::bfloat16_t>(thread_keys(j)) : cute::bfloat16_t{};
      }
    }

    for (int head = 0; head < kHeads; ++head) {
      float score[kRowsPerTile] = {};
      auto c = cute::make_tensor(cute::make_rmem_ptr(score), cute::make_layout(cute::Shape<cute::Int<kRowsPerTile>>{}));
      for (int step = 0; step < kDim / kSubgroup; ++step) {
        const auto g_query = cute::make_tensor(
            cute::make_gmem_ptr(q + (first_row * kHeads + head) * kDim + step * kSubgroup),
            cute::make_layout(a_shape, cute::make_stride(cute::Int<kHeads * kDim>{}, cute::Int<1>{})));
        const auto thread_query = thread_mma.partition_A(g_query);
        cute::bfloat16_t query[kRowsPerTile];
        for (int row = 0; row < kRowsPerTile; ++row) {
          const int64_t query_row = first_row + cute::get<0>(a_coords(row));
          query[row] = query_row < rows ? sycl::bit_cast<cute::bfloat16_t>(thread_query(row)) : cute::bfloat16_t{};
        }
        auto a =
            cute::make_tensor(cute::make_rmem_ptr(query), cute::make_layout(cute::Shape<cute::Int<kRowsPerTile>>{}));
        auto b =
            cute::make_tensor(cute::make_rmem_ptr(keys[step]), cute::make_layout(cute::Shape<cute::Int<kSubgroup>>{}));
        cute::gemm(cute::MMA_Atom<Dpas>{}, a, b, c);
      }
      for (int j = 0; j < kRowsPerTile; ++j) {
        const int coordinate = DpasTraits::CLayout{}(lane, j);
        totals[coordinate % kRowsPerTile] += sycl::fmax(score[j], 0.0f);
      }
    }

    for (int row = 0; row < kRowsPerTile; ++row) {
      const int64_t r = first_row + row;
      if (r < rows && col < columns)
        logits[r * columns + col] = col >= starts[r] && col < ends[r] ? totals[row] * inv_scale : -INFINITY;
    }
  }
};

struct QSAMqaDecodeKernel {
  const bf16* q;
  const bf16* cache;
  const int32_t* pages;
  const int32_t* lengths;
  float* logits;
  int columns;
  int capacity;
  int page_size;
  int table_width;
  int num_pages;
  int tiles;
  float inv_scale;

  [[sycl::reqd_sub_group_size(kSubgroup)]]
  void operator()(sycl::nd_item<1> item) const {
    const int64_t group = item.get_group(0);
    const int64_t row = group / tiles;
    const int64_t first_col = (group % tiles) * kColumnsPerTile + (item.get_local_id(0) / kSubgroup) * kSubgroup;
    const int64_t col = first_col + item.get_local_id(0) % kSubgroup;
    const int lane = item.get_local_id(0) % kSubgroup;
    float score[kHeads] = {};
    auto c = cute::make_tensor(cute::make_rmem_ptr(score), cute::make_layout(cute::Shape<cute::Int<kHeads>>{}));
    for (int step = 0; step < kDim / kSubgroup; ++step) {
      cute::bfloat16_t query[kHeads];
      for (int head = 0; head < kHeads; ++head) {
        const int coordinate = DpasTraits::ALayout{}(lane, head);
        query[head] = sycl::bit_cast<cute::bfloat16_t>(
            q[(row * kHeads + coordinate % kHeads) * kDim + step * kSubgroup + coordinate / kHeads]);
      }
      cute::bfloat16_t key[kSubgroup];
      // Gather paged keys using the same VNNI layout as packed prefill keys.
      for (int j = 0; j < kSubgroup; ++j) {
        const int coordinate = DpasTraits::BLayout{}(lane, j);
        const int64_t column = first_col + coordinate % kSubgroup;
        const int64_t logical_page = column / page_size;
        const int page = column < capacity && logical_page < table_width ? pages[row * table_width + logical_page] : -1;
        const bool valid = column < columns && column < lengths[row] && page >= 0 && page < num_pages;
        const int64_t index = (static_cast<int64_t>(page) * page_size + column % page_size) * kDim + step * kSubgroup +
                              coordinate / kSubgroup;
        key[j] = valid ? sycl::bit_cast<cute::bfloat16_t>(cache[index]) : cute::bfloat16_t{};
      }
      auto a = cute::make_tensor(cute::make_rmem_ptr(query), cute::make_layout(cute::Shape<cute::Int<kHeads>>{}));
      auto b = cute::make_tensor(cute::make_rmem_ptr(key), cute::make_layout(cute::Shape<cute::Int<kSubgroup>>{}));
      cute::gemm(cute::MMA_Atom<Dpas>{}, a, b, c);
    }
    if (col < columns) {
      float total = 0.0f;
      for (int head = 0; head < kHeads; ++head)
        total += sycl::fmax(score[head], 0.0f);
      const int page = col < capacity ? pages[row * table_width + col / page_size] : -1;
      const bool valid = col < lengths[row] && col < capacity && page >= 0 && page < num_pages;
      logits[row * columns + col] = valid ? total * inv_scale : -INFINITY;
    }
  }
};

void check_q(const at::Tensor& q) {
  CHECK_INPUT(q);
  TORCH_CHECK(
      q.scalar_type() == at::kBFloat16 && q.dim() == 3 && q.size(1) == kHeads && q.size(2) == kDim,
      "QSA MQA requires BF16 q [rows, 4, 128]");
  TORCH_CHECK(q.size(0) <= std::numeric_limits<int>::max(), "QSA MQA row count exceeds int32");
}

float inverse_scale(std::optional<double> score_scale) {
  const double scale = score_scale.value_or(0.0) == 0.0 ? std::sqrt(kDim) : *score_scale;
  TORCH_CHECK(std::isfinite(scale), "QSA MQA score scale must be finite");
  return static_cast<float>(1.0 / scale);
}

void check_same_device(const at::Tensor& q, const at::Tensor& tensor) {
  CHECK_INPUT(tensor);
  TORCH_CHECK(q.device() == tensor.device(), "QSA MQA tensors must be on the same device");
}

}  // namespace

SGL_KERNEL_EXPORT at::Tensor qsa_mqa_prefill(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& row_starts,
    const at::Tensor& row_ends,
    std::optional<double> score_scale) {
  check_q(q);
  for (const auto* tensor : {&k, &row_starts, &row_ends})
    check_same_device(q, *tensor);
  TORCH_CHECK(
      k.scalar_type() == at::kBFloat16 && k.dim() == 3 && k.size(1) == 1 && k.size(2) == kDim,
      "QSA MQA prefill requires BF16 k [columns, 1, 128]");
  TORCH_CHECK(
      row_starts.scalar_type() == at::kInt && row_ends.scalar_type() == at::kInt && row_starts.dim() == 1 &&
          row_ends.dim() == 1 && row_starts.numel() == q.size(0) && row_ends.numel() == q.size(0),
      "QSA MQA prefill requires int32 row ranges [rows]");
  TORCH_CHECK(k.size(0) <= std::numeric_limits<int>::max(), "QSA MQA column count exceeds int32");
  const float scale = inverse_scale(score_scale);
  c10::DeviceGuard guard(q.device());
  auto out = at::empty({q.size(0), k.size(0)}, q.options().dtype(at::kFloat));
  const int rows = static_cast<int>(q.size(0));
  const int columns = static_cast<int>(k.size(0));
  if (!rows || !columns) return out;
  const int tiles = (columns - 1) / kColumnsPerTile + 1;
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(static_cast<size_t>((rows - 1) / kRowsPerTile + 1) * tiles * kThreads, kThreads),
        QSAMqaPrefillKernel{
            reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(k.data_ptr<at::BFloat16>()),
            row_starts.data_ptr<int32_t>(),
            row_ends.data_ptr<int32_t>(),
            out.data_ptr<float>(),
            rows,
            columns,
            tiles,
            scale});
  });
  return out;
}

SGL_KERNEL_EXPORT at::Tensor qsa_mqa_decode(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& page_table,
    const at::Tensor& context_lens,
    int64_t max_model_len,
    std::optional<double> score_scale) {
  check_q(q);
  for (const auto* tensor : {&k_cache, &page_table, &context_lens})
    check_same_device(q, *tensor);
  TORCH_CHECK(
      k_cache.scalar_type() == at::kBFloat16 && k_cache.dim() == 4 && k_cache.size(2) == 1 && k_cache.size(3) == kDim,
      "QSA MQA decode requires BF16 k_cache [pages, page_size, 1, 128]");
  TORCH_CHECK(
      page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.size(0) == q.size(0) &&
          context_lens.scalar_type() == at::kInt && context_lens.dim() == 1 && context_lens.numel() == q.size(0),
      "QSA MQA decode requires int32 page_table [rows, pages] and context_lens [rows]");
  TORCH_CHECK(
      k_cache.size(0) <= std::numeric_limits<int>::max() && k_cache.size(1) > 0 &&
          k_cache.size(1) <= std::numeric_limits<int>::max() && page_table.size(1) <= std::numeric_limits<int>::max() &&
          max_model_len >= 0 && max_model_len <= std::numeric_limits<int>::max() &&
          page_table.size(1) * k_cache.size(1) <= std::numeric_limits<int>::max(),
      "QSA MQA decode cache dimensions or output width are unsupported");
  const float scale = inverse_scale(score_scale);
  c10::DeviceGuard guard(q.device());
  auto out = at::empty({q.size(0), max_model_len}, q.options().dtype(at::kFloat));
  const int rows = static_cast<int>(q.size(0));
  const int columns = static_cast<int>(max_model_len);
  if (!rows || !columns) return out;
  TORCH_CHECK(k_cache.size(0) > 0, "QSA MQA decode requires at least one cache page");
  const int tiles = (columns - 1) / kColumnsPerTile + 1;
  const int page_size = static_cast<int>(k_cache.size(1));
  const int width = static_cast<int>(page_table.size(1));
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(rows) * tiles * kThreads, kThreads),
        QSAMqaDecodeKernel{
            reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(k_cache.data_ptr<at::BFloat16>()),
            page_table.data_ptr<int32_t>(),
            context_lens.data_ptr<int32_t>(),
            out.data_ptr<float>(),
            columns,
            width * page_size,
            page_size,
            width,
            static_cast<int>(k_cache.size(0)),
            tiles,
            scale});
  });
  return out;
}
