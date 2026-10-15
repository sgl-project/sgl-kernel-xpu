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
    const int group = item.get_group(0);
    const int first_row = (group / tiles) * kRowsPerTile;
    const int first_col = (group % tiles) * kColumnsPerTile + (item.get_local_id(0) / kSubgroup) * kSubgroup;
    const int col = first_col + item.get_local_id(0) % kSubgroup;
    const int lane = item.get_local_id(0) % kSubgroup;
    float totals[kRowsPerTile] = {};
    // DPAS repeats four rows against each key tile; CuTe maps its B operand into VNNI order.
    Dpas::BVector keys[kDim / kSubgroup];
    for (int step = 0; step < kDim / kSubgroup; ++step) {
      uint16_t key[kSubgroup];
      for (int j = 0; j < kSubgroup; ++j) {
        const int coordinate = DpasTraits::BLayout{}(lane, j);
        const int column = first_col + coordinate % kSubgroup;
        const int dimension = step * kSubgroup + coordinate / kSubgroup;
        key[j] = column < columns ? sycl::bit_cast<uint16_t>(k[column * kDim + dimension]) : 0;
      }
      __builtin_memcpy(&keys[step], key, sizeof(key));
    }

    for (int head = 0; head < kHeads; ++head) {
      Dpas::DVector score{};
      for (int step = 0; step < kDim / kSubgroup; ++step) {
        uint16_t query[kRowsPerTile];
        for (int row = 0; row < kRowsPerTile; ++row) {
          const int coordinate = DpasTraits::ALayout{}(lane, row);
          const int query_row = first_row + coordinate % kRowsPerTile;
          query[row] = query_row < rows
                           ? sycl::bit_cast<uint16_t>(
                                 q[(query_row * kHeads + head) * kDim + step * kSubgroup + coordinate / kRowsPerTile])
                           : 0;
        }
        Dpas::AVector a;
        __builtin_memcpy(&a, query, sizeof(query));
        Dpas::fma(score, a, keys[step], score);
      }
      for (int j = 0; j < kRowsPerTile; ++j) {
        const int coordinate = DpasTraits::CLayout{}(lane, j);
        totals[coordinate % kRowsPerTile] += sycl::fmax(score[j], 0.0f);
      }
    }

    for (int row = 0; row < kRowsPerTile; ++row) {
      const int r = first_row + row;
      if (r < rows && col < columns)
        logits[static_cast<int64_t>(r) * columns + col] =
            col >= starts[r] && col < ends[r] ? totals[row] * inv_scale : -INFINITY;
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
    const int group = item.get_group(0);
    const int row = group / tiles;
    const int first_col = (group % tiles) * kColumnsPerTile + (item.get_local_id(0) / kSubgroup) * kSubgroup;
    const int col = first_col + item.get_local_id(0) % kSubgroup;
    const int lane = item.get_local_id(0) % kSubgroup;
    Dpas::DVector score{};
    for (int step = 0; step < kDim / kSubgroup; ++step) {
      uint16_t query[kHeads];
      for (int head = 0; head < kHeads; ++head) {
        const int coordinate = DpasTraits::ALayout{}(lane, head);
        query[head] = sycl::bit_cast<uint16_t>(
            q[(row * kHeads + coordinate % kHeads) * kDim + step * kSubgroup + coordinate / kHeads]);
      }
      Dpas::AVector a;
      __builtin_memcpy(&a, query, sizeof(query));
      uint16_t key[kSubgroup];
      // Gather paged keys using the same VNNI layout as packed prefill keys.
      for (int j = 0; j < kSubgroup; ++j) {
        const int coordinate = DpasTraits::BLayout{}(lane, j);
        const int column = first_col + coordinate % kSubgroup;
        const int logical_page = column / page_size;
        const int page = column < capacity && logical_page < table_width ? pages[row * table_width + logical_page] : -1;
        const bool valid = column < columns && column < lengths[row] && page >= 0 && page < num_pages;
        const int64_t index = (static_cast<int64_t>(page) * page_size + column % page_size) * kDim + step * kSubgroup +
                              coordinate / kSubgroup;
        key[j] = valid ? sycl::bit_cast<uint16_t>(cache[index]) : 0;
      }
      Dpas::BVector b;
      __builtin_memcpy(&b, key, sizeof(key));
      Dpas::fma(score, a, b, score);
    }
    if (col < columns) {
      float total = 0.0f;
      for (int head = 0; head < kHeads; ++head)
        total += sycl::fmax(score[head], 0.0f);
      const int page = col < capacity ? pages[row * table_width + col / page_size] : -1;
      const bool valid = col < lengths[row] && col < capacity && page >= 0 && page < num_pages;
      logits[static_cast<int64_t>(row) * columns + col] = valid ? total * inv_scale : -INFINITY;
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
