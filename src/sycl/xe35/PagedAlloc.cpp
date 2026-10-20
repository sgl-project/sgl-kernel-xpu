#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include "sgl_kernel/xe35/memory_ops.h"

#include <algorithm>
#include <sycl/sycl.hpp>

#include "../SYCLHelpers.h"

namespace at::native::xpu {
namespace {

constexpr int kWorkgroup = 256;
constexpr int kDecodeWorkgroup = 32;

inline int64_t ceil_pages(int64_t tokens, int64_t page_size) {
  return (tokens + page_size - 1) / page_size;
}

// Positive shifts are compile-time page-size variants; zero is a dynamic
// power-of-two size, and -1 keeps the general division path.
template <int PageShift>
inline int64_t effective_page_size(int64_t page_size) {
  if constexpr (PageShift > 0) {
    return int64_t{1} << PageShift;
  } else {
    return page_size;
  }
}

template <int PageShift>
inline int page_shift(int64_t page_size) {
  if constexpr (PageShift > 0) {
    return PageShift;
  } else if constexpr (PageShift == 0) {
    return sycl::ctz(static_cast<uint64_t>(page_size));
  } else {
    return 0;
  }
}

template <int PageShift>
inline int64_t page_count(int64_t tokens, int64_t page_size, int shift) {
  if constexpr (PageShift >= 0) {
    return (tokens + page_size - 1) >> shift;
  } else {
    return ceil_pages(tokens, page_size);
  }
}

template <int PageShift>
inline int64_t page_index(int64_t position, int64_t page_size, int shift) {
  if constexpr (PageShift >= 0) {
    return position >> shift;
  } else {
    return position / page_size;
  }
}

template <int PageShift>
inline int64_t page_offset(int64_t position, int64_t page_size) {
  if constexpr (PageShift >= 0) {
    return position & (page_size - 1);
  } else {
    return position % page_size;
  }
}

template <typename LastScalar, int TilesPerReq, int PageShift>
struct PagedAllocExtendKernel {
  const int64_t* __restrict__ prefix_lens;
  const int64_t* __restrict__ seq_lens;
  const LastScalar* __restrict__ last_loc;
  const int64_t* __restrict__ free_pages;
  int64_t* __restrict__ out_indices;
  int64_t batch;
  int64_t page_size;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t req = item.get_group(0) / TilesPerReq;
    const int64_t tile = item.get_group(0) % TilesPerReq;
    const int64_t lane = item.get_local_id(0);
    if (req >= batch) {
      return;
    }
    const int64_t size = effective_page_size<PageShift>(page_size);
    const int shift = page_shift<PageShift>(page_size);
    int64_t output_start = 0;
    int64_t new_page_start = 0;
    for (int64_t i = 0; i < req; ++i) {
      output_start += seq_lens[i] - prefix_lens[i];
      new_page_start +=
          page_count<PageShift>(seq_lens[i], size, shift) - page_count<PageShift>(prefix_lens[i], size, shift);
    }
    const int64_t prefix_len = prefix_lens[req];
    const int64_t seq_len = seq_lens[req];
    const int64_t old_boundary = page_count<PageShift>(prefix_len, size, shift) * size;
    for (int64_t offset = tile * kWorkgroup + lane; offset < seq_len - prefix_len; offset += TilesPerReq * kWorkgroup) {
      const int64_t position = prefix_len + offset;
      int64_t location;
      if (position < old_boundary) {
        location = last_loc[req] + 1 + offset;
      } else {
        const int64_t new_page = page_index<PageShift>(position - old_boundary, size, shift);
        location = free_pages[new_page_start + new_page] * size + page_offset<PageShift>(position, size);
      }
      out_indices[output_start + offset] = location;
    }
  }
};

// One launch produces the paged locations and writes the request table.
template <typename LastScalar, int TilesPerReq, int PageShift>
struct PagedAllocExtendAndWriteKernel {
  const int64_t* __restrict__ prefix_lens;
  const int64_t* __restrict__ seq_lens;
  const LastScalar* __restrict__ last_loc;
  const int64_t* __restrict__ free_pages;
  int64_t* __restrict__ out_indices;
  int32_t* __restrict__ req_to_token;
  const int64_t* __restrict__ req_pool_indices;
  const uint64_t* __restrict__ prefix_pointers;
  int64_t batch;
  int64_t page_size;
  int64_t row_stride;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t req = item.get_group(0) / TilesPerReq;
    const int64_t tile = item.get_group(0) % TilesPerReq;
    const int64_t lane = item.get_local_id(0);
    if (req >= batch) return;
    const int64_t size = effective_page_size<PageShift>(page_size);
    const int shift = page_shift<PageShift>(page_size);
    int64_t output_start = 0;
    int64_t new_page_start = 0;
    for (int64_t i = 0; i < req; ++i) {
      output_start += seq_lens[i] - prefix_lens[i];
      new_page_start +=
          page_count<PageShift>(seq_lens[i], size, shift) - page_count<PageShift>(prefix_lens[i], size, shift);
    }
    const int64_t prefix_len = prefix_lens[req];
    const int64_t seq_len = seq_lens[req];
    const int64_t row = req_pool_indices[req] * row_stride;
    if (prefix_len == 0) {
      for (int64_t offset = tile * kWorkgroup + lane; offset < seq_len; offset += TilesPerReq * kWorkgroup) {
        const int64_t location = free_pages[new_page_start + page_index<PageShift>(offset, size, shift)] * size +
                                 page_offset<PageShift>(offset, size);
        out_indices[output_start + offset] = location;
        req_to_token[row + offset] = static_cast<int32_t>(location);
      }
      return;
    }
    const auto* prefix = reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(prefix_pointers[req]));
    for (int64_t pos = tile * kWorkgroup + lane; pos < prefix_len; pos += TilesPerReq * kWorkgroup) {
      req_to_token[row + pos] = static_cast<int32_t>(prefix[pos]);
    }
    const int64_t old_boundary = page_count<PageShift>(prefix_len, size, shift) * size;
    for (int64_t offset = tile * kWorkgroup + lane; offset < seq_len - prefix_len; offset += TilesPerReq * kWorkgroup) {
      const int64_t position = prefix_len + offset;
      int64_t location;
      if (position < old_boundary) {
        location = last_loc[req] + 1 + offset;
      } else {
        location = free_pages[new_page_start + page_index<PageShift>(position - old_boundary, size, shift)] * size +
                   page_offset<PageShift>(position, size);
      }
      out_indices[output_start + offset] = location;
      req_to_token[row + position] = static_cast<int32_t>(location);
    }
  }
};

template <typename LastScalar, int PageShift>
struct PagedAllocDecodeKernel {
  const int64_t* __restrict__ seq_lens;
  const LastScalar* __restrict__ last_loc;
  const int64_t* __restrict__ free_pages;
  int64_t* __restrict__ out_indices;
  int64_t batch;
  int64_t page_size;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t req = item.get_global_id(0);
    if (req >= batch) {
      return;
    }
    const int64_t size = effective_page_size<PageShift>(page_size);
    const int shift = page_shift<PageShift>(page_size);
    int64_t new_page_start = 0;
    for (int64_t i = 0; i < req; ++i) {
      const int64_t seq = seq_lens[i];
      new_page_start += page_count<PageShift>(seq, size, shift) - page_count<PageShift>(seq - 1, size, shift);
    }
    const int64_t seq = seq_lens[req];
    const bool need_page = page_count<PageShift>(seq, size, shift) > page_count<PageShift>(seq - 1, size, shift);
    out_indices[req] = need_page ? free_pages[new_page_start] * size : last_loc[req] + 1;
  }
};

void check_vector(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.is_xpu() && tensor.dim() == 1 && tensor.is_contiguous() && tensor.scalar_type() == at::kLong,
      name,
      " must be a contiguous XPU int64 vector");
}

void check_last_loc(const torch::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_xpu() && tensor.dim() == 1 && tensor.is_contiguous() &&
          (tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong),
      "last_loc must be a contiguous XPU int32 or int64 vector");
}

template <typename LastScalar, int TilesPerReq, int PageShift>
void launch_extend(
    const torch::Tensor& prefix_lens,
    const torch::Tensor& seq_lens,
    const torch::Tensor& last_loc,
    const torch::Tensor& free_pages,
    torch::Tensor& out_indices,
    int64_t page_size,
    sycl::queue& queue) {
  const int64_t batch = prefix_lens.numel();
  PagedAllocExtendKernel<LastScalar, TilesPerReq, PageShift> kernel{
      prefix_lens.data_ptr<int64_t>(),
      seq_lens.data_ptr<int64_t>(),
      last_loc.data_ptr<LastScalar>(),
      free_pages.data_ptr<int64_t>(),
      out_indices.data_ptr<int64_t>(),
      batch,
      page_size};
  sycl_kernel_submit(sycl::range<1>(batch * TilesPerReq * kWorkgroup), sycl::range<1>(kWorkgroup), queue, kernel);
}

template <typename LastScalar, int PageShift>
void dispatch_extend_tiles(
    const torch::Tensor& prefix_lens,
    const torch::Tensor& seq_lens,
    const torch::Tensor& last_loc,
    const torch::Tensor& free_pages,
    torch::Tensor& out_indices,
    int64_t page_size,
    int64_t output_per_req,
    sycl::queue& queue) {
  if (output_per_req >= 8 * kWorkgroup) {
    launch_extend<LastScalar, 16, PageShift>(
        prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, queue);
  } else if (output_per_req >= 2 * kWorkgroup) {
    launch_extend<LastScalar, 4, PageShift>(prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, queue);
  } else {
    launch_extend<LastScalar, 1, PageShift>(prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, queue);
  }
}

template <typename LastScalar, int TilesPerReq, int PageShift>
void launch_extend_and_write(
    const torch::Tensor& prefix_lens,
    const torch::Tensor& seq_lens,
    const torch::Tensor& last_loc,
    const torch::Tensor& free_pages,
    torch::Tensor& out_indices,
    torch::Tensor& req_to_token,
    const torch::Tensor& req_pool_indices,
    const torch::Tensor& prefix_pointers,
    int64_t page_size,
    sycl::queue& queue) {
  const int64_t batch = prefix_lens.numel();
  PagedAllocExtendAndWriteKernel<LastScalar, TilesPerReq, PageShift> kernel{
      prefix_lens.data_ptr<int64_t>(),
      seq_lens.data_ptr<int64_t>(),
      last_loc.data_ptr<LastScalar>(),
      free_pages.data_ptr<int64_t>(),
      out_indices.data_ptr<int64_t>(),
      req_to_token.data_ptr<int32_t>(),
      req_pool_indices.data_ptr<int64_t>(),
      prefix_pointers.data_ptr<uint64_t>(),
      batch,
      page_size,
      req_to_token.stride(0)};
  sycl_kernel_submit(sycl::range<1>(batch * TilesPerReq * kWorkgroup), sycl::range<1>(kWorkgroup), queue, kernel);
}

template <typename LastScalar, int PageShift>
void dispatch_extend_and_write(
    const torch::Tensor& prefix_lens,
    const torch::Tensor& seq_lens,
    const torch::Tensor& last_loc,
    const torch::Tensor& free_pages,
    torch::Tensor& out_indices,
    torch::Tensor& req_to_token,
    const torch::Tensor& req_pool_indices,
    const torch::Tensor& prefix_pointers,
    int64_t page_size,
    sycl::queue& queue) {
  const int64_t batch = prefix_lens.numel();
  const int64_t output_per_req = std::max(out_indices.numel() / batch, req_to_token.size(1));
  if (output_per_req >= 8 * kWorkgroup) {
    launch_extend_and_write<LastScalar, 16, PageShift>(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        req_to_token,
        req_pool_indices,
        prefix_pointers,
        page_size,
        queue);
  } else if (output_per_req >= 2 * kWorkgroup) {
    launch_extend_and_write<LastScalar, 4, PageShift>(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        req_to_token,
        req_pool_indices,
        prefix_pointers,
        page_size,
        queue);
  } else {
    launch_extend_and_write<LastScalar, 1, PageShift>(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        req_to_token,
        req_pool_indices,
        prefix_pointers,
        page_size,
        queue);
  }
}

template <typename LastScalar, int PageShift>
void launch_decode(
    const torch::Tensor& seq_lens,
    const torch::Tensor& last_loc,
    const torch::Tensor& free_pages,
    torch::Tensor& out_indices,
    int64_t page_size,
    sycl::queue& queue) {
  const int64_t batch = seq_lens.numel();
  PagedAllocDecodeKernel<LastScalar, PageShift> kernel{
      seq_lens.data_ptr<int64_t>(),
      last_loc.data_ptr<LastScalar>(),
      free_pages.data_ptr<int64_t>(),
      out_indices.data_ptr<int64_t>(),
      batch,
      page_size};
  sycl_kernel_submit(
      sycl::range<1>(((batch + kDecodeWorkgroup - 1) / kDecodeWorkgroup) * kDecodeWorkgroup),
      sycl::range<1>(kDecodeWorkgroup),
      queue,
      kernel);
}

template <typename Launch>
void dispatch_page_family(int64_t page_size, Launch&& launch) {
  switch (page_size) {
    case 32:
      launch.template operator()<5>();
      break;
    case 64:
      launch.template operator()<6>();
      break;
    case 128:
      launch.template operator()<7>();
      break;
    default:
      if ((page_size & (page_size - 1)) == 0) {
        launch.template operator()<0>();
      } else {
        launch.template operator()<-1>();
      }
  }
}

}  // namespace

void paged_alloc_extend_xe35(
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    int64_t page_size) {
  check_vector(prefix_lens, "prefix_lens");
  check_vector(seq_lens, "seq_lens");
  check_last_loc(last_loc);
  check_vector(free_pages, "free_pages");
  check_vector(out_indices, "out_indices");
  TORCH_CHECK(
      page_size > 0 && seq_lens.numel() == prefix_lens.numel() && last_loc.numel() == prefix_lens.numel(),
      "invalid batch or page_size");
  const int64_t batch = prefix_lens.numel();
  if (batch == 0) {
    return;
  }
  auto queue = at::xpu::getCurrentXPUStream().queue();
  const int64_t output_per_req = out_indices.numel() / batch;
  if (last_loc.scalar_type() == at::kInt) {
    dispatch_page_family(page_size, [&]<int PageShift>() {
      dispatch_extend_tiles<int32_t, PageShift>(
          prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, output_per_req, queue);
    });
  } else {
    dispatch_page_family(page_size, [&]<int PageShift>() {
      dispatch_extend_tiles<int64_t, PageShift>(
          prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, output_per_req, queue);
    });
  }
}

void paged_alloc_extend_and_write_xe35(
    torch::Tensor prefix_lens,
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    torch::Tensor req_to_token,
    torch::Tensor req_pool_indices,
    torch::Tensor prefix_pointers,
    int64_t page_size) {
  check_vector(prefix_lens, "prefix_lens");
  check_vector(seq_lens, "seq_lens");
  check_last_loc(last_loc);
  check_vector(free_pages, "free_pages");
  check_vector(out_indices, "out_indices");
  check_vector(req_pool_indices, "req_pool_indices");
  TORCH_CHECK(
      prefix_pointers.is_xpu() && prefix_pointers.dim() == 1 && prefix_pointers.is_contiguous() &&
          prefix_pointers.scalar_type() == at::kUInt64,
      "prefix_pointers must be a contiguous XPU uint64 vector");
  TORCH_CHECK(
      req_to_token.is_xpu() && req_to_token.dim() == 2 && req_to_token.is_contiguous() &&
          req_to_token.scalar_type() == at::kInt,
      "req_to_token must be a contiguous XPU int32 matrix");
  const int64_t batch = prefix_lens.numel();
  TORCH_CHECK(
      page_size > 0 && seq_lens.numel() == batch && last_loc.numel() == batch && req_pool_indices.numel() == batch &&
          prefix_pointers.numel() == batch,
      "invalid batch or page_size");
  if (!batch) return;
  auto queue = at::xpu::getCurrentXPUStream().queue();
  if (last_loc.scalar_type() == at::kInt) {
    dispatch_page_family(page_size, [&]<int PageShift>() {
      dispatch_extend_and_write<int32_t, PageShift>(
          prefix_lens,
          seq_lens,
          last_loc,
          free_pages,
          out_indices,
          req_to_token,
          req_pool_indices,
          prefix_pointers,
          page_size,
          queue);
    });
  } else {
    dispatch_page_family(page_size, [&]<int PageShift>() {
      dispatch_extend_and_write<int64_t, PageShift>(
          prefix_lens,
          seq_lens,
          last_loc,
          free_pages,
          out_indices,
          req_to_token,
          req_pool_indices,
          prefix_pointers,
          page_size,
          queue);
    });
  }
}

void paged_alloc_decode_xe35(
    torch::Tensor seq_lens,
    torch::Tensor last_loc,
    torch::Tensor free_pages,
    torch::Tensor out_indices,
    int64_t page_size) {
  check_vector(seq_lens, "seq_lens");
  check_last_loc(last_loc);
  check_vector(free_pages, "free_pages");
  check_vector(out_indices, "out_indices");
  TORCH_CHECK(
      page_size > 0 && last_loc.numel() == seq_lens.numel() && out_indices.numel() == seq_lens.numel(),
      "invalid batch or page_size");
  const int64_t batch = seq_lens.numel();
  if (batch == 0) {
    return;
  }
  auto queue = at::xpu::getCurrentXPUStream().queue();
  if (last_loc.scalar_type() == at::kInt) {
    dispatch_page_family(page_size, [&]<int PageShift>() {
      launch_decode<int32_t, PageShift>(seq_lens, last_loc, free_pages, out_indices, page_size, queue);
    });
  } else {
    dispatch_page_family(page_size, [&]<int PageShift>() {
      launch_decode<int64_t, PageShift>(seq_lens, last_loc, free_pages, out_indices, page_size, queue);
    });
  }
}

}  // namespace at::native::xpu
