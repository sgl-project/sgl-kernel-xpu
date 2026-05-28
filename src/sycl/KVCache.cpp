#include <ATen/ATen.h>

#include "Utils.h"
#include "comm/General.h"

namespace at::native::xpu {

template <typename scalar_t>
struct StoreCacheKernel {
  void operator()(sycl::nd_item<1> item) const {
    int64_t token_id = static_cast<int64_t>(item.get_group(0));
    int64_t local_id = static_cast<int64_t>(item.get_local_id(0));
    int64_t local_range = static_cast<int64_t>(item.get_local_range(0));

    int64_t cache_idx = indices_[token_id];
    if (cache_idx < 0) return;

    scalar_t* k_src = k_ + token_id * row_dim_;
    scalar_t* v_src = v_ + token_id * row_dim_;
    scalar_t* k_dst = k_cache_ + cache_idx * row_dim_;
    scalar_t* v_dst = v_cache_ + cache_idx * row_dim_;

    // Vectorized copy: 4 elements at a time (8 bytes for bf16/fp16)
    constexpr int64_t vec_width = sizeof(uint64_t) / sizeof(scalar_t);
    int64_t row_dim_vec = (row_dim_ / vec_width) * vec_width;

    auto* k_src_vec = reinterpret_cast<const uint64_t*>(k_src);
    auto* v_src_vec = reinterpret_cast<const uint64_t*>(v_src);
    auto* k_dst_vec = reinterpret_cast<uint64_t*>(k_dst);
    auto* v_dst_vec = reinterpret_cast<uint64_t*>(v_dst);
    int64_t vec_count = row_dim_vec / vec_width;

    for (int64_t i = local_id; i < vec_count; i += local_range) {
      k_dst_vec[i] = k_src_vec[i];
      v_dst_vec[i] = v_src_vec[i];
    }
    // Handle remaining elements
    for (int64_t i = row_dim_vec + local_id; i < row_dim_; i += local_range) {
      k_dst[i] = k_src[i];
      v_dst[i] = v_src[i];
    }
  }

  scalar_t* k_;
  scalar_t* v_;
  scalar_t* k_cache_;
  scalar_t* v_cache_;
  int64_t* indices_;
  int64_t row_dim_;
};

void store_cache_xpu(at::Tensor& k, at::Tensor& v, at::Tensor& k_cache, at::Tensor& v_cache, at::Tensor& indices) {
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(k_cache);
  CHECK_DEVICE(v_cache);
  CHECK_DEVICE(indices);

  CHECK_CONTIGUOUS(k);
  CHECK_CONTIGUOUS(v);
  CHECK_CONTIGUOUS(k_cache);
  CHECK_CONTIGUOUS(v_cache);
  CHECK_CONTIGUOUS(indices);

  TORCH_CHECK(k.dim() == 2, "k must be 2D [num_tokens, row_dim]");
  TORCH_CHECK(v.dim() == 2, "v must be 2D [num_tokens, row_dim]");
  TORCH_CHECK(k_cache.dim() == 2, "k_cache must be 2D [cache_size, row_dim]");
  TORCH_CHECK(v_cache.dim() == 2, "v_cache must be 2D [cache_size, row_dim]");
  TORCH_CHECK(indices.dim() == 1, "indices must be 1D [num_tokens]");

  TORCH_CHECK(v.sizes() == k.sizes(), "v shape must match k shape");
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "v_cache shape must match k_cache shape");
  TORCH_CHECK(k.size(1) == k_cache.size(1), "k row_dim must match k_cache row_dim");
  TORCH_CHECK(indices.size(0) == k.size(0), "indices length must match num_tokens");

  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(k.dtype() == v.dtype(), "k and v must have the same dtype");
  TORCH_CHECK(k.dtype() == k_cache.dtype(), "k and k_cache must have the same dtype");
  TORCH_CHECK(k.dtype() == v_cache.dtype(), "k and v_cache must have the same dtype");

  int64_t num_tokens = k.size(0);
  int64_t row_dim = k.size(1);

  if (num_tokens == 0) return;

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t group_size = std::min<int64_t>(std::min<int64_t>(row_dim, 512), max_wg_size);

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, k.scalar_type(), "store_cache_xpu", [&]() {
        auto cgf = DPCPP_Q_CGF(cgh) {
          StoreCacheKernel<scalar_t> kernel = {
              .k_ = k.data_ptr<scalar_t>(),
              .v_ = v.data_ptr<scalar_t>(),
              .k_cache_ = k_cache.data_ptr<scalar_t>(),
              .v_cache_ = v_cache.data_ptr<scalar_t>(),
              .indices_ = indices.data_ptr<int64_t>(),
              .row_dim_ = row_dim,
          };
          cgh.parallel_for<decltype(kernel)>(
              sycl::nd_range<1>(sycl::range<1>(num_tokens * group_size), sycl::range<1>(group_size)), kernel);
        };
        dpcppGetCurrentQueue().submit(cgf);
      });
}

}  // namespace at::native::xpu
