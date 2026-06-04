#include <ATen/ATen.h>

#include <cstdint>

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

    // Source rows are addressed with the tensor's actual row stride
    // (k_row_stride_ / v_row_stride_), so a non-contiguous K/V whose rows are
    // spaced apart in a wider buffer (e.g. a per-head slice [tokens, 1, dim] of
    // a [tokens, n_heads, dim] tensor) is handled directly — no host-side copy.
    // The destination cache is always dense, so its row stride is row_dim_.
    // Within a row, elements are contiguous (stride(1) == 1 is enforced on host).
    scalar_t* k_src = k_ + token_id * k_row_stride_;
    scalar_t* v_src = v_ + token_id * v_row_stride_;
    scalar_t* k_dst = k_cache_ + cache_idx * row_dim_;
    scalar_t* v_dst = v_cache_ + cache_idx * row_dim_;

    // K/V are a pure copy. For 2-byte scalar_t (bf16/half) we move 16 bytes at a
    // time as one Intel Xe LSC OWord message via sycl::vec<uint32_t, 4>, using the
    // native sycl::vec::load/store(offset, ptr) API (no memcpy) — offset is in units
    // of the 16-byte pack, ptr is the reinterpreted uint32_t row base. This mirrors
    // the B1 V-write idiom in Rope.cpp and src/sycl/merge_states.cpp:106. A native
    // sycl::vec<scalar_t, 8> would hit the c10::BFloat16/Half element-type-trait
    // mismatch and (measured on the rope kernel) does not coalesce to one OWord.
    //
    // vec_count_ is computed on the host: it is row_dim_ / vec_width when every
    // row base is 16-byte aligned (so the OWord load/store is legal), and 0
    // otherwise. When it is 0 the vectorized loop is skipped and the scalar loop
    // below copies the whole row — correctness never depends on alignment.
    using pack_t = sycl::vec<uint32_t, 4>;
    constexpr int64_t vec_width = sizeof(pack_t) / sizeof(scalar_t);

    auto* k_src_p = reinterpret_cast<uint32_t*>(k_src);
    auto* v_src_p = reinterpret_cast<uint32_t*>(v_src);
    auto* k_dst_p = reinterpret_cast<uint32_t*>(k_dst);
    auto* v_dst_p = reinterpret_cast<uint32_t*>(v_dst);
    for (int64_t i = local_id; i < vec_count_; i += local_range) {
      pack_t kp, vp;
      kp.load(i, k_src_p);
      vp.load(i, v_src_p);
      kp.store(i, k_dst_p);
      vp.store(i, v_dst_p);
    }
    for (int64_t i = vec_count_ * vec_width + local_id; i < row_dim_; i += local_range) {
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
  int64_t k_row_stride_;
  int64_t v_row_stride_;
  int64_t vec_count_;
};

void store_cache(at::Tensor& k, at::Tensor& v, at::Tensor& k_cache, at::Tensor& v_cache, at::Tensor& indices) {
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(k_cache);
  CHECK_DEVICE(v_cache);
  CHECK_DEVICE(indices);

  // k/v may be non-contiguous across rows (e.g. a per-head slice of a wider
  // [num_tokens, n_heads, head_dim] tensor): only the inner row must be
  // contiguous so the 16-byte vectorized copy is valid. The dense cache buffers
  // and indices are still required fully contiguous.
  CHECK_CONTIGUOUS(k_cache);
  CHECK_CONTIGUOUS(v_cache);
  CHECK_CONTIGUOUS(indices);

  TORCH_CHECK(k.dim() == 2, "k must be 2D [num_tokens, row_dim]");
  TORCH_CHECK(v.dim() == 2, "v must be 2D [num_tokens, row_dim]");
  TORCH_CHECK(k_cache.dim() == 2, "k_cache must be 2D [cache_size, row_dim]");
  TORCH_CHECK(v_cache.dim() == 2, "v_cache must be 2D [cache_size, row_dim]");
  TORCH_CHECK(indices.dim() == 1, "indices must be 1D [num_tokens]");

  TORCH_CHECK(k.stride(1) == 1, "k rows must be contiguous (k.stride(1) == 1)");
  TORCH_CHECK(v.stride(1) == 1, "v rows must be contiguous (v.stride(1) == 1)");

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
  int64_t k_row_stride = k.stride(0);
  int64_t v_row_stride = v.stride(0);

  if (num_tokens == 0) return;

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t group_size = std::min<int64_t>(std::min<int64_t>(row_dim, 512), max_wg_size);

  SYCL_DISPATCH_FLOATING_TYPES(at::ScalarType::Half, at::ScalarType::BFloat16, k.scalar_type(), "store_cache", [&]() {
    // The 16-byte OWord load/store is only legal when every row base is
    // 16-byte aligned. Row bases are at multiples of the row stride (source) or
    // row_dim (dense cache), plus the tensor's own base offset. Enable the
    // vectorized path only when all four strides keep every row base aligned;
    // otherwise vec_count = 0 and the scalar loop copies the whole row.
    constexpr int64_t vec_width = 16 / sizeof(scalar_t);
    auto aligned = [](int64_t elems) { return (elems % vec_width) == 0; };
    int64_t vec_count = (aligned(k_row_stride) && aligned(v_row_stride) && aligned(row_dim) &&
                         (reinterpret_cast<uintptr_t>(k.data_ptr<scalar_t>()) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(v.data_ptr<scalar_t>()) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(k_cache.data_ptr<scalar_t>()) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(v_cache.data_ptr<scalar_t>()) % 16 == 0))
                            ? row_dim / vec_width
                            : 0;
    auto cgf = DPCPP_Q_CGF(cgh) {
      StoreCacheKernel<scalar_t> kernel = {
          .k_ = k.data_ptr<scalar_t>(),
          .v_ = v.data_ptr<scalar_t>(),
          .k_cache_ = k_cache.data_ptr<scalar_t>(),
          .v_cache_ = v_cache.data_ptr<scalar_t>(),
          .indices_ = indices.data_ptr<int64_t>(),
          .row_dim_ = row_dim,
          .k_row_stride_ = k_row_stride,
          .v_row_stride_ = v_row_stride,
          .vec_count_ = vec_count,
      };
      cgh.parallel_for<decltype(kernel)>(
          sycl::nd_range<1>(sycl::range<1>(num_tokens * group_size), sycl::range<1>(group_size)), kernel);
    };
    dpcppGetCurrentQueue().submit(cgf);
  });
}

}  // namespace at::native::xpu
