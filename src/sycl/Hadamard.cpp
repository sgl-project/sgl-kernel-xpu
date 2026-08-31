#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace at::native::xpu {

namespace {

constexpr int kMinLogN = 3;

constexpr int hadamard_ceil_log2(int val) {
  int log = 0;
  int p = 1;
  while (p < val) {
    p <<= 1;
    ++log;
  }
  return log;
}

template <typename T, int kLogN_, int kNThreads_>
struct FastHadamardTraits {
  using input_t = T;
  static constexpr int kLogN = kLogN_;
  static constexpr int kN = 1 << kLogN;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kNBytes = sizeof(T);
  static_assert(kNBytes == 2 || kNBytes == 4, "hadamard supports fp16/bf16/fp32 only");
  static constexpr int kNElts = (kNBytes == 4) ? 4 : 8;
  static_assert(kN % (kNElts * kNThreads) == 0, "N must be a multiple of kNElts * kNThreads");
  static constexpr int kNChunks = kN / (kNElts * kNThreads);
  static constexpr int kSlmFloats = kN;
};

template <typename Traits>
class FastHadamardKernel {
 public:
  using T = typename Traits::input_t;
  static constexpr int kN = Traits::kN;
  static constexpr int kLogN = Traits::kLogN;
  static constexpr int kNThreads = Traits::kNThreads;
  static constexpr int kNElts = Traits::kNElts;
  static constexpr int kNChunks = Traits::kNChunks;

  FastHadamardKernel(
      const T* x,
      T* out,
      int64_t x_batch_stride,
      int64_t out_batch_stride,
      float scale,
      ::sycl::local_accessor<float, 1> smem)
      : x_(x),
        out_(out),
        x_batch_stride_(x_batch_stride),
        out_batch_stride_(out_batch_stride),
        scale_(scale),
        smem_(smem) {}

  void operator()(::sycl::nd_item<1> item) const {
    const int batch_id = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));

    float* smem_ptr = smem_.template get_multi_ptr<::sycl::access::decorated::no>().get();

    const T* xp = x_ + batch_id * x_batch_stride_;
    T* op = out_ + batch_id * out_batch_stride_;

#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
      const int base = (c * kNThreads + tid) * kNElts;
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        smem_ptr[base + i] = static_cast<float>(xp[base + i]);
      }
    }

    ::sycl::group_barrier(item.get_group());

#pragma unroll
    for (int s = 0; s < kLogN; ++s) {
      const int h = 1 << s;
      const int mask_lo = h - 1;
      for (int k = tid; k < kN / 2; k += kNThreads) {
        const int i = ((k & ~mask_lo) << 1) | (k & mask_lo);
        const int j = i | h;
        const float a = smem_ptr[i];
        const float b = smem_ptr[j];
        smem_ptr[i] = a + b;
        smem_ptr[j] = a - b;
      }
      ::sycl::group_barrier(item.get_group());
    }

#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
      const int base = (c * kNThreads + tid) * kNElts;
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        op[base + i] = static_cast<T>(smem_ptr[base + i] * scale_);
      }
    }
  }

 private:
  const T* x_;
  T* out_;
  int64_t x_batch_stride_;
  int64_t out_batch_stride_;
  float scale_;
  ::sycl::local_accessor<float, 1> smem_;
};

template <typename T, int kLogN, int kNThreads>
inline void hadamard_launch(
    ::sycl::queue& q,
    const T* x,
    T* out,
    int64_t batch,
    int64_t x_batch_stride,
    int64_t out_batch_stride,
    float scale) {
  using Traits = FastHadamardTraits<T, kLogN, kNThreads>;
  q.submit([&](::sycl::handler& cgh) {
    ::sycl::local_accessor<float, 1> smem(::sycl::range<1>(Traits::kSlmFloats), cgh);
    cgh.parallel_for(
        ::sycl::nd_range<1>(::sycl::range<1>(static_cast<size_t>(batch) * kNThreads), ::sycl::range<1>(kNThreads)),
        FastHadamardKernel<Traits>(x, out, x_batch_stride, out_batch_stride, scale, smem));
  });
}

#define _LAUNCH(LOGN) \
  hadamard_launch<T, LOGN, hadamard_nthreads<T, LOGN>()>(q, x, out, batch, x_batch_stride, out_batch_stride, scale)

// Threads per workgroup: enough to cover kN elements at kNElts per thread, capped at 256.
template <typename T, int kLogN>
constexpr int hadamard_nthreads() {
  constexpr int kNElts = (sizeof(T) == 4) ? 4 : 8;
  constexpr int kNThreads = (1 << kLogN) / kNElts;
  return kNThreads < 256 ? kNThreads : 256;
}

template <typename T>
inline void hadamard_dispatch(
    ::sycl::queue& q,
    const T* x,
    T* out,
    int64_t batch,
    int64_t x_batch_stride,
    int64_t out_batch_stride,
    int log_N,
    float scale) {
  switch (log_N) {
    case 3:
      _LAUNCH(3);
      break;
    case 4:
      _LAUNCH(4);
      break;
    case 5:
      _LAUNCH(5);
      break;
    case 6:
      _LAUNCH(6);
      break;
    case 7:
      _LAUNCH(7);
      break;
    case 8:
      _LAUNCH(8);
      break;
    case 9:
      _LAUNCH(9);
      break;
    case 10:
      _LAUNCH(10);
      break;
    case 11:
      _LAUNCH(11);
      break;
    case 12:
      _LAUNCH(12);
      break;
    case 13:
      _LAUNCH(13);
      break;
    case 14:
      _LAUNCH(14);
      break;
    case 15:
      _LAUNCH(15);
      break;
    default:
      TORCH_CHECK(false, "hadamard_transform: unsupported log_N=", log_N);
  }
#undef _LAUNCH
}

}  // anonymous namespace

SGL_KERNEL_EXPORT at::Tensor hadamard_transform(const at::Tensor& input, double scale) {
  CHECK_INPUT(input);
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float || input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16,
      "hadamard_transform: unsupported dtype ",
      input.scalar_type());

  const int64_t dim_og = input.size(-1);
  TORCH_CHECK(dim_og > 0 && dim_og <= 32768, "hadamard_transform: last dim must be in [1, 32768], got ", dim_og);

  const auto shapes_og = input.sizes().vec();
  at::Tensor x_flat = input.view({-1, dim_og});

  const int log_N = std::max(kMinLogN, hadamard_ceil_log2(static_cast<int>(dim_og)));
  const int64_t padded_dim = int64_t{1} << log_N;
  if (padded_dim != dim_og) {
    x_flat = at::constant_pad_nd(x_flat, {0, padded_dim - dim_og}, 0);
  }

  at::Tensor out_flat = at::empty_like(x_flat);
  const int64_t batch = x_flat.size(0);

  if (batch > 0) {
    const int64_t x_batch_stride = x_flat.stride(0);
    const int64_t out_batch_stride = out_flat.stride(0);
    const float scale_f = static_cast<float>(scale);

    auto& q = dpcppGetCurrentQueue();

    SYCL_DISPATCH_FLOATING_TYPES_AND3(
        at::ScalarType::Float,
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        x_flat.scalar_type(),
        "hadamard_transform",
        [&]() {
          hadamard_dispatch<scalar_t>(
              q,
              reinterpret_cast<const scalar_t*>(x_flat.data_ptr()),
              reinterpret_cast<scalar_t*>(out_flat.data_ptr()),
              batch,
              x_batch_stride,
              out_batch_stride,
              log_N,
              scale_f);
        });
  }

  at::Tensor out = padded_dim != dim_og ? out_flat.slice(1, 0, dim_og) : out_flat;
  return out.reshape(shapes_og);
}

}  // namespace at::native::xpu
