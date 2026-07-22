#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

namespace {

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
  constexpr bool kIsFp32 = std::is_same_v<T, float>;

#define _LAUNCH(LOGN, NT) hadamard_launch<T, LOGN, NT>(q, x, out, batch, x_batch_stride, out_batch_stride, scale)

  switch (log_N) {
    case 3:
      if constexpr (kIsFp32) {
        _LAUNCH(3, 2);
      } else {
        _LAUNCH(3, 1);
      }
      break;
    case 4:
      if constexpr (kIsFp32) {
        _LAUNCH(4, 4);
      } else {
        _LAUNCH(4, 2);
      }
      break;
    case 5:
      if constexpr (kIsFp32) {
        _LAUNCH(5, 8);
      } else {
        _LAUNCH(5, 4);
      }
      break;
    case 6:
      if constexpr (kIsFp32) {
        _LAUNCH(6, 16);
      } else {
        _LAUNCH(6, 8);
      }
      break;
    case 7:
      if constexpr (kIsFp32) {
        _LAUNCH(7, 32);
      } else {
        _LAUNCH(7, 16);
      }
      break;
    case 8:
      if constexpr (kIsFp32) {
        _LAUNCH(8, 64);
      } else {
        _LAUNCH(8, 32);
      }
      break;
    case 9:
      if constexpr (kIsFp32) {
        _LAUNCH(9, 128);
      } else {
        _LAUNCH(9, 64);
      }
      break;
    case 10:
      if constexpr (kIsFp32) {
        _LAUNCH(10, 256);
      } else {
        _LAUNCH(10, 128);
      }
      break;
    case 11:
      _LAUNCH(11, 256);
      break;
    case 12:
      _LAUNCH(12, 256);
      break;
    case 13:
      _LAUNCH(13, 256);
      break;
    case 14:
      _LAUNCH(14, 256);
      break;
    case 15:
      _LAUNCH(15, 256);
      break;
    default:
      TORCH_CHECK(false, "hadamard_transform: unsupported log_N=", log_N);
  }
#undef _LAUNCH
}

}  // anonymous namespace

void hadamard_transform(at::Tensor& output, const at::Tensor& input, double scale) {
  CHECK_INPUT(output);
  CHECK_INPUT(input);
  TORCH_CHECK(input.is_xpu(), "hadamard_transform: input must be an XPU tensor");
  TORCH_CHECK(output.is_xpu(), "hadamard_transform: output must be an XPU tensor");
  TORCH_CHECK(input.dim() == 2, "hadamard_transform: input must be 2D (batch, dim)");
  TORCH_CHECK(output.dim() == 2, "hadamard_transform: output must be 2D (batch, dim)");
  TORCH_CHECK(
      input.sizes() == output.sizes(),
      "hadamard_transform: input/output shape mismatch: ",
      input.sizes(),
      " vs ",
      output.sizes());
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "hadamard_transform: input/output dtype mismatch");
  TORCH_CHECK(input.stride(-1) == 1, "hadamard_transform: input's last dim must be contiguous");
  TORCH_CHECK(output.stride(-1) == 1, "hadamard_transform: output's last dim must be contiguous");

  const int64_t batch = input.size(0);
  const int64_t dim = input.size(1);

  TORCH_CHECK(dim >= 8, "hadamard_transform: dim must be >= 8");
  TORCH_CHECK(dim <= 32768, "hadamard_transform: dim must be <= 32768");
  TORCH_CHECK((dim & (dim - 1)) == 0, "hadamard_transform: dim must be a power of two");

  if (batch == 0) return;

  const int log_N = hadamard_ceil_log2(static_cast<int>(dim));
  const int64_t x_batch_stride = input.stride(0);
  const int64_t out_batch_stride = output.stride(0);
  const float scale_f = static_cast<float>(scale);

  auto& q = dpcppGetCurrentQueue();

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "hadamard_transform",
      AT_DISPATCH_CASE(at::ScalarType::Float, ([&] {
                         hadamard_dispatch<float>(
                             q,
                             input.data_ptr<float>(),
                             output.data_ptr<float>(),
                             batch,
                             x_batch_stride,
                             out_batch_stride,
                             log_N,
                             scale_f);
                       })) AT_DISPATCH_CASE(at::ScalarType::Half, ([&] {
                                              hadamard_dispatch<::sycl::half>(
                                                  q,
                                                  reinterpret_cast<const ::sycl::half*>(input.data_ptr<at::Half>()),
                                                  reinterpret_cast<::sycl::half*>(output.data_ptr<at::Half>()),
                                                  batch,
                                                  x_batch_stride,
                                                  out_batch_stride,
                                                  log_N,
                                                  scale_f);
                                            }))
          AT_DISPATCH_CASE(at::ScalarType::BFloat16, ([&] {
                             hadamard_dispatch<::sycl::ext::oneapi::bfloat16>(
                                 q,
                                 reinterpret_cast<const ::sycl::ext::oneapi::bfloat16*>(input.data_ptr<at::BFloat16>()),
                                 reinterpret_cast<::sycl::ext::oneapi::bfloat16*>(output.data_ptr<at::BFloat16>()),
                                 batch,
                                 x_batch_stride,
                                 out_batch_stride,
                                 log_N,
                                 scale_f);
                           })));
}

}  // namespace at::native::xpu
