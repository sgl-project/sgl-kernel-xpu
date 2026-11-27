#include <torch/all.h>

#include <atomic>
#include <cstdint>
#include <cstdio>

#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include "sgl_kernel_export.h"

namespace {

// lm_head weight is [vocab, hidden] row-major, so a vocabulary row is contiguous
// and neighboring rows are not. One work-group owns one row and the threads
// stream that row together, then reduce. A thread-per-row walk was uncoalesced
// and no faster than torch.matmul.
template <typename T>
struct LmHeadGemv {
  static constexpr int kWork = 64;

  const T* hidden;
  const T* weight;
  T* out;
  int K;
  int N;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    const int n = static_cast<int>(item.get_group(0));
    const int lid = static_cast<int>(item.get_local_id(0));
    float acc0 = 0.f;
    float acc1 = 0.f;
    float acc2 = 0.f;
    float acc3 = 0.f;
    if (n < N) {
      const T* row = weight + static_cast<int64_t>(n) * K;
      constexpr int kVec = 4;
      const int step = kVec * kWork;
      int k = lid * kVec;
      if (k + 3 < K) {
        T h0 = hidden[k];
        T h1 = hidden[k + 1];
        T h2 = hidden[k + 2];
        T h3 = hidden[k + 3];
        T w0 = row[k];
        T w1 = row[k + 1];
        T w2 = row[k + 2];
        T w3 = row[k + 3];
        for (int next = k + step; next + 3 < K; next += step) {
          const T nh0 = hidden[next];
          const T nh1 = hidden[next + 1];
          const T nh2 = hidden[next + 2];
          const T nh3 = hidden[next + 3];
          const T nw0 = row[next];
          const T nw1 = row[next + 1];
          const T nw2 = row[next + 2];
          const T nw3 = row[next + 3];
          acc0 += static_cast<float>(w0) * static_cast<float>(h0);
          acc1 += static_cast<float>(w1) * static_cast<float>(h1);
          acc2 += static_cast<float>(w2) * static_cast<float>(h2);
          acc3 += static_cast<float>(w3) * static_cast<float>(h3);
          k = next;
          h0 = nh0;
          h1 = nh1;
          h2 = nh2;
          h3 = nh3;
          w0 = nw0;
          w1 = nw1;
          w2 = nw2;
          w3 = nw3;
        }
        acc0 += static_cast<float>(w0) * static_cast<float>(h0);
        acc1 += static_cast<float>(w1) * static_cast<float>(h1);
        acc2 += static_cast<float>(w2) * static_cast<float>(h2);
        acc3 += static_cast<float>(w3) * static_cast<float>(h3);
        k += step;
      }
      for (; k < K; k += kWork) {
        acc0 += static_cast<float>(row[k]) * static_cast<float>(hidden[k]);
      }
    }
    const float acc = sycl::reduce_over_group(item.get_group(), acc0 + acc1 + acc2 + acc3, sycl::plus<float>());
    if (n < N && lid == 0) {
      out[n] = static_cast<T>(acc);
    }
  }
};

template <typename T>
void launch_lm_head_row(sycl::queue& queue, const T* hidden, const T* weight, T* out, int K, int N) {
  constexpr int kWork = LmHeadGemv<T>::kWork;
  const size_t rows = static_cast<size_t>(N);
  queue.submit([&](sycl::handler& cgh) {
    LmHeadGemv<T> fn{hidden, weight, out, K, N};
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(rows * kWork), sycl::range<1>(kWork)), fn);
  });
}

// Same 4-wide software pipeline as LmHeadGemv. Each hidden row keeps its own
// accumulator, and each vocabulary row is loaded once for the rows in the tile.
template <typename T, int kSlots>
struct LmHeadPipelined {
  static constexpr int kWork = 64;

  const T* hidden;
  const T* weight;
  T* out;
  int M;
  int K;
  int N;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    const int n = static_cast<int>(item.get_group(0));
    const int lid = static_cast<int>(item.get_local_id(0));
    float acc[kSlots];
#pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
      acc[slot] = 0.f;
    }
    if (n < N) {
      const T* row = weight + static_cast<int64_t>(n) * K;
      constexpr int kVec = 4;
      const int step = kVec * kWork;
      int k = lid * kVec;
      if (k + 3 < K) {
        T w0 = row[k];
        T w1 = row[k + 1];
        T w2 = row[k + 2];
        T w3 = row[k + 3];
        T h0[kSlots];
        T h1[kSlots];
        T h2[kSlots];
        T h3[kSlots];
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
          if (slot < M) {
            const T* hs = hidden + static_cast<int64_t>(slot) * K + k;
            h0[slot] = hs[0];
            h1[slot] = hs[1];
            h2[slot] = hs[2];
            h3[slot] = hs[3];
          }
        }
        for (int next = k + step; next + 3 < K; next += step) {
          const T nw0 = row[next];
          const T nw1 = row[next + 1];
          const T nw2 = row[next + 2];
          const T nw3 = row[next + 3];
          T nh0[kSlots];
          T nh1[kSlots];
          T nh2[kSlots];
          T nh3[kSlots];
#pragma unroll
          for (int slot = 0; slot < kSlots; ++slot) {
            if (slot < M) {
              const T* hs = hidden + static_cast<int64_t>(slot) * K + next;
              nh0[slot] = hs[0];
              nh1[slot] = hs[1];
              nh2[slot] = hs[2];
              nh3[slot] = hs[3];
            }
          }
#pragma unroll
          for (int slot = 0; slot < kSlots; ++slot) {
            if (slot < M) {
              acc[slot] += static_cast<float>(w0) * static_cast<float>(h0[slot]);
              acc[slot] += static_cast<float>(w1) * static_cast<float>(h1[slot]);
              acc[slot] += static_cast<float>(w2) * static_cast<float>(h2[slot]);
              acc[slot] += static_cast<float>(w3) * static_cast<float>(h3[slot]);
            }
          }
          k = next;
          w0 = nw0;
          w1 = nw1;
          w2 = nw2;
          w3 = nw3;
#pragma unroll
          for (int slot = 0; slot < kSlots; ++slot) {
            h0[slot] = nh0[slot];
            h1[slot] = nh1[slot];
            h2[slot] = nh2[slot];
            h3[slot] = nh3[slot];
          }
        }
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
          if (slot < M) {
            acc[slot] += static_cast<float>(w0) * static_cast<float>(h0[slot]);
            acc[slot] += static_cast<float>(w1) * static_cast<float>(h1[slot]);
            acc[slot] += static_cast<float>(w2) * static_cast<float>(h2[slot]);
            acc[slot] += static_cast<float>(w3) * static_cast<float>(h3[slot]);
          }
        }
        k += step;
      }
      for (; k < K; k += kWork) {
        const float w = static_cast<float>(row[k]);
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
          if (slot < M) {
            acc[slot] += w * static_cast<float>(hidden[static_cast<int64_t>(slot) * K + k]);
          }
        }
      }
    }
#pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
      const float sum = sycl::reduce_over_group(item.get_group(), acc[slot], sycl::plus<float>());
      if (n < N && lid == 0 && slot < M) {
        out[static_cast<int64_t>(slot) * N + n] = static_cast<T>(sum);
      }
    }
  }
};

template <typename T, int kSlots>
void launch_lm_head_slots(
    sycl::queue& queue,
    const T* hidden,
    const T* weight,
    T* out,
    int M,
    int K,
    int N) {
  constexpr int kWork = LmHeadPipelined<T, kSlots>::kWork;
  const size_t rows = static_cast<size_t>(N);
  queue.submit([&](sycl::handler& cgh) {
    LmHeadPipelined<T, kSlots> fn{hidden, weight, out, M, K, N};
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(rows * kWork), sycl::range<1>(kWork)), fn);
  });
}

// One vocabulary pass covers up to 8 hidden rows. A 4-row verify is one pass.
template <typename T>
void launch_lm_head_rows(sycl::queue& queue, const T* hidden, const T* weight, T* out, int M, int K, int N) {
  for (int m0 = 0; m0 < M;) {
    const int left = M - m0;
    const T* h = hidden + static_cast<int64_t>(m0) * K;
    T* o = out + static_cast<int64_t>(m0) * N;
    if (left >= 8) {
      launch_lm_head_slots<T, 8>(queue, h, weight, o, 8, K, N);
      m0 += 8;
    } else if (left > 4) {
      launch_lm_head_slots<T, 8>(queue, h, weight, o, left, K, N);
      m0 += left;
    } else {
      launch_lm_head_slots<T, 4>(queue, h, weight, o, left, K, N);
      m0 += left;
    }
  }
}


}  // namespace

SGL_KERNEL_EXPORT torch::Tensor lm_head_gemv(torch::Tensor hidden, torch::Tensor weight) {
  TORCH_CHECK(hidden.is_xpu() && weight.is_xpu(), "lm_head_gemv tensors must be XPU");
  TORCH_CHECK(hidden.dim() == 2 && weight.dim() == 2, "lm_head_gemv expects hidden[M,K] and weight[N,K]");
  TORCH_CHECK(hidden.is_contiguous() && weight.is_contiguous(), "lm_head_gemv tensors must be contiguous");
  TORCH_CHECK(hidden.size(1) == weight.size(1), "lm_head_gemv K mismatch");
  TORCH_CHECK(
      hidden.scalar_type() == weight.scalar_type() &&
          (hidden.scalar_type() == at::kHalf || hidden.scalar_type() == at::kBFloat16),
      "lm_head_gemv dtype must be fp16 or bf16");
  const int M = static_cast<int>(hidden.size(0));
  const int K = static_cast<int>(hidden.size(1));
  const int N = static_cast<int>(weight.size(0));
  auto out = torch::empty({hidden.size(0), weight.size(0)}, hidden.options());
  auto queue = at::xpu::getCurrentXPUStream().queue();
  static std::atomic<int> once{0};
  if (once.fetch_add(1) == 0) {
    std::fprintf(stderr, "SGL_GPTQ_LMHEAD\n");
    std::fflush(stderr);
  }
  if (hidden.scalar_type() == at::kHalf) {
    const auto* h = reinterpret_cast<const sycl::half*>(hidden.data_ptr<at::Half>());
    const auto* w = reinterpret_cast<const sycl::half*>(weight.data_ptr<at::Half>());
    auto* o = reinterpret_cast<sycl::half*>(out.data_ptr<at::Half>());
    if (M == 1) {
      launch_lm_head_row<sycl::half>(queue, h, w, o, K, N);
    } else {
      launch_lm_head_rows<sycl::half>(queue, h, w, o, M, K, N);
    }
  } else {
    using bf16 = sycl::ext::oneapi::bfloat16;
    const auto* h = reinterpret_cast<const bf16*>(hidden.data_ptr<at::BFloat16>());
    const auto* w = reinterpret_cast<const bf16*>(weight.data_ptr<at::BFloat16>());
    auto* o = reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>());
    if (M == 1) {
      launch_lm_head_row<bf16>(queue, h, w, o, K, N);
    } else {
      launch_lm_head_rows<bf16>(queue, h, w, o, M, K, N);
    }
  }
  return out;
}
