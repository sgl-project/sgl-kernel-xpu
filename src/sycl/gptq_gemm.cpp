#include <torch/all.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include "sgl_kernel_export.h"

#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#define SGL_GPTQ_DPAS 1
#endif

namespace {

constexpr int kGroup = 128;

inline int nibble_from_word(int word, int nibble) {
  return (word >> (nibble * 4)) & 0x0F;
}

inline int effective_zero(int stored, int zero_format) {
  return zero_format == 1 ? stored + 1 : stored;
}

// qweight[K/8, N]: nibble (k % 8) of qweight[k/8, n] is weight row k.
inline int load_q(const int* qweight, int k, int n, int N) {
  return nibble_from_word(qweight[(k >> 3) * N + n], k & 7);
}

// qzeros[K/128, N/8]: nibble (n % 8) of qzeros[group, n/8].
inline int load_z(const int* qzeros, int group, int n, int N) {
  return nibble_from_word(qzeros[group * (N >> 3) + (n >> 3)], n & 7);
}

template <typename T>
struct GptqGemvFunctor {
  const T* a;
  const int* qweight;
  const int* qzeros;
  const T* scales;
  const int* g_idx;
  T* out;
  int K;
  int N;
  int zero_format;
  bool scalar;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    const int n = static_cast<int>(item.get_global_id(0));
    if (n >= N) {
      return;
    }
    float acc = 0.f;
    int prev_group = -1;
    float scale = 0.f;
    int zero = 0;
    if (scalar) {
      for (int k = 0; k < K; ++k) {
        const int group = g_idx[k];
        if (group != prev_group) {
          scale = static_cast<float>(scales[group * N + n]);
          zero = effective_zero(load_z(qzeros, group, n, N), zero_format);
          prev_group = group;
        }
        const int q = load_q(qweight, k, n, N);
        acc += static_cast<float>(a[k]) * (static_cast<float>(q - zero) * scale);
      }
    } else {
      // Eight K rows share one int32. Load that word once, then apply each nibble.
      for (int k0 = 0; k0 < K; k0 += 8) {
        const int word = qweight[(k0 >> 3) * N + n];
#pragma unroll
        for (int nib = 0; nib < 8; ++nib) {
          const int k = k0 + nib;
          const int group = g_idx[k];
          if (group != prev_group) {
            scale = static_cast<float>(scales[group * N + n]);
            zero = effective_zero(load_z(qzeros, group, n, N), zero_format);
            prev_group = group;
          }
          const int q = (word >> (nib * 4)) & 0x0F;
          acc += static_cast<float>(a[k]) * (static_cast<float>(q - zero) * scale);
        }
      }
    }
    out[n] = static_cast<T>(acc);
  }
};

template <typename T>
struct GptqGemmFunctor {
  const T* a;
  const int* qweight;
  const int* qzeros;
  const T* scales;
  const int* g_idx;
  T* out;
  int M;
  int K;
  int N;
  int zero_format;
  bool scalar;

  void operator()(sycl::nd_item<2> item) const {
    const int m = static_cast<int>(item.get_global_id(0));
    const int n = static_cast<int>(item.get_global_id(1));
    if (m >= M || n >= N) {
      return;
    }
    float acc = 0.f;
    int prev_group = -1;
    float scale = 0.f;
    int zero = 0;
    const T* arow = a + static_cast<int64_t>(m) * K;
    if (scalar) {
      for (int k = 0; k < K; ++k) {
        const int group = g_idx[k];
        if (group != prev_group) {
          scale = static_cast<float>(scales[group * N + n]);
          zero = effective_zero(load_z(qzeros, group, n, N), zero_format);
          prev_group = group;
        }
        const int q = load_q(qweight, k, n, N);
        acc += static_cast<float>(arow[k]) * (static_cast<float>(q - zero) * scale);
      }
    } else {
      for (int k0 = 0; k0 < K; k0 += 8) {
        const int word = qweight[(k0 >> 3) * N + n];
#pragma unroll
        for (int nib = 0; nib < 8; ++nib) {
          const int k = k0 + nib;
          const int group = g_idx[k];
          if (group != prev_group) {
            scale = static_cast<float>(scales[group * N + n]);
            zero = effective_zero(load_z(qzeros, group, n, N), zero_format);
            prev_group = group;
          }
          const int q = (word >> (nib * 4)) & 0x0F;
          acc += static_cast<float>(arow[k]) * (static_cast<float>(q - zero) * scale);
        }
      }
    }
    out[static_cast<int64_t>(m) * N + n] = static_cast<T>(acc);
  }
};

#if defined(SGL_GPTQ_DPAS)
// BMG XMX atom is 8x16x16 BF16. B is dequantized into one K-slice in SLM, then
// VNNI-packed: low 16 bits are row k, high 16 bits are row k+1. That is the
// layout DPAS reads. Rows past M and columns past N are zero, so M=1 decode
// still issues the 8-row atom.
template <typename T>
struct GptqDpasFunctor {
  const T* a;
  const int* qweight;
  const int* qzeros;
  const T* scales;
  const int* g_idx;
  T* out;
  int M;
  int K;
  int N;
  int zero_format;
  sycl::local_accessor<T, 1> a_tile;
  sycl::local_accessor<T, 1> b_tile;
  sycl::local_accessor<float, 2> c_tile;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<2> item) const {
    using namespace sycl::ext::oneapi::experimental::matrix;
    constexpr int TM = 8;
    constexpr int TN = 16;
    constexpr int TK = 16;
    constexpr int kPackedStride = TN * 2;
    const int tile_m = static_cast<int>(item.get_group(0));
    const int tile_n = static_cast<int>(item.get_group(1));
    const int m0 = tile_m * TM;
    const int n0 = tile_n * TN;
    if (m0 >= M || n0 >= N) {
      return;
    }
    auto sg = item.get_sub_group();
    // SPIR joint_matrix is not copyable. mad writes the accumulator in place.
    joint_matrix<sycl::sub_group, T, use::a, TM, TK, layout::row_major> tA;
    joint_matrix<sycl::sub_group, T, use::b, TK, TN, layout::ext_intel_packed> tB;
    joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> tC;
    joint_matrix_fill(sg, tC, 0.f);
    const int lane = static_cast<int>(sg.get_local_id());
    for (int k0 = 0; k0 < K; k0 += TK) {
      for (int idx = lane; idx < TM * TK; idx += 16) {
        const int row = idx / TK;
        const int kk = idx % TK;
        const int m = m0 + row;
        const int k = k0 + kk;
        T val = static_cast<T>(0.f);
        if (m < M && k < K) {
          val = a[static_cast<int64_t>(m) * K + k];
        }
        a_tile[idx] = val;
      }
      for (int idx = lane; idx < TK * TN; idx += 16) {
        const int kk = idx / TN;
        const int nn = idx % TN;
        const int k = k0 + kk;
        const int n = n0 + nn;
        T val = static_cast<T>(0.f);
        if (k < K && n < N) {
          const int group = g_idx[k];
          const int q = load_q(qweight, k, n, N);
          const int z = effective_zero(load_z(qzeros, group, n, N), zero_format);
          const float s = static_cast<float>(scales[group * N + n]);
          val = static_cast<T>((static_cast<float>(q - z)) * s);
        }
        b_tile[(kk >> 1) * kPackedStride + nn * 2 + (kk & 1)] = val;
      }
      sycl::group_barrier(sg);
      auto a_ptr = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
          &a_tile[0]);
      auto b_ptr = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
          &b_tile[0]);
      joint_matrix_load(sg, tA, a_ptr, static_cast<size_t>(TK));
      joint_matrix_load(sg, tB, b_ptr, static_cast<size_t>(kPackedStride));
      joint_matrix_mad(sg, tC, tA, tB, tC);
      sycl::group_barrier(sg);
    }
    auto c_ptr = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        &c_tile[0][0]);
    joint_matrix_store(sg, tC, c_ptr, static_cast<size_t>(TN), layout::row_major);
    sycl::group_barrier(sg);
    for (int idx = lane; idx < TM * TN; idx += 16) {
      const int row = idx / TN;
      const int col = idx % TN;
      const int m = m0 + row;
      const int n = n0 + col;
      if (m < M && n < N) {
        out[static_cast<int64_t>(m) * N + n] = static_cast<T>(c_tile[row][col]);
      }
    }
  }
};

struct XmxStats {
  std::atomic<long> launches{0};
  std::atomic<long> fallbacks{0};
  ~XmxStats() {
    std::fprintf(
        stderr,
        "SGL_KERNEL_XPU_GPTQ_XMX launches=%ld fallbacks=%ld\n",
        launches.load(),
        fallbacks.load());
    std::fflush(stderr);
  }
};

XmxStats& xmx_stats() {
  static XmxStats stats;
  return stats;
}
#endif

bool env_flag(const char* name) {
  const char* flag = std::getenv(name);
  return flag != nullptr && flag[0] == '1' && flag[1] == '\0';
}

#if defined(SGL_GPTQ_DPAS)
bool dpas_enabled() {
  // XMX is an 8-row atom. Decode is one row, so the bandwidth GEMV is the
  // default. SGL_GPTQ_DPAS=1 opts into the packed DPAS path for wide prefill.
  return env_flag("SGL_GPTQ_DPAS");
}
#endif

// This checkpoint stores g_idx[k] == k/128. The pointer is stable after load,
// so the device scan runs once per weight and the decode loop skips it.
bool cached_contiguous_groups(sycl::queue& queue, const int* g_idx, int K, const void* key) {
  static std::mutex mu;
  static std::unordered_map<const void*, bool> cache;
  std::lock_guard<std::mutex> lock(mu);
  const auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  int* flag = sycl::malloc_device<int>(1, queue);
  queue.memset(flag, 0, sizeof(int)).wait();
  queue
      .parallel_for(sycl::range<1>(static_cast<size_t>(K)), [=](sycl::id<1> id) {
        const int k = static_cast<int>(id[0]);
        if (g_idx[k] != k / kGroup) {
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> seen(flag[0]);
          seen.store(1);
        }
      })
      .wait();
  int host = 0;
  queue.memcpy(&host, flag, sizeof(int)).wait();
  sycl::free(flag, queue);
  const bool ok = host == 0;
  cache.emplace(key, ok);
  return ok;
}

struct BandwidthStats {
  std::atomic<long> gemv{0};
  std::atomic<long> gemm{0};
  std::atomic<long long> decode_us{0};
  std::atomic<long long> prefill_us{0};
  ~BandwidthStats() {
    std::fprintf(
        stderr,
        "SGL_GPTQ_BANDWIDTH gemv=%ld gemm=%ld decode_us=%lld prefill_us=%lld\n",
        gemv.load(),
        gemm.load(),
        decode_us.load(),
        prefill_us.load());
    std::fflush(stderr);
  }
};

BandwidthStats& bandwidth_stats() {
  static BandwidthStats stats;
  return stats;
}

// One column per thread. qweight is [K/8, N], so a subgroup reading
// consecutive columns issues one coalesced int32 vector. The activation is
// the same address on every lane, so it stays in cache. Four accumulators
// keep the dequant chain from serializing the loads. No shared-memory barrier.
template <typename T>
struct GptqBandwidthGemv {
  static constexpr int kWork = 64;

  const T* a;
  const int* qweight;
  const int* qzeros;
  const T* scales;
  T* out;
  int K;
  int N;
  int zero_format;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    const int lid = static_cast<int>(item.get_local_id(0));
    const int n = static_cast<int>(item.get_group(0)) * kWork + lid;
    const bool live = n < N;
    float acc0 = 0.f;
    float acc1 = 0.f;
    float acc2 = 0.f;
    float acc3 = 0.f;
    const int words = K >> 3;
    for (int base = 0; base < words; base += 16) {
      float scale = 0.f;
      float zero = 0.f;
      if (live) {
        const int group = base >> 4;
        scale = static_cast<float>(scales[group * N + n]);
        zero = static_cast<float>(effective_zero(load_z(qzeros, group, n, N), zero_format));
      }
      int w0 = 0;
      int w1 = 0;
      int w2 = 0;
      int w3 = 0;
      if (live) {
        w0 = qweight[(base + 0) * N + n];
        w1 = qweight[(base + 1) * N + n];
        w2 = qweight[(base + 2) * N + n];
        w3 = qweight[(base + 3) * N + n];
      }
#pragma unroll
      for (int step = 0; step < 16; step += 4) {
        int n0 = 0;
        int n1 = 0;
        int n2 = 0;
        int n3 = 0;
        if (live && step + 4 < 16) {
          n0 = qweight[(base + step + 4) * N + n];
          n1 = qweight[(base + step + 5) * N + n];
          n2 = qweight[(base + step + 6) * N + n];
          n3 = qweight[(base + step + 7) * N + n];
        }
#pragma unroll
        for (int nib = 0; nib < 8; ++nib) {
          const int k = ((base + step) << 3) + nib;
          const float act = static_cast<float>(a[k]);
          const float d0 = (static_cast<float>((w0 >> (nib * 4)) & 0x0F) - zero) * scale;
          const float d1 = (static_cast<float>((w1 >> (nib * 4)) & 0x0F) - zero) * scale;
          const float d2 = (static_cast<float>((w2 >> (nib * 4)) & 0x0F) - zero) * scale;
          const float d3 = (static_cast<float>((w3 >> (nib * 4)) & 0x0F) - zero) * scale;
          acc0 += act * d0;
          const float act1 = static_cast<float>(a[k + 8]);
          acc1 += act1 * d1;
          const float act2 = static_cast<float>(a[k + 16]);
          acc2 += act2 * d2;
          const float act3 = static_cast<float>(a[k + 24]);
          acc3 += act3 * d3;
        }
        w0 = n0;
        w1 = n1;
        w2 = n2;
        w3 = n3;
      }
    }
    if (live) {
      out[n] = static_cast<T>(acc0 + acc1 + acc2 + acc3);
    }
  }
};

// lm_head hits about half of device bandwidth by giving one work-group a
// contiguous row. qweight is [K/8, N], so decode transposes it once and then
// uses that same pattern. Prefill keeps the original layout.
template <typename T>
struct GptqPackedGemv {
  static constexpr int kWork = 64;

  const T* a;
  const int* packed;
  const int* qzeros;
  const T* scales;
  T* out;
  int K;
  int N;
  int zero_format;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    const int n = static_cast<int>(item.get_group(0));
    const int lid = static_cast<int>(item.get_local_id(0));
    float acc0 = 0.f;
    float acc1 = 0.f;
    float acc2 = 0.f;
    float acc3 = 0.f;
    if (n < N) {
      const int words = K >> 3;
      const int* row = packed + static_cast<int64_t>(n) * words;
      for (int word = lid; word + 3 * kWork < words; word += 4 * kWork) {
        const int w0 = row[word];
        const int w1 = row[word + kWork];
        const int w2 = row[word + 2 * kWork];
        const int w3 = row[word + 3 * kWork];
        const int g0 = word >> 4;
        const int g1 = (word + kWork) >> 4;
        const int g2 = (word + 2 * kWork) >> 4;
        const int g3 = (word + 3 * kWork) >> 4;
        const float s0 = static_cast<float>(scales[g0 * N + n]);
        const float s1 = static_cast<float>(scales[g1 * N + n]);
        const float s2 = static_cast<float>(scales[g2 * N + n]);
        const float s3 = static_cast<float>(scales[g3 * N + n]);
        const float z0 = static_cast<float>(effective_zero(load_z(qzeros, g0, n, N), zero_format));
        const float z1 = static_cast<float>(effective_zero(load_z(qzeros, g1, n, N), zero_format));
        const float z2 = static_cast<float>(effective_zero(load_z(qzeros, g2, n, N), zero_format));
        const float z3 = static_cast<float>(effective_zero(load_z(qzeros, g3, n, N), zero_format));
#pragma unroll
        for (int nib = 0; nib < 8; ++nib) {
          acc0 += static_cast<float>(a[(word << 3) + nib]) *
                  ((static_cast<float>((w0 >> (nib * 4)) & 0x0F) - z0) * s0);
          acc1 += static_cast<float>(a[((word + kWork) << 3) + nib]) *
                  ((static_cast<float>((w1 >> (nib * 4)) & 0x0F) - z1) * s1);
          acc2 += static_cast<float>(a[((word + 2 * kWork) << 3) + nib]) *
                  ((static_cast<float>((w2 >> (nib * 4)) & 0x0F) - z2) * s2);
          acc3 += static_cast<float>(a[((word + 3 * kWork) << 3) + nib]) *
                  ((static_cast<float>((w3 >> (nib * 4)) & 0x0F) - z3) * s3);
        }
      }
      for (int word = lid + ((words / (4 * kWork)) * 4 * kWork); word < words; word += kWork) {
        const int group = word >> 4;
        const float scale = static_cast<float>(scales[group * N + n]);
        const float zero = static_cast<float>(effective_zero(load_z(qzeros, group, n, N), zero_format));
        const int bits = row[word];
        const int k = word << 3;
#pragma unroll
        for (int nib = 0; nib < 8; ++nib) {
          const int q = (bits >> (nib * 4)) & 0x0F;
          acc0 += static_cast<float>(a[k + nib]) * ((static_cast<float>(q) - zero) * scale);
        }
      }
    }
    const float acc = sycl::reduce_over_group(
        item.get_group(), acc0 + acc1 + acc2 + acc3, sycl::plus<float>());
    if (n < N && lid == 0) {
      out[n] = static_cast<T>(acc);
    }
  }
};

struct PackEntry {
  int words = 0;
  int N = 0;
  int head = 0;
  int tail = 0;
  at::Tensor packed;
};

void ensure_transposed(sycl::queue& queue, int* qweight, int K, int N, const void* key) {
  const int words = K >> 3;
  static std::mutex mu;
  static std::unordered_map<const void*, std::pair<int, int>> done;
  std::lock_guard<std::mutex> lock(mu);
  const auto it = done.find(key);
  if (it != done.end() && it->second.first == words && it->second.second == N) {
    return;
  }
  auto scratch = torch::empty(
      {static_cast<int64_t>(words) * N}, torch::TensorOptions().dtype(torch::kInt).device(torch::kXPU));
  int* dst = scratch.data_ptr<int>();
  const int* src = qweight;
  const size_t total = static_cast<size_t>(words) * static_cast<size_t>(N);
  queue
      .submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
          const int idx = static_cast<int>(id);
          const int k8 = idx / N;
          const int col = idx - k8 * N;
          dst[(static_cast<int64_t>(col) * words) + k8] = src[idx];
        });
      })
      .wait();
  queue.memcpy(qweight, dst, total * sizeof(int)).wait();
  done[key] = {words, N};
}

const int* repacked_qweight(sycl::queue& queue, const int* qweight, int K, int N, const void* key) {
  const int words = K >> 3;
  static std::mutex mu;
  static std::unordered_map<const void*, PackEntry> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto& entry = cache[key];
  if (entry.packed.defined() && entry.words == words && entry.N == N) {
    return entry.packed.data_ptr<int>();
  }
  int head = 0;
  int tail = 0;
  queue.memcpy(&head, qweight, sizeof(int));
  queue.memcpy(&tail, qweight + static_cast<int64_t>(words) * N - 1, sizeof(int)).wait();
  auto packed = torch::empty(
      {static_cast<int64_t>(N) * words}, torch::TensorOptions().dtype(torch::kInt).device(torch::kXPU));
  const int* src = qweight;
  int* dst = packed.data_ptr<int>();
  const size_t total = static_cast<size_t>(words) * static_cast<size_t>(N);
  queue
      .submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
          const int idx = static_cast<int>(id);
          const int k8 = idx / N;
          const int col = idx - k8 * N;
          dst[(static_cast<int64_t>(col) * words) + k8] = src[idx];
        });
      })
      .wait();
  entry.words = words;
  entry.N = N;
  entry.head = head;
  entry.tail = tail;
  entry.packed = std::move(packed);
  return entry.packed.data_ptr<int>();
}

// Prefill keeps one K-group of every row in SLM, then every column reads it.
// Reloading those activations from global memory once per column made the
// short prompt cost about 27 seconds.
template <typename T, int MT>
struct GptqStreamGemm {
  static constexpr int kWork = 128;

  const T* a;
  const int* qweight;
  const int* qzeros;
  const T* scales;
  T* out;
  int M;
  int K;
  int N;
  int zero_format;
  int m0;
  sycl::local_accessor<T, 1> a_tile;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    const int lid = static_cast<int>(item.get_local_id(0));
    const int n = static_cast<int>(item.get_group(0)) * kWork + lid;
    float acc[MT];
#pragma unroll
    for (int r = 0; r < MT; ++r) {
      acc[r] = 0.f;
    }
    const int groups = K / kGroup;
    for (int g = 0; g < groups; ++g) {
      for (int i = lid; i < MT * kGroup; i += kWork) {
        const int row = i / kGroup;
        const int kk = i - row * kGroup;
        const int m = m0 + row;
        T val = static_cast<T>(0.f);
        if (m < M) {
          val = a[static_cast<int64_t>(m) * K + g * kGroup + kk];
        }
        a_tile[i] = val;
      }
      sycl::group_barrier(item.get_group());
      if (n < N) {
        const float scale = static_cast<float>(scales[g * N + n]);
        const int zero = effective_zero(load_z(qzeros, g, n, N), zero_format);
        const int words = K >> 3;
        const int* col = qweight + static_cast<int64_t>(n) * words + g * (kGroup / 8);
#pragma unroll 2
        for (int step = 0; step < kGroup / 8; ++step) {
          const int word = col[step];
#pragma unroll
          for (int nib = 0; nib < 8; ++nib) {
            const float w = static_cast<float>((word >> (nib * 4)) & 0x0F) - static_cast<float>(zero);
            const float weight = w * scale;
#pragma unroll
            for (int r = 0; r < MT; ++r) {
              acc[r] += static_cast<float>(a_tile[r * kGroup + step * 8 + nib]) * weight;
            }
          }
        }
      }
      sycl::group_barrier(item.get_group());
    }
    if (n < N) {
#pragma unroll
      for (int r = 0; r < MT; ++r) {
        const int m = m0 + r;
        if (m < M) {
          out[static_cast<int64_t>(m) * N + n] = static_cast<T>(acc[r]);
        }
      }
    }
  }
};

bool profile_enabled() {
  return env_flag("SGL_GPTQ_PROFILE") || env_flag("SGLANG_GPTQ_PROFILE");
}

void note_profile(sycl::queue& queue, bool decode) {
  if (!profile_enabled()) {
    return;
  }
  const auto started = std::chrono::steady_clock::now();
  queue.wait();
  const auto us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
  auto& stats = bandwidth_stats();
  const long long total = decode ? stats.decode_us.fetch_add(us) + us : stats.prefill_us.fetch_add(us) + us;
  const long calls = stats.gemv.load() + stats.gemm.load();
  if (calls % 32 == 0) {
    std::fprintf(
        stderr,
        "SGL_GPTQ_PROFILE calls=%ld decode_us=%lld prefill_us=%lld\n",
        calls,
        stats.decode_us.load(),
        stats.prefill_us.load());
    std::fflush(stderr);
  }
  (void)total;
}

template <typename T>
void launch_gptq(
    sycl::queue& queue,
    const T* a,
    const int* qweight,
    const int* qzeros,
    const T* scales,
    const int* g_idx,
    T* out,
    int M,
    int K,
    int N,
    int zero_format,
    const void* weight_key,
    const void* gidx_key) {
  const bool scalar = env_flag("SGL_GPTQ_SCALAR");
  const bool contiguous = !scalar && cached_contiguous_groups(queue, g_idx, K, gidx_key);
  if (contiguous) {
    ensure_transposed(queue, const_cast<int*>(qweight), K, N, weight_key);
  }
  if (contiguous && M == 1) {
    constexpr int kWork = GptqPackedGemv<T>::kWork;
    auto& stats = bandwidth_stats();
    if (stats.gemv.fetch_add(1) == 0) {
      std::fprintf(stderr, "SGL_GPTQ_PACKED\n");
      std::fflush(stderr);
    }
    queue.submit([&](sycl::handler& cgh) {
      GptqPackedGemv<T> fn{a, qweight, qzeros, scales, out, K, N, zero_format};
      cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(N) * kWork), sycl::range<1>(kWork)), fn);
    });
    note_profile(queue, true);
    return;
  }
  if (contiguous) {
    constexpr int kMt = 8;
    constexpr int kLocal = GptqStreamGemm<T, 8>::kWork;
    const int groups = (N + kLocal - 1) / kLocal;
    auto& stats = bandwidth_stats();
    stats.gemm.fetch_add(1);
    for (int m0 = 0; m0 < M; m0 += kMt) {
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<T, 1> a_tile(sycl::range<1>(kMt * kGroup), cgh);
        GptqStreamGemm<T, kMt> fn{a, qweight, qzeros, scales, out, M, K, N, zero_format, m0, a_tile};
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * kLocal), sycl::range<1>(kLocal)), fn);
      });
    }
    note_profile(queue, false);
    return;
  }
#if defined(SGL_GPTQ_DPAS)
  if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
    if (!scalar && dpas_enabled() && K % 16 == 0) {
      constexpr int TM = 8;
      constexpr int TN = 16;
      constexpr int TK = 16;
      const int tile_m = (M + TM - 1) / TM;
      const int tile_n = (N + TN - 1) / TN;
      auto& stats = xmx_stats();
      if (stats.launches.fetch_add(1) == 0) {
        std::fprintf(stderr, "SGL_KERNEL_XPU_GPTQ_XMX\n");
        std::fflush(stderr);
      }
      sycl::range<2> local(1, 16);
      sycl::range<2> global(tile_m, tile_n * 16);
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<T, 1> a_tile(sycl::range<1>(TM * TK), cgh);
        sycl::local_accessor<T, 1> b_tile(sycl::range<1>((TK / 2) * (TN * 2)), cgh);
        sycl::local_accessor<float, 2> c_tile(sycl::range<2>(TM, TN), cgh);
        GptqDpasFunctor<T> fn{
            a, qweight, qzeros, scales, g_idx, out, M, K, N, zero_format, a_tile, b_tile, c_tile};
        cgh.parallel_for(sycl::nd_range<2>(global, local), fn);
      });
      return;
    }
    if (!scalar && dpas_enabled()) {
      auto& stats = xmx_stats();
      if (stats.fallbacks.fetch_add(1) == 0) {
        std::fprintf(stderr, "SGL_KERNEL_XPU_GPTQ_XMX_FALLBACK M=%d K=%d N=%d\n", M, K, N);
        std::fflush(stderr);
      }
    }
  }
#endif
  if (M == 1) {
    constexpr int kLocal = 16;
    const int groups = (N + kLocal - 1) / kLocal;
    GptqGemvFunctor<T> fn{a, qweight, qzeros, scales, g_idx, out, K, N, zero_format, scalar};
    queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * kLocal), sycl::range<1>(kLocal)), fn);
    return;
  }
  constexpr int kLocalM = 4;
  constexpr int kLocalN = 16;
  const int grid_m = ((M + kLocalM - 1) / kLocalM) * kLocalM;
  const int grid_n = ((N + kLocalN - 1) / kLocalN) * kLocalN;
  GptqGemmFunctor<T> fn{a, qweight, qzeros, scales, g_idx, out, M, K, N, zero_format, scalar};
  queue.parallel_for(
      sycl::nd_range<2>(sycl::range<2>(grid_m, grid_n), sycl::range<2>(kLocalM, kLocalN)), fn);
}

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
    float acc = 0.f;
    if (n < N) {
      const T* row = weight + static_cast<int64_t>(n) * K;
      for (int k = lid; k < K; k += kWork) {
        acc += static_cast<float>(row[k]) * static_cast<float>(hidden[k]);
      }
    }
    acc = sycl::reduce_over_group(item.get_group(), acc, sycl::plus<float>());
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

// Several hidden rows share one vocabulary. One pass over each weight row
// keeps that matrix off the bus for every row in the tile.
template <typename T, int kSlots>
struct LmHeadMulti {
  static constexpr int kWork = 64;

  const T* hidden;
  const T* weight;
  T* out;
  int K;
  int N;
  int M;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    const int n = static_cast<int>(item.get_group(0));
    const int lid = static_cast<int>(item.get_local_id(0));
    float acc[kSlots];
    for (int slot = 0; slot < kSlots; ++slot) {
      acc[slot] = 0.f;
    }
    if (n < N) {
      const T* row = weight + static_cast<int64_t>(n) * K;
      for (int k = lid; k < K; k += kWork) {
        const float w = static_cast<float>(row[k]);
        for (int slot = 0; slot < kSlots; ++slot) {
          if (slot < M) {
            acc[slot] += w * static_cast<float>(hidden[static_cast<int64_t>(slot) * K + k]);
          }
        }
      }
    }
    for (int slot = 0; slot < kSlots; ++slot) {
      const float sum = sycl::reduce_over_group(item.get_group(), acc[slot], sycl::plus<float>());
      if (n < N && lid == 0 && slot < M) {
        out[static_cast<int64_t>(slot) * N + n] = static_cast<T>(sum);
      }
    }
  }
};

template <typename T>
void launch_lm_head_multi(sycl::queue& queue, const T* hidden, const T* weight, T* out, int M, int K, int N) {
  constexpr int kWork = LmHeadMulti<T, 8>::kWork;
  const size_t rows = static_cast<size_t>(N);
  queue.submit([&](sycl::handler& cgh) {
    LmHeadMulti<T, 8> fn{hidden, weight, out, K, N, M};
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(rows * kWork), sycl::range<1>(kWork)), fn);
  });
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
      for (int m0 = 0; m0 < M; m0 += 8) {
        const int tile = std::min(8, M - m0);
        launch_lm_head_multi<sycl::half>(
            queue, h + static_cast<int64_t>(m0) * K, w, o + static_cast<int64_t>(m0) * N, tile, K, N);
      }
    }
  } else {
    using bf16 = sycl::ext::oneapi::bfloat16;
    const auto* h = reinterpret_cast<const bf16*>(hidden.data_ptr<at::BFloat16>());
    const auto* w = reinterpret_cast<const bf16*>(weight.data_ptr<at::BFloat16>());
    auto* o = reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>());
    if (M == 1) {
      launch_lm_head_row<bf16>(queue, h, w, o, K, N);
    } else {
      for (int m0 = 0; m0 < M; m0 += 8) {
        const int tile = std::min(8, M - m0);
        launch_lm_head_multi<bf16>(
            queue, h + static_cast<int64_t>(m0) * K, w, o + static_cast<int64_t>(m0) * N, tile, K, N);
      }
    }
  }
  return out;
}

SGL_KERNEL_EXPORT torch::Tensor gptq_gemm(
    torch::Tensor a,
    torch::Tensor b_q_weight,
    torch::Tensor b_gptq_qzeros,
    torch::Tensor b_gptq_scales,
    torch::Tensor b_g_idx,
    bool use_shuffle,
    int64_t bit,
    int64_t zero_format) {
  TORCH_CHECK(!use_shuffle, "gptq_gemm use_shuffle=true is not supported");
  TORCH_CHECK(bit == 4, "gptq_gemm bit must be 4, got ", bit);
  TORCH_CHECK(
      zero_format == 1 || zero_format == 2,
      "gptq_gemm zero_format must be 1 (v1, stored+1) or 2 (v2, stored as-is), got ",
      zero_format);
  TORCH_CHECK(a.is_xpu(), "gptq_gemm activation must be an XPU tensor");
  TORCH_CHECK(b_q_weight.is_xpu() && b_gptq_qzeros.is_xpu() && b_gptq_scales.is_xpu() && b_g_idx.is_xpu(),
              "gptq_gemm weight tensors must be XPU tensors");
  TORCH_CHECK(a.dim() == 2 && b_q_weight.dim() == 2 && b_gptq_qzeros.dim() == 2 && b_gptq_scales.dim() == 2 &&
                  b_g_idx.dim() == 1,
              "gptq_gemm ranks must be a[M,K], qweight[K/8,N], qzeros[K/128,N/8], scales[K/128,N], g_idx[K]");
  TORCH_CHECK(a.is_contiguous() && b_q_weight.is_contiguous() && b_gptq_qzeros.is_contiguous() &&
                  b_gptq_scales.is_contiguous() && b_g_idx.is_contiguous(),
              "gptq_gemm tensors must be contiguous");
  TORCH_CHECK(
      a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16, "gptq_gemm activation dtype must be fp16 or bf16");
  TORCH_CHECK(b_gptq_scales.scalar_type() == a.scalar_type(), "gptq_gemm scales dtype must match activations");
  TORCH_CHECK(b_q_weight.scalar_type() == at::kInt && b_gptq_qzeros.scalar_type() == at::kInt &&
                  b_g_idx.scalar_type() == at::kInt,
              "gptq_gemm qweight, qzeros, and g_idx must be int32");

  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b_q_weight.size(1);
  TORCH_CHECK(K % kGroup == 0, "gptq_gemm K must be divisible by 128, got ", K);
  TORCH_CHECK(N % 8 == 0, "gptq_gemm N must be divisible by 8, got ", N);
  TORCH_CHECK(b_q_weight.size(0) == K / 8, "gptq_gemm qweight rows must be K/8");
  TORCH_CHECK(b_gptq_scales.size(0) == K / kGroup && b_gptq_scales.size(1) == N, "gptq_gemm scales shape must be [K/128, N]");
  TORCH_CHECK(
      b_gptq_qzeros.size(0) == K / kGroup && b_gptq_qzeros.size(1) == N / 8,
      "gptq_gemm qzeros shape must be [K/128, N/8]");
  TORCH_CHECK(b_g_idx.size(0) == K, "gptq_gemm g_idx length must be K, got ", b_g_idx.size(0));

  auto out = torch::empty({M, N}, a.options());
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  const int zf = static_cast<int>(zero_format);
  const void* weight_key = b_q_weight.storage().unsafeGetStorageImpl();
  const void* gidx_key = b_g_idx.storage().unsafeGetStorageImpl();
  if (a.scalar_type() == at::kHalf) {
    launch_gptq<sycl::half>(
        queue,
        reinterpret_cast<const sycl::half*>(a.data_ptr<at::Half>()),
        b_q_weight.data_ptr<int>(),
        b_gptq_qzeros.data_ptr<int>(),
        reinterpret_cast<const sycl::half*>(b_gptq_scales.data_ptr<at::Half>()),
        b_g_idx.data_ptr<int>(),
        reinterpret_cast<sycl::half*>(out.data_ptr<at::Half>()),
        static_cast<int>(M),
        static_cast<int>(K),
        static_cast<int>(N),
        zf,
        weight_key,
        gidx_key);
  } else {
    using bf16 = sycl::ext::oneapi::bfloat16;
    launch_gptq<bf16>(
        queue,
        reinterpret_cast<const bf16*>(a.data_ptr<at::BFloat16>()),
        b_q_weight.data_ptr<int>(),
        b_gptq_qzeros.data_ptr<int>(),
        reinterpret_cast<const bf16*>(b_gptq_scales.data_ptr<at::BFloat16>()),
        b_g_idx.data_ptr<int>(),
        reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
        static_cast<int>(M),
        static_cast<int>(K),
        static_cast<int>(N),
        zf,
        weight_key,
        gidx_key);
  }
  return out;
}
