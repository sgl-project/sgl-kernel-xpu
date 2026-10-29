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

// One launch covers every row of M (chunked only past 512) and reads each
// INT4 column once. A work-group owns 64 output columns. Its subgroups each
// keep an 8x16 accumulator in registers for the whole K loop. Every 128 K
// rows are dequantized into SLM in the VNNI layout DPAS consumes, then reused
// for every resident row. Adjacent threads own adjacent columns so the VNNI
// stores land on consecutive addresses. SGL_GPTQ_DPAS is not consulted.
template <typename T, bool kAligned>
struct GptqOnePassXmx {
  const T* a;
  const int* qweight;
  const int* qzeros;
  const T* scales;
  T* out;
  int M;
  int K;
  int N;
  int zero_format;
  sycl::local_accessor<T, 1> a_tile;
  sycl::local_accessor<T, 1> b_tile;
  sycl::local_accessor<float, 1> c_tile;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    using namespace sycl::ext::oneapi::experimental::matrix;
    constexpr int kTM = 8;
    constexpr int kTN = 16;
    constexpr int kTK = 16;
    constexpr int kNAtoms = 4;
    constexpr int kSg = 16;
    constexpr int kPackedStride = kTN * 2;
    constexpr int kTileElems = kTK * kTN;

    auto sg = item.get_sub_group();
    const int lane = static_cast<int>(sg.get_local_id());
    const int sg_id = static_cast<int>(sg.get_group_linear_id());
    const int m0 = sg_id * kTM;
    const int n0 = static_cast<int>(item.get_group(0)) * (kNAtoms * kTN);
    const int local = static_cast<int>(item.get_local_range(0));

    joint_matrix<sycl::sub_group, T, use::a, kTM, kTK, layout::row_major> tA;
    joint_matrix<sycl::sub_group, T, use::b, kTK, kTN, layout::ext_intel_packed> tB;
    joint_matrix<sycl::sub_group, float, use::accumulator, kTM, kTN> tC0;
    joint_matrix<sycl::sub_group, float, use::accumulator, kTM, kTN> tC1;
    joint_matrix<sycl::sub_group, float, use::accumulator, kTM, kTN> tC2;
    joint_matrix<sycl::sub_group, float, use::accumulator, kTM, kTN> tC3;
    joint_matrix_fill(sg, tC0, 0.f);
    joint_matrix_fill(sg, tC1, 0.f);
    joint_matrix_fill(sg, tC2, 0.f);
    joint_matrix_fill(sg, tC3, 0.f);

    const int words = K >> 3;
    constexpr int kCols = kNAtoms * kTN;
    const int nwords = (kGroup / 8) * kCols;
    const int tid = static_cast<int>(item.get_local_id(0));
    const int n_slices = K / kGroup;
    T* const slab = &b_tile[0];

    for (int slice = 0; slice < n_slices; ++slice) {
      for (int widx = tid; widx < nwords; widx += local) {
        const int wk = widx / kCols;
        const int nn = widx - wk * kCols;
        const int n = n0 + nn;
        const int k_base = slice * kGroup + wk * 8;
        int word = 0;
        float scale = 0.f;
        float zero = 0.f;
        if (n < N) {
          word = qweight[static_cast<int64_t>(n) * words + (k_base >> 3)];
          scale = static_cast<float>(scales[slice * N + n]);
          zero = static_cast<float>(effective_zero(load_z(qzeros, slice, n, N), zero_format));
        }
#pragma unroll
        for (int nib = 0; nib < 8; ++nib) {
          const int kk = wk * 8 + nib;
          const float dq =
              n < N ? (static_cast<float>((word >> (nib * 4)) & 0x0F) - zero) * scale : 0.f;
          const int ka = kk >> 4;
          const int kk_local = kk & 15;
          const int na = nn >> 4;
          const int nn_local = nn & 15;
          const int packed = (kk_local >> 1) * kPackedStride + nn_local * 2 + (kk_local & 1);
          slab[(ka * kNAtoms + na) * kTileElems + packed] = static_cast<T>(dq);
        }
      }
      sycl::group_barrier(item.get_group());

      const int k0 = slice * kGroup;
      for (int ka = 0; ka < kGroup / kTK; ++ka) {
        const int k_a = k0 + ka * kTK;
        if constexpr (kAligned) {
          auto a_ptr = sycl::address_space_cast<
              sycl::access::address_space::global_space, sycl::access::decorated::yes>(
              const_cast<T*>(a + static_cast<int64_t>(m0) * K + k_a));
          joint_matrix_load(sg, tA, a_ptr, static_cast<size_t>(K));
        } else {
          for (int idx = lane; idx < kTM * kTK; idx += kSg) {
            const int row = idx / kTK;
            const int kk = idx - row * kTK;
            const int m = m0 + row;
            const int k = k_a + kk;
            T val = static_cast<T>(0.f);
            if (m < M && k < K) {
              val = a[static_cast<int64_t>(m) * K + k];
            }
            a_tile[sg_id * (kTM * kTK) + idx] = val;
          }
          sycl::group_barrier(sg);
          auto a_ptr = sycl::address_space_cast<
              sycl::access::address_space::local_space, sycl::access::decorated::yes>(
              &a_tile[sg_id * (kTM * kTK)]);
          joint_matrix_load(sg, tA, a_ptr, static_cast<size_t>(kTK));
        }
        {
          auto b_ptr = sycl::address_space_cast<
              sycl::access::address_space::local_space, sycl::access::decorated::yes>(
              slab + (ka * kNAtoms + 0) * kTileElems);
          joint_matrix_load(sg, tB, b_ptr, static_cast<size_t>(kPackedStride));
          joint_matrix_mad(sg, tC0, tA, tB, tC0);
        }
        {
          auto b_ptr = sycl::address_space_cast<
              sycl::access::address_space::local_space, sycl::access::decorated::yes>(
              slab + (ka * kNAtoms + 1) * kTileElems);
          joint_matrix_load(sg, tB, b_ptr, static_cast<size_t>(kPackedStride));
          joint_matrix_mad(sg, tC1, tA, tB, tC1);
        }
        {
          auto b_ptr = sycl::address_space_cast<
              sycl::access::address_space::local_space, sycl::access::decorated::yes>(
              slab + (ka * kNAtoms + 2) * kTileElems);
          joint_matrix_load(sg, tB, b_ptr, static_cast<size_t>(kPackedStride));
          joint_matrix_mad(sg, tC2, tA, tB, tC2);
        }
        {
          auto b_ptr = sycl::address_space_cast<
              sycl::access::address_space::local_space, sycl::access::decorated::yes>(
              slab + (ka * kNAtoms + 3) * kTileElems);
          joint_matrix_load(sg, tB, b_ptr, static_cast<size_t>(kPackedStride));
          joint_matrix_mad(sg, tC3, tA, tB, tC3);
        }
      }
      sycl::group_barrier(item.get_group());
    }

    auto c_ptr = sycl::address_space_cast<
        sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        &c_tile[sg_id * (kTM * kTN)]);
    joint_matrix_store(sg, tC0, c_ptr, static_cast<size_t>(kTN), layout::row_major);
    sycl::group_barrier(sg);
    for (int idx = lane; idx < kTM * kTN; idx += kSg) {
      const int row = idx / kTN;
      const int col = idx - row * kTN;
      const int m = m0 + row;
      const int n = n0 + col;
      if (m < M && n < N) {
        out[static_cast<int64_t>(m) * N + n] = static_cast<T>(c_tile[sg_id * (kTM * kTN) + idx]);
      }
    }
    joint_matrix_store(sg, tC1, c_ptr, static_cast<size_t>(kTN), layout::row_major);
    sycl::group_barrier(sg);
    for (int idx = lane; idx < kTM * kTN; idx += kSg) {
      const int row = idx / kTN;
      const int col = idx - row * kTN;
      const int m = m0 + row;
      const int n = n0 + kTN + col;
      if (m < M && n < N) {
        out[static_cast<int64_t>(m) * N + n] = static_cast<T>(c_tile[sg_id * (kTM * kTN) + idx]);
      }
    }
    joint_matrix_store(sg, tC2, c_ptr, static_cast<size_t>(kTN), layout::row_major);
    sycl::group_barrier(sg);
    for (int idx = lane; idx < kTM * kTN; idx += kSg) {
      const int row = idx / kTN;
      const int col = idx - row * kTN;
      const int m = m0 + row;
      const int n = n0 + 2 * kTN + col;
      if (m < M && n < N) {
        out[static_cast<int64_t>(m) * N + n] = static_cast<T>(c_tile[sg_id * (kTM * kTN) + idx]);
      }
    }
    joint_matrix_store(sg, tC3, c_ptr, static_cast<size_t>(kTN), layout::row_major);
    sycl::group_barrier(sg);
    for (int idx = lane; idx < kTM * kTN; idx += kSg) {
      const int row = idx / kTN;
      const int col = idx - row * kTN;
      const int m = m0 + row;
      const int n = n0 + 3 * kTN + col;
      if (m < M && n < N) {
        out[static_cast<int64_t>(m) * N + n] = static_cast<T>(c_tile[sg_id * (kTM * kTN) + idx]);
      }
    }
  }
};

template <typename T, bool kAligned>
void submit_gptq_xmx_chunk(
    sycl::queue& queue,
    const T* a,
    const int* qweight,
    const int* qzeros,
    const T* scales,
    T* out,
    int rows,
    int K,
    int N,
    int zero_format,
    int tiles) {
  constexpr int kTM = 8;
  constexpr int kTN = 16;
  constexpr int kTK = 16;
  constexpr int kNAtoms = 4;
  constexpr int kSg = 16;
  constexpr int kTileN = kNAtoms * kTN;
  const int local = tiles * kSg;
  const int n_tiles = (N + kTileN - 1) / kTileN;
  const size_t b_elems = static_cast<size_t>(kGroup) * kTileN;
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<T, 1> a_tile(
        sycl::range<1>(kAligned ? size_t{1} : static_cast<size_t>(tiles) * kTM * kTK), cgh);
    sycl::local_accessor<T, 1> b_tile(sycl::range<1>(b_elems), cgh);
    sycl::local_accessor<float, 1> c_tile(sycl::range<1>(static_cast<size_t>(tiles) * kTM * kTN), cgh);
    GptqOnePassXmx<T, kAligned> fn{
        a, qweight, qzeros, scales, out, rows, K, N, zero_format, a_tile, b_tile, c_tile};
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(n_tiles) * static_cast<size_t>(local)),
            sycl::range<1>(static_cast<size_t>(local))),
        fn);
  });
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

// Expand one INT4 word once. Every hidden row reuses those eight weights.
inline void gptq_dequant8(int packed, float zero, float scale, float out[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = (static_cast<float>((packed >> (i * 4)) & 0x0F) - zero) * scale;
  }
}

template <typename T>
inline float gptq_lane(uint16_t bits) {
  if constexpr (std::is_same_v<T, sycl::half>) {
    return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
  } else {
    return sycl::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
  }
}

template <typename T>
inline sycl::vec<uint32_t, 4> gptq_load8(const T* row, int k) {
  sycl::vec<uint32_t, 4> bits;
  bits.load(0, reinterpret_cast<const uint32_t*>(row + k));
  return bits;
}

template <typename T>
inline float gptq_dot_ready(sycl::vec<uint32_t, 4> bits, const float w[8]) {
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t u = bits[i];
    acc += gptq_lane<T>(static_cast<uint16_t>(u)) * w[i * 2];
    acc += gptq_lane<T>(static_cast<uint16_t>(u >> 16)) * w[i * 2 + 1];
  }
  return acc;
}

template <typename T>
inline void gptq_apply4(
    sycl::vec<uint32_t, 4> bits,
    const float w0[8],
    const float w1[8],
    const float w2[8],
    const float w3[8],
    float& a,
    float& b,
    float& c,
    float& d) {
  float act[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t u = bits[i];
    act[i * 2] = gptq_lane<T>(static_cast<uint16_t>(u));
    act[i * 2 + 1] = gptq_lane<T>(static_cast<uint16_t>(u >> 16));
  }
  float s0 = 0.f;
  float s1 = 0.f;
  float s2 = 0.f;
  float s3 = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    s0 += act[i] * w0[i];
    s1 += act[i] * w1[i];
    s2 += act[i] * w2[i];
    s3 += act[i] * w3[i];
  }
  a += s0;
  b += s1;
  c += s2;
  d += s3;
}

template <typename T>
inline void gptq_quad_step(
    const int* const rows[4],
    const int safe[4],
    const T* const act[4],
    const T* scales,
    const int* qzeros,
    int word,
    int N,
    int zero_format,
    float* col0,
    float* col1,
    float* col2,
    float* col3) {
  const int group = word >> 4;
  float dw[4][8];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float scale = static_cast<float>(scales[group * N + safe[c]]);
    const float zero =
        static_cast<float>(effective_zero(load_z(qzeros, group, safe[c], N), zero_format));
    gptq_dequant8(rows[c][word], zero, scale, dw[c]);
  }
  const int k = word << 3;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    gptq_apply4<T>(
        gptq_load8(act[r], k), dw[0], dw[1], dw[2], dw[3], col0[r], col1[r], col2[r], col3[r]);
  }
}

// Several decode rows share one INT4 column. M==1 is GptqPackedGemv.
// The column is read once, four words ahead, the same pipeline as M=1.
// Each hidden row keeps its own accumulator.
template <typename T, int MT>
struct GptqPackedRows {
  static constexpr int kWork = 64;

  const T* a;
  const int* qweight;
  const int* qzeros;
  const T* scales;
  T* out;
  int M;
  int K;
  int N;
  int zero_format;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    (void)M;
    const int n = static_cast<int>(item.get_group(0));
    const int lid = static_cast<int>(item.get_local_id(0));
    float acc0[MT];
    float acc1[MT];
    float acc2[MT];
    float acc3[MT];
#pragma unroll
    for (int r = 0; r < MT; ++r) {
      acc0[r] = 0.f;
      acc1[r] = 0.f;
      acc2[r] = 0.f;
      acc3[r] = 0.f;
    }
    if (n < N) {
      const int words = K >> 3;
      const int* row = qweight + static_cast<int64_t>(n) * words;
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
        const int k0 = word << 3;
        const int k1 = (word + kWork) << 3;
        const int k2 = (word + 2 * kWork) << 3;
        const int k3 = (word + 3 * kWork) << 3;
        float dw0[8];
        float dw1[8];
        float dw2[8];
        float dw3[8];
        gptq_dequant8(w0, z0, s0, dw0);
        gptq_dequant8(w1, z1, s1, dw1);
        gptq_dequant8(w2, z2, s2, dw2);
        gptq_dequant8(w3, z3, s3, dw3);
        if constexpr (MT == 4) {
          const T* r0 = a;
          const T* r1 = a + static_cast<int64_t>(K);
          const T* r2 = a + static_cast<int64_t>(2) * K;
          const T* r3 = a + static_cast<int64_t>(3) * K;
          const auto b00 = gptq_load8(r0, k0);
          const auto b10 = gptq_load8(r1, k0);
          const auto b20 = gptq_load8(r2, k0);
          const auto b30 = gptq_load8(r3, k0);
          const auto b01 = gptq_load8(r0, k1);
          const auto b11 = gptq_load8(r1, k1);
          const auto b21 = gptq_load8(r2, k1);
          const auto b31 = gptq_load8(r3, k1);
          acc0[0] += gptq_dot_ready<T>(b00, dw0);
          acc0[1] += gptq_dot_ready<T>(b10, dw0);
          acc0[2] += gptq_dot_ready<T>(b20, dw0);
          acc0[3] += gptq_dot_ready<T>(b30, dw0);
          acc1[0] += gptq_dot_ready<T>(b01, dw1);
          acc1[1] += gptq_dot_ready<T>(b11, dw1);
          acc1[2] += gptq_dot_ready<T>(b21, dw1);
          acc1[3] += gptq_dot_ready<T>(b31, dw1);
          const auto b02 = gptq_load8(r0, k2);
          const auto b12 = gptq_load8(r1, k2);
          const auto b22 = gptq_load8(r2, k2);
          const auto b32 = gptq_load8(r3, k2);
          const auto b03 = gptq_load8(r0, k3);
          const auto b13 = gptq_load8(r1, k3);
          const auto b23 = gptq_load8(r2, k3);
          const auto b33 = gptq_load8(r3, k3);
          acc2[0] += gptq_dot_ready<T>(b02, dw2);
          acc2[1] += gptq_dot_ready<T>(b12, dw2);
          acc2[2] += gptq_dot_ready<T>(b22, dw2);
          acc2[3] += gptq_dot_ready<T>(b32, dw2);
          acc3[0] += gptq_dot_ready<T>(b03, dw3);
          acc3[1] += gptq_dot_ready<T>(b13, dw3);
          acc3[2] += gptq_dot_ready<T>(b23, dw3);
          acc3[3] += gptq_dot_ready<T>(b33, dw3);
        } else {
#pragma unroll
          for (int r = 0; r < MT; ++r) {
            const T* row_a = a + static_cast<int64_t>(r) * K;
#pragma unroll
            for (int nib = 0; nib < 8; ++nib) {
              acc0[r] += static_cast<float>(row_a[k0 + nib]) * dw0[nib];
              acc1[r] += static_cast<float>(row_a[k1 + nib]) * dw1[nib];
              acc2[r] += static_cast<float>(row_a[k2 + nib]) * dw2[nib];
              acc3[r] += static_cast<float>(row_a[k3 + nib]) * dw3[nib];
            }
          }
        }
      }
      for (int word = lid + ((words / (4 * kWork)) * 4 * kWork); word < words; word += kWork) {
        const int group = word >> 4;
        const float scale = static_cast<float>(scales[group * N + n]);
        const float zero = static_cast<float>(effective_zero(load_z(qzeros, group, n, N), zero_format));
        const int bits = row[word];
        const int k = word << 3;
        float dw[8];
        gptq_dequant8(bits, zero, scale, dw);
        if constexpr (MT == 4) {
          const auto t0 = gptq_load8(a, k);
          const auto t1 = gptq_load8(a + static_cast<int64_t>(K), k);
          const auto t2 = gptq_load8(a + static_cast<int64_t>(2) * K, k);
          const auto t3 = gptq_load8(a + static_cast<int64_t>(3) * K, k);
          acc0[0] += gptq_dot_ready<T>(t0, dw);
          acc0[1] += gptq_dot_ready<T>(t1, dw);
          acc0[2] += gptq_dot_ready<T>(t2, dw);
          acc0[3] += gptq_dot_ready<T>(t3, dw);
        } else {
#pragma unroll
          for (int r = 0; r < MT; ++r) {
            const T* row_a = a + static_cast<int64_t>(r) * K;
#pragma unroll
            for (int nib = 0; nib < 8; ++nib) {
              acc0[r] += static_cast<float>(row_a[k + nib]) * dw[nib];
            }
          }
        }
      }
    }
#pragma unroll
    for (int r = 0; r < MT; ++r) {
      const float sum = sycl::reduce_over_group(
          item.get_group(), acc0[r] + acc1[r] + acc2[r] + acc3[r], sycl::plus<float>());
      if (n < N && lid == 0) {
        out[static_cast<int64_t>(r) * N + n] = static_cast<T>(sum);
      }
    }
  }
};

// M=4: one work-group owns two adjacent columns and applies each activation
// block to both before the next load. Weight streams stay coalesced because
// every lane still reads the same K index of each column.
template <typename T>
struct GptqPackedPair4 {
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
    const int n0 = static_cast<int>(item.get_group(0)) * 2;
    const int n1 = n0 + 1;
    const int lid = static_cast<int>(item.get_local_id(0));
    float a0[4], a1[4], a2[4], a3[4];
    float b0[4], b1[4], b2[4], b3[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      a0[r] = 0.f;
      a1[r] = 0.f;
      a2[r] = 0.f;
      a3[r] = 0.f;
      b0[r] = 0.f;
      b1[r] = 0.f;
      b2[r] = 0.f;
      b3[r] = 0.f;
    }
    if (n0 < N) {
      const bool live1 = n1 < N;
      const int words = K >> 3;
      const int* row0 = qweight + static_cast<int64_t>(n0) * words;
      const int* row1 = qweight + static_cast<int64_t>(live1 ? n1 : n0) * words;
      const T* r0 = a;
      const T* r1 = a + static_cast<int64_t>(K);
      const T* r2 = a + static_cast<int64_t>(2) * K;
      const T* r3 = a + static_cast<int64_t>(3) * K;
      for (int word = lid; word + 3 * kWork < words; word += 4 * kWork) {
        const int u0 = row0[word];
        const int u1 = row0[word + kWork];
        const int u2 = row0[word + 2 * kWork];
        const int u3 = row0[word + 3 * kWork];
        const int v0 = row1[word];
        const int v1 = row1[word + kWork];
        const int v2 = row1[word + 2 * kWork];
        const int v3 = row1[word + 3 * kWork];
        const int g0 = word >> 4;
        const int g1 = (word + kWork) >> 4;
        const int g2 = (word + 2 * kWork) >> 4;
        const int g3 = (word + 3 * kWork) >> 4;
        const float su0 = static_cast<float>(scales[g0 * N + n0]);
        const float su1 = static_cast<float>(scales[g1 * N + n0]);
        const float su2 = static_cast<float>(scales[g2 * N + n0]);
        const float su3 = static_cast<float>(scales[g3 * N + n0]);
        const float sv0 = static_cast<float>(scales[g0 * N + (live1 ? n1 : n0)]);
        const float sv1 = static_cast<float>(scales[g1 * N + (live1 ? n1 : n0)]);
        const float sv2 = static_cast<float>(scales[g2 * N + (live1 ? n1 : n0)]);
        const float sv3 = static_cast<float>(scales[g3 * N + (live1 ? n1 : n0)]);
        const float zu0 = static_cast<float>(effective_zero(load_z(qzeros, g0, n0, N), zero_format));
        const float zu1 = static_cast<float>(effective_zero(load_z(qzeros, g1, n0, N), zero_format));
        const float zu2 = static_cast<float>(effective_zero(load_z(qzeros, g2, n0, N), zero_format));
        const float zu3 = static_cast<float>(effective_zero(load_z(qzeros, g3, n0, N), zero_format));
        const float zv0 = static_cast<float>(effective_zero(load_z(qzeros, g0, live1 ? n1 : n0, N), zero_format));
        const float zv1 = static_cast<float>(effective_zero(load_z(qzeros, g1, live1 ? n1 : n0, N), zero_format));
        const float zv2 = static_cast<float>(effective_zero(load_z(qzeros, g2, live1 ? n1 : n0, N), zero_format));
        const float zv3 = static_cast<float>(effective_zero(load_z(qzeros, g3, live1 ? n1 : n0, N), zero_format));
        const int k0 = word << 3;
        const int k1 = (word + kWork) << 3;
        const int k2 = (word + 2 * kWork) << 3;
        const int k3 = (word + 3 * kWork) << 3;
        const auto p00 = gptq_load8(r0, k0);
        const auto p10 = gptq_load8(r1, k0);
        const auto p20 = gptq_load8(r2, k0);
        const auto p30 = gptq_load8(r3, k0);
        const auto p01 = gptq_load8(r0, k1);
        const auto p11 = gptq_load8(r1, k1);
        const auto p21 = gptq_load8(r2, k1);
        const auto p31 = gptq_load8(r3, k1);
        float dw[8];
        gptq_dequant8(u0, zu0, su0, dw);
        a0[0] += gptq_dot_ready<T>(p00, dw);
        a0[1] += gptq_dot_ready<T>(p10, dw);
        a0[2] += gptq_dot_ready<T>(p20, dw);
        a0[3] += gptq_dot_ready<T>(p30, dw);
        gptq_dequant8(v0, zv0, sv0, dw);
        b0[0] += gptq_dot_ready<T>(p00, dw);
        b0[1] += gptq_dot_ready<T>(p10, dw);
        b0[2] += gptq_dot_ready<T>(p20, dw);
        b0[3] += gptq_dot_ready<T>(p30, dw);
        gptq_dequant8(u1, zu1, su1, dw);
        a1[0] += gptq_dot_ready<T>(p01, dw);
        a1[1] += gptq_dot_ready<T>(p11, dw);
        a1[2] += gptq_dot_ready<T>(p21, dw);
        a1[3] += gptq_dot_ready<T>(p31, dw);
        gptq_dequant8(v1, zv1, sv1, dw);
        b1[0] += gptq_dot_ready<T>(p01, dw);
        b1[1] += gptq_dot_ready<T>(p11, dw);
        b1[2] += gptq_dot_ready<T>(p21, dw);
        b1[3] += gptq_dot_ready<T>(p31, dw);
        const auto p02 = gptq_load8(r0, k2);
        const auto p12 = gptq_load8(r1, k2);
        const auto p22 = gptq_load8(r2, k2);
        const auto p32 = gptq_load8(r3, k2);
        const auto p03 = gptq_load8(r0, k3);
        const auto p13 = gptq_load8(r1, k3);
        const auto p23 = gptq_load8(r2, k3);
        const auto p33 = gptq_load8(r3, k3);
        gptq_dequant8(u2, zu2, su2, dw);
        a2[0] += gptq_dot_ready<T>(p02, dw);
        a2[1] += gptq_dot_ready<T>(p12, dw);
        a2[2] += gptq_dot_ready<T>(p22, dw);
        a2[3] += gptq_dot_ready<T>(p32, dw);
        gptq_dequant8(v2, zv2, sv2, dw);
        b2[0] += gptq_dot_ready<T>(p02, dw);
        b2[1] += gptq_dot_ready<T>(p12, dw);
        b2[2] += gptq_dot_ready<T>(p22, dw);
        b2[3] += gptq_dot_ready<T>(p32, dw);
        gptq_dequant8(u3, zu3, su3, dw);
        a3[0] += gptq_dot_ready<T>(p03, dw);
        a3[1] += gptq_dot_ready<T>(p13, dw);
        a3[2] += gptq_dot_ready<T>(p23, dw);
        a3[3] += gptq_dot_ready<T>(p33, dw);
        gptq_dequant8(v3, zv3, sv3, dw);
        b3[0] += gptq_dot_ready<T>(p03, dw);
        b3[1] += gptq_dot_ready<T>(p13, dw);
        b3[2] += gptq_dot_ready<T>(p23, dw);
        b3[3] += gptq_dot_ready<T>(p33, dw);
      }
      for (int word = lid + ((words / (4 * kWork)) * 4 * kWork); word < words; word += kWork) {
        const int group = word >> 4;
        const float su = static_cast<float>(scales[group * N + n0]);
        const float sv = static_cast<float>(scales[group * N + (live1 ? n1 : n0)]);
        const float zu = static_cast<float>(effective_zero(load_z(qzeros, group, n0, N), zero_format));
        const float zv = static_cast<float>(effective_zero(load_z(qzeros, group, live1 ? n1 : n0, N), zero_format));
        const int ku = row0[word];
        const int kv = row1[word];
        const int k = word << 3;
        const auto t0 = gptq_load8(r0, k);
        const auto t1 = gptq_load8(r1, k);
        const auto t2 = gptq_load8(r2, k);
        const auto t3 = gptq_load8(r3, k);
        float dw[8];
        gptq_dequant8(ku, zu, su, dw);
        a0[0] += gptq_dot_ready<T>(t0, dw);
        a0[1] += gptq_dot_ready<T>(t1, dw);
        a0[2] += gptq_dot_ready<T>(t2, dw);
        a0[3] += gptq_dot_ready<T>(t3, dw);
        gptq_dequant8(kv, zv, sv, dw);
        b0[0] += gptq_dot_ready<T>(t0, dw);
        b0[1] += gptq_dot_ready<T>(t1, dw);
        b0[2] += gptq_dot_ready<T>(t2, dw);
        b0[3] += gptq_dot_ready<T>(t3, dw);
      }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const float sum0 = sycl::reduce_over_group(
          item.get_group(), a0[r] + a1[r] + a2[r] + a3[r], sycl::plus<float>());
      const float sum1 = sycl::reduce_over_group(
          item.get_group(), b0[r] + b1[r] + b2[r] + b3[r], sycl::plus<float>());
      if (lid == 0 && n0 < N) {
        out[static_cast<int64_t>(r) * N + n0] = static_cast<T>(sum0);
      }
      if (lid == 0 && n1 < N) {
        out[static_cast<int64_t>(r) * N + n1] = static_cast<T>(sum1);
      }
    }
  }
};

// Wide M=4 layers only take a couple of K passes per thread. Each pass owns
// four adjacent columns. One accumulator per column and row stays in registers;
// the two K steps add into those same accumulators.
template <typename T>
struct GptqPackedQuad4 {
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
    const int base = static_cast<int>(item.get_group(0)) * 4;
    const int lid = static_cast<int>(item.get_local_id(0));
    const int nc[4] = {base, base + 1, base + 2, base + 3};
    float a0[4], b0[4], c0[4], d0[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      a0[r] = 0.f;
      b0[r] = 0.f;
      c0[r] = 0.f;
      d0[r] = 0.f;
    }
    if (base < N) {
      const int words = K >> 3;
      const int safe[4] = {
          nc[0] < N ? nc[0] : base,
          nc[1] < N ? nc[1] : base,
          nc[2] < N ? nc[2] : base,
          nc[3] < N ? nc[3] : base};
      const int* rows[4] = {
          qweight + static_cast<int64_t>(safe[0]) * words,
          qweight + static_cast<int64_t>(safe[1]) * words,
          qweight + static_cast<int64_t>(safe[2]) * words,
          qweight + static_cast<int64_t>(safe[3]) * words};
      const T* act[4] = {
          a,
          a + static_cast<int64_t>(K),
          a + static_cast<int64_t>(2) * K,
          a + static_cast<int64_t>(3) * K};
      for (int word = lid; word + kWork < words; word += 2 * kWork) {
        gptq_quad_step<T>(rows, safe, act, scales, qzeros, word, N, zero_format, a0, b0, c0, d0);
        gptq_quad_step<T>(
            rows, safe, act, scales, qzeros, word + kWork, N, zero_format, a0, b0, c0, d0);
      }
      for (int word = lid + ((words / (2 * kWork)) * 2 * kWork); word < words; word += kWork) {
        gptq_quad_step<T>(rows, safe, act, scales, qzeros, word, N, zero_format, a0, b0, c0, d0);
      }
    }
    float* cols[4] = {a0, b0, c0, d0};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const float sum = sycl::reduce_over_group(item.get_group(), cols[c][r], sycl::plus<float>());
        if (lid == 0 && nc[c] < N) {
          out[static_cast<int64_t>(r) * N + nc[c]] = static_cast<T>(sum);
        }
      }
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

#if defined(SGL_GPTQ_DPAS)
template <typename T>
void launch_gptq_xmx(
    sycl::queue& queue,
    const T* a,
    const int* qweight,
    const int* qzeros,
    const T* scales,
    T* out,
    int M,
    int K,
    int N,
    int zero_format) {
  constexpr int kTM = 8;
  constexpr int kTN = 16;
  constexpr int kNAtoms = 4;
  constexpr int kMaxTiles = 64;
  // Unaligned chunks also stage A in SLM. Cap them so A+B+C stays under 60 KB.
  constexpr int kUnalignedMaxTiles = 48;
  constexpr int kTileN = kNAtoms * kTN;
  auto& stats = bandwidth_stats();
  stats.gemm.fetch_add(1);
  xmx_stats().launches.fetch_add(1);
  const bool n_aligned = N % kTileN == 0;
  int cursor = 0;
  while (cursor < M) {
    int rows = std::min(M - cursor, kMaxTiles * kTM);
    int tiles = (rows + kTM - 1) / kTM;
    bool aligned = rows % kTM == 0 && n_aligned;
    if (!aligned && tiles > kUnalignedMaxTiles) {
      rows = std::min(M - cursor, kUnalignedMaxTiles * kTM);
      tiles = (rows + kTM - 1) / kTM;
      aligned = false;
    }
    const T* a_rows = a + static_cast<int64_t>(cursor) * K;
    T* out_rows = out + static_cast<int64_t>(cursor) * N;
    if (aligned) {
      submit_gptq_xmx_chunk<T, true>(
          queue, a_rows, qweight, qzeros, scales, out_rows, rows, K, N, zero_format, tiles);
    } else {
      submit_gptq_xmx_chunk<T, false>(
          queue, a_rows, qweight, qzeros, scales, out_rows, rows, K, N, zero_format, tiles);
    }
    cursor += rows;
  }
}
#endif

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
  // DPAS atom is 8 rows. Contiguous bf16 M>=8 (speculative verify and
  // prefill) takes the one-pass XMX GEMM. M==1 stays the packed GEMV, which
  // is also the draft-model forward. M==2 and M==4 stay the multi-row EU GEMV.
#if defined(SGL_GPTQ_DPAS)
  const bool xmx_rows = std::is_same_v<T, sycl::ext::oneapi::bfloat16> && M >= 8;
#else
  const bool xmx_rows = false;
#endif
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
  if (contiguous && !xmx_rows && (M == 2 || M == 4 || M == 8)) {
    bandwidth_stats().gemm.fetch_add(1);
    auto launch_rows = [&](auto kernel) {
      constexpr int kWork = decltype(kernel)::kWork;
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(N) * kWork), sycl::range<1>(kWork)), kernel);
      });
    };
    if (M == 2) {
      launch_rows(GptqPackedRows<T, 2>{a, qweight, qzeros, scales, out, M, K, N, zero_format});
    } else if (M == 4) {
      constexpr int kWork = GptqPackedQuad4<T>::kWork;
      // Long K hides latency better with two columns and four pipeline stages.
      // Wide layers only take a couple of K passes, so four columns reuse each
      // activation block more times.
      if (K >= N) {
        const int pairs = (N + 1) / 2;
        queue.submit([&](sycl::handler& cgh) {
          GptqPackedPair4<T> fn{a, qweight, qzeros, scales, out, K, N, zero_format};
          cgh.parallel_for(
              sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(pairs) * kWork), sycl::range<1>(kWork)),
              fn);
        });
      } else {
        const int quads = (N + 3) / 4;
        queue.submit([&](sycl::handler& cgh) {
          GptqPackedQuad4<T> fn{a, qweight, qzeros, scales, out, K, N, zero_format};
          cgh.parallel_for(
              sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(quads) * kWork), sycl::range<1>(kWork)),
              fn);
        });
      }
    } else {
      launch_rows(GptqPackedRows<T, 8>{a, qweight, qzeros, scales, out, M, K, N, zero_format});
    }
    note_profile(queue, true);
    return;
  }
  if (contiguous) {
#if defined(SGL_GPTQ_DPAS)
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
      launch_gptq_xmx<T>(queue, a, qweight, qzeros, scales, out, M, K, N, zero_format);
      note_profile(queue, false);
      return;
    }
#endif
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
