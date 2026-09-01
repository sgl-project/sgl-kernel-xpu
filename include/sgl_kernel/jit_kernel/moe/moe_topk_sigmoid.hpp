/**
 * Fused Sigmoid Top-K MoE Gate SYCL JIT Kernel for SGLang XPU
 *
 * Fuses sigmoid + top-k selection + (optional) bias-aware ranking, renormalize,
 * routed scaling, and a fused shared expert into a single kernel:
 *
 *   scores = sigmoid(gating_output)                  (fp32)
 *   rank   = scores + correction_bias                (bias affects selection only)
 *   pick top-k experts per token; output weights are the *unbiased* sigmoid
 *   optionally append a shared expert and renormalize by routed_scaling_factor
 *
 *   gating_output : [num_tokens, num_experts]   (fp32/fp16/bf16)
 *   topk_weights  : [num_tokens, topk]          (fp32, written)
 *   topk_ids      : [num_tokens, topk]          (int32, written)
 *   correction_bias : [num_experts] fp32 or null
 *
 * Self-contained JIT port of the AOT kernel in src/sycl/TopKSigMoid.cpp. Keeps
 * both AOT code paths, selected at runtime by num_experts (dtype is the only
 * compile-time specialization, so one .so per dtype serves every expert count,
 * mirroring the CUDA JIT moe_topk_sigmoid.cuh design):
 *   - TopkGatingSigmoid<NUM_EXPERTS>: power-of-2 experts (1..256), warp-per-rows,
 *     vectorized interleaved loads, butterfly argmax over the sub-group.
 *   - FusedTopkSigmoid: general fallback (non-pow2 / >256), one sub-group per
 *     token, fp32 ranking scores.
 * fp32 sigmoid/ranking throughout; no .wait() in the launcher.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl_kernel {
namespace topk_sigmoid {

static constexpr int kSubGroupSize = 32;
static constexpr int kWarpSize = kSubGroupSize;

#define SGL_TOPK_MAX(a, b) ((a) > (b) ? (a) : (b))
#define SGL_TOPK_MIN(a, b) ((a) < (b) ? (a) : (b))

inline int div_up(int a, int b) {
  return (a + b - 1) / b;
}

template <int Dim, typename KernelT>
inline void submit_nd(::sycl::queue& q, ::sycl::range<Dim> global, ::sycl::range<Dim> local, const KernelT& ker) {
  q.submit([&](::sycl::handler& cgh) { cgh.parallel_for(::sycl::nd_range<Dim>(global, local), ker); });
}

// Aligned array for vectorized loads (avoids a CUTLASS dependency).
template <typename T, int N, int Alignment = static_cast<int>(sizeof(T) * N)>
class alignas(Alignment) AlignedArray {
  T data[N];
};

// ---------------------------------------------------------------------------
// General fallback: one sub-group per token, fp32 ranking. Handles any
// num_experts (used for non-power-of-2 / >256).
// ---------------------------------------------------------------------------
template <typename T>
struct FusedTopkSigmoid {
  static constexpr int sub_group_size = kSubGroupSize;
  static constexpr int max_group_size = 1024;
  static constexpr int malloc_per_item = 8;
  static constexpr float kNegInfinity = -std::numeric_limits<float>::infinity();

  FusedTopkSigmoid(
      float* topk_weights,
      int* topk_ids,
      const T* gating_output,
      const float* correction_bias,
      const bool renormalize,
      const float routed_scaling_factor,
      const int num_fused_shared_experts,
      const int tokens,
      const int experts,
      const int top_k)
      : topk_weights(topk_weights),
        topk_ids(topk_ids),
        gating_output(gating_output),
        correction_bias(correction_bias),
        renormalize(renormalize),
        routed_scaling_factor(routed_scaling_factor),
        num_fused_shared_experts(num_fused_shared_experts),
        tokens(tokens),
        experts(experts),
        top_k(top_k) {}

  static inline ::sycl::nd_range<3> get_nd_range(const int tokens, const int experts) {
    int calc_per_item = div_up(experts, sub_group_size);
    int group_size = div_up(experts, calc_per_item);
    group_size = group_size < sub_group_size ? sub_group_size : group_size;
    group_size = group_size < max_group_size ? group_size : max_group_size;
    int sub_groups_per_group = div_up(group_size, sub_group_size);
    group_size = sub_groups_per_group * sub_group_size;
    int global_size = div_up(tokens, sub_groups_per_group);
    ::sycl::range<3> local(1, 1, group_size);
    ::sycl::range<3> global(1, 1, global_size);
    return ::sycl::nd_range<3>(global * local, local);
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void operator()(::sycl::nd_item<3> item) const {
    int group_id = item.get_group_linear_id();
    int local_range = item.get_local_range(2);
    int sub_groups_per_group = local_range / sub_group_size;
    int calc_per_item = (experts + sub_group_size - 1) / sub_group_size;

    ::sycl::sub_group sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();

    int tid = group_id * sub_groups_per_group + sg_id;
    if (tid >= tokens) return;

    float local_elems[malloc_per_item];
    int local_idx[malloc_per_item];

    int start_offset = sg_local_id * calc_per_item;
    int local_num = calc_per_item;
    if (start_offset + local_num >= experts) local_num = experts - start_offset;
    if (local_num < 0) local_num = 0;

    for (int e = 0; e < calc_per_item; ++e) {
      local_elems[e] = kNegInfinity;
      local_idx[e] = -1;
    }

    for (int e = 0; e < local_num; ++e) {
      int expert_id = start_offset + e;
      float logit = static_cast<float>(gating_output[tid * experts + expert_id]);
      float sig = 1.0f / (1.0f + ::sycl::native::exp(-logit));
      local_idx[e] = expert_id;
      if (correction_bias != nullptr) sig += correction_bias[expert_id];
      local_elems[e] = sig;
    }

    float topk_weights_local[malloc_per_item];
    int topk_ids_local[malloc_per_item];

    const int routed_topk = top_k - num_fused_shared_experts;
    float row_sum_for_renormalize = 0.0f;

    for (int k = 0; k < routed_topk; ++k) {
      float k_max = kNegInfinity;
      int k_max_idx = -1;
      int remove_ix = -1;

      for (int e = 0; e < calc_per_item; ++e) {
        float my_val = local_elems[e];
        int my_idx = local_idx[e];

        for (int offset = sub_group_size / 2; offset > 0; offset /= 2) {
          float other_val = ::sycl::permute_group_by_xor(sg, my_val, offset);
          int other_idx = ::sycl::permute_group_by_xor(sg, my_idx, offset);
          if (other_val > my_val || (other_val == my_val && other_idx < my_idx)) {
            my_val = other_val;
            my_idx = other_idx;
          }
        }

        if (my_val > k_max || (my_val == k_max && my_idx < k_max_idx)) {
          k_max = my_val;
          k_max_idx = my_idx;
          remove_ix = (k_max_idx == local_idx[e]) ? e : -1;
        }
      }
      topk_weights_local[k] = k_max;
      topk_ids_local[k] = k_max_idx;

      if (remove_ix != -1) {
        local_elems[remove_ix] = kNegInfinity;
        local_idx[remove_ix] = -1;
      }
    }

    if (correction_bias != nullptr) {
      for (int i = 0; i < routed_topk; ++i) {
        int id = topk_ids_local[i];
        topk_weights_local[i] -= correction_bias[id];
      }
    }

    for (int i = 0; i < routed_topk; ++i) {
      row_sum_for_renormalize += topk_weights_local[i];
    }

    if (sg_local_id == 0) {
      int offset = tid * top_k;
      for (int i = 0; i < routed_topk; ++i) {
        topk_weights[offset + i] = topk_weights_local[i];
        topk_ids[offset + i] = (topk_ids_local[i] >= 0) ? topk_ids_local[i] : 0;
      }

      if (num_fused_shared_experts > 0) {
        const int shared_idx = offset + routed_topk;
        topk_ids[shared_idx] = experts;
        topk_weights[shared_idx] = renormalize ? 1.0f : row_sum_for_renormalize / routed_scaling_factor;
      }

      if (renormalize) {
        const float scale = routed_scaling_factor / (row_sum_for_renormalize + 1e-20f);
        for (int i = 0; i < routed_topk; ++i) {
          topk_weights[offset + i] *= scale;
        }
      }
    }
  }

  float* topk_weights;
  int* topk_ids;
  const T* gating_output;
  const float* correction_bias;
  const bool renormalize;
  const float routed_scaling_factor;
  const int num_fused_shared_experts;
  const int tokens;
  const int experts;
  const int top_k;
};

// ---------------------------------------------------------------------------
// Power-of-2-experts optimized path: warp packs multiple rows, vectorized
// interleaved loads, butterfly argmax across THREADS_PER_ROW.
// ---------------------------------------------------------------------------
template <typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
struct TopkGatingSigmoid {
  [[sycl::reqd_sub_group_size(kSubGroupSize)]] void operator()(::sycl::nd_item<2> item) const {
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static_assert(VPT % ELTS_PER_LDG == 0, "");
    static_assert(kWarpSize % THREADS_PER_ROW == 0, "");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "");
    static_assert(THREADS_PER_ROW <= kWarpSize, "");

    static constexpr int ELTS_PER_WARP = kWarpSize * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "");

    auto sg = item.get_sub_group();
    auto local_id_x = item.get_local_id(1);
    auto local_id_y = item.get_local_id(0);
    auto group_id_x = item.get_group(1);

    const int cta_base_row = group_id_x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + local_id_y * ROWS_PER_WARP;
    const int thread_row_in_warp = local_id_x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;
    const int topk = k + num_fused_shared_experts;

    if (thread_row >= num_rows) return;
    const bool row_is_active = finished ? !finished[thread_row] : true;

    const T* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = local_id_x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<T, ELTS_PER_LDG>;

    T row_chunk_temp[VPT];
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk_temp);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    float row_chunk[VPT];
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      float val = static_cast<float>(row_chunk_temp[ii]);
      val = 1.0f / (1.0f + ::sycl::native::exp(-val));
      if (correction_bias != nullptr) {
        const int group_id = ii / ELTS_PER_LDG;
        const int local_id = ii % ELTS_PER_LDG;
        const int expert_idx = first_elt_read_by_thread + group_id * THREADS_PER_ROW * ELTS_PER_LDG + local_id;
        val = val + correction_bias[expert_idx];
      }
      row_chunk[ii] = val;
    }

    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    float row_sum_for_renormalize = 0;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      float max_val = row_chunk[0];
      int expert = start_col;
#pragma unroll
      for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          float val = row_chunk[ldg * ELTS_PER_LDG + ii];
          if (val > max_val) {
            max_val = val;
            expert = col + ii;
          }
        }
      }

#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max = ::sycl::permute_group_by_xor(sg, max_val, mask);
        int other_expert = ::sycl::permute_group_by_xor(sg, expert, mask);
        if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
          max_val = other_max;
          expert = other_expert;
        }
      }

      if (thread_group_idx == 0) {
        const bool node_uses_expert = expert >= start_expert && expert < end_expert;
        const bool should_process_row = row_is_active && node_uses_expert;

        const int idx = topk * thread_row + k_idx;
        if (correction_bias != nullptr) {
          max_val -= correction_bias[expert];
        }
        output[idx] = max_val;
        indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
        row_sum_for_renormalize += max_val;
      }

      if (k_idx + 1 < k) {
        const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
        const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
        if (thread_group_idx == thread_to_clear_in_group) {
          const int offset_for_expert = expert % ELTS_PER_LDG;
          row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
        }
      }
    }

    if (num_fused_shared_experts > 0 && thread_group_idx == 0) {
      const int last_idx = topk * thread_row + k;
      output[last_idx] = renormalize ? 1.0f : row_sum_for_renormalize / routed_scaling_factor;
      indices[last_idx] = NUM_EXPERTS;
    }

    if (renormalize && thread_group_idx == 0) {
      float row_sum_for_renormalize_inv = routed_scaling_factor / (row_sum_for_renormalize + 1e-20f);
#pragma unroll
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        const int idx = topk * thread_row + k_idx;
        output[idx] = output[idx] * row_sum_for_renormalize_inv;
      }
    }
  }

  const T* input;
  const bool* finished;
  float* output;
  const int num_rows;
  int* indices;
  const float* correction_bias;
  const int k;
  const int start_expert;
  const int end_expert;
  const bool renormalize;
  const float routed_scaling_factor;
  const int num_fused_shared_experts;
};

namespace detail {
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static_assert(EXPERTS / (ELTS_PER_LDG * kWarpSize) == 0 || EXPERTS % (ELTS_PER_LDG * kWarpSize) == 0, "");
  static constexpr int VECs_PER_THREAD = SGL_TOPK_MAX(1, EXPERTS / (ELTS_PER_LDG * kWarpSize));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = kWarpSize / THREADS_PER_ROW;
};
}  // namespace detail

template <typename T, int NUM_EXPERTS, int WARPS_PER_TB>
void launch_topk_gating_sigmoid(
    ::sycl::queue& queue,
    const T* input,
    const bool* finished,
    float* output,
    int* indices,
    const float* correction_bias,
    const int num_rows,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize,
    const float routed_scaling_factor,
    const int num_fused_shared_experts) {
  static constexpr std::size_t MAX_BYTES_PER_LDG = 16;
  static constexpr int BYTES_PER_LDG = SGL_TOPK_MIN(MAX_BYTES_PER_LDG, sizeof(T) * NUM_EXPERTS);
  using Constants = detail::TopkConstants<T, NUM_EXPERTS, BYTES_PER_LDG>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  ::sycl::range<2> grid(1, num_blocks);
  ::sycl::range<2> block(WARPS_PER_TB, kWarpSize);

  TopkGatingSigmoid<T, VPT, NUM_EXPERTS, WARPS_PER_TB, BYTES_PER_LDG> task{
      input,
      finished,
      output,
      num_rows,
      indices,
      correction_bias,
      k,
      start_expert,
      end_expert,
      renormalize,
      routed_scaling_factor,
      num_fused_shared_experts};

  submit_nd<2>(queue, grid * block, block, task);
}

template <typename T>
void launch_fused_topk_sigmoid(
    ::sycl::queue& queue,
    const T* gating_output,
    float* topk_weights,
    int* topk_indices,
    const float* correction_bias,
    const bool renormalize,
    const float routed_scaling_factor,
    const int num_fused_shared_experts,
    const int top_k,
    const int num_tokens,
    const int num_experts) {
  using Kernel = FusedTopkSigmoid<T>;
  auto range = Kernel::get_nd_range(num_tokens, num_experts);
  Kernel task(
      topk_weights,
      topk_indices,
      gating_output,
      correction_bias,
      renormalize,
      routed_scaling_factor,
      num_fused_shared_experts,
      num_tokens,
      num_experts,
      top_k);
  submit_nd<3>(queue, range.get_global_range(), range.get_local_range(), task);
}

// ---------------------------------------------------------------------------
// Runtime dispatch on num_experts (power-of-2 fast path 1..256, else fallback).
// ---------------------------------------------------------------------------
template <typename T>
void fused_topk_sigmoid(
    ::sycl::queue& queue,
    const T* gating_output,
    float* topk_weights,
    int* topk_indices,
    const float* correction_bias,
    const bool renormalize,
    const float routed_scaling_factor,
    const int num_fused_shared_experts,
    const int num_tokens,
    const int num_experts,
    const int topk) {
  constexpr int WARPS_PER_TB = 4;

#define SGL_LAUNCH_GATING_SIGMOID(NUM_EXPERTS)              \
  launch_topk_gating_sigmoid<T, NUM_EXPERTS, WARPS_PER_TB>( \
      queue,                                                \
      gating_output,                                        \
      nullptr,                                              \
      topk_weights,                                         \
      topk_indices,                                         \
      correction_bias,                                      \
      num_tokens,                                           \
      topk - num_fused_shared_experts,                      \
      0,                                                    \
      num_experts,                                          \
      renormalize,                                          \
      routed_scaling_factor,                                \
      num_fused_shared_experts)

  switch (num_experts) {
    case 1:
      SGL_LAUNCH_GATING_SIGMOID(1);
      break;
    case 2:
      SGL_LAUNCH_GATING_SIGMOID(2);
      break;
    case 4:
      SGL_LAUNCH_GATING_SIGMOID(4);
      break;
    case 8:
      SGL_LAUNCH_GATING_SIGMOID(8);
      break;
    case 16:
      SGL_LAUNCH_GATING_SIGMOID(16);
      break;
    case 32:
      SGL_LAUNCH_GATING_SIGMOID(32);
      break;
    case 64:
      SGL_LAUNCH_GATING_SIGMOID(64);
      break;
    case 128:
      SGL_LAUNCH_GATING_SIGMOID(128);
      break;
    case 256:
      SGL_LAUNCH_GATING_SIGMOID(256);
      break;
    default:
      launch_fused_topk_sigmoid<T>(
          queue,
          gating_output,
          topk_weights,
          topk_indices,
          correction_bias,
          renormalize,
          routed_scaling_factor,
          num_fused_shared_experts,
          topk,
          num_tokens,
          num_experts);
  }
#undef SGL_LAUNCH_GATING_SIGMOID
}

}  // namespace topk_sigmoid

// ---------------------------------------------------------------------------
// C API for the Python ctypes wrapper. One symbol per dtype; num_experts and
// all runtime flags are arguments (correction_bias may be null).
// ---------------------------------------------------------------------------
#define DEFINE_TOPK_SIGMOID_FORWARD(DTYPE_SUFFIX, DTYPE)   \
  extern "C" void topk_sigmoid_forward_##DTYPE_SUFFIX(     \
      void* queue_ptr,                                     \
      const void* gating_output,                           \
      void* topk_weights,                                  \
      void* topk_ids,                                      \
      const void* correction_bias,                         \
      int64_t num_tokens,                                  \
      int64_t num_experts,                                 \
      int64_t topk,                                        \
      int32_t renormalize,                                 \
      float routed_scaling_factor,                         \
      int64_t num_fused_shared_experts) {                  \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr); \
    topk_sigmoid::fused_topk_sigmoid<DTYPE>(               \
        queue,                                             \
        static_cast<const DTYPE*>(gating_output),          \
        static_cast<float*>(topk_weights),                 \
        static_cast<int*>(topk_ids),                       \
        static_cast<const float*>(correction_bias),        \
        renormalize != 0,                                  \
        routed_scaling_factor,                             \
        static_cast<int>(num_fused_shared_experts),        \
        static_cast<int>(num_tokens),                      \
        static_cast<int>(num_experts),                     \
        static_cast<int>(topk));                           \
  }

#if defined(SGL_TOPK_SIGMOID_DTYPE_fp32)
DEFINE_TOPK_SIGMOID_FORWARD(fp32, float)
#elif defined(SGL_TOPK_SIGMOID_DTYPE_fp16)
DEFINE_TOPK_SIGMOID_FORWARD(fp16, ::sycl::half)
#elif defined(SGL_TOPK_SIGMOID_DTYPE_bf16)
DEFINE_TOPK_SIGMOID_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16)
#else
DEFINE_TOPK_SIGMOID_FORWARD(fp32, float)
DEFINE_TOPK_SIGMOID_FORWARD(fp16, ::sycl::half)
DEFINE_TOPK_SIGMOID_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16)
#endif

#undef DEFINE_TOPK_SIGMOID_FORWARD
#undef SGL_TOPK_MAX
#undef SGL_TOPK_MIN

}  // namespace sycl_kernel
}  // namespace sgl
