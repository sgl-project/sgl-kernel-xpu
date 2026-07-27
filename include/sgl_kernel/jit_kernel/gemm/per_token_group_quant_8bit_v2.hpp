/**
 * Per-Token-Group 8-bit Quantization v2 SYCL JIT Kernel for SGLang XPU
 *
 * DeepGEMM/DeepSeek-style grouped quantization to int8 or fp8 (e4m3fn), with:
 *   - group sizes 16 / 32 / 64 / 128 (THREADS_PER_SUBWARP = GROUP_SIZE / 16)
 *   - optional fused SiLU-and-mul: input [.., 2H] -> silu(gate)*up, then quantize
 *   - optional masked MoE layout: [num_experts, tokens_pad, 2H] (masked_m per expert)
 *   - optional UE8M0 (power-of-2) scales in a column-major, TMA-packed layout
 *   - one scale per group; 256-bit (32-byte) vectorized loads/stores
 *
 * Self-contained JIT port of the AOT kernel in
 * src/sycl/per_token_group_quant_8bit_v2.cpp. Kept numerically close to it:
 *   - 32-wide sub-group holding THREADS_PER_SUBWARP-lane logical groups; absmax
 *     reduced with an XOR butterfly (permute_group_by_xor) within each group
 *   - fp32 absmax / scale accumulation; eps == LOCAL_ABSMAX_ABS == 1e-10
 *   - NaiveScheduler (flattened 1-D grid) + MaskedLayoutScheduler (expert x
 *     token x chunk 3-D grid), same exec-config heuristics
 *
 * Differences from AOT (behaviorally identical):
 *   - FP8 e4m3fn conversion uses a self-contained c10-style routine instead of
 *     CUTLASS (bit-exact with torch's `.to(torch.float8_e4m3fn)` in range).
 *   - Input dtype and output dtype are compile-time specialized via -D macros;
 *     group_size and all runtime flags are arguments dispatched in the launcher,
 *     so one .so per (in,out) dtype serves every config (mirrors the CUDA JIT).
 */

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl_kernel {
namespace ptgq_v2 {

constexpr float LOCAL_ABSMAX_ABS = 1e-10f;
constexpr uint32_t INPUT_PRIMARY_VEC_NUM_BYTES = 32;

// c10-style round-to-nearest-even float -> e4m3fn (finite; NaN at 0x7f/0xff).
// Inputs are pre-clamped to +-448, so the saturation branch is unreached.
inline uint8_t fp32_to_e4m3fn(float f) {
  const uint32_t fp8_max_bits = static_cast<uint32_t>(1087) << 20;
  const uint32_t denorm_mask = static_cast<uint32_t>(141) << 23;
  uint32_t f_bits = ::sycl::bit_cast<uint32_t>(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & static_cast<uint32_t>(0x80000000);
  f_bits ^= sign;
  if (f_bits >= fp8_max_bits) {
    result = 0x7f;
  } else if (f_bits < (static_cast<uint32_t>(121) << 23)) {
    float shifted = ::sycl::bit_cast<float>(f_bits) + ::sycl::bit_cast<float>(denorm_mask);
    f_bits = ::sycl::bit_cast<uint32_t>(shifted);
    result = static_cast<uint8_t>(f_bits - denorm_mask);
  } else {
    uint8_t mant_odd = (f_bits >> 20) & 1u;
    f_bits += (static_cast<uint32_t>(7 - 127) << 23) + 0x7FFFFu;
    f_bits += mant_odd;
    result = static_cast<uint8_t>(f_bits >> 20);
  }
  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

struct dim3 {
  int x, y, z;
};

template <typename T>
struct DtypeInfo;
template <>
struct DtypeInfo<int8_t> {
  static constexpr float MIN = -128;
  static constexpr float MAX = 127;
  static constexpr bool kIsFp8 = false;
};
struct e4m3_tag {
  static constexpr float MIN = -448;
  static constexpr float MAX = 448;
  static constexpr bool kIsFp8 = true;
};

// ---------------------------------------------------------------------------
// Shared per-group compute: load (+ optional silu*mul), XOR-reduce absmax over
// the logical sub-group, compute the group scale, quantize + vectorized store.
// DST_INFO is DtypeInfo<int8_t> or e4m3_tag; output is written as uint8 bytes.
// ---------------------------------------------------------------------------
template <
    int GROUP_SIZE,
    int THREADS_PER_SUBWARP,
    typename T,
    typename DST_INFO,
    bool IS_COLUMN_MAJOR,
    bool SCALE_UE8M0,
    bool FUSE_SILU_AND_MUL,
    typename scale_packed_t>
struct MainKernel {
  const T* input;
  uint8_t* output_q;
  scale_packed_t* output_s;
  const int32_t* masked_m;
  float eps;
  float min_8bit;
  float max_8bit;
  int subwarps_per_block;
  int hidden_dim_num_groups;
  int scale_expert_stride;
  int scale_hidden_stride;
  int num_tokens_per_expert;

  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

  inline float silu(float val) const {
    float half = 0.5f * val;
    return half * (1.0f + ::sycl::tanh(half));
  }

  void compute(
      const int expert_idx,
      const int token_idx,
      const int hidden_dim_group_idx,
      const int lane_id,
      const int64_t input_group_start_offset,
      ::sycl::sub_group sg) const {
    constexpr uint32_t INPUT_PRIMARY_VEC_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / sizeof(T);
    constexpr uint32_t INPUT_PRIMARY_INT4_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / (4 * sizeof(int));

    const int offset_num_groups = expert_idx * num_tokens_per_expert * hidden_dim_num_groups +
                                  token_idx * hidden_dim_num_groups + hidden_dim_group_idx;

    using int4 = ::sycl::vec<int, 4>;
    int4 input_primary_int4[INPUT_PRIMARY_INT4_SIZE];
    T* input_primary_vec = reinterpret_cast<T*>(input_primary_int4);
    int4 input_secondary_int4[INPUT_PRIMARY_INT4_SIZE];
    T* input_secondary_vec = reinterpret_cast<T*>(input_secondary_int4);

    auto primary_base_ptr =
        reinterpret_cast<const int4*>(input + input_group_start_offset + lane_id * INPUT_PRIMARY_VEC_SIZE);
#pragma unroll
    for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
      input_primary_int4[j] = primary_base_ptr[j];
    }
    if constexpr (FUSE_SILU_AND_MUL) {
      const int secondary_offset = hidden_dim_num_groups * GROUP_SIZE;
      auto secondary_base_ptr = reinterpret_cast<const int4*>(
          input + input_group_start_offset + lane_id * INPUT_PRIMARY_VEC_SIZE + secondary_offset);
#pragma unroll
      for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
        input_secondary_int4[j] = secondary_base_ptr[j];
      }
    }

    constexpr int num_elems_per_pack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
    scale_element_t* scale_output;
    if constexpr (IS_COLUMN_MAJOR) {
      const int flat_idx = token_idx + hidden_dim_group_idx * num_tokens_per_expert;
      const int output_token_idx = flat_idx / hidden_dim_num_groups;
      const int output_group_idx = flat_idx % hidden_dim_num_groups;
      const int hidden_idx_packed = output_group_idx / num_elems_per_pack;
      const int pack_idx = output_group_idx % num_elems_per_pack;
      constexpr int scale_token_stride = 1;
      scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                     (expert_idx * scale_expert_stride * num_elems_per_pack +
                      hidden_idx_packed * scale_hidden_stride * num_elems_per_pack +
                      output_token_idx * scale_token_stride * num_elems_per_pack + pack_idx);
    } else {
      static_assert(!SCALE_UE8M0);
      scale_output = reinterpret_cast<scale_element_t*>(output_s) + offset_num_groups;
    }

    // TMA-alignment padding: last pack may have unused slots; zero them once.
    if constexpr (IS_COLUMN_MAJOR && SCALE_UE8M0) {
      const int flat_idx = token_idx + hidden_dim_group_idx * num_tokens_per_expert;
      const int output_group_idx = flat_idx % hidden_dim_num_groups;
      const int pack_idx = output_group_idx % num_elems_per_pack;
      const int remainder_num_groups = hidden_dim_num_groups % num_elems_per_pack;
      if (lane_id == 0 && remainder_num_groups != 0 && output_group_idx == hidden_dim_num_groups - 1) {
        for (int i = pack_idx + 1; i < num_elems_per_pack; i++) {
          *(scale_output + (i - pack_idx)) = 0;
        }
      }
    }

    float local_absmax = LOCAL_ABSMAX_ABS;
#pragma unroll
    for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
      float val;
      if constexpr (FUSE_SILU_AND_MUL) {
        T val_lowprec = static_cast<T>(silu(static_cast<float>(input_primary_vec[j]))) * input_secondary_vec[j];
        val = static_cast<float>(val_lowprec);
        input_primary_vec[j] = val_lowprec;
      } else {
        val = static_cast<float>(input_primary_vec[j]);
      }
      local_absmax = ::sycl::fmax(local_absmax, ::sycl::fabs(val));
    }

    // XOR-butterfly max within the logical THREADS_PER_SUBWARP-lane group.
    const uint32_t lane = sg.get_local_id()[0];
    const uint32_t logical_lane = lane & (THREADS_PER_SUBWARP - 1);
    const uint32_t group_base = lane & ~(THREADS_PER_SUBWARP - 1);
#pragma unroll
    for (int mask = THREADS_PER_SUBWARP / 2; mask > 0; mask >>= 1) {
      const uint32_t target_lane = group_base + (logical_lane ^ mask);
      const uint32_t xor_mask = lane ^ target_lane;
      float other_max = ::sycl::permute_group_by_xor(sg, local_absmax, xor_mask);
      local_absmax = ::sycl::fmax(local_absmax, other_max);
    }

    float y_s = local_absmax / max_8bit;
    scale_element_t y_s_quant;
    if constexpr (SCALE_UE8M0) {
      float exp_s = ::sycl::ceil(::sycl::log2(::sycl::fmax(y_s, 1e-10f)));
      y_s = ::sycl::exp2(exp_s);
      y_s_quant = static_cast<scale_element_t>(static_cast<int>(exp_s) + 127);
    } else {
      y_s_quant = y_s;
    }

    if (lane_id == 0) {
      *scale_output = y_s_quant;
    }

    const float inv_y_s = 1.0f / y_s;
    using output_vec_type = ::sycl::vec<uint8_t, INPUT_PRIMARY_VEC_SIZE>;
    output_vec_type output_vec;
#pragma unroll
    for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
      float q_val = ::sycl::fmin(::sycl::fmax(static_cast<float>(input_primary_vec[j]) * inv_y_s, min_8bit), max_8bit);
      if constexpr (DST_INFO::kIsFp8) {
        output_vec[j] = fp32_to_e4m3fn(q_val);
      } else {
        output_vec[j] = ::sycl::bit_cast<uint8_t>(static_cast<int8_t>(q_val));
      }
    }
    output_vec.store(
        0,
        ::sycl::address_space_cast<::sycl::access::address_space::global_space, ::sycl::access::decorated::yes>(
            output_q + static_cast<int64_t>(offset_num_groups) * GROUP_SIZE + lane_id * INPUT_PRIMARY_VEC_SIZE));
  }
};

// Naive path: flattened 1-D grid, one logical sub-group per group.
template <
    int GROUP_SIZE,
    int THREADS_PER_SUBWARP,
    typename T,
    typename DST_INFO,
    bool IS_COLUMN_MAJOR,
    bool SCALE_UE8M0,
    bool FUSE_SILU_AND_MUL,
    typename scale_packed_t>
struct NaiveKernel {
  MainKernel<
      GROUP_SIZE,
      THREADS_PER_SUBWARP,
      T,
      DST_INFO,
      IS_COLUMN_MAJOR,
      SCALE_UE8M0,
      FUSE_SILU_AND_MUL,
      scale_packed_t>
      k;

  [[sycl::reqd_sub_group_size(32)]] void operator()(::sycl::nd_item<3> item) const {
    constexpr int expert_idx = 0;
    const int threadIdx_x = item.get_local_linear_id();
    const int blockIdx_x = item.get_group().get_group_linear_id();

    const int64_t subwarp_id = threadIdx_x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx_x % THREADS_PER_SUBWARP;
    const int64_t group_id = static_cast<int64_t>(blockIdx_x) * k.subwarps_per_block + subwarp_id;

    const int token_idx = group_id / k.hidden_dim_num_groups;
    const int hidden_dim_group_idx = group_id % k.hidden_dim_num_groups;

    int64_t input_group_start_offset;
    if constexpr (FUSE_SILU_AND_MUL) {
      const int hidden_size = k.hidden_dim_num_groups * GROUP_SIZE;
      input_group_start_offset =
          static_cast<int64_t>(token_idx) * hidden_size * 2 + static_cast<int64_t>(hidden_dim_group_idx) * GROUP_SIZE;
    } else {
      input_group_start_offset = group_id * GROUP_SIZE;
    }

    k.compute(expert_idx, token_idx, hidden_dim_group_idx, lane_id, input_group_start_offset, item.get_sub_group());
  }
};

// Masked MoE path: 3-D grid (expert, token-start, chunk), grid-strided tokens.
template <
    int GROUP_SIZE,
    int THREADS_PER_SUBWARP,
    typename T,
    typename DST_INFO,
    bool IS_COLUMN_MAJOR,
    bool SCALE_UE8M0,
    bool FUSE_SILU_AND_MUL,
    typename scale_packed_t,
    int SUBWARPS_PER_BLOCK,
    int TOKEN_DIM_BLOCK_NUM_PER_EXPERT>
struct MaskedKernel {
  MainKernel<
      GROUP_SIZE,
      THREADS_PER_SUBWARP,
      T,
      DST_INFO,
      IS_COLUMN_MAJOR,
      SCALE_UE8M0,
      FUSE_SILU_AND_MUL,
      scale_packed_t>
      k;

  [[sycl::reqd_sub_group_size(32)]] void operator()(::sycl::nd_item<3> item) const {
    const int threadIdx_x = item.get_local_linear_id();
    const int64_t subwarp_id = threadIdx_x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx_x % THREADS_PER_SUBWARP;

    auto group = item.get_group();
    const int expert_idx = group.get_group_id(0);
    const int token_idx_start = group.get_group_id(1);
    const int chunk_id = group.get_group_id(2);

    const int64_t hidden_dim_group_idx = static_cast<int64_t>(chunk_id) * SUBWARPS_PER_BLOCK + subwarp_id;
    const int curr_expert_token_num = k.masked_m[expert_idx];

    for (int token_idx = token_idx_start; token_idx < curr_expert_token_num;
         token_idx += TOKEN_DIM_BLOCK_NUM_PER_EXPERT) {
      const int hidden_size = k.hidden_dim_num_groups * GROUP_SIZE;
      const int mul = FUSE_SILU_AND_MUL ? 2 : 1;
      const int64_t input_group_start_offset =
          static_cast<int64_t>(expert_idx) * k.num_tokens_per_expert * hidden_size * mul +
          static_cast<int64_t>(token_idx) * hidden_size * mul + static_cast<int64_t>(hidden_dim_group_idx) * GROUP_SIZE;
      k.compute(expert_idx, token_idx, hidden_dim_group_idx, lane_id, input_group_start_offset, item.get_sub_group());
    }
  }
};

struct NaiveScheduler {
  static void compute_exec_config(
      int threads_per_subwarp,
      int num_local_experts,
      int hidden_dim_num_groups,
      int num_groups,
      int& subwarps_per_block,
      dim3& grid,
      dim3& block) {
    subwarps_per_block = (num_groups % 16 == 0)  ? 16
                         : (num_groups % 8 == 0) ? 8
                         : (num_groups % 4 == 0) ? 4
                         : (num_groups % 2 == 0) ? 2
                                                 : 1;
    grid = dim3{num_groups / subwarps_per_block, 1, 1};
    block = dim3{subwarps_per_block * threads_per_subwarp, 1, 1};
  }
};

struct MaskedLayoutScheduler {
  static constexpr int TOKEN_DIM_BLOCK_NUM_PER_EXPERT = 1024;
  static constexpr int SUBWARPS_PER_BLOCK = 16;

  static void compute_exec_config(
      int threads_per_subwarp,
      int num_local_experts,
      int hidden_dim_num_groups,
      int num_groups,
      int& subwarps_per_block,
      dim3& grid,
      dim3& block) {
    subwarps_per_block = SUBWARPS_PER_BLOCK;
    grid = dim3{hidden_dim_num_groups / subwarps_per_block, TOKEN_DIM_BLOCK_NUM_PER_EXPERT, num_local_experts};
    block = dim3{subwarps_per_block * threads_per_subwarp, 1, 1};
  }
};

// ---------------------------------------------------------------------------
// Templated launch for one fully-resolved config.
// ---------------------------------------------------------------------------
template <
    bool MASKED,
    int GROUP_SIZE,
    int THREADS_PER_SUBWARP,
    typename T,
    typename DST_INFO,
    bool IS_COLUMN_MAJOR,
    bool SCALE_UE8M0,
    bool FUSE_SILU_AND_MUL>
void launch_one(
    ::sycl::queue& queue,
    const void* input,
    void* output_q,
    void* output_s,
    const int32_t* masked_m,
    float eps,
    float min_8bit,
    float max_8bit,
    int num_local_experts,
    int hidden_dim_num_groups,
    int num_groups,
    int scale_expert_stride,
    int scale_hidden_stride,
    int num_tokens_per_expert) {
  using scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>;
  using Main = MainKernel<
      GROUP_SIZE,
      THREADS_PER_SUBWARP,
      T,
      DST_INFO,
      IS_COLUMN_MAJOR,
      SCALE_UE8M0,
      FUSE_SILU_AND_MUL,
      scale_packed_t>;

  int subwarps_per_block;
  dim3 grid, block;
  using Scheduler = std::conditional_t<MASKED, MaskedLayoutScheduler, NaiveScheduler>;
  Scheduler::compute_exec_config(
      THREADS_PER_SUBWARP, num_local_experts, hidden_dim_num_groups, num_groups, subwarps_per_block, grid, block);

  Main k{
      static_cast<const T*>(input),
      static_cast<uint8_t*>(output_q),
      static_cast<scale_packed_t*>(output_s),
      masked_m,
      eps,
      min_8bit,
      max_8bit,
      subwarps_per_block,
      hidden_dim_num_groups,
      scale_expert_stride,
      scale_hidden_stride,
      num_tokens_per_expert};

  if constexpr (MASKED) {
    MaskedKernel<
        GROUP_SIZE,
        THREADS_PER_SUBWARP,
        T,
        DST_INFO,
        IS_COLUMN_MAJOR,
        SCALE_UE8M0,
        FUSE_SILU_AND_MUL,
        scale_packed_t,
        MaskedLayoutScheduler::SUBWARPS_PER_BLOCK,
        MaskedLayoutScheduler::TOKEN_DIM_BLOCK_NUM_PER_EXPERT>
        task{k};
    const ::sycl::range<3> global_range{
        static_cast<size_t>(grid.z),
        static_cast<size_t>(grid.y),
        static_cast<size_t>(grid.x) * static_cast<size_t>(block.x)};
    const ::sycl::range<3> local_range{1, 1, static_cast<size_t>(block.x)};
    queue.submit([&](::sycl::handler& cgh) { cgh.parallel_for(::sycl::nd_range<3>(global_range, local_range), task); });
  } else {
    NaiveKernel<
        GROUP_SIZE,
        THREADS_PER_SUBWARP,
        T,
        DST_INFO,
        IS_COLUMN_MAJOR,
        SCALE_UE8M0,
        FUSE_SILU_AND_MUL,
        scale_packed_t>
        task{k};
    const ::sycl::range<3> global_range{static_cast<size_t>(grid.x) * static_cast<size_t>(block.x), 1, 1};
    const ::sycl::range<3> local_range{static_cast<size_t>(block.x), 1, 1};
    queue.submit([&](::sycl::handler& cgh) { cgh.parallel_for(::sycl::nd_range<3>(global_range, local_range), task); });
  }
}

// Resolve the runtime bool flags to compile-time template args (matches the AOT
// LAUNCH_KERNEL nesting: only column-major supports ue8m0; masked implies both).
template <int GROUP_SIZE, typename T, typename DST_INFO>
void dispatch_flags(
    ::sycl::queue& queue,
    const void* input,
    void* output_q,
    void* output_s,
    const int32_t* masked_m,
    float eps,
    float min_8bit,
    float max_8bit,
    bool is_column_major,
    bool scale_ue8m0,
    bool fuse_silu_and_mul,
    bool masked_layout,
    int num_local_experts,
    int hidden_dim_num_groups,
    int num_groups,
    int scale_expert_stride,
    int scale_hidden_stride,
    int num_tokens_per_expert) {
  constexpr int TPS = GROUP_SIZE / 16;
#define SGL_PTGQV2_CALL(MASKED, COL, UE8M0, SILU)                     \
  launch_one<MASKED, GROUP_SIZE, TPS, T, DST_INFO, COL, UE8M0, SILU>( \
      queue,                                                          \
      input,                                                          \
      output_q,                                                       \
      output_s,                                                       \
      masked_m,                                                       \
      eps,                                                            \
      min_8bit,                                                       \
      max_8bit,                                                       \
      num_local_experts,                                              \
      hidden_dim_num_groups,                                          \
      num_groups,                                                     \
      scale_expert_stride,                                            \
      scale_hidden_stride,                                            \
      num_tokens_per_expert)

  if (is_column_major) {
    if (scale_ue8m0) {
      if (fuse_silu_and_mul) {
        if (masked_layout) {
          SGL_PTGQV2_CALL(true, true, true, true);
        } else {
          SGL_PTGQV2_CALL(false, true, true, true);
        }
      } else {
        SGL_PTGQV2_CALL(false, true, true, false);
      }
    } else {
      SGL_PTGQV2_CALL(false, true, false, false);
    }
  } else {
    SGL_PTGQV2_CALL(false, false, false, false);
  }
#undef SGL_PTGQV2_CALL
}

template <typename T, typename DST_INFO>
void dispatch_group_size(
    ::sycl::queue& queue,
    const void* input,
    void* output_q,
    void* output_s,
    const int32_t* masked_m,
    int64_t group_size,
    float eps,
    float min_8bit,
    float max_8bit,
    bool is_column_major,
    bool scale_ue8m0,
    bool fuse_silu_and_mul,
    bool masked_layout,
    int num_local_experts,
    int hidden_dim_num_groups,
    int num_groups,
    int scale_expert_stride,
    int scale_hidden_stride,
    int num_tokens_per_expert) {
#define SGL_PTGQV2_GS(GS)          \
  dispatch_flags<GS, T, DST_INFO>( \
      queue,                       \
      input,                       \
      output_q,                    \
      output_s,                    \
      masked_m,                    \
      eps,                         \
      min_8bit,                    \
      max_8bit,                    \
      is_column_major,             \
      scale_ue8m0,                 \
      fuse_silu_and_mul,           \
      masked_layout,               \
      num_local_experts,           \
      hidden_dim_num_groups,       \
      num_groups,                  \
      scale_expert_stride,         \
      scale_hidden_stride,         \
      num_tokens_per_expert)
  switch (group_size) {
    case 16:
      SGL_PTGQV2_GS(16);
      break;
    case 32:
      SGL_PTGQV2_GS(32);
      break;
    case 64:
      SGL_PTGQV2_GS(64);
      break;
    case 128:
      SGL_PTGQV2_GS(128);
      break;
    default:
      break;  // validated in Python
  }
#undef SGL_PTGQV2_GS
}

}  // namespace ptgq_v2

// ---------------------------------------------------------------------------
// C API for the Python ctypes wrapper. One symbol per (in dtype, out dtype);
// group_size and all flags/shape scalars are runtime arguments.
// ---------------------------------------------------------------------------
#define _DEFINE_PTGQV2_FORWARD(IN_SUFFIX, IN_T, OUT_SUFFIX, DST_INFO)               \
  extern "C" void per_token_group_quant_8bit_v2_forward_##IN_SUFFIX##_##OUT_SUFFIX( \
      void* queue_ptr,                                                              \
      const void* input,                                                            \
      void* output_q,                                                               \
      void* output_s,                                                               \
      const void* masked_m,                                                         \
      int64_t group_size,                                                           \
      float eps,                                                                    \
      float min_8bit,                                                               \
      float max_8bit,                                                               \
      int32_t scale_ue8m0,                                                          \
      int32_t fuse_silu_and_mul,                                                    \
      int32_t masked_layout,                                                        \
      int32_t is_column_major,                                                      \
      int64_t num_local_experts,                                                    \
      int64_t hidden_dim_num_groups,                                                \
      int64_t num_groups,                                                           \
      int64_t scale_expert_stride,                                                  \
      int64_t scale_hidden_stride,                                                  \
      int64_t num_tokens_per_expert) {                                              \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);                          \
    ptgq_v2::dispatch_group_size<IN_T, DST_INFO>(                                   \
        queue,                                                                      \
        input,                                                                      \
        output_q,                                                                   \
        output_s,                                                                   \
        static_cast<const int32_t*>(masked_m),                                      \
        group_size,                                                                 \
        eps,                                                                        \
        min_8bit,                                                                   \
        max_8bit,                                                                   \
        is_column_major != 0,                                                       \
        scale_ue8m0 != 0,                                                           \
        fuse_silu_and_mul != 0,                                                     \
        masked_layout != 0,                                                         \
        static_cast<int>(num_local_experts),                                        \
        static_cast<int>(hidden_dim_num_groups),                                    \
        static_cast<int>(num_groups),                                               \
        static_cast<int>(scale_expert_stride),                                      \
        static_cast<int>(scale_hidden_stride),                                      \
        static_cast<int>(num_tokens_per_expert));                                   \
  }
#define DEFINE_PTGQV2_FORWARD(IN_SUFFIX, IN_T, OUT_SUFFIX, DST_INFO) \
  _DEFINE_PTGQV2_FORWARD(IN_SUFFIX, IN_T, OUT_SUFFIX, DST_INFO)

#define DEFINE_PTGQV2_IN(IN_SUFFIX, IN_T)                                  \
  DEFINE_PTGQV2_FORWARD(IN_SUFFIX, IN_T, int8, ptgq_v2::DtypeInfo<int8_t>) \
  DEFINE_PTGQV2_FORWARD(IN_SUFFIX, IN_T, fp8, ptgq_v2::e4m3_tag)

#if defined(SGL_PTGQV2_IN_fp16) || defined(SGL_PTGQV2_IN_bf16)
#if defined(SGL_PTGQV2_IN_fp16)
#define _PTGQV2_IN_SUFFIX fp16
#define _PTGQV2_IN_T ::sycl::half
#else
#define _PTGQV2_IN_SUFFIX bf16
#define _PTGQV2_IN_T ::sycl::ext::oneapi::bfloat16
#endif
#if defined(SGL_PTGQV2_OUT_fp8)
DEFINE_PTGQV2_FORWARD(_PTGQV2_IN_SUFFIX, _PTGQV2_IN_T, fp8, ptgq_v2::e4m3_tag)
#elif defined(SGL_PTGQV2_OUT_int8)
DEFINE_PTGQV2_FORWARD(_PTGQV2_IN_SUFFIX, _PTGQV2_IN_T, int8, ptgq_v2::DtypeInfo<int8_t>)
#else
DEFINE_PTGQV2_IN(_PTGQV2_IN_SUFFIX, _PTGQV2_IN_T)
#endif
#else
DEFINE_PTGQV2_IN(fp16, ::sycl::half)
DEFINE_PTGQV2_IN(bf16, ::sycl::ext::oneapi::bfloat16)
#endif

#undef DEFINE_PTGQV2_IN
#undef DEFINE_PTGQV2_FORWARD
#undef _DEFINE_PTGQV2_FORWARD

}  // namespace sycl_kernel
}  // namespace sgl
