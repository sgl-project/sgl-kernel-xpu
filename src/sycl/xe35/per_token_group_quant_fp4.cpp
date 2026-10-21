// SPDX-License-Identifier: Apache-2.0
/*
 * SYCL kernel for per-token group quantization to MXFP4 (E2M1) format.
 *
 * MXFP4 follows the OpenCompute MX (Microscaling) format specification:
 * - Data type: E2M1 (4-bit float with 2-bit exponent, 1-bit mantissa)
 * - Block size: 32 elements per scale factor
 * - Scale format: UE8M0 (unsigned 8-bit exponent-only, no mantissa)
 *
 * E2M1 representable values (magnitude): 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
 * With sign bit, we have 16 total values.
 *
 * Bit layout of E2M1:
 *   Bit 3: Sign (0 = positive, 1 = negative)
 *   Bits 0-2: Magnitude index (0-7)
 *
 * Two FP4 values are packed into a single uint8_t:
 *   - Lower nibble (bits 0-3): First value
 *   - Upper nibble (bits 4-7): Second value
 *
 * Rounding: Per OCP MX spec (section 5.3.3), FP4 conversion uses
 * roundTiesToEven — at midpoints between representable values, the
 * value with even mantissa (mantissa bit = 0) is chosen.
 */

#include "sgl_kernel/xe35/routing_quant.h"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <limits>
#include <sycl/sycl.hpp>

#include "sycl/SYCLHelpers.h"
#include "sycl/Utils.h"

namespace at::native::xpu {

constexpr float FLOAT4_E2M1_MAX_xe35 = 6.0f;

// Reduce `val` across the WIDTH consecutive lanes that own one quantization
// group. A group is a power-of-two-sized, WIDTH-aligned span of the sub-group
// (the work-group size is a whole number of groups), so the xor butterfly never
// crosses a group boundary and WIDTH must not exceed the sub-group size.
template <int WIDTH, typename T>
inline T QuantGroupReduceMaxFP4_xe35(T val, sycl::sub_group sg) {
  static_assert(WIDTH >= 1 && (WIDTH & (WIDTH - 1)) == 0, "WIDTH must be a power of two");
#pragma unroll
  for (int mask = WIDTH / 2; mask > 0; mask >>= 1) {
    val = sycl::fmax(val, sycl::permute_group_by_xor(sg, val, mask));
  }
  return val;
}

// E2M1 format (4-bit float): 1 sign bit, 2 exponent bits, 1 mantissa bit
// Encoding: exp=00 (subnormal), exp=01/10/11 (normal with bias=1)
// Result: bits[3]=sign, bits[2:1]=exponent, bits[0]=mantissa
//
// Representable values and their codes:
//   0.0 -> 0b000   (subnormal, m=0, even)
//   0.5 -> 0b001   (subnormal, m=1, odd)
//   1.0 -> 0b010   (e=01, m=0, even)
//   1.5 -> 0b011   (e=01, m=1, odd)
//   2.0 -> 0b100   (e=10, m=0, even)
//   3.0 -> 0b101   (e=10, m=1, odd)
//   4.0 -> 0b110   (e=11, m=0, even)
//   6.0 -> 0b111   (e=11, m=1, odd)
//
// RoundTiesToEven: At exact midpoints between two representable values,
// we round to the one whose mantissa bit is 0 (even).
//
// Midpoints and their rounding targets:
//   0.25  -> midpoint of (0.0, 0.5)  -> round to 0.0  (m=0, even)
//   0.75  -> midpoint of (0.5, 1.0)  -> round to 1.0  (m=0, even)
//   1.25  -> midpoint of (1.0, 1.5)  -> round to 1.0  (m=0, even)
//   1.75  -> midpoint of (1.5, 2.0)  -> round to 2.0  (m=0, even)
//   2.5   -> midpoint of (2.0, 3.0)  -> round to 2.0  (m=0, even)
//   3.5   -> midpoint of (3.0, 4.0)  -> round to 4.0  (m=0, even)
//   5.0   -> midpoint of (4.0, 6.0)  -> round to 4.0  (m=0, even)
inline uint8_t quantize_to_e2m1_xe35(float val) {
  uint8_t sign = (val < 0.0f) ? 1 : 0;
  float abs_val = sycl::fabs(val);

  // Branchless form of the interval ladder: the magnitude code is the number of
  // interval bounds `abs_val` has passed. The bounds are the midpoints between
  // consecutive representable magnitudes, and the comparison at each one is
  // strict or non-strict exactly so that a tie lands on the even-mantissa side
  // (roundTiesToEven, OCP MX 5.3.3):
  //
  //   code  1  2     3     4    5     6    7
  //   at   >.25 >=.75 >1.25 >=1.75 >2.5 >=3.5 >5.0
  //
  // e.g. 0.25 -> 0 (even), 0.75 -> 2 i.e. 1.0 (even), 2.5 -> 4 i.e. 2.0 (even).
  // This replaces an 8-way branch chain evaluated once per element; the packed
  // loop below calls it VEC_SIZE times per work item.
  const uint8_t code = static_cast<uint8_t>(
      static_cast<int>(abs_val > 0.25f) + static_cast<int>(abs_val >= 0.75f) + static_cast<int>(abs_val > 1.25f) +
      static_cast<int>(abs_val >= 1.75f) + static_cast<int>(abs_val > 2.5f) + static_cast<int>(abs_val >= 3.5f) +
      static_cast<int>(abs_val > 5.0f));

  return (sign << 3) | code;
}

// Use SYCL native vector type for efficient loading
template <typename T, uint32_t N>
using vec_t = sycl::vec<T, N>;

template <typename T>
inline T silu_mul_xe35(T a, T b) {
  float gate = static_cast<float>(a);
  float up = static_cast<float>(b);
  return static_cast<T>(gate / (1.0f + expf(-gate)) * up);
}

// `LANES_PER_GROUP` work items cooperate on one quantization group, each moving
// `GROUP_SIZE / LANES_PER_GROUP` elements as whole 16-byte vectors. At the
// default of 1 a lane owns a whole group, which is what this shape wants: the
// shared-exponent math (`log2`, `exp2`, the reciprocal) and the absmax reduction
// are per-group, so spreading a group over N lanes runs that arithmetic N times
// and adds N-1 sub-group permutes on top. One lane per group also turns the
// packed output into a single 16-byte store. Small surfaces are the exception;
// see `fp4_quant_lanes_per_group_xe35`.
//
// `FUSE_SILU_AND_MUL` folds SwiGLU into the quantizer: the input row is twice as
// wide as the output row, the gate half first and the up half `row_groups *
// GROUP_SIZE` elements later, which is the layout `silu_and_mul` and
// `sgl_per_token_group_quant_8bit_v2` already use. The group's absmax is taken
// over the *activated* values, so the scale is the one the fused row needs.
template <
    typename T,
    int GROUP_SIZE = 32,
    bool FUSE_SILU_AND_MUL = false,
    int LANES_PER_GROUP = 1,
    bool PADDED_GATHER = false,
    typename IndexTy = int64_t>
struct PerTokenGroupQuantFP4Kernel_xe35 : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr uint32_t VEC_SIZE = 16 / sizeof(T);
  static constexpr int32_t THREADS_PER_GROUP = LANES_PER_GROUP;
  static constexpr uint32_t ELEMS_PER_LANE = GROUP_SIZE / LANES_PER_GROUP;
  static constexpr uint32_t VECS_PER_LANE = ELEMS_PER_LANE / VEC_SIZE;
  static constexpr uint32_t PACKED_BYTES = ELEMS_PER_LANE / 2;
  static_assert(GROUP_SIZE % LANES_PER_GROUP == 0, "a group must split evenly across its lanes");
  static_assert(ELEMS_PER_LANE % VEC_SIZE == 0, "a lane's share must be a whole number of 16-byte loads");
  static_assert(
      THREADS_PER_GROUP >= 1 && (THREADS_PER_GROUP & (THREADS_PER_GROUP - 1)) == 0,
      "THREADS_PER_GROUP must be a power of two for the sub-group butterfly");
  static_assert(THREADS_PER_GROUP <= 32, "a group must fit inside one sub-group");
  static_assert(!(FUSE_SILU_AND_MUL && PADDED_GATHER), "the fused input transforms are mutually exclusive");

  PerTokenGroupQuantFP4Kernel_xe35(
      const T* input,
      const IndexTy* row_map,
      uint8_t* output_q,
      uint8_t* output_s,
      const int32_t* scale_map,
      int num_groups,
      int groups_per_block,
      int row_groups,
      float eps)
      : input(input),
        row_map(row_map),
        output_q(output_q),
        output_s(output_s),
        scale_map(scale_map),
        num_groups(num_groups),
        groups_per_block(groups_per_block),
        row_groups(row_groups),
        eps(eps) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {}

  [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t local_group_id = item.get_local_id(0) / THREADS_PER_GROUP;
    const int lane_id = item.get_local_id(0) % THREADS_PER_GROUP;

    const int64_t block_group_id = item.get_group(0) * groups_per_block;
    const int64_t global_group_id = block_group_id + local_group_id;

    if (global_group_id >= num_groups) return;

    // `global_group_id` always enumerates *output* groups. Without the fused
    // activation the input is the same surface, so the offsets coincide.
    const int64_t out_group_offset = global_group_id * GROUP_SIZE;
    int64_t in_group_offset = out_group_offset;
    if constexpr (FUSE_SILU_AND_MUL) {
      const int64_t row = global_group_id / row_groups;
      const int64_t g = global_group_id - row * row_groups;
      in_group_offset = (row * 2 * row_groups + g) * GROUP_SIZE;
    } else if constexpr (PADDED_GATHER) {
      const int64_t row = global_group_id / row_groups;
      const int64_t g = global_group_id - row * row_groups;
      const int64_t source_row = static_cast<int64_t>(row_map[row]);
      in_group_offset = (source_row * row_groups + g) * GROUP_SIZE;
    }

    const T* gate_ptr = input + in_group_offset + lane_id * ELEMS_PER_LANE;
    // Output is packed FP4 (2 values per byte), so the offset is halved.
    uint8_t* out_ptr = output_q + (out_group_offset / 2) + lane_id * PACKED_BYTES;
    // Scales are row-major by default: one byte per output group, contiguous per
    // row. A `scale_map` names, per row, the (first row, row count) of the
    // expert block holding it, and the scales go to the per-expert MN-major
    // surface the Xe3.5 MXFP grouped GEMM reads (mxfp4_moe_scale_layout.cpp):
    //   out[start * G + g * rows + (row - start)]
    // A negative row count marks alignment padding inside a block: its GEMM
    // output is discarded, so only a finite scale (2^-127) is stored and the
    // input and packed values are left untouched. A row count of 0 marks
    // buffer slack nothing reads, so the row is skipped entirely.
    uint8_t* scale_output = output_s + global_group_id;
    if (scale_map != nullptr) {
      const int64_t row = global_group_id / row_groups;
      const int64_t g = global_group_id - row * row_groups;
      const int64_t start = scale_map[2 * row];
      const int64_t rows = scale_map[2 * row + 1];
      if (rows == 0) return;
      scale_output = output_s + start * row_groups + g * sycl::abs(rows) + (row - start);
      if (rows < 0) {
        if (lane_id == 0) *scale_output = 0;
        return;
      }
    }

    using vec_type = vec_t<T, VEC_SIZE>;
    const int64_t up_stride = FUSE_SILU_AND_MUL ? static_cast<int64_t>(row_groups) * GROUP_SIZE : 0;

    float vals[ELEMS_PER_LANE];
    float local_absmax = eps;
#pragma unroll
    for (uint32_t v = 0; v < VECS_PER_LANE; ++v) {
      const T* vec_ptr = gate_ptr + v * VEC_SIZE;
      vec_type gate_vec;
      gate_vec.load(0, sycl::multi_ptr<const T, sycl::access::address_space::global_space>(vec_ptr));
      vec_type up_vec;
      if constexpr (FUSE_SILU_AND_MUL) {
        up_vec.load(0, sycl::multi_ptr<const T, sycl::access::address_space::global_space>(vec_ptr + up_stride));
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        float val = static_cast<float>(gate_vec[j]);
        if constexpr (FUSE_SILU_AND_MUL) {
          val = static_cast<float>(silu_mul_xe35<T>(gate_vec[j], up_vec[j]));
        }
        vals[v * VEC_SIZE + j] = val;
        local_absmax = sycl::fmax(local_absmax, sycl::fabs(val));
      }
    }

    // Reduce across the threads in the quantization group to find the maximum
    local_absmax = QuantGroupReduceMaxFP4_xe35<THREADS_PER_GROUP>(local_absmax, item.get_sub_group());

    // Shared exponent per OCP MX spec / Microsoft micro-scaling:
    //   shared_exp = floor(log2(absmax)) - E2M1_EMAX
    // where E2M1_EMAX = 2.  eps already lower-limits local_absmax so
    // log2 is well-defined.
    float log2_scale = sycl::floor(sycl::log2(local_absmax)) - 2.0f;
    int clamped_exponent = sycl::clamp(static_cast<int>(log2_scale), -127, 127);
    float scale_value = sycl::ldexp(1.0f, clamped_exponent);

    if (lane_id == 0) {
      // Store scale as UE8M0: exponent + 127 bias
      uint8_t scale_ue8m0 = static_cast<uint8_t>(clamped_exponent + 127);
      *scale_output = scale_ue8m0;
    }

    const float inv_scale = 1.0f / scale_value;

    // Quantize and pack: two E2M1 nibbles per byte, one vector store per lane.
    vec_t<uint8_t, PACKED_BYTES> packed;
#pragma unroll
    for (uint32_t j = 0; j < ELEMS_PER_LANE; j += 2) {
      uint8_t q0 = quantize_to_e2m1_xe35(vals[j] * inv_scale);
      uint8_t q1 = quantize_to_e2m1_xe35(vals[j + 1] * inv_scale);
      // Pack: first value in lower nibble, second in upper nibble
      // No masking needed — quantize_to_e2m1_xe35 returns values in [0, 15]
      packed[j / 2] = q0 | (q1 << 4);
    }
    packed.store(
        0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(out_ptr));
  }

 private:
  const T* input;
  const IndexTy* row_map;
  uint8_t* output_q;
  uint8_t* output_s;
  const int32_t* scale_map;
  int num_groups;
  int groups_per_block;
  int row_groups;
  float eps;
};

// GPU graph timing shows four lanes are useful for small fused BF16 surfaces
// across output widths 2048, 4096 and 8192. The first crossover is near
// 48 Ki groups, independent of how those groups are divided into rows.
// Plain quantization and padded gather have no stable four-lane advantage.
// One lane is the default for input dtypes without a measured crossover.
static int fp4_quant_lanes_per_group_xe35(int64_t num_groups, bool bf16_input, bool fused_swiglu) {
  constexpr int64_t kFourLaneFusedGroupLimit = 48 * 1024;
  return bf16_input && fused_swiglu && num_groups <= kFourLaneFusedGroupLimit ? 4 : 1;
}

// `scale_map` is an optional [rows, 2] int32 (block start, block rows) table
// redirecting each output row's scales; see the kernel. Offsets are not bounds-checked on the host
// (they are device data), so the caller sizes `output_s` for the surface it names.
static const int32_t* checked_scale_map_xe35(const std::optional<at::Tensor>& scale_map, int64_t rows, const char* op) {
  if (!scale_map.has_value()) return nullptr;
  const at::Tensor& m = *scale_map;
  TORCH_CHECK(m.device().is_xpu() && m.is_contiguous(), op, ": scale_map must be a contiguous XPU tensor");
  TORCH_CHECK(m.scalar_type() == at::kInt, op, ": scale_map must be int32, got ", m.scalar_type());
  TORCH_CHECK(
      m.dim() == 2 && m.size(0) == rows && m.size(1) == 2,
      op,
      ": scale_map must have shape [",
      rows,
      ", 2], got ",
      m.sizes());
  return m.data_ptr<int32_t>();
}

void sgl_per_token_group_quant_fp4_xe35(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    bool fuse_silu_and_mul,
    const std::optional<at::Tensor>& scale_map) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output_q);

  TORCH_CHECK(group_size == 32, "sgl_per_token_group_quant_fp4_xe35: group_size must be 32 for MXFP4, got ", group_size);

  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16 ||
          input.scalar_type() == at::ScalarType::Float,
      "sgl_per_token_group_quant_fp4_xe35: input dtype must be Float16, BFloat16, or Float32, got ",
      input.scalar_type());

  TORCH_CHECK(
      output_q.scalar_type() == at::ScalarType::Byte,
      "output_q must be uint8 (packed FP4), got ",
      output_q.scalar_type());
  TORCH_CHECK(
      output_s.scalar_type() == at::ScalarType::Byte,
      "output_s must be uint8 (UE8M0 scales), got ",
      output_s.scalar_type());

  TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
  TORCH_CHECK(
      input.size(-1) % group_size == 0,
      "sgl_per_token_group_quant_fp4_xe35: last dimension of input (",
      input.size(-1),
      ") must be divisible by group_size (",
      group_size,
      ")");

  // With the fused activation the input row carries the gate and up halves, so
  // it holds twice the elements the output groups cover.
  const int64_t in_per_out = fuse_silu_and_mul ? 2 : 1;
  TORCH_CHECK(
      input.numel() % (group_size * in_per_out) == 0,
      "sgl_per_token_group_quant_fp4_xe35: input numel (",
      input.numel(),
      ") must be divisible by group_size x ",
      in_per_out);
  const int num_groups = static_cast<int>(input.numel() / (group_size * in_per_out));

  // Output should be half the size (2 FP4 values per byte)
  CHECK_EQ(output_q.numel(), input.numel() / (2 * in_per_out));
  CHECK_EQ(output_s.numel(), num_groups);

  // Groups per output row, which is what the fused path needs to find the up
  // half of a row; `output_q` holds two nibbles per byte.
  const int row_groups = static_cast<int>(output_q.size(-1) * 2 / group_size);
  if (fuse_silu_and_mul) {
    TORCH_CHECK(input.dim() == 2 && output_q.dim() == 2, "fuse_silu_and_mul needs 2D input/output_q");
    TORCH_CHECK(
        input.size(-1) == output_q.size(-1) * 4,
        "sgl_per_token_group_quant_fp4_xe35: with fuse_silu_and_mul the input row (",
        input.size(-1),
        ") must be 2x the output row (",
        output_q.size(-1) * 2,
        " values)");
  }

  const int32_t* scale_map_ptr = checked_scale_map_xe35(scale_map, num_groups / row_groups, "sgl_per_token_group_quant_fp4_xe35");

  // Ensure eps is positive to prevent NaN from log2(0)
  float eps_f = static_cast<float>(eps);
  if (eps_f <= 0.0f) {
    eps_f = 1e-10f;
  }

  auto queue = dpcppGetCurrentQueue();
  const int lanes =
      fp4_quant_lanes_per_group_xe35(num_groups, input.scalar_type() == at::ScalarType::BFloat16, fuse_silu_and_mul);

// A block holds a whole number of quantization groups, so the sub-group
// butterfly in `QuantGroupReduceMaxFP4_xe35` stays inside one group (a no-op at the
// default one lane per group). The ladder aims at `kTargetThreads` work items
// and steps down to whatever divides `num_groups`.
#define LAUNCH_FP4_KERNEL_WITH_GROUP_SIZE(T, GS, FUSE, LANES)                                                       \
  do {                                                                                                              \
    using Kernel = PerTokenGroupQuantFP4Kernel_xe35<T, GS, FUSE, LANES>;                                                 \
    constexpr int kTargetThreads = 256;                                                                             \
    int groups_per_block = kTargetThreads / Kernel::THREADS_PER_GROUP;                                              \
    while (groups_per_block > 1 && num_groups % groups_per_block != 0) {                                            \
      groups_per_block >>= 1;                                                                                       \
    }                                                                                                               \
    const int num_blocks = num_groups / groups_per_block;                                                           \
    const int num_threads = groups_per_block * Kernel::THREADS_PER_GROUP;                                           \
    auto kernel = Kernel(                                                                                           \
        static_cast<const T*>(input.data_ptr()),                                                                    \
        nullptr,                                                                                                    \
        static_cast<uint8_t*>(output_q.data_ptr()),                                                                 \
        static_cast<uint8_t*>(output_s.data_ptr()),                                                                 \
        scale_map_ptr,                                                                                              \
        num_groups,                                                                                                 \
        groups_per_block,                                                                                           \
        row_groups,                                                                                                 \
        eps_f);                                                                                                     \
    sycl_kernel_submit(                                                                                             \
        sycl::range<1>(static_cast<size_t>(num_blocks) * num_threads), sycl::range<1>(num_threads), queue, kernel); \
  } while (0)

#define LAUNCH_FP4_KERNEL_LANES(T, FUSE)                 \
  do {                                                   \
    if (lanes == 4) {                                    \
      LAUNCH_FP4_KERNEL_WITH_GROUP_SIZE(T, 32, FUSE, 4); \
    } else {                                             \
      LAUNCH_FP4_KERNEL_WITH_GROUP_SIZE(T, 32, FUSE, 1); \
    }                                                    \
  } while (0)

#define LAUNCH_FP4_KERNEL(T)             \
  do {                                   \
    if (fuse_silu_and_mul) {             \
      LAUNCH_FP4_KERNEL_LANES(T, true);  \
    } else {                             \
      LAUNCH_FP4_KERNEL_LANES(T, false); \
    }                                    \
  } while (0)

  // Dispatch based on input type
  if (input.scalar_type() == at::ScalarType::Half) {
    LAUNCH_FP4_KERNEL(sycl::half);
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    LAUNCH_FP4_KERNEL(sycl::ext::oneapi::bfloat16);
  } else if (input.scalar_type() == at::ScalarType::Float) {
    LAUNCH_FP4_KERNEL(float);
  }

#undef LAUNCH_FP4_KERNEL
#undef LAUNCH_FP4_KERNEL_LANES
#undef LAUNCH_FP4_KERNEL_WITH_GROUP_SIZE
}

void sgl_padded_gather_per_token_group_quant_fp4_xe35(
    torch::Tensor input,
    torch::Tensor row_map,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    const std::optional<at::Tensor>& scale_map) {
  CHECK_INPUT(input);
  CHECK_INPUT(row_map);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);

  TORCH_CHECK(
      group_size == 32,
      "sgl_padded_gather_per_token_group_quant_fp4_xe35: group_size must be 32 for MXFP4, got ",
      group_size);
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16,
      "sgl_padded_gather_per_token_group_quant_fp4_xe35: input dtype must be Float16 or BFloat16, got ",
      input.scalar_type());
  TORCH_CHECK(
      row_map.scalar_type() == at::ScalarType::Int || row_map.scalar_type() == at::ScalarType::Long,
      "sgl_padded_gather_per_token_group_quant_fp4_xe35: row_map dtype must be Int32 or Int64, got ",
      row_map.scalar_type());
  TORCH_CHECK(
      output_q.scalar_type() == at::ScalarType::Byte,
      "sgl_padded_gather_per_token_group_quant_fp4_xe35: output_q must be uint8, got ",
      output_q.scalar_type());
  TORCH_CHECK(
      output_s.scalar_type() == at::ScalarType::Byte,
      "sgl_padded_gather_per_token_group_quant_fp4_xe35: output_s must be uint8, got ",
      output_s.scalar_type());
  TORCH_CHECK(input.dim() == 2, "sgl_padded_gather_per_token_group_quant_fp4_xe35: input must be 2D");
  TORCH_CHECK(row_map.dim() == 1, "sgl_padded_gather_per_token_group_quant_fp4_xe35: row_map must be 1D");
  TORCH_CHECK(
      output_q.dim() == 2 && output_s.dim() == 2, "sgl_padded_gather_per_token_group_quant_fp4_xe35: outputs must be 2D");
  TORCH_CHECK(
      input.size(1) > 0 && input.size(1) % group_size == 0,
      "sgl_padded_gather_per_token_group_quant_fp4_xe35: input width (",
      input.size(1),
      ") must be positive and divisible by ",
      group_size);

  const int64_t output_rows = row_map.numel();
  const int64_t row_width = input.size(1);
  const int64_t row_groups_i64 = row_width / group_size;
  TORCH_CHECK(
      output_q.size(0) == output_rows && output_q.size(1) == row_width / 2,
      "sgl_padded_gather_per_token_group_quant_fp4_xe35: output_q must have shape [",
      output_rows,
      ", ",
      row_width / 2,
      "], got ",
      output_q.sizes());
  TORCH_CHECK(
      output_s.size(0) == output_rows && output_s.size(1) == row_groups_i64,
      "sgl_padded_gather_per_token_group_quant_fp4_xe35: output_s must have shape [",
      output_rows,
      ", ",
      row_groups_i64,
      "], got ",
      output_s.sizes());
  TORCH_CHECK(
      input.get_device() == row_map.get_device() && input.get_device() == output_q.get_device() &&
          input.get_device() == output_s.get_device(),
      "sgl_padded_gather_per_token_group_quant_fp4_xe35: all tensors must be on the same XPU");

  if (output_rows == 0) return;

  TORCH_CHECK(
      input.size(0) > 0,
      "sgl_padded_gather_per_token_group_quant_fp4_xe35: input must have at least one row when row_map is nonempty");
  TORCH_CHECK(
      output_rows <= std::numeric_limits<int>::max() / row_groups_i64,
      "sgl_padded_gather_per_token_group_quant_fp4_xe35: output surface is too large");
  const int num_groups = static_cast<int>(output_rows * row_groups_i64);
  const int row_groups = static_cast<int>(row_groups_i64);
  const int32_t* scale_map_ptr =
      checked_scale_map_xe35(scale_map, output_rows, "sgl_padded_gather_per_token_group_quant_fp4_xe35");
  float eps_f = static_cast<float>(eps);
  if (eps_f <= 0.0f) eps_f = 1e-10f;

  auto queue = dpcppGetCurrentQueue();
#define LAUNCH_PADDED_GATHER_FP4(T, INDEX_T)                                                                        \
  do {                                                                                                              \
    using Kernel = PerTokenGroupQuantFP4Kernel_xe35<T, 32, false, 1, true, INDEX_T>;                                     \
    constexpr int kTargetThreads = 256;                                                                             \
    int groups_per_block = kTargetThreads / Kernel::THREADS_PER_GROUP;                                              \
    while (groups_per_block > 1 && num_groups % groups_per_block != 0) {                                            \
      groups_per_block >>= 1;                                                                                       \
    }                                                                                                               \
    const int num_blocks = num_groups / groups_per_block;                                                           \
    const int num_threads = groups_per_block * Kernel::THREADS_PER_GROUP;                                           \
    auto kernel = Kernel(                                                                                           \
        static_cast<const T*>(input.data_ptr()),                                                                    \
        static_cast<const INDEX_T*>(row_map.data_ptr()),                                                            \
        static_cast<uint8_t*>(output_q.data_ptr()),                                                                 \
        static_cast<uint8_t*>(output_s.data_ptr()),                                                                 \
        scale_map_ptr,                                                                                              \
        num_groups,                                                                                                 \
        groups_per_block,                                                                                           \
        row_groups,                                                                                                 \
        eps_f);                                                                                                     \
    sycl_kernel_submit(                                                                                             \
        sycl::range<1>(static_cast<size_t>(num_blocks) * num_threads), sycl::range<1>(num_threads), queue, kernel); \
  } while (0)

#define DISPATCH_PADDED_GATHER_INDEX(T)                 \
  do {                                                  \
    if (row_map.scalar_type() == at::ScalarType::Int) { \
      LAUNCH_PADDED_GATHER_FP4(T, int32_t);             \
    } else {                                            \
      LAUNCH_PADDED_GATHER_FP4(T, int64_t);             \
    }                                                   \
  } while (0)

  if (input.scalar_type() == at::ScalarType::Half) {
    DISPATCH_PADDED_GATHER_INDEX(sycl::half);
  } else {
    DISPATCH_PADDED_GATHER_INDEX(sycl::ext::oneapi::bfloat16);
  }

#undef DISPATCH_PADDED_GATHER_INDEX
#undef LAUNCH_PADDED_GATHER_FP4
}

}  // namespace at::native::xpu
