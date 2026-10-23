#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sycl/sycl.hpp>
#include <vector>

#include "MemoryAccess.h"
#include "Norm.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
#include <cutlass/util/GPU_Clock.hpp>
#endif

namespace at::native::xpu {
// Flatten tensor to 2D (M, N) for the kernel.  If the tensor is already 2D it
// is returned unchanged; 3D tensors are viewed as 2D.  Uses view() so that the
// returned tensor always shares storage with the original (no copy).
static inline Tensor flatten_to_2d(const Tensor& t, int64_t M, int64_t N) {
  if (t.dim() == 2) {
    return t;
  }
  return t.view({M, N});
}

// Describes how a flattened row index (0 .. M-1) maps to a byte offset on a
// 2D/3D/4D tensor without requiring a contiguous copy.  A row is decomposed
// into up to 3 nested indices (outer, inner0, inner1), from outermost to
// innermost:
//
//   outer  = r / (inner0_size * inner1_size)
//   inner0 = (r % (inner0_size * inner1_size)) / inner1_size
//   inner1 = r % inner1_size
//   offset(r) = outer * batch_stride + inner0 * inner0_stride + inner1 * inner1_stride
//
// For 2D and flattenable 3D tensors, inner0_size == inner1_size == 1 so the
// formula collapses to `offset(r) = r * batch_stride`.  For non-flattenable
// 3D tensors (e.g. a per-head slice of a packed QKV buffer reshaped to
// (tokens, heads, head_dim)) inner0_size/inner0_stride describe the second
// dim, with inner1_size == 1.  4D tensors additionally use inner1 when the
// leading dim (dim 0) is independent of dim 1 (e.g. a batched QKV slice),
// otherwise dim 0 folds away and only the 2-level form is needed.
struct RowStrides {
  int64_t batch_stride;
  int64_t inner0_size;
  int64_t inner0_stride;
  int64_t inner1_size;
  int64_t inner1_stride;
};

static inline RowStrides get_row_strides(const Tensor& t) {
  TORCH_CHECK(
      t.dim() == 2 || t.dim() == 3 || t.dim() == 4, "get_row_strides: expected a 2D/3D/4D tensor, got ", t.dim(), "D");
  if (t.dim() == 2) {
    return {t.stride(0), 1, 0, 1, 0};
  }

  // dim0/dim1 always participate in both the 3D and 4D shapes, so the fold
  // check between them is shared; dim2 only exists for 4D.
  int64_t d0_size = t.size(0);
  int64_t d0_stride = t.stride(0);
  int64_t d1_size = t.size(1);
  int64_t d1_stride = t.stride(1);
  bool d0_folds_into_d1 = (d0_size == 1) || (d0_stride == d1_size * d1_stride);

  if (t.dim() == 3) {
    if (d0_folds_into_d1) {
      return {d1_stride, 1, 0, 1, 0};
    }
    return {d0_stride, d1_size, d1_stride, 1, 0};
  }

  // t.dim() == 4: shape (d0, d1, d2, plane); row = d0*d1*d2.
  int64_t d2_size = t.size(2);
  int64_t d2_stride = t.stride(2);
  bool d1_folds_into_d2 = (d1_size == 1) || (d1_stride == d2_size * d2_stride);

  if (d0_folds_into_d1 && d1_folds_into_d2) {
    // Everything collapses into a single stride.
    return {d2_stride, 1, 0, 1, 0};
  }
  if (d0_folds_into_d1) {
    // dim0 contributes nothing; (d1, d2) alone need the 2-level form.
    return {d1_stride, d2_size, d2_stride, 1, 0};
  }
  // Fully independent: need the full 3-level (d0, d1, d2) decomposition.
  return {d0_stride, d1_size, d1_stride, d2_size, d2_stride};
}

// Shared 3-level row-offset computation used by every kernel site below; see
// the RowStrides comment for the (outer, inner0, inner1) decomposition.
template <typename index_t>
static inline index_t compute_row_offset(
    index_t row,
    index_t batch_stride,
    index_t inner0_size,
    index_t inner0_stride,
    index_t inner1_size,
    index_t inner1_stride) {
  index_t combined_inner = inner0_size * inner1_size;
  index_t outer_idx = row / combined_inner;
  index_t rem = row % combined_inner;
  index_t inner0_idx = rem / inner1_size;
  index_t inner1_idx = rem % inner1_size;
  return outer_idx * batch_stride + inner0_idx * inner0_stride + inner1_idx * inner1_stride;
}

template <typename scalar_t>
class NormNoRstdForward {
 public:
  using accscalar_t = acc_type<scalar_t>;
  NormNoRstdForward() = delete;
  NormNoRstdForward(scalar_t* X_data, scalar_t* Y_data, accscalar_t eps) : X_data(X_data), Y_data(Y_data), eps(eps) {}

  int get_update_vec_size(int Plane, int vec_size) const {
    return get_aligned_update_vec_size(Plane, vec_size, X_data, Y_data);
  }

 protected:
  template <typename... Args>
  int get_aligned_update_vec_size(int Plane, int vec_size, Args*... args) const {
    vec_size = get_min_vec_size(vec_size, args...);
    while (Plane % vec_size != 0) {
      vec_size = vec_size >> 1;
    }
    return vec_size;
  }

 public:
  scalar_t* X_data;
  scalar_t* Y_data;
  accscalar_t eps;
};

template <typename scalar_t, typename weight_t>
class RMSNormNoRstdForward : public NormNoRstdForward<scalar_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef NormNoRstdForward<scalar_t> NF;
  RMSNormNoRstdForward() = delete;
  RMSNormNoRstdForward(scalar_t* X_data, scalar_t* Y_data, weight_t* gamma_data, accscalar_t eps)
      : NormNoRstdForward<scalar_t>(X_data, Y_data, eps), gamma_data(gamma_data) {}

  int get_update_vec_size(int Plane, int vec_size) const {
    return NF::get_aligned_update_vec_size(Plane, vec_size, NF::X_data, NF::Y_data, gamma_data);
  }

  // Per-element weight application, the single place the normalized value meets
  // gamma.  Shared by the cached and non-cached update() loops and by the
  // AddNoRstdForward mixin; GemmaRMSNormNoRstdForward hides it with the
  // `1 + gamma` variant.
  static inline scalar_t apply_weight(accscalar_t x, accscalar_t rstd, accscalar_t gamma) {
    return static_cast<scalar_t>(x * rstd * gamma);
  }

  template <int vec_size, int ITERS, bool cache_inputs, typename vec_t, typename index_t, typename Cfg>
  void reduce_combine(
      sycl::nd_item<1> item_id,
      const Cfg& cfg,
      index_t x_group_offset,
      accscalar_t& sum_value,
      vec_t (&reg)[cache_inputs ? ITERS : 1]) const {
    const index_t lid = item_id.get_local_id(0);

    if constexpr (cache_inputs) {
#pragma unroll
      for (int it = 0; it < ITERS; ++it) {
        const index_t plane_offset = (static_cast<index_t>(it) * cfg.workgroup_size + lid) * vec_size;
        if (plane_offset < cfg.Plane) {
          vec_t x_val = *(reinterpret_cast<vec_t*>(NF::X_data + x_group_offset + plane_offset));
          reg[it] = x_val;
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            const accscalar_t x = static_cast<accscalar_t>(x_val[v]);
            sum_value += x * x;
          }
        }
      }
    } else {
      for (index_t plane_offset = lid * vec_size; plane_offset < cfg.Plane;
           plane_offset += cfg.workgroup_size * vec_size) {
        vec_t x_val = *(reinterpret_cast<vec_t*>(NF::X_data + x_group_offset + plane_offset));
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          const accscalar_t x = static_cast<accscalar_t>(x_val[v]);
          sum_value += x * x;
        }
      }
    }
  }

  template <typename Cfg>
  accscalar_t reduce_project(sycl::nd_item<1> item_id, const Cfg& cfg, accscalar_t sum_value) const {
    sum_value = sycl::reduce_over_group(item_id.get_group(), sum_value, sycl::plus<accscalar_t>());
    sum_value = sum_value < static_cast<accscalar_t>(0) ? static_cast<accscalar_t>(0) : sum_value;
    return Numerics<accscalar_t>::rsqrt(
        sum_value / static_cast<accscalar_t>(cfg.Plane) + static_cast<accscalar_t>(NF::eps));
  }

  template <
      int vec_size,
      int ITERS,
      bool cache_inputs,
      typename vec_t,
      typename weight_vec_t,
      typename index_t,
      typename Cfg>
  void update(
      sycl::nd_item<1> item_id,
      const Cfg& cfg,
      index_t x_group_offset,
      index_t y_group_offset,
      accscalar_t rstd,
      const vec_t (&reg)[cache_inputs ? ITERS : 1]) const {
    const index_t lid = item_id.get_local_id(0);

    if constexpr (cache_inputs) {
#pragma unroll
      for (int it = 0; it < ITERS; ++it) {
        const index_t plane_offset = (static_cast<index_t>(it) * cfg.workgroup_size + lid) * vec_size;
        if (plane_offset < cfg.Plane) {
          vec_t x_val = reg[it];
          weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(gamma_data + plane_offset));
          vec_t y_val;
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            y_val[v] = apply_weight(static_cast<accscalar_t>(x_val[v]), rstd, static_cast<accscalar_t>(gamma_val[v]));
          }
          *(reinterpret_cast<vec_t*>(NF::Y_data + y_group_offset + plane_offset)) = y_val;
        }
      }
    } else {
      for (index_t plane_offset = lid * vec_size; plane_offset < cfg.Plane;
           plane_offset += cfg.workgroup_size * vec_size) {
        vec_t x_val = *(reinterpret_cast<vec_t*>(NF::X_data + x_group_offset + plane_offset));
        weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(gamma_data + plane_offset));
        vec_t y_val;
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          y_val[v] = apply_weight(static_cast<accscalar_t>(x_val[v]), rstd, static_cast<accscalar_t>(gamma_val[v]));
        }
        *(reinterpret_cast<vec_t*>(NF::Y_data + y_group_offset + plane_offset)) = y_val;
      }
    }
  }

 public:
  weight_t* gamma_data;
};

// Residual-add mixin for the register-cached no-rstd norms.  `BaseNorm` supplies
// reduce_project() and the per-element weight application (apply_weight); this
// only adds `x += add` in front of the reduction, so the llama-style and the
// Gemma-style fused-add norms share one implementation.
//
// reduce_combine() never stores the sum back to X_data: it goes to add_data,
// plus reg[] when the inputs are cached.  The only write to the x buffer is
// update()'s store through Y_data (which for the in-place export *is* x, after
// the reduction).  So the non-cached update() must re-read the sum from
// add_data and NOT from X_data, which still holds the original, un-summed x.
template <typename BaseNorm, typename scalar_t, typename weight_t>
class AddNoRstdForward : public BaseNorm {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef BaseNorm Base;
  typedef NormNoRstdForward<scalar_t> NF;
  AddNoRstdForward() = delete;
  AddNoRstdForward(scalar_t* X_data, scalar_t* Y_data, weight_t* gamma_data, accscalar_t eps, scalar_t* add_data)
      : BaseNorm(X_data, Y_data, gamma_data, eps), add_data(add_data) {}

  int get_update_vec_size(int Plane, int vec_size) const {
    return NF::get_aligned_update_vec_size(Plane, vec_size, NF::X_data, NF::Y_data, Base::gamma_data, add_data);
  }

  template <int vec_size, int ITERS, bool cache_inputs, typename vec_t, typename index_t, typename Cfg>
  void reduce_combine(
      sycl::nd_item<1> item_id,
      const Cfg& cfg,
      index_t x_group_offset,
      accscalar_t& sum_value,
      vec_t (&reg)[cache_inputs ? ITERS : 1]) const {
    const index_t lid = item_id.get_local_id(0);

    if constexpr (cache_inputs) {
#pragma unroll
      for (int it = 0; it < ITERS; ++it) {
        const index_t plane_offset = (static_cast<index_t>(it) * cfg.workgroup_size + lid) * vec_size;
        if (plane_offset < cfg.Plane) {
          vec_t x_val = *(reinterpret_cast<vec_t*>(NF::X_data + x_group_offset + plane_offset));
          vec_t add_val = *(reinterpret_cast<vec_t*>(add_data + x_group_offset + plane_offset));
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            x_val[v] = static_cast<scalar_t>(static_cast<accscalar_t>(x_val[v]) + static_cast<accscalar_t>(add_val[v]));
          }
          *(reinterpret_cast<vec_t*>(add_data + x_group_offset + plane_offset)) = x_val;
          reg[it] = x_val;
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            const accscalar_t x = static_cast<accscalar_t>(x_val[v]);
            sum_value += x * x;
          }
        }
      }
    } else {
      for (index_t plane_offset = lid * vec_size; plane_offset < cfg.Plane;
           plane_offset += cfg.workgroup_size * vec_size) {
        vec_t x_val = *(reinterpret_cast<vec_t*>(NF::X_data + x_group_offset + plane_offset));
        vec_t add_val = *(reinterpret_cast<vec_t*>(add_data + x_group_offset + plane_offset));
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          x_val[v] = static_cast<scalar_t>(static_cast<accscalar_t>(x_val[v]) + static_cast<accscalar_t>(add_val[v]));
        }
        *(reinterpret_cast<vec_t*>(add_data + x_group_offset + plane_offset)) = x_val;
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          const accscalar_t x = static_cast<accscalar_t>(x_val[v]);
          sum_value += x * x;
        }
      }
    }
  }

  template <
      int vec_size,
      int ITERS,
      bool cache_inputs,
      typename vec_t,
      typename weight_vec_t,
      typename index_t,
      typename Cfg>
  void update(
      sycl::nd_item<1> item_id,
      const Cfg& cfg,
      index_t x_group_offset,
      index_t y_group_offset,
      accscalar_t rstd,
      const vec_t (&reg)[cache_inputs ? ITERS : 1]) const {
    if constexpr (cache_inputs) {
      Base::template update<vec_size, ITERS, cache_inputs, vec_t, weight_vec_t, index_t>(
          item_id, cfg, x_group_offset, y_group_offset, rstd, reg);
    } else {
      const index_t lid = item_id.get_local_id(0);
      for (index_t plane_offset = lid * vec_size; plane_offset < cfg.Plane;
           plane_offset += cfg.workgroup_size * vec_size) {
        // add_data, not X_data: the residual sum was stored there by
        // reduce_combine, while X_data still holds the un-summed x.
        vec_t x_val = *(reinterpret_cast<vec_t*>(add_data + x_group_offset + plane_offset));
        weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(Base::gamma_data + plane_offset));
        vec_t y_val;
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          y_val[v] =
              Base::apply_weight(static_cast<accscalar_t>(x_val[v]), rstd, static_cast<accscalar_t>(gamma_val[v]));
        }
        *(reinterpret_cast<vec_t*>(NF::Y_data + y_group_offset + plane_offset)) = y_val;
      }
    }
  }

 public:
  scalar_t* add_data;
};

template <typename scalar_t, typename weight_t>
using AddRMSNormNoRstdForward = AddNoRstdForward<RMSNormNoRstdForward<scalar_t, weight_t>, scalar_t, weight_t>;

template <typename scalar_t, typename weight_t>
class GemmaRMSNormNoRstdForward : public RMSNormNoRstdForward<scalar_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef RMSNormNoRstdForward<scalar_t, weight_t> RNF;
  GemmaRMSNormNoRstdForward() = delete;
  using RNF::RNF;

  // Hides RMSNormNoRstdForward::apply_weight: Gemma stores gamma - 1.
  static inline scalar_t apply_weight(accscalar_t x, accscalar_t rstd, accscalar_t gamma) {
    return static_cast<scalar_t>(x * rstd * (static_cast<accscalar_t>(1.0) + gamma));
  }

  template <
      int vec_size,
      int ITERS,
      bool cache_inputs,
      typename vec_t,
      typename weight_vec_t,
      typename index_t,
      typename Cfg>
  void update(
      sycl::nd_item<1> item_id,
      const Cfg& cfg,
      index_t x_group_offset,
      index_t y_group_offset,
      accscalar_t rstd,
      const vec_t (&reg)[cache_inputs ? ITERS : 1]) const {
    const index_t lid = item_id.get_local_id(0);

    if constexpr (cache_inputs) {
#pragma unroll
      for (int it = 0; it < ITERS; ++it) {
        const index_t plane_offset = (static_cast<index_t>(it) * cfg.workgroup_size + lid) * vec_size;
        if (plane_offset < cfg.Plane) {
          vec_t x_val = reg[it];
          weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(RNF::gamma_data + plane_offset));
          vec_t y_val;
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            y_val[v] = apply_weight(static_cast<accscalar_t>(x_val[v]), rstd, static_cast<accscalar_t>(gamma_val[v]));
          }
          *(reinterpret_cast<vec_t*>(RNF::Y_data + y_group_offset + plane_offset)) = y_val;
        }
      }
    } else {
      for (index_t plane_offset = lid * vec_size; plane_offset < cfg.Plane;
           plane_offset += cfg.workgroup_size * vec_size) {
        vec_t x_val = *(reinterpret_cast<vec_t*>(RNF::X_data + x_group_offset + plane_offset));
        weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(RNF::gamma_data + plane_offset));
        vec_t y_val;
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          y_val[v] = apply_weight(static_cast<accscalar_t>(x_val[v]), rstd, static_cast<accscalar_t>(gamma_val[v]));
        }
        *(reinterpret_cast<vec_t*>(RNF::Y_data + y_group_offset + plane_offset)) = y_val;
      }
    }
  }
};

template <typename scalar_t, typename weight_t>
using GemmaAddRMSNormNoRstdForward =
    AddNoRstdForward<GemmaRMSNormNoRstdForward<scalar_t, weight_t>, scalar_t, weight_t>;

template <
    typename scalar_t,
    typename weight_t,
    int vec_size,
    int ITERS,
    typename Norm,
    bool cache_inputs,
    typename index_t = uint32_t,
    typename Cfg = NormConfig>
struct RMSNormNoRstdKernelFunctor {
  using accscalar_t = acc_type<scalar_t>;
  using vec_t = aligned_vector_loop<scalar_t, vec_size>;
  using weight_vec_t = aligned_vector_loop<weight_t, vec_size>;

  [[sycl::reqd_sub_group_size(NUM_REDUCE_STAGES)]] void operator()(sycl::nd_item<1> item_id) const {
    const index_t row = item_id.get_group(0);
    const index_t x_group_offset = compute_row_offset<index_t>(
        row,
        cfg.input_batch_stride,
        cfg.input_inner0_size,
        cfg.input_inner0_stride,
        cfg.input_inner1_size,
        cfg.input_inner1_stride);
    const index_t y_group_offset = compute_row_offset<index_t>(
        row,
        cfg.output_batch_stride,
        cfg.output_inner0_size,
        cfg.output_inner0_stride,
        cfg.output_inner1_size,
        cfg.output_inner1_stride);

    accscalar_t sum_value = 0;
    vec_t reg[cache_inputs ? ITERS : 1];
    norm.template reduce_combine<vec_size, ITERS, cache_inputs, vec_t, index_t>(
        item_id, cfg, x_group_offset, sum_value, reg);
    const accscalar_t rstd = norm.reduce_project(item_id, cfg, sum_value);
    norm.template update<vec_size, ITERS, cache_inputs, vec_t, weight_vec_t, index_t>(
        item_id, cfg, x_group_offset, y_group_offset, rstd, reg);
  }

  RMSNormNoRstdKernelFunctor(Norm norm_, Cfg cfg_) : norm(norm_), cfg(cfg_) {}

 private:
  Norm norm;
  const Cfg cfg;
};

template <typename scalar_t, typename weight_t, int vec_size, int ITERS, typename Norm, bool cache_inputs>
void rmsnorm_no_rstd_kernel(Norm& norm, const NormConfig& config) {
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using KernelFunctor = RMSNormNoRstdKernelFunctor<scalar_t, weight_t, vec_size, ITERS, Norm, cache_inputs>;

  KernelFunctor kfn(norm, config);
  sycl::range<1> local_range{static_cast<size_t>(config.workgroup_size)};
  sycl::range<1> global_range{static_cast<size_t>(config.workgroup_num * config.workgroup_size)};
  sycl_kernel_submit(global_range, local_range, queue, kfn);
}

template <typename scalar_t, typename weight_t, int vec_size, typename Norm>
void dispatch_rmsnorm_no_rstd_iters(int iters, Norm& norm, const NormConfig& config) {
#define RMSNORM_NO_RSTD_CACHED_CASE(ITERS) \
  rmsnorm_no_rstd_kernel<scalar_t, weight_t, vec_size, ITERS, Norm, true>(norm, config)

  if (iters <= 1) {
    RMSNORM_NO_RSTD_CACHED_CASE(1);
  } else if (iters == 2) {
    RMSNORM_NO_RSTD_CACHED_CASE(2);
  } else if (iters == 3) {
    RMSNORM_NO_RSTD_CACHED_CASE(3);
  } else if (iters == 4) {
    RMSNORM_NO_RSTD_CACHED_CASE(4);
  } else if (iters == 5) {
    RMSNORM_NO_RSTD_CACHED_CASE(5);
  } else if (iters == 6) {
    RMSNORM_NO_RSTD_CACHED_CASE(6);
  } else if (iters == 7) {
    RMSNORM_NO_RSTD_CACHED_CASE(7);
  } else if (iters == 8) {
    RMSNORM_NO_RSTD_CACHED_CASE(8);
  } else {
    rmsnorm_no_rstd_kernel<scalar_t, weight_t, vec_size, 1, Norm, false>(norm, config);
  }
#undef RMSNORM_NO_RSTD_CACHED_CASE
}

template <typename scalar_t, typename weight_t, typename Norm>
void launch_vectorized_rmsnorm_no_rstd_kernel(Norm& norm, const NormConfig& config) {
  const int vec_size = config.update_vec_size;
  const int iters = (config.WGPlane + config.workgroup_size * vec_size - 1) / (config.workgroup_size * vec_size);

#define DISPATCH_RMSNORM_NO_RSTD_VEC(VEC_SIZE) \
  dispatch_rmsnorm_no_rstd_iters<scalar_t, weight_t, VEC_SIZE>(iters, norm, config)

  switch (vec_size) {
    case 8: {
      DISPATCH_RMSNORM_NO_RSTD_VEC(8);
      break;
    }
    case 4: {
      DISPATCH_RMSNORM_NO_RSTD_VEC(4);
      break;
    }
    case 2: {
      DISPATCH_RMSNORM_NO_RSTD_VEC(2);
      break;
    }
    default: {
      DISPATCH_RMSNORM_NO_RSTD_VEC(1);
      break;
    }
  }
#undef DISPATCH_RMSNORM_NO_RSTD_VEC
}

// Launch for contiguous rows built with NormConfig(FastRowTag): NormFastParams makes every inner
// size/stride a compile-time constant, so compute_row_offset folds its integer div/mod (Xe has no
// hardware divide) to row * batch_stride. 1, 2, 4 or 8 passes per row are instantiated; any other
// count takes the general launcher.
template <typename scalar_t, typename weight_t, int vec_size, typename Norm>
void launch_fast_rmsnorm_no_rstd_kernel(Norm& norm, const NormConfig& config) {
  const int iters = (config.WGPlane + config.workgroup_size * vec_size - 1) / (config.workgroup_size * vec_size);
  auto launch = [&](auto iters_tag) {
    using KernelFunctor = RMSNormNoRstdKernelFunctor<
        scalar_t,
        weight_t,
        vec_size,
        decltype(iters_tag)::value,
        Norm,
        /*cache_inputs=*/true,
        uint32_t,
        NormFastParams>;
    const NormFastParams fast_config{
        config.Plane, config.workgroup_size, config.input_batch_stride, config.output_batch_stride};
    const sycl::range<1> local_range{static_cast<size_t>(config.workgroup_size)};
    const sycl::range<1> global_range{
        static_cast<size_t>(config.workgroup_num) * static_cast<size_t>(config.workgroup_size)};
    sycl_kernel_submit(
        global_range, local_range, at::xpu::getCurrentXPUStream().queue(), KernelFunctor(norm, fast_config));
  };
  if (iters == 1) {
    launch(std::integral_constant<int, 1>{});
  } else if (iters == 2) {
    launch(std::integral_constant<int, 2>{});
  } else if (iters == 4) {
    launch(std::integral_constant<int, 4>{});
  } else if (iters == 8) {
    launch(std::integral_constant<int, 8>{});
  } else {
    launch_vectorized_rmsnorm_no_rstd_kernel<scalar_t, weight_t>(norm, config);
  }
}

template <typename scalar_t, typename weight_t>
void RMSNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gemma,
    int64_t M,
    int64_t N,
    acc_type<scalar_t> eps,
    Tensor& Y,
    int64_t input_batch_stride,
    int64_t output_batch_stride,
    int64_t input_inner0_size,
    int64_t input_inner0_stride,
    int64_t output_inner0_size,
    int64_t output_inner0_stride,
    int64_t input_inner1_size,
    int64_t input_inner1_stride,
    int64_t output_inner1_size,
    int64_t output_inner1_stride) {
  scalar_t* X_data = X.data_ptr<scalar_t>();
  scalar_t* Y_data = Y.data_ptr<scalar_t>();
  weight_t* gemma_data = gemma.data_ptr<weight_t>();
  RMSNormNoRstdForward<scalar_t, weight_t> rms_norm_no_rstd_forward(X_data, Y_data, gemma_data, eps);

  auto config = NormConfig(
      M,
      N,
      1,
      sizeof(scalar_t),
      input_batch_stride,
      output_batch_stride,
      [&](int plane, int max_vec_size) { return rms_norm_no_rstd_forward.get_update_vec_size(plane, max_vec_size); },
      input_inner0_size,
      input_inner0_stride,
      output_inner0_size,
      output_inner0_stride,
      input_inner1_size,
      input_inner1_stride,
      output_inner1_size,
      output_inner1_stride);
  launch_vectorized_rmsnorm_no_rstd_kernel<scalar_t, weight_t>(rms_norm_no_rstd_forward, config);
}

template <typename scalar_t, typename weight_t>
void FusedAddRMSNormKernelImplInternal(
    const Tensor& X, const Tensor& gemma, int64_t M, int64_t N, acc_type<scalar_t> eps, Tensor& residual) {
  scalar_t* X_data = X.data_ptr<scalar_t>();
  weight_t* gemma_data = gemma.data_ptr<weight_t>();
  scalar_t* residual_data = residual.data_ptr<scalar_t>();
  AddRMSNormNoRstdForward<scalar_t, weight_t> add_rms_norm_no_rstd_forward(
      X_data, X_data, gemma_data, eps, residual_data);

  // A 2-byte dtype with a same-dtype weight whose rows split into full 16-byte vectors takes the
  // fast launch: FastRowTag caches the workgroup-sizing device properties (this op is host-bound
  // at decode sizes) and NormFastParams drops the per-row div/mod.
  if constexpr (std::is_same_v<scalar_t, weight_t> && sizeof(scalar_t) == 2) {
    // Alignment is tested directly: get_update_vec_size() looks up device properties once per
    // pointer, which cost 0.44 us per call here (9% of a decode-size call on Arc Pro B60).
    constexpr std::uintptr_t kAlignBytes = 16;
    constexpr int kVec = kAlignBytes / sizeof(scalar_t);
    const auto aligned = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % kAlignBytes == 0; };
    if (N % kVec == 0 && aligned(X_data) && aligned(gemma_data) && aligned(residual_data)) {
      const NormConfig config(
          NormConfig::FastRowTag{},
          static_cast<int>(M),
          static_cast<int>(N),
          static_cast<int>(sizeof(scalar_t)),
          kVec,
          static_cast<int>(N));
      launch_fast_rmsnorm_no_rstd_kernel<scalar_t, weight_t, kVec>(add_rms_norm_no_rstd_forward, config);
      return;
    }
  }

  auto config = NormConfig(
      M,
      N,
      1,
      sizeof(scalar_t),
      N,
      N,
      [&](int plane, int max_vec_size) {
        return add_rms_norm_no_rstd_forward.get_update_vec_size(plane, max_vec_size);
      },
      1,
      0,
      1,
      0);

  launch_vectorized_rmsnorm_no_rstd_kernel<scalar_t, weight_t>(add_rms_norm_no_rstd_forward, config);
}

template <typename scalar_t, typename weight_t>
void GemmaRMSNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gemma,
    int64_t M,
    int64_t N,
    acc_type<scalar_t> eps,
    Tensor& Y,
    int64_t input_batch_stride,
    int64_t output_batch_stride,
    int64_t input_inner0_size,
    int64_t input_inner0_stride,
    int64_t output_inner0_size,
    int64_t output_inner0_stride,
    int64_t input_inner1_size,
    int64_t input_inner1_stride,
    int64_t output_inner1_size,
    int64_t output_inner1_stride) {
  scalar_t* X_data = X.data_ptr<scalar_t>();
  scalar_t* Y_data = Y.data_ptr<scalar_t>();
  weight_t* gemma_data = gemma.data_ptr<weight_t>();
  GemmaRMSNormNoRstdForward<scalar_t, weight_t> gemma_rms_norm_no_rstd_forward(X_data, Y_data, gemma_data, eps);

  auto config = NormConfig(
      M,
      N,
      1,
      sizeof(scalar_t),
      input_batch_stride,
      output_batch_stride,
      [&](int plane, int max_vec_size) {
        return gemma_rms_norm_no_rstd_forward.get_update_vec_size(plane, max_vec_size);
      },
      input_inner0_size,
      input_inner0_stride,
      output_inner0_size,
      output_inner0_stride,
      input_inner1_size,
      input_inner1_stride,
      output_inner1_size,
      output_inner1_stride);
  launch_vectorized_rmsnorm_no_rstd_kernel<scalar_t, weight_t>(gemma_rms_norm_no_rstd_forward, config);
}

template <typename scalar_t, typename weight_t>
void GemmaFusedAddRMSNormKernelImplInternal(
    Tensor& X, const Tensor& gemma, int64_t M, int64_t N, acc_type<scalar_t> eps, Tensor& residual) {
  scalar_t* X_data = X.data_ptr<scalar_t>();
  weight_t* gemma_data = gemma.data_ptr<weight_t>();
  scalar_t* residual_data = residual.data_ptr<scalar_t>();
  GemmaAddRMSNormNoRstdForward<scalar_t, weight_t> gemma_add_rms_norm_no_rstd_forward(
      X_data, X_data, gemma_data, eps, residual_data);

  auto config = NormConfig(
      M,
      N,
      1,
      sizeof(scalar_t),
      N,
      N,
      [&](int plane, int max_vec_size) {
        return gemma_add_rms_norm_no_rstd_forward.get_update_vec_size(plane, max_vec_size);
      },
      1,
      0,
      1,
      0);

  launch_vectorized_rmsnorm_no_rstd_kernel<scalar_t, weight_t>(gemma_add_rms_norm_no_rstd_forward, config);
}

SGL_KERNEL_EXPORT void rmsnorm(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps) {
  std::optional<torch::Tensor> opt_weight = weight;
  std::optional<torch::Tensor> opt_bias;
  auto [M, N] = _check_layer_norm_inputs(input, c10::IntArrayRef({input.size(-1)}), opt_weight, opt_bias);

  // Derive row-stride info directly from input/output so the kernel can
  // handle non-flattenable 3D tensors (e.g. QKV slices) natively.
  RowStrides in_strides = get_row_strides(input);
  RowStrides out_strides = get_row_strides(output);
  Tensor weight_ = (weight.dim() == 1) ? weight.reshape({N}) : weight;

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  // Follows the PR#269 flash_attentionXe35_common.hpp GPU_Clock pattern. The
  // sycl_kernel_submit helper returns void so we cannot forward the event to
  // EventManager here -- host-side timing + queue.wait() is the meaningful
  // subset for this launcher.
  auto profiling_queue = at::xpu::getCurrentXPUStream().queue();
  GPU_Clock timer;
  timer.start();
#endif

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "RMSNormKernelImpl", [&]() {
        SYCL_DISPATCH_WEIGHT_TYPES(
            at::ScalarType::Half, at::ScalarType::BFloat16, weight_.scalar_type(), "RMSNormKernelImpl", [&]() {
              RMSNormKernelImplInternal<scalar_t, weight_t>(
                  input,
                  weight_,
                  M,
                  N,
                  static_cast<acc_type<scalar_t>>(eps),
                  output,
                  in_strides.batch_stride,
                  out_strides.batch_stride,
                  in_strides.inner0_size,
                  in_strides.inner0_stride,
                  out_strides.inner0_size,
                  out_strides.inner0_stride,
                  in_strides.inner1_size,
                  in_strides.inner1_stride,
                  out_strides.inner1_size,
                  out_strides.inner1_stride);
            });
      });

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  profiling_queue.wait();
  const double elapsed_s = timer.seconds();

  // RMSNorm per-row work: N muls + N adds for sum-of-squares, one rsqrt, then
  // N muls (scale) + N muls (weight). ~= 4 * M * N flops end-to-end.
  const double flops = 4.0 * static_cast<double>(M) * static_cast<double>(N);
  const double tflops = elapsed_s > 0.0 ? (flops * 1e-12) / elapsed_s : 0.0;

  const double in_elem = static_cast<double>(input.element_size());
  const double w_elem = static_cast<double>(weight_.element_size());
  const double out_elem = static_cast<double>(output.element_size());
  const double bytes = static_cast<double>(M) * static_cast<double>(N) * in_elem + static_cast<double>(N) * w_elem +
                       static_cast<double>(M) * static_cast<double>(N) * out_elem;
  const double gbps = elapsed_s > 0.0 ? (bytes * 1e-9) / elapsed_s : 0.0;

  const double elapsed_ms = elapsed_s * 1000.0;
  const double elapsed_us = elapsed_s * 1e6;
  const double gflops = tflops * 1e3;

  ::printf(
      "rmsnorm perf(gpu_clock): time=%.9f ms (%.3f us), bandwidth=%.6f GB/s, "
      "compute=%.6f TFLOPS (%.3f GFLOPS)\n",
      elapsed_ms,
      elapsed_us,
      gbps,
      tflops,
      gflops);

  if (elapsed_s <= 0.0) {
    ::printf(
        "rmsnorm perf(gpu_clock): unavailable (reported 0). Check SYCL profiling event path and "
        "CUTLASS_SYCL_PROFILING_ENABLED build flag.\n");
  }
#endif
}

SGL_KERNEL_EXPORT void
fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps) {
  TORCH_CHECK(input.is_contiguous(), "fused_add_rmsnorm: input must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "fused_add_rmsnorm: residual must be contiguous");

  TORCH_CHECK(
      residual.sizes() == input.sizes(),
      "fused_add_rmsnorm: residual shape ",
      residual.sizes(),
      " must match input shape ",
      input.sizes());
  std::optional<torch::Tensor> opt_weight = weight;
  std::optional<torch::Tensor> opt_bias;
  auto [M, N] = _check_layer_norm_inputs(input, c10::IntArrayRef({input.size(-1)}), opt_weight, opt_bias);

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  auto profiling_queue = at::xpu::getCurrentXPUStream().queue();
  GPU_Clock timer;
  timer.start();
#endif

  // input and residual are contiguous and the same shape, so the kernel indexes them as M rows of N.
  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "FusedAddRMSNormKernelImpl", [&]() {
        SYCL_DISPATCH_WEIGHT_TYPES(
            at::ScalarType::Half, at::ScalarType::BFloat16, weight.scalar_type(), "FusedAddRMSNormKernelImpl", [&]() {
              FusedAddRMSNormKernelImplInternal<scalar_t, weight_t>(
                  input, weight, M, N, static_cast<acc_type<scalar_t>>(eps), residual);
            });
      });

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  profiling_queue.wait();
  const double elapsed_s = timer.seconds();

  // Residual add (1 flop/elem) + RMSNorm (4 flop/elem) = 5 * M * N.
  const double flops = 5.0 * static_cast<double>(M) * static_cast<double>(N);
  const double tflops = elapsed_s > 0.0 ? (flops * 1e-12) / elapsed_s : 0.0;

  const double in_elem = static_cast<double>(input.element_size());
  const double res_elem = static_cast<double>(residual.element_size());
  const double w_elem = static_cast<double>(weight.element_size());
  // read input, read+write residual, write normalized output back to input, read weight.
  const double bytes = 2.0 * static_cast<double>(M) * static_cast<double>(N) * in_elem +
                       2.0 * static_cast<double>(M) * static_cast<double>(N) * res_elem +
                       static_cast<double>(N) * w_elem;
  const double gbps = elapsed_s > 0.0 ? (bytes * 1e-9) / elapsed_s : 0.0;

  const double elapsed_ms = elapsed_s * 1000.0;
  const double elapsed_us = elapsed_s * 1e6;
  const double gflops = tflops * 1e3;

  ::printf(
      "fused_add_rmsnorm perf(gpu_clock): time=%.9f ms (%.3f us), bandwidth=%.6f GB/s, "
      "compute=%.6f TFLOPS (%.3f GFLOPS)\n",
      elapsed_ms,
      elapsed_us,
      gbps,
      tflops,
      gflops);

  if (elapsed_s <= 0.0) {
    ::printf(
        "fused_add_rmsnorm perf(gpu_clock): unavailable (reported 0). Check SYCL profiling event path and "
        "CUTLASS_SYCL_PROFILING_ENABLED build flag.\n");
  }
#endif
}

SGL_KERNEL_EXPORT void gemma_rmsnorm(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps) {
  std::optional<torch::Tensor> opt_weight = weight;
  std::optional<torch::Tensor> opt_bias;
  auto [M, N] = _check_layer_norm_inputs(input, c10::IntArrayRef({input.size(-1)}), opt_weight, opt_bias);

  RowStrides in_strides = get_row_strides(input);
  RowStrides out_strides = get_row_strides(output);
  Tensor weight_ = (weight.dim() == 1) ? weight.reshape({N}) : weight;

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  auto profiling_queue = at::xpu::getCurrentXPUStream().queue();
  GPU_Clock timer;
  timer.start();
#endif

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "GemmaRMSNormKernelImpl", [&]() {
        SYCL_DISPATCH_WEIGHT_TYPES(
            at::ScalarType::Half, at::ScalarType::BFloat16, weight_.scalar_type(), "GemmaRMSNormKernelImpl", [&]() {
              GemmaRMSNormKernelImplInternal<scalar_t, weight_t>(
                  input,
                  weight_,
                  M,
                  N,
                  static_cast<acc_type<scalar_t>>(eps),
                  output,
                  in_strides.batch_stride,
                  out_strides.batch_stride,
                  in_strides.inner0_size,
                  in_strides.inner0_stride,
                  out_strides.inner0_size,
                  out_strides.inner0_stride,
                  in_strides.inner1_size,
                  in_strides.inner1_stride,
                  out_strides.inner1_size,
                  out_strides.inner1_stride);
            });
      });

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  profiling_queue.wait();
  const double elapsed_s = timer.seconds();

  // Gemma RMSNorm: sum-of-squares (2N) + rsqrt/scale + (1+weight) mul (~3N) = ~5 * M * N.
  const double flops = 5.0 * static_cast<double>(M) * static_cast<double>(N);
  const double tflops = elapsed_s > 0.0 ? (flops * 1e-12) / elapsed_s : 0.0;

  const double in_elem = static_cast<double>(input.element_size());
  const double w_elem = static_cast<double>(weight_.element_size());
  const double out_elem = static_cast<double>(output.element_size());
  const double bytes = static_cast<double>(M) * static_cast<double>(N) * in_elem + static_cast<double>(N) * w_elem +
                       static_cast<double>(M) * static_cast<double>(N) * out_elem;
  const double gbps = elapsed_s > 0.0 ? (bytes * 1e-9) / elapsed_s : 0.0;

  const double elapsed_ms = elapsed_s * 1000.0;
  const double elapsed_us = elapsed_s * 1e6;
  const double gflops = tflops * 1e3;

  ::printf(
      "gemma_rmsnorm perf(gpu_clock): time=%.9f ms (%.3f us), bandwidth=%.6f GB/s, "
      "compute=%.6f TFLOPS (%.3f GFLOPS)\n",
      elapsed_ms,
      elapsed_us,
      gbps,
      tflops,
      gflops);

  if (elapsed_s <= 0.0) {
    ::printf(
        "gemma_rmsnorm perf(gpu_clock): unavailable (reported 0). Check SYCL profiling event path and "
        "CUTLASS_SYCL_PROFILING_ENABLED build flag.\n");
  }
#endif
}

SGL_KERNEL_EXPORT void
gemma_fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, double eps) {
  TORCH_CHECK(input.is_contiguous(), "gemma_fused_add_rmsnorm: input must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "gemma_fused_add_rmsnorm: residual must be contiguous");
  std::optional<torch::Tensor> opt_weight = weight;
  std::optional<torch::Tensor> opt_bias;
  auto [M, N] = _check_layer_norm_inputs(input, c10::IntArrayRef({input.size(-1)}), opt_weight, opt_bias);

  // Flatten leading dimensions to 2D for the kernel
  Tensor input_ = flatten_to_2d(input, M, N);
  Tensor residual_ = flatten_to_2d(residual, M, N);
  Tensor weight_ = (weight.dim() == 1) ? weight.reshape({N}) : weight;

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  auto profiling_queue = at::xpu::getCurrentXPUStream().queue();
  GPU_Clock timer;
  timer.start();
#endif

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, input_.scalar_type(), "GemmaFusedAddRMSNormKernelImpl", [&]() {
        SYCL_DISPATCH_WEIGHT_TYPES(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            weight_.scalar_type(),
            "GemmaFusedAddRMSNormKernelImpl",
            [&]() {
              GemmaFusedAddRMSNormKernelImplInternal<scalar_t, weight_t>(
                  input_, weight_, M, N, static_cast<acc_type<scalar_t>>(eps), residual_);
            });
      });

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  profiling_queue.wait();
  const double elapsed_s = timer.seconds();

  // Residual add (1N) + Gemma RMSNorm (~5N with (1+weight) mul) = ~6 * M * N.
  const double flops = 6.0 * static_cast<double>(M) * static_cast<double>(N);
  const double tflops = elapsed_s > 0.0 ? (flops * 1e-12) / elapsed_s : 0.0;

  const double in_elem = static_cast<double>(input_.element_size());
  const double res_elem = static_cast<double>(residual_.element_size());
  const double w_elem = static_cast<double>(weight_.element_size());
  const double bytes = 2.0 * static_cast<double>(M) * static_cast<double>(N) * in_elem +
                       2.0 * static_cast<double>(M) * static_cast<double>(N) * res_elem +
                       static_cast<double>(N) * w_elem;
  const double gbps = elapsed_s > 0.0 ? (bytes * 1e-9) / elapsed_s : 0.0;

  const double elapsed_ms = elapsed_s * 1000.0;
  const double elapsed_us = elapsed_s * 1e6;
  const double gflops = tflops * 1e3;

  ::printf(
      "gemma_fused_add_rmsnorm perf(gpu_clock): time=%.9f ms (%.3f us), bandwidth=%.6f GB/s, "
      "compute=%.6f TFLOPS (%.3f GFLOPS)\n",
      elapsed_ms,
      elapsed_us,
      gbps,
      tflops,
      gflops);

  if (elapsed_s <= 0.0) {
    ::printf(
        "gemma_fused_add_rmsnorm perf(gpu_clock): unavailable (reported 0). Check SYCL profiling event path and "
        "CUTLASS_SYCL_PROFILING_ENABLED build flag.\n");
  }
#endif
}

// One work-group per (row, group) pair of group_size elements; weights are loaded together with the input so their
// latency overlaps the reduction.
template <typename scalar_t, int vec_size, int ITERS>
struct GroupedGemmaRMSNormFunctor {
  using accscalar_t = acc_type<scalar_t>;
  using vec_t = aligned_vector_loop<scalar_t, vec_size>;

  [[sycl::reqd_sub_group_size(NUM_REDUCE_STAGES)]] void operator()(sycl::nd_item<1> item) const {
    const uint32_t wg = item.get_group(0);
    const uint32_t lid = item.get_local_id(0);
    const uint32_t wg_size = item.get_local_range(0);
    const uint32_t num_vecs = group_size / vec_size;
    const vec_t* x = reinterpret_cast<const vec_t*>(input + static_cast<int64_t>(wg) * group_size);
    vec_t* y = reinterpret_cast<vec_t*>(output + static_cast<int64_t>(wg) * group_size);
    const vec_t* w = reinterpret_cast<const vec_t*>(weight + (wg % num_groups) * group_size);

    vec_t reg[ITERS];
    vec_t wreg[ITERS];
    accscalar_t sum_sq = 0;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const uint32_t idx = it * wg_size + lid;
      if (idx < num_vecs) {
        reg[it] = x[idx];
        wreg[it] = w[idx];
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
          const accscalar_t f = static_cast<accscalar_t>(reg[it][i]);
          sum_sq += f * f;
        }
      }
    }
    sum_sq = sycl::reduce_over_group(item.get_group(), sum_sq, sycl::plus<accscalar_t>());
    const accscalar_t rstd = sycl::rsqrt(sum_sq / static_cast<accscalar_t>(group_size) + eps);
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const uint32_t idx = it * wg_size + lid;
      if (idx < num_vecs) {
        vec_t out;
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
          out[i] = GemmaRMSNormNoRstdForward<scalar_t, scalar_t>::apply_weight(
              static_cast<accscalar_t>(reg[it][i]), rstd, static_cast<accscalar_t>(wreg[it][i]));
        }
        y[idx] = out;
      }
    }
  }

  const scalar_t* input;
  const scalar_t* weight;
  scalar_t* output;
  uint32_t group_size;
  uint32_t num_groups;
  accscalar_t eps;
};

template <typename scalar_t, int vec_size, int ITERS>
void launch_grouped_gemma_rmsnorm(
    const scalar_t* x, const scalar_t* w, scalar_t* y, int64_t rows, int64_t N, int64_t num_groups, double eps) {
  const int64_t vecs = N / vec_size;
  const int64_t wg_size = std::min<int64_t>(256, (((vecs + ITERS - 1) / ITERS) + 15) / 16 * 16);
  GroupedGemmaRMSNormFunctor<scalar_t, vec_size, ITERS> fn{
      x, w, y, static_cast<uint32_t>(N), static_cast<uint32_t>(num_groups), static_cast<acc_type<scalar_t>>(eps)};
  sycl_kernel_submit(
      sycl::range<1>(rows * wg_size), sycl::range<1>(wg_size), at::xpu::getCurrentXPUStream().queue(), fn);
}

template <typename scalar_t>
void GroupedGemmaRMSNormKernelImplInternal(
    const Tensor& X, const Tensor& gemma, int64_t M, int64_t N, int64_t num_groups, double eps, Tensor& Y) {
  constexpr int kVec = 16 / sizeof(scalar_t);
  const int64_t want = (N / kVec + 255) / 256;
  scalar_t* x = X.data_ptr<scalar_t>();
  scalar_t* w = gemma.data_ptr<scalar_t>();
  scalar_t* y = Y.data_ptr<scalar_t>();
  if (want <= 1) {
    launch_grouped_gemma_rmsnorm<scalar_t, kVec, 1>(x, w, y, M, N, num_groups, eps);
  } else if (want <= 2) {
    launch_grouped_gemma_rmsnorm<scalar_t, kVec, 2>(x, w, y, M, N, num_groups, eps);
  } else if (want <= 4) {
    launch_grouped_gemma_rmsnorm<scalar_t, kVec, 4>(x, w, y, M, N, num_groups, eps);
  } else {
    launch_grouped_gemma_rmsnorm<scalar_t, kVec, 8>(x, w, y, M, N, num_groups, eps);
  }
}

// Gemma RMSNorm with the variance computed per group_size slice of the last dim. weight has the full hidden size.
SGL_KERNEL_EXPORT void grouped_gemma_rmsnorm(
    torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps, int64_t group_size) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(output);
  TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dim");
  TORCH_CHECK(input.sizes() == output.sizes(), "output must have the same shape as input");
  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
      "grouped_gemma_rmsnorm supports bfloat16 and float16, got ",
      input.scalar_type());
  TORCH_CHECK(output.scalar_type() == input.scalar_type(), "output dtype must match input");
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "weight dtype must match input");
  constexpr int64_t kVecElems = 8;
  TORCH_CHECK(group_size > 0 && group_size % kVecElems == 0, "group_size must be a positive multiple of ", kVecElems);
  constexpr int64_t kMaxGroupSize = 8 * 256 * kVecElems;  // 8 register-cached passes of a 256-thread work-group
  TORCH_CHECK(
      group_size <= kMaxGroupSize,
      "grouped_gemma_rmsnorm: group_size ",
      group_size,
      " is too large (max ",
      kMaxGroupSize,
      ")");

  const int64_t hidden = input.size(-1);
  TORCH_CHECK(hidden % group_size == 0, "hidden size ", hidden, " must be divisible by group_size ", group_size);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == hidden, "weight must be 1-D with hidden_size elements");
  TORCH_CHECK(
      weight.device() == input.device() && output.device() == input.device(),
      "input, weight and output must be on the same device");
  for (const auto* t : {&input, &weight, &output}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "tensors must be 16-byte aligned");
  }

  if (input.numel() == 0) {
    return;
  }
  at::assert_no_partial_overlap(output, input);
  at::assert_no_overlap(output, weight);
  TORCH_CHECK(
      input.numel() <= std::numeric_limits<int32_t>::max(),
      "grouped_gemma_rmsnorm: input is too large (",
      input.numel(),
      " elements)");

  const int64_t M = input.numel() / group_size;
  c10::DeviceGuard guard(input.device());
  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "grouped_gemma_rmsnorm", [&]() {
        GroupedGemmaRMSNormKernelImplInternal<scalar_t>(input, weight, M, group_size, hidden / group_size, eps, output);
      });
}

}  // namespace at::native::xpu
