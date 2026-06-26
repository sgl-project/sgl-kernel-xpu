#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "MemoryAccess.h"
#include "Norm.h"
#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {
template <typename ScalarType, int Dims = 1>
using sycl_local_acc_t = sycl::local_accessor<ScalarType, Dims>;

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
// 2D or 3D tensor without requiring a contiguous copy.  For 2D and
// flattenable 3D tensors, (inner_size == 1, inner_stride == 0) reduces the
// kernel's per-row offset formula
//
//   offset(r) = (r / inner_size) * batch_stride + (r % inner_size) * inner_stride
//
// to the existing behaviour `offset(r) = r * batch_stride`.  For
// non-flattenable 3D tensors (e.g. a per-head slice of a packed QKV buffer
// reshaped to (tokens, heads, head_dim)) we fall back to the general formula
// by setting inner_size = size(1) and inner_stride = stride(1).
struct RowStrides {
  int64_t batch_stride;
  int64_t inner_size;
  int64_t inner_stride;
};

static inline RowStrides get_row_strides(const Tensor& t) {
  TORCH_CHECK(t.dim() == 2 || t.dim() == 3, "get_row_strides: expected a 2D or 3D tensor, got ", t.dim(), "D");
  if (t.dim() == 2) {
    return {t.stride(0), 1, 0};
  }
  // 3D
  int64_t outer_stride = t.stride(0);
  int64_t inner_size = t.size(1);
  int64_t inner_stride = t.stride(1);
  if (t.size(0) == 1 || outer_stride == inner_size * inner_stride) {
    // Flattenable: a single stride describes all rows.
    return {inner_stride, 1, 0};
  }
  return {outer_stride, inner_size, inner_stride};
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

  template <int vec_size, int ITERS, bool cache_inputs, typename vec_t, typename index_t>
  void reduce_combine(
      sycl::nd_item<1> item_id,
      const NormConfig& cfg,
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

  accscalar_t reduce_project(sycl::nd_item<1> item_id, const NormConfig& cfg, accscalar_t sum_value) const {
    sum_value = sycl::reduce_over_group(item_id.get_group(), sum_value, sycl::plus<accscalar_t>());
    sum_value = sum_value < static_cast<accscalar_t>(0) ? static_cast<accscalar_t>(0) : sum_value;
    return Numerics<accscalar_t>::rsqrt(
        sum_value / static_cast<accscalar_t>(cfg.Plane) + static_cast<accscalar_t>(NF::eps));
  }

  template <int vec_size, int ITERS, bool cache_inputs, typename vec_t, typename weight_vec_t, typename index_t>
  void update(
      sycl::nd_item<1> item_id,
      const NormConfig& cfg,
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
            y_val[v] = static_cast<scalar_t>(
                static_cast<accscalar_t>(x_val[v]) * rstd * static_cast<accscalar_t>(gamma_val[v]));
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
          y_val[v] =
              static_cast<scalar_t>(static_cast<accscalar_t>(x_val[v]) * rstd * static_cast<accscalar_t>(gamma_val[v]));
        }
        *(reinterpret_cast<vec_t*>(NF::Y_data + y_group_offset + plane_offset)) = y_val;
      }
    }
  }

 public:
  weight_t* gamma_data;
};

template <typename scalar_t, typename weight_t>
class GemmaRMSNormNoRstdForward : public RMSNormNoRstdForward<scalar_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef RMSNormNoRstdForward<scalar_t, weight_t> RNF;
  GemmaRMSNormNoRstdForward() = delete;
  using RNF::RNF;

  template <int vec_size, int ITERS, bool cache_inputs, typename vec_t, typename weight_vec_t, typename index_t>
  void update(
      sycl::nd_item<1> item_id,
      const NormConfig& cfg,
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
            y_val[v] = static_cast<scalar_t>(
                static_cast<accscalar_t>(x_val[v]) * rstd *
                (static_cast<accscalar_t>(1.0) + static_cast<accscalar_t>(gamma_val[v])));
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
          y_val[v] = static_cast<scalar_t>(
              static_cast<accscalar_t>(x_val[v]) * rstd *
              (static_cast<accscalar_t>(1.0) + static_cast<accscalar_t>(gamma_val[v])));
        }
        *(reinterpret_cast<vec_t*>(RNF::Y_data + y_group_offset + plane_offset)) = y_val;
      }
    }
  }
};

template <typename scalar_t, typename weight_t>
class AddRMSNormNoRstdForward : public RMSNormNoRstdForward<scalar_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef RMSNormNoRstdForward<scalar_t, weight_t> RNF;
  typedef NormNoRstdForward<scalar_t> NF;
  AddRMSNormNoRstdForward() = delete;
  AddRMSNormNoRstdForward(scalar_t* X_data, scalar_t* Y_data, weight_t* gamma_data, accscalar_t eps, scalar_t* add_data)
      : RMSNormNoRstdForward<scalar_t, weight_t>(X_data, Y_data, gamma_data, eps), add_data(add_data) {}

  int get_update_vec_size(int Plane, int vec_size) const {
    return NF::get_aligned_update_vec_size(Plane, vec_size, NF::X_data, NF::Y_data, RNF::gamma_data, add_data);
  }

  template <int vec_size, int ITERS, bool cache_inputs, typename vec_t, typename index_t>
  void reduce_combine(
      sycl::nd_item<1> item_id,
      const NormConfig& cfg,
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

  template <int vec_size, int ITERS, bool cache_inputs, typename vec_t, typename weight_vec_t, typename index_t>
  void update(
      sycl::nd_item<1> item_id,
      const NormConfig& cfg,
      index_t x_group_offset,
      index_t y_group_offset,
      accscalar_t rstd,
      const vec_t (&reg)[cache_inputs ? ITERS : 1]) const {
    if constexpr (cache_inputs) {
      RNF::template update<vec_size, ITERS, cache_inputs, vec_t, weight_vec_t, index_t>(
          item_id, cfg, x_group_offset, y_group_offset, rstd, reg);
    } else {
      const index_t lid = item_id.get_local_id(0);
      for (index_t plane_offset = lid * vec_size; plane_offset < cfg.Plane;
           plane_offset += cfg.workgroup_size * vec_size) {
        vec_t x_val = *(reinterpret_cast<vec_t*>(add_data + x_group_offset + plane_offset));
        weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(RNF::gamma_data + plane_offset));
        vec_t y_val;
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          y_val[v] =
              static_cast<scalar_t>(static_cast<accscalar_t>(x_val[v]) * rstd * static_cast<accscalar_t>(gamma_val[v]));
        }
        *(reinterpret_cast<vec_t*>(NF::Y_data + y_group_offset + plane_offset)) = y_val;
      }
    }
  }

 public:
  scalar_t* add_data;
};

template <typename scalar_t, typename weight_t>
class GemmaAddRMSNormNoRstdForward : public GemmaRMSNormNoRstdForward<scalar_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef GemmaRMSNormNoRstdForward<scalar_t, weight_t> Base;
  typedef NormNoRstdForward<scalar_t> NF;
  GemmaAddRMSNormNoRstdForward() = delete;
  GemmaAddRMSNormNoRstdForward(
      scalar_t* X_data, scalar_t* Y_data, weight_t* gamma_data, accscalar_t eps, scalar_t* add_data)
      : GemmaRMSNormNoRstdForward<scalar_t, weight_t>(X_data, Y_data, gamma_data, eps), add_data(add_data) {}

  int get_update_vec_size(int Plane, int vec_size) const {
    return NF::get_aligned_update_vec_size(Plane, vec_size, NF::X_data, NF::Y_data, Base::gamma_data, add_data);
  }

  template <int vec_size, int ITERS, bool cache_inputs, typename vec_t, typename index_t>
  void reduce_combine(
      sycl::nd_item<1> item_id,
      const NormConfig& cfg,
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

  template <int vec_size, int ITERS, bool cache_inputs, typename vec_t, typename weight_vec_t, typename index_t>
  void update(
      sycl::nd_item<1> item_id,
      const NormConfig& cfg,
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
        vec_t x_val = *(reinterpret_cast<vec_t*>(add_data + x_group_offset + plane_offset));
        weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(Base::gamma_data + plane_offset));
        vec_t y_val;
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          y_val[v] = static_cast<scalar_t>(
              static_cast<accscalar_t>(x_val[v]) * rstd *
              (static_cast<accscalar_t>(1.0) + static_cast<accscalar_t>(gamma_val[v])));
        }
        *(reinterpret_cast<vec_t*>(NF::Y_data + y_group_offset + plane_offset)) = y_val;
      }
    }
  }

 public:
  scalar_t* add_data;
};

template <
    typename scalar_t,
    typename weight_t,
    int vec_size,
    int ITERS,
    typename Norm,
    bool cache_inputs,
    typename index_t = uint32_t>
struct RMSNormNoRstdKernelFunctor {
  using accscalar_t = acc_type<scalar_t>;
  using vec_t = aligned_vector_loop<scalar_t, vec_size>;
  using weight_vec_t = aligned_vector_loop<weight_t, vec_size>;

  [[sycl::reqd_sub_group_size(NUM_REDUCE_STAGES)]] void operator()(sycl::nd_item<1> item_id) const {
    const index_t row = item_id.get_group(0);
    const index_t x_group_offset =
        (row / cfg.input_inner_size) * cfg.input_batch_stride + (row % cfg.input_inner_size) * cfg.input_inner_stride;
    const index_t y_group_offset = (row / cfg.output_inner_size) * cfg.output_batch_stride +
                                   (row % cfg.output_inner_size) * cfg.output_inner_stride;

    accscalar_t sum_value = 0;
    vec_t reg[cache_inputs ? ITERS : 1];
    norm.template reduce_combine<vec_size, ITERS, cache_inputs, vec_t, index_t>(
        item_id, cfg, x_group_offset, sum_value, reg);
    const accscalar_t rstd = norm.reduce_project(item_id, cfg, sum_value);
    norm.template update<vec_size, ITERS, cache_inputs, vec_t, weight_vec_t, index_t>(
        item_id, cfg, x_group_offset, y_group_offset, rstd, reg);
  }

  RMSNormNoRstdKernelFunctor(Norm norm_, NormConfig cfg_) : norm(norm_), cfg(cfg_) {}

 private:
  Norm norm;
  const NormConfig cfg;
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
    int64_t input_inner_size,
    int64_t input_inner_stride,
    int64_t output_inner_size,
    int64_t output_inner_stride) {
  scalar_t* X_data = X.data_ptr<scalar_t>();
  scalar_t* Y_data = Y.data_ptr<scalar_t>();
  weight_t* gemma_data = gemma.data_ptr<weight_t>();
  RMSNormNoRstdForward<scalar_t, weight_t> rms_norm_forward(X_data, Y_data, gemma_data, eps);

  auto config = NormConfig(
      M,
      N,
      1,
      sizeof(scalar_t),
      input_batch_stride,
      output_batch_stride,
      [&](int plane, int max_vec_size) { return rms_norm_forward.get_update_vec_size(plane, max_vec_size); },
      input_inner_size,
      input_inner_stride,
      output_inner_size,
      output_inner_stride);
  launch_vectorized_rmsnorm_no_rstd_kernel<scalar_t, weight_t>(rms_norm_forward, config);
}

template <typename scalar_t, typename weight_t>
void FusedAddRMSNormKernelImplInternal(
    const Tensor& X, const Tensor& gemma, int64_t M, int64_t N, acc_type<scalar_t> eps, Tensor& residual) {
  scalar_t* X_data = X.data_ptr<scalar_t>();
  weight_t* gemma_data = gemma.data_ptr<weight_t>();
  scalar_t* residual_data = residual.data_ptr<scalar_t>();
  AddRMSNormNoRstdForward<scalar_t, weight_t> norm_forward(X_data, X_data, gemma_data, eps, residual_data);

  auto config = NormConfig(
      M,
      N,
      1,
      sizeof(scalar_t),
      N,
      N,
      [&](int plane, int max_vec_size) { return norm_forward.get_update_vec_size(plane, max_vec_size); },
      1,
      0,
      1,
      0);
  launch_vectorized_rmsnorm_no_rstd_kernel<scalar_t, weight_t>(norm_forward, config);
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
    int64_t input_inner_size,
    int64_t input_inner_stride,
    int64_t output_inner_size,
    int64_t output_inner_stride) {
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
      input_inner_size,
      input_inner_stride,
      output_inner_size,
      output_inner_stride);
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

void rmsnorm(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps) {
  std::optional<torch::Tensor> opt_weight = weight;
  std::optional<torch::Tensor> opt_bias;
  auto [M, N] = _check_layer_norm_inputs(input, c10::IntArrayRef({input.size(-1)}), opt_weight, opt_bias);

  // Derive row-stride info directly from input/output so the kernel can
  // handle non-flattenable 3D tensors (e.g. QKV slices) natively.
  RowStrides in_strides = get_row_strides(input);
  RowStrides out_strides = get_row_strides(output);
  Tensor weight_ = (weight.dim() == 1) ? weight.reshape({N}) : weight;

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
                  in_strides.inner_size,
                  in_strides.inner_stride,
                  out_strides.inner_size,
                  out_strides.inner_stride);
            });
      });
}

void fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps) {
  TORCH_CHECK(input.is_contiguous(), "fused_add_rmsnorm: input must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "fused_add_rmsnorm: residual must be contiguous");
  std::optional<torch::Tensor> opt_weight = weight;
  std::optional<torch::Tensor> opt_bias;
  auto [M, N] = _check_layer_norm_inputs(input, c10::IntArrayRef({input.size(-1)}), opt_weight, opt_bias);

  // Flatten leading dimensions to 2D for the kernel
  Tensor input_ = flatten_to_2d(input, M, N);
  Tensor residual_ = flatten_to_2d(residual, M, N);

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, input_.scalar_type(), "FusedAddRMSNormKernelImpl", [&]() {
        SYCL_DISPATCH_WEIGHT_TYPES(
            at::ScalarType::Half, at::ScalarType::BFloat16, weight.scalar_type(), "FusedAddRMSNormKernelImpl", [&]() {
              FusedAddRMSNormKernelImplInternal<scalar_t, weight_t>(
                  input_, weight, M, N, static_cast<acc_type<scalar_t>>(eps), residual_);
            });
      });
}

void gemma_rmsnorm(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps) {
  std::optional<torch::Tensor> opt_weight = weight;
  std::optional<torch::Tensor> opt_bias;
  auto [M, N] = _check_layer_norm_inputs(input, c10::IntArrayRef({input.size(-1)}), opt_weight, opt_bias);

  RowStrides in_strides = get_row_strides(input);
  RowStrides out_strides = get_row_strides(output);
  Tensor weight_ = (weight.dim() == 1) ? weight.reshape({N}) : weight;

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
                  in_strides.inner_size,
                  in_strides.inner_stride,
                  out_strides.inner_size,
                  out_strides.inner_stride);
            });
      });
}

void gemma_fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, double eps) {
  TORCH_CHECK(input.is_contiguous(), "gemma_fused_add_rmsnorm: input must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "gemma_fused_add_rmsnorm: residual must be contiguous");
  std::optional<torch::Tensor> opt_weight = weight;
  std::optional<torch::Tensor> opt_bias;
  auto [M, N] = _check_layer_norm_inputs(input, c10::IntArrayRef({input.size(-1)}), opt_weight, opt_bias);

  // Flatten leading dimensions to 2D for the kernel
  Tensor input_ = flatten_to_2d(input, M, N);
  Tensor residual_ = flatten_to_2d(residual, M, N);
  Tensor weight_ = (weight.dim() == 1) ? weight.reshape({N}) : weight;

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
}

}  // namespace at::native::xpu
