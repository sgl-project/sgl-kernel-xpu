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

template <typename scalar_t, typename weight_t, typename mean_t = float>
class RMSNormForward : public NormForward<scalar_t, weight_t, true> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef NormForward<scalar_t, weight_t, true> NF;
  RMSNormForward() = delete;
  RMSNormForward(
      scalar_t* X_data, scalar_t* Y_data, mean_t* var_data, weight_t* gamma_data, accscalar_t eps, int64_t M, int64_t N)
      : NormForward<scalar_t, weight_t, true>(X_data, Y_data, nullptr, var_data, gamma_data, nullptr, eps), M(M), N(N) {
    numel = M * N;
  };

  template <int vec_size, typename vec_t, typename weight_vec_t, typename index_t, typename nd_item_id>
  void reduce_combine(nd_item_id item_id, const NormConfig& cfg, accscalar_t& sum_value, accscalar_t& sum_tmp) const {
    auto group_id = item_id.get_group(0);
    auto group_id_foreach = item_id.get_group(1);
    auto local_id = item_id.get_local_id(2);
    index_t group_offset = group_id * cfg.Plane;

    for (index_t j = local_id * vec_size; j < cfg.WGPlane; j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < cfg.Plane) {
        vec_t value = *(reinterpret_cast<vec_t*>(NF::X_data + group_offset + plane_offset));
        for (int v = 0; v < vec_size; ++v) {
          sum_value += Numerics<accscalar_t>::pow(value[v], 2);
        }
      }
    }
  }

  template <typename nd_item_id>
  void reduce_project(nd_item_id item_id, accscalar_t sum_value, accscalar_t sum_tmp, const NormConfig& cfg) const {
    auto group_id = item_id.get_group(0);
    accscalar_t scale = static_cast<accscalar_t>(cfg.Plane);
    NF::var_data[group_id] = static_cast<accscalar_t>(
        Numerics<accscalar_t>::rsqrt(sum_value < 0 ? 0 : sum_value / scale + static_cast<accscalar_t>(NF::eps)));
  }

  template <int vec_size, typename index_t, typename vec_t, typename weight_vec_t, typename nd_item_id>
  void update(nd_item_id item_id, const NormConfig& cfg, accscalar_t sum_value = 0, accscalar_t sum_tmp = 0) const {
    auto group_id = item_id.get_group(0);
    auto group_id_foreach = item_id.get_group(1);
    auto local_id = item_id.get_local_id(2);

    index_t group_offset = group_id * cfg.Plane;
    if (cfg.workgroup_num_foreach == 1) {
      if (local_id == 0) {
        reduce_project(item_id, sum_value, sum_tmp, cfg);
      }
      item_id.barrier(DECLARE_SYCL_GLOBAL_FENCE);
    }

    auto var_val = NF::var_data[group_id];
    for (index_t j = local_id * vec_size; j < cfg.WGPlane; j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < cfg.Plane) {
        vec_t X_val = *(reinterpret_cast<vec_t*>(NF::X_data + group_offset + plane_offset));
        vec_t Y_val;
        weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(NF::gamma_data + plane_offset));

        for (int v = 0; v < vec_size; ++v) {
          Y_val[v] = static_cast<scalar_t>(gamma_val[v] * var_val * X_val[v]);
        }
        *(reinterpret_cast<vec_t*>(NF::Y_data + group_offset + plane_offset)) = Y_val;
      }
    }
  }

  int64_t M;
  int64_t N;
  int64_t numel;
};

template <typename scalar_t, typename weight_t, typename mean_t = float>
class AddRMSNormForward : public RMSNormForward<scalar_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef NormForward<scalar_t, weight_t, true> NF;
  AddRMSNormForward() = delete;
  AddRMSNormForward(
      scalar_t* X_data,
      scalar_t* Y_data,
      mean_t* var_data,
      weight_t* gamma_data,
      accscalar_t eps,
      scalar_t* add_data,
      int64_t M,
      int64_t N)
      : RMSNormForward<scalar_t, weight_t>(X_data, Y_data, var_data, gamma_data, eps, M, N), add_data(add_data){};
  template <int vec_size, typename vec_t, typename weight_vec_t, typename index_t, typename nd_item_id>
  void reduce_combine(nd_item_id item_id, const NormConfig& cfg, accscalar_t& sum_value, accscalar_t& sum_tmp) const {
    auto group_id = item_id.get_group(0);
    auto group_id_foreach = item_id.get_group(1);
    auto local_id = item_id.get_local_id(2);
    index_t group_offset = group_id * cfg.Plane;

    for (index_t j = local_id * vec_size; j < cfg.WGPlane; j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < cfg.Plane) {
        vec_t X_value = *(reinterpret_cast<vec_t*>(NF::X_data + group_offset + plane_offset));
        vec_t add_value = *(reinterpret_cast<vec_t*>(add_data + group_offset + plane_offset));
        for (int v = 0; v < vec_size; ++v) {
          X_value[v] += add_value[v];
          sum_value += Numerics<accscalar_t>::pow(X_value[v], 2);
        }
        *(reinterpret_cast<vec_t*>(add_data + group_offset + plane_offset)) = X_value;
        *(reinterpret_cast<vec_t*>(NF::X_data + group_offset + plane_offset)) = X_value;
      }
    }
  }
  scalar_t* add_data;
};

template <typename scalar_t, typename weight_t, typename mean_t = float>
class GemmaRMSNormForward : public RMSNormForward<scalar_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef NormForward<scalar_t, weight_t, true> NF;
  typedef RMSNormForward<scalar_t, weight_t> RNF;
  GemmaRMSNormForward() = delete;
  GemmaRMSNormForward(
      scalar_t* X_data, scalar_t* Y_data, mean_t* var_data, weight_t* gamma_data, accscalar_t eps, int64_t M, int64_t N)
      : RMSNormForward<scalar_t, weight_t>(X_data, Y_data, var_data, gamma_data, eps, M, N){};
  template <int vec_size, typename index_t, typename vec_t, typename weight_vec_t, typename nd_item_id>
  void update(nd_item_id item_id, const NormConfig& cfg, accscalar_t sum_value = 0, accscalar_t sum_tmp = 0) const {
    auto group_id = item_id.get_group(0);
    auto group_id_foreach = item_id.get_group(1);
    auto local_id = item_id.get_local_id(2);

    index_t group_offset = group_id * cfg.Plane;
    if (cfg.workgroup_num_foreach == 1) {
      if (local_id == 0) {
        RNF::reduce_project(item_id, sum_value, sum_tmp, cfg);
      }
      item_id.barrier(DECLARE_SYCL_GLOBAL_FENCE);
    }

    auto var_val = NF::var_data[group_id];
    for (index_t j = local_id * vec_size; j < cfg.WGPlane; j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < cfg.Plane) {
        vec_t X_val = *(reinterpret_cast<vec_t*>(NF::X_data + group_offset + plane_offset));
        vec_t Y_val;
        weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(NF::gamma_data + plane_offset));

        for (int v = 0; v < vec_size; ++v) {
          Y_val[v] = static_cast<scalar_t>((accscalar_t(1.0) + gamma_val[v]) * var_val * X_val[v]);
        }
        *(reinterpret_cast<vec_t*>(NF::Y_data + group_offset + plane_offset)) = Y_val;
      }
    }
  }
};

template <typename scalar_t, typename weight_t, typename mean_t = float>
class GemmaAddRMSNormForward : public AddRMSNormForward<scalar_t, weight_t>,
                               public GemmaRMSNormForward<scalar_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef NormForward<scalar_t, weight_t, true> NF;
  typedef RMSNormForward<scalar_t, weight_t> RNF;
  typedef AddRMSNormForward<scalar_t, weight_t> ARNF;
  typedef GemmaRMSNormForward<scalar_t, weight_t> GRNF;
  GemmaAddRMSNormForward() = delete;
  GemmaAddRMSNormForward(
      scalar_t* X_data,
      scalar_t* Y_data,
      mean_t* var_data,
      weight_t* gamma_data,
      accscalar_t eps,
      scalar_t* add_data,
      int64_t M,
      int64_t N)
      : AddRMSNormForward<scalar_t, weight_t>(X_data, Y_data, var_data, gamma_data, eps, add_data, M, N),
        GemmaRMSNormForward<scalar_t, weight_t>(X_data, Y_data, var_data, gamma_data, eps, M, N) {}
  int get_update_vec_size(int Plane, int vec_size) {
    return ARNF::get_update_vec_size(Plane, vec_size);
  }
  template <int vec_size, typename index_t, typename vec_t, typename weight_vec_t, typename nd_item_id>
  void update(nd_item_id item_id, const NormConfig& cfg, accscalar_t sum_value = 0, accscalar_t sum_tmp = 0) const {
    GRNF::template update<vec_size, index_t, vec_t, weight_vec_t, nd_item_id>(item_id, cfg, sum_value, sum_tmp);
  }
  template <int vec_size, typename vec_t, typename weight_vec_t, typename index_t, typename nd_item_id>
  void reduce_combine(nd_item_id item_id, const NormConfig& cfg, accscalar_t& sum_value, accscalar_t& sum_tmp) const {
    ARNF::template reduce_combine<vec_size, vec_t, weight_vec_t, index_t, nd_item_id>(item_id, cfg, sum_value, sum_tmp);
  }
};
template <
    typename scalar_t,
    typename weight_t,
    int vec_size,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false,
    typename mean_t = float,
    typename index_t = uint32_t>
struct FusedNormKernelFunctor {
  using accscalar_t = acc_type<scalar_t>;
  using vec_t = aligned_vector_loop<scalar_t, vec_size>;
  using weight_vec_t = aligned_vector_loop<weight_t, vec_size>;
  [[sycl::reqd_sub_group_size(NUM_REDUCE_STAGES)]] void operator()(sycl::nd_item<3> item_id) const {
    accscalar_t sum1 = 0;
    accscalar_t sum2 = 0;
    norm.template reduce_combine<vec_size, vec_t, weight_vec_t, index_t>(item_id, cfg, sum1, sum2);

    if constexpr (one_moment) {
      sum1 = sycl::reduce_over_group(item_id.get_group(), sum1, sycl::plus<accscalar_t>());
    } else {
      norm_group_reduce<accscalar_t>(
          item_id, cfg.sub_group_num, sum1, sum2, local_sum1, local_sum2, [](accscalar_t a, accscalar_t b) {
            return a + b;
          });
    }
    norm.template update<vec_size, index_t, vec_t, weight_vec_t>(item_id, cfg, sum1, sum2);
  }
  FusedNormKernelFunctor(
      sycl_local_acc_t<accscalar_t> local_sum1_,
      sycl_local_acc_t<accscalar_t> local_sum2_,
      Norm<scalar_t, weight_t, mean_t> norm_,
      NormConfig cfg_)
      : local_sum1(local_sum1_), local_sum2(local_sum2_), norm(norm_), cfg(cfg_) {}

 private:
  sycl_local_acc_t<accscalar_t> local_sum1;
  sycl_local_acc_t<accscalar_t> local_sum2;
  Norm<scalar_t, weight_t, mean_t> norm;
  const NormConfig cfg;
};

template <
    typename scalar_t,
    typename weight_t,
    int vec_size,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false,
    typename mean_t = float,
    typename index_t = uint32_t>
void fused_norm_kernel(Norm<scalar_t, weight_t, mean_t>& norm, const NormConfig& cfg) {
  using accscalar_t = acc_type<scalar_t>;
  sycl::range<3> local_range{
      1, static_cast<size_t>(cfg.workgroup_num_foreach), static_cast<size_t>(cfg.workgroup_size)};
  sycl::range<3> global_range{
      static_cast<size_t>(cfg.workgroup_num),
      static_cast<size_t>(cfg.workgroup_num_foreach),
      static_cast<size_t>(cfg.workgroup_size)};

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  auto cgf = [&](sycl::handler& cgh) {
    sycl_local_acc_t<accscalar_t> local_sum1(cfg.sub_group_num, cgh);
    sycl_local_acc_t<accscalar_t> local_sum2(cfg.sub_group_num, cgh);
    FusedNormKernelFunctor<scalar_t, weight_t, vec_size, Norm, one_moment> kfn(local_sum1, local_sum2, norm, cfg);
    cgh.parallel_for<decltype(kfn)>(sycl::nd_range<3>(sycl::range<3>(global_range), sycl::range<3>(local_range)), kfn);
  };
  queue.submit(cgf);
}

template <
    typename scalar_t,
    typename weight_t,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false,
    typename mean_t = float>
void launch_vectorized_fused_norm_kernel(Norm<scalar_t, weight_t, mean_t>& norm, const NormConfig& config) {
  int vec_size = norm.get_update_vec_size(config.WGPlane, config.max_vec_size);
#define VECTORIZED_FUSED_NORM_KERNEL(vec_size)                                       \
  {                                                                                  \
    fused_norm_kernel<scalar_t, weight_t, vec_size, Norm, one_moment>(norm, config); \
    break;                                                                           \
  }
  switch (vec_size) {
    case 8: {
      VECTORIZED_FUSED_NORM_KERNEL(8);
    }
    case 4: {
      VECTORIZED_FUSED_NORM_KERNEL(4);
    }
    case 2: {
      VECTORIZED_FUSED_NORM_KERNEL(2);
    }
    default: {
      VECTORIZED_FUSED_NORM_KERNEL(1);
    }
  }
}

template <typename scalar_t, typename weight_t, typename mean_t = float>
void RMSNormKernelImplInternal(
    const Tensor& X, const Tensor& gemma, int64_t M, int64_t N, acc_type<scalar_t> eps, Tensor& Y, Tensor& rstd) {
  scalar_t* X_data = X.data_ptr<scalar_t>();
  scalar_t* Y_data = Y.data_ptr<scalar_t>();
  mean_t* var_data = rstd.data_ptr<mean_t>();
  weight_t* gemma_data = gemma.defined() ? gemma.data_ptr<weight_t>() : nullptr;

  auto config = NormConfig(M, N, 1, sizeof(scalar_t));
  RMSNormForward<scalar_t, weight_t> rms_norm_forward(X_data, Y_data, var_data, gemma_data, eps, M, N);
  config.workgroup_num_foreach = 1;
  config.WGPlane = config.Plane;

  launch_vectorized_fused_norm_kernel<scalar_t, weight_t, RMSNormForward, true>(rms_norm_forward, config);
}

template <typename scalar_t, typename weight_t, typename mean_t = float>
void FusedAddRMSNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gemma,
    int64_t M,
    int64_t N,
    acc_type<scalar_t> eps,
    Tensor& rstd,
    Tensor& residual) {
  scalar_t* X_data = X.data_ptr<scalar_t>();
  mean_t* var_data = rstd.data_ptr<mean_t>();
  weight_t* gemma_data = gemma.defined() ? gemma.data_ptr<weight_t>() : nullptr;
  scalar_t* residual_data = residual.data_ptr<scalar_t>();

  auto config = NormConfig(M, N, 1, sizeof(scalar_t));
  AddRMSNormForward<scalar_t, weight_t> add_rms_norm_forward(
      X_data, X_data, var_data, gemma_data, eps, residual_data, M, N);
  config.workgroup_num_foreach = 1;
  config.WGPlane = config.Plane;

  launch_vectorized_fused_norm_kernel<scalar_t, weight_t, AddRMSNormForward, true>(add_rms_norm_forward, config);
}

template <typename scalar_t, typename weight_t, typename mean_t = float>
void GemmaRMSNormKernelImplInternal(
    const Tensor& X, const Tensor& gemma, int64_t M, int64_t N, acc_type<scalar_t> eps, Tensor& Y, Tensor& rstd) {
  scalar_t* X_data = X.data_ptr<scalar_t>();
  scalar_t* Y_data = Y.data_ptr<scalar_t>();
  mean_t* var_data = rstd.data_ptr<mean_t>();
  weight_t* gemma_data = gemma.defined() ? gemma.data_ptr<weight_t>() : nullptr;

  auto config = NormConfig(M, N, 1, sizeof(scalar_t));
  GemmaRMSNormForward<scalar_t, weight_t> gemma_rms_norm_forward(X_data, Y_data, var_data, gemma_data, eps, M, N);
  config.workgroup_num_foreach = 1;
  config.WGPlane = config.Plane;

  launch_vectorized_fused_norm_kernel<scalar_t, weight_t, GemmaRMSNormForward, true>(gemma_rms_norm_forward, config);
}

template <typename scalar_t, typename weight_t, typename mean_t = float>
void GemmaFusedAddRMSNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gemma,
    int64_t M,
    int64_t N,
    acc_type<scalar_t> eps,
    Tensor& rstd,
    Tensor& residual) {
  scalar_t* X_data = X.data_ptr<scalar_t>();
  mean_t* var_data = rstd.data_ptr<mean_t>();
  weight_t* gemma_data = gemma.defined() ? gemma.data_ptr<weight_t>() : nullptr;
  scalar_t* residual_data = residual.data_ptr<scalar_t>();

  auto config = NormConfig(M, N, 1, sizeof(scalar_t));
  GemmaAddRMSNormForward<scalar_t, weight_t> gemma_add_rms_norm_forward(
      X_data, X_data, var_data, gemma_data, eps, residual_data, M, N);
  config.workgroup_num_foreach = 1;
  config.WGPlane = config.Plane;

  launch_vectorized_fused_norm_kernel<scalar_t, weight_t, GemmaAddRMSNormForward, true>(
      gemma_add_rms_norm_forward, config);
}

void rmsnorm(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps) {
  std::optional<torch::Tensor> opt_weight = weight;
  std::optional<torch::Tensor> opt_bias;
  auto M_N = _check_layer_norm_inputs(input, c10::IntArrayRef({input.size(-1)}), opt_weight, opt_bias);
  auto M = M_N.first;
  auto N = M_N.second;

  Tensor input_ = (input.dim() == 1) ? input.reshape({M, N}) : input;
  Tensor output_ = (output.dim() == 1) ? output.reshape({M, N}) : output;
  Tensor weight_ = (weight.dim() == 1) ? weight.reshape({N}) : weight;
  Tensor rstd = at::empty({M}, input_.options().dtype(kFloat));

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, input_.scalar_type(), "RMSNormKernelImpl", [&]() {
        RMSNormKernelImplInternal<scalar_t, scalar_t>(
            input_, weight_, M, N, static_cast<acc_type<scalar_t>>(eps), output_, rstd);
      });
}

void fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps) {
  std::optional<torch::Tensor> opt_weight = weight;
  std::optional<torch::Tensor> opt_bias;
  auto M_N = _check_layer_norm_inputs(input, c10::IntArrayRef({input.size(-1)}), opt_weight, opt_bias);
  auto M = M_N.first;
  auto N = M_N.second;

  Tensor input_ = (input.dim() == 1) ? input.reshape({M, N}) : input;
  Tensor residual_ = (residual.dim() == 1) ? residual.reshape({M, N}) : residual;
  Tensor weight_ = (weight.dim() == 1) ? weight.reshape({N}) : weight;
  Tensor rstd = at::empty({M}, input_.options().dtype(kFloat));

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, input_.scalar_type(), "FusedAddRMSNormKernelImpl", [&]() {
        FusedAddRMSNormKernelImplInternal<scalar_t, scalar_t>(
            input_, weight_, M, N, static_cast<acc_type<scalar_t>>(eps), rstd, residual_);
      });
}

void gemma_rmsnorm(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, double eps) {
  std::optional<torch::Tensor> opt_weight = weight;
  std::optional<torch::Tensor> opt_bias;
  auto M_N = _check_layer_norm_inputs(input, c10::IntArrayRef({input.size(-1)}), opt_weight, opt_bias);
  auto M = M_N.first;
  auto N = M_N.second;

  Tensor input_ = (input.dim() == 1) ? input.reshape({M, N}) : input;
  Tensor output_ = (output.dim() == 1) ? output.reshape({M, N}) : output;
  Tensor weight_ = (weight.dim() == 1) ? weight.reshape({N}) : weight;
  Tensor rstd = at::empty({M}, input_.options().dtype(kFloat));

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, input_.scalar_type(), "GemmaRMSNormKernelImpl", [&]() {
        GemmaRMSNormKernelImplInternal<scalar_t, scalar_t>(
            input_, weight_, M, N, static_cast<acc_type<scalar_t>>(eps), output_, rstd);
      });
}

void gemma_fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, double eps) {
  std::optional<torch::Tensor> opt_weight = weight;
  std::optional<torch::Tensor> opt_bias;
  auto M_N = _check_layer_norm_inputs(input, c10::IntArrayRef({input.size(-1)}), opt_weight, opt_bias);
  auto M = M_N.first;
  auto N = M_N.second;

  Tensor input_ = (input.dim() == 1) ? input.reshape({M, N}) : input;
  Tensor residual_ = (residual.dim() == 1) ? residual.reshape({M, N}) : residual;
  Tensor weight_ = (weight.dim() == 1) ? weight.reshape({N}) : weight;
  Tensor rstd = at::empty({M}, input_.options().dtype(kFloat));

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, input_.scalar_type(), "GemmaFusedAddRMSNormKernelImpl", [&]() {
        GemmaFusedAddRMSNormKernelImplInternal<scalar_t, scalar_t>(
            input_, weight_, M, N, static_cast<acc_type<scalar_t>>(eps), rstd, residual_);
      });
}

}  // namespace at::native::xpu