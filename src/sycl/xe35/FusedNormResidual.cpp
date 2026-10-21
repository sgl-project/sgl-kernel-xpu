#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

#include "../MemoryAccess.h"
#include "../SYCLHelpers.h"
#include "sgl_kernel/xe35/residual_norm_ops.h"

namespace at::native::xpu {
namespace {

constexpr int kVecSize_xe35 = 8;
using fp32_vec_t_xe35 = aligned_vector_loop<float, kVecSize_xe35>;
using bf16_vec_t_xe35 = aligned_vector_loop<at::BFloat16, kVecSize_xe35>;

inline at::BFloat16 accurate_bfloat16_xe35(double value) {
  float high = static_cast<float>(value);
  if ((sycl::bit_cast<uint32_t>(high) & 0xffffu) == 0x8000u && value != static_cast<double>(high)) {
    high = sycl::nextafter(
        high,
        value > static_cast<double>(high) ? std::numeric_limits<float>::infinity()
                                          : -std::numeric_limits<float>::infinity());
  }
  return static_cast<at::BFloat16>(high);
}

struct NormArgs_xe35 {
  const at::BFloat16* __restrict__ branch;
  const float* __restrict__ stream;
  const at::BFloat16* __restrict__ post_weight;
  const at::BFloat16* __restrict__ pre_weight;
  const float* __restrict__ alpha;
  const float* __restrict__ beta;
  at::BFloat16* __restrict__ output;
  float* __restrict__ new_stream;
  float* __restrict__ output_fp32;
  int64_t hidden_size;
  float post_eps;
  float pre_eps;
  bool has_post_weight;
  bool post_norm_input_scale;
  bool round_pre_norm_input;
};



template <int WorkGroupSize, int MaxIters>
struct FusedPostNormResidualPreNorm_xe35 {
  NormArgs_xe35 args;
  [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t row_offset = item.get_group(0) * args.hidden_size;
    const int64_t lane = item.get_local_id(0);

    bf16_vec_t_xe35 branch_reg[MaxIters];
    double branch_sum_sq = 0.0;

#pragma unroll
    for (int it = 0; it < MaxIters; ++it) {
      const int64_t offset = (it * WorkGroupSize + lane) * kVecSize_xe35;
      if (offset >= args.hidden_size) {
        continue;
      }
      bf16_vec_t_xe35 value = *reinterpret_cast<const bf16_vec_t_xe35*>(args.branch + row_offset + offset);
#pragma unroll
      for (int v = 0; v < kVecSize_xe35; ++v) {
        float x = static_cast<float>(value[v]);
        if (args.has_post_weight && args.post_norm_input_scale) {
          value[v] = static_cast<at::BFloat16>(x * static_cast<float>(args.post_weight[offset + v]));
          x = static_cast<float>(value[v]);
        }
        branch_sum_sq += static_cast<double>(x) * static_cast<double>(x);
      }
      branch_reg[it] = value;
    }

    branch_sum_sq = sycl::reduce_over_group(item.get_group(), branch_sum_sq, sycl::plus<double>());
    const double post_rstd = sycl::rsqrt(branch_sum_sq / static_cast<double>(args.hidden_size) + args.post_eps);

    fp32_vec_t_xe35 new_stream_reg[MaxIters];
    float stream_sum_sq = 0.0f;

#pragma unroll
    for (int it = 0; it < MaxIters; ++it) {
      const int64_t offset = (it * WorkGroupSize + lane) * kVecSize_xe35;
      if (offset >= args.hidden_size) {
        continue;
      }
      const fp32_vec_t_xe35 stream_value = *reinterpret_cast<const fp32_vec_t_xe35*>(args.stream + row_offset + offset);
      const fp32_vec_t_xe35 alpha_value = *reinterpret_cast<const fp32_vec_t_xe35*>(args.alpha + offset);
      const fp32_vec_t_xe35 beta_value = *reinterpret_cast<const fp32_vec_t_xe35*>(args.beta + offset);
      fp32_vec_t_xe35 updated;

#pragma unroll
      for (int v = 0; v < kVecSize_xe35; ++v) {

        float post_norm_value = static_cast<float>(branch_reg[it][v]);
        double normalized_post = static_cast<double>(post_norm_value) * post_rstd;
        if (args.has_post_weight && !args.post_norm_input_scale) {
          normalized_post *= static_cast<double>(static_cast<float>(args.post_weight[offset + v]));
        }
        const at::BFloat16 normed_bf16 = accurate_bfloat16_xe35(normalized_post);
        const float value = static_cast<float>(
            static_cast<double>(alpha_value[v]) * static_cast<double>(stream_value[v]) +
            static_cast<double>(beta_value[v]) * static_cast<double>(static_cast<float>(normed_bf16)));
        updated[v] = value;
        const float pre_norm_value =
            args.round_pre_norm_input ? static_cast<float>(static_cast<at::BFloat16>(value)) : value;
        stream_sum_sq += pre_norm_value * pre_norm_value;
      }
      new_stream_reg[it] = updated;
    }

    stream_sum_sq = sycl::reduce_over_group(item.get_group(), stream_sum_sq, sycl::plus<float>());
    const float pre_rstd = sycl::rsqrt(stream_sum_sq / static_cast<float>(args.hidden_size) + args.pre_eps);

#pragma unroll
    for (int it = 0; it < MaxIters; ++it) {
      const int64_t offset = (it * WorkGroupSize + lane) * kVecSize_xe35;
      if (offset >= args.hidden_size) {
        continue;
      }
      const bf16_vec_t_xe35 weight = *reinterpret_cast<const bf16_vec_t_xe35*>(args.pre_weight + offset);
      bf16_vec_t_xe35 normalized;
#pragma unroll
      for (int v = 0; v < kVecSize_xe35; ++v) {
        const float pre_norm_value = args.round_pre_norm_input
                                         ? static_cast<float>(static_cast<at::BFloat16>(new_stream_reg[it][v]))
                                         : new_stream_reg[it][v];
        normalized[v] = static_cast<at::BFloat16>(pre_norm_value * pre_rstd * static_cast<float>(weight[v]));
      }
      *reinterpret_cast<fp32_vec_t_xe35*>(args.new_stream + row_offset + offset) = new_stream_reg[it];
      *reinterpret_cast<bf16_vec_t_xe35*>(args.output + row_offset + offset) = normalized;
      if (args.output_fp32 != nullptr) {
        fp32_vec_t_xe35 widened;
#pragma unroll
        for (int v = 0; v < kVecSize_xe35; ++v) {
          widened[v] = static_cast<float>(normalized[v]);
        }
        *reinterpret_cast<fp32_vec_t_xe35*>(args.output_fp32 + row_offset + offset) = widened;
      }
    }
  }
};

template <int WorkGroupSize, int MaxIters>
void launch_fused_norm_xe35(const NormArgs_xe35& args, int64_t rows, sycl::queue& queue) {
  FusedPostNormResidualPreNorm_xe35<WorkGroupSize, MaxIters> kernel{args};
  sycl_kernel_submit(sycl::range<1>(rows * WorkGroupSize), sycl::range<1>(WorkGroupSize), queue, kernel);
}

template <int WorkGroupSize>
void launch_for_group_xe35(const NormArgs_xe35& args, int64_t rows, sycl::queue& queue) {
  if (args.hidden_size <= WorkGroupSize * kVecSize_xe35) {
    launch_fused_norm_xe35<WorkGroupSize, 1>(args, rows, queue);
  } else if (args.hidden_size <= WorkGroupSize * kVecSize_xe35 * 2) {
    launch_fused_norm_xe35<WorkGroupSize, 2>(args, rows, queue);
  } else {
    TORCH_CHECK(
        args.hidden_size <= WorkGroupSize * kVecSize_xe35 * 4,
        "hidden size exceeds the four-vector per lane register budget");
    launch_fused_norm_xe35<WorkGroupSize, 4>(args, rows, queue);
  }
}

}

void fused_post_norm_residual_pre_norm_xe35(
    torch::Tensor branch,
    torch::Tensor stream,
    torch::Tensor post_weight,
    torch::Tensor pre_weight,
    torch::Tensor alpha,
    torch::Tensor beta,
    torch::Tensor output,
    torch::Tensor new_stream,
    double post_eps,
    double pre_eps,
    bool post_norm_input_scale,
    bool round_pre_norm_input,
    const std::optional<torch::Tensor>& output_fp32) {
  TORCH_CHECK(branch.device().is_xpu(), "branch must be an XPU tensor");
  TORCH_CHECK(branch.scalar_type() == at::kBFloat16, "branch must be bfloat16");
  TORCH_CHECK(stream.scalar_type() == at::kFloat, "stream must be float32");
  TORCH_CHECK(post_weight.scalar_type() == at::kBFloat16, "post_weight must be bfloat16");
  TORCH_CHECK(pre_weight.scalar_type() == at::kBFloat16, "pre_weight must be bfloat16");
  TORCH_CHECK(alpha.scalar_type() == at::kFloat, "alpha must be float32");
  TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be float32");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");
  TORCH_CHECK(new_stream.scalar_type() == at::kFloat, "new_stream must be float32");
  TORCH_CHECK(
      branch.dim() == 2 && branch.size(1) > 0 && branch.size(1) % kVecSize_xe35 == 0 && branch.size(1) <= 32768,
      "branch must have shape [M, H] with positive H divisible by 8 and H <= 32768");
  const int64_t hidden_size = branch.size(1);
  TORCH_CHECK(stream.device() == branch.device(), "stream must be on the branch device");
  TORCH_CHECK(post_weight.device() == branch.device(), "post_weight must be on the branch device");
  TORCH_CHECK(pre_weight.device() == branch.device(), "pre_weight must be on the branch device");
  TORCH_CHECK(alpha.device() == branch.device(), "alpha must be on the branch device");
  TORCH_CHECK(beta.device() == branch.device(), "beta must be on the branch device");
  TORCH_CHECK(output.device() == branch.device(), "output must be on the branch device");
  TORCH_CHECK(new_stream.device() == branch.device(), "new_stream must be on the branch device");
  TORCH_CHECK(stream.sizes() == branch.sizes(), "stream must match branch shape");
  TORCH_CHECK(output.sizes() == branch.sizes(), "output must match branch shape");
  TORCH_CHECK(new_stream.sizes() == branch.sizes(), "new_stream must match branch shape");
  TORCH_CHECK(post_weight.numel() == 0 || post_weight.numel() == hidden_size, "post_weight must be empty or match H");
  TORCH_CHECK(pre_weight.numel() == hidden_size, "pre_weight must match H");
  TORCH_CHECK(alpha.numel() == hidden_size, "alpha must match H");
  TORCH_CHECK(beta.numel() == hidden_size, "beta must match H");
  TORCH_CHECK(branch.is_contiguous(), "branch must be contiguous");
  TORCH_CHECK(stream.is_contiguous(), "stream must be contiguous");
  TORCH_CHECK(post_weight.is_contiguous(), "post_weight must be contiguous");
  TORCH_CHECK(pre_weight.is_contiguous(), "pre_weight must be contiguous");
  TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");
  TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(new_stream.is_contiguous(), "new_stream must be contiguous");


  float* output_fp32_ptr = nullptr;
  if (output_fp32.has_value()) {
    TORCH_CHECK(output_fp32->device() == branch.device(), "output_fp32 must be on the branch device");
    TORCH_CHECK(output_fp32->scalar_type() == at::kFloat, "output_fp32 must be float32");
    TORCH_CHECK(output_fp32->sizes() == branch.sizes(), "output_fp32 must match branch shape");
    TORCH_CHECK(output_fp32->is_contiguous(), "output_fp32 must be contiguous");
    output_fp32_ptr = output_fp32->data_ptr<float>();
  }

  const int64_t rows = branch.size(0);
  if (rows == 0) {
    return;
  }

  auto queue = at::xpu::getCurrentXPUStream().queue();
  TORCH_CHECK(
      queue.get_device().has(sycl::aspect::fp64), "Accurate fused residual normalization requires FP64 capability");
  NormArgs_xe35 args{
      branch.data_ptr<at::BFloat16>(),
      stream.data_ptr<float>(),
      post_weight.data_ptr<at::BFloat16>(),
      pre_weight.data_ptr<at::BFloat16>(),
      alpha.data_ptr<float>(),
      beta.data_ptr<float>(),
      output.data_ptr<at::BFloat16>(),
      new_stream.data_ptr<float>(),
      output_fp32_ptr,
      hidden_size,
      static_cast<float>(post_eps),
      static_cast<float>(pre_eps),
      post_weight.numel() != 0,
      post_norm_input_scale,
      round_pre_norm_input};



  int workgroup_size;
  if (hidden_size <= 512 * kVecSize_xe35) {
    if (rows <= 128 || (rows >= 768 && rows < 3584)) {
      workgroup_size = 512;
    } else if (rows < 768) {
      workgroup_size = 256;
    } else {
      workgroup_size = 1024;
    }
  } else if (hidden_size <= 1024 * kVecSize_xe35) {
    workgroup_size = rows > 128 && rows <= 256 ? 256 : 1024;
  } else if (hidden_size <= 3 * 512 * kVecSize_xe35) {
    workgroup_size = rows > 64 && rows <= 384 ? 512 : 1024;
  } else {
    workgroup_size = 1024;
  }
  if (workgroup_size == 256) {
    launch_for_group_xe35<256>(args, rows, queue);
  } else if (workgroup_size == 512) {
    launch_for_group_xe35<512>(args, rows, queue);
  } else {
    launch_for_group_xe35<1024>(args, rows, queue);
  }
}

}
