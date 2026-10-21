#pragma once

#include <torch/all.h>

#pragma GCC visibility push(default)

namespace at::native::xpu {
void muse_spark_moe_routing_xe35(
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    at::Tensor& gating_output,
    const c10::optional<at::Tensor>& correction_bias,
    double router_output_multiplier,
    double route_scale,
    double division_eps,
    const c10::optional<at::Tensor>& local_expert_mapping,
    const c10::optional<at::Tensor>& num_valid_tokens);
void sgl_per_token_group_quant_fp4_xe35(
    at::Tensor input,
    at::Tensor output_q,
    at::Tensor output_s,
    int64_t group_size,
    double eps,
    bool fuse_silu_and_mul,
    const std::optional<at::Tensor>& scale_map);
void sgl_padded_gather_per_token_group_quant_fp4_xe35(
    at::Tensor input,
    at::Tensor row_map,
    at::Tensor output_q,
    at::Tensor output_s,
    int64_t group_size,
    double eps,
    const std::optional<at::Tensor>& scale_map);
void topk_sigmoid_xe35(
    at::Tensor& topk_weights,
    at::Tensor& topk_indices,
    at::Tensor& gating_output,
    bool renormalize,
    const c10::optional<at::Tensor>& correction_bias);
}
void mxfp4_moe_scale_to_mn_major_xe35(torch::Tensor& out, const torch::Tensor& in, const torch::Tensor& rows_per_expert);

#pragma GCC visibility pop
