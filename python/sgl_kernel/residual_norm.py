import torch


def fused_post_norm_residual_pre_norm_xe35(
    branch: torch.Tensor,
    stream: torch.Tensor,
    post_weight: torch.Tensor,
    pre_weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    new_stream: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    post_norm_input_scale: bool = False,
    round_pre_norm_input: bool = False,
    output_fp32: torch.Tensor | None = None,
) -> None:
    """Xe35 BF16 post-norm and pre-norm with an FP32 residual stream."""
    torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm_xe35(
        branch, stream, post_weight, pre_weight, alpha, beta, output,
        new_stream, post_eps, pre_eps, post_norm_input_scale,
        round_pre_norm_input, output_fp32,
    )
