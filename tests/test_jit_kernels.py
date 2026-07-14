"""
Tests for SYCL JIT kernel accuracy against AOT kernels.

These tests validate that JIT-compiled kernels produce the same results as
their AOT counterparts or PyTorch reference implementations.
"""

import pytest
import torch

try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False

try:
    from sgl_kernel.jit import apply_rope_inplace as jit_rope
    from sgl_kernel.jit import fused_inplace_qknorm as jit_qknorm
    from sgl_kernel.jit import moe_fused_gate as jit_moe_fused_gate
    from sgl_kernel.jit import rmsnorm as jit_rmsnorm
    from sgl_kernel.jit import timestep_embedding as jit_timestep_embedding

    HAS_SGLANG_JIT = True
except ImportError:
    HAS_SGLANG_JIT = False

HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


# PyTorch reference implementations


def reference_qknorm(q, k, q_weight, k_weight, eps=1e-6):
    """PyTorch reference implementation of QKNorm."""
    q_rms = torch.sqrt(torch.mean(q**2, dim=-1, keepdim=True) + eps)
    k_rms = torch.sqrt(torch.mean(k**2, dim=-1, keepdim=True) + eps)
    q_out = (q / q_rms) * q_weight
    k_out = (k / k_rms) * k_weight
    return q_out, k_out


def reference_timestep_embedding(
    t: torch.Tensor,
    dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """PyTorch reference implementation of Timestep Embedding."""
    half_dim = dim // 2

    # Compute frequency schedule
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32, device=t.device))
        * torch.arange(0, half_dim, dtype=torch.float32, device=t.device)
        / (half_dim - downscale_freq_shift)
    )

    # Compute angles
    t_float = t.float().view(-1, 1)
    args = scale * t_float * freqs.view(1, -1)

    # Compute embeddings
    cos_emb = torch.cos(args)
    sin_emb = torch.sin(args)

    if flip_sin_to_cos:
        output = torch.cat([cos_emb, sin_emb], dim=-1)
    else:
        output = torch.cat([sin_emb, cos_emb], dim=-1)

    return output


@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not HAS_SGL_KERNEL, reason="Requires sgl_kernel for AOT comparison")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_rmsnorm_jit_vs_aot():
    """Test RMSNorm JIT accuracy vs AOT for hidden_size=4096, dtype=float16."""
    device = "xpu"
    dtype = torch.float16
    hidden_size = 4096
    batch_size = 32
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # AOT kernel: sgl_kernel.rmsnorm(input, weight, eps, out=None) -> Tensor
    y_aot = sgl_kernel.rmsnorm(x.clone(), weight, eps)

    # JIT kernel (in-place operation, modifies out parameter)
    y_jit = torch.empty_like(x)
    jit_rmsnorm(x.clone(), weight, out=y_jit, eps=eps)

    # Compare accuracy
    torch.testing.assert_close(y_jit, y_aot, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_qknorm_jit_vs_reference():
    """Test QKNorm JIT accuracy vs PyTorch reference for hidden_size=128, dtype=float16."""
    device = "xpu"
    dtype = torch.float16
    batch_size = 32
    seq_len = 64
    num_heads = 8
    head_dim = 128
    eps = 1e-6

    # Create input tensors (3D: batch_size, num_heads, head_dim)
    # Note: fused_inplace_qknorm expects 3D tensors
    q = torch.randn(
        batch_size * seq_len, num_heads, head_dim, dtype=dtype, device=device
    )
    k = torch.randn(
        batch_size * seq_len, num_heads, head_dim, dtype=dtype, device=device
    )
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)

    # PyTorch reference (on 4D reshaped)
    q_4d = q.view(batch_size, seq_len, num_heads, head_dim)
    k_4d = k.view(batch_size, seq_len, num_heads, head_dim)
    q_ref, k_ref = reference_qknorm(q_4d, k_4d, q_weight, k_weight, eps)
    q_ref = q_ref.reshape(-1, num_heads, head_dim)
    k_ref = k_ref.reshape(-1, num_heads, head_dim)

    # JIT kernel (in-place operation)
    q_jit = q.clone().contiguous()
    k_jit = k.clone().contiguous()
    jit_qknorm(q_jit, k_jit, q_weight, k_weight, eps, head_dim=head_dim)

    # Compare accuracy
    torch.testing.assert_close(q_jit, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_jit, k_ref, rtol=1e-2, atol=1e-2)


def reference_rope(q, k, cos_sin_cache, positions, is_neox, rope_dim):
    """PyTorch reference implementation of RoPE (both neox and GPT-J styles).

    cos_sin_cache: [max_pos, rope_dim] float32, first half cos, second half sin.
    q/k: [batch, num_heads, head_dim] - head_dim >= rope_dim.
    positions: [batch] int64.
    """
    half = rope_dim // 2
    cos = cos_sin_cache[positions, :half].float()  # [batch, half]
    sin = cos_sin_cache[positions, half:rope_dim].float()  # [batch, half]

    def apply(x):
        x_rot = x[..., :rope_dim].float()
        if is_neox:
            x1 = x_rot[..., :half]
            x2 = x_rot[..., half:]
            rotated = torch.cat([-x2, x1], dim=-1)
        else:
            x1 = x_rot[..., 0::2]
            x2 = x_rot[..., 1::2]
            rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        cos_b = cos[:, None, :]  # [batch, 1, half]
        sin_b = sin[:, None, :]
        if is_neox:
            out = x_rot * torch.cat([cos_b, cos_b], dim=-1) + rotated * torch.cat(
                [sin_b, sin_b], dim=-1
            )
        else:
            cos_full = torch.stack([cos_b, cos_b], dim=-1).flatten(-2)
            sin_full = torch.stack([sin_b, sin_b], dim=-1).flatten(-2)
            out = x_rot * cos_full + rotated * sin_full
        return torch.cat([out.to(x.dtype), x[..., rope_dim:]], dim=-1)

    return apply(q), apply(k)


@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_rope_jit_vs_reference(is_neox, dtype):
    """Test JIT RoPE accuracy vs PyTorch reference for both neox/GPT-J styles."""
    device = "xpu"
    batch_size = 16
    num_heads = 8
    head_dim = 128
    rope_dim = 128
    max_pos = 4096

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
    positions = torch.arange(batch_size, dtype=torch.int64, device=device)
    cos_sin_cache = torch.randn(max_pos, rope_dim, dtype=torch.float32, device=device)

    q_ref, k_ref = reference_rope(
        q.cpu().float(),
        k.cpu().float(),
        cos_sin_cache.cpu(),
        positions.cpu(),
        is_neox,
        rope_dim,
    )
    q_ref = q_ref.to(dtype=dtype, device=device)
    k_ref = k_ref.to(dtype=dtype, device=device)

    q_jit = q.clone()
    k_jit = k.clone()
    jit_rope(q_jit, k_jit, cos_sin_cache, positions, is_neox=is_neox, rope_dim=rope_dim)

    torch.testing.assert_close(q_jit, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_jit, k_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_timestep_embedding_jit_vs_reference():
    """Test Timestep Embedding JIT accuracy vs PyTorch reference for dim=256, dtype=float32."""
    device = "xpu"
    dtype = torch.float32
    batch_size = 128
    dim = 256

    timesteps = torch.randn(batch_size, dtype=dtype, device=device)

    # PyTorch reference with default parameters
    y_ref = reference_timestep_embedding(
        timesteps.clone(),
        dim,
        flip_sin_to_cos=False,
        downscale_freq_shift=0.0,
        scale=1.0,
        max_period=10000,
    )

    # JIT kernel with default parameters
    y_jit = jit_timestep_embedding(
        timesteps.clone(),
        dim,
        flip_sin_to_cos=False,
        downscale_freq_shift=0.0,
        scale=1.0,
        max_period=10000,
        dtype=dtype,
    )

    # Compare accuracy
    torch.testing.assert_close(y_jit, y_ref, rtol=1e-3, atol=1e-3)


def reference_moe_fused_gate(scores, bias, num_expert_group, topk_group, topk):
    """PyTorch reference for the hierarchical grouped-topk MoE gate.

    Mirrors biased_grouped_topk (renormalize=True, no fused shared experts) and
    serves as ground truth for the JIT kernel. The reference computes in fp32.

    Note: the AOT *dynamic* fallback kernel groups by topk_group instead of
    num_expert_group, so it disagrees with this reference (and with the JIT
    kernel) whenever topk_group != num_expert_group; the JIT kernel
    intentionally follows this reference.
    """
    s = scores.float().sigmoid()
    n, e = s.shape
    scores_for_choice = s + bias.float().unsqueeze(0)
    group_scores = (
        scores_for_choice.view(n, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(n, num_expert_group, e // num_expert_group)
        .reshape(n, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = s.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


@pytest.mark.parametrize(
    "num_experts,num_expert_group,topk_group,topk",
    [
        (256, 8, 4, 8),  # DeepSeek-V3  (AOT templated path)
        (128, 4, 2, 4),  # AOT templated path
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not HAS_SGL_KERNEL, reason="Requires AOT sgl_kernel")
@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_moe_fused_gate_jit_vs_aot(
    num_experts, num_expert_group, topk_group, topk, dtype
):
    """JIT vs AOT for configs AOT handles via its (correct) templated path.

    The native-dtype compute paths are bit-identical, so this is an exact match
    even in bf16. (The AOT dynamic fallback is not exercised here — it groups by
    topk_group rather than num_expert_group; see test below.)
    """
    device = "xpu"
    num_rows = 64
    routed_scaling_factor = 2.5

    torch.manual_seed(0)
    x = torch.randn(num_rows, num_experts, dtype=dtype, device=device)
    bias = torch.randn(num_experts, dtype=dtype, device=device)

    out_aot, idx_aot = sgl_kernel.moe_fused_gate(
        x.clone(),
        bias.clone(),
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=False,
    )
    out_jit, idx_jit = jit_moe_fused_gate(
        x.clone(),
        bias.clone(),
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=False,
    )

    assert torch.equal(
        idx_jit.sort(dim=-1).values, idx_aot.sort(dim=-1).values
    ), "JIT and AOT selected different experts"
    torch.testing.assert_close(
        out_jit.sort(dim=-1).values,
        out_aot.sort(dim=-1).values,
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize(
    "num_experts,num_expert_group,topk_group,topk",
    [
        (256, 8, 4, 8),  # DeepSeek-V3
        (128, 4, 2, 4),
        (64, 8, 4, 6),  # dynamic config, topk_group != num_expert_group
    ],
)
@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_moe_fused_gate_jit_vs_reference(
    num_experts, num_expert_group, topk_group, topk
):
    """JIT vs PyTorch grouped-topk ground truth (covers the dynamic-only config).

    Uses fp16. In bf16 the sigmoid scores carry only ~2-3 significant digits, so
    many group/expert scores collide to identical values; the kernel's
    shuffle-argmax tie-break and torch.topk's tie-break then legitimately pick
    different (equal-scoring) experts. bf16 correctness is covered instead by
    test_moe_fused_gate_jit_vs_aot, which is bit-exact against the AOT kernel.
    """
    device = "xpu"
    num_rows = 64
    dtype = torch.float16

    torch.manual_seed(0)
    x = torch.randn(num_rows, num_experts, dtype=dtype, device=device)
    bias = torch.randn(num_experts, dtype=dtype, device=device)

    out_ref, idx_ref = reference_moe_fused_gate(
        x, bias, num_expert_group, topk_group, topk
    )
    out_jit, idx_jit = jit_moe_fused_gate(
        x.clone(),
        bias.clone(),
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
    )

    assert torch.equal(
        idx_jit.sort(dim=-1).values, idx_ref.sort(dim=-1).values
    ), "JIT and reference selected different experts"
    torch.testing.assert_close(
        out_jit.sort(dim=-1).values,
        out_ref.sort(dim=-1).values,
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize(
    "seq_length",
    list(range(1, 10))
    + [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
)
@pytest.mark.parametrize(
    "params",
    [
        (128, 4, 2, 4),
        (256, 8, 4, 8),  # deepseek v3
        (512, 16, 8, 16),
    ],
)
# @pytest.mark.parametrize("num_fused_shared_experts", [0, 1, 2])
@pytest.mark.parametrize("num_fused_shared_experts", [0])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
@pytest.mark.skipif(not HAS_SGL_KERNEL, reason="Requires AOT sgl_kernel")
@pytest.mark.skipif(not HAS_SGLANG_JIT, reason="Requires SGLang for JIT compilation")
@pytest.mark.skipif(not HAS_XPU, reason="Requires XPU device")
def test_moe_fused_gate_jit_vs_aot_combined(
    seq_length, params, num_fused_shared_experts, apply_routed_scaling_factor_on_output
):
    """Sweep JIT vs AOT across the full AOT test matrix (fp32).

    fp32 avoids the bf16 tie-break ambiguity, and all three configs route
    through AOT's (correct) templated path, so JIT must match AOT exactly.
    """
    num_experts, num_expert_group, topk_group, topk = params
    dtype = torch.float32
    routed_scaling_factor = 2.5

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts), dtype=dtype, device="xpu")
    bias = torch.rand(num_experts, dtype=dtype, device="xpu")
    topk = topk + num_fused_shared_experts

    out_aot, idx_aot = sgl_kernel.moe_fused_gate(
        tensor.clone(),
        bias.clone(),
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
    out_jit, idx_jit = jit_moe_fused_gate(
        tensor.clone(),
        bias.clone(),
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    # When num_fused_shared_experts > 0, the last topk slots are shared experts;
    # only validate that both place them in [num_experts, num_experts + n).
    if num_fused_shared_experts > 0:
        valid_min = num_experts
        valid_max = num_experts + num_fused_shared_experts
        for tag, ids in (("JIT", idx_jit), ("AOT", idx_aot)):
            shared = ids[:, -num_fused_shared_experts:]
            assert torch.all(
                (shared >= valid_min) & (shared < valid_max)
            ), f"{tag} shared expert indices out of [{valid_min}, {valid_max})"
        idx_jit = idx_jit[:, :-num_fused_shared_experts]
        idx_aot = idx_aot[:, :-num_fused_shared_experts]
        out_jit = out_jit[:, :-num_fused_shared_experts]
        out_aot = out_aot[:, :-num_fused_shared_experts]

    idx_check = torch.allclose(
        idx_jit.sort(dim=-1)[0].to(torch.int32),
        idx_aot.sort(dim=-1)[0].to(torch.int32),
        rtol=1e-04,
        atol=1e-05,
    )
    output_check = torch.allclose(
        out_jit.sort(dim=-1)[0].to(torch.float32),
        out_aot.sort(dim=-1)[0].to(torch.float32),
        rtol=1e-02,
        atol=1e-03,
    )

    assert idx_check, (
        f"JIT/AOT indices mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}, num_fused_shared_experts {num_fused_shared_experts}"
    )
    assert output_check, (
        f"JIT/AOT output mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}, num_fused_shared_experts {num_fused_shared_experts}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
