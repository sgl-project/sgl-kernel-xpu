from typing import Optional

import torch


def embedding_lora_a_fwd(
    input_ids: torch.Tensor,
    weights: torch.Tensor,
    vocab_size: int,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    extra_embeddings: Optional[torch.Tensor] = None,
    seg_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""LoRA embedding forward pass.

    This kernel computes the forward pass for the LoRA embedding layer. Each token
    is processed independently, and the output is computed based on the LoRA adapter
    for the corresponding segment.

    Each token `tok = input_ids[i]` belongs to a segment identified by the `seg_indptr` tensor.
    Corresponding to each segment, there is a LoRA adapter defined by the `weight_indices` tensor.
    Once the LoRA adapter `l` is identified for a token, the output is computed as follows:

    - The rank of the adapter is determined by `lora_ranks[l]`.
    - The embedding is zero-padded to match the maximum rank.
    - If `extra_embeddings` is provided, tokens beyond the vocabulary size are mapped to the extra embeddings.

    Parameters
    ----------
    input_ids : torch.Tensor
        Input tensor containing token IDs, shape ``(num_tokens,)``.
    weights : torch.Tensor
        Weight tensor for embeddings, shape ``(num_loras, max_rank, vocab_size)``.
    vocab_size : int
        Vocabulary size.
    seg_indptr : torch.Tensor
        Segment index pointer tensor, shape ``(num_segments + 1,)``.
    weight_indices : torch.Tensor
        Weight indices tensor, shape ``(num_segments,)``.
    lora_ranks : torch.Tensor
        LoRA ranks tensor, shape ``(num_loras,)``.
    extra_embeddings : Optional[torch.Tensor], optional
        Optional extra embeddings tensor, shape ``(num_loras, num_extra_tokens, max_rank)``.
    seg_lens : Optional[torch.Tensor], optional
        Optional segment lengths tensor, shape ``(num_segments,)``.

    Returns
    -------
    output : torch.Tensor
        Output tensor containing the computed embeddings, shape ``(num_tokens, max_rank)``.

    Notes
    -----
    - The output tensor is created with zeros in torch and is allocated by the C++ kernel
    - The output tensor is zero-padded for ranks smaller than `max_rank`.
    - Tokens with IDs beyond the vocabulary size are mapped to `extra_embeddings` if provided else zero-padded.
    - Tokens with negative IDs are treated as invalid and their output embeddings are zeroed out.
    """
    # Create zero output tensor
    output = torch.zeros(
        (input_ids.size(0), weights.size(1)), dtype=weights.dtype, device=weights.device
    )
    # Call the kernel
    torch.ops.sgl_kernel.embedding_lora_a_fwd(
        output,
        input_ids,
        weights,
        vocab_size,
        seg_indptr,
        weight_indices,
        lora_ranks,
        extra_embeddings,
        seg_lens,
    )

    return output


def sgemm_lora_a_fwd(
    input_x: torch.Tensor,
    weights: torch.Tensor,
    stack_num: int,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    seg_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""LoRA A-matrix SGEMM forward pass.

    This kernel computes the forward pass through the LoRA ``A`` matrices as a
    grouped (segmented) matrix multiplication. The input tokens are partitioned
    into contiguous segments by ``seg_indptr``, and each segment is multiplied by
    the LoRA adapter selected via ``weight_indices``.

    For each segment ``s``, the token rows ``input_x[seg_indptr[s]:seg_indptr[s+1]]``
    are matrix-multiplied by the transpose of the adapter weights
    ``weights[weight_indices[s]]``:

    .. math::

        output[i] = input\_x[i] \; @ \; weights[l]^{T}

    where ``l = weight_indices[s]`` is the adapter for the segment containing token ``i``.

    - ``stack_num`` accounts for adapters that stack multiple projections (e.g. the
      fused ``qkv`` projection), so the output column dimension is ``stack_num * max_rank``.
    - For FP16/BF16 weights a single grouped GEMM is issued on the XMX DPAS path.
    - For FP32 weights there is no native XMX path, so a 3xTF32 emulation (three
      chained TF32 grouped GEMMs) is used to recover near-FP32 accuracy.

    Parameters
    ----------
    input_x : torch.Tensor
        Input activation tensor, shape ``(num_tokens, input_dim)``.
    weights : torch.Tensor
        LoRA A-matrix weight tensor, shape ``(num_loras, stack_num * max_rank, input_dim)``.
    stack_num : int
        Number of stacked projections packed along the weight rank dimension; must be > 0
        and must divide ``weights.size(1)``.
    seg_indptr : torch.Tensor
        Segment index pointer tensor, shape ``(num_segments + 1,)``. Must start at 0,
        end at ``num_tokens``, and be non-decreasing.
    weight_indices : torch.Tensor
        Per-segment adapter indices into ``weights``, shape ``(num_segments,)``. Values
        must be in ``[0, num_loras)``.
    lora_ranks : torch.Tensor
        LoRA ranks tensor, shape ``(num_loras,)``. Values must be in ``[0, max_rank]``.
        Note that ``lora_ranks`` is only range-validated: it does **not** shrink the
        per-segment GEMM, which always computes the full ``stack_num * max_rank`` output
        columns. The caller is therefore expected to pre-zero the weight rows beyond
        each adapter's rank ``R_l`` (rows ``j >= R_l`` within every stacked block) so
        that the extra output columns come out zero-padded.
    seg_lens : Optional[torch.Tensor], optional
        Optional segment lengths tensor, shape ``(num_segments,)``. Currently unused,
        reserved for future per-segment optimizations.

    Returns
    -------
    output : torch.Tensor
        Output tensor containing the LoRA A projection, shape ``(num_tokens, stack_num * max_rank)``.

    Notes
    -----
    - The output tensor is created with ``torch.empty`` and populated by the C++ kernel.
    - ``output`` and ``input_x`` must share the same dtype as ``weights``.
    - Supported weight dtypes are FP16, BF16 and FP32.
    """
    # Create empty output tensor
    output = torch.empty(
        (input_x.size(0), weights.size(1)), dtype=weights.dtype, device=weights.device
    )
    # Call the kernel
    torch.ops.sgl_kernel.sgemm_lora_a_fwd(
        output,
        input_x,
        weights,
        stack_num,
        seg_indptr,
        weight_indices,
        lora_ranks,
        seg_lens,
    )

    return output


def sgemm_lora_b_fwd(
    input_x: torch.Tensor,
    weights: torch.Tensor,
    seg_indptr: torch.Tensor,
    weight_indices: torch.Tensor,
    lora_ranks: torch.Tensor,
    scalings: torch.Tensor,
    seg_lens: Optional[torch.Tensor] = None,
    base_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""LoRA B-matrix SGEMM forward pass.

    This kernel computes the forward pass through the LoRA ``B`` matrices as a
    grouped (segmented) matrix multiplication, applying the per-adapter scaling
    and (optionally) adding a residual base-model output in a single fused pass.

    The input tokens are partitioned into contiguous segments by ``seg_indptr``,
    and each segment is multiplied by the transpose of the LoRA adapter selected
    via ``weight_indices``. For each segment ``s`` (adapter ``l = weight_indices[s]``):

    .. math::

        output[i] = scalings[l] \; \cdot \; \left( input\_x[i] \; @ \; weights[l]^{T} \right)
                    \; + \; base\_output[i]

    where the ``base_output`` term is only added when it is supplied.

    - Unlike LoRA-A, the epilogue scaling is **per segment**: each adapter carries
      its own ``scalings[l]`` (typically ``lora_alpha / rank``), broadcast over the
      segment's rows via the grouped-GEMM epilogue.
    - The reduction dimension ``K`` is ``max_rank`` (``input_x.size(1)``), so the
      caller is expected to have zero-padded the rank columns beyond each adapter's
      ``R_l`` in both ``input_x`` and ``weights``.

    Parameters
    ----------
    input_x : torch.Tensor
        Input activation tensor (the LoRA-A projection), shape ``(num_tokens, max_rank)``.
    weights : torch.Tensor
        LoRA B-matrix weight tensor, shape ``(num_loras, output_dim, max_rank)``.
    seg_indptr : torch.Tensor
        Segment index pointer tensor, shape ``(num_segments + 1,)``. Must start at 0,
        end at ``num_tokens``, and be non-decreasing.
    weight_indices : torch.Tensor
        Per-segment adapter indices into ``weights``, shape ``(num_segments,)``. Values
        must be in ``[0, num_loras)``.
    lora_ranks : torch.Tensor
        LoRA ranks tensor, shape ``(num_loras,)``. Values must be in ``[0, max_rank]``.
        Range-validated only; it does not shrink the per-segment GEMM.
    scalings : torch.Tensor
        Per-adapter scaling factors (e.g. ``lora_alpha / rank``), shape ``(num_loras,)``.
    seg_lens : Optional[torch.Tensor], optional
        Optional segment lengths tensor, shape ``(num_segments,)``. Currently unused,
        reserved for future per-segment optimizations.
    base_output : Optional[torch.Tensor], optional
        Optional base-model output added as a residual, shape ``(num_tokens, output_dim)``.
        When provided, the kernel computes ``scalings[l] * (input_x @ weights[l]^T) + base_output``
        in a single fused pass; otherwise only the scaled LoRA projection is returned.

    Returns
    -------
    output : torch.Tensor
        Output tensor, shape ``(num_tokens, output_dim)``.

    Notes
    -----
    - The output tensor is created with ``torch.empty`` and populated by the C++ kernel.
    - ``output`` and ``input_x`` must share the same dtype as ``weights``.
    - Supported weight dtypes are FP16 and BF16.
    """
    # Create empty output tensor
    output = torch.empty(
        (input_x.size(0), weights.size(1)), dtype=weights.dtype, device=weights.device
    )
    # Call the kernel
    torch.ops.sgl_kernel.sgemm_lora_b_fwd(
        output,
        input_x,
        weights,
        seg_indptr,
        weight_indices,
        lora_ranks,
        scalings,
        seg_lens,
        base_output,
    )

    return output
