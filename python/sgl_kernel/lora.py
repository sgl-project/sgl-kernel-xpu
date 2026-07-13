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
        seg_lens
    )

    return output