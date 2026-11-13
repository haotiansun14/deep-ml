import torch
from torch import einsum

def moe(x: torch.Tensor, We: torch.Tensor, Wg: torch.Tensor, n_experts: int, top_k: int) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    logits = x @ Wg # b, l, e
    values, idx = logits.topk(top_k, dim=-1) # b, l, k
    weights = values.softmax(dim=-1) # b, l, k

    selected_We = We[idx] # b, l, k, d, d
    output = einsum("bld, blkdm, blk -> blm", x, selected_We, weights)
    return output

