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
    B, L, D = x.shape
    x_flat = x.reshape(-1, D)
    logits = x_flat @ Wg # bl, e
    values, idx = logits.topk(top_k, dim=-1) # bl, k
    weights = values.softmax(dim=-1) # bl, k

    out = torch.zeros_like(x_flat) # bl, d
    for e in range(n_experts):
        tok, slot = (idx == e).nonzero(as_tuple=True)
        if tok.numel():
            out.index_add_(
                0, tok, 
                weights[tok, slot, None] * (x_flat[tok] @ We[e]) # bl, d
            )

    return out.reshape(B, L, D)

