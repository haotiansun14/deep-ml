import torch

def softmax(x):
    m = x.max(-1, keepdim=True).values
    return torch.exp(x - m) / torch.exp(x - m).sum(-1,keepdim=True)

def gated_attention(
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_g: torch.Tensor
) -> torch.Tensor:
    """
    Compute Gated Attention output.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_q: Query projection of shape (d_model, d_k)
        W_k: Key projection of shape (d_model, d_k)
        W_v: Value projection of shape (d_model, d_v)
        W_g: Gate projection of shape (d_model, d_v)
    
    Returns:
        Gated attention output of shape (seq_len, d_v), rounded to 4 decimal places
    
    Hint: First compute standard scaled dot-product attention, then apply
    a sigmoid gate to modulate the output.
    """
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    gate = torch.nn.functional.sigmoid(X @ W_g) # s, dv
    d_k = W_q.shape[-1]
    attn = softmax(Q @ K.T / d_k ** 0.5) @ V # s, dv
    return attn * gate
