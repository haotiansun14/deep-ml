import torch

def top_p_sampling(logits: list[float], p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a torch.Tensor
    """
    assert 0 < p <= 1
    logits = torch.as_tensor(logits, dtype=torch.float32)
    l = (logits - logits.max(-1, keepdims=True).values).exp()
    probs = l / l.sum(-1, keepdims=True)
    sorted_probs, idx = torch.sort(probs, dim=-1, descending=True)
    cdf = sorted_probs.cumsum(dim=-1)
    remove = (cdf - sorted_probs) >= p

    probs[idx[remove]] = 0.
    return probs / probs.sum(-1, keepdims=True)