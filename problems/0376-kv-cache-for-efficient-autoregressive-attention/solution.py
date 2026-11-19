import numpy as np

def softmax(x):
    m = x.max(-1, keepdims=True)
    exp_x = np.exp(x - m)
    return exp_x / exp_x.sum(-1, keepdims=True)

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    d_k = W_Q.shape[1]
    Q_cur = x_new @ W_Q # (dk,)
    K_cur = x_new @ W_K # (dk,)
    V_cur = x_new @ W_V # (dk,)
    K_cur, V_cur = K_cur[None, :], V_cur[None, :]
    if cache is None:
        K, V = K_cur, V_cur
    else:
        K_cache, V_cache = cache
        K = np.concatenate([K_cache, K_cur], axis=0)
        V = np.concatenate([V_cache, V_cur], axis=0) # s+1, dk
    updated_cache = (K, V)
    attn_scores = softmax(K @ Q_cur / d_k ** 0.5) # s+1,
    output = attn_scores @ V
    return (
        output, updated_cache
    )

