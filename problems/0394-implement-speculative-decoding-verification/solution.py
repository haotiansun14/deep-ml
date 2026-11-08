import numpy as np

def speculative_decode_verify(draft_tokens: list, draft_probs: list, target_probs: list, coin_flips: list, resample_coin: float) -> list:
    """
    Verify draft tokens using speculative decoding.
    
    Args:
        draft_tokens: List of K drafted token indices
        draft_probs: K x V array, draft model distributions at each position
        target_probs: K x V array, target model distributions at each position
        coin_flips: K random values in [0,1) for acceptance decisions
        resample_coin: Random value in [0,1) for resampling on rejection
    
    Returns:
        List of accepted/resampled token indices
    """
    draft_tokens = np.array(draft_tokens)
    draft_probs = np.array(draft_probs)
    target_probs = np.array(target_probs)
    coin_flips = np.array(coin_flips)
    K = draft_tokens.shape[0]
    p = draft_probs[np.arange(K), draft_tokens]
    q = target_probs[np.arange(K), draft_tokens]

    rejection = coin_flips >= np.minimum(1, q / p)
    if not rejection.any():
        return draft_tokens.tolist()

    reject_from_idx = np.argmax(rejection)
    
    residual_probs = np.maximum(0, target_probs[reject_from_idx] - draft_probs[reject_from_idx]) 
    residual_probs /= np.sum(residual_probs, axis=-1, keepdims= True)

    resample_cdf = np.cumsum(residual_probs)
    resampled = np.searchsorted(resample_cdf, resample_coin)
    resampled = min(resampled, len(residual_probs) - 1)
    draft_tokens[reject_from_idx] = resampled
    
    return draft_tokens[:reject_from_idx+1].tolist()
    