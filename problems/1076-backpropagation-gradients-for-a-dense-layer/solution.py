import numpy as np

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.

    Args:
        a_prev: activations from previous layer, shape (K,)
        W: weight matrix, shape (J, K)
        b: bias vector, shape (J,)
        y: target vector, shape (J,)

    Returns:
        dict with keys 'dW' (J x K nested list), 'db' (list of length J),
        and 'da_prev' (list of length K).
    """
    # Your code here
    # dC/dW = dC/da * da/dz * dz/dW # J, K
    # dC/da_prev = dC/da * da/dz * dz/daprev
    # dC/db = dC/da * da/dz * dz/db
    
    # dC/da = 2(a-y) # J,
    # da/dz = a * (1 - a) # J,
    # dz/dW = a_prev

    z = W @ a_prev + b # J
    a = 1 / (1 + np.exp(-z)) # J
    C = np.sum((a - y) ** 2) # 1

    da = 2 * (a - y) # J
    dz = da * a * (1 - a) # J
    dW = dz[..., None] @ a_prev[..., None].T # j, k
    da_prev = W.T @ dz # K
    db = dz
    return {
        "dW": dW.tolist(),
        "db": db.tolist(),
        "da_prev": da_prev.tolist()
    }
    