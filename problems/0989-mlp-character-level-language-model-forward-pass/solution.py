import numpy as np

def tanh(x):
    # m = x.max(-1, keepdims=True)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) 

def logsumexp(x):
    m = x.max(-1, keepdims=True)
    return m + np.log(
        np.sum(
            np.exp(x - m), axis=-1, keepdims=True
        )
    ).squeeze(-1)

def cross_entropy(logits, targets):
    x = logits[np.arange(logits.shape[0]), targets] # N, 1
    return (
        -x + logsumexp(logits) 
    ).mean()



def mlp_forward_loss(X, Y, C, W1, b1, W2, b2):
    """
    Forward pass of an MLP character-level language model.
    Returns mean cross-entropy loss as a float.
    """
    inputs = C[X] # N, BLK, D
    N, B, D = inputs.shape
    inputs = inputs.reshape(N, -1)
    hidden = tanh(inputs @ W1 + b1) # N, BD -> N, H
    logits = hidden @ W2 + b2 # N, H -> N, V
    return cross_entropy(logits, Y)