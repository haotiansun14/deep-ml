import torch



def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                           block_size: int = 2) -> torch.Tensor:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (seq_len, d_model)
        K: Key matrix (seq_len, d_model)
        V: Value matrix (seq_len, d_model)
        block_size: Size of blocks for tiled computation
    
    Returns:
        Output matrix (seq_len, d_model)
    """
    # Your code here
    # l_cur = l_prev * exp(m_prev - m_cur) + exp(q @ k.T / d**0.5 - m_cur)   
    # out_cur = out_prev * exp(m_prev - m_cur) + sum(exp(q @ k.T / d**0.5 - m_cur) @ v)
    # out /= l
    seq_len, d = Q.shape
    output = torch.zeros_like(Q) # s, d
    num_blk = (seq_len + block_size - 1) // block_size
    
    for i in range(0, num_blk):
        q_start = i * block_size
        q_end = min((i + 1) * block_size, seq_len)
        Q_tile = Q[q_start: q_end] # b, d
        tile_size = q_end - q_start
        m_prev = torch.full((tile_size, 1), fill_value=-torch.inf)
        output_tile = torch.zeros_like(Q_tile) # b, d
        sum_exp = torch.zeros((tile_size, 1)) # b, 1
        for j in range(0, num_blk):
            kv_start = j * block_size
            kv_end = min((j + 1) * block_size, seq_len)
            K_tile = K[kv_start: kv_end] # b, d
            V_tile = V[kv_start: kv_end]
            # tile attn
            attn = Q_tile @ K_tile.T / d ** 0.5  # b, b
            m_cur = torch.maximum(m_prev, attn.amax(dim=-1, keepdims=True)) # b, 1
            attn = torch.exp(attn - m_cur) # b, b
            # recal
            alpha = torch.exp(m_prev - m_cur) # b, 1
            sum_exp = sum_exp * alpha + attn.sum(-1, keepdims=True) # b, 1
            output_tile = output_tile * alpha + attn @ V_tile # b, d
            m_prev = m_cur
        output[q_start: q_end] = output_tile / sum_exp
    return output


