import torch

def qlora_forward(
    x: torch.Tensor,
    quantized_W: torch.Tensor,
    scale: float,
    zero_point: float,
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    QLoRA forward pass with quantized frozen weights using PyTorch.
    
    Args:
        x: Input tensor (batch_size x in_features)
        quantized_W: 4-bit quantized weights as integers (in_features x out_features)
        scale: Quantization scale factor
        zero_point: Quantization zero point
        A: LoRA matrix A (rank x out_features)
        B: LoRA matrix B (in_features x rank)
        alpha: LoRA scaling factor
        
    Returns:
        Output tensor (batch_size x out_features)
    """
    # dequantize
    W_deq = quantized_W * scale + zero_point
    r = A.shape[0]
    return (
        x @ W_deq + alpha / r * x @ B @ A
    )
    