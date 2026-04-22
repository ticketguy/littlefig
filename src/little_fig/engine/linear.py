"""
Fig Engine — FigLinear Layer

A linear layer with INT4 quantized base weights and trainable LoRA adapters.
Base weights are stored in FIG4 format and dequantized on-the-fly during
forward pass. Only LoRA parameters consume gradient/optimizer memory.

The dequantization is designed to work with torch.compile's inductor backend,
which fuses the dequant + matmul into vectorized C++ code on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .quantize import FIG4Tensor, _dequantize_int4


class DequantMatmul(torch.autograd.Function):
    """
    Custom autograd function that dequantizes INT4 weights and performs matmul.
    
    Forward:  y = x @ dequant(W_int4).T
    Backward: dx = dy @ dequant(W_int4), no gradient for frozen INT4 weights
    
    This avoids storing the full FP32 weight tensor — we re-dequantize in
    backward. Trades compute for memory.
    """
    
    @staticmethod
    def forward(ctx, x, packed, scales, zeros, shape, n_groups, group_size, numel):
        # Build FIG4Tensor for dequant
        q = FIG4Tensor(
            packed=packed, scales=scales, zeros=zeros,
            shape=shape, n_groups=n_groups, group_size=group_size, numel=numel,
        )
        W = _dequantize_int4(q)
        
        # F.linear expects weight shape (out_features, in_features)
        y = F.linear(x, W)
        
        # Save quantized data for backward (NOT the full FP32 weight)
        ctx.save_for_backward(x, packed, scales, zeros)
        ctx.shape = shape
        ctx.n_groups = n_groups
        ctx.group_size = group_size
        ctx.numel = numel
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, packed, scales, zeros = ctx.saved_tensors
        
        # Re-dequantize for backward (memory-compute tradeoff)
        q = FIG4Tensor(
            packed=packed, scales=scales, zeros=zeros,
            shape=ctx.shape, n_groups=ctx.n_groups,
            group_size=ctx.group_size, numel=ctx.numel,
        )
        W = _dequantize_int4(q)
        
        # Gradient for input: dx = dy @ W
        grad_x = grad_output @ W
        
        # No gradient for quantized weights (frozen)
        return grad_x, None, None, None, None, None, None, None


class FigLinear(nn.Module):
    """
    Linear layer with INT4 frozen base weights + trainable LoRA adapters.
    
    Architecture:
        y = DequantMatmul(x, W_int4) + (x @ A) @ B * scale
        
    Where:
        W_int4: frozen INT4 weights (dequantized on-the-fly)
        A: LoRA down-projection (in_features × lora_r), trainable
        B: LoRA up-projection (lora_r × out_features), trainable
        scale: lora_alpha / lora_r
    
    Memory per layer (example: 2048 × 5632):
        INT4 weights: 2048 * 5632 / 2 = 5.8 MB (packed)
        LoRA A (r=16): 2048 * 16 * 4 = 128 KB
        LoRA B (r=16): 16 * 5632 * 4 = 352 KB
        Total: ~6.3 MB vs 46.1 MB for FP32
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        fig4: FIG4Tensor with quantized weights
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA scaling factor (default: 32)
        lora_dropout: Dropout on LoRA path (default: 0.0 for speed)
        bias: Optional bias tensor (FP32, not quantized)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        fig4: FIG4Tensor,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_alpha / lora_r
        
        # Frozen INT4 base weights (registered as buffers — not parameters)
        self.register_buffer("q_packed", fig4.packed)
        self.register_buffer("q_scales", fig4.scales)
        self.register_buffer("q_zeros", fig4.zeros)
        self.q_shape = fig4.shape
        self.q_n_groups = fig4.n_groups
        self.q_group_size = fig4.group_size
        self.q_numel = fig4.numel
        
        # Bias (optional, FP32)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None
        
        # LoRA adapters (trainable)
        self.lora_A = nn.Parameter(torch.empty(in_features, lora_r))
        self.lora_B = nn.Parameter(torch.empty(lora_r, out_features))
        
        # Initialize LoRA: A with Kaiming, B with zeros (output starts at 0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Dropout
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base: dequant + matmul (custom autograd saves memory)
        h = DequantMatmul.apply(
            x, self.q_packed, self.q_scales, self.q_zeros,
            self.q_shape, self.q_n_groups, self.q_group_size, self.q_numel,
        )
        
        # LoRA correction
        lora_out = (self.lora_dropout(x) @ self.lora_A) @ self.lora_B * self.lora_scale
        h = h + lora_out
        
        # Bias
        if self.bias is not None:
            h = h + self.bias
        
        return h
    
    def merge_lora(self) -> torch.Tensor:
        """
        Merge LoRA adapters into the dequantized base weight.
        Returns full FP32 weight tensor (for export/inference).
        """
        q = FIG4Tensor(
            packed=self.q_packed, scales=self.q_scales, zeros=self.q_zeros,
            shape=self.q_shape, n_groups=self.q_n_groups,
            group_size=self.q_group_size, numel=self.q_numel,
        )
        W = _dequantize_int4(q)
        
        # LoRA merge: W_merged = W + (B @ A).T * scale
        # W shape: (out_features, in_features)
        # A shape: (in_features, r), B shape: (r, out_features)
        lora_weight = (self.lora_A @ self.lora_B).T * self.lora_scale
        
        return W + lora_weight
    
    @property
    def trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()
    
    @property
    def total_params(self) -> int:
        return self.q_numel + self.trainable_params
    
    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"lora_r={self.lora_r}, lora_alpha={self.lora_alpha}, "
            f"base=INT4({self.q_packed.numel()} bytes), "
            f"trainable={self.trainable_params:,}"
        )


class FigLinearFull(nn.Module):
    """
    Linear layer with INT4 base weights + FULL trainable FP32 overlay.
    Used for LISA (unfrozen layers) and LOMO training where selected layers
    are trained at full rank.
    
    Architecture:
        y = (dequant(W_int4) + delta_W) @ x
        
    Where delta_W is a full-rank FP32 update matrix initialized to zeros.
    This is more expressive than LoRA but uses more memory per active layer.
    Only used for the small number of layers LISA selects.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        fig4: FIG4Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Frozen INT4 base weights
        self.register_buffer("q_packed", fig4.packed)
        self.register_buffer("q_scales", fig4.scales)
        self.register_buffer("q_zeros", fig4.zeros)
        self.q_shape = fig4.shape
        self.q_n_groups = fig4.n_groups
        self.q_group_size = fig4.group_size
        self.q_numel = fig4.numel
        
        # Full-rank trainable delta (initialized to zero)
        self.delta_weight = nn.Parameter(torch.zeros(out_features, in_features))
        
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = FIG4Tensor(
            packed=self.q_packed, scales=self.q_scales, zeros=self.q_zeros,
            shape=self.q_shape, n_groups=self.q_n_groups,
            group_size=self.q_group_size, numel=self.q_numel,
        )
        W = _dequantize_int4(q)
        W_effective = W + self.delta_weight
        
        h = F.linear(x, W_effective, self.bias)
        return h
