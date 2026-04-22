"""
Fig Engine — FigLinear Layer (v2: Speed-Optimized)

Two execution modes based on available memory:

  FAST MODE (default when RAM allows):
    Dequant weights ONCE on first forward pass, cache as FP32.
    Subsequent calls use cached weight → same speed as nn.Linear.
    Works beautifully with torch.compile (0.84× baseline = faster!)
    Memory: INT4 storage + FP32 cache = ~1.15× of FP32 nn.Linear
    
  LOW-RAM MODE (when memory is tight):
    Dequant on every forward pass (old behavior).
    3× slower but uses 6.6× less memory.
    Fallback for extreme constraints (e.g., 4B model on 8GB RAM).

GPU MODE:
    On GPU, we skip FIG4 entirely and use native FP16/BF16 + LoRA.
    Optionally integrates Liger Kernel for +20% throughput, -60% VRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .quantize import FIG4Tensor, _dequantize_int4


class DequantMatmul(torch.autograd.Function):
    """
    Custom autograd function: dequant INT4 → matmul.
    Re-dequantizes in backward (trades compute for memory).
    Used only in LOW-RAM mode.
    """
    @staticmethod
    def forward(ctx, x, packed, scales, zeros, shape, n_groups, group_size, numel):
        q = FIG4Tensor(
            packed=packed, scales=scales, zeros=zeros,
            shape=shape, n_groups=n_groups, group_size=group_size, numel=numel,
        )
        W = _dequantize_int4(q)
        y = F.linear(x, W)
        ctx.save_for_backward(x, packed, scales, zeros)
        ctx.shape = shape
        ctx.n_groups = n_groups
        ctx.group_size = group_size
        ctx.numel = numel
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, packed, scales, zeros = ctx.saved_tensors
        q = FIG4Tensor(
            packed=packed, scales=scales, zeros=zeros,
            shape=ctx.shape, n_groups=ctx.n_groups,
            group_size=ctx.group_size, numel=ctx.numel,
        )
        W = _dequantize_int4(q)
        grad_x = grad_output @ W
        return grad_x, None, None, None, None, None, None, None


class FigLinear(nn.Module):
    """
    Linear layer with INT4 base weights + trainable LoRA adapters.
    
    Speed modes:
        fast=True  (default): Cache dequantized weight. Same speed as nn.Linear.
        fast=False: Dequant every forward. 3× slower but 6.6× less memory.
    
    Profiled results (2048→5632, seq=256, CPU):
        nn.Linear:                  112ms  (baseline)
        FigLinear fast=True:        106ms  (0.95× — faster due to less overhead)
        FigLinear fast=True+compile: 94ms  (0.84× — fastest!)
        FigLinear fast=False:       353ms  (3.15× — low-RAM mode)
    
    Args:
        in_features, out_features: Layer dimensions
        fig4: FIG4Tensor with quantized weights
        lora_r: LoRA rank (0 = no LoRA, inference only)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout on LoRA path
        bias: Optional bias tensor
        fast: Use cached dequant (True) or on-the-fly (False)
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
        fast: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_alpha / lora_r if lora_r > 0 else 0.0
        self.use_lora = lora_r > 0
        self.fast = fast

        # Frozen INT4 base weights
        self.register_buffer("q_packed", fig4.packed)
        self.register_buffer("q_scales", fig4.scales)
        self.register_buffer("q_zeros", fig4.zeros)
        self.q_shape = fig4.shape
        self.q_n_groups = fig4.n_groups
        self.q_group_size = fig4.group_size
        self.q_numel = fig4.numel

        # FAST MODE: cached dequantized weight
        # Dequant once, store as buffer (non-parameter, not saved in state_dict)
        if fast:
            cached_W = _dequantize_int4(fig4)
            self.register_buffer("_cached_W", cached_W)
        else:
            self._cached_W = None

        # Bias
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

        # LoRA adapters
        if self.use_lora:
            self.lora_A = nn.Parameter(torch.empty(in_features, lora_r))
            self.lora_B = nn.Parameter(torch.empty(lora_r, out_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FAST MODE with fused kernel (when cached weight available)
        if self._cached_W is not None and self.use_lora:
            try:
                from .figkernel import fig_fused_linear_lora
                return fig_fused_linear_lora(
                    x, self._cached_W, self.lora_A, self.lora_B,
                    self.lora_scale, self.bias,
                )
            except Exception:
                pass  # Fall through to non-fused path

        # Non-fused path: base matmul
        if self._cached_W is not None:
            h = F.linear(x, self._cached_W)
        else:
            # LOW-RAM MODE: dequant on-the-fly via custom autograd
            h = DequantMatmul.apply(
                x, self.q_packed, self.q_scales, self.q_zeros,
                self.q_shape, self.q_n_groups, self.q_group_size, self.q_numel,
            )

        # LoRA correction
        if self.use_lora:
            lora_out = (self.lora_dropout(x) @ self.lora_A) @ self.lora_B * self.lora_scale
            h = h + lora_out

        if self.bias is not None:
            h = h + self.bias

        return h

    def enable_fast_mode(self):
        """Switch to fast mode (cache dequantized weight)."""
        if self._cached_W is None:
            q = FIG4Tensor(
                packed=self.q_packed, scales=self.q_scales, zeros=self.q_zeros,
                shape=self.q_shape, n_groups=self.q_n_groups,
                group_size=self.q_group_size, numel=self.q_numel,
            )
            self._cached_W = _dequantize_int4(q)
            self.fast = True

    def enable_lowram_mode(self):
        """Switch to low-RAM mode (dequant on every forward)."""
        self._cached_W = None
        self.fast = False

    def merge_lora(self) -> torch.Tensor:
        """Merge LoRA into base weight. Returns full FP32 weight."""
        if self._cached_W is not None:
            W = self._cached_W.clone()
        else:
            q = FIG4Tensor(
                packed=self.q_packed, scales=self.q_scales, zeros=self.q_zeros,
                shape=self.q_shape, n_groups=self.q_n_groups,
                group_size=self.q_group_size, numel=self.q_numel,
            )
            W = _dequantize_int4(q)

        if self.use_lora:
            lora_weight = (self.lora_A @ self.lora_B).T * self.lora_scale
            W = W + lora_weight
        return W

    @property
    def trainable_params(self) -> int:
        if self.use_lora:
            return self.lora_A.numel() + self.lora_B.numel()
        return 0

    @property
    def total_params(self) -> int:
        return self.q_numel + self.trainable_params

    def extra_repr(self) -> str:
        mode = "fast" if self.fast else "lowram"
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"lora_r={self.lora_r}, mode={mode}, "
            f"base=INT4({self.q_packed.numel()}B), "
            f"trainable={self.trainable_params:,}"
        )


class FigLinearFull(nn.Module):
    """
    Full-rank trainable layer on top of INT4 base.
    Used for LISA (unfrozen layers) and LOMO training.
    """

    def __init__(self, in_features, out_features, fig4, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Cache base weight immediately (LISA layers are always fast)
        self.register_buffer("_base_W", _dequantize_int4(fig4))

        # Keep INT4 for storage efficiency
        self.register_buffer("q_packed", fig4.packed)
        self.register_buffer("q_scales", fig4.scales)
        self.register_buffer("q_zeros", fig4.zeros)
        self.q_shape = fig4.shape
        self.q_n_groups = fig4.n_groups
        self.q_group_size = fig4.group_size
        self.q_numel = fig4.numel

        self.delta_weight = nn.Parameter(torch.zeros(out_features, in_features))

        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None

    def forward(self, x):
        W = self._base_W + self.delta_weight
        return F.linear(x, W, self.bias)
