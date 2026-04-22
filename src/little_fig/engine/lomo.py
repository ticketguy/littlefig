"""
Fig Engine — LOMO (Low-Memory Optimization)

Implementation based on arxiv 2306.09782 (OpenLMLab).

Key idea: Fuse gradient computation with parameter update during backprop.
    - During backward pass, as each gradient is computed:
        1. Immediately apply the SGD update: θ = θ - lr * grad
        2. Free the gradient (don't accumulate)
    - At any moment, only ONE parameter's gradient exists in memory
    - Memory for gradients: O(1) instead of O(N)

This enables full-parameter fine-tuning at LoRA-level memory cost.
No optimizer states (no momentum, no exp_avg_sq).

Limitation: Only SGD-level optimization (no Adam).
For Adam-level quality at low memory, see AdaLomo (arxiv 2310.10195)
which adds grouped normalization for adaptive learning rates.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, List
from dataclasses import dataclass


@dataclass
class LOMOConfig:
    """Configuration for LOMO optimizer."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    clip_grad_norm: Optional[float] = 1.0
    clip_grad_value: Optional[float] = None


class LOMOOptimizer:
    """
    LOMO: Fused backward + parameter update with O(1) gradient memory.
    
    Usage:
        optimizer = LOMOOptimizer(model, config)
        
        loss = model(**batch).loss
        optimizer.fused_backward(loss)
        # Parameters are already updated — no separate .step() needed
    
    How it works:
        Registers backward hooks on each parameter. When a gradient is
        computed during loss.backward(), the hook immediately applies
        the SGD update and zeros the gradient. This means:
        - Only 1 gradient tensor exists at any time (O(1) memory)
        - No separate optimizer.step() needed
        - No optimizer states (pure SGD)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: LOMOConfig,
    ):
        self.model = model
        self.config = config
        self._hooks = []
        self._step_count = 0
        
        # Collect trainable parameters
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.n_params = sum(p.numel() for p in self.params)
        
        # For gradient clipping, we need a two-pass approach:
        # Pass 1: compute gradient norm (hooks accumulate squared norms)
        # Pass 2: apply clipped update (hooks apply update)
        self._grad_norm_sq = 0.0
        self._is_norm_pass = False
        
        self._register_hooks()
        
        print(f"🍐 LOMO: {self.n_params:,} trainable parameters, "
              f"lr={config.learning_rate}, O(1) gradient memory")
    
    def _register_hooks(self):
        """Register backward hooks for fused gradient + update."""
        for param in self.params:
            hook = param.register_post_accumulate_grad_hook(self._grad_hook)
            self._hooks.append(hook)
    
    def _grad_hook(self, param: torch.Tensor):
        """
        Called after gradient is computed for each parameter.
        Either accumulates norm (pass 1) or applies update (pass 2).
        """
        if param.grad is None:
            return
        
        if self._is_norm_pass:
            # Pass 1: accumulate gradient norm
            self._grad_norm_sq += param.grad.data.norm(2).item() ** 2
            param.grad = None  # free immediately
        else:
            # Pass 2: apply fused SGD update
            grad = param.grad.data
            
            # Clip gradient value
            if self.config.clip_grad_value is not None:
                grad.clamp_(
                    -self.config.clip_grad_value,
                    self.config.clip_grad_value
                )
            
            # Clip gradient norm (using pre-computed norm)
            if self.config.clip_grad_norm is not None and self._grad_norm > 0:
                clip_coef = self.config.clip_grad_norm / (self._grad_norm + 1e-6)
                if clip_coef < 1.0:
                    grad.mul_(clip_coef)
            
            # Weight decay (decoupled, like AdamW)
            if self.config.weight_decay > 0:
                param.data.mul_(1.0 - self.config.learning_rate * self.config.weight_decay)
            
            # SGD update
            param.data.add_(grad, alpha=-self.config.learning_rate)
            
            # Free gradient immediately (the key memory savings)
            param.grad = None
    
    def fused_backward(self, loss: torch.Tensor):
        """
        Run backward pass with fused parameter updates.
        
        If gradient clipping is enabled, runs two backward passes:
        1. First pass: compute gradient norm
        2. Second pass: apply clipped updates
        
        If no clipping, runs one backward pass with immediate updates.
        """
        self._step_count += 1
        
        if self.config.clip_grad_norm is not None:
            # Two-pass approach for gradient clipping
            # Pass 1: Compute gradient norm
            self._is_norm_pass = True
            self._grad_norm_sq = 0.0
            loss.backward(retain_graph=True)
            self._grad_norm = self._grad_norm_sq ** 0.5
            
            # Pass 2: Apply clipped updates
            self._is_norm_pass = False
            self.model.zero_grad()
            loss.backward()
        else:
            # Single pass: no clipping needed
            self._grad_norm = 0.0
            self._is_norm_pass = False
            loss.backward()
    
    def fused_backward_no_clip(self, loss: torch.Tensor):
        """
        Single-pass backward with fused updates (no gradient clipping).
        Faster than fused_backward when clipping is not needed.
        """
        self._step_count += 1
        self._grad_norm = 0.0
        self._is_norm_pass = False
        loss.backward()
    
    @property
    def step_count(self) -> int:
        return self._step_count
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
