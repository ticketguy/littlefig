"""
Fig Engine — MeZO (Memory-efficient Zeroth-Order Optimizer)

Implementation based on arxiv 2305.17333 (Princeton NLP).

Key idea: Estimate gradients using only forward passes (no backward pass).
    1. Save random seed z
    2. Perturb parameters: θ+ = θ + ε * perturbation(z)
    3. Compute loss L+
    4. Perturb parameters: θ- = θ - ε * perturbation(z)  (using same z)
    5. Compute loss L-
    6. Estimated gradient: ĝ = (L+ - L-) / (2ε) * perturbation(z)
    7. Update: θ = θ - lr * ĝ

Memory: Only need the model weights + 1 scalar (the loss difference).
No gradients, no activations, no optimizer states.

Tradeoff: Needs 10-100× more steps than backprop, but each step uses
inference-level memory. On 7/11 benchmarks, achieves comparable results
to full fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class MeZOConfig:
    """Configuration for MeZO optimizer."""
    learning_rate: float = 1e-5
    epsilon: float = 1e-3       # Perturbation scale
    weight_decay: float = 0.0
    seed: int = 42


class MeZOOptimizer:
    """
    Zeroth-order optimizer that estimates gradients with 2 forward passes.
    
    Usage:
        optimizer = MeZOOptimizer(model, config)
        
        for batch in dataloader:
            loss = optimizer.step(
                forward_fn=lambda: model(**batch).loss
            )
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MeZOConfig,
    ):
        self.model = model
        self.config = config
        self._step_count = 0
        
        # Collect trainable parameters
        self.params = [p for p in model.parameters() if p.requires_grad]
        if not self.params:
            # If no params require grad (e.g., INT4 model), train all
            self.params = list(model.parameters())
            for p in self.params:
                p.requires_grad = False  # MeZO doesn't need autograd
        
        self.n_params = sum(p.numel() for p in self.params)
        print(f"🍐 MeZO: {self.n_params:,} parameters, ε={config.epsilon}, lr={config.learning_rate}")
    
    @torch.no_grad()
    def step(self, forward_fn: Callable[[], torch.Tensor]) -> float:
        """
        One MeZO step: perturb → forward → perturb back → forward → update.
        
        Args:
            forward_fn: Callable that returns scalar loss
            
        Returns:
            Estimated loss (average of L+ and L-)
        """
        self._step_count += 1
        seed = self.config.seed + self._step_count
        eps = self.config.epsilon
        lr = self.config.learning_rate
        wd = self.config.weight_decay
        
        # Step 1: Perturb θ → θ+ε*z
        self._perturb(seed, eps)
        
        # Step 2: Forward pass → L+
        loss_plus = forward_fn().item()
        
        # Step 3: Perturb θ+ε*z → θ-ε*z (subtract 2ε*z)
        self._perturb(seed, -2.0 * eps)
        
        # Step 4: Forward pass → L-
        loss_minus = forward_fn().item()
        
        # Step 5: Restore θ-ε*z → θ (add ε*z back)
        self._perturb(seed, eps)
        
        # Step 6: Compute projected gradient and update
        grad_scale = (loss_plus - loss_minus) / (2.0 * eps)
        
        # Step 7: SGD update using the same random direction
        self._update(seed, lr, grad_scale, wd)
        
        return (loss_plus + loss_minus) / 2.0
    
    @torch.no_grad()
    def _perturb(self, seed: int, scale: float):
        """
        Perturb all parameters by scale * N(0, 1) with deterministic seed.
        Using the same seed always generates the same perturbation direction.
        """
        gen = torch.Generator()
        gen.manual_seed(seed)
        
        for param in self.params:
            z = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
            param.data.add_(z, alpha=scale)
    
    @torch.no_grad()
    def _update(self, seed: int, lr: float, grad_scale: float, wd: float):
        """
        Apply the ZO-SGD update:
            θ = θ - lr * (grad_scale * z + wd * θ)
        """
        gen = torch.Generator()
        gen.manual_seed(seed)
        
        for param in self.params:
            z = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
            
            # Weight decay
            if wd > 0:
                param.data.mul_(1.0 - lr * wd)
            
            # ZO gradient step
            param.data.add_(z, alpha=-lr * grad_scale)
    
    @property
    def step_count(self) -> int:
        return self._step_count
