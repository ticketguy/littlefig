"""
FigMeZO — Error-Shaped Zeroth-Order Optimizer

Original research contribution of the Fig Engine project.

Core insight:
    Standard MeZO uses z ~ N(0, I) — isotropic perturbations that waste the
    single probe direction equally across all weight dimensions, regardless of
    where quantization has introduced the most error.

    FigMeZO shapes the perturbation by quantization error:
        z ~ N(0, diag(σ²))  where σ² ∝ q_scales

    High-error dimensions (large q_scales → large absmax in that group → more
    quantization noise) receive larger perturbations, generating more gradient
    signal in the exact directions where LoRA needs to compensate.

    For LoRA params:
        lora_A [in, r]  → shaped by column-wise error of parent weight
                          (lora_A row i handles input dim i, so it should be
                           sensitive to error in column i of W)
        lora_B [r, out] → shaped by row-wise error of parent weight
                          (lora_B column j produces output dim j, so it should
                           be sensitive to error in row j of W)

Zero extra memory cost: q_scales already live in the FigLinear buffer.

hyperparameter:
    shaping_strength ∈ [0, 1]
        0 → standard MeZO (isotropic, z ~ N(0, I))
        1 → full error shaping (z ~ N(0, diag(σ²_normalised)))
        Interpolates between the two.  Start with 0.7.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class FigMeZOConfig:
    """Configuration for FigMeZO optimizer."""
    learning_rate: float = 1e-5
    epsilon: float = 1e-3       # Perturbation scale
    weight_decay: float = 0.0
    seed: int = 42
    shaping_strength: float = 0.7   # 0 = isotropic MeZO, 1 = full error shaping


class FigMeZO:
    """
    Error-shaped zeroth-order optimizer for FigQuant-quantized models.

    Extends MeZO by using quantization error (q_scales) from FigLinear layers
    to concentrate the perturbation budget where it matters most.

    Usage:
        optimizer = FigMeZO(model, config)

        for batch in dataloader:
            loss = optimizer.step(
                forward_fn=lambda: model(**batch).loss
            )

    Shaping map is built once at construction and cached (free at step time).
    Falls back to standard MeZO for non-LoRA / non-FigLinear parameters.
    """

    def __init__(self, model: nn.Module, config: FigMeZOConfig):
        self.model = model
        self.config = config
        self._step_count = 0

        # Collect trainable parameters
        self.params = [p for p in model.parameters() if p.requires_grad]
        if not self.params:
            self.params = list(model.parameters())
            for p in self.params:
                p.requires_grad = False

        self.n_params = sum(p.numel() for p in self.params)

        # Build error-shaping scales (σ) for each trainable parameter.
        # For lora_A / lora_B: derived from parent FigLinear's q_scales.
        # For everything else: unit scale (→ standard MeZO).
        self._sigma: Dict[int, torch.Tensor] = {}   # id(param) → σ tensor, same shape as param
        self._build_shaping_map()

        print(
            f"🍐 FigMeZO: {self.n_params:,} params | "
            f"ε={config.epsilon} | lr={config.learning_rate} | "
            f"shaping={config.shaping_strength} | "
            f"shaped params: {sum(1 for v in self._sigma.values() if v is not None)}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Shaping map construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_shaping_map(self):
        """
        Walk all FigLinear modules and assign σ tensors to their LoRA params.

        σ_lora_A[i, :] = column_error[i]   (broadcast across rank dim)
        σ_lora_B[:, j] = row_error[j]       (broadcast across rank dim)

        where column_error / row_error come from q_scales reshaped to weight dims.
        """
        try:
            from .linear import FigLinear
        except ImportError:
            return  # Graceful degradation — no shaping

        for module in self.model.modules():
            if not isinstance(module, FigLinear):
                continue
            if not module.use_lora:
                continue

            lora_A: nn.Parameter = module.lora_A  # [in_features, r]
            lora_B: nn.Parameter = module.lora_B  # [r, out_features]
            q_scales: torch.Tensor = module.q_scales  # [n_groups]

            out_f = module.out_features
            in_f = module.in_features
            n_groups = module.q_n_groups
            group_size = module.q_group_size

            # Expand q_scales → per-element error proxy [out_features, in_features]
            # q_scales index = flat_element_index // group_size
            # weights are stored [out_features, in_features] (standard Linear layout)
            numel = out_f * in_f
            # Build per-element sigma by repeating each group scale group_size times
            # Clamp to actual numel (last group may be padded)
            per_elem = q_scales.repeat_interleave(group_size)[:numel]  # [numel]
            W_sigma = per_elem.reshape(out_f, in_f)  # [out, in]

            # Normalise so mean σ = 1 (preserves effective step size)
            W_sigma = W_sigma / W_sigma.mean().clamp(min=1e-10)

            # lora_A [in, r]:  sensitivity = column error = mean over output dim
            col_error = W_sigma.mean(dim=0)   # [in_features]
            # Broadcast to [in, r]
            sigma_A = col_error.unsqueeze(1).expand_as(lora_A).clone()
            sigma_A = sigma_A / sigma_A.mean().clamp(min=1e-10)

            # lora_B [r, out]: sensitivity = row error = mean over input dim
            row_error = W_sigma.mean(dim=1)   # [out_features]
            # Broadcast to [r, out]
            sigma_B = row_error.unsqueeze(0).expand_as(lora_B).clone()
            sigma_B = sigma_B / sigma_B.mean().clamp(min=1e-10)

            self._sigma[id(lora_A)] = sigma_A
            self._sigma[id(lora_B)] = sigma_B

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(self, forward_fn: Callable[[], torch.Tensor]) -> float:
        """
        One FigMeZO step.

        Returns average of (L+ + L-) / 2 as loss estimate.
        """
        self._step_count += 1
        seed = self.config.seed + self._step_count
        eps = self.config.epsilon
        lr = self.config.learning_rate
        wd = self.config.weight_decay

        self._perturb(seed, eps)
        loss_plus = forward_fn().item()

        self._perturb(seed, -2.0 * eps)
        loss_minus = forward_fn().item()

        self._perturb(seed, eps)   # restore

        grad_scale = (loss_plus - loss_minus) / (2.0 * eps)
        self._update(seed, lr, grad_scale, wd)

        return (loss_plus + loss_minus) / 2.0

    @property
    def step_count(self) -> int:
        return self._step_count

    # ──────────────────────────────────────────────────────────────────────────
    # Internal: shaped perturbation
    # ──────────────────────────────────────────────────────────────────────────

    def _shaped_noise(
        self,
        param: torch.Tensor,
        gen: torch.Generator,
    ) -> torch.Tensor:
        """
        Generate shaped noise for param.

        z = α * σ * z_iso + (1 - α) * z_iso
          = z_iso * (α * σ + (1 - α))

        where α = shaping_strength, σ = normalised q_scale-derived sigma.
        Reduces to isotropic z_iso when α = 0 or param has no sigma entry.
        """
        z_iso = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)

        sigma = self._sigma.get(id(param))
        if sigma is None or self.config.shaping_strength == 0.0:
            return z_iso

        alpha = self.config.shaping_strength
        sigma_dev = sigma.to(device=param.device, dtype=param.dtype)
        # Blend: z_iso * (1 + α * (σ - 1))  ≡  z_iso * (α*σ + (1-α))
        # This preserves E[z²] ≈ 1 for α=0 and scales by σ for α=1
        shaped = z_iso * (alpha * sigma_dev + (1.0 - alpha))
        return shaped

    @torch.no_grad()
    def _perturb(self, seed: int, scale: float):
        gen = torch.Generator()
        gen.manual_seed(seed)
        for param in self.params:
            z = self._shaped_noise(param, gen)
            param.data.add_(z, alpha=scale)

    @torch.no_grad()
    def _update(self, seed: int, lr: float, grad_scale: float, wd: float):
        gen = torch.Generator()
        gen.manual_seed(seed)
        for param in self.params:
            z = self._shaped_noise(param, gen)
            if wd > 0:
                param.data.mul_(1.0 - lr * wd)
            param.data.add_(z, alpha=-lr * grad_scale)
