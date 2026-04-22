"""
Fig Engine — Training Tier Selection

Automatically selects the best training method based on available RAM,
model size, and user preference. Four tiers from most to least memory:

    Tier 1: Streaming LoRA — INT4 base + LoRA adapters (lowest memory)
    Tier 2: LISA — Layerwise Importance Sampled AdamW (better than LoRA)
    Tier 3: MeZO — Zeroth-order optimization (no backward pass)
    Tier 4: LOMO — Full-parameter with fused gradient update

References:
    LISA: arxiv 2403.17919
    MeZO: arxiv 2305.17333
    LOMO: arxiv 2306.09782
"""

import os
import enum
import psutil
from dataclasses import dataclass
from typing import Optional


class TrainingTier(enum.Enum):
    """Training tiers ordered by memory usage (low → high)."""
    STREAMING_LORA = "streaming_lora"
    LISA = "lisa"
    MEZO = "mezo"
    LOMO = "lomo"


@dataclass
class MemoryEstimate:
    """Estimated memory breakdown for a training configuration."""
    tier: TrainingTier
    model_params: int
    
    # Breakdown (bytes)
    base_weights: int      # INT4 quantized weights in RAM
    trainable_weights: int # FP32 trainable parameters
    gradients: int         # Gradient memory
    optimizer_states: int  # Optimizer state memory
    activations: int       # Estimated activation memory
    
    @property
    def total_bytes(self) -> int:
        return (self.base_weights + self.trainable_weights + 
                self.gradients + self.optimizer_states + self.activations)
    
    @property
    def total_mb(self) -> float:
        return self.total_bytes / 1024 / 1024
    
    @property
    def total_gb(self) -> float:
        return self.total_bytes / 1024 / 1024 / 1024
    
    def __repr__(self):
        return (
            f"MemoryEstimate({self.tier.value}: "
            f"total={self.total_mb:.0f}MB, "
            f"base={self.base_weights/1e6:.0f}MB, "
            f"train={self.trainable_weights/1e6:.0f}MB, "
            f"grad={self.gradients/1e6:.0f}MB, "
            f"optim={self.optimizer_states/1e6:.0f}MB, "
            f"act={self.activations/1e6:.0f}MB)"
        )


def estimate_memory(
    model_params: int,
    tier: TrainingTier,
    lora_r: int = 16,
    lisa_active_layers: int = 2,
    seq_length: int = 512,
    batch_size: int = 1,
    n_layers: int = 22,
    hidden_dim: int = 2048,
) -> MemoryEstimate:
    """
    Estimate memory requirements for a training configuration.
    
    Args:
        model_params: Total model parameters
        tier: Training tier
        lora_r: LoRA rank (for Tier 1)
        lisa_active_layers: Number of unfrozen layers (for Tier 2)
        seq_length: Sequence length
        batch_size: Batch size
        n_layers: Number of transformer layers
        hidden_dim: Hidden dimension
    """
    # INT4 base weights: ~0.5 bytes per param
    base_int4 = int(model_params * 0.55)  # packed + scales + zeros
    
    # Activation memory estimate (rough: batch * seq * hidden * n_active_layers * 4 bytes)
    act_per_layer = batch_size * seq_length * hidden_dim * 4  # one activation tensor
    
    if tier == TrainingTier.STREAMING_LORA:
        # LoRA: ~2 * r * hidden_dim * n_target_modules * n_layers * 4 bytes
        n_target = 7  # q, k, v, o, gate, up, down
        lora_params = 2 * lora_r * hidden_dim * n_target * n_layers
        trainable = lora_params * 4
        gradients = trainable  # same size as trainable params
        optimizer = trainable * 2  # AdamW: exp_avg + exp_avg_sq
        # Only need activations for 2 layers (gradient checkpointing)
        activations = act_per_layer * 2 * 4  # 2 layers, ~4 tensors each
        
        return MemoryEstimate(
            tier=tier, model_params=model_params,
            base_weights=base_int4, trainable_weights=trainable,
            gradients=gradients, optimizer_states=optimizer,
            activations=activations,
        )
    
    elif tier == TrainingTier.LISA:
        # LISA: embeddings + LM head always trained (full rank)
        # + lisa_active_layers randomly sampled layers (full rank)
        embed_params = hidden_dim * 50000  # rough vocab size
        active_layer_params = int(model_params / n_layers) * lisa_active_layers
        trainable_params = embed_params + active_layer_params
        trainable = trainable_params * 4
        gradients = trainable
        optimizer = trainable * 2  # AdamW
        activations = act_per_layer * (lisa_active_layers + 2) * 4
        
        return MemoryEstimate(
            tier=tier, model_params=model_params,
            base_weights=base_int4, trainable_weights=trainable,
            gradients=gradients, optimizer_states=optimizer,
            activations=activations,
        )
    
    elif tier == TrainingTier.MEZO:
        # MeZO: no backward pass, no gradients, no optimizer states
        # Just weights + perturbation buffer (1 param-sized tensor)
        perturbation = int(model_params * 0.55)  # same size as INT4 weights
        
        return MemoryEstimate(
            tier=tier, model_params=model_params,
            base_weights=base_int4, trainable_weights=0,
            gradients=0, optimizer_states=0,
            activations=act_per_layer * 2,  # just forward pass
        )
    
    elif tier == TrainingTier.LOMO:
        # LOMO: full-param but O(1) gradient memory
        # Need FP32 copy of weights for updates
        fp32_weights = model_params * 4
        # Only 1 gradient tensor at a time (fused update)
        max_param_size = hidden_dim * hidden_dim * 4 * 4  # largest single param
        activations = act_per_layer * 2 * 4  # with gradient checkpointing
        
        return MemoryEstimate(
            tier=tier, model_params=model_params,
            base_weights=base_int4, trainable_weights=fp32_weights,
            gradients=max_param_size, optimizer_states=0,
            activations=activations,
        )
    
    raise ValueError(f"Unknown tier: {tier}")


def get_available_ram_bytes() -> int:
    """Get available system RAM in bytes."""
    return psutil.virtual_memory().available


def select_tier(
    model_params: int,
    available_ram: Optional[int] = None,
    preferred_tier: Optional[TrainingTier] = None,
    lora_r: int = 16,
    seq_length: int = 512,
    n_layers: int = 22,
    hidden_dim: int = 2048,
) -> TrainingTier:
    """
    Automatically select the best training tier that fits in available RAM.
    
    Priority order (quality): LISA > LOMO > Streaming LoRA > MeZO
    Memory order (ascending): Streaming LoRA < MeZO < LISA < LOMO
    
    We pick the highest-quality tier that fits within 70% of available RAM
    (leaving 30% headroom for OS and other processes).
    
    Args:
        model_params: Total model parameters
        available_ram: Available RAM in bytes (auto-detected if None)
        preferred_tier: Force a specific tier (skips auto-selection)
        lora_r: LoRA rank
        seq_length: Sequence length
        n_layers: Number of transformer layers
        hidden_dim: Hidden dimension
    
    Returns:
        Best training tier
    """
    if preferred_tier is not None:
        return preferred_tier
    
    if available_ram is None:
        available_ram = get_available_ram_bytes()
    
    # Use 70% of available RAM (30% headroom)
    budget = int(available_ram * 0.7)
    
    # Try tiers in quality order (best first)
    quality_order = [
        TrainingTier.LISA,
        TrainingTier.LOMO,
        TrainingTier.STREAMING_LORA,
        TrainingTier.MEZO,
    ]
    
    estimates = {}
    for tier in quality_order:
        est = estimate_memory(
            model_params, tier, lora_r=lora_r,
            seq_length=seq_length, n_layers=n_layers, hidden_dim=hidden_dim,
        )
        estimates[tier] = est
        
        if est.total_bytes <= budget:
            return tier
    
    # Nothing fits — return MeZO (minimum memory) with a warning
    print(f"⚠ Warning: Even MeZO requires {estimates[TrainingTier.MEZO].total_mb:.0f} MB "
          f"but only {budget/1e6:.0f} MB available. Training may be slow or fail.")
    return TrainingTier.MEZO


def print_tier_comparison(
    model_params: int,
    n_layers: int = 22,
    hidden_dim: int = 2048,
    seq_length: int = 512,
):
    """Print a comparison table of all tiers for a given model size."""
    available = get_available_ram_bytes()
    budget = int(available * 0.7)
    
    print(f"\n🍐 Training Tier Comparison ({model_params/1e9:.1f}B params)")
    print(f"   Available RAM: {available/1e9:.1f} GB (budget: {budget/1e9:.1f} GB)")
    print(f"   {'Tier':<20s} {'Memory':>10s} {'Fits?':>8s} {'Quality':>10s} {'Speed':>10s}")
    print(f"   {'-'*60}")
    
    tiers_info = [
        (TrainingTier.STREAMING_LORA, "Good", "Fast"),
        (TrainingTier.LISA, "Better", "Moderate"),
        (TrainingTier.MEZO, "Acceptable", "Slow"),
        (TrainingTier.LOMO, "Best", "Fast"),
    ]
    
    for tier, quality, speed in tiers_info:
        est = estimate_memory(
            model_params, tier, n_layers=n_layers,
            hidden_dim=hidden_dim, seq_length=seq_length,
        )
        fits = "✓" if est.total_bytes <= budget else "✗"
        print(f"   {tier.value:<20s} {est.total_mb:>8.0f}MB {fits:>8s} {quality:>10s} {speed:>10s}")
