"""
Fig Engine — LISA (Layerwise Importance Sampled AdamW)

Implementation of the LISA training strategy from arxiv 2403.17919.

Key idea: Instead of LoRA (low-rank adapters on all layers), LISA:
    1. Always trains embedding layer + LM head at full rank
    2. Randomly samples γ middle layers to unfreeze every K steps
    3. Frozen layers have no gradients and no optimizer states
    
Results: 10-35% better than LoRA on instruction tuning benchmarks,
while using the same memory as LoRA.

On CPU with Fig Engine:
    - Unfrozen layers use FP32 delta_weight on top of INT4 base
    - When layers rotate, delta is merged back and new layer unfrozen
    - Memory = INT4 base + γ full-rank layers + embeddings/LM head
"""

import torch
import torch.nn as nn
import random
from typing import List, Optional, Set
from dataclasses import dataclass


@dataclass
class LISAConfig:
    """Configuration for LISA training."""
    active_layers: int = 2       # γ: number of middle layers to unfreeze
    switch_interval: int = 5     # K: switch layers every K steps
    always_train_embed: bool = True   # Always train embedding layer
    always_train_head: bool = True    # Always train LM head
    seed: Optional[int] = None   # Random seed for reproducibility


class LISAScheduler:
    """
    Manages which layers are active (unfrozen) during LISA training.
    
    Usage:
        scheduler = LISAScheduler(model, config)
        
        for step in range(total_steps):
            scheduler.step(step)  # May switch active layers
            # ... training step ...
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: LISAConfig,
    ):
        self.model = model
        self.config = config
        self.rng = random.Random(config.seed)
        
        # Discover layer structure
        self._layers = self._find_layers()
        self._embed_params = self._find_embed_params()
        self._head_params = self._find_head_params()
        self._middle_layer_indices = list(range(len(self._layers)))
        
        # Current state
        self._active_indices: Set[int] = set()
        self._current_step = -1
        
        # Initial freeze
        self._freeze_all()
        self._unfreeze_embed_head()
        self._sample_and_unfreeze()
        
        n_middle = len(self._middle_layer_indices)
        print(f"🍐 LISA: {n_middle} middle layers, "
              f"γ={config.active_layers} active, "
              f"switch every {config.switch_interval} steps")
    
    def _find_layers(self) -> List[nn.Module]:
        """Find transformer decoder layers in the model."""
        layers = []
        
        for name, module in self.model.named_modules():
            # Common patterns for decoder layers
            class_name = module.__class__.__name__.lower()
            if any(x in class_name for x in ["decoderlayer", "block", "transformerblock"]):
                layers.append(module)
        
        # Fallback: look for numbered sequential children
        if not layers:
            for name, module in self.model.named_modules():
                if hasattr(module, "layers") and isinstance(module.layers, nn.ModuleList):
                    layers = list(module.layers)
                    break
                if hasattr(module, "h") and isinstance(module.h, nn.ModuleList):
                    layers = list(module.h)
                    break
        
        return layers
    
    def _find_embed_params(self) -> List[nn.Parameter]:
        """Find embedding parameters."""
        params = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                for p in module.parameters():
                    params.append(p)
        return params
    
    def _find_head_params(self) -> List[nn.Parameter]:
        """Find LM head parameters."""
        params = []
        for name, module in self.model.named_modules():
            class_name = module.__class__.__name__.lower()
            if "lmhead" in class_name or "lm_head" in name:
                for p in module.parameters():
                    params.append(p)
        # Also check for final linear layer
        if not params:
            for name, param in self.model.named_parameters():
                if "lm_head" in name or name.endswith("weight") and "embed" not in name:
                    if len(param.shape) == 2 and param.shape[0] > 1000:  # likely vocab size
                        params.append(param)
                        break
        return params
    
    def _freeze_all(self):
        """Freeze all parameters in the model."""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _unfreeze_embed_head(self):
        """Unfreeze embedding and LM head if configured."""
        if self.config.always_train_embed:
            for p in self._embed_params:
                p.requires_grad = True
        
        if self.config.always_train_head:
            for p in self._head_params:
                p.requires_grad = True
    
    def _freeze_layers(self, indices: Set[int]):
        """Freeze specific middle layers."""
        for idx in indices:
            if idx < len(self._layers):
                for param in self._layers[idx].parameters():
                    param.requires_grad = False
    
    def _unfreeze_layers(self, indices: Set[int]):
        """Unfreeze specific middle layers."""
        for idx in indices:
            if idx < len(self._layers):
                for param in self._layers[idx].parameters():
                    param.requires_grad = True
    
    def _sample_and_unfreeze(self):
        """Sample new active layers and unfreeze them."""
        n_layers = len(self._middle_layer_indices)
        n_active = min(self.config.active_layers, n_layers)
        
        # Freeze currently active layers
        self._freeze_layers(self._active_indices)
        
        # Sample new active layers
        self._active_indices = set(
            self.rng.sample(self._middle_layer_indices, n_active)
        )
        
        # Unfreeze new active layers
        self._unfreeze_layers(self._active_indices)
    
    def step(self, global_step: int):
        """
        Called every training step. Switches active layers every K steps.
        
        Returns True if layers were switched.
        """
        if global_step == self._current_step:
            return False
        
        self._current_step = global_step
        
        if global_step > 0 and global_step % self.config.switch_interval == 0:
            self._sample_and_unfreeze()
            return True
        
        return False
    
    @property
    def active_layer_indices(self) -> Set[int]:
        """Currently active (unfrozen) layer indices."""
        return self._active_indices.copy()
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all currently trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def get_trainable_count(self) -> int:
        """Count of currently trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
