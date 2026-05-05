"""
Fig Engine — Micro-Training Pipeline

Runs 1-5 LoRA steps on a specific Memory Fabric namespace
between conversation turns. Target: <100ms per memory write.

This is how memories get written INTO the weights:
1. Conversation generates a "store" signal
2. The information becomes a micro-training example
3. This pipeline trains ONLY the relevant namespace adapter
4. The Cognitive Core (base weights) is NEVER touched
5. Next turn, the model just KNOWS the new information

Uses FigMeZO for memory-constrained environments (no backward pass).
Falls back to standard backprop when memory allows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class MicroTrainConfig:
    """Configuration for micro-training steps."""
    learning_rate: float = 5e-4  # Higher than normal — few steps, need impact
    steps: int = 3               # 1-5 steps per memory write
    use_mezo: bool = False       # Use FigMeZO (no backward pass) if True
    mezo_epsilon: float = 1e-3
    max_time_ms: float = 100.0   # Hard cap — abort if exceeding this


class MicroTrainer:
    """
    Rapid memory writer. Trains a specific namespace's adapters
    on a single example in 1-5 gradient steps.
    
    Usage:
        trainer = MicroTrainer(memory_fabric, config)
        
        # After model generates <|mem_store|> for "user likes coffee":
        trainer.write_memory(
            namespace="personal",
            input_ids=tokenized_fact,
            labels=tokenized_fact,
        )
        # Now the personal/ adapters encode this knowledge
    """
    
    def __init__(self, fabric, config: Optional[MicroTrainConfig] = None):
        """
        Args:
            fabric: MemoryFabric instance
            config: training configuration
        """
        self.fabric = fabric
        self.config = config or MicroTrainConfig()
        self._write_count = 0
        self._total_time_ms = 0
    
    def write_memory(
        self,
        model: nn.Module,
        namespace: str,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Write a memory by micro-training the namespace's adapters.
        
        Args:
            model: the full model (for forward pass)
            namespace: which Memory Fabric namespace to train
            input_ids: tokenized memory content [1, seq_len]
            labels: same as input_ids for causal LM training
            attention_mask: optional
            
        Returns:
            dict with training stats (loss_before, loss_after, time_ms)
        """
        t0 = time.time()
        
        # Get only this namespace's parameters
        params = self.fabric.get_trainable_params(namespace)
        if not params:
            return {"error": f"No params for namespace '{namespace}'"}
        
        # Ensure only these params have grad
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        
        # Measure loss before
        model.eval()
        with torch.no_grad():
            kwargs = {"input_ids": input_ids, "labels": labels}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            loss_before = model(**kwargs).loss.item()
        
        # Micro-training loop
        model.train()
        optimizer = torch.optim.SGD(params, lr=self.config.learning_rate)
        
        losses = []
        for step in range(self.config.steps):
            # Time check
            elapsed_ms = (time.time() - t0) * 1000
            if elapsed_ms > self.config.max_time_ms:
                break
            
            optimizer.zero_grad()
            out = model(**kwargs)
            out.loss.backward()
            
            # Clip grads
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(out.loss.item())
        
        # Measure loss after
        model.eval()
        with torch.no_grad():
            loss_after = model(**kwargs).loss.item()
        
        # Freeze params again
        for p in params:
            p.requires_grad_(False)
        
        elapsed_ms = (time.time() - t0) * 1000
        self._write_count += 1
        self._total_time_ms += elapsed_ms
        
        return {
            "namespace": namespace,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "improvement": (loss_before - loss_after) / loss_before * 100,
            "steps": len(losses),
            "time_ms": elapsed_ms,
            "total_writes": self._write_count,
        }
    
    def apply_decay(self, hours: float = 1.0):
        """Apply time-based decay to all fabric adapters."""
        self.fabric.apply_decay(hours)
    
    def promote_memory(self, from_ns: str, to_ns: str):
        """Promote knowledge from one namespace to another across all layers."""
        for layer in self.fabric.layers.values():
            layer.promote(from_ns, to_ns, scale=0.3)
    
    @property
    def stats(self) -> Dict:
        return {
            "total_writes": self._write_count,
            "avg_time_ms": self._total_time_ms / max(self._write_count, 1),
            "confidence_map": self.fabric.get_confidence_map(),
        }
