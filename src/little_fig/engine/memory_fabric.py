"""
Fig Engine — Memory Fabric

The weight-space memory system. Multiple LoRA adapters per layer,
organized by namespace, with learned gating and confidence decay.

This is NOT external retrieval. Memory lives IN the model weights.
The model doesn't "look up" information — it KNOWS it.

Architecture:
    Each FigLinear layer gets N parallel adapter pairs (Aᵢ, Bᵢ),
    one per memory namespace. A learned gate per namespace controls
    how much each adapter contributes to the output:

    output = base(x) + Σᵢ gate_i(x) × (x @ Aᵢ) @ Bᵢ × scaleᵢ

    The gate is a simple linear projection of the input mean-pooled
    activation → sigmoid → per-namespace weight.

Memory namespaces:
    personal/  — facts about the user (rank 8)
    episodic/  — conversation history, events (rank 16)
    wiki/      — verified knowledge, permanent (rank 32)
    schedule/  — time-sensitive info (rank 4)
    contested/ — conflicting information (rank 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MemoryNamespace:
    """Configuration for a single memory namespace."""
    name: str
    rank: int
    decay_rate: float  # weight decay per hour of non-access
    min_magnitude: float = 0.01  # floor before considered "forgotten"


DEFAULT_NAMESPACES = [
    MemoryNamespace("personal", rank=8, decay_rate=0.001),
    MemoryNamespace("episodic", rank=16, decay_rate=0.01),
    MemoryNamespace("wiki", rank=32, decay_rate=0.0001),  # near-permanent
    MemoryNamespace("schedule", rank=4, decay_rate=0.05),
    MemoryNamespace("contested", rank=4, decay_rate=0.02),
]


class MemoryGate(nn.Module):
    """
    Learned gate that decides which memory namespaces to activate.
    
    Input: hidden state from the current forward pass
    Output: per-namespace activation weights in [0, 1]
    
    The gate learns WHEN each namespace is relevant based on
    the input content. Not keyword matching — learned routing.
    """
    
    def __init__(self, hidden_size: int, n_namespaces: int):
        super().__init__()
        # Small projection: hidden → n_namespaces scores
        self.proj = nn.Linear(hidden_size, n_namespaces, bias=True)
        # Initialize biases slightly negative so gates start mostly closed
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, -1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq, hidden] or [batch, hidden]
        Returns: [batch, n_namespaces] gate values in [0, 1]
        """
        if x.dim() == 3:
            # Mean-pool across sequence dimension
            pooled = x.mean(dim=1)  # [batch, hidden]
        else:
            pooled = x
        
        return torch.sigmoid(self.proj(pooled))  # [batch, n_namespaces]


class MultiAdapterLayer(nn.Module):
    """
    A single layer's Memory Fabric: N parallel LoRA adapters + gating.
    
    Replaces the single (lora_A, lora_B) with N pairs, each gated.
    The base weight computation is unchanged — this only adds the
    memory adapter contributions on top.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        namespaces: List[MemoryNamespace],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.namespaces = namespaces
        self.n_namespaces = len(namespaces)
        
        # Gate
        self.gate = MemoryGate(in_features, self.n_namespaces)
        
        # Per-namespace LoRA adapters
        self.adapters_A = nn.ParameterDict()
        self.adapters_B = nn.ParameterDict()
        self.scales = {}
        
        for ns in namespaces:
            A = nn.Parameter(torch.empty(in_features, ns.rank))
            B = nn.Parameter(torch.zeros(ns.rank, out_features))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            self.adapters_A[ns.name] = A
            self.adapters_B[ns.name] = B
            self.scales[ns.name] = 32.0 / ns.rank  # alpha/rank
        
        # Access tracking for decay
        self._last_accessed: Dict[str, float] = {ns.name: time.time() for ns in namespaces}
        self._access_counts: Dict[str, int] = {ns.name: 0 for ns in namespaces}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute memory fabric contribution for this layer.
        Returns the DELTA to add to the base layer output.
        
        x: [batch, seq, in_features]
        Returns: [batch, seq, out_features]
        """
        # Get gate values: [batch, n_namespaces]
        gates = self.gate(x)
        
        # Compute gated sum of adapter outputs
        dtype = x.dtype
        result = torch.zeros(x.shape[0], x.shape[1], self.out_features,
                           device=x.device, dtype=dtype)
        
        for i, ns in enumerate(self.namespaces):
            A = self.adapters_A[ns.name].to(dtype=dtype)
            B = self.adapters_B[ns.name].to(dtype=dtype)
            gate_i = gates[:, i].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
            
            # Only compute if gate is open (save compute for closed namespaces)
            if gate_i.max().item() > 0.01:
                adapter_out = (x @ A) @ B * self.scales[ns.name]  # [batch, seq, out]
                result = result + gate_i * adapter_out
                
                # Track access
                self._last_accessed[ns.name] = time.time()
                self._access_counts[ns.name] += 1
        
        return result
    
    def get_namespace_adapter(self, namespace: str) -> Tuple[nn.Parameter, nn.Parameter]:
        """Get (A, B) pair for a specific namespace."""
        return self.adapters_A[namespace], self.adapters_B[namespace]
    
    def get_namespace_magnitude(self, namespace: str) -> float:
        """Get adapter norm as confidence proxy."""
        A = self.adapters_A[namespace]
        B = self.adapters_B[namespace]
        return (A.norm() * B.norm()).item()
    
    def apply_decay(self, hours_elapsed: float = 1.0):
        """
        Apply Ebbinghaus-style decay to adapter weights.
        Namespaces not accessed recently lose magnitude.
        """
        now = time.time()
        with torch.no_grad():
            for ns in self.namespaces:
                hours_since_access = (now - self._last_accessed[ns.name]) / 3600.0
                if hours_since_access < hours_elapsed:
                    continue  # Recently accessed — no decay
                
                # Decay factor: exp(-decay_rate × hours)
                decay = math.exp(-ns.decay_rate * hours_since_access)
                self.adapters_A[ns.name].data.mul_(decay)
                self.adapters_B[ns.name].data.mul_(decay)
    
    def promote(self, from_ns: str, to_ns: str, scale: float = 0.5):
        """
        Promote knowledge from one namespace to another.
        Copies a fraction of the source adapter into the target.
        Used for: episodic → personal → wiki promotion.
        """
        with torch.no_grad():
            src_A = self.adapters_A[from_ns]
            src_B = self.adapters_B[from_ns]
            dst_A = self.adapters_A[to_ns]
            dst_B = self.adapters_B[to_ns]
            
            # Project source into target rank (truncate or pad)
            src_rank = src_A.shape[1]
            dst_rank = dst_A.shape[1]
            r = min(src_rank, dst_rank)
            
            dst_A.data[:, :r] += src_A.data[:, :r] * scale
            dst_B.data[:r, :] += src_B.data[:r, :] * scale


class MemoryFabric(nn.Module):
    """
    The complete Memory Fabric for a model.
    One MultiAdapterLayer per FigLinear layer in the model.
    
    Provides:
    - forward() contribution to add to base model output
    - train_namespace() for micro-training specific adapters
    - decay() for time-based forgetting
    - promote() for knowledge consolidation
    - conflict_detect() for opposing adapter signals
    """
    
    def __init__(
        self,
        layer_configs: List[Tuple[str, int, int]],  # [(name, in_f, out_f), ...]
        namespaces: Optional[List[MemoryNamespace]] = None,
    ):
        super().__init__()
        self.namespaces = namespaces or DEFAULT_NAMESPACES
        
        self.layers = nn.ModuleDict()
        for name, in_f, out_f in layer_configs:
            safe_name = name.replace(".", "_")
            self.layers[safe_name] = MultiAdapterLayer(in_f, out_f, self.namespaces)
        
        self._layer_name_map = {name: name.replace(".", "_") for name, _, _ in layer_configs}
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🧠 Memory Fabric: {len(self.layers)} layers, "
              f"{len(self.namespaces)} namespaces, "
              f"{total_params:,} params ({total_params*4/1e6:.1f} MB)")
    
    def get_layer(self, layer_name: str) -> Optional[MultiAdapterLayer]:
        """Get the multi-adapter layer for a given FigLinear layer name."""
        safe = self._layer_name_map.get(layer_name)
        if safe and safe in self.layers:
            return self.layers[safe]
        return None
    
    def apply_decay(self, hours: float = 1.0):
        """Apply decay across all layers."""
        for layer in self.layers.values():
            layer.apply_decay(hours)
    
    def get_confidence_map(self) -> Dict[str, Dict[str, float]]:
        """Get confidence (adapter magnitude) for each namespace across layers."""
        result = {}
        for ns in self.namespaces:
            mags = []
            for layer in self.layers.values():
                mags.append(layer.get_namespace_magnitude(ns.name))
            result[ns.name] = {
                "mean_magnitude": sum(mags) / len(mags) if mags else 0,
                "max_magnitude": max(mags) if mags else 0,
                "total_access": sum(layer._access_counts[ns.name] for layer in self.layers.values()),
            }
        return result
    
    def detect_conflicts(self, x: torch.Tensor, layer_name: str) -> Optional[Tuple[str, str]]:
        """
        Check if two namespaces produce opposing outputs for the same input.
        Returns (ns_a, ns_b) if conflict detected, None otherwise.
        """
        layer = self.get_layer(layer_name)
        if layer is None:
            return None
        
        outputs = {}
        dtype = x.dtype
        for ns in self.namespaces:
            A = layer.adapters_A[ns.name].to(dtype=dtype)
            B = layer.adapters_B[ns.name].to(dtype=dtype)
            mag = A.norm() * B.norm()
            if mag.item() > 0.1:  # Only check active namespaces
                out = (x @ A) @ B
                outputs[ns.name] = out
        
        # Check pairwise cosine similarity — opposing = conflict
        ns_names = list(outputs.keys())
        for i in range(len(ns_names)):
            for j in range(i+1, len(ns_names)):
                a = outputs[ns_names[i]].flatten()
                b = outputs[ns_names[j]].flatten()
                cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                if cos < -0.5:  # Strong opposition
                    return (ns_names[i], ns_names[j])
        
        return None
    
    def get_trainable_params(self, namespace: str) -> List[nn.Parameter]:
        """Get all trainable parameters for a specific namespace (for micro-training)."""
        params = []
        for layer in self.layers.values():
            params.append(layer.adapters_A[namespace])
            params.append(layer.adapters_B[namespace])
        # Also include gates (they learn routing)
        for layer in self.layers.values():
            params.extend(layer.gate.parameters())
        return params
