"""
Fig Engine — INT4 Quantization Engine

Implements the FIG4 format: asymmetric per-group INT4 quantization with
mmap-friendly binary storage. Designed for CPU streaming training where
weights are loaded from disk and dequantized on-the-fly.

Format:
    - Group size 128 (sweet spot: accuracy vs overhead)
    - Asymmetric quantization (min/max per group)
    - 2 weights packed per uint8 byte → 7.1× compression
    - Binary files with JSON header for mmap access
"""

import torch
import torch.nn.functional as F
import os
import json
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict


@dataclass
class FIG4Tensor:
    """
    A quantized INT4 tensor with per-group scale and zero-point.
    
    Memory layout:
        packed: uint8 tensor, 2 values per byte (low nibble first)
        scales: float32 tensor, one per group
        zeros:  float32 tensor, one per group (min value)
        shape:  original tensor shape
    """
    packed: torch.Tensor     # uint8, [numel_padded / 2]
    scales: torch.Tensor     # float32, [n_groups]
    zeros: torch.Tensor      # float32, [n_groups]
    shape: tuple             # original shape
    n_groups: int
    group_size: int
    numel: int               # original number of elements (before padding)
    
    @property
    def nbytes_quantized(self) -> int:
        """Total bytes of quantized representation."""
        return (self.packed.numel() + 
                self.scales.numel() * 4 + 
                self.zeros.numel() * 4)
    
    @property
    def nbytes_original(self) -> int:
        """Bytes of original FP32 tensor."""
        return self.numel * 4
    
    @property
    def compression_ratio(self) -> float:
        return self.nbytes_original / max(self.nbytes_quantized, 1)
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize back to FP32."""
        return _dequantize_int4(self)
    
    def save(self, path: str):
        """Save to binary file for mmap access."""
        _save_fig4(self, path)
    
    @staticmethod
    def load(path: str) -> "FIG4Tensor":
        """Load from binary file."""
        return _load_fig4(path)


class FigQuantizer:
    """
    Quantizes PyTorch tensors and HuggingFace models to INT4 (FIG4 format).
    
    Usage:
        # Single tensor
        quantizer = FigQuantizer(group_size=128)
        q = quantizer.quantize(weight_tensor)
        w_fp32 = q.dequantize()
        
        # Full model to disk
        quantizer.quantize_model_to_disk(model, save_dir)
    """
    
    def __init__(self, group_size: int = 128):
        self.group_size = group_size
    
    def quantize(self, tensor: torch.Tensor) -> FIG4Tensor:
        """Quantize a tensor to INT4 with per-group asymmetric quantization."""
        return _quantize_int4(tensor, self.group_size)
    
    def quantize_model_to_disk(
        self,
        model_name_or_path: str,
        save_dir: str,
        target_modules: Optional[list] = None,
    ) -> Dict[str, str]:
        """
        Load a HuggingFace model, quantize all Linear layers to INT4,
        save each as a .fig4 file for mmap access.
        
        Returns dict mapping layer_name → file_path.
        """
        from transformers import AutoModelForCausalLM, AutoConfig
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Load model config first to get architecture info
        config = AutoConfig.from_pretrained(model_name_or_path)
        
        # Load model with low memory
        print(f"🍐 Loading model: {model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        
        layer_map = {}
        total_original = 0
        total_quantized = 0
        
        print(f"🍐 Quantizing to INT4 (group_size={self.group_size})...")
        
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            
            if target_modules and not any(t in name for t in target_modules):
                continue
            
            weight = module.weight.data
            total_original += weight.numel() * 4
            
            # Quantize
            q = self.quantize(weight)
            total_quantized += q.nbytes_quantized
            
            # Save
            safe_name = name.replace(".", "__")
            path = os.path.join(save_dir, f"{safe_name}.fig4")
            q.save(path)
            
            layer_map[name] = {
                "path": path,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": module.bias is not None,
            }
            
            # Also save bias if present
            if module.bias is not None:
                bias_path = os.path.join(save_dir, f"{safe_name}.bias.pt")
                torch.save(module.bias.data, bias_path)
                layer_map[name]["bias_path"] = bias_path
        
        # Save non-linear modules (embeddings, norms, LM head)
        non_linear = {}
        for name, param in model.named_parameters():
            # Skip parameters that belong to quantized linear layers
            is_quantized = any(
                name.startswith(ln) for ln in layer_map.keys()
            )
            if not is_quantized:
                param_path = os.path.join(save_dir, f"_param__{name.replace('.', '__')}.pt")
                torch.save(param.data, param_path)
                non_linear[name] = param_path
        
        # Save metadata
        metadata = {
            "model_name": model_name_or_path,
            "config_class": config.__class__.__name__,
            "group_size": self.group_size,
            "layer_map": layer_map,
            "non_linear": non_linear,
            "total_original_bytes": total_original,
            "total_quantized_bytes": total_quantized,
            "compression_ratio": total_original / max(total_quantized, 1),
        }
        
        meta_path = os.path.join(save_dir, "fig_model.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save the original config
        config.save_pretrained(save_dir)
        
        # Save tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            tokenizer.save_pretrained(save_dir)
        except Exception:
            pass
        
        ratio = total_original / max(total_quantized, 1)
        print(f"🍐 ✓ Quantized {len(layer_map)} layers")
        print(f"   Weights: {total_original/1e6:.1f} MB → {total_quantized/1e6:.1f} MB ({ratio:.1f}× compression)")
        print(f"   Saved to: {save_dir}")
        
        # Free the original model
        del model
        
        return metadata


# ═══════════════════════════════════════════════════════════════════════════════
# Core INT4 Operations
# ═══════════════════════════════════════════════════════════════════════════════

def _quantize_int4(tensor: torch.Tensor, group_size: int = 128) -> FIG4Tensor:
    """Quantize FP32 → INT4 (asymmetric, per-group)."""
    original_shape = tensor.shape
    numel = tensor.numel()
    flat = tensor.reshape(-1).clone()
    
    # Pad to multiple of group_size
    pad = (group_size - numel % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])
    
    grouped = flat.reshape(-1, group_size)
    n_groups = grouped.shape[0]
    
    # Per-group min/max
    mins = grouped.min(dim=1).values
    maxs = grouped.max(dim=1).values
    
    # Scale: (max - min) / 15  (4-bit range: 0-15)
    scales = (maxs - mins) / 15.0
    scales = scales.clamp(min=1e-10)
    
    # Quantize to 0-15
    quantized = ((grouped - mins.unsqueeze(1)) / scales.unsqueeze(1))
    quantized = quantized.round().clamp(0, 15).to(torch.uint8)
    
    # Pack 2 values per byte: low nibble = even indices, high nibble = odd indices
    flat_q = quantized.reshape(-1)
    packed = (flat_q[0::2] | (flat_q[1::2] << 4)).to(torch.uint8)
    
    return FIG4Tensor(
        packed=packed,
        scales=scales.to(torch.float32),
        zeros=mins.to(torch.float32),
        shape=original_shape,
        n_groups=n_groups,
        group_size=group_size,
        numel=numel,
    )


def _dequantize_int4(q: FIG4Tensor) -> torch.Tensor:
    """Dequantize INT4 → FP32."""
    # Unpack: low nibble and high nibble
    low = (q.packed & 0x0F).to(torch.float32)
    high = ((q.packed >> 4) & 0x0F).to(torch.float32)
    
    # Interleave: [low0, high0, low1, high1, ...]
    unpacked = torch.stack([low, high], dim=1).reshape(-1)
    
    # Trim to grouped size and reshape
    total = q.n_groups * q.group_size
    unpacked = unpacked[:total].reshape(-1, q.group_size)
    
    # Dequantize: val = quantized * scale + zero
    result = unpacked * q.scales.unsqueeze(1) + q.zeros.unsqueeze(1)
    
    # Flatten and trim to original size, reshape
    return result.reshape(-1)[:q.numel].reshape(q.shape)


# ═══════════════════════════════════════════════════════════════════════════════
# File I/O (mmap-friendly binary format)
# ═══════════════════════════════════════════════════════════════════════════════

_FIG4_MAGIC = b"FIG4"
_FIG4_VERSION = 1

def _save_fig4(q: FIG4Tensor, path: str):
    """Save FIG4Tensor to a binary file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    header = {
        "version": _FIG4_VERSION,
        "shape": list(q.shape),
        "n_groups": q.n_groups,
        "group_size": q.group_size,
        "numel": q.numel,
        "packed_size": q.packed.numel(),
        "scales_size": q.scales.numel(),
    }
    header_json = json.dumps(header).encode("utf-8")
    
    with open(path, "wb") as f:
        # Magic + version
        f.write(_FIG4_MAGIC)
        # Header length + header
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        # Data: packed, scales, zeros (contiguous binary)
        f.write(q.packed.numpy().tobytes())
        f.write(q.scales.numpy().tobytes())
        f.write(q.zeros.numpy().tobytes())


def _load_fig4(path: str) -> FIG4Tensor:
    """Load FIG4Tensor from binary file."""
    with open(path, "rb") as f:
        # Verify magic
        magic = f.read(4)
        if magic != _FIG4_MAGIC:
            raise ValueError(f"Not a FIG4 file: {path} (magic={magic})")
        
        # Read header
        header_len = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        
        # Read data
        packed_bytes = header["packed_size"]
        scales_count = header["scales_size"]
        
        packed = torch.frombuffer(
            bytearray(f.read(packed_bytes)), dtype=torch.uint8
        ).clone()
        
        scales = torch.frombuffer(
            bytearray(f.read(scales_count * 4)), dtype=torch.float32
        ).clone()
        
        zeros = torch.frombuffer(
            bytearray(f.read(scales_count * 4)), dtype=torch.float32
        ).clone()
    
    return FIG4Tensor(
        packed=packed,
        scales=scales,
        zeros=zeros,
        shape=tuple(header["shape"]),
        n_groups=header["n_groups"],
        group_size=header["group_size"],
        numel=header["numel"],
    )
