"""
Fig Engine — Streaming Model Loader (v3: FigQuant + FigKernel)

Loads a HuggingFace model with FigQuant adaptive codebook INT4 quantization
and optional FigKernel fused ops. Supports Ember memory token injection.

Two modes:
    1. From HuggingFace Hub: downloads, quantizes with FigQuant, loads
    2. From pre-quantized directory: loads directly (fast)

The model streams layers from INT4 files on disk. Only LoRA adapters and
active layers (for LISA) live in RAM as FP32.
"""

import torch
import torch.nn as nn
import os
import json
import gc
from typing import Optional, Dict, List, Tuple

from .figquant import figquant_quantize, FigQuantTensor
from .linear import FigLinear, FigLinearFull
from .tier import TrainingTier


# Standard target modules for different architectures
ARCH_TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "default": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def _detect_arch(model_name: str) -> str:
    """Detect model architecture from name."""
    name_lower = model_name.lower()
    for arch in ARCH_TARGET_MODULES:
        if arch in name_lower:
            return arch
    return "default"


def _get_cache_dir(model_name: str) -> str:
    """Get the cache directory for quantized model files."""
    cache_base = os.path.join(os.path.expanduser("~"), ".cache", "fig_engine")
    safe_name = model_name.replace("/", "__")
    return os.path.join(cache_base, safe_name)


def _swap_rmsnorm(model: nn.Module):
    """Replace all RMSNorm / LayerNorm modules with FigRMSNorm for fused ops."""
    try:
        from .figkernel import FigRMSNorm
    except ImportError:
        return  # FigKernel not available, skip

    count = 0
    for name, module in list(model.named_modules()):
        # Match common RMSNorm class names across architectures
        cls_name = module.__class__.__name__
        is_rmsnorm = "RMSNorm" in cls_name or "rmsnorm" in cls_name.lower()

        if is_rmsnorm and hasattr(module, "weight"):
            hidden_size = module.weight.shape[0]
            eps = getattr(module, "eps", getattr(module, "variance_epsilon", 1e-6))
            fig_norm = FigRMSNorm(hidden_size, eps=eps)
            fig_norm.weight.data.copy_(module.weight.data)

            # Replace in parent
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], fig_norm)
            count += 1

    if count > 0:
        print(f"   ✓ FigKernel: swapped {count} RMSNorm → FigRMSNorm (fused)")


class FigModel(nn.Module):
    """
    A HuggingFace causal LM with INT4 quantized base weights and trainable adapters.
    
    Usage:
        model = FigModel.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            lora_r=16,
            tier=TrainingTier.STREAMING_LORA,
        )
        
        # Forward pass works like any nn.Module
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
    
    The model internally:
        1. Stores base weights as INT4 on disk (mmap-friendly)
        2. Dequantizes on-the-fly during forward pass
        3. Only LoRA/LISA parameters are trainable FP32
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.model_name = ""
        self.lora_r = 16
        self.lora_alpha = 32
        self.tier = TrainingTier.STREAMING_LORA
        self._fig_layers = {}  # name → FigLinear mapping
        self._config = None
    
    @staticmethod
    def from_pretrained(
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        tier: TrainingTier = TrainingTier.STREAMING_LORA,
        target_modules: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        group_size: int = 128,
        compile_model: bool = False,
        fast: bool = True,
        use_liger: bool = False,
        ember_mode: bool = False,
        fuse_kernels: bool = True,
    ) -> "FigModel":
        """
        Load a model with FigQuant INT4 quantized base weights + LoRA adapters.
        
        Args:
            model_name: HuggingFace model name or path
            lora_r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: LoRA dropout rate
            tier: Training tier (affects which layers are trainable)
            target_modules: Which linear layers to quantize + add LoRA
            cache_dir: Directory for cached quantized weights
            group_size: FigQuant quantization group size
            compile_model: Whether to apply torch.compile
            fast: Use cached dequant (True=fast, False=low-RAM)
            use_liger: Apply Liger Kernel optimizations (GPU, +20% speed, -60% VRAM)
            ember_mode: Inject Ember memory tokens into tokenizer + resize embeddings
            fuse_kernels: Replace RMSNorm with FigRMSNorm for fused ops
        
        Returns:
            FigModel ready for training
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        fig = FigModel()
        fig.model_name = model_name
        fig.lora_r = lora_r
        fig.lora_alpha = lora_alpha
        fig.tier = tier
        
        # Detect architecture
        arch = _detect_arch(model_name)
        if target_modules is None:
            target_modules = ARCH_TARGET_MODULES.get(arch, ARCH_TARGET_MODULES["default"])
        
        mode_label = "fast" if fast else "low-RAM"
        print(f"🍐 Fig Engine: Loading {model_name}")
        print(f"   Architecture: {arch}")
        print(f"   Quantization: FigQuant (adaptive codebook INT4)")
        print(f"   Target modules: {target_modules}")
        print(f"   Training tier: {tier.value}")
        print(f"   LoRA rank: {lora_r}, alpha: {lora_alpha}")
        print(f"   Speed mode: {mode_label}")
        if ember_mode:
            print(f"   🔥 Ember mode: memory tokens will be injected")
        
        # Apply Liger Kernel if requested (must happen BEFORE model load)
        if use_liger:
            try:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM
                print(f"   ✓ Liger Kernel enabled (+20% speed, -60% VRAM)")
            except ImportError:
                print(f"   ⚠ liger-kernel not installed. pip install liger-kernel")
                use_liger = False
        
        # Load tokenizer
        fig.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if fig.tokenizer.pad_token is None:
            fig.tokenizer.pad_token = fig.tokenizer.eos_token
            fig.tokenizer.pad_token_id = fig.tokenizer.eos_token_id
        
        # Ember mode: inject memory tokens into tokenizer
        if ember_mode:
            from .ember_integration import MEMORY_TOKENS
            new_tokens = list(MEMORY_TOKENS.values())
            n_added = fig.tokenizer.add_special_tokens(
                {"additional_special_tokens": new_tokens}
            )
            if n_added > 0:
                print(f"   🔥 Added {n_added} Ember memory tokens to tokenizer")
        
        # Load model
        print(f"   Loading original model...")
        load_kwargs = {
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
        }
        try:
            load_kwargs["device_map"] = "cpu"
            orig_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except (ImportError, ValueError):
            # accelerate not installed — load without device_map
            del load_kwargs["device_map"]
            orig_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        fig._config = orig_model.config
        
        # Ember mode: resize embeddings to fit new tokens
        if ember_mode:
            orig_model.resize_token_embeddings(len(fig.tokenizer))
            print(f"   🔥 Resized embeddings to {len(fig.tokenizer)} tokens")
        
        # FigKernel: swap RMSNorm → FigRMSNorm for fused ops
        if fuse_kernels:
            _swap_rmsnorm(orig_model)
        
        # Quantize target linear layers with FigQuant and replace with FigLinear
        quantized_count = 0
        original_bytes = 0
        quantized_bytes = 0
        
        print(f"   Quantizing linear layers with FigQuant...")
        
        replacements = {}
        for name, module in orig_model.named_modules():
            # Support both nn.Linear and transformers Conv1D (used by GPT-2)
            is_linear = isinstance(module, nn.Linear)
            is_conv1d = module.__class__.__name__ == "Conv1D"
            
            if not (is_linear or is_conv1d):
                continue
            
            # Check if this module matches target_modules
            module_type = name.split(".")[-1]
            if not any(t in module_type for t in target_modules):
                continue
            
            weight = module.weight.data
            
            # Conv1D stores weight as (in_features, out_features) — transpose it
            if is_conv1d:
                in_features = weight.shape[0]
                out_features = weight.shape[1]
                weight = weight.T.contiguous()  # → (out_features, in_features)
            else:
                in_features = module.in_features
                out_features = module.out_features
            
            original_bytes += weight.numel() * 4
            
            # Quantize with FigQuant (adaptive codebook)
            fq = figquant_quantize(weight, group_size=group_size)
            quantized_bytes += fq.nbytes
            
            # Create FigLinear replacement
            bias = module.bias.data if module.bias is not None else None
            fig_layer = FigLinear(
                in_features=in_features,
                out_features=out_features,
                fq=fq,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
                fast=fast,
            )
            
            replacements[name] = (fig_layer, is_conv1d)
            quantized_count += 1
        
        # Apply replacements
        for name, (fig_layer, was_conv1d) in replacements.items():
            parts = name.split(".")
            parent = orig_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], fig_layer)
            fig._fig_layers[name] = fig_layer
        
        # Freeze all non-LoRA parameters
        for name, param in orig_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
        
        fig.model = orig_model
        
        # Compile if requested
        if compile_model:
            try:
                print(f"   Applying torch.compile (inductor backend)...")
                fig.model = torch.compile(fig.model, backend="inductor", mode="reduce-overhead")
                print(f"   ✓ Model compiled")
            except Exception as e:
                print(f"   ⚠ torch.compile failed: {e}")
        
        # Stats
        trainable = sum(p.numel() for p in fig.parameters() if p.requires_grad)
        total = sum(p.numel() for p in fig.parameters())
        ratio = original_bytes / max(quantized_bytes, 1)
        
        print(f"   ✓ Quantized {quantized_count} layers via FigQuant ({ratio:.1f}× compression)")
        print(f"   ✓ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print(f"   ✓ Base weights: {original_bytes/1e6:.1f} MB → {quantized_bytes/1e6:.1f} MB")
        
        return fig
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Forward pass — delegates to the underlying HF model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
    
    def generate(self, **kwargs):
        """Generate text — delegates to the underlying HF model."""
        self.model.eval()
        with torch.no_grad():
            return self.model.generate(**kwargs)
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get only the trainable (LoRA) parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_lora_state_dict(self) -> dict:
        """Get state dict of only LoRA parameters."""
        return {
            name: param.data
            for name, param in self.named_parameters()
            if param.requires_grad
        }
    
    def save_adapter(self, save_dir: str):
        """Save only the LoRA adapter weights."""
        os.makedirs(save_dir, exist_ok=True)
        
        state = self.get_lora_state_dict()
        torch.save(state, os.path.join(save_dir, "fig_adapter.pt"))
        
        # Save metadata
        meta = {
            "model_name": self.model_name,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "tier": self.tier.value,
            "trainable_params": sum(p.numel() for p in state.values()),
        }
        with open(os.path.join(save_dir, "fig_adapter_config.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)
        
        total_kb = sum(p.numel() * 4 for p in state.values()) / 1024
        print(f"🍐 Adapter saved: {save_dir} ({total_kb:.0f} KB)")
    
    def load_adapter(self, adapter_dir: str):
        """Load LoRA adapter weights."""
        state = torch.load(
            os.path.join(adapter_dir, "fig_adapter.pt"),
            map_location="cpu",
            weights_only=True,
        )
        
        # Load into model
        current_state = dict(self.named_parameters())
        loaded = 0
        for name, tensor in state.items():
            if name in current_state:
                current_state[name].data.copy_(tensor)
                loaded += 1
        
        print(f"🍐 Adapter loaded: {loaded} parameters from {adapter_dir}")
    
    def merge_and_export(self, save_dir: str):
        """
        Merge LoRA adapters into base weights and save as a standard
        HuggingFace model (FP32). For deployment/sharing.
        """
        print(f"🍐 Merging LoRA adapters into base model...")
        
        for name, fig_layer in self._fig_layers.items():
            if isinstance(fig_layer, FigLinear):
                # Get merged weight
                merged_weight = fig_layer.merge_lora()
                
                # Replace FigLinear with standard nn.Linear
                parts = name.split(".")
                parent = self.model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                
                new_linear = nn.Linear(
                    fig_layer.in_features, fig_layer.out_features,
                    bias=fig_layer.bias is not None,
                )
                new_linear.weight.data = merged_weight
                if fig_layer.bias is not None:
                    new_linear.bias.data = fig_layer.bias.data
                
                setattr(parent, parts[-1], new_linear)
        
        # Save as HF model
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)
        
        print(f"🍐 ✓ Merged model saved: {save_dir}")
    
    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload model trained with Fig Engine",
        private: bool = False,
        merge_before_push: bool = True,
    ):
        """
        Push the trained model to HuggingFace Hub.
        
        Args:
            repo_id: HuggingFace repo ID (e.g., "username/my-model")
            commit_message: Commit message
            private: Whether to make the repo private
            merge_before_push: If True, merges LoRA into base weights first
        """
        import tempfile
        
        if merge_before_push:
            print(f"🍐 Merging LoRA and pushing to Hub: {repo_id}")
            with tempfile.TemporaryDirectory() as tmpdir:
                self.merge_and_export(tmpdir)
                
                from transformers import AutoModelForCausalLM
                merged = AutoModelForCausalLM.from_pretrained(tmpdir)
                merged.push_to_hub(repo_id, commit_message=commit_message, private=private)
                
                if self.tokenizer:
                    self.tokenizer.push_to_hub(repo_id, commit_message=commit_message)
        else:
            # Push adapter only
            print(f"🍐 Pushing adapter to Hub: {repo_id}")
            with tempfile.TemporaryDirectory() as tmpdir:
                self.save_adapter(tmpdir)
                from huggingface_hub import HfApi
                api = HfApi()
                api.create_repo(repo_id, exist_ok=True, private=private)
                api.upload_folder(folder_path=tmpdir, repo_id=repo_id,
                                  commit_message=commit_message)
        
        print(f"🍐 ✓ Pushed to https://huggingface.co/{repo_id}")
    
    def print_trainable_summary(self):
        """Print a summary of trainable vs frozen parameters."""
        trainable = 0
        frozen = 0
        lora_params = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable += param.numel()
                if "lora_" in name:
                    lora_params += param.numel()
            else:
                frozen += param.numel()
        
        total = trainable + frozen
        print(f"\n🍐 Model Summary:")
        print(f"   Total parameters:     {total:>12,}")
        print(f"   Frozen (INT4 base):   {frozen:>12,}  ({100*frozen/total:.1f}%)")
        print(f"   Trainable:            {trainable:>12,}  ({100*trainable/total:.1f}%)")
        if lora_params > 0:
            print(f"   └─ LoRA parameters:   {lora_params:>12,}")
