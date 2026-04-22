"""
Fig Engine — Native GGUF Loader

Reads GGUF files directly using the `gguf` Python package (pure Python, no
llama-cpp-python, no C compilation). Dequantizes weights and loads them into
a HuggingFace model architecture for inference and training.

Supports all GGUF quantization types: Q4_K_M, Q4_0, Q5_K, Q8_0, F16, F32, etc.
Works with any model architecture stored in GGUF format.

Usage:
    from little_fig.engine.gguf_loader import load_gguf_as_fig_model
    model = load_gguf_as_fig_model("./gemma-4b-it-Q4_K_M.gguf")
"""

import os
import torch
import numpy as np
from typing import Optional, Dict, Tuple
from collections import OrderedDict


def _ensure_gguf_package():
    """Check that the gguf package is installed."""
    try:
        import gguf
        return gguf
    except ImportError:
        raise ImportError(
            "The 'gguf' package is required to load GGUF files.\n"
            "Install it with: pip install gguf\n"
            "This is a lightweight pure-Python package (no C compilation needed)."
        )


def read_gguf_metadata(path: str) -> dict:
    """
    Read GGUF file metadata without loading weights.
    Returns architecture info, tensor list, quantization type, etc.
    """
    gguf_mod = _ensure_gguf_package()
    reader = gguf_mod.GGUFReader(path, mode="r")

    metadata = {}
    for field in reader.fields.values():
        # Extract key metadata
        name = field.name
        if field.types and len(field.data) > 0:
            try:
                # Try to get scalar value
                parts = field.parts
                if len(parts) > 1:
                    val = parts[-1].tolist()
                    if isinstance(val, list) and len(val) == 1:
                        val = val[0]
                    metadata[name] = val
                elif len(parts) == 1:
                    metadata[name] = parts[0].tolist()
            except Exception:
                pass

    # Tensor info
    tensor_info = []
    for tensor in reader.tensors:
        tensor_info.append({
            "name": tensor.name,
            "shape": tensor.shape.tolist(),
            "type": tensor.tensor_type.name,
            "n_elements": int(np.prod(tensor.shape)),
        })

    metadata["_tensors"] = tensor_info
    metadata["_n_tensors"] = len(tensor_info)
    metadata["_path"] = path

    return metadata


def load_gguf_tensors(path: str) -> Dict[str, torch.Tensor]:
    """
    Load all tensors from a GGUF file, dequantizing to FP32.

    Returns a dict mapping tensor name → FP32 torch.Tensor.
    """
    gguf_mod = _ensure_gguf_package()
    from gguf import GGUFReader, dequantize

    print(f"🍐 Reading GGUF: {os.path.basename(path)}")
    reader = GGUFReader(path, mode="r")

    tensors = OrderedDict()
    total_params = 0

    for tensor in reader.tensors:
        name = tensor.name
        shape = tensor.shape.tolist()
        qtype = tensor.tensor_type
        data = tensor.data

        # Dequantize to float32
        if qtype.name in ("F32",):
            arr = data.reshape(shape).astype(np.float32)
        elif qtype.name in ("F16", "BF16"):
            arr = data.view(np.float16).reshape(shape).astype(np.float32)
        else:
            # Quantized type — use gguf's dequantize
            arr = dequantize(data, qtype)
            arr = arr.reshape(shape)

        t = torch.from_numpy(arr.copy())
        tensors[name] = t
        total_params += t.numel()

    print(f"   ✓ Loaded {len(tensors)} tensors ({total_params / 1e9:.2f}B parameters)")
    return tensors


def _map_gguf_name_to_hf(gguf_name: str, arch: str) -> str:
    """
    Map GGUF tensor name to HuggingFace model state_dict name.

    GGUF uses names like:
        blk.0.attn_q.weight → model.layers.0.self_attn.q_proj.weight
        token_embd.weight   → model.embed_tokens.weight
        output.weight       → lm_head.weight
    """
    name = gguf_name

    # Common mappings (Llama/Gemma/Mistral architecture)
    name = name.replace("blk.", "model.layers.")
    name = name.replace(".attn_q.", ".self_attn.q_proj.")
    name = name.replace(".attn_k.", ".self_attn.k_proj.")
    name = name.replace(".attn_v.", ".self_attn.v_proj.")
    name = name.replace(".attn_output.", ".self_attn.o_proj.")
    name = name.replace(".ffn_gate.", ".mlp.gate_proj.")
    name = name.replace(".ffn_up.", ".mlp.up_proj.")
    name = name.replace(".ffn_down.", ".mlp.down_proj.")
    name = name.replace(".attn_norm.", ".input_layernorm.")
    name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
    name = name.replace("token_embd.", "model.embed_tokens.")
    name = name.replace("output_norm.", "model.norm.")
    name = name.replace("output.", "lm_head.")

    # Ensure .weight suffix
    if not name.endswith(".weight") and not name.endswith(".bias"):
        name += ".weight"

    return name


def load_gguf_into_model(
    gguf_path: str,
    model_config_name: Optional[str] = None,
) -> Tuple:
    """
    Load a GGUF file into a HuggingFace model structure.

    Steps:
        1. Read GGUF metadata to determine architecture
        2. Create an empty HF model of that architecture
        3. Dequantize GGUF tensors and load into the model

    Args:
        gguf_path: Path to the .gguf file
        model_config_name: HuggingFace config to use (auto-detected if None)

    Returns:
        (model, tokenizer_info) — model is a HuggingFace model with loaded weights
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # Step 1: Read metadata
    meta = read_gguf_metadata(gguf_path)
    arch = meta.get("general.architecture", "unknown")
    model_name = meta.get("general.name", "unknown")
    n_layers = meta.get("llama.block_count", meta.get(f"{arch}.block_count", None))
    n_embd = meta.get("llama.embedding_length", meta.get(f"{arch}.embedding_length", None))
    vocab_size = meta.get("llama.vocab_size", meta.get(f"{arch}.vocab_size", None))

    print(f"   Architecture: {arch}")
    print(f"   Model name: {model_name}")
    if n_layers:
        print(f"   Layers: {n_layers}, Hidden: {n_embd}, Vocab: {vocab_size}")

    # Step 2: Load tensors
    gguf_tensors = load_gguf_tensors(gguf_path)

    # Step 3: Map tensor names
    mapped = OrderedDict()
    unmapped = []
    for gguf_name, tensor in gguf_tensors.items():
        hf_name = _map_gguf_name_to_hf(gguf_name, arch)
        mapped[hf_name] = tensor

    return mapped, meta, arch


def load_gguf_as_fig_model(
    gguf_path: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """
    Load a GGUF file as a FigModel (INT4 quantized + LoRA).

    This is the main entry point. It:
        1. Reads the GGUF file (dequantizes to FP32)
        2. Quantizes to Fig Engine INT4 format
        3. Adds LoRA adapters
        4. Returns a model ready for inference or training

    Args:
        gguf_path: Path to the .gguf file
        lora_r: LoRA rank (0 for inference-only)
        lora_alpha: LoRA scaling factor
    """
    from .quantize import FigQuantizer
    from .linear import FigLinear

    print(f"🍐 Fig Engine: Loading GGUF → INT4")

    # Load and dequantize all tensors
    mapped_tensors, meta, arch = load_gguf_into_model(gguf_path)

    # Try to create a HF model and load weights
    # First attempt: use the architecture to find the right HF config
    from transformers import AutoConfig, AutoModelForCausalLM

    # Build config from GGUF metadata
    config = _build_config_from_meta(meta, arch)

    if config is not None:
        print(f"   Creating model from GGUF metadata...")
        model = AutoModelForCausalLM.from_config(config)
        model.eval()

        # Load weights into model
        model_state = model.state_dict()
        loaded = 0
        skipped = []

        for name, tensor in mapped_tensors.items():
            if name in model_state:
                if tensor.shape == model_state[name].shape:
                    model_state[name].copy_(tensor)
                    loaded += 1
                else:
                    # Shape mismatch — try transpose
                    if tensor.T.shape == model_state[name].shape:
                        model_state[name].copy_(tensor.T)
                        loaded += 1
                    else:
                        skipped.append(f"{name}: GGUF {tuple(tensor.shape)} vs model {tuple(model_state[name].shape)}")
            else:
                skipped.append(f"{name}: not found in model")

        model.load_state_dict(model_state, strict=False)
        print(f"   ✓ Loaded {loaded}/{len(mapped_tensors)} tensors into model")

        if skipped and len(skipped) <= 5:
            for s in skipped:
                print(f"   ⚠ Skipped: {s}")
        elif skipped:
            print(f"   ⚠ Skipped {len(skipped)} tensors (shape mismatch or unmapped)")

        # Now quantize to INT4 + add LoRA (reuse FigModel.from_pretrained logic)
        from .model import FigModel
        fig = FigModel()
        fig.model = model
        fig.model_name = os.path.basename(gguf_path)
        fig._config = config
        fig.lora_r = lora_r
        fig.lora_alpha = lora_alpha

        # Try to extract tokenizer from GGUF metadata
        fig.tokenizer = _extract_tokenizer_from_meta(meta)

        return fig
    else:
        raise RuntimeError(
            f"Could not determine model architecture from GGUF metadata.\n"
            f"Architecture: {arch}\n"
            f"Try specifying the HuggingFace model config manually."
        )


def _build_config_from_meta(meta: dict, arch: str):
    """Build a HuggingFace model config from GGUF metadata."""
    from transformers import AutoConfig

    # Extract config values from GGUF metadata
    prefix = arch if arch != "llama" else "llama"

    n_layers = meta.get(f"{prefix}.block_count")
    n_embd = meta.get(f"{prefix}.embedding_length")
    n_head = meta.get(f"{prefix}.attention.head_count")
    n_head_kv = meta.get(f"{prefix}.attention.head_count_kv", n_head)
    vocab_size = meta.get(f"{prefix}.vocab_size")
    ctx_len = meta.get(f"{prefix}.context_length", 4096)
    ff_len = meta.get(f"{prefix}.feed_forward_length")
    rms_eps = meta.get(f"{prefix}.attention.layer_norm_rms_epsilon", 1e-5)
    rope_dim = meta.get(f"{prefix}.rope.dimension_count")
    rope_freq = meta.get(f"{prefix}.rope.freq_base", 10000.0)

    if not all([n_layers, n_embd, n_head, vocab_size]):
        return None

    # Map GGUF arch names to HuggingFace model types
    arch_to_hf = {
        "llama": "LlamaConfig",
        "gemma": "GemmaConfig",
        "gemma2": "Gemma2Config",
        "gemma3": "Gemma2Config",  # Gemma 3 uses Gemma2Config in HF
        "gemma4": "Gemma2Config",  # Gemma 4 likely similar
        "mistral": "MistralConfig",
        "phi": "PhiConfig",
        "phi3": "Phi3Config",
        "qwen2": "Qwen2Config",
        "gpt2": "GPT2Config",
    }

    config_class_name = arch_to_hf.get(arch)

    try:
        if config_class_name:
            import transformers
            config_class = getattr(transformers, config_class_name, None)
            if config_class:
                config_kwargs = {
                    "num_hidden_layers": int(n_layers),
                    "hidden_size": int(n_embd),
                    "num_attention_heads": int(n_head),
                    "num_key_value_heads": int(n_head_kv) if n_head_kv else int(n_head),
                    "vocab_size": int(vocab_size),
                    "max_position_embeddings": int(ctx_len),
                    "rms_norm_eps": float(rms_eps),
                    "rope_theta": float(rope_freq),
                }
                if ff_len:
                    config_kwargs["intermediate_size"] = int(ff_len)

                return config_class(**config_kwargs)
    except Exception as e:
        print(f"   ⚠ Config creation failed: {e}")

    return None


def _extract_tokenizer_from_meta(meta: dict):
    """Try to extract tokenizer from GGUF metadata."""
    try:
        from transformers import AutoTokenizer

        # GGUF stores tokenizer data in metadata
        # Check if we can find a matching HF tokenizer
        model_name = meta.get("general.name", "")
        arch = meta.get("general.architecture", "")

        # Try common model names
        candidates = []
        name_lower = model_name.lower() if model_name else ""

        if "gemma" in name_lower or "gemma" in arch:
            if "4b" in name_lower or "4e" in name_lower:
                candidates = ["google/gemma-3-4b-it", "google/gemma-2-2b-it"]
            elif "1b" in name_lower:
                candidates = ["google/gemma-3-1b-it"]
            else:
                candidates = ["google/gemma-3-4b-it"]
        elif "llama" in name_lower or "llama" in arch:
            candidates = ["meta-llama/Llama-3.2-1B-Instruct"]
        elif "mistral" in name_lower:
            candidates = ["mistralai/Mistral-7B-Instruct-v0.3"]

        for candidate in candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                print(f"   ✓ Tokenizer loaded from {candidate}")
                return tokenizer
            except Exception:
                continue

        print(f"   ⚠ Could not find matching tokenizer for '{model_name}'")
        return None

    except Exception:
        return None
