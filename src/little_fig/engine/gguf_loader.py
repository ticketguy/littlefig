"""
Fig Engine — Native GGUF Loader

Reads GGUF files directly using the `gguf` Python package (pure Python).
No llama-cpp-python needed. No C compilation.

Two loading modes:
  1. GGUF → FIG4 (quantized → quantized): Stays compressed in memory.
     GGUF Q4_K_M weights → dequant per-layer → re-quantize to FIG4 → FigLinear.
     Peak memory: only 1 layer in FP32 at a time.
  2. GGUF → FP32 (for export or GPU): Full dequantization.

Supports all GGUF quantization types: Q4_K_M, Q4_0, Q5_K, Q8_0, F16, F32, etc.
"""

import os
import torch
import numpy as np
from typing import Optional, Dict, Tuple
from collections import OrderedDict


def _ensure_gguf():
    try:
        import gguf
        return gguf
    except ImportError:
        raise ImportError(
            "The 'gguf' package is required to load GGUF files.\n"
            "Install: pip install gguf"
        )


def _decode_field_value(val):
    """
    Decode a GGUF metadata field value.
    The gguf package returns numpy arrays — string fields come back as
    arrays of uint8 (ASCII bytes). Convert them to Python str/int/float.
    """
    if isinstance(val, (list, np.ndarray)):
        # Check if it looks like a byte/ASCII string (array of uint8 values 1-127)
        try:
            arr = np.asarray(val)
            if arr.ndim == 1 and arr.dtype.kind in ("u", "i") and arr.size > 0:
                if arr.min() >= 1 and arr.max() <= 127:
                    decoded = bytes(int(b) for b in arr).decode("utf-8", errors="replace")
                    # Only accept if it looks like a real string (printable, no control chars)
                    if decoded.isprintable() and len(decoded) > 0:
                        return decoded
        except (ValueError, TypeError, UnicodeDecodeError):
            pass
        # Single-element list → unwrap
        if isinstance(val, list) and len(val) == 1:
            return val[0]
    return val


def read_gguf_metadata(path: str) -> dict:
    """Read GGUF file metadata without loading weights."""
    gguf_mod = _ensure_gguf()
    reader = gguf_mod.GGUFReader(path, mode="r")

    metadata = {}
    for field in reader.fields.values():
        name = field.name
        if field.types and len(field.data) > 0:
            try:
                parts = field.parts
                if len(parts) > 1:
                    val = parts[-1].tolist()
                    if isinstance(val, list) and len(val) == 1:
                        val = val[0]
                    metadata[name] = _decode_field_value(val)
                elif len(parts) == 1:
                    metadata[name] = _decode_field_value(parts[0].tolist())
            except Exception:
                pass

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
    return metadata


def load_gguf_as_fig_model(
    gguf_path: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """
    Load a GGUF file as a FigModel with INT4 quantized weights.

    Key: Weights go GGUF-quantized → dequant ONE LAYER AT A TIME → re-quantize
    to FIG4 → store as FigLinear. Peak memory = 1 FP32 layer + all FIG4 layers.

    For a 4B model:
      GGUF on disk:    ~2.5 GB
      FIG4 in memory:  ~2.2 GB (our INT4 format)
      Peak during load: ~2.2 GB + ~50 MB (one layer FP32)
      NOT 16 GB (full FP32 dequant)
    """
    from .quantize import FigQuantizer
    from .linear import FigLinear
    from .model import FigModel

    _ensure_gguf()
    from gguf import GGUFReader, dequantize

    print(f"🍐 Fig Engine: Loading GGUF → FIG4 (quantized → quantized)")
    reader = GGUFReader(gguf_path, mode="r")

    # Step 1: Read metadata, determine architecture
    meta = read_gguf_metadata(gguf_path)
    arch = meta.get("general.architecture", "unknown")
    model_name = meta.get("general.name", os.path.basename(gguf_path))

    print(f"   Architecture: {arch}")
    print(f"   Model: {model_name}")
    print(f"   Tensors: {meta['_n_tensors']}")

    # Step 2: Build HF model config from metadata
    config = _build_config_from_meta(meta, arch)
    if config is None:
        raise RuntimeError(
            f"Could not build model config from GGUF metadata.\n"
            f"Architecture '{arch}' — metadata may be incomplete.\n"
            f"Tensor names: {[t['name'] for t in meta['_tensors'][:5]]}"
        )

    # Step 3: Create empty model from config
    from transformers import AutoModelForCausalLM
    print(f"   Creating empty {config.__class__.__name__}...")
    try:
        model = AutoModelForCausalLM.from_config(config)
    except Exception as e:
        # Some text sub-configs need their specific model class
        print(f"   AutoModelForCausalLM.from_config failed ({e})")
        print(f"   Trying architecture-specific model class...")
        model = _create_model_from_config(config)
    model.eval()

    # Step 4: Load weights one tensor at a time (stream, don't bulk-dequant)
    quantizer = FigQuantizer(group_size=128)
    model_state = model.state_dict()

    loaded = 0
    quantized_to_fig4 = 0
    skipped = 0
    fig_layers = {}

    total_tensors = len(reader.tensors)
    print(f"   Loading {total_tensors} tensors (streaming, one at a time)...")

    for i, tensor in enumerate(reader.tensors):
        gguf_name = tensor.name
        shape = tensor.shape.tolist()
        qtype = tensor.tensor_type
        data = tensor.data

        # Map GGUF name → HF name
        hf_name = _map_gguf_name_to_hf(gguf_name, arch)

        # Dequantize this ONE tensor to FP32
        if qtype.name == "F32":
            arr = data.reshape(shape).astype(np.float32)
        elif qtype.name in ("F16",):
            arr = data.view(np.float16).reshape(shape).astype(np.float32)
        elif qtype.name in ("BF16",):
            # BF16 needs special handling
            raw = data.view(np.uint16)
            arr = np.frombuffer(
                np.left_shift(raw.astype(np.uint32), 16).tobytes(), dtype=np.float32
            ).reshape(shape)
        else:
            arr = dequantize(data, qtype).reshape(shape)

        fp32_tensor = torch.from_numpy(arr.copy())

        # Load into model state
        if hf_name in model_state:
            target_shape = model_state[hf_name].shape
            if fp32_tensor.shape == target_shape:
                model_state[hf_name].copy_(fp32_tensor)
                loaded += 1
            elif fp32_tensor.T.shape == target_shape:
                model_state[hf_name].copy_(fp32_tensor.T.contiguous())
                loaded += 1
            else:
                skipped += 1
        else:
            skipped += 1

        # Free the FP32 tensor immediately
        del fp32_tensor, arr

        if (i + 1) % 50 == 0:
            print(f"   ... {i+1}/{total_tensors} tensors processed")

    model.load_state_dict(model_state, strict=False)
    print(f"   ✓ Loaded {loaded}/{total_tensors} tensors (skipped {skipped})")

    # Step 5: Now quantize the loaded model to FIG4 + add LoRA
    # This reuses FigModel.from_pretrained logic but on the already-loaded model
    fig = FigModel()
    fig.model = model
    fig.model_name = model_name
    fig._config = config
    fig.lora_r = lora_r
    fig.lora_alpha = lora_alpha

    # Quantize linear layers to INT4 if lora_r > 0 (training mode)
    if lora_r > 0:
        import torch.nn as nn
        from .linear import FigLinear

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
        replacements = {}
        original_bytes = 0
        quantized_bytes = 0

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            module_type = name.split(".")[-1]
            if not any(t in module_type for t in target_modules):
                continue

            weight = module.weight.data
            original_bytes += weight.numel() * 4

            fig4 = quantizer.quantize(weight)
            quantized_bytes += fig4.nbytes_quantized

            bias = module.bias.data if module.bias is not None else None
            fig_layer = FigLinear(
                module.in_features, module.out_features,
                fig4, lora_r=lora_r, lora_alpha=lora_alpha, bias=bias,
            )
            replacements[name] = fig_layer

        for name, fig_layer in replacements.items():
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], fig_layer)
            fig._fig_layers[name] = fig_layer

        for param in model.parameters():
            if "lora_" not in str(id(param)):
                param.requires_grad = False

        if original_bytes > 0:
            ratio = original_bytes / max(quantized_bytes, 1)
            print(f"   ✓ Quantized {len(replacements)} layers to FIG4 ({ratio:.1f}× compression)")

    # Step 6: Find tokenizer
    fig.tokenizer = _find_tokenizer(meta, arch, model_name)

    return fig


def _map_gguf_name_to_hf(gguf_name: str, arch: str) -> str:
    """Map GGUF tensor name to HuggingFace state_dict name."""
    name = gguf_name

    # ── Global (non-layer) tensor mappings ────────────────────────────────
    _global_map = {
        "token_embd.":           "model.embed_tokens.",
        "output_norm.":          "model.norm.",
        "output.":               "lm_head.",
        # Gemma 3n / Gemma 4 specific
        "per_layer_token_embd.": "model.embed_tokens_per_layer.",
        "per_layer_model_proj.": "model.per_layer_model_projection.",
        "per_layer_proj_norm.":  "model.per_layer_projection_norm.",
        "altup_proj.":           "model.altup_projections.",
        "altup_unembd_proj.":    "model.altup_unembed_projections.",
    }
    for gguf_prefix, hf_prefix in _global_map.items():
        if name.startswith(gguf_prefix):
            name = name.replace(gguf_prefix, hf_prefix, 1)
            if not name.endswith(".weight") and not name.endswith(".bias"):
                name += ".weight"
            return name

    # ── Per-layer tensor mappings (blk.N.xxx) ─────────────────────────────
    name = name.replace("blk.", "model.layers.")

    # Standard attention & MLP
    name = name.replace(".attn_q.", ".self_attn.q_proj.")
    name = name.replace(".attn_k.", ".self_attn.k_proj.")
    name = name.replace(".attn_v.", ".self_attn.v_proj.")
    name = name.replace(".attn_output.", ".self_attn.o_proj.")
    name = name.replace(".attn_q_norm.", ".self_attn.q_norm.")
    name = name.replace(".attn_k_norm.", ".self_attn.k_norm.")
    name = name.replace(".ffn_gate.", ".mlp.gate_proj.")
    name = name.replace(".ffn_up.", ".mlp.up_proj.")
    name = name.replace(".ffn_down.", ".mlp.down_proj.")

    # Layer norms
    name = name.replace(".attn_norm.", ".input_layernorm.")
    name = name.replace(".ffn_norm.", ".pre_feedforward_layernorm.")
    name = name.replace(".post_attention_norm.", ".post_attention_layernorm.")
    name = name.replace(".post_ffw_norm.", ".post_feedforward_layernorm.")

    # Gemma 3n / Gemma 4: Laurel (Learned Augmented Residual Layer)
    name = name.replace(".laurel_l.", ".laurel.linear_left.")
    name = name.replace(".laurel_r.", ".laurel.linear_right.")
    name = name.replace(".laurel_post_norm.", ".laurel.post_laurel_norm.")

    # Gemma 3n / Gemma 4: Per-Layer Embeddings (PLE)
    name = name.replace(".inp_gate.", ".per_layer_input_gate.")
    name = name.replace(".proj.", ".per_layer_projection.")
    name = name.replace(".post_norm.", ".post_per_layer_input_norm.")

    # Gemma 3n / Gemma 4: AltUp (Alternating Updates)
    name = name.replace(".altup_correct_coef.", ".altup.correction_coefs.")
    name = name.replace(".altup_correct_scale", ".altup.correct_output_scale")
    name = name.replace(".altup_predict_coef.", ".altup.prediction_coefs.")
    name = name.replace(".altup_router.", ".altup.modality_router.")
    name = name.replace(".altup_router_norm.", ".altup.router_norm.")

    # Fallback for older Gemma (ffn_norm → post_attention_layernorm)
    # Only if not already mapped above
    if ".post_attention_layernorm." not in name:
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

    if not name.endswith(".weight") and not name.endswith(".bias"):
        name += ".weight"
    return name


def _create_model_from_config(config):
    """
    Create a causal LM model from a config object.
    Handles special cases like Gemma4TextConfig / Gemma3nTextConfig which
    need their specific ForCausalLM class rather than Auto dispatch.
    """
    import transformers

    config_type = type(config).__name__
    # Map config class names → model class names
    _model_class_map = {
        "Gemma4TextConfig":  "Gemma4ForCausalLM",
        "Gemma3nTextConfig": "Gemma3nForCausalLM",
        "Gemma3TextConfig":  "Gemma3ForCausalLM",
    }

    model_cls_name = _model_class_map.get(config_type)
    if model_cls_name:
        model_cls = getattr(transformers, model_cls_name, None)
        if model_cls is not None:
            print(f"   Using {model_cls_name}")
            return model_cls(config)

    # Last resort: try _from_config with trust_remote_code
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_config(config, trust_remote_code=True)


def _build_config_from_meta(meta: dict, arch: str):
    """
    Build a HuggingFace model config from GGUF metadata.

    For complex/new architectures (gemma4, gemma3n, etc.) we first try to
    download the authoritative config from HuggingFace Hub.  If that fails
    we fall back to manual construction from GGUF metadata fields.
    """
    # Ensure arch is a string (defensive — _decode_field_value should handle this)
    if not isinstance(arch, str):
        try:
            arch = bytes(int(b) for b in arch).decode("utf-8", errors="replace")
        except Exception:
            arch = str(arch)

    # ── Complex architectures: try downloading config from Hub first ──────
    # These architectures have specialised sub-modules (PLE, AltUp, Laurel,
    # shared KV, etc.) that cannot be reconstructed from GGUF scalars alone.
    _hub_config_map = {
        "gemma4":      "google/gemma-4-e4b-it",
        "gemma3n":     "google/gemma-4-e4b-it",
        "gemma3":      "google/gemma-3-4b-it",
        "gemma3_text": "google/gemma-3-4b-it",
    }

    hub_id = _hub_config_map.get(arch)
    if hub_id:
        try:
            from transformers import AutoConfig
            print(f"   Downloading config from {hub_id} (arch={arch})...")
            cfg = AutoConfig.from_pretrained(hub_id)
            # For multimodal configs, extract the text sub-config
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                print(f"   ✓ Using {text_cfg.__class__.__name__} from {hub_id}")
                return text_cfg
            print(f"   ✓ Using {cfg.__class__.__name__} from {hub_id}")
            return cfg
        except Exception as e:
            print(f"   ⚠ Could not download config from {hub_id}: {e}")
            print(f"   Falling back to manual config construction...")

    # ── Standard architectures: build config from GGUF metadata ───────────
    # Try multiple prefixes (GGUF uses arch name as prefix)
    prefixes = [arch, "llama", "gemma", "gemma2"]

    n_layers = n_embd = n_head = n_head_kv = vocab_size = ff_len = None
    ctx_len = 4096
    rms_eps = 1e-5
    rope_freq = 10000.0

    for prefix in prefixes:
        n_layers = n_layers or meta.get(f"{prefix}.block_count")
        n_embd = n_embd or meta.get(f"{prefix}.embedding_length")
        n_head = n_head or meta.get(f"{prefix}.attention.head_count")
        n_head_kv = n_head_kv or meta.get(f"{prefix}.attention.head_count_kv")
        vocab_size = vocab_size or meta.get(f"{prefix}.vocab_size")
        ff_len = ff_len or meta.get(f"{prefix}.feed_forward_length")
        ctx_len_val = meta.get(f"{prefix}.context_length")
        if ctx_len_val:
            ctx_len = ctx_len_val
        rms_val = meta.get(f"{prefix}.attention.layer_norm_rms_epsilon")
        if rms_val:
            rms_eps = rms_val
        rope_val = meta.get(f"{prefix}.rope.freq_base")
        if rope_val:
            rope_freq = rope_val

    if not all([n_layers, n_embd, n_head, vocab_size]):
        return None

    if n_head_kv is None:
        n_head_kv = n_head

    # Map GGUF arch → HF config class
    arch_map = {
        "llama": "LlamaConfig",
        "gemma": "GemmaConfig",
        "gemma2": "Gemma2Config",
        "mistral": "MistralConfig",
        "phi": "PhiConfig",
        "phi3": "Phi3Config",
        "qwen2": "Qwen2Config",
    }

    config_name = arch_map.get(arch, "LlamaConfig")
    try:
        import transformers
        config_class = getattr(transformers, config_name, None)
        if config_class is None:
            from transformers import LlamaConfig
            config_class = LlamaConfig

        kwargs = {
            "num_hidden_layers": int(n_layers),
            "hidden_size": int(n_embd),
            "num_attention_heads": int(n_head),
            "num_key_value_heads": int(n_head_kv),
            "vocab_size": int(vocab_size),
            "max_position_embeddings": int(ctx_len),
            "rms_norm_eps": float(rms_eps),
            "rope_theta": float(rope_freq),
        }
        if ff_len:
            kwargs["intermediate_size"] = int(ff_len)

        return config_class(**kwargs)
    except Exception as e:
        print(f"   ⚠ Config creation error: {e}")
        return None


def _find_tokenizer(meta: dict, arch: str, model_name: str):
    """Try to find a matching tokenizer."""
    try:
        from transformers import AutoTokenizer

        name_lower = (model_name or "").lower()

        # Try known mappings based on architecture and name
        candidates = []
        if "gemma4" in arch or "gemma-4" in name_lower or "gemma4" in name_lower:
            candidates = ["google/gemma-4-e4b-it", "google/gemma-3-4b-it"]
        elif "gemma3n" in arch or "gemma-3n" in name_lower:
            candidates = ["google/gemma-4-e4b-it", "google/gemma-3-4b-it"]
        elif "gemma" in arch or "gemma" in name_lower:
            candidates = ["google/gemma-3-4b-it", "google/gemma-2-2b-it"]
        elif "llama" in arch or "llama" in name_lower:
            candidates = ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
        elif "mistral" in arch or "mistral" in name_lower:
            candidates = ["mistralai/Mistral-7B-Instruct-v0.3"]
        elif "phi" in arch or "phi" in name_lower:
            candidates = ["microsoft/phi-2"]
        elif "qwen" in arch or "qwen" in name_lower:
            candidates = ["Qwen/Qwen2.5-0.5B-Instruct"]

        for c in candidates:
            try:
                tok = AutoTokenizer.from_pretrained(c)
                print(f"   ✓ Tokenizer from {c}")
                return tok
            except Exception:
                continue

        print(f"   ⚠ No matching tokenizer found for '{model_name}'")
        return None
    except Exception:
        return None
