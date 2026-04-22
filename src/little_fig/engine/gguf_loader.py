"""
Fig Engine — Universal GGUF Loader (v2)

Loads ANY GGUF model automatically. Zero hardcoded architecture mappings.

Loading strategy (in order):
  1. transformers built-in GGUF loader (22+ architectures, fully automatic)
  2. gguf-py TensorNameMap fallback (122+ architectures known to gguf-py)
  3. Graceful error with actionable message

No llama-cpp-python needed. No C compilation. Pure Python.
"""

import os
import sys
import torch
import numpy as np
from typing import Optional


def _ensure_gguf():
    try:
        import gguf
        return gguf
    except ImportError:
        raise ImportError(
            "The 'gguf' package is required to load GGUF files.\n"
            "Install: pip install gguf"
        )


# ── Architecture detection ────────────────────────────────────────────────────

def detect_gguf_arch(gguf_path: str) -> str:
    """
    Read the architecture string from a GGUF file.
    Handles the gguf package returning raw bytes as numpy arrays.
    """
    gguf_mod = _ensure_gguf()
    reader = gguf_mod.GGUFReader(gguf_path, mode="r")
    field = reader.fields.get("general.architecture")
    if field is None:
        raise ValueError(f"GGUF file missing general.architecture: {gguf_path}")
    # The value is in field.parts[-1] as a numpy array of uint8 (ASCII bytes)
    raw = field.parts[-1]
    if hasattr(raw, "tobytes"):
        return raw.tobytes().decode("utf-8", errors="replace").rstrip("\x00")
    return bytes(raw).decode("utf-8", errors="replace").rstrip("\x00")


def detect_gguf_name(gguf_path: str) -> str:
    """Read the model name from a GGUF file."""
    gguf_mod = _ensure_gguf()
    reader = gguf_mod.GGUFReader(gguf_path, mode="r")
    field = reader.fields.get("general.name")
    if field is None:
        return os.path.basename(gguf_path)
    raw = field.parts[-1]
    if hasattr(raw, "tobytes"):
        return raw.tobytes().decode("utf-8", errors="replace").rstrip("\x00")
    return bytes(raw).decode("utf-8", errors="replace").rstrip("\x00")


# ── Primary loader: transformers built-in ─────────────────────────────────────

def _load_via_transformers(gguf_path: str):
    """
    Use transformers' built-in GGUF support.
    Works for all architectures in GGUF_SUPPORTED_ARCHITECTURES (22+).
    Returns (model, tokenizer) or raises on failure.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # transformers needs accelerate for GGUF loading
    try:
        import accelerate  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'accelerate' package is required for GGUF loading.\n"
            "Install: pip install accelerate"
        )

    gguf_dir = os.path.dirname(os.path.abspath(gguf_path))
    gguf_file = os.path.basename(gguf_path)

    print(f"   Loading via transformers built-in GGUF loader...")
    model = AutoModelForCausalLM.from_pretrained(
        gguf_dir,
        gguf_file=gguf_file,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    # Tokenizer is also extracted from the GGUF file by transformers
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            gguf_dir,
            gguf_file=gguf_file,
        )
    except Exception:
        tokenizer = None

    return model, tokenizer


def _is_transformers_supported(arch: str) -> bool:
    """Check if transformers knows how to load this GGUF architecture."""
    try:
        from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
        # Also check common name normalizations
        normalised = arch.replace("-", "_").replace("moe", "_moe")
        return arch in GGUF_SUPPORTED_ARCHITECTURES or normalised in GGUF_SUPPORTED_ARCHITECTURES
    except ImportError:
        return False


# ── Fallback loader: gguf-py TensorNameMap ────────────────────────────────────

def _get_arch_enum(arch_str: str):
    """Look up the gguf-py MODEL_ARCH enum for an architecture string."""
    try:
        from gguf import MODEL_ARCH_NAMES
        for enum_val, name in MODEL_ARCH_NAMES.items():
            if name == arch_str:
                return enum_val
    except ImportError:
        pass
    return None


def _read_gguf_int_field(reader, key: str) -> Optional[int]:
    """Read a single integer field from a GGUF reader."""
    field = reader.fields.get(key)
    if field is None:
        return None
    try:
        val = field.parts[-1]
        if hasattr(val, "item"):
            return int(val.item())
        if hasattr(val, "__len__") and len(val) == 1:
            return int(val[0])
        return int(val)
    except (ValueError, TypeError, IndexError):
        return None


def _load_via_tensor_name_map(gguf_path: str, arch: str):
    """
    Fallback: use gguf-py's TensorNameMap for architectures not yet in
    transformers' GGUF loader.  gguf-py knows 122+ architectures.

    Strategy:
      1. Read metadata → build HF config
      2. Create empty HF model
      3. Use TensorNameMap to map GGUF tensor names → HF param names
      4. Dequantize and load weights
    """
    from gguf import GGUFReader, dequantize, get_tensor_name_map

    arch_enum = _get_arch_enum(arch)
    if arch_enum is None:
        raise RuntimeError(
            f"Architecture '{arch}' is not known to gguf-py.\n"
            f"This is a very new model format. Try updating:\n"
            f"  pip install --upgrade gguf transformers"
        )

    reader = GGUFReader(gguf_path, mode="r")

    # Read layer count (needed for TensorNameMap)
    n_layers = _read_gguf_int_field(reader, f"{arch}.block_count")
    if n_layers is None:
        # Try common fallback prefix
        for prefix in ["llama", "gemma", "qwen2"]:
            n_layers = _read_gguf_int_field(reader, f"{prefix}.block_count")
            if n_layers:
                break
    if not n_layers:
        raise RuntimeError(f"Could not determine layer count from GGUF metadata.")

    # Build tensor name map (GGUF canonical → HF param names)
    name_map = get_tensor_name_map(arch_enum, n_layers)

    # Build reverse map: GGUF tensor name → HF param name
    # name_map.mapping is { hf_name_prefix: (tensor_type, gguf_canonical_prefix) }
    hf_to_gguf = {}
    for hf_key, (tensor_type, gguf_canonical) in name_map.mapping.items():
        hf_to_gguf[hf_key] = gguf_canonical

    # Invert: gguf_canonical → hf_key
    gguf_to_hf = {v: k for k, v in hf_to_gguf.items()}

    # Now we need a model. Try downloading config from Hub based on model name.
    model_name = detect_gguf_name(gguf_path)
    model, config = _create_model_for_arch(arch, model_name, reader)

    model_state = model.state_dict()
    loaded = 0
    skipped = 0
    total_tensors = len(reader.tensors)

    print(f"   Loading {total_tensors} tensors via TensorNameMap (arch={arch})...")

    for i, tensor in enumerate(reader.tensors):
        gguf_name = tensor.name
        shape = tensor.shape.tolist()

        # Try TensorNameMap lookup
        # Strip .weight/.bias suffix for lookup, then re-add
        base_name = gguf_name
        for suffix in (".weight", ".bias"):
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break

        hf_base = gguf_to_hf.get(base_name)
        if hf_base is None:
            skipped += 1
            continue

        # Try with .weight and .bias
        hf_name = None
        for suffix in (".weight", ".bias"):
            candidate = hf_base + suffix
            if candidate in model_state:
                hf_name = candidate
                break
        if hf_name is None:
            # Maybe the HF name already includes the suffix
            if hf_base in model_state:
                hf_name = hf_base
        if hf_name is None:
            skipped += 1
            continue

        # Dequantize
        qtype = tensor.tensor_type
        data = tensor.data
        if qtype.name == "F32":
            arr = data.reshape(shape).astype(np.float32)
        elif qtype.name == "F16":
            arr = data.view(np.float16).reshape(shape).astype(np.float32)
        elif qtype.name == "BF16":
            raw = data.view(np.uint16)
            arr = np.frombuffer(
                np.left_shift(raw.astype(np.uint32), 16).tobytes(), dtype=np.float32
            ).reshape(shape)
        else:
            arr = dequantize(data, qtype).reshape(shape)

        fp32_tensor = torch.from_numpy(arr.copy())
        target_shape = model_state[hf_name].shape

        if fp32_tensor.shape == target_shape:
            model_state[hf_name].copy_(fp32_tensor)
            loaded += 1
        elif fp32_tensor.T.shape == target_shape:
            model_state[hf_name].copy_(fp32_tensor.T.contiguous())
            loaded += 1
        else:
            skipped += 1

        del fp32_tensor, arr

        if (i + 1) % 100 == 0:
            print(f"   ... {i+1}/{total_tensors} tensors processed")

    model.load_state_dict(model_state, strict=False)
    print(f"   ✓ Loaded {loaded}/{total_tensors} tensors (skipped {skipped})")

    # Find a tokenizer
    tokenizer = _find_tokenizer_for_arch(arch, model_name)

    return model, tokenizer


def _create_model_for_arch(arch: str, model_name: str, reader):
    """
    Create an empty HF model for a given architecture.
    Tries Hub config download first, then manual construction.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    # Try to find a Hub model ID that matches this architecture
    hub_candidates = _guess_hub_id(arch, model_name)

    for hub_id in hub_candidates:
        try:
            print(f"   Trying config from {hub_id}...")
            cfg = AutoConfig.from_pretrained(hub_id)
            # For multimodal models, use the text sub-config
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                cfg = text_cfg
            print(f"   ✓ Config: {cfg.__class__.__name__} from {hub_id}")

            try:
                model = AutoModelForCausalLM.from_config(cfg)
            except Exception:
                # Some configs need their specific model class
                model = _instantiate_model_from_config(cfg)
            model.eval()
            return model, cfg
        except Exception as e:
            print(f"   ⚠ {hub_id}: {e}")
            continue

    raise RuntimeError(
        f"Could not create model for architecture '{arch}'.\n"
        f"No matching HuggingFace config found.\n"
        f"Try updating: pip install --upgrade transformers"
    )


def _guess_hub_id(arch: str, model_name: str) -> list:
    """
    Guess HuggingFace Hub model IDs that might have the right config
    for a given GGUF architecture. Uses the model name from the GGUF
    metadata to make educated guesses.
    """
    candidates = []
    name_lower = (model_name or "").lower()

    # Common arch → Hub ID mappings (only as hints, not a hard requirement)
    _known = {
        "gemma4":  ["google/gemma-4-e4b-it"],
        "gemma3n": ["google/gemma-4-e4b-it"],
        "gemma3":  ["google/gemma-3-4b-it"],
        "gemma2":  ["google/gemma-2-2b-it"],
        "gemma":   ["google/gemma-2-2b-it"],
        "llama":   ["meta-llama/Llama-3.2-1B-Instruct"],
        "qwen2":   ["Qwen/Qwen2.5-0.5B-Instruct"],
        "qwen3":   ["Qwen/Qwen3-0.6B"],
        "phi3":    ["microsoft/Phi-3-mini-4k-instruct"],
        "phi":     ["microsoft/phi-2"],
    }

    # Add known candidates
    candidates.extend(_known.get(arch, []))

    # Also try to construct a candidate from the model name
    # e.g. "Gemma-4-E4B-It" → "google/gemma-4-e4b-it"
    if "gemma" in name_lower:
        if "4" in name_lower:
            candidates.append("google/gemma-4-e4b-it")
        elif "3" in name_lower:
            candidates.append("google/gemma-3-4b-it")
    if "llama" in name_lower:
        candidates.append("meta-llama/Llama-3.2-1B-Instruct")
    if "qwen" in name_lower:
        candidates.append("Qwen/Qwen2.5-0.5B-Instruct")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _instantiate_model_from_config(config):
    """
    Instantiate a CausalLM model from a config that AutoModelForCausalLM
    can't auto-dispatch (e.g. text sub-configs from multimodal models).
    """
    import transformers

    config_type = type(config).__name__
    # Try common pattern: XxxConfig → XxxForCausalLM
    base = config_type.replace("TextConfig", "").replace("Config", "")
    for suffix in ["ForCausalLM", "ForCausalLM"]:
        cls_name = f"{base}{suffix}"
        cls = getattr(transformers, cls_name, None)
        if cls is not None:
            print(f"   Using {cls_name}")
            return cls(config)

    raise RuntimeError(f"No ForCausalLM class found for {config_type}")


def _find_tokenizer_for_arch(arch: str, model_name: str):
    """Try to find a matching tokenizer from HuggingFace Hub."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    candidates = _guess_hub_id(arch, model_name)
    for c in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(c)
            print(f"   ✓ Tokenizer from {c}")
            return tok
        except Exception:
            continue

    print(f"   ⚠ No matching tokenizer found for '{model_name}'")
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def load_gguf_as_fig_model(
    gguf_path: str,
    lora_r: int = 0,
    lora_alpha: int = 0,
):
    """
    Load ANY GGUF file as a FigModel.  Fully automatic — no hardcoded
    architecture mappings needed.

    Loading strategy:
      1. Try transformers built-in GGUF loader (fast, 22+ archs)
      2. Fall back to gguf-py TensorNameMap (122+ archs)
      3. Raise with actionable error message

    For inference (lora_r=0): loads model in FP32.
    For training (lora_r>0): quantizes to FIG4 + adds LoRA adapters.
    """
    from .model import FigModel

    arch = detect_gguf_arch(gguf_path)
    model_name = detect_gguf_name(gguf_path)

    print(f"🍐 Fig Engine: Loading GGUF (universal loader)")
    print(f"   Architecture: {arch}")
    print(f"   Model: {model_name}")

    # ── Strategy 1: transformers built-in ─────────────────────────────────
    if _is_transformers_supported(arch):
        print(f"   ✓ Architecture '{arch}' supported by transformers")
        try:
            model, tokenizer = _load_via_transformers(gguf_path)
            print(f"   ✓ Model loaded via transformers ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")
        except Exception as e:
            print(f"   ⚠ transformers loader failed: {e}")
            print(f"   Trying gguf-py TensorNameMap fallback...")
            model, tokenizer = _load_via_tensor_name_map(gguf_path, arch)
    else:
        # ── Strategy 2: gguf-py TensorNameMap ─────────────────────────────
        print(f"   Architecture '{arch}' not in transformers GGUF loader")
        print(f"   Using gguf-py TensorNameMap fallback...")
        try:
            model, tokenizer = _load_via_tensor_name_map(gguf_path, arch)
        except Exception as e:
            raise RuntimeError(
                f"Could not load GGUF file: {gguf_path}\n"
                f"Architecture: {arch}\n"
                f"Error: {e}\n\n"
                f"This architecture may be too new. Try:\n"
                f"  pip install --upgrade transformers gguf\n"
                f"Or use a different model format (HuggingFace safetensors)."
            ) from e

    model.eval()

    # Wrap in FigModel
    fig = FigModel()
    fig.model = model
    fig.model_name = model_name if isinstance(model_name, str) else os.path.basename(gguf_path)
    fig._config = getattr(model, "config", None)
    fig.tokenizer = tokenizer
    fig.lora_r = lora_r
    fig.lora_alpha = lora_alpha

    # Optionally quantize to FIG4 + add LoRA (for training)
    if lora_r > 0:
        _apply_fig4_quantization(fig, lora_r, lora_alpha)

    return fig


def _apply_fig4_quantization(fig, lora_r: int, lora_alpha: int):
    """Quantize linear layers to FIG4 INT4 and add LoRA adapters."""
    import torch.nn as nn
    from .quantize import FigQuantizer
    from .linear import FigLinear

    quantizer = FigQuantizer(group_size=128)
    model = fig.model

    # Auto-detect target modules from the model
    target_keywords = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",
                       # GPT-2 style
                       "c_attn", "c_proj", "c_fc",
                       # Phi style
                       "dense", "fc1", "fc2"]

    replacements = {}
    original_bytes = 0
    quantized_bytes = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        module_type = name.split(".")[-1]
        if not any(t in module_type for t in target_keywords):
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


# ── Metadata reading (kept for other modules that may use it) ─────────────────

def read_gguf_metadata(path: str) -> dict:
    """Read GGUF file metadata. Returns decoded string values."""
    gguf_mod = _ensure_gguf()
    reader = gguf_mod.GGUFReader(path, mode="r")

    metadata = {}
    for field in reader.fields.values():
        name = field.name
        if field.types and len(field.data) > 0:
            try:
                raw = field.parts[-1]
                if hasattr(raw, "tobytes"):
                    # Try string decode first
                    try:
                        val = raw.tobytes().decode("utf-8", errors="replace").rstrip("\x00")
                        if val.isprintable() and val:
                            metadata[name] = val
                            continue
                    except Exception:
                        pass
                    # Numeric value
                    if hasattr(raw, "item"):
                        metadata[name] = raw.item()
                    else:
                        metadata[name] = raw.tolist()
                elif hasattr(raw, "item"):
                    metadata[name] = raw.item()
                else:
                    metadata[name] = raw
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
