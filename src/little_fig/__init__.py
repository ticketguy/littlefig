"""
Little Fig — Entry Point
Detects hardware at startup and configures the runtime accordingly.
"""

import sys
import os
import torch


def detect_hardware():
    """
    Detect available hardware and return a config dict.
    This runs at startup before anything else.
    """
    hw = {
        "device": "cpu",
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "cpu_cores": os.cpu_count(),
        "torch_version": torch.__version__,
        "recommended_dtype": torch.float32,
        "recommended_backend": "hf",  # "hf" or "gguf"
    }

    if torch.cuda.is_available():
        hw["gpu_available"] = True
        hw["device"] = "cuda"
        hw["gpu_name"] = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        hw["gpu_vram_gb"] = round(vram, 1)
        hw["recommended_dtype"] = torch.float16
        hw["recommended_backend"] = "hf"

        # VRAM guidance for model selection
        if vram >= 16:
            hw["model_tier"] = "large"     # 7B+ fine-tuning possible
        elif vram >= 8:
            hw["model_tier"] = "medium"    # 4B fine-tuning, 7B inference
        else:
            hw["model_tier"] = "small"     # 1-3B fine-tuning
    else:
        # CPU — force off CUDA even if something claims otherwise
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FORCE_CPU"] = "1"
        hw["model_tier"] = "cpu"
        hw["recommended_backend"] = "gguf"  # GGUF is faster on CPU

    return hw


def print_startup_banner(hw: dict):
    fig = """
  ╔══════════════════════════════════════════╗
  ║           🍐  L I T T L E  F I G        ║
  ║      CPU-native LLM engine v0.2.0        ║
  ╚══════════════════════════════════════════╝"""
    print(fig)

    if hw["gpu_available"]:
        print(f"  ⚡ GPU detected : {hw['gpu_name']}")
        print(f"  ⚡ VRAM         : {hw['gpu_vram_gb']} GB")
        print(f"  ⚡ dtype        : float16 (GPU mode)")
    else:
        print(f"  💻 Device       : CPU ({hw['cpu_cores']} cores)")
        print(f"  💻 dtype        : float32")
        print(f"  💡 Tip          : GGUF models run 4-8x faster on CPU")
        print(f"                    See README for download instructions")

    print(f"  🔧 PyTorch      : {hw['torch_version']}")
    print(f"  🔧 Python       : {sys.version.split()[0]}")
    print()


# ── Global hardware config (importable by other modules) ─────────────────────
HW = detect_hardware()


def start():
    print_startup_banner(HW)

    try:
        from .studio.app import run_studio
        run_studio(hw=HW)
    except ImportError as e:
        print(f"❌ Could not load studio module: {e}")
        print("   Make sure studio/__init__.py exists.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Startup error: {e}")
        raise


if __name__ == "__main__":
    start()