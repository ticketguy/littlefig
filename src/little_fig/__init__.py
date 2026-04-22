"""
Little Fig — Entry Point
Detects hardware at startup and configures the runtime accordingly.
"""

import sys
import os


def detect_hardware():
    """
    Detect available hardware and return a config dict.
    This runs at startup before anything else.
    """
    import psutil

    hw = {
        "device": "cpu",
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "cpu_cores": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 1),
        "recommended_dtype": "float32",
        "recommended_backend": "fig_engine",
    }

    try:
        import torch
        hw["torch_version"] = torch.__version__

        if torch.cuda.is_available():
            hw["gpu_available"] = True
            hw["device"] = "cuda"
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            hw["gpu_vram_gb"] = round(vram, 1)
            hw["recommended_dtype"] = "float16"

            if vram >= 16:
                hw["model_tier"] = "large"
            elif vram >= 8:
                hw["model_tier"] = "medium"
            else:
                hw["model_tier"] = "small"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            hw["model_tier"] = "cpu"
    except ImportError:
        hw["torch_version"] = "not installed"
        hw["model_tier"] = "cpu"

    return hw


def print_startup_banner(hw: dict):
    fig = """
  ╔══════════════════════════════════════════╗
  ║           🍐  L I T T L E  F I G        ║
  ║     CPU-native LLM engine  v0.4.0       ║
  ║         Powered by Fig Engine           ║
  ╚══════════════════════════════════════════╝"""
    print(fig)

    if hw.get("gpu_available"):
        print(f"  ⚡ GPU detected : {hw['gpu_name']}")
        print(f"  ⚡ VRAM         : {hw['gpu_vram_gb']} GB")
    else:
        print(f"  💻 Device       : CPU ({hw['cpu_cores']} cores)")
        print(f"  💻 RAM          : {hw.get('ram_available_gb', '?')} GB available / {hw.get('ram_total_gb', '?')} GB total")

    print(f"  🔧 PyTorch      : {hw.get('torch_version', 'N/A')}")
    print(f"  🔧 Python       : {sys.version.split()[0]}")
    print(f"  🔧 Backend      : Fig Engine (INT4 streaming)")
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
        print("   Install with: pip install -e '.[full]'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Startup error: {e}")
        raise


if __name__ == "__main__":
    start()
