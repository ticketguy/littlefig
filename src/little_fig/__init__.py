import sys
import torch
import os

def start():
    print("🍐 Little Fig (CPU Edition) Starting...")
    print(f"Platform: {sys.platform} | Torch: {torch.__version__}")
    
    # Force CPU environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FORCE_CPU"] = "1"

    try:
        from .studio.app import run_studio
        run_studio()
    except ImportError as e:
        print(f"Error: Could not find internal modules. {e}")

if __name__ == "__main__":
    start()
