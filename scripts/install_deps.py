"""
Little Fig — Dependency installer
Run this ONCE before pip install -e .

Handles the two packages that can't be auto-installed safely:
  - torch         : default pip wheel is CUDA (~2.5GB), we want CPU (~180MB)
  - llama-cpp-python : compiles C++ code, needs CUDA flag set explicitly

Usage:
    python scripts/install_deps.py           # auto-detect
    python scripts/install_deps.py --cpu     # force CPU builds
    python scripts/install_deps.py --gpu     # force GPU builds (if you have one)
"""

import subprocess
import sys
import os


def run(cmd: list, env=None):
    print(f"   > {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\n❌ Command failed. See error above.")
        sys.exit(1)


def has_cuda() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def install_cpu():
    print("── Installing torch (CPU-only, ~180MB) ─────────────────────")
    run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu",
    ])

    print("\n── Installing Little Fig core ───────────────────────────────")
    run([sys.executable, "-m", "pip", "install", "-e", "."])

    print("\n── Installing llama-cpp-python (CPU-only, pre-built) ────────")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "llama-cpp-python",
        "--extra-index-url",
        "https://abetlen.github.io/llama-cpp-python/whl/cpu",
        "--prefer-binary",
    ])
    if result.returncode != 0:
        print("   Pre-built wheel unavailable. Building from source (CPU)...")
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=off -DLLAMA_METAL=off"
        run([
            sys.executable, "-m", "pip", "install",
            "llama-cpp-python", "--no-cache-dir",
        ], env=env)


def install_gpu():
    print("── Installing torch (CUDA) ──────────────────────────────────")
    print("   Detecting CUDA version...")
    # Default torch CUDA build — matches most CUDA 11.8/12.x setups
    run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])

    print("\n── Installing Little Fig core ───────────────────────────────")
    run([sys.executable, "-m", "pip", "install", "-e", "."])

    print("\n── Installing llama-cpp-python (CUDA) ───────────────────────")
    env = os.environ.copy()
    env["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=on"
    env["FORCE_CMAKE"] = "1"
    run([
        sys.executable, "-m", "pip", "install",
        "llama-cpp-python", "--no-cache-dir",
    ], env=env)


def install_train_deps():
    print("\n── Installing training deps (peft, datasets, trl) ───────────")
    run([
        sys.executable, "-m", "pip", "install",
        "peft>=0.9.0", "datasets>=2.18.0", "trl>=0.8.0",
    ])


if __name__ == "__main__":
    print("🍐 Little Fig — Dependency Installer\n")

    force_cpu = "--cpu" in sys.argv
    force_gpu = "--gpu" in sys.argv
    with_train = "--train" in sys.argv

    cuda = has_cuda()
    print(f"   CUDA detected : {'Yes' if cuda else 'No'}")

    if force_cpu:
        print("   Mode          : CPU (forced)\n")
        install_cpu()
    elif force_gpu or cuda:
        print("   Mode          : GPU (CUDA)\n")
        install_gpu()
    else:
        print("   Mode          : CPU (auto)\n")
        install_cpu()

    if with_train:
        install_train_deps()

    print("\n✓ All done. Run: little-fig")
    print("\nVerify torch:")
    print("  python -c \"import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())\"")