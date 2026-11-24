from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fgaesq_project.paths import default_checkpoint


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {default_checkpoint()} ({'present' if default_checkpoint().is_file() else 'missing'})")
    gpd_root = ROOT / "data" / "gpd"
    print(f"GPD directory: {gpd_root} ({'present' if gpd_root.is_dir() else 'missing'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
