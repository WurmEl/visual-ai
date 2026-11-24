from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
RESULTS_DIR = PROJECT_ROOT / "results"


def default_checkpoint() -> Path:
    return CHECKPOINT_DIR / "FGAesQ.pt"
