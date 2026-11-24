from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fgaesq_project.model import IMAGE_EXTENSIONS, infer_image, load_model, resolve_device
from fgaesq_project.paths import default_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify FGAesQ scores and visual features.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    images = (
        [input_path]
        if input_path.is_file()
        else sorted(p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    )
    if not images:
        raise RuntimeError(f"No supported images found under {input_path}")

    device = resolve_device(args.device)
    model = load_model(args.checkpoint.resolve(), device)
    expected_dimension = getattr(model, "feature_dim", None)
    for image_path in images:
        score, features = infer_image(model, image_path, device, return_features=True)
        if features is None or features.ndim != 1 or not math.isfinite(score):
            raise RuntimeError(f"Invalid output for {image_path}")
        if expected_dimension is not None and features.shape[0] != expected_dimension:
            raise RuntimeError(
                f"Expected {expected_dimension} features, got {tuple(features.shape)}"
            )
        print(
            f"{image_path}: score={score:.6f}, feature_shape={tuple(features.shape)}, "
            f"device={device}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
