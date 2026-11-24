from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fgaesq_project.model import IMAGE_EXTENSIONS, infer_image, load_model, resolve_device
from fgaesq_project.paths import default_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score images with pretrained FGAesQ.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    return parser.parse_args()


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    device = resolve_device(args.device)
    images = collect_images(input_path)
    if not images:
        raise RuntimeError(f"No supported images found under {input_path}")

    model = load_model(args.checkpoint.resolve(), device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "score", "error"])
        writer.writeheader()
        for image_path in images:
            try:
                score, _ = infer_image(model, image_path, device)
                relative_image = (
                    image_path.name
                    if input_path.is_file()
                    else image_path.relative_to(input_path).as_posix()
                )
                writer.writerow(
                    {"image": relative_image, "score": score, "error": ""}
                )
            except Exception as error:
                relative_image = (
                    image_path.name
                    if input_path.is_file()
                    else image_path.relative_to(input_path).as_posix()
                )
                writer.writerow(
                    {
                        "image": relative_image,
                        "score": "",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    print(f"Scored {len(images)} image(s) on {device}; wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
