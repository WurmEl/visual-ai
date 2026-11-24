from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image

from .model import IMAGE_EXTENSIONS


def _image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_label(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if parts.intersection({"positive", "pos", "high", "good"}):
        return "positive"
    if parts.intersection({"negative", "neg", "low", "bad"}):
        return "negative"
    return None


def infer_split(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "test" in parts:
        return "test"
    if "train" in parts:
        return "train"
    if "val" in parts or "validation" in parts:
        return "validation"
    return "unspecified"


def build_manifest(dataset_root: Path, manifest_path: Path) -> dict[str, int]:
    rows: list[dict[str, str | int]] = []
    seen_hashes: dict[str, str] = {}
    invalid = 0
    duplicates = 0

    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = infer_label(path)
        if label is None:
            continue
        relative = path.relative_to(dataset_root)
        try:
            with Image.open(path) as image:
                image.verify()
            file_hash = _image_hash(path)
        except Exception:
            invalid += 1
            continue
        duplicate_of = seen_hashes.get(file_hash, "")
        if duplicate_of:
            duplicates += 1
        else:
            seen_hashes[file_hash] = relative.as_posix()
        rows.append(
            {
                "path": relative.as_posix(),
                "label": label,
                "split": infer_split(path),
                "sha256": file_hash,
                "duplicate_of": duplicate_of,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["path", "label", "split", "sha256", "duplicate_of"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return {
        "images": len(rows),
        "positive": sum(row["label"] == "positive" for row in rows),
        "negative": sum(row["label"] == "negative" for row in rows),
        "invalid": invalid,
        "duplicates": duplicates,
    }
