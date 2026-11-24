from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fgaesq_project.gpd import build_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Index a downloaded GPD directory.")
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "gpd")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "gpd_manifest.csv")
    args = parser.parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(
            f"GPD directory not found: {args.root}. Download the partitions first; "
            "see README.md for the official links."
        )
    summary = build_manifest(args.root.resolve(), args.output.resolve())
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
