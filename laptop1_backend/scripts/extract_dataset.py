"""Safely extract supplied data, preserving bytes and recording SHA-256 evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.extraction import extract_dataset


if __name__ == "__main__":
    backend = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=backend.parents[1] / "Student folder.zip")
    parser.add_argument("--destination", type=Path, default=backend / "data/raw")
    parser.add_argument("--manifest", type=Path, default=backend / "outputs/eda/source_manifest.csv")
    args = parser.parse_args()
    count = extract_dataset(args.archive, args.destination, args.manifest)
    print(f"Verified/extracted {count} files to {args.destination}")
