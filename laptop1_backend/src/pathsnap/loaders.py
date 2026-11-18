"""Typed readers for the documented, headerless whitespace-separated source files.

IDs are physical zero-based row positions, never renumbered or inferred from
coordinates. Blank/malformed records reject the entire file, preventing ID drift.
Numerical NaN/NA/null values in floating-point fields are retained for auditing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator

import pandas as pd

SPLITS = ("train", "student_test")
EXTENSIONS = ("track", "nodes", "arcs", "route")
SCHEMAS = {
    "track": (("longitude", "latitude", "timestamp_s"), "observation_id", "float64"),
    "nodes": (("longitude", "latitude"), "node_id", "float64"),
    "arcs": (("from_node_id", "to_node_id"), "arc_id", "int64"),
    "route": (("arc_id",), "route_position", "int64"),
}


@dataclass(frozen=True)
class RecordIssue:
    line: int
    message: str


class DataFormatError(ValueError):
    def __init__(self, path: Path, issues: list[RecordIssue], record_count: int):
        self.path, self.issues, self.record_count = path, issues, record_count
        detail = "; ".join(f"line {i.line}: {i.message}" for i in issues[:5])
        super().__init__(f"{path}: {len(issues)} invalid record(s): {detail}")


def _load(path: Path, kind: str) -> pd.DataFrame:
    path = Path(path)
    columns, index_name, dtype = SCHEMAS[kind]
    rows, issues = [], []
    line_count = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_count, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != len(columns):
                issues.append(RecordIssue(line_count, f"expected {len(columns)} fields, got {len(fields)}"))
                continue
            try:
                if dtype == "int64":
                    if any(not re.fullmatch(r"[+-]?\d+", token) for token in fields):
                        raise ValueError("IDs must be integer literals, not floats or missing values")
                    row = [int(token) for token in fields]
                    if any(value < -(2**63) or value >= 2**63 for value in row):
                        raise ValueError("integer outside int64 range")
                else:
                    row = [float("nan") if token.lower() in {"na", "null"} else float(token) for token in fields]
                rows.append(row)
            except ValueError as exc:
                issues.append(RecordIssue(line_count, str(exc)))
    if line_count == 0:
        issues.append(RecordIssue(0, "empty file; expected at least one record"))
    if issues:
        raise DataFormatError(path, issues, line_count)
    result = pd.DataFrame(rows, columns=columns, dtype=dtype)
    result.index = pd.RangeIndex(len(result), name=index_name)
    return result


def load_track(path: Path) -> pd.DataFrame:
    return _load(path, "track")


def load_nodes(path: Path) -> pd.DataFrame:
    return _load(path, "nodes")


def load_arcs(path: Path) -> pd.DataFrame:
    return _load(path, "arcs")


def load_route(path: Path) -> pd.DataFrame:
    return _load(path, "route")


@dataclass(frozen=True)
class TripFiles:
    split: str
    trip_id: str
    directory: Path

    def path(self, kind: str) -> Path:
        if kind not in EXTENSIONS:
            raise ValueError(f"Unknown source type: {kind}")
        return self.directory / f"{self.trip_id}.{kind}"


@dataclass
class Trip:
    files: TripFiles
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: pd.DataFrame


def discover_trips(root: Path, split: str | None = None) -> list[TripFiles]:
    """Require explicit source splits; fail on misplaced or overlapping trips."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {root}. Run scripts/extract_dataset.py first.")
    if split is not None and split not in SPLITS:
        raise ValueError(f"Split must be one of {SPLITS}; evaluator data is unsupported in Step 1")
    unexpected = [p.name for p in root.iterdir() if p.name not in {*SPLITS, ".gitkeep"}]
    if unexpected:
        raise ValueError(f"Unexpected entries in raw dataset: {unexpected}; expected train/ and student_test/")
    trips, seen = [], set()
    for partition in SPLITS:
        folder = root / partition
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing dataset partition: {folder}")
        for directory in sorted(folder.iterdir()):
            if not directory.is_dir() or not re.fullmatch(r"\d{8}", directory.name):
                raise ValueError(f"Unexpected trip entry: {directory}")
            if directory.name in seen:
                raise ValueError(f"Trip ID {directory.name} appears in both dataset splits")
            seen.add(directory.name)
            if split is None or split == partition:
                trips.append(TripFiles(partition, directory.name, directory))
    return trips


def load_trip(files: TripFiles) -> Trip:
    return Trip(files, load_track(files.path("track")), load_nodes(files.path("nodes")),
                load_arcs(files.path("arcs")), load_route(files.path("route")))


def iter_trips(root: Path, split: str) -> Iterator[Trip]:
    """Explicit split is required to avoid accidentally feeding test data to training."""
    for files in discover_trips(root, split):
        yield load_trip(files)
