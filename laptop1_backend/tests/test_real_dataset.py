"""Integration checks against supplied files; extraction is a documented prerequisite."""
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from pathsnap.eda import verify_manifest
from pathsnap.loaders import EXTENSIONS, discover_trips, load_trip

BACKEND = Path(__file__).resolve().parents[1]
RAW = BACKEND / "data/raw"


def test_official_partition_counts_and_companion_files():
    trips = discover_trips(RAW)
    train = {f.trip_id for f in trips if f.split == "train"}
    student = {f.trip_id for f in trips if f.split == "student_test"}
    assert len(train) == 80 and len(student) == 10
    assert train.isdisjoint(student)
    assert all(f.path(ext).is_file() for f in trips for ext in EXTENSIONS)


def test_all_extracted_bytes_match_archive_and_manifest():
    manifest = BACKEND / "outputs/eda/source_manifest.csv"
    verify_manifest(RAW, manifest)
    # Re-check against the original archive, not just the generated manifest.
    with ZipFile(BACKEND.parents[1] / "Student folder.zip") as archive:
        for row in pd.read_csv(manifest).itertuples():
            assert (RAW / row.relative_path).read_bytes() == archive.read(row.archive_member)


def test_real_trips_load_both_splits_and_resolve_row_ids():
    for split in ("train", "student_test"):
        files = discover_trips(RAW, split)[0]
        trip = load_trip(files)
        assert trip.track.index[0] == 0 and trip.nodes.index[0] == 0 and trip.arcs.index[0] == 0
        assert trip.arcs.from_node_id.isin(trip.nodes.index).all()
        assert trip.arcs.to_node_id.isin(trip.nodes.index).all()
        assert trip.route.arc_id.isin(trip.arcs.index).all()
        assert len(trip.track) != len(trip.route)  # These supplied sequences are not pointwise labels.
