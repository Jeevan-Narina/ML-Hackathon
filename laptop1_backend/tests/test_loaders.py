from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathsnap.loaders import (DataFormatError, TripFiles, discover_trips, iter_trips,
                             load_arcs, load_nodes, load_route, load_track, load_trip)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_track_whitespace_dtype_order_and_index(tmp_path):
    track = load_track(write(tmp_path, "x.track", "8.5\t49.2 3.0\n 8.6  49.3\t1.0\n"))
    assert list(track.columns) == ["longitude", "latitude", "timestamp_s"]
    assert track.index.name == "observation_id"
    assert track.index.tolist() == [0, 1]
    assert track.timestamp_s.tolist() == [3.0, 1.0]  # Must not sort source order.
    assert all(dtype == np.dtype("float64") for dtype in track.dtypes)


def test_nodes_arcs_routes_keep_zero_based_ids_and_repeats(tmp_path):
    nodes = load_nodes(write(tmp_path, "x.nodes", "8 49\n8.1 49.1\n8.2 49.2\n"))
    arcs = load_arcs(write(tmp_path, "x.arcs", "2 0\n0 1\n1 2\n"))
    route = load_route(write(tmp_path, "x.route", "2\n0\n1\n2\n"))
    assert nodes.index.name == "node_id"
    assert arcs.index.name == "arc_id"
    assert route.index.name == "route_position"
    assert arcs.loc[0].tolist() == [2, 0]
    assert route.arc_id.tolist() == [2, 0, 1, 2]
    assert all(dtype == np.dtype("int64") for dtype in arcs.dtypes)
    assert route.arc_id.dtype == np.dtype("int64")


@pytest.mark.parametrize("kind,text", [
    ("track", "8 49\n"), ("nodes", "8 49 extra\n"),
    ("arcs", "0 1.0\n"), ("route", "nan\n"),
    ("track", "8 invalid 2\n"), ("arcs", "0 9223372036854775808\n"),
])
def test_malformed_records_report_path_and_line(tmp_path, kind, text):
    loader = {"track": load_track, "nodes": load_nodes, "arcs": load_arcs, "route": load_route}[kind]
    path = write(tmp_path, f"broken.{kind}", text)
    with pytest.raises(DataFormatError, match="line 1") as error:
        loader(path)
    assert error.value.path == path
    assert error.value.issues[0].line == 1


def test_blank_node_record_rejects_file_instead_of_shifting_ids(tmp_path):
    with pytest.raises(DataFormatError, match="line 2") as error:
        load_nodes(write(tmp_path, "x.nodes", "8 49\n\n9 50\n"))
    assert error.value.record_count == 3


def test_all_malformed_lines_collected(tmp_path):
    with pytest.raises(DataFormatError) as error:
        load_track(write(tmp_path, "x.track", "8 49 1\n8\n8 invalid 3\n"))
    assert [issue.line for issue in error.value.issues] == [2, 3]


def test_nan_float_records_retained_for_quality_audit(tmp_path):
    track = load_track(write(tmp_path, "x.track", "NaN 49 1\n8 NA 2\n8 49 null\n"))
    assert len(track) == 3 and track.isna().sum().sum() == 3


def test_empty_and_missing_file(tmp_path):
    with pytest.raises(DataFormatError, match="empty file"):
        load_route(write(tmp_path, "empty.route", ""))
    with pytest.raises(FileNotFoundError):
        load_track(tmp_path / "missing.track")


def make_trip(raw, split, trip_id):
    directory = raw / split / trip_id
    directory.mkdir(parents=True)
    for ext, text in {"track": "8 49 1\n8.01 49 2\n", "nodes": "8 49\n8.01 49\n",
                      "arcs": "0 1\n1 0\n", "route": "0\n1\n"}.items():
        (directory / f"{trip_id}.{ext}").write_text(text)
    return TripFiles(split, trip_id, directory)


def test_explicit_partition_iteration_and_complete_trip(tmp_path):
    make_trip(tmp_path, "train", "00000001")
    make_trip(tmp_path, "student_test", "00000002")
    training = list(iter_trips(tmp_path, "train"))
    assert len(training) == 1 and training[0].files.split == "train"
    assert len(training[0].track) == 2 and len(training[0].route) == 2
    assert len(discover_trips(tmp_path)) == 2
    with pytest.raises(ValueError, match="Split must"):
        discover_trips(tmp_path, "evaluator_test")


def test_overlapping_trip_ids_rejected_even_when_selecting_train(tmp_path):
    make_trip(tmp_path, "train", "00000001")
    make_trip(tmp_path, "student_test", "00000001")
    with pytest.raises(ValueError, match="both dataset splits"):
        discover_trips(tmp_path, "train")


def test_missing_partition_and_unexpected_raw_entry(tmp_path):
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="partition"):
        discover_trips(tmp_path)
    (tmp_path / "student_test").mkdir()
    (tmp_path / "evaluator_test").mkdir()
    with pytest.raises(ValueError, match="Unexpected entries"):
        discover_trips(tmp_path)


def test_missing_trip_companion_informative(tmp_path):
    files = make_trip(tmp_path, "train", "00000001")
    files.path("arcs").unlink()
    with pytest.raises(FileNotFoundError, match="00000001.arcs"):
        load_trip(files)
