import numpy as np
import pandas as pd
import pytest

from pathsnap.quality import Thresholds, gps_quality, haversine_m, network_quality, valid_coordinates


def track(rows):
    return pd.DataFrame(rows, columns=["longitude", "latitude", "timestamp_s"], dtype="float64")


def test_haversine_known_distances_symmetry_and_antimeridian():
    # One degree along the equator is approximately 111.195 km, not one metre.
    assert haversine_m(0, 0, 1, 0) == pytest.approx(111_195.08, abs=.1)
    assert haversine_m(8, 49, 8, 49) == 0
    assert haversine_m(179.9, 0, -179.9, 0) == pytest.approx(22_239.016, abs=.1)
    assert haversine_m(8, 49, 9, 50) == pytest.approx(haversine_m(9, 50, 8, 49))
    assert np.isfinite(haversine_m(0, 0, 180, 0))


def test_coordinate_validation_missing_infinite_and_ranges():
    frame = track([[8, 49, 1], [181, 49, 2], [8, -91, 3], [np.nan, 49, 4], [8, np.inf, 5], [-180, 90, 6]])
    assert valid_coordinates(frame).tolist() == [True, False, False, False, False, True]
    stats, _ = gps_quality(frame)
    assert stats["gps_invalid_coordinates"] == 4
    assert stats["gps_missing_cells"] == 1


def test_timestamps_unsorted_duplicate_nonfinite_and_gap():
    frame = track([[8, 49, t] for t in [1, 2, 2, 1, 8, np.nan, np.inf]])
    stats, steps = gps_quality(frame)
    assert stats["timestamp_backwards"] == 1
    assert stats["timestamp_equal_adjacent"] == 1
    assert stats["timestamp_duplicate_values"] == 2
    assert stats["timestamp_nonfinite"] == 2
    assert stats["gaps_gt_5s"] == 1
    assert not stats["timestamp_strictly_increasing"]
    assert np.isnan(steps.loc[1, "speed_kmh"])
    assert np.isnan(steps.loc[2, "speed_kmh"])
    assert steps.interval_s.iloc[:4].tolist() == [1, 0, -1, 7]


def test_diagnostic_thresholds_strict_and_source_not_modified():
    frame = track([[0, 0, 1], [.001, 0, 6], [.002, 0, 12]])
    before = frame.copy(deep=True)
    stats, steps = gps_quality(frame, Thresholds(speed_kmh=50, jump_m=100))
    assert stats["gaps_gt_5s"] == 1  # Exactly 5 seconds is not > 5.
    assert stats["distance_jumps"] == 2
    assert stats["speed_outliers"] == 2
    pd.testing.assert_frame_equal(frame, before)


def test_invalid_coordinate_steps_are_not_bridged():
    frame = track([[0, 0, 1], [np.nan, 0, 2], [.01, 0, 3]])
    stats, steps = gps_quality(frame)
    assert stats["unmeasurable_steps"] == 2
    assert stats["trajectory_distance_m"] == 0
    assert steps.distance_m.isna().all()


def test_single_observation_and_stationary_duplicates():
    stats, steps = gps_quality(track([[8, 49, 1]]))
    assert steps.empty and stats["timestamp_strictly_increasing"]
    stats, _ = gps_quality(track([[8, 49, 1], [8, 49, 2], [8, 49, 2]]))
    assert stats["gps_duplicate_records"] == 1 and stats["gps_repeated_coordinates"] == 2


def network():
    nodes = pd.DataFrame([[0, 0], [.01, 0], [.02, 0], [1, 1]], columns=["longitude", "latitude"])
    nodes.index.name = "node_id"
    arcs = pd.DataFrame([[0, 1], [1, 2], [2, 1], [0, 1], [2, 99]], columns=["from_node_id", "to_node_id"], dtype="int64")
    arcs.index.name = "arc_id"
    return nodes, arcs


def test_directed_connectivity_invalid_references_parallel_arcs_and_isolates():
    nodes, arcs = network()
    route = pd.DataFrame({"arc_id": [0, 1, 2]}, dtype="int64")
    stats, details = network_quality(nodes, arcs, route)
    assert stats["directed_arcs"] == 5
    assert stats["invalid_node_reference_arcs"] == 1
    assert details["invalid_node_reference_arcs"] == [4]
    assert stats["duplicate_directed_arcs"] == 1
    assert stats["isolated_nodes"] == 1 and details["isolated_nodes"] == [3]
    assert stats["weak_components"] == 2 and stats["strong_components"] == 3
    assert stats["largest_weak_component_fraction"] == .75
    assert stats["route_disconnected_pairs"] == 0
    assert stats["reference_route_distance_m"] == pytest.approx(3335.8524, abs=.1)


def test_reference_breaks_invalid_ids_and_uncheckable_pairs():
    nodes, arcs = network()
    # 0->1 followed by 2->1 is disconnected; invalid arc 99 is not interpreted as a row.
    route = pd.DataFrame({"arc_id": [0, 2, 99, 4, -1]}, dtype="int64")
    stats, details = network_quality(nodes, arcs, route)
    assert stats["route_invalid_arc_ids"] == 2
    assert stats["route_invalid_endpoint_arcs"] == 1
    assert stats["route_disconnected_pairs"] == 1
    assert details["route_disconnected_pairs"] == [1]
    assert stats["route_uncheckable_pairs"] == 3
    assert details["route_invalid_arc_ids"] == [2, 4]


def test_route_duplicate_arc_is_not_silently_deduplicated():
    nodes, arcs = network()
    stats, _ = network_quality(nodes, arcs, pd.DataFrame({"arc_id": [0, 0]}))
    assert stats["reference_route_arcs"] == 2
    assert stats["route_disconnected_pairs"] == 1


def test_invalid_network_coordinates_reported_without_fabricating_route_length():
    nodes, arcs = network()
    nodes.loc[0, "longitude"] = np.inf
    with np.errstate(invalid="raise"):
        stats, details = network_quality(nodes, arcs, pd.DataFrame({"arc_id": [0]}))
    assert stats["node_invalid_coordinates"] == 1
    assert details["node_invalid_coordinates"] == [0]
    assert stats["route_unmeasurable_arcs"] == 1
    assert stats["reference_route_distance_m"] == 0
    assert stats["network_nodes"] == 4  # Topological IDs remain intact.
