from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import shapely

from pathsnap.candidates import CandidateConfig, generate_candidates
from pathsnap.projection import MetricProjection
from pathsnap.road_network import RoadNetwork
from pathsnap.step2 import validate_candidate_batch


def fixture_network():
    projection = MetricProjection(8, 49)
    # Two long parallel segments; one duplicate and one reverse-direction arc.
    lon, lat = projection.lonlat(np.array([-1000, 1000, -1000, 1000]), np.array([0, 0, 30, 30]))
    nodes = pd.DataFrame({"longitude": lon, "latitude": lat})
    arcs = pd.DataFrame({"from_node_id": [0, 1, 0, 2], "to_node_id": [1, 0, 1, 3]},
                        index=pd.Index([2, 9, 15, 23], name="arc_id"))
    return RoadNetwork(nodes, arcs)


def observations_at(network, xy, timestamps=None, indices=None):
    lon, lat = MetricProjection(8, 49).lonlat(np.array(xy)[:,0], np.array(xy)[:,1])
    return pd.DataFrame({"longitude": lon, "latitude": lat,
                         "timestamp_s": timestamps if timestamps is not None else np.arange(len(xy), dtype=float)},
                        index=pd.Index(indices if indices is not None else np.arange(len(xy)), name="observation_id"))


def generate(network, track, **kwargs):
    config = CandidateConfig(**kwargs)
    batches = list(generate_candidates(network, track, "train", "00000001", config))
    candidates = pd.concat([c for c,_ in batches], ignore_index=True)
    statuses = pd.concat([o for _,o in batches], ignore_index=True)
    validate_candidate_batch(network, candidates, config)
    return candidates, statuses


def test_distances_ranks_original_ids_and_multiple_candidates():
    network = fixture_network()
    track = observations_at(network, [[-900, 5]], indices=[42])
    candidates, status = generate(network, track, search_radius_m=40, fallback_radius_m=None)
    assert len(candidates) == 4 and status.candidate_count.iloc[0] == 4
    assert candidates.gps_observation_index.tolist() == [42]*4
    assert set(candidates.candidate_arc_id) == {2,9,15,23}
    assert candidates.distance_to_arc_m.min() == pytest.approx(5, abs=.01)
    assert candidates.candidate_rank.tolist() == [1,2,3,4]
    assert candidates.distance_to_arc_m.is_monotonic_increasing
    assert not candidates.fallback_used.any()
    assert "ground_truth" not in candidates.columns


def test_long_segment_end_is_retrieved_without_midpoint_shortcut():
    network = fixture_network()
    track = observations_at(network, [[-999, 1]])
    candidates, _ = generate(network, track, search_radius_m=5, fallback_radius_m=None)
    assert set(candidates.candidate_arc_id) == {2,9,15}
    assert candidates.distance_to_arc_m.max() < 2


def test_maximum_candidate_limit_and_tie_order():
    network = fixture_network()
    track = observations_at(network, [[0,5]])
    candidates, status = generate(network, track, max_candidates=2)
    assert len(candidates) == 2 and status.candidate_count.iloc[0] == 2
    assert status.candidates_before_limit.iloc[0] == 4 and status.candidate_limit_applied.iloc[0]
    # Identical-direction arcs have exactly identical geometry/distance, sorted by ID.
    full, _ = generate(network, track)
    duplicates = full.loc[full.candidate_arc_id.isin([2,15])]
    assert duplicates.candidate_arc_id.tolist() == [2,15]


def test_no_candidates_and_bounded_fallback():
    network = fixture_network()
    track = observations_at(network, [[0,100], [0,300]])
    candidates, status = generate(network, track)
    assert status.status.tolist() == ["matched_fallback", "unmatched"]
    assert status.fallback_attempted.tolist() == [True,True]
    assert status.fallback_used.tolist() == [True,False]
    assert candidates.fallback_used.all() and candidates.distance_to_arc_m.max() <= 150
    assert set(candidates.gps_observation_index) == {0}
    disabled, disabled_status = generate(network, track, fallback_radius_m=None)
    assert disabled.empty and disabled_status.status.eq("unmatched").all()


def test_primary_candidates_do_not_get_extra_fallback_candidates():
    network = fixture_network()
    track = observations_at(network, [[0,-25]])
    candidates, status = generate(network, track, search_radius_m=30)
    assert set(candidates.candidate_arc_id) == {2,9,15}
    assert not status.fallback_attempted.iloc[0]


def test_closest_point_fraction_is_directed_not_vehicle_position():
    network = fixture_network()
    track = observations_at(network, [[-500,5]])
    candidates, _ = generate(network, track)
    forward = candidates.set_index("candidate_arc_id").loc[2]
    reverse = candidates.set_index("candidate_arc_id").loc[9]
    assert forward.position_along_arc == pytest.approx(.25, abs=.001)
    assert reverse.position_along_arc == pytest.approx(.75, abs=.001)
    assert forward.projected_longitude == pytest.approx(reverse.projected_longitude, abs=1e-9)
    assert forward.projected_latitude == pytest.approx(reverse.projected_latitude, abs=1e-9)


def test_invalid_far_coordinates_and_nonfinite_timestamps_preserved():
    network = fixture_network()
    track = observations_at(network, [[0,5],[0,5],[0,5],[0,5]], timestamps=[1,153065,np.nan,2], indices=[3,7,11,20])
    track.loc[11,"longitude"] = np.nan
    track.loc[20,["longitude","latitude"]] = [100,0]
    _, status = generate(network, track, batch_size=1)
    assert status.gps_observation_index.tolist() == [3,7,11,20]
    assert status.loc[1,"dt_previous_s"] == 153064 and status.loc[1,"gap_gt_5s"]
    assert status.status.tolist() == ["matched_radius","matched_radius","invalid_coordinates","projection_out_of_extent"]
    assert status.loc[2,"timestamp_nonfinite"]
    assert status.previous_observation_index.tolist()[1:] == [3,7,11]


def test_bad_timestamp_does_not_suppress_spatial_candidates():
    network = fixture_network()
    track = observations_at(network, [[0,5],[0,5],[0,5]], timestamps=[2,1,np.nan])
    candidates, status = generate(network, track, batch_size=1)
    assert (status.candidate_count > 0).all()
    assert status.nonincreasing_timestamp.tolist() == [False,True,False]
    assert candidates.gps_observation_index.nunique() == 3


def test_partition_validation_and_deterministic_batching():
    network = fixture_network()
    track = observations_at(network, [[-800,3],[0,120],[900,0],[0,300]])
    c1, o1 = generate(network, track, batch_size=1)
    c2, o2 = generate(network, track, batch_size=3)
    pd.testing.assert_frame_equal(c1,c2)
    pd.testing.assert_frame_equal(o1,o2)
    test_rows = list(generate_candidates(network, track, "student_test", "00000007"))[0][0]
    assert test_rows.dataset_partition.eq("student_test").all() and test_rows.trip_id.eq("00000007").all()
    with pytest.raises(ValueError, match="partition"):
        list(generate_candidates(network, track, "evaluator_test", "00000007"))


def test_strtree_matches_independent_brute_force_on_small_fixture():
    network = fixture_network()
    rng = np.random.default_rng(42)
    xy = np.column_stack([rng.uniform(-1200,1200,40),rng.uniform(-100,150,40)])
    track = observations_at(network, xy)
    candidates, _ = generate(network, track, search_radius_m=60, fallback_radius_m=None, max_candidates=10)
    px,py = network.projection.xy(track.longitude.to_numpy(), track.latitude.to_numpy())
    for i,(x,y) in enumerate(zip(px,py)):
        distances = np.array([shapely.Point(x,y).distance(line) for line in network.geometries])
        ids = network.index_arc_ids[distances<=60]
        actual = candidates.loc[candidates.gps_observation_index==i]
        assert set(actual.candidate_arc_id)==set(ids)
        for row in actual.itertuples():
            position=np.flatnonzero(network.index_arc_ids==row.candidate_arc_id)[0]
            assert row.distance_to_arc_m==pytest.approx(distances[position],abs=1e-9)


@pytest.mark.parametrize("kwargs", [{"search_radius_m":0},{"max_candidates":0},{"max_candidates":1.5},
    {"batch_size":0},{"fallback_radius_m":40},{"search_radius_m":float('nan')},{"max_projection_scale_error":-1}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        CandidateConfig(**kwargs)
