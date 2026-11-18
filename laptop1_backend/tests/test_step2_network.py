import numpy as np
import pandas as pd
import pytest

from pathsnap.projection import GEOD, MetricProjection, geodesic_point_segment_m
from pathsnap.road_network import RoadNetwork


def tables():
    nodes = pd.DataFrame({"longitude": [8, 8.001, 8.002, 8.001], "latitude": [49, 49, 49, 49.001]},
                         index=pd.Index([0, 1, 2, 3], name="node_id"))
    arcs = pd.DataFrame({"from_node_id": [0, 1, 0, 1], "to_node_id": [1, 0, 1, 3]},
                        index=pd.Index([0, 1, 7, 12], name="arc_id"))
    return nodes, arcs


def test_directed_graph_parallel_edges_original_ids_and_lazy_index():
    network = RoadNetwork(*tables())
    graph = network.to_multidigraph()
    assert set(graph.nodes) == {0, 1, 2, 3}
    assert graph.has_edge(0, 1, key=0) and graph.has_edge(0, 1, key=7)
    assert graph.has_edge(1, 0, key=1) and not graph.has_edge(3, 1)
    assert graph.number_of_edges() == 4
    assert network.arc(12).from_node_id == 1
    assert set(network.index_arc_ids) == {0, 1, 7, 12}
    assert network.summary()["duplicate_endpoint_pairs"] == 1
    assert network._tree is None
    assert network.build_spatial_index() is network.build_spatial_index()
    with pytest.raises(KeyError):
        network.arc(2)  # Arc ID is not its position in the indexed geometry array.


@pytest.mark.parametrize("longitude,latitude", [(8,49),(-93,45),(137,36),(151,-34),(179.999,10),(0,85)])
def test_metric_roundtrip_and_independent_geodesic(longitude, latitude):
    nodes = pd.DataFrame({"longitude": [longitude, longitude+.0001], "latitude": [latitude, latitude+.0001]})
    projection = MetricProjection.from_nodes(nodes)
    x, y = projection.xy(nodes.longitude.to_numpy(), nodes.latitude.to_numpy())
    lon, lat = projection.lonlat(x, y)
    np.testing.assert_allclose(lon, nodes.longitude, atol=1e-9)
    np.testing.assert_allclose(lat, nodes.latitude, atol=1e-9)
    geodesic = GEOD.inv(*nodes.iloc[0], *nodes.iloc[1])[2]
    assert np.hypot(x[1]-x[0], y[1]-y[0]) == pytest.approx(geodesic, rel=1e-5)


def test_dateline_center_and_projection_extent_guard():
    local = pd.DataFrame({"longitude": [179.99, -179.99], "latitude": [10, 10]})
    projection = MetricProjection.from_nodes(local)
    assert abs(projection.center_longitude) > 179
    assert projection.network_extent_m < 2000
    worldwide = pd.DataFrame({"longitude": [0, 100], "latitude": [0, 50]})
    with pytest.raises(ValueError, match="extent"):
        MetricProjection.from_nodes(worldwide)


def test_length_bearing_and_all_arc_projection_checks():
    network = RoadNetwork(*tables())
    assert network.arc(0).arc_bearing_deg == pytest.approx(90, abs=.001)
    assert network.arc(1).arc_bearing_deg == pytest.approx(270, abs=.001)
    assert network.arc(12).arc_bearing_deg == pytest.approx(0, abs=.001)
    assert 70 < network.arc(0).arc_length_m < 75
    assert network.max_length_scale_error < 1e-5
    assert len(network.projection_checks) == 4


def test_invalid_and_degenerate_arcs_retained_but_not_spatially_indexed():
    nodes, arcs = tables()
    nodes.loc[5] = [np.nan, 49]
    arcs.loc[20] = [0, 0]
    arcs.loc[21] = [0, 999]
    arcs.loc[22] = [0, 5]
    network = RoadNetwork(nodes, arcs)
    assert len(network.arcs) == 7 and len(network.index_arc_ids) == 4
    assert network.arc(20).geometry_status == "degenerate_segment"
    assert network.arc(21).geometry_status == "invalid_node_reference"
    assert network.arc(22).geometry_status == "invalid_node_coordinates"
    assert np.isnan(network.arc(20).arc_bearing_deg)
    graph = network.to_multidigraph()
    assert graph.has_edge(0, 0, key=20)  # Zero-cost connectors retain directional topology.
    assert graph[0][0][20]["length_m"] == 0


def test_empty_index_is_supported():
    nodes, _ = tables()
    arcs = pd.DataFrame({"from_node_id": [0], "to_node_id": [0]}, index=pd.Index([9], name="arc_id"))
    network = RoadNetwork(nodes, arcs)
    assert len(network.geometries) == 0
    assert len(network.build_spatial_index().geometries) == 0


def test_independent_geodesic_point_to_segment_minimisation():
    # Equatorial segment 0..0.02 longitude; the closest point is (0.01, 0).
    expected = GEOD.inv(.01, .001, .01, 0)[2]
    assert geodesic_point_segment_m(.01, .001, 0, 0, .02, 0) == pytest.approx(expected, abs=1e-5)
