"""Arc-indexed directed network and reusable metric STRtree.

The full arc table retains excluded geometries with explicit reasons. Tree
positions are a separate dense index mapping to original arc IDs, never labels.
"""
from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from .projection import GEOD, MetricProjection
from .quality import valid_coordinates


class RoadNetwork:
    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame, *, max_extent_m=300_000.0,
                 max_scale_error=0.005, min_segment_m=1e-6):
        started = time.perf_counter()
        if not nodes.index.is_unique or not arcs.index.is_unique:
            raise ValueError("Node and arc IDs must be unique within each source table")
        if not np.isfinite(max_scale_error) or max_scale_error <= 0 or not np.isfinite(min_segment_m) or min_segment_m < 0:
            raise ValueError("Invalid projection tolerance or minimum segment length")
        self.nodes = nodes
        self.projection = MetricProjection.from_nodes(nodes, max_extent_m)
        valid_nodes = valid_coordinates(nodes)
        xy = np.full((len(nodes), 2), np.nan)
        xy[valid_nodes.to_numpy()] = np.column_stack(self.projection.xy(
            nodes.loc[valid_nodes].longitude.to_numpy(), nodes.loc[valid_nodes].latitude.to_numpy()))
        self.node_xy = pd.DataFrame(xy, index=nodes.index, columns=["x_m", "y_m"])
        table = arcs.copy()
        refs = arcs.from_node_id.isin(nodes.index) & arcs.to_node_id.isin(nodes.index)
        starts = self.node_xy.reindex(arcs.from_node_id).to_numpy()
        ends = self.node_xy.reindex(arcs.to_node_id).to_numpy()
        finite = np.isfinite(starts).all(axis=1) & np.isfinite(ends).all(axis=1)
        projected = np.linalg.norm(ends - starts, axis=1)
        length, bearing = np.full(len(arcs), np.nan), np.full(len(arcs), np.nan)
        positions = np.flatnonzero(refs.to_numpy() & finite)
        left = nodes.reindex(arcs.from_node_id.iloc[positions])
        right = nodes.reindex(arcs.to_node_id.iloc[positions])
        azimuth, _, geodesic = GEOD.inv(left.longitude.to_numpy(), left.latitude.to_numpy(),
                                       right.longitude.to_numpy(), right.latitude.to_numpy())
        length[positions] = geodesic
        bearing[positions] = np.mod(azimuth, 360)
        degenerate = finite & (projected <= min_segment_m)
        bearing[degenerate] = np.nan
        status = np.full(len(arcs), "valid", dtype=object)
        status[degenerate] = "degenerate_segment"
        status[refs.to_numpy() & ~finite] = "invalid_node_coordinates"
        status[~refs.to_numpy()] = "invalid_node_reference"
        table["arc_length_m"] = length
        table["projected_length_m"] = projected
        table["arc_bearing_deg"] = bearing
        table["geometry_status"] = status
        table.index.name = "arc_id"
        self.arcs = table
        eligible = status == "valid"
        self.index_arc_ids = arcs.index.to_numpy(dtype="int64")[eligible]
        self.geometries = shapely.linestrings(np.stack([starts[eligible], ends[eligible]], axis=1))
        self._tree = None
        self.index_build_seconds = 0.0
        self.build_seconds = time.perf_counter() - started
        # Validate EVERY valid arc; report representative rows separately.
        rel = np.abs(projected[eligible] - length[eligible]) / length[eligible]
        self.max_length_scale_error = float(rel.max()) if len(rel) else 0.0
        if self.max_length_scale_error > max_scale_error:
            raise ValueError(f"Metric/geodesic arc length disagreement {self.max_length_scale_error:.6g} "
                             f"exceeds {max_scale_error}; local projection is inappropriate")
        ids = np.flatnonzero(eligible)
        selected = ids[np.unique(np.linspace(0, len(ids)-1, min(32, len(ids)), dtype=int))] if len(ids) else []
        self.projection_checks = pd.DataFrame({"arc_id": arcs.index.to_numpy()[selected],
            "geodesic_length_m": length[selected], "projected_length_m": projected[selected],
            "relative_error": np.abs(projected[selected]-length[selected])/length[selected]})

    def build_spatial_index(self) -> STRtree:
        if self._tree is None:
            started = time.perf_counter()
            self._tree = STRtree(self.geometries)
            self.index_build_seconds = time.perf_counter() - started
        return self._tree

    def arc(self, arc_id: int) -> pd.Series:
        """Lookup by the original source ID; missing IDs raise KeyError."""
        return self.arcs.loc[arc_id]

    def to_multidigraph(self) -> nx.MultiDiGraph:
        """Optional graph for later transitions; parallel arcs use arc_id as key.

        Valid zero-length connectors remain in the directed graph at zero cost,
        even though they are excluded from the spatial candidate index. Invalid
        endpoint/coordinate arcs remain in self.arcs with explicit reasons.
        """
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(self.nodes.index)
        for arc_id in self.arcs.index[self.arcs.geometry_status.isin(["valid", "degenerate_segment"])]:
            arc = self.arc(int(arc_id))
            graph.add_edge(int(arc.from_node_id), int(arc.to_node_id), key=int(arc_id),
                           arc_id=int(arc_id), length_m=float(arc.arc_length_m), geometry_status=arc.geometry_status)
        return graph

    def summary(self):
        return {"nodes": len(self.nodes), "directed_arcs": len(self.arcs), "indexed_arcs": len(self.geometries),
                "invalid_node_references": int(self.arcs.geometry_status.eq("invalid_node_reference").sum()),
                "invalid_node_coordinates": int(self.arcs.geometry_status.eq("invalid_node_coordinates").sum()),
                "degenerate_segments": int(self.arcs.geometry_status.eq("degenerate_segment").sum()),
                "duplicate_endpoint_pairs": int(self.arcs.duplicated(["from_node_id", "to_node_id"]).sum()),
                "network_build_seconds": self.build_seconds, "index_build_seconds": self.index_build_seconds,
                "network_extent_m": self.projection.network_extent_m,
                "max_length_scale_error": self.max_length_scale_error}
