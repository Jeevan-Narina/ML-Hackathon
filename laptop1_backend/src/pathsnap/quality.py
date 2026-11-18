"""Descriptive GPS and directed-network diagnostics; no changes to inputs."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    nominal_interval_s: float = 1.0
    interval_tolerance_s: float = 1e-6
    gap_s: float = 5.0
    speed_kmh: float = 180.0
    jump_m: float = 500.0
    map_padding_deg: float = 0.01


EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1, lat1, lon2, lat2):
    """Spherical geodesic distance in metres; inputs are degrees, lon before lat."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def valid_coordinates(frame: pd.DataFrame) -> pd.Series:
    return (np.isfinite(frame.longitude) & np.isfinite(frame.latitude)
            & frame.longitude.between(-180, 180) & frame.latitude.between(-90, 90))


def gps_quality(track: pd.DataFrame, thresholds: Thresholds = Thresholds()) -> tuple[dict, pd.DataFrame]:
    valid = valid_coordinates(track)
    coords = track[["longitude", "latitude"]].to_numpy(copy=True)
    coords[~valid.to_numpy()] = np.nan
    distance = haversine_m(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])
    timestamps = track.timestamp_s.to_numpy(copy=True)
    timestamps[~np.isfinite(timestamps)] = np.nan
    dt = np.diff(timestamps)
    speed = np.full(len(dt), np.nan)
    np.divide(distance * 3.6, dt, out=speed, where=np.isfinite(dt) & (dt > 0))
    irregular = np.isfinite(dt) & ~np.isclose(dt, thresholds.nominal_interval_s, atol=thresholds.interval_tolerance_s, rtol=0)
    steps = pd.DataFrame({"from_observation_id": track.index[:-1], "to_observation_id": track.index[1:],
                          "interval_s": dt, "distance_m": distance, "speed_kmh": speed,
                          "irregular_interval": irregular, "gap_gt_5s": dt > thresholds.gap_s,
                          "speed_outlier": speed > thresholds.speed_kmh,
                          "distance_jump": distance > thresholds.jump_m})
    usable = track.loc[valid]
    stats = {
        "gps_observations": len(track), "gps_missing_cells": int(track.isna().sum().sum()),
        "gps_rows_with_missing": int(track.isna().any(axis=1).sum()),
        "gps_invalid_coordinates": int((~valid).sum()),
        "gps_duplicate_records": int(track.duplicated().sum()),
        "gps_repeated_coordinates": int(track.duplicated(["longitude", "latitude"]).sum()),
        "timestamp_nonfinite": int((~np.isfinite(track.timestamp_s)).sum()),
        "timestamp_backwards": int((dt < 0).sum()), "timestamp_equal_adjacent": int((dt == 0).sum()),
        "timestamp_duplicate_values": int(track.loc[np.isfinite(track.timestamp_s), "timestamp_s"].duplicated().sum()),
        "timestamp_strictly_increasing": bool(np.isfinite(timestamps).all() and (dt > 0).all()),
        "irregular_intervals": int(irregular.sum()), "gaps_gt_5s": int((dt > thresholds.gap_s).sum()),
        "interval_min_s": float(np.nanmin(dt)) if np.isfinite(dt).any() else np.nan,
        "interval_median_s": float(np.nanmedian(dt)) if np.isfinite(dt).any() else np.nan,
        "interval_max_s": float(np.nanmax(dt)) if np.isfinite(dt).any() else np.nan,
        "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) else np.nan,
        "longitude_min": usable.longitude.min(), "longitude_max": usable.longitude.max(),
        "latitude_min": usable.latitude.min(), "latitude_max": usable.latitude.max(),
        "trajectory_distance_m": float(np.nansum(distance)), "unmeasurable_steps": int((~np.isfinite(distance)).sum()),
        "movement_median_m": float(np.nanmedian(distance)) if np.isfinite(distance).any() else np.nan,
        "movement_max_m": float(np.nanmax(distance)) if np.isfinite(distance).any() else np.nan,
        "speed_median_kmh": float(np.nanmedian(speed)) if np.isfinite(speed).any() else np.nan,
        "speed_max_kmh": float(np.nanmax(speed)) if np.isfinite(speed).any() else np.nan,
        "speed_outliers": int((speed > thresholds.speed_kmh).sum()),
        "distance_jumps": int((distance > thresholds.jump_m).sum()),
    }
    return stats, steps


def network_quality(nodes: pd.DataFrame, arcs: pd.DataFrame, route: pd.DataFrame) -> tuple[dict, dict]:
    valid_refs = arcs.from_node_id.isin(nodes.index) & arcs.to_node_id.isin(nodes.index)
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes.index)
    graph.add_edges_from((int(a), int(b), int(i)) for i, (a, b) in
                         zip(arcs.index[valid_refs], arcs.loc[valid_refs].itertuples(index=False, name=None)))
    weak = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
    strong = sorted((len(c) for c in nx.strongly_connected_components(graph)), reverse=True)
    valid_route = route.arc_id.isin(arcs.index)
    route_edges = arcs.reindex(route.arc_id.to_numpy())
    endpoints_valid = (route_edges.from_node_id.isin(nodes.index) & route_edges.to_node_id.isin(nodes.index)).to_numpy()
    comparable = valid_route.to_numpy() & endpoints_valid
    pair_valid = comparable[:-1] & comparable[1:]
    breaks = pair_valid & (route_edges.to_node_id.to_numpy()[:-1] != route_edges.from_node_id.to_numpy()[1:])
    lengths = np.full(len(route), np.nan)
    coord_valid = valid_coordinates(nodes)
    good_positions = np.flatnonzero(comparable)
    if len(good_positions):
        edges = route_edges.iloc[good_positions].astype("int64")
        left = nodes.loc[edges.from_node_id]
        right = nodes.loc[edges.to_node_id]
        usable = coord_valid.loc[edges.from_node_id].to_numpy() & coord_valid.loc[edges.to_node_id].to_numpy()
        measured = haversine_m(left.longitude.to_numpy()[usable], left.latitude.to_numpy()[usable],
                              right.longitude.to_numpy()[usable], right.latitude.to_numpy()[usable])
        lengths[good_positions[usable]] = measured
    stats = {
        "network_nodes": len(nodes), "directed_arcs": len(arcs),
        "node_missing_cells": int(nodes.isna().sum().sum()), "node_invalid_coordinates": int((~coord_valid).sum()),
        "invalid_node_reference_arcs": int((~valid_refs).sum()),
        "duplicate_directed_arcs": int(arcs.duplicated(["from_node_id", "to_node_id"]).sum()),
        "self_loops": int((arcs.from_node_id == arcs.to_node_id).sum()),
        "isolated_nodes": nx.number_of_isolates(graph), "weak_components": len(weak), "strong_components": len(strong),
        "largest_weak_component_nodes": weak[0] if weak else 0,
        "largest_weak_component_fraction": weak[0] / len(nodes) if weak else np.nan,
        "largest_strong_component_nodes": strong[0] if strong else 0,
        "reference_route_arcs": len(route), "route_invalid_arc_ids": int((~valid_route).sum()),
        "route_invalid_endpoint_arcs": int((valid_route.to_numpy() & ~endpoints_valid).sum()),
        "route_disconnected_pairs": int(breaks.sum()), "route_uncheckable_pairs": int((~pair_valid).sum()),
        "reference_route_distance_m": float(np.nansum(lengths)),
        "route_unmeasurable_arcs": int((~np.isfinite(lengths)).sum()),
    }
    details = {
        "invalid_node_reference_arcs": arcs.index[~valid_refs].tolist(),
        "duplicate_directed_arcs": arcs.index[arcs.duplicated()].tolist(),
        "node_invalid_coordinates": nodes.index[~coord_valid].tolist(),
        "isolated_nodes": list(nx.isolates(graph)),
        "route_invalid_arc_ids": route.index[~valid_route].tolist(),
        "route_invalid_endpoint_arcs": np.flatnonzero(valid_route.to_numpy() & ~endpoints_valid).tolist(),
        "route_disconnected_pairs": (np.flatnonzero(breaks) + 1).tolist(),
        "route_uncheckable_pairs": (np.flatnonzero(~pair_valid) + 1).tolist(),
    }
    return stats, details
