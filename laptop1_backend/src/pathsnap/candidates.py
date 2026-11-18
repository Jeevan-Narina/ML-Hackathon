"""Label-free, deterministic geometry candidates. This module never loads routes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import shapely

from .loaders import SPLITS
from .quality import valid_coordinates
from .road_network import RoadNetwork

CANDIDATE_SCHEMA = pa.schema([
    ("dataset_partition", pa.string()), ("trip_id", pa.string()), ("gps_observation_index", pa.int64()),
    ("timestamp_s", pa.float64()), ("gps_longitude", pa.float64()), ("gps_latitude", pa.float64()),
    ("candidate_arc_id", pa.int64()), ("from_node_id", pa.int64()), ("to_node_id", pa.int64()),
    ("distance_to_arc_m", pa.float64()), ("candidate_rank", pa.int32()), ("arc_length_m", pa.float64()),
    ("arc_bearing_deg", pa.float64()), ("projected_longitude", pa.float64()), ("projected_latitude", pa.float64()),
    ("position_along_arc", pa.float64()), ("candidate_source", pa.string()), ("fallback_used", pa.bool_()),
    ("search_radius_used_m", pa.float64()),
])
OBSERVATION_SCHEMA = pa.schema([
    ("dataset_partition", pa.string()), ("trip_id", pa.string()), ("gps_observation_index", pa.int64()),
    ("timestamp_s", pa.float64()), ("gps_longitude", pa.float64()), ("gps_latitude", pa.float64()),
    ("previous_observation_index", pa.int64()), ("dt_previous_s", pa.float64()), ("gap_gt_5s", pa.bool_()),
    ("timestamp_nonfinite", pa.bool_()), ("nonincreasing_timestamp", pa.bool_()),
    ("primary_candidate_count", pa.int32()), ("candidates_before_limit", pa.int32()),
    ("candidate_count", pa.int32()), ("candidate_limit_applied", pa.bool_()),
    ("fallback_attempted", pa.bool_()), ("fallback_used", pa.bool_()), ("status", pa.string()),
    ("spatial_bbox_hits", pa.int32()), ("step1_gap_flag", pa.bool_()),
    ("step1_speed_flag", pa.bool_()), ("step1_jump_flag", pa.bool_()),
])


@dataclass(frozen=True)
class CandidateConfig:
    search_radius_m: float = 50.0
    max_candidates: int = 8
    fallback_radius_m: float | None = 150.0
    batch_size: int = 1024
    max_projection_extent_m: float = 300_000.0
    max_projection_scale_error: float = 0.005
    min_segment_m: float = 1e-6

    def __post_init__(self):
        for field in ("search_radius_m", "max_projection_extent_m", "max_projection_scale_error"):
            value = getattr(self, field)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if not np.isfinite(self.min_segment_m) or self.min_segment_m < 0:
            raise ValueError("min_segment_m must be finite and nonnegative")
        for field in ("max_candidates", "batch_size"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.fallback_radius_m is not None and (
            not np.isfinite(self.fallback_radius_m) or self.fallback_radius_m <= self.search_radius_m):
            raise ValueError("fallback_radius_m must exceed primary radius, or be null to disable")

    def as_dict(self):
        return asdict(self)


def _search(network, points, positions, radius):
    """Full-segment bounding boxes, then exact metric distance; no midpoint proxy."""
    if len(positions) == 0 or len(network.geometries) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float), np.zeros(len(points), dtype="int32")
    selected = points[positions]
    x, y = shapely.get_x(selected), shapely.get_y(selected)
    pairs = network.build_spatial_index().query(shapely.box(x-radius, y-radius, x+radius, y+radius))
    observation_positions = positions[pairs[0]]
    geometry_positions = pairs[1]
    bbox_hits = np.bincount(observation_positions, minlength=len(points)).astype("int32")
    distance = shapely.distance(points[observation_positions], network.geometries[geometry_positions])
    within = distance <= radius  # Inclusive boundary; no widening of the configured search.
    return observation_positions[within], geometry_positions[within], distance[within], bbox_hits


def generate_candidates(network: RoadNetwork, track: pd.DataFrame, dataset_partition: str,
                        trip_id: str, config: CandidateConfig = CandidateConfig()) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield candidate rows and one status row per source observation, in batches.

    Candidate counts are geometric retrieval coverage, not correct-label coverage.
    No reference route is accepted as an argument. Time does not influence search.
    """
    if dataset_partition not in SPLITS:
        raise ValueError(f"Unsupported dataset partition: {dataset_partition}")
    if not track.index.is_unique or not pd.api.types.is_integer_dtype(track.index.dtype):
        raise ValueError("GPS observation index must contain unique original integer IDs")
    if config.max_projection_extent_m != network.projection.max_extent_m:
        raise ValueError("Candidate configuration and network projection extent limit disagree")
    network.build_spatial_index()
    original_ids = track.index.to_numpy(dtype="int64")
    timestamp = track.timestamp_s.to_numpy(copy=True)
    timestamp[~np.isfinite(timestamp)] = np.nan
    intervals = np.r_[np.nan, np.diff(timestamp)] if len(track) else np.array([])
    previous = pd.array([pd.NA, *original_ids[:-1]], dtype="Int64") if len(track) else pd.array([], dtype="Int64")
    for start in range(0, len(track), config.batch_size):
        batch = track.iloc[start:start + config.batch_size]
        n = len(batch)
        ids = batch.index.to_numpy(dtype="int64")
        coordinate_valid = valid_coordinates(batch).to_numpy()
        within_projection = np.zeros(n, dtype=bool)
        valid_positions = np.flatnonzero(coordinate_valid)
        if len(valid_positions):
            radial = network.projection.radial_distance(batch.longitude.to_numpy()[valid_positions], batch.latitude.to_numpy()[valid_positions])
            within_projection[valid_positions] = radial <= config.max_projection_extent_m
        usable = coordinate_valid & within_projection
        xy = np.full((n, 2), np.nan)
        xy[usable] = np.column_stack(network.projection.xy(batch.longitude.to_numpy()[usable], batch.latitude.to_numpy()[usable]))
        points = shapely.points(xy)
        positions, geometries, distances, hits = _search(network, points, np.flatnonzero(usable), config.search_radius_m)
        primary_counts = np.bincount(positions, minlength=n).astype("int32")
        sources = np.zeros(len(positions), dtype=bool)
        fallback_attempted = usable & (primary_counts == 0) & (config.fallback_radius_m is not None)
        if fallback_attempted.any():
            extra_p, extra_g, extra_d, extra_hits = _search(network, points, np.flatnonzero(fallback_attempted), config.fallback_radius_m)
            positions = np.r_[positions, extra_p]
            geometries = np.r_[geometries, extra_g]
            distances = np.r_[distances, extra_d]
            sources = np.r_[sources, np.ones(len(extra_p), dtype=bool)]
            hits += extra_hits
        before_limit = np.bincount(positions, minlength=n).astype("int32")
        if len(positions):
            # STRtree traversal order is not a ranking. Explicit sort is required.
            sort = np.lexsort((network.index_arc_ids[geometries], distances, positions))
            positions, geometries, distances, sources = (a[sort] for a in (positions, geometries, distances, sources))
            starts = np.r_[0, np.flatnonzero(np.diff(positions)) + 1]
            rank = np.arange(len(positions)) - np.repeat(starts, np.diff(np.r_[starts, len(positions)])) + 1
            keep = rank <= config.max_candidates
            positions, geometries, distances, sources, rank = (a[keep] for a in (positions, geometries, distances, sources, rank))
            arc_ids = network.index_arc_ids[geometries]
            arcs = network.arcs.loc[arc_ids]
            fraction = shapely.line_locate_point(network.geometries[geometries], points[positions], normalized=True)
            closest = shapely.line_interpolate_point(network.geometries[geometries], fraction, normalized=True)
            lon, lat = network.projection.lonlat(shapely.get_x(closest), shapely.get_y(closest))
            candidates = pd.DataFrame({"dataset_partition": dataset_partition, "trip_id": trip_id,
                "gps_observation_index": ids[positions], "timestamp_s": batch.timestamp_s.to_numpy()[positions],
                "gps_longitude": batch.longitude.to_numpy()[positions], "gps_latitude": batch.latitude.to_numpy()[positions],
                "candidate_arc_id": arc_ids, "from_node_id": arcs.from_node_id.to_numpy(), "to_node_id": arcs.to_node_id.to_numpy(),
                "distance_to_arc_m": distances, "candidate_rank": rank.astype("int32"),
                "arc_length_m": arcs.arc_length_m.to_numpy(), "arc_bearing_deg": arcs.arc_bearing_deg.to_numpy(),
                "projected_longitude": lon, "projected_latitude": lat, "position_along_arc": fraction,
                "candidate_source": np.where(sources, "fallback", "radius"), "fallback_used": sources,
                "search_radius_used_m": np.where(sources, config.fallback_radius_m if config.fallback_radius_m is not None else config.search_radius_m, config.search_radius_m)})
        else:
            candidates = CANDIDATE_SCHEMA.empty_table().to_pandas()
        counts = np.bincount(positions, minlength=n).astype("int32")
        fallback_used = (primary_counts == 0) & (counts > 0)
        status = np.full(n, "unmatched", dtype=object)
        status[counts > 0] = "matched_radius"
        status[fallback_used] = "matched_fallback"
        status[coordinate_valid & ~within_projection] = "projection_out_of_extent"
        status[~coordinate_valid] = "invalid_coordinates"
        dt = intervals[start:start+n]
        observations = pd.DataFrame({"dataset_partition": dataset_partition, "trip_id": trip_id,
            "gps_observation_index": ids, "timestamp_s": batch.timestamp_s.to_numpy(),
            "gps_longitude": batch.longitude.to_numpy(), "gps_latitude": batch.latitude.to_numpy(),
            "previous_observation_index": previous[start:start+n], "dt_previous_s": dt, "gap_gt_5s": dt > 5,
            "timestamp_nonfinite": ~np.isfinite(batch.timestamp_s.to_numpy()), "nonincreasing_timestamp": np.isfinite(dt) & (dt <= 0),
            "primary_candidate_count": primary_counts, "candidates_before_limit": before_limit, "candidate_count": counts,
            "candidate_limit_applied": before_limit > config.max_candidates, "fallback_attempted": fallback_attempted,
            "fallback_used": fallback_used, "status": status, "spatial_bbox_hits": hits,
            "step1_gap_flag": False, "step1_speed_flag": False, "step1_jump_flag": False})
        yield candidates, observations
