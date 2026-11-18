"""Per-trip metric projections, with WGS84 explicitly treated as an assumption."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import CRS, Geod, Transformer

from .quality import valid_coordinates

GEOD = Geod(ellps="WGS84")


@dataclass
class MetricProjection:
    """Local ellipsoidal azimuthal equidistant CRS, longitude-first transforms."""
    center_longitude: float
    center_latitude: float
    max_extent_m: float = 300_000.0

    def __post_init__(self):
        if not np.isfinite(self.max_extent_m) or self.max_extent_m <= 0:
            raise ValueError("max_extent_m must be finite and positive")
        self.crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={self.center_latitude:.12f} +lon_0={self.center_longitude:.12f} "
            "+datum=WGS84 +units=m +no_defs")
        self.forward = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self.inverse = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    @classmethod
    def from_nodes(cls, nodes: pd.DataFrame, max_extent_m: float = 300_000.0):
        valid = nodes.loc[valid_coordinates(nodes)]
        if valid.empty:
            raise ValueError("Cannot choose a metric CRS: no valid network coordinates")
        # Circular longitude average supports local trips crossing the dateline.
        radians = np.radians(valid.longitude.to_numpy())
        lon = float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())))
        lat = float(valid.latitude.mean())
        result = cls(lon, lat, max_extent_m)
        extent = result.radial_distance(valid.longitude.to_numpy(), valid.latitude.to_numpy())
        result.network_extent_m = float(np.max(extent))
        if result.network_extent_m > max_extent_m:
            raise ValueError(f"Network extent {result.network_extent_m:.1f} m exceeds local projection limit "
                             f"{max_extent_m:.1f} m; partition the network or choose a justified CRS")
        return result

    def radial_distance(self, longitude, latitude):
        longitude, latitude = np.broadcast_arrays(longitude, latitude)
        _, _, distance = GEOD.inv(np.full(longitude.shape, self.center_longitude),
                                  np.full(latitude.shape, self.center_latitude), longitude, latitude)
        return np.asarray(distance)

    def xy(self, longitude, latitude):
        return self.forward.transform(longitude, latitude)

    def lonlat(self, x, y):
        return self.inverse.transform(x, y)

    def metadata(self):
        return {"method": "local ellipsoidal azimuthal equidistant", "source_crs_assumption": "EPSG:4326 / WGS84 (not declared by dataset)",
                "center_longitude": self.center_longitude, "center_latitude": self.center_latitude,
                "network_extent_m": getattr(self, "network_extent_m", None),
                "max_extent_m": self.max_extent_m, "metric_crs_wkt": self.crs.to_wkt(),
                "axis_order": "always_xy: longitude, latitude -> x, y metres"}


def geodesic_point_segment_m(lon, lat, start_lon, start_lat, end_lon, end_lat):
    """Independent 1-D geodesic minimisation for diagnostics, not retrieval.

    Golden-section search along the short WGS84 geodesic connecting endpoints;
    endpoints are checked explicitly. Suitable for local segments in this task.
    """
    azimuth, _, length = GEOD.inv(start_lon, start_lat, end_lon, end_lat)

    def distance(position):
        x, y, _ = GEOD.fwd(start_lon, start_lat, azimuth, position)
        return GEOD.inv(lon, lat, x, y)[2]

    lo, hi = 0.0, length
    ratio = (np.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    da, db = distance(a), distance(b)
    for _ in range(45):
        if da < db:
            hi, b, db = b, a, da
            a = hi - ratio * (hi - lo)
            da = distance(a)
        else:
            lo, a, da = a, b, db
            b = lo + ratio * (hi - lo)
            db = distance(b)
    return float(min(da, db, distance(0), distance(length)))
