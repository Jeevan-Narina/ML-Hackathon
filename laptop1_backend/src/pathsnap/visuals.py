"""Figures and a reference-only Folium map, generated from audited source data."""
from __future__ import annotations

from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .loaders import Trip
from .quality import Thresholds, valid_coordinates

COLORS = {"train": "#2366ad", "student_test": "#d97521"}


def make_plots(inventory: pd.DataFrame, gps: pd.DataFrame, network: pd.DataFrame,
               steps: pd.DataFrame, coordinates: pd.DataFrame, anomalies: pd.DataFrame, output: Path) -> list[str]:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=160)
        plt.close(fig)
        names.append(name)

    fig, ax = plt.subplots(figsize=(16, 5))
    for split, color in COLORS.items():
        subset = gps[gps.split == split]
        ax.bar(subset.split + "/" + subset.trip_id, subset.gps_observations, color=color, label=split)
    ax.set(xlabel="Trip (partition/ID)", ylabel="GPS observations", title="GPS observations in all available trips")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.legend()
    save(fig, "gps_observations_per_trip.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    distance_bins = np.histogram_bin_edges(gps.trajectory_distance_m / 1000, bins=15)
    for split, color in COLORS.items():
        values = steps.loc[steps.split == split, "interval_s"].dropna()
        counts = values.value_counts().sort_index()
        axes[0].plot(counts.index, counts.values, "o-", color=color, label=split)
        axes[1].hist(gps.loc[gps.split == split, "trajectory_distance_m"] / 1000,
                     bins=distance_bins, alpha=.6, color=color, label=split)
    axes[0].set(xlabel="Timestamp interval (s; symlog, linear up to 5 s)", ylabel="Number of intervals (log scale)",
                title="Sampling intervals; all finite intervals")
    axes[0].set_xscale("symlog", linthresh=5)
    axes[0].xaxis.set_major_locator(matplotlib.ticker.SymmetricalLogLocator(base=10, linthresh=10))
    axes[0].set_yscale("log")
    axes[1].set(xlabel="Noisy GPS trajectory distance (km)", ylabel="Trips", title="Trip distance distribution")
    for ax in axes:
        ax.legend(loc="upper right")
    save(fig, "timestamp_intervals_and_trip_distances.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for split, color in COLORS.items():
        subset = network[network.split == split]
        axes[0].scatter(subset.network_nodes, subset.directed_arcs, alpha=.7, color=color, label=split)
        axes[1].scatter(subset.weak_components, subset.largest_weak_component_fraction * 100,
                        alpha=.7, color=color, label=split)
    axes[0].set(xlabel="Nodes per trip", ylabel="Directed arcs per trip", title="Road-network size")
    axes[1].set(xlabel="Weakly connected components", ylabel="Nodes in largest weak component (%)",
                title="Road-network connectivity")
    for ax in axes:
        ax.legend(loc="upper right")
    save(fig, "road_network_statistics.png")

    fig, ax = plt.subplots(figsize=(13, 6))
    fields = ["gps_missing_cells", "gps_invalid_coordinates", "gps_duplicate_records", "timestamp_backwards",
              "gaps_gt_5s", "speed_outliers", "distance_jumps", "invalid_node_reference_arcs",
              "duplicate_directed_arcs", "isolated_nodes", "route_invalid_arc_ids", "route_disconnected_pairs"]
    merged = gps.merge(network, on=["split", "trip_id"])
    x = np.arange(len(fields))
    for offset, (split, color) in enumerate(COLORS.items()):
        counts = merged.loc[merged.split == split, fields].sum()
        bars = ax.bar(x + (offset - .5) * .38, counts, .38, label=split, color=color)
        ax.bar_label(bars, fontsize=7, rotation=90, padding=3)
    ax.set_xticks(x, [field.replace("_", " ") for field in fields], rotation=35, ha="right")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0, top=max(10, float(merged[fields].sum().max()) * 5))
    ax.set(ylabel="Flag count (symlog; zero shown)", title="Missing values and diagnostic flags (flags may overlap)")
    ax.legend()
    save(fig, "missing_and_anomalous_data.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    speed_bins = np.histogram_bin_edges(steps.loc[np.isfinite(steps.speed_kmh), "speed_kmh"], bins=70)
    for split, color in COLORS.items():
        subset = coordinates[coordinates.split == split]
        axes[0].scatter(subset.longitude, subset.latitude, s=.5, alpha=.2, color=color, label=split, rasterized=True)
        speeds = steps.loc[(steps.split == split) & np.isfinite(steps.speed_kmh), "speed_kmh"]
        axes[1].hist(speeds, bins=speed_bins, alpha=.6, color=color, label=split)
    axes[0].set(xlabel="Longitude (degrees)", ylabel="Latitude (degrees)", title="All valid GPS coordinates (no sampling)")
    axes[0].set_aspect(1 / np.cos(np.radians(coordinates.latitude.mean())))
    axes[1].set(xlabel="Approximate speed (km/h)", ylabel="Steps (log scale)", title="Consecutive GPS displacement speeds")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.legend(loc="upper right", markerscale=4 if ax is axes[0] else 1)
    save(fig, "gps_extent_and_speeds.png")
    return names


def make_map(trip: Trip, output: Path, thresholds: Thresholds) -> dict:
    gps = trip.track.loc[valid_coordinates(trip.track)]
    if gps.empty:
        raise ValueError("Representative trip has no valid coordinates to map")
    node_valid = valid_coordinates(trip.nodes)
    valid_edges = trip.arcs.from_node_id.isin(trip.nodes.index) & trip.arcs.to_node_id.isin(trip.nodes.index)
    edges = trip.arcs.loc[valid_edges]
    edges = edges.loc[node_valid.loc[edges.from_node_id].to_numpy() & node_valid.loc[edges.to_node_id].to_numpy()]
    left = trip.nodes.loc[edges.from_node_id].to_numpy()
    right = trip.nodes.loc[edges.to_node_id].to_numpy()
    pad = thresholds.map_padding_deg
    lo = gps[["longitude", "latitude"]].min().to_numpy() - pad
    hi = gps[["longitude", "latitude"]].max().to_numpy() + pad
    visible = ((np.maximum(left, right) >= lo) & (np.minimum(left, right) <= hi)).all(axis=1)
    segments = np.stack((left[visible, ::-1], right[visible, ::-1]), axis=1).tolist()
    map_ = folium.Map(location=[gps.latitude.mean(), gps.longitude.mean()], prefer_canvas=True, tiles="OpenStreetMap")
    roads = folium.FeatureGroup(name="Road network (GPS bounds + 0.01 degrees)")
    if segments:
        folium.PolyLine(segments, color="#888888", weight=1, opacity=.5, tooltip="Supplied directed network; display uses straight node-to-node segments").add_to(roads)
    roads.add_to(map_)
    reference = folium.FeatureGroup(name="Supplied reference route (not a prediction)")
    route_segments = []
    for arc_id in trip.route.arc_id:
        if arc_id not in edges.index:
            continue
        edge = edges.loc[arc_id]
        points = trip.nodes.loc[[edge.from_node_id, edge.to_node_id], ["latitude", "longitude"]].values.tolist()
        route_segments.append(points)
    if route_segments:
        folium.PolyLine(route_segments, color="#18843b", weight=5, opacity=.85,
                        tooltip="Supplied reference arcs; disconnected pairs are not bridged").add_to(reference)
    reference.add_to(map_)
    observations = folium.FeatureGroup(name="Original noisy GPS observations")
    for i, row in gps.iterrows():
        folium.CircleMarker([row.latitude, row.longitude], radius=2.5, color="#1767d2", weight=1,
                            fill=True, fill_opacity=.7, tooltip=f"Observation {i}; timestamp {row.timestamp_s:g} s").add_to(observations)
    observations.add_to(map_)
    folium.Marker(gps.iloc[0][["latitude", "longitude"]].tolist(), tooltip="GPS start", icon=folium.Icon(color="purple", icon="play")).add_to(map_)
    folium.Marker(gps.iloc[-1][["latitude", "longitude"]].tolist(), tooltip="GPS end", icon=folium.Icon(color="red", icon="stop")).add_to(map_)
    bounds = pd.concat([gps[["latitude", "longitude"]], trip.nodes.loc[
        trip.nodes.index.isin(edges.loc[edges.index.isin(trip.route.arc_id)].values.flatten()), ["latitude", "longitude"]]])
    map_.fit_bounds([bounds.min().tolist(), bounds.max().tolist()])
    folium.LayerControl().add_to(map_)
    map_.get_root().html.add_child(folium.Element(f'''
    <div style="position:fixed;bottom:25px;left:25px;z-index:1000;background:white;padding:12px;border:1px solid #777;font:13px Arial">
    <b>Step 1 EDA: train/{trip.files.trip_id}</b><br>
    <span style="color:#1767d2">Blue dots</span>: original GPS<br>
    <span style="color:#18843b">Green</span>: supplied reference route<br>
    <span style="color:#888">Grey</span>: local road network<br>
    Purple marker: start; red marker: end<br>
    No model predictions. Reference arcs are drawn individually.<br>
    Network display: {int(visible.sum())} / {len(trip.arcs)} arcs; all GPS points shown.
    </div>'''))
    map_.save(str(output / "representative_training_trip.html"))
    return {"split": trip.files.split, "trip_id": trip.files.trip_id,
            "gps_points_displayed": len(gps), "network_arcs_total": len(trip.arcs),
            "network_arcs_displayed": int(visible.sum()), "reference_arcs_total": len(trip.route),
            "reference_arcs_displayed": len(route_segments), "network_padding_degrees": pad,
            "selection": "Training trip nearest median training GPS observation count; tie by trip ID",
            "display": "All valid GPS observations and valid reference arcs; network arcs whose bounding boxes overlap padded GPS bounds; no sampling or simplification"}
