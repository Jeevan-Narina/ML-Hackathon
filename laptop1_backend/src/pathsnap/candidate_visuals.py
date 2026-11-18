"""Step 2 plots and candidate-only Folium layers; Step 1 visuals stay intact."""
from __future__ import annotations

import json
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely

from .quality import valid_coordinates


def candidate_map(network, track, candidates, observations, output: Path, partition: str, trip_id: str):
    valid = track.loc[valid_coordinates(track)]
    if valid.empty:
        raise ValueError("No valid GPS coordinates for candidate map")
    # Show a local inspection view, avoiding thousands of overlapping candidates.
    matched = observations.loc[observations.candidate_count > 0]
    if matched.empty:
        selected_id = int(valid.index[len(valid)//2])
    else:
        multiple = matched.loc[matched.candidate_count > 1]
        pool = multiple if len(multiple) else matched
        selected_id = int(pool.iloc[len(pool)//2].gps_observation_index)
    selected = track.loc[selected_id]
    nearby_ids = network.build_spatial_index().query(shapely.box(
        *np.r_[np.array(network.projection.xy(selected.longitude, selected.latitude))-1000,
               np.array(network.projection.xy(selected.longitude, selected.latitude))+1000]))
    roads = folium.FeatureGroup(name="Road network near selected observation (1 km square)")
    segments = []
    for arc_id in network.index_arc_ids[nearby_ids]:
        arc = network.arc(int(arc_id))
        segments.append(network.nodes.loc[[arc.from_node_id, arc.to_node_id], ["latitude", "longitude"]].values.tolist())
    if segments:
        folium.PolyLine(segments, color="#999999", weight=1, opacity=.5).add_to(roads)
    map_ = folium.Map(location=[selected.latitude, selected.longitude], zoom_start=17, prefer_canvas=True)
    roads.add_to(map_)
    gps = folium.FeatureGroup(name="All original GPS observations")
    for i, row in valid.iterrows():
        folium.CircleMarker([row.latitude, row.longitude], radius=2, color="#1767d2", fill=True,
                            tooltip=f"GPS observation {i}; t={row.timestamp_s:g} s").add_to(gps)
    gps.add_to(map_)
    local = candidates.loc[candidates.gps_observation_index.isin(
        observations.loc[(observations.gps_observation_index-selected_id).abs() <= 10, "gps_observation_index"])]
    local_ids = sorted(local.candidate_arc_id.unique())
    layer = folium.FeatureGroup(name="Candidate segments for selected GPS and up to 10 neighbours each side")
    for arc_id in local_ids:
        arc = network.arc(int(arc_id))
        folium.PolyLine(network.nodes.loc[[arc.from_node_id, arc.to_node_id], ["latitude", "longitude"]].values.tolist(),
                        color="#e48a16", weight=4, opacity=.6, tooltip=f"Candidate arc {arc_id}").add_to(layer)
    layer.add_to(map_)
    chosen = candidates.loc[candidates.gps_observation_index == selected_id]
    focus = folium.FeatureGroup(name="Selected observation candidates and closest points")
    for row in chosen.itertuples():
        arc = network.arc(row.candidate_arc_id)
        folium.PolyLine(network.nodes.loc[[arc.from_node_id, arc.to_node_id], ["latitude", "longitude"]].values.tolist(),
                        color="#a72f9c", weight=6, opacity=.8,
                        tooltip=f"arc_id {row.candidate_arc_id}; rank {row.candidate_rank}; {row.distance_to_arc_m:.2f} m; {row.candidate_source}").add_to(focus)
        folium.CircleMarker([row.projected_latitude, row.projected_longitude], radius=5, color="#138f60", fill=True,
                            tooltip=f"Closest point on arc {row.candidate_arc_id}; fraction {row.position_along_arc:.4f}").add_to(focus)
        folium.PolyLine([[selected.latitude, selected.longitude], [row.projected_latitude, row.projected_longitude]],
                        color="#138f60", weight=2, dash_array="4 5").add_to(focus)
    folium.Marker([selected.latitude, selected.longitude], tooltip=f"Selected GPS observation {selected_id}",
                  icon=folium.Icon(color="red", icon="info-sign")).add_to(focus)
    focus.add_to(map_)
    folium.LayerControl().add_to(map_)
    map_.get_root().html.add_child(folium.Element(f'''
    <div style="position:fixed;bottom:25px;left:25px;z-index:1000;background:white;padding:12px;border:1px solid #777;font:13px Arial">
    <b>Step 2 candidates: {partition}/{trip_id}</b><br>Selected observation: {selected_id}<br>
    Blue: all original GPS points; grey: local roads<br>Orange: nearby candidate segments<br>
    Purple: selected observation's candidates<br>Green: closest points and distance connectors<br>
    Red: selected GPS observation<br>No ML prediction or reconstructed route; no reference labels used.<br>
    Candidate layers are bounded to a local inspection window; raw outputs remain complete.
    </div>'''))
    map_.save(str(output / "candidate_training_trip.html"))
    metadata = {"dataset_partition": partition, "trip_id": trip_id, "selected_observation": selected_id,
                "gps_points_displayed": len(valid), "road_arcs_displayed": len(segments),
                "local_candidate_arcs_displayed": len(local_ids), "selected_candidate_rows": len(chosen),
                "network_display_window_half_width_m": 1000, "candidate_observation_window": 10,
                "selection": "Training trip nearest median training GPS count; middle observation with multiple candidates if available",
                "reference_route_used": False}
    (output / "candidate_map_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def candidate_plots(summary, counts_by_partition, distances_by_partition, output: Path):
    colors = {"train": "#2366ad", "student_test": "#d97521"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    max_count = max((int(a.max()) for a in counts_by_partition.values() if len(a)), default=1)
    for split, counts in counts_by_partition.items():
        axes[0].hist(counts, bins=np.arange(max_count+2)-.5, alpha=.6, label=split, color=colors[split])
    max_distance = max((float(a.max()) for a in distances_by_partition.values() if len(a)), default=1)
    bins = np.linspace(0, max(1, max_distance), 51)
    for split, distances in distances_by_partition.items():
        axes[1].hist(distances, bins=bins, alpha=.6, label=split, color=colors[split])
    axes[0].set(xlabel="Retained candidates per observation (zero = unmatched)", ylabel="GPS observations", title="Candidate count distribution")
    axes[1].set(xlabel="Point-to-segment distance (m)", ylabel="Candidate rows", title="Candidate distances (all retained rows)")
    for ax in axes:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "candidate_distributions.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = summary.dataset_partition + "/" + summary.trip_id
    axes[0].bar(labels, summary.unmatched_percent, color=[colors[s] for s in summary.dataset_partition])
    axes[0].set(xlabel="Partition / trip ID", ylabel="Unmatched observations (%)", title="Geometric retrieval gaps by trip")
    axes[0].tick_params(axis="x", labelrotation=90, labelsize=6)
    for split, color in colors.items():
        subset = summary.loc[summary.dataset_partition == split]
        axes[1].scatter(subset.directed_arcs, subset.total_seconds, label=split, color=color)
    axes[1].set(xlabel="Directed arcs", ylabel="Measured trip processing time (s)", title="Actual runtime by network size")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "candidate_coverage_and_runtime.png", dpi=160)
    plt.close(fig)
