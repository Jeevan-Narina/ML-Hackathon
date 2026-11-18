"""Reproducible full-partition EDA. No training labels or features are generated."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .loaders import (DataFormatError, EXTENSIONS, SCHEMAS, discover_trips,
                      load_arcs, load_nodes, load_route, load_track, load_trip)
from .quality import EARTH_RADIUS_M, Thresholds, gps_quality, network_quality, valid_coordinates

BACKEND = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(raw: Path, manifest: Path) -> None:
    table = pd.read_csv(manifest)
    expected = set(table.relative_path)
    actual = {p.relative_to(raw).as_posix() for p in raw.rglob("*") if p.is_file() and p.name != ".gitkeep"}
    if expected != actual:
        raise ValueError(f"Source manifest file-set mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    for row in table.itertuples():
        if sha256(raw / row.relative_path) != row.sha256:
            raise ValueError(f"Source bytes changed since extraction: {row.relative_path}")


def run_eda(raw: Path, output: Path, thresholds: Thresholds, manifest: Path | None = None) -> dict:
    from .visuals import make_map, make_plots

    started = time.perf_counter()
    trips = discover_trips(raw)
    if not trips:
        raise ValueError("No trips discovered")
    if manifest is not None:
        verify_manifest(raw, manifest)
    output.mkdir(parents=True, exist_ok=True)
    inventory, file_rows, gps_rows, net_rows, step_frames, coord_frames, events = [], [], [], [], [], [], []
    track_hashes = {}
    readers = {"track": load_track, "nodes": load_nodes, "arcs": load_arcs, "route": load_route}

    for number, files in enumerate(trips, 1):
        key = {"split": files.split, "trip_id": files.trip_id}
        loaded, file_stats = {}, {}

        def event(category, record_id="", detail="", severity="warning"):
            events.append({**key, "category": category, "severity": severity,
                           "record_id": record_id, "detail": detail})

        for kind in EXTENSIONS:
            path = files.path(kind)
            info = {**key, "file_type": kind, "relative_path": path.relative_to(raw).as_posix(),
                    "available": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0,
                    "sha256": sha256(path) if path.is_file() else "", "record_count": np.nan,
                    "malformed_records": 0, "status": "missing", "columns": "|".join(SCHEMAS[kind][0]),
                    "dtype": SCHEMAS[kind][2], "index_name": SCHEMAS[kind][1]}
            if not path.is_file():
                event("missing_file", detail=kind, severity="error")
            else:
                try:
                    loaded[kind] = readers[kind](path)
                    info.update(record_count=len(loaded[kind]), status="loaded")
                except DataFormatError as exc:
                    info.update(record_count=exc.record_count, malformed_records=len(exc.issues), status="malformed")
                    for issue in exc.issues:
                        event("malformed_record", issue.line, f"{kind}: {issue.message}", "error")
                except (OSError, UnicodeError) as exc:
                    info["status"] = "unreadable"
                    event("unreadable_file", detail=f"{kind}: {exc}", severity="error")
            file_rows.append(info)
            file_stats[kind] = info
        unexpected = [p.name for p in files.directory.iterdir() if p.name not in {files.path(e).name for e in EXTENSIONS}]
        for name in unexpected:
            event("unexpected_trip_file", detail=name)
        inv = {**key, "files_available": sum(s["available"] for s in file_stats.values()),
               "missing_files": "|".join(k for k, s in file_stats.items() if not s["available"]),
               "malformed_records": sum(s["malformed_records"] for s in file_stats.values()),
               "unreadable_files": sum(s["status"] == "unreadable" for s in file_stats.values()),
               "unexpected_files": "|".join(unexpected)}
        for kind, count in [("track", "gps_observations"), ("nodes", "network_nodes"),
                            ("arcs", "directed_arcs"), ("route", "reference_route_arcs")]:
            inv[count] = file_stats[kind]["record_count"]
            inv[f"{kind}_available"] = file_stats[kind]["available"]
            inv[f"{kind}_bytes"] = file_stats[kind]["bytes"]
        inventory.append(inv)
        if "track" in loaded:
            track = loaded["track"]
            digest = file_stats["track"]["sha256"]
            if digest in track_hashes and track_hashes[digest]["split"] != files.split:
                event("cross_split_duplicate_track", detail=str(track_hashes[digest]), severity="error")
            track_hashes[digest] = key
            stats, steps = gps_quality(track, thresholds)
            gps_rows.append({**key, **stats})
            steps.insert(0, "trip_id", files.trip_id)
            steps.insert(0, "split", files.split)
            step_frames.append(steps)
            valid = valid_coordinates(track)
            coords = track.loc[valid, ["longitude", "latitude"]].assign(**key)
            coord_frames.append(coords)
            masks = {
                "gps_missing_values": track.isna().any(axis=1),
                "gps_invalid_coordinates": ~valid,
                "gps_duplicate_record": track.duplicated(),
                "gps_repeated_coordinates": track.duplicated(["longitude", "latitude"]),
                "timestamp_nonfinite": ~np.isfinite(track.timestamp_s),
                "timestamp_duplicate_value": track.timestamp_s.notna() & track.timestamp_s.duplicated(),
            }
            for category, mask in masks.items():
                for index in track.index[mask]:
                    event(category, int(index), "Source observation ID; repeated coordinates may be stationary observations")
            for category, mask in {
                "timestamp_backwards": steps.interval_s < 0,
                "timestamp_equal_adjacent": steps.interval_s == 0,
                "irregular_interval": steps.irregular_interval,
                "gap_gt_5s": steps.gap_gt_5s,
                "speed_outlier": steps.speed_outlier,
                "distance_jump": steps.distance_jump,
            }.items():
                for row in steps.loc[mask].itertuples():
                    event(category, int(row.to_observation_id),
                          f"from={row.from_observation_id}; dt_s={row.interval_s:g}; distance_m={row.distance_m:g}; speed_kmh={row.speed_kmh:g}")
        if all(kind in loaded for kind in ("nodes", "arcs", "route")):
            stats, details = network_quality(loaded["nodes"], loaded["arcs"], loaded["route"])
            net_rows.append({**key, **stats})
            for category, positions in details.items():
                for position in positions:
                    identity = "route position (second arc of pair where applicable)" if category.startswith("route_") else "source node/arc row ID"
                    detail = identity
                    if category == "route_disconnected_pairs":
                        prev_id = int(loaded["route"].iloc[position - 1].arc_id)
                        next_id = int(loaded["route"].iloc[position].arc_id)
                        prev = loaded["arcs"].loc[prev_id]
                        next_ = loaded["arcs"].loc[next_id]
                        detail = (f"reference positions {position-1}->{position}; arc IDs {prev_id}->{next_id}; "
                                  f"previous endpoint={prev.to_node_id}; next start={next_.from_node_id}")
                    elif category == "duplicate_directed_arcs":
                        arc = loaded["arcs"].loc[position]
                        detail = f"arc_id={position}; from_node_id={arc.from_node_id}; to_node_id={arc.to_node_id}; repeated directed endpoints"
                    elif category == "route_invalid_arc_ids":
                        detail = f"reference position={position}; arc_id={loaded['route'].iloc[position].arc_id}"
                    event(category, int(position), detail)
        print(f"[{number:02d}/{len(trips)}] {files.split}/{files.trip_id}: "
              f"GPS={inv['gps_observations']}, nodes={inv['network_nodes']}, arcs={inv['directed_arcs']}", flush=True)

    inv = pd.DataFrame(inventory)
    file_table = pd.DataFrame(file_rows)
    gps = pd.DataFrame(gps_rows)
    network = pd.DataFrame(net_rows)
    steps = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    coords = pd.concat(coord_frames, ignore_index=True) if coord_frames else pd.DataFrame()
    anomalies = pd.DataFrame(events, columns=["split", "trip_id", "category", "severity", "record_id", "detail"])
    anomaly_summary = anomalies.groupby(["split", "category", "severity"]).size().rename("flag_count").reset_index()
    totals = inv.groupby("split", sort=False).agg(
        trips=("trip_id", "size"), files_available=("files_available", "sum"),
        gps_observations=("gps_observations", "sum"), network_nodes=("network_nodes", "sum"),
        directed_arcs=("directed_arcs", "sum"), reference_route_arcs=("reference_route_arcs", "sum"),
        malformed_records=("malformed_records", "sum"), unreadable_files=("unreadable_files", "sum")).reset_index()
    tables = {"dataset_inventory.csv": inv, "file_inventory.csv": file_table,
              "partition_summary.csv": totals, "gps_quality.csv": gps, "network_quality.csv": network,
              "gps_steps.csv": steps, "anomalies.csv": anomalies, "anomaly_summary.csv": anomaly_summary}
    for name, table in tables.items():
        table.to_csv(output / name, index=False, float_format="%.10g")
    structural_errors = int((anomalies.severity == "error").sum())
    complete_analysis = len(gps) == len(inv) and len(network) == len(inv)
    if gps.empty or network.empty or steps.empty or coords.empty:
        raise ValueError("CSV audit saved, but insufficient valid data for requested plots/map; inspect anomalies.csv")
    plots = make_plots(inv, gps, network, steps, coords, anomalies, output)
    candidates = gps[gps.split == "train"].copy()
    candidates = candidates[candidates.trip_id.isin(network[network.split == "train"].trip_id)]
    if candidates.empty:
        raise ValueError("No complete training trip available for the representative map")
    median = candidates.gps_observations.median()
    selected = candidates.assign(deviation=(candidates.gps_observations - median).abs()).sort_values(["deviation", "trip_id"]).iloc[0]
    trip_files = next(f for f in trips if f.split == "train" and f.trip_id == selected.trip_id)
    map_metadata = make_map(load_trip(trip_files), output, thresholds)
    (output / "map_metadata.json").write_text(json.dumps(map_metadata, indent=2), encoding="utf-8")
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "dependencies": {name: importlib.metadata.version(name) for name in
                                 ["numpy", "pandas", "matplotlib", "networkx", "folium", "pytest"]},
                "thresholds": asdict(thresholds), "earth_radius_m": EARTH_RADIUS_M,
                "source_manifest_verified": manifest is not None,
                "source_manifest_sha256": sha256(manifest) if manifest is not None else None,
                "source_fingerprint_sha256": hashlib.sha256("\n".join(
                    f"{r.relative_path}:{r.sha256}" for r in file_table.sort_values("relative_path").itertuples()).encode()).hexdigest(),
                "files_audited": len(file_table), "trips_audited": len(inv),
                "gps_trips_analyzed": len(gps), "network_trips_analyzed": len(network),
                "structural_errors": structural_errors, "complete_analysis": complete_analysis,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "outputs": list(tables) + plots + ["representative_training_trip.html", "map_metadata.json", "EDA_SUMMARY.md", "run_metadata.json"]}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(output, totals, gps, network, file_table, anomalies, map_metadata, metadata)
    print(f"EDA complete: {len(inv)} trips; {structural_errors} structural error flags; outputs: {output}", flush=True)
    return metadata


def write_summary(output, totals, gps, network, files, anomalies, map_info, metadata):
    def total(frame, field):
        return int(frame[field].sum())

    lines = ["# PathSnap Step 1: measured dataset exploration", "",
             f"Execution UTC: {metadata['created_utc']}. Python {metadata['python']}. "
             f"Inspected {metadata['trips_audited']} trips and {metadata['files_audited']} expected source files. "
             f"GPS analysis completed for {len(gps)} trips; network/reference analysis for {len(network)} trips.", "",
             "## Partition inventory", "",
             "| Partition | Trips | Files | GPS observations | Node rows | Directed arc rows | Reference arc entries |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in totals.itertuples():
        lines.append(f"| {row.split} | {row.trips} | {row.files_available} | {row.gps_observations} | {row.network_nodes} | {row.directed_arcs} | {row.reference_route_arcs} |")
    lines += ["", "Node and arc totals sum per-trip networks; they are not globally unique road features. "
              "Evaluator-testing data was not supplied, created, or inspected. IDs remain local to their trip.", "",
              "## GPS quality (partition-specific measurements)", ""]
    steps = pd.read_csv(output / "gps_steps.csv", dtype={"trip_id": str})
    for split in ("train", "student_test"):
        g = gps[gps.split == split]
        n = network[network.split == split]
        s = steps[steps.split == split]
        finite = np.isfinite(s.interval_s)
        nominal = np.isclose(s.loc[finite, "interval_s"], metadata['thresholds']['nominal_interval_s'],
                             atol=metadata['thresholds']['interval_tolerance_s'], rtol=0)
        lines += [f"- **{split}:** {total(g, 'gps_missing_cells')} missing GPS cells; "
                  f"{total(g, 'gps_invalid_coordinates')} invalid coordinate rows; {total(g, 'gps_duplicate_records')} duplicate full records; "
                  f"{total(g, 'gps_repeated_coordinates')} repeated coordinate records (including stationary movement). "
                  f"{total(g, 'timestamp_backwards')} backwards timestamp steps; {total(g, 'timestamp_equal_adjacent')} zero intervals; "
                  f"{total(g, 'timestamp_nonfinite')} nonfinite timestamps. "
                  f"{total(g, 'irregular_intervals')} non-nominal finite intervals; {total(g, 'gaps_gt_5s')} gaps > 5 s. "
                  f"Nominal intervals: {nominal.mean()*100:.3f}% of {finite.sum()} finite steps. "
                  f"Finite intervals range {s.interval_s.min():g}–{s.interval_s.max():g} s; "
                  f"per-trip durations range {g.duration_s.min():g}–{g.duration_s.max():g} s.",
                  f"  GPS distance sum {g.trajectory_distance_m.sum()/1000:.3f} km; per-trip range "
                  f"{g.trajectory_distance_m.min()/1000:.3f}–{g.trajectory_distance_m.max()/1000:.3f} km. "
                  f"Maximum step distance {g.movement_max_m.max():.3f} m; maximum approximate speed {g.speed_max_kmh.max():.3f} km/h. "
                  f"{total(g, 'speed_outliers')} speed flags; {total(g, 'distance_jumps')} distance jump flags. "
                  f"{total(g, 'unmeasurable_steps')} unmeasurable coordinate steps.",
                  f"  Longitude {g.longitude_min.min():.6f} to {g.longitude_max.max():.6f}; "
                  f"latitude {g.latitude_min.min():.6f} to {g.latitude_max.max():.6f}.",
                  f"  Network/reference: {total(n, 'invalid_node_reference_arcs')} invalid endpoint arcs; "
                  f"{total(n, 'duplicate_directed_arcs')} duplicate directed endpoint pairs; {total(n, 'isolated_nodes')} isolated nodes. "
                  f"Weak components per trip {n.weak_components.min()}–{n.weak_components.max()}; "
                  f"strong components {n.strong_components.min()}–{n.strong_components.max()}. "
                  f"Largest weak-component coverage {n.largest_weak_component_fraction.min()*100:.2f}%–{n.largest_weak_component_fraction.max()*100:.2f}%. "
                  f"{total(n, 'node_invalid_coordinates')} invalid network coordinate rows. "
                  f"{total(n, 'route_invalid_arc_ids')} invalid reference arc IDs; {total(n, 'route_invalid_endpoint_arcs')} reference arcs with invalid endpoints; "
                  f"{total(n, 'route_disconnected_pairs')} disconnected consecutive reference pairs; "
                  f"{total(n, 'route_uncheckable_pairs')} uncheckable pairs. "
                  f"Reference length (sum of straight endpoint distances): {n.reference_route_distance_m.sum()/1000:.3f} km; "
                  f"{total(n, 'route_unmeasurable_arcs')} unmeasurable reference entries."]
    lines += ["", "## Notable measured anomalies", "",
              "Largest finite timestamp intervals (source observation IDs):", "",
              "| Partition | Trip | From observation | To observation | Interval (s) | Displacement (m) |",
              "|---|---|---:|---:|---:|---:|"]
    for row in steps.nlargest(5, "interval_s").itertuples():
        lines.append(f"| {row.split} | {row.trip_id} | {row.from_observation_id} | {row.to_observation_id} | {row.interval_s:g} | {row.distance_m:.3f} |")
    if not steps.speed_kmh.dropna().empty:
        fastest = steps.loc[steps.speed_kmh.idxmax()]
        lines += ["", f"Maximum approximate speed is {fastest.speed_kmh:.3f} km/h in "
                  f"{fastest['split']}/{fastest.trip_id}, observation {fastest.from_observation_id} to {fastest.to_observation_id}. "
                  "A speed flag measures noisy GPS displacement over elapsed time, not verified vehicle speed."]
    broken = network[network.route_disconnected_pairs > 0]
    lines += ["", "Trips with disconnected consecutive reference arcs: " + (
        "; ".join(f"{row.split}/{row.trip_id} ({row.route_disconnected_pairs})" for row in broken.itertuples()) or "none") + ". "
        "Exact reference positions, arc IDs and endpoint mismatches are recorded in anomalies.csv. "
        "These are supplied-reference limitations to investigate; no replacement route or pointwise labels have been invented.", "",
        "Coordinates span several world regions, so no single local projected CRS is assumed for the full dataset. "
        "Timestamp and distance histograms use the complete supplied data; common histogram bins permit partition comparison. "
        "The timestamp axis uses a symmetric logarithmic scale outside 5 s so long gaps do not hide nominal sampling."]
    status_counts = files.status.value_counts().to_dict()
    lines += ["", "## File and route validation", "",
              f"File statuses: {status_counts}. Malformed records: {int(files.malformed_records.sum())}. "
              f"Source archive manifest verified before EDA: {metadata['source_manifest_verified']}. "
              f"Structural error flags: {metadata['structural_errors']}.", "",
              "`anomalies.csv` records each flagged observation, node, arc or reference position; "
              "flags can overlap. `anomaly_summary.csv` aggregates by partition and category. "
              "Component counts describe topology, not necessarily corrupt data. Duplicate endpoint pairs may be valid parallel arcs. "
              "Reference continuity checks exact directed endpoint IDs; adjacent repeated reference arc IDs are not silently collapsed.", "",
              "## Assumptions and thresholds", "",
              "- Official schema: headerless whitespace records. Track = longitude, latitude, timestamp_s (float64); "
              "nodes = longitude, latitude (float64); arcs = from_node_id, to_node_id (int64); route = arc_id (int64). "
              "Node/arc IDs are zero-based row positions. GPS IDs and reference sequence positions are separate indices.",
              "- Source order and bytes are preserved. Blank/wrong-width/nonnumeric records reject a file with line numbers; "
              "no row deletion or ID renumbering. NaN/NA/null floating-point tokens are retained as missing values. "
              "Coordinates must be finite, longitude in [-180,180], latitude in [-90,90]. This checks global plausibility, not location truth.",
              "- Timestamps are seconds as documented; no timezone or absolute epoch is inferred. "
              "A positive interval is required for speed. Ordering is checked in original row order, without sorting. "
              "Missing/invalid coordinate steps are not bridged; reported distances sum measurable adjacent steps only.",
              f"- Diagnostic configuration: `{json.dumps(metadata['thresholds'], sort_keys=True)}`. "
              "Non-nominal intervals differ from 1 s beyond the stated tolerance. Gap, speed and jump comparisons are strictly greater than their thresholds. "
              "180 km/h and 500 m flag observations for review; they are not learned cutoffs, physical certainty, or cleaning rules. "
              "A large jump across a large time gap can be plausible.",
              f"- Haversine uses radius {EARTH_RADIUS_M:,.1f} m. Coordinates are interpreted as geographic degrees "
              "consistent with the official document and supplied values; an explicit source CRS/datum declaration is absent. "
              "Spherical distances are approximate and GPS noise inflates trajectory lengths. "
              "Route length uses straight node-to-node segments because no detailed arc polyline geometry is supplied.",
              "- Directed multigraph keeps every valid arc ID, including parallel/reversed edges, and every node. "
              "Invalid endpoint arcs cannot enter connectivity diagnostics and are reported explicitly. "
              "Weak connectivity ignores direction; strong connectivity requires directed reachability. Isolated nodes have degree zero.",
              "- EDA reads supplied student-test references for descriptive validation only. It fits no transformations or models, "
              "creates no observation-to-arc labels, and does not use student-test statistics to set thresholds or select the map trip.", "",
              "## Visualisations and evidence", "",
              *[f"- [{name}]({name})" for name in metadata['outputs'] if name not in {"EDA_SUMMARY.md", "run_metadata.json"}],
              f"- Representative map: train/{map_info['trip_id']}; {map_info['selection']}. "
              f"All {map_info['gps_points_displayed']} valid GPS points, {map_info['reference_arcs_displayed']} reference entries, "
              f"{map_info['network_arcs_displayed']} of {map_info['network_arcs_total']} network arcs displayed. "
              "Only network display is spatially bounded; no raw records are sampled, simplified, or changed. "
              "Reference arcs are drawn separately so gaps are not fabricated into continuous lines. "
              "Folium markers use latitude/longitude; distance functions use longitude/latitude. "
              "The map needs internet for Leaflet assets and OpenStreetMap tiles; embedded coordinates remain in the HTML.", "",
              "## Limitations and Stage 2 requirements", "",
              "The supplied references are not independently verified physical truth. GPS observations and reference arc entries "
              "are unequal-length sequences with no supplied pointwise alignment. A defensible alignment/label construction "
              "and candidate coverage evaluation must be designed and validated in a later stage; neither exists in Step 1. "
              "Review flagged gaps, jumps and reference continuity before defining preprocessing. Do not silently repair them. "
              "Design trip-grouped cross-validation using training trips and fit learned preprocessing on training folds only. "
              "See [requirements review](../../../docs/REQUIREMENTS_REVIEW.md) for the official-versus-prompt student-test policy discrepancy. "
              "Evaluator data remains unavailable. Two-laptop API/UI integration, models, HMM/Viterbi, optimisation and deployment "
              "packages are later-stage work and have not been implemented.", "",
              "This is an AI-assisted, programmatically generated working EDA record from actual executed analysis, "
              "not the final university technical report. Team members should review and be able to explain all findings "
              "and disclose assistance according to the event policy.", ""]
    (output / "EDA_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=BACKEND / "data/raw")
    parser.add_argument("--output", type=Path, default=BACKEND / "outputs/eda")
    parser.add_argument("--config", type=Path, default=BACKEND / "config/eda.json")
    parser.add_argument("--manifest", type=Path, help="Optional extraction manifest; standard raw dataset uses its default manifest automatically")
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8"))
    thresholds = Thresholds(**values)
    if any(not np.isfinite(v) or v <= 0 for v in asdict(thresholds).values()):
        parser.error("All diagnostic thresholds must be finite and positive")
    manifest = args.manifest
    if manifest is None and args.raw.resolve() == (BACKEND / "data/raw").resolve():
        manifest = BACKEND / "outputs/eda/source_manifest.csv"
    result = run_eda(args.raw.resolve(), args.output.resolve(), thresholds, manifest)
    if result["structural_errors"] or not result["complete_analysis"]:
        raise SystemExit("Audit completed with structural errors or incomplete analysis; see anomalies.csv")
