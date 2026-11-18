"""Incremental Step 2 runner: one trip/index at a time, no route-file reads."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import threading
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from .candidates import CANDIDATE_SCHEMA, OBSERVATION_SCHEMA, CandidateConfig, generate_candidates
from .candidate_visuals import candidate_map, candidate_plots
from .loaders import discover_trips, load_arcs, load_nodes, load_track
from .projection import GEOD, geodesic_point_segment_m
from .quality import valid_coordinates
from .road_network import RoadNetwork

BACKEND = Path(__file__).resolve().parents[2]
VERSION = "1.0"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MemoryMonitor:
    """Sample process RSS, including Python/native allocations, every 20 ms."""
    def __enter__(self):
        self.process = psutil.Process()
        self.initial = self.peak = self.process.memory_info().rss
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stopped.wait(.02):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *_):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stopped.set()
        self.thread.join()


def _write_batch(writer, frame, schema):
    writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False), row_group_size=8192)


def validate_candidate_batch(network, frame, config):
    """Check emitted IDs, endpoints, bounds and ordering before persisting them."""
    if frame.empty:
        return
    arcs = network.arcs.loc[frame.candidate_arc_id]
    assert arcs.geometry_status.eq("valid").all(), "Candidate includes excluded geometry"
    assert np.array_equal(arcs.from_node_id.to_numpy(), frame.from_node_id.to_numpy()), "from-node mapping changed"
    assert np.array_equal(arcs.to_node_id.to_numpy(), frame.to_node_id.to_numpy()), "to-node mapping changed"
    assert frame.position_along_arc.between(0, 1).all(), "Invalid closest-point fraction"
    assert (frame.distance_to_arc_m >= 0).all() and (frame.distance_to_arc_m <= frame.search_radius_used_m).all(), "Radius violation"
    assert frame.candidate_rank.between(1, config.max_candidates).all(), "Candidate limit violation"
    groups = frame.groupby("gps_observation_index", sort=False)
    assert (groups.distance_to_arc_m.diff().dropna() >= -1e-9).all(), "Candidate distances are not ranked"
    assert not frame.duplicated(["gps_observation_index", "candidate_arc_id"]).any(), "Repeated observation/arc candidate"


def _independent_checks(network, frame, files):
    checks = []
    sample = frame.loc[frame.candidate_rank == 1]
    if len(sample):
        sample = sample.iloc[np.unique(np.linspace(0, len(sample)-1, min(8, len(sample)), dtype=int))]
    for row in sample.itertuples():
        start = network.nodes.loc[row.from_node_id]
        end = network.nodes.loc[row.to_node_id]
        independent = geodesic_point_segment_m(row.gps_longitude, row.gps_latitude,
            start.longitude, start.latitude, end.longitude, end.latitude)
        foot_distance = GEOD.inv(row.gps_longitude, row.gps_latitude, row.projected_longitude, row.projected_latitude)[2]
        error = abs(row.distance_to_arc_m-independent)
        tolerance = .1 + .005 * independent
        checks.append({"dataset_partition": files.split, "trip_id": files.trip_id,
            "gps_observation_index": row.gps_observation_index, "arc_id": row.candidate_arc_id,
            "projected_distance_m": row.distance_to_arc_m, "geodesic_closest_distance_m": independent,
            "geodesic_distance_to_output_foot_m": foot_distance, "absolute_error_m": error,
            "tolerance_m": tolerance, "passed": error <= tolerance})
    return checks


def _training_geometry_sample(network, track, files):
    """Unbounded nearest geometry on a deterministic TRAINING-ONLY sample."""
    if files.split != "train" or not len(network.geometries):
        return []
    valid = track.loc[valid_coordinates(track)]
    valid = valid.loc[network.projection.radial_distance(valid.longitude.to_numpy(), valid.latitude.to_numpy()) <= network.projection.max_extent_m]
    sample = valid.iloc[np.unique(np.linspace(0, len(valid)-1, min(32, len(valid)), dtype=int))] if len(valid) else valid
    if sample.empty:
        return []
    points = shapely.points(np.column_stack(network.projection.xy(sample.longitude.to_numpy(), sample.latitude.to_numpy())))
    pairs, distances = network.build_spatial_index().query_nearest(points, all_matches=False, return_distance=True)
    return [{"dataset_partition": "train", "trip_id": files.trip_id,
             "gps_observation_index": int(sample.index[p]), "nearest_distance_m": float(d)} for p, d in zip(pairs[0], distances)]


def load_step1_flags(path):
    if path is None or not Path(path).is_file():
        return {}, {"available": False}
    frame = pd.read_csv(path, dtype={"trip_id": str})
    categories = {"gap_gt_5s": "step1_gap_flag", "speed_outlier": "step1_speed_flag", "distance_jump": "step1_jump_flag"}
    flags = {}
    for (split, trip_id, category), subset in frame.loc[frame.category.isin(categories)].groupby(["split", "trip_id", "category"]):
        flags.setdefault((split, trip_id), {})[categories[category]] = set(subset.record_id.astype(int))
    return flags, {"available": True, "sha256": digest(path), "categories": categories,
                   "use": "Join GPS diagnostic flags AFTER geometric candidate decisions; no route anomalies enter the generator"}


def process_trip(files, processed, config, flags=None, network_only=False):
    start = time.perf_counter()
    directory = processed / files.split / files.trip_id
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {kind: digest(files.path(kind)) for kind in ("nodes", "arcs", "track")}
    (directory / "status.json").write_text(json.dumps({"state": "processing", "source_sha256": hashes,
                                                       "configuration": asdict(config)}), encoding="utf-8")
    load_start = time.perf_counter()
    nodes, arcs, track = load_nodes(files.path("nodes")), load_arcs(files.path("arcs")), load_track(files.path("track"))
    load_seconds = time.perf_counter()-load_start
    with MemoryMonitor() as memory:
        network = RoadNetwork(nodes, arcs, max_extent_m=config.max_projection_extent_m,
            max_scale_error=config.max_projection_scale_error, min_segment_m=config.min_segment_m)
        network.build_spatial_index()
        network.arcs.reset_index().to_parquet(directory / "road_arcs.parquet", index=False, compression="zstd")
        network_info = {"dataset_partition": files.split, "trip_id": files.trip_id, **network.summary()}
        network_meta = {"schema_version": VERSION, "source_sha256": hashes, "projection": network.projection.metadata(),
                        "configuration": asdict(config), "summary": network_info,
                        "index_lifecycle": "Built once and reused within this trip; rebuilt on each process run, not pickled"}
        if network_only:
            (directory / "network.json").write_text(json.dumps(network_meta, indent=2), encoding="utf-8")
            (directory / "status.json").write_text(json.dumps({"state": "network_only", "source_sha256": hashes}), encoding="utf-8")
            return network_info, None, [], [], [], []
        writers = {}
        paths = {"candidates": CANDIDATE_SCHEMA, "observations": OBSERVATION_SCHEMA, "unmatched": OBSERVATION_SCHEMA}
        candidate_rows, observations, geodesic_checks, sample_rows, nearest_sample = 0, [], [], [], []
        check_ids = set(track.index[np.unique(np.linspace(0, len(track)-1, min(8,len(track)), dtype=int))])
        query_seconds = write_seconds = 0.0
        try:
            for name, schema in paths.items():
                writers[name] = pq.ParquetWriter(directory / f"{name}.partial", schema, compression="zstd")
            iterator = iter(generate_candidates(network, track, files.split, files.trip_id, config))
            while True:
                tick = time.perf_counter()
                try:
                    candidates, status = next(iterator)
                except StopIteration:
                    break
                query_seconds += time.perf_counter()-tick
                validate_candidate_batch(network, candidates, config)
                for name, observation_ids in (flags or {}).items():
                    status[name] = status.gps_observation_index.isin(observation_ids)
                candidate_rows += len(candidates)
                observations.append(status)
                if len(sample_rows) < 20 and len(candidates):
                    sample_rows.extend(candidates.head(20-len(sample_rows)).to_dict("records"))
                selected_checks = candidates.loc[candidates.gps_observation_index.isin(check_ids)]
                if len(selected_checks):
                    batch_checks = _independent_checks(network, selected_checks, files)
                    geodesic_checks.extend(batch_checks)
                    if not all(c["passed"] for c in batch_checks):
                        raise ValueError("Independent geodesic candidate-distance validation failed")
                tick = time.perf_counter()
                _write_batch(writers["candidates"], candidates, CANDIDATE_SCHEMA)
                _write_batch(writers["observations"], status, OBSERVATION_SCHEMA)
                _write_batch(writers["unmatched"], status.loc[status.candidate_count == 0], OBSERVATION_SCHEMA)
                write_seconds += time.perf_counter()-tick
        finally:
            for writer in writers.values():
                writer.close()
        for name in paths:
            (directory / f"{name}.partial").replace(directory / f"{name}.parquet")
        status = pd.concat(observations, ignore_index=True)
        nearest_sample = _training_geometry_sample(network, track, files)
        distances = pd.read_parquet(directory / "candidates.parquet", columns=["distance_to_arc_m", "candidate_rank"])
        rank1 = distances.loc[distances.candidate_rank == 1, "distance_to_arc_m"]
        counts = status.candidate_count
        info = {**network_info, "gps_observations": len(track), "candidate_rows": candidate_rows,
                "matched_observations": int((counts > 0).sum()), "unmatched_observations": int((counts == 0).sum()),
                "unmatched_percent": float((counts == 0).mean()*100), "mean_candidates": float(counts.mean()),
                "fallback_attempted": int(status.fallback_attempted.sum()), "fallback_observations": int(status.fallback_used.sum()),
                "fallback_percent": float(status.fallback_used.mean()*100), "truncated_observations": int(status.candidate_limit_applied.sum()),
                "invalid_gps_observations": int(status.status.eq("invalid_coordinates").sum()),
                "projection_out_of_extent": int(status.status.eq("projection_out_of_extent").sum()),
                "gap_observations_gt_5s": int(status.gap_gt_5s.sum()), "maximum_gap_s": float(status.dt_previous_s.max()),
                "step1_speed_flags_preserved": int(status.step1_speed_flag.sum()), "step1_jump_flags_preserved": int(status.step1_jump_flag.sum()),
                "candidate_distance_mean_m": float(distances.distance_to_arc_m.mean()),
                "candidate_distance_p50_m": float(distances.distance_to_arc_m.median()),
                "candidate_distance_p95_m": float(distances.distance_to_arc_m.quantile(.95)),
                "candidate_distance_max_m": float(distances.distance_to_arc_m.max()),
                "nearest_retained_distance_p50_m": float(rank1.median()), "nearest_retained_distance_p95_m": float(rank1.quantile(.95)),
                "spatial_bbox_hits": int(status.spatial_bbox_hits.sum()), "load_seconds": load_seconds,
                "candidate_generation_seconds": query_seconds, "parquet_write_seconds": write_seconds,
                "total_seconds": time.perf_counter()-start}
        info = {key: (None if isinstance(value, float) and not np.isfinite(value) else value) for key, value in info.items()}
        network_meta["candidate_summary"] = info
        (directory / "network.json").write_text(json.dumps(network_meta, indent=2, allow_nan=False), encoding="utf-8")
        checks = network.projection_checks.assign(dataset_partition=files.split, trip_id=files.trip_id).to_dict("records")
    info.update(peak_process_rss_mib=memory.peak/2**20, initial_process_rss_mib=memory.initial/2**20)
    (directory / "status.json").write_text(json.dumps({"state": "complete", "source_sha256": hashes,
        "configuration": asdict(config), "candidate_rows": candidate_rows, "observations": len(track)}, indent=2), encoding="utf-8")
    return network_info, info, checks, geodesic_checks, sample_rows, nearest_sample


def run_step2(raw, output, processed, config, partitions=("train", "student_test"), trip_ids=None,
              step1_anomalies=None, make_map=True, network_only=False):
    started = time.perf_counter()
    raw, output, processed = Path(raw).resolve(), Path(output).resolve(), Path(processed).resolve()
    if output.is_relative_to(raw) or processed.is_relative_to(raw):
        raise ValueError("Derived data and reports must not be written inside the raw source directory")
    output.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    trips = [f for f in discover_trips(raw) if f.split in partitions and (trip_ids is None or f.trip_id in trip_ids)]
    if not trips or (trip_ids and set(trip_ids) != {f.trip_id for f in trips}):
        raise ValueError("No matching trips, or one or more requested trip IDs do not exist in the selected partition")
    flags, flag_meta = load_step1_flags(step1_anomalies)
    network_rows, summaries, length_checks, point_checks, samples, training_samples, failures = [], [], [], [], [], [], []
    for number, files in enumerate(trips, 1):
        print(f"[{number:02}/{len(trips)}] {files.split}/{files.trip_id}: building network and spatial index", flush=True)
        try:
            with MemoryMonitor() as trip_memory:
                result = process_trip(files, processed, config, flags.get((files.split, files.trip_id)), network_only)
            net, summary, lengths, points, sample, training = result
            network_rows.append(net)
            if summary is not None:
                summary.update(peak_process_rss_mib=trip_memory.peak/2**20, initial_process_rss_mib=trip_memory.initial/2**20)
                summaries.append(summary)
                print(f"  {summary['candidate_rows']} candidates; {summary['matched_observations']}/{summary['gps_observations']} observations matched; "
                      f"{summary['total_seconds']:.2f} s; peak process RSS {summary['peak_process_rss_mib']:.1f} MiB", flush=True)
            length_checks.extend(lengths)
            point_checks.extend(points)
            samples.extend(sample)
            training_samples.extend(training)
        except Exception as exc:
            failures.append({"dataset_partition": files.split, "trip_id": files.trip_id, "error": str(exc), "traceback": traceback.format_exc()})
            directory = processed / files.split / files.trip_id
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "status.json").write_text(json.dumps({"state": "failed", "error": str(exc)}), encoding="utf-8")
            print(f"  FAILED: {exc}", flush=True)
        gc.collect()
    pd.DataFrame(network_rows).to_csv(output / "road_network_summary.csv", index=False)
    pd.DataFrame(failures, columns=["dataset_partition", "trip_id", "error", "traceback"]).to_csv(output / "failures.csv", index=False)
    metadata = {"schema_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "configuration": asdict(config),
        "python": platform.python_version(), "dependencies": {n: importlib.metadata.version(n) for n in
        ["numpy", "pandas", "shapely", "pyproj", "pyarrow", "psutil", "folium", "networkx", "pytest"]},
        "requested_trips": len(trips), "successful_trips": len(network_rows), "failed_trips": len(failures),
        "network_only": network_only, "partitions": list(partitions), "step1_diagnostic_flags": flag_meta,
        "map_generated": False,
        "processed_directory": str(processed),
        "candidate_decisions_use_reference_routes": False, "memory_measurement": "Process RSS sampled every 20 ms per trip; includes Python, native libraries, file buffers and retained allocator memory; not incremental allocation or a comparative benchmark"}
    if summaries:
        summary = pd.DataFrame(summaries)
        summary.to_csv(output / "candidate_summary.csv", index=False)
        for name, rows in [("projection_length_checks.csv", length_checks), ("geodesic_candidate_checks.csv", point_checks),
                           ("candidate_sample.csv", samples), ("training_geometry_sample.csv", training_samples)]:
            pd.DataFrame(rows).to_csv(output / name, index=False)
        counts_by_partition, distances_by_partition, partitions_rows = {}, {}, []
        for partition, group in summary.groupby("dataset_partition", sort=False):
            counts, distances, nearest = [], [], []
            for trip_id in group.trip_id:
                directory = processed / partition / trip_id
                counts.append(pd.read_parquet(directory / "observations.parquet", columns=["candidate_count"]).candidate_count.to_numpy())
                d = pd.read_parquet(directory / "candidates.parquet", columns=["distance_to_arc_m", "candidate_rank"])
                distances.append(d.distance_to_arc_m.to_numpy())
                nearest.append(d.loc[d.candidate_rank == 1, "distance_to_arc_m"].to_numpy())
            counts, distances, nearest = np.concatenate(counts), np.concatenate(distances), np.concatenate(nearest)
            counts_by_partition[partition], distances_by_partition[partition] = counts, distances
            partitions_rows.append({"dataset_partition": partition, "trips": len(group), "gps_observations": int(group.gps_observations.sum()),
                "candidate_rows": int(group.candidate_rows.sum()), "matched_observations": int((counts > 0).sum()),
                "unmatched_observations": int((counts == 0).sum()), "unmatched_percent": float((counts == 0).mean()*100),
                "fallback_observations": int(group.fallback_observations.sum()), "fallback_percent": float(group.fallback_observations.sum()/len(counts)*100),
                "mean_candidates": float(counts.mean()), "candidate_distance_mean_m": float(distances.mean()) if len(distances) else None,
                "candidate_distance_p50_m": float(np.median(distances)) if len(distances) else None,
                "candidate_distance_p95_m": float(np.quantile(distances,.95)) if len(distances) else None,
                "nearest_retained_distance_p50_m": float(np.median(nearest)) if len(nearest) else None,
                "nearest_retained_distance_p95_m": float(np.quantile(nearest,.95)) if len(nearest) else None,
                "total_trip_seconds": float(group.total_seconds.sum()), "peak_process_rss_mib": float(group.peak_process_rss_mib.max())})
        pd.DataFrame(partitions_rows).to_csv(output / "partition_summary.csv", index=False)
        candidate_plots(summary, counts_by_partition, distances_by_partition, output)
        if make_map and (summary.dataset_partition == "train").any():
            training = summary.loc[summary.dataset_partition == "train"].copy()
            chosen = training.assign(deviation=(training.gps_observations-training.gps_observations.median()).abs()).sort_values(["deviation", "trip_id"]).iloc[0]
            files = next(f for f in trips if f.split == "train" and f.trip_id == chosen.trip_id)
            net = RoadNetwork(load_nodes(files.path("nodes")), load_arcs(files.path("arcs")), max_extent_m=config.max_projection_extent_m,
                              max_scale_error=config.max_projection_scale_error, min_segment_m=config.min_segment_m)
            directory = processed / "train" / chosen.trip_id
            candidate_map(net, load_track(files.path("track")), pd.read_parquet(directory / "candidates.parquet"),
                          pd.read_parquet(directory / "observations.parquet"), output, "train", chosen.trip_id)
            metadata["map_generated"] = True
            del net
    metadata["runtime_seconds"] = time.perf_counter()-started
    metadata["source_code_sha256"] = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "candidate_schema.json").write_text(json.dumps({"version": VERSION,
        "candidate_fields": {f.name: str(f.type) for f in CANDIDATE_SCHEMA},
        "observation_fields": {f.name: str(f.type) for f in OBSERVATION_SCHEMA},
        "position_along_arc": "closest point's normalized fraction along directed projected segment, 0=from node, 1=to node; not a vehicle position or ground-truth label"}, indent=2), encoding="utf-8")
    print(f"Step 2 finished: {len(network_rows)}/{len(trips)} successful trips; {len(failures)} failures; {metadata['runtime_seconds']:.2f} s", flush=True)
    return metadata


def main(network_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=BACKEND / "data/raw")
    parser.add_argument("--output", type=Path, default=BACKEND / "outputs/step2")
    parser.add_argument("--processed", type=Path, default=BACKEND / "data/processed/step2")
    parser.add_argument("--config", type=Path, default=BACKEND / "config/candidates.json")
    parser.add_argument("--partition", choices=["all", "train", "student_test"], default="all")
    parser.add_argument("--trip", action="append", help="Original eight-digit trip ID; repeat for multiple trips")
    parser.add_argument("--radius", type=float)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--fallback-radius", type=float)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--no-map", action="store_true")
    parser.add_argument("--network-only", action="store_true", default=network_only)
    parser.add_argument("--step1-anomalies", type=Path, default=BACKEND / "outputs/eda/anomalies.csv")
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8"))
    for argument, field in [("radius", "search_radius_m"), ("max_candidates", "max_candidates"),
                            ("fallback_radius", "fallback_radius_m"), ("batch_size", "batch_size")]:
        if getattr(args, argument) is not None:
            values[field] = getattr(args, argument)
    if args.no_fallback:
        values["fallback_radius_m"] = None
    config = CandidateConfig(**values)
    partitions = ("train", "student_test") if args.partition == "all" else (args.partition,)
    meta = run_step2(args.raw.resolve(), args.output.resolve(), args.processed.resolve(), config, partitions,
                     args.trip, args.step1_anomalies, not args.no_map, args.network_only)
    if meta["failed_trips"]:
        raise SystemExit("Step 2 has failed trips; inspect failures.csv")
