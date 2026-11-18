"""Regenerate the technical summary and small diagnostics from executed outputs."""
from pathlib import Path
import argparse
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]


def summarize(output: Path, processed: Path, test_results: Path):
    meta = json.loads((output / "run_metadata.json").read_text())
    if meta["network_only"]:
        network = pd.read_csv(output / "road_network_summary.csv")
        (output / "STEP2_SUMMARY.md").write_text(
            f"# Step 2 network-only run\n\nProcessed {meta['successful_trips']}/{meta['requested_trips']} networks; "
            f"{int(network.directed_arcs.sum())} source arcs. Candidate generation, plots and tests are not claimed "
            "by this network-only command. Inspect road_network_summary.csv and run_metadata.json.\n", encoding="utf-8")
        return
    def read_or_empty(name, columns):
        try:
            return pd.read_csv(output / name)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=columns)
    summary = pd.read_csv(output / "candidate_summary.csv", dtype={"trip_id": str})
    partitions = pd.read_csv(output / "partition_summary.csv")
    network = pd.read_csv(output / "road_network_summary.csv", dtype={"trip_id": str})
    checks = read_or_empty("geodesic_candidate_checks.csv", ["passed", "absolute_error_m"])
    training = read_or_empty("training_geometry_sample.csv", ["nearest_distance_m"])
    config = meta["configuration"]
    issues, distributions, statuses = [], [], []
    for row in summary.itertuples():
        directory = processed / row.dataset_partition / row.trip_id
        arcs = pd.read_parquet(directory / "road_arcs.parquet")
        bad = arcs.loc[arcs.geometry_status != "valid"].assign(dataset_partition=row.dataset_partition, trip_id=row.trip_id)
        issues.append(bad)
        observations = pd.read_parquet(directory / "observations.parquet", columns=["candidate_count", "status"])
        counts = observations.candidate_count.value_counts().sort_index()
        distributions.extend({"dataset_partition": row.dataset_partition, "trip_id": row.trip_id,
                              "candidate_count": int(k), "observations": int(v)} for k,v in counts.items())
        statuses.extend({"dataset_partition": row.dataset_partition, "trip_id": row.trip_id,
                         "status": k, "observations": int(v)} for k,v in observations.status.value_counts().items())
    pd.concat(issues,ignore_index=True).to_csv(output / "geometry_issues.csv", index=False)
    pd.DataFrame(distributions).to_csv(output / "candidate_count_distribution.csv", index=False)
    pd.DataFrame(statuses).to_csv(output / "observation_status_summary.csv", index=False)
    if test_results.is_file():
        suites = ET.parse(test_results).getroot().iter("testsuite")
        totals = {k: 0 for k in ["tests", "failures", "errors", "skipped"]}
        seconds = 0.0
        for suite in suites:
            for k in totals:
                totals[k] += int(suite.get(k,0))
            seconds += float(suite.get("time",0))
        passed = totals["tests"]-totals["failures"]-totals["errors"]-totals["skipped"]
        test_text = f"{passed} passed; {totals['failures']} failed; {totals['errors']} errors; {totals['skipped']} skipped. JUnit suite time {seconds:.3f} s."
    else:
        test_text = "No JUnit test result exists at the requested path; tests are not claimed as executed."
    lines = ["# Step 2: executed road-network and candidate-generation summary", "",
        f"Generated from actual outputs for run UTC {meta['created_utc']}. Python {meta['python']}. "
        f"Trips: {meta['successful_trips']}/{meta['requested_trips']} successful; {meta['failed_trips']} failed.", "",
        "## Methodology", "",
        "The existing Step 1 readers load only `.nodes`, `.arcs` and `.track`. No `.route` is read by the Step 2 runner. "
        "An arc-indexed table retains every source arc ID, directed endpoints, geodesic length, projected length, initial "
        "geodesic bearing and geometry status. Distinct parallel/reversed arcs keep separate IDs. NetworkX multigraph "
        "construction is optional and preserves zero-length connectors at zero cost; the primary representation avoids "
        "allocating Python graph objects for millions of edges. Invalid geometries are retained and recorded, not repaired.", "",
        "Each trip gets a local WGS84 ellipsoidal azimuthal equidistant projection centred on its valid network nodes "
        "(circular mean longitude, mean latitude). WGS84 is an explicit assumption because the dataset omits a datum. "
        "All transforms use longitude/latitude ordering with `always_xy=True`. Networks and observations outside the "
        "configured local extent are rejected or marked unmatched. Every valid arc's projected length is checked against "
        "independent PyProj Geod geodesic length. Representative candidate distances are independently minimised along "
        "the short geodesic between source endpoints, including endpoints; tolerance is 0.1 m + 0.5% of that distance.", "",
        f"Maximum observed network radius: {network.network_extent_m.max():.3f} m; maximum relative projected arc-length "
        f"error: {network.max_length_scale_error.max():.9g}. {len(checks)} independent point-to-segment checks: "
        f"{int(checks.passed.sum())} passed; maximum absolute discrepancy {checks.absolute_error_m.max():.9g} m.", "",
        "One Shapely STRtree per trip indexes complete projected LineStrings; a separate array maps tree positions to "
        "original arc IDs, preserving coincident geometries. Batches query square bounding boxes, then exact projected "
        "point-to-segment distance enforces the circular radius. Results sort by distance, then original arc ID for exact "
        "ties, and are capped per observation. Only observations with no primary result attempt the bounded fallback. "
        "No candidate is fabricated for an unmatched observation. Closest-point fraction is measured along the directed "
        "projected segment, not an inferred vehicle position. There is no GPS/reference alignment, label, model or decoder.", "",
        "## Configuration and training-only justification", "", "```json", json.dumps(config,indent=2), "```", "",
        "The initial 50 m / 150 m radii are conservative engineering search limits, not learned accuracy thresholds. "
        "Eight candidates bounds storage and later state-space size while retaining multiple directed/parallel alternatives. "
        "The 1,024-observation batch bounds transient query arrays; output writes incrementally using Zstandard Parquet. "
        "The 300 km local extent and 0.5% length tolerance are safeguards, not inferred properties of all future trips.", ""]
    if len(training):
        distances = training.nearest_distance_m
        lines += [f"A separate unbounded-nearest geometry diagnostic sampled {len(training)} training observations "
            "(32 evenly spaced observations per training trip; equal weighting by trip, not observation). "
            f"Nearest-distance median {distances.median():.3f} m; p95 {distances.quantile(.95):.3f} m; "
            f"p99 {distances.quantile(.99):.3f} m; maximum {distances.max():.3f} m. "
            f"{(distances<=config['search_radius_m']).mean()*100:.5f}% lie within the primary radius and "
            + (f"{(distances<=config['fallback_radius_m']).mean()*100:.5f}% within the fallback bound. " if config['fallback_radius_m'] is not None else "Fallback is disabled. ") +
            "Full-data results below are observation-weighted and can differ substantially from this trip-balanced sample. "
            "Student-test labels/statistics were not used to choose these defaults.", ""]
    lines += ["## Measured candidate results", "",
        "| Partition | Trips | GPS observations | Candidate rows | Matched | Unmatched | Fallback observations |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in partitions.itertuples():
        lines.append(f"| {r.dataset_partition} | {r.trips} | {r.gps_observations} | {r.candidate_rows} | {r.matched_observations} | {r.unmatched_observations} | {r.fallback_observations} |")
    for r in partitions.itertuples():
        lines += ["", f"- **{r.dataset_partition}:** {(1-r.unmatched_percent/100)*100:.6f}% have at least one candidate; "
            f"unmatched {r.unmatched_percent:.6f}%; fallback {r.fallback_percent:.6f}%; mean retained count {r.mean_candidates:.4f}. "
            f"All retained candidate distances: mean {r.candidate_distance_mean_m:.3f} m, median {r.candidate_distance_p50_m:.3f} m, "
            f"p95 {r.candidate_distance_p95_m:.3f} m. Rank-1 retained distances: median {r.nearest_retained_distance_p50_m:.3f} m, "
            f"p95 {r.nearest_retained_distance_p95_m:.3f} m. Rank-1 statistics exclude unmatched observations."]
    lines += ["", "These are geometric retrieval statistics, not label coverage, accuracy or route reconstruction quality. "
        f"The top-{config['max_candidates']} limit truncated {int(summary.truncated_observations.sum())} observation candidate sets; "
        "the true reference arc might be absent even when several nearby candidates exist. Step 3 must evaluate that after defensible alignment.", "",
        "## Geometry and anomaly handling", "",
        f"Inspected {int(network.nodes.sum())} node rows and {int(network.directed_arcs.sum())} arc rows across per-trip networks. "
        f"Indexed {int(network.indexed_arcs.sum())} arcs; {int(network.degenerate_segments.sum())} degenerate segments excluded only "
        f"from spatial retrieval. Invalid node-reference arcs: {int(network.invalid_node_references.sum())}; "
        f"invalid-coordinate arcs: {int(network.invalid_node_coordinates.sum())}. "
        "`geometry_issues.csv` lists source IDs; full road tables retain them. Node/arc totals are not globally unique roads.", "",
        f"All {int(summary.gps_observations.sum())} GPS observations have explicit status rows, including "
        f"{int(summary.unmatched_observations.sum())} unmatched rows. Preserved {int(summary.gap_observations_gt_5s.sum())} "
        f"gaps >5 s, maximum {summary.maximum_gap_s.max():g} s, {int(summary.step1_speed_flags_preserved.sum())} Step 1 speed "
        f"flags and {int(summary.step1_jump_flags_preserved.sum())} jump flags. GPS anomaly flags are joined after geometric "
        "selection; observations are not dropped. Reference discontinuities are neither read into selection nor rewritten. "
        "Large gaps need later temporal handling, while large spatial jumps can leave observations unmatched.", ""]
    broken = summary.loc[summary.unmatched_observations>0]
    for r in broken.itertuples():
        lines.append(f"- {r.dataset_partition}/{r.trip_id}: {r.unmatched_observations} unmatched observations.")
    lines += ["", "## Performance and storage", "",
        f"Measured wall time: {meta['runtime_seconds']:.3f} s, including reports/plots/map; sum of trip processing times "
        f"{summary.total_seconds.sum():.3f} s. Load time {summary.load_seconds.sum():.3f} s; network construction "
        f"{summary.network_build_seconds.sum():.3f} s; STRtree construction {summary.index_build_seconds.sum():.3f} s; "
        f"candidate generation {summary.candidate_generation_seconds.sum():.3f} s; incremental Parquet writes "
        f"{summary.parquet_write_seconds.sum():.3f} s. Remaining time includes validation, geodesic diagnostics and reporting. "
        f"Maximum sampled process RSS during trip processing: {summary.peak_process_rss_mib.max():.3f} MiB.", "",
        meta['memory_measurement'] + ". This is an execution measurement on this laptop, not a comparative benchmark. "
        "Only one trip network/index is held at a time; objects are released between trips. Aggregate reporting retains "
        "only candidate-distance/count vectors, not all networks. STRtrees are reused in memory and deliberately rebuilt "
        "on later runs; portable arc metrics and CRS/source hashes are persisted, not version-fragile tree pickles. "
        "Partial Parquet writes become final files only after successful completion. `status.json` must say `complete` "
        "before downstream consumers use a trip's outputs.", "",
        "## Actual candidate example", ""]
    examples = read_or_empty("candidate_sample.csv", [])
    if len(examples):
        examples["trip_id"] = examples.trip_id.astype(str).str.zfill(8)
    example_text = json.dumps(examples.iloc[0].to_dict(),indent=2,default=str) if len(examples) else "No candidate records were generated; unmatched observations remain explicit."
    lines += ["```json", example_text, "```", "",
        "See `candidate_schema.json` for types and meanings. IDs identify source roads, not classifier labels; each "
        "observation can have multiple candidate rows. Timestamp and GPS coordinates are original values; projected "
        "longitude/latitude describe the nearest point on that candidate arc.", "",
        "## Outputs", "",
        "Small reports: `road_network_summary.csv`, `candidate_summary.csv`, `partition_summary.csv`, "
        "`candidate_count_distribution.csv`, `observation_status_summary.csv`, `geometry_issues.csv`, "
        "`projection_length_checks.csv`, `geodesic_candidate_checks.csv`, `training_geometry_sample.csv`, "
        "`candidate_sample.csv`, `failures.csv`, `candidate_schema.json`, `run_metadata.json`. "
        "Plots: `candidate_distributions.png`, `candidate_coverage_and_runtime.png`. Map: "
        "`candidate_training_trip.html` with `candidate_map_metadata.json`. Tests: `pytest-results.xml`, `test-results.txt`. "
        "Final verification: `verification.json` when the verification command has been run.", "",
        "Reusable per-trip files in `data/processed/step2/<partition>/<trip_id>/`: `road_arcs.parquet`, "
        "`candidates.parquet`, `observations.parquet`, `unmatched.parquet`, `network.json`, `status.json`. "
        "Processed data is excluded from Git and belongs only on Laptop 1.", "",
        "## Tests and Step 1 preservation", "", test_text, "",
        "The original 37 tests remain unchanged. New tests cover projections, original IDs, directed/parallel arcs, "
        "degenerate geometries, ranking/radius/limits, unmatched/fallback handling, partition separation, determinism, "
        "search agreement with brute force, typed Parquet, time gaps and reference-independent processing. "
        "The Step 1 full EDA was re-executed in a separate ignored verification directory. Original Step 1 code, tests "
        "and output byte hashes are checked by `verify_step2.py`; original outputs are preserved.", "",
        "## Limitations and Step 3 readiness", "",
        "No supervised alignment, artificial labels, model training, HMM/Viterbi, optimisation, API, UI or deployment "
        "ZIPs were implemented. Endpoint segments approximate roads without detailed curvature; WGS84 is assumed. "
        "Distance alone cannot distinguish direction or identify truth. Candidate caps, fallback and degenerate "
        "connectors need careful assessment in later alignment/temporal modelling. Step 3 must construct defensible "
        "observation/reference alignment, assess candidate misses, engineer features and retain trip-level separation. "
        "Student-test labels must not guide candidate settings. All later work requires explicit approval.", "",
        "This is an AI-assisted working summary of actually executed processing. Review the code and findings for "
        "technical understanding and disclose assistance under the event policy.", ""]
    verification_path = output / "verification.json"
    if verification_path.is_file():
        verified = json.loads(verification_path.read_text())
        lines += ["", "## Final persisted-output verification", "",
            f"Verification status: {verified['status']}. Checked {verified['trips_verified']} trips, "
            f"{verified['observations_verified']} observations and {verified['candidate_rows_verified']} candidate rows "
            "against original source IDs/coordinates and recorded configuration. "
            f"Independent brute-force retrieval agrees on {verified['real_brute_force_observations']} actual observations "
            "across selected training/student-test trips, including fallback/unmatched cases. "
            f"{verified['step1_artifact_hashes_unchanged']} original Step 1 artifact hashes remain unchanged. "
            f"Repeated full-data runs produced byte-identical output in {verified['deterministic_candidate_and_status_files']} "
            "candidate/status Parquet files. PNGs and Folium layer/marker counts were checked. "
            "See verification.json for complete measurements.", ""]
    (output / "STEP2_SUMMARY.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"Summary and small diagnostics updated: {output / 'STEP2_SUMMARY.md'}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=BACKEND/'outputs/step2')
    parser.add_argument("--processed",type=Path,default=BACKEND/'data/processed/step2')
    parser.add_argument("--test-results",type=Path)
    args=parser.parse_args()
    summarize(args.output,args.processed,args.test_results or args.output/'pytest-results.xml')
