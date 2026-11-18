# Laptop 1: data processing, baseline ML and evaluation (Steps 1–9)

This backend directory owns all raw data and analysis. Step 1 implements safe
extraction, typed source-file loaders, GPS/network/reference validation, CSV
reports, Matplotlib plots, a Folium reference map and pytest tests. Step 2 adds
metric roads, reusable spatial indexes and label-free GPS candidates. Step 3 implements supervised alignment/features; Step 4 adds five trained baseline pipelines and trip-held-out CV. Step 7 adds HMM/Viterbi route reconstruction. Step 9 adds the FastAPI server and new-input inference service; Step 10 adds the standalone Streamlit frontend; physical two-laptop checks are owner-reported successful; deployment ZIPs are complete.

## Install on Windows

Use 64-bit CPython 3.12–3.14. This workspace was executed with Python 3.14.0;
other supported versions have not been exercised here. Start PowerShell in the
existing `ML hackathon` workspace containing both original ZIP files:

```powershell
cd .\PathSnap\laptop1_backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No activation or administrator rights are required. `requirements.txt` pins the
eighteen direct dependencies; `requirements-lock.txt` records the complete tested
Python 3.14 environment, excluding pip. To reproduce that environment, substitute
`-r requirements-lock.txt` for `-r requirements.txt`. The lock is environment
specific; use direct requirements for other Python versions. Internet is needed
for dependency installation and browser map assets/tiles. There are no API keys.

## Execute extraction, EDA and tests

From `PathSnap\laptop1_backend`:

```powershell
.\.venv\Scripts\python.exe scripts\extract_dataset.py
.\.venv\Scripts\python.exe scripts\run_eda.py
.\.venv\Scripts\python.exe -m pytest -q --junitxml=outputs\eda\pytest-results.xml
```

Or run the Step 1 Windows startup script after installation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_step1.ps1
```

All Python script defaults resolve from their own source location, so they also
work from another working directory. By default, extraction reads
`..\..\Student folder.zip` and writes `data\raw\train\<trip-ID>\` and
`data\raw\student_test\<trip-ID>\`. It removes only the archive's outer
`Student folder/` directory name, preserving file bytes. Rerunning verifies
identical files and refuses to overwrite a differing raw file. All members are
validated before extraction (paths, permitted splits/extensions, duplicate names,
symlinks); archive CRC and post-extraction SHA-256 are checked.

Paths and thresholds can be configured explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\extract_dataset.py --archive "..\..\Student folder.zip" --destination data\raw --manifest outputs\eda\source_manifest.csv
.\.venv\Scripts\python.exe scripts\run_eda.py --raw data\raw --output outputs\eda --config config\eda.json --manifest outputs\eda\source_manifest.csv
```

`run_step1.ps1` also accepts `-Archive`, `-Raw`, `-Output`. The real-data integration
tests target the standard workspace layout and original ZIP; when using custom
dataset paths, run the EDA with those arguments and keep the supplied data in the
standard layout to run those tests. Missing test prerequisites fail explicitly;
tests are not silently skipped. Full EDA records all supplied trips. Structural
errors produce audit reports and a nonzero exit status; diagnostic quality flags
are reported without changing input data. Manifest mismatches fail before analysis.

## Source schema and reusable Python readers

| Extension | Columns | Dtype | Index |
|---|---|---|---|
| `.track` | longitude, latitude, timestamp_s | float64 | observation_id |
| `.nodes` | longitude, latitude | float64 | node_id |
| `.arcs` | from_node_id, to_node_id | int64 | arc_id |
| `.route` | arc_id | int64 | route_position |

All files are headerless and whitespace-separated. IDs are zero-based source row
positions. Route order and repeated arc entries are preserved. GPS observations
do not have one-to-one labels in the route file. Invalid rows are not dropped.

The importable package is under `src/pathsnap/`. Scripts and pytest configure that
path automatically. For other Python projects, optionally install this package:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

```python
from pathlib import Path
from pathsnap.loaders import discover_trips, load_trip, iter_trips

files = discover_trips(Path("data/raw"), split="train")
trip = load_trip(files[0])
for training_trip in iter_trips(Path("data/raw"), split="train"):
    pass  # Step 1 exposes source data; no training happens here.
```

`iter_trips` requires an explicit split. Discovery rejects overlapping trip IDs
and unrecognised partitions. EDA also flags identical track files across splits.
Each trip's node and arc IDs are local, so totals across trips are row totals,
not counts of unique roads across the dataset.

## Generated evidence in `outputs/eda/`

| File | Contents |
|---|---|
| `source_manifest.csv` | Archive member, relative extracted path, bytes, SHA-256 for all source files |
| `dataset_inventory.csv` | One row per trip: files available/missing, bytes, record counts, malformed/unreadable records |
| `file_inventory.csv` | One row per expected source file: availability, status, row count, size, hash, columns, dtype, index name |
| `partition_summary.csv` | Train/student-test trip and record totals |
| `gps_quality.csv` | Per-trip missing/invalid/duplicate coordinates, time ordering/intervals, geographic extent, distances, speeds, outlier counts |
| `gps_steps.csv` | Every consecutive source observation pair with interval, haversine displacement, speed and flags |
| `network_quality.csv` | Per-trip directed-network topology and reference validity/continuity/approximate length |
| `anomalies.csv` | Individual flags with partition, trip, category, source row/route position and explanation; flags may overlap |
| `anomaly_summary.csv` | Flag totals by partition/category/severity |
| `*.png` | Five figures covering per-trip GPS counts, time intervals/distances, network size/connectivity, anomalies, GPS extent/speed |
| `representative_training_trip.html` | Interactive original GPS/network/reference map with start/end, legend and layer controls |
| `map_metadata.json` | Training-only selection criterion, displayed counts and network bounds |
| `run_metadata.json` | Runtime, Python/dependency versions, thresholds, source fingerprint and audit completeness |
| `EDA_SUMMARY.md` | Measured findings, assumptions, limitations and later-stage requirements |
| `pytest-results.xml` | Actual test execution result generated by pytest |
| `test-results.txt` | Captured final test run output |

CSV trip identifiers retain zero padding in the file; specify
`dtype={"trip_id": str}` when reading them with pandas (spreadsheet software may
otherwise infer integers). Counts of anomalies are per diagnostic category and
can overlap. Zero flags do not prove error-free physical truth. GPS distance is
the sum of consecutive valid haversine steps; speed requires a positive time
interval; missing steps are not bridged. Network/reference distance uses straight
endpoint segments, not detailed road curvature. See `config/eda.json` and the EDA
summary for every threshold and map display rule. No cleaning is performed.

The map embeds all valid GPS points and reference arcs for its selected training
trip. Road-network display uses only arcs whose bounding boxes overlap GPS bounds
padded by 0.01 degrees; the underlying network and connectivity analysis remain
complete. Folium/Leaflet/OpenStreetMap assets require internet in the browser.

## Separation and next steps

Raw/processed data and models remain only on Laptop 1. `data/processed/` and
`models/` were empty in Step 1. Step 2 populates processed candidates; models
remain empty. Do not send these inputs, reference
labels or this backend directory to Laptop 2. Later, a FastAPI service will own
inference and server-side evaluation; Laptop 2 will consume HTTP responses using
an environment-configured backend URL.

Raw files, source ZIPs, virtual environments, caches, secrets and temporary
artifacts are excluded from Git. Measured EDA reports/plots and source code are
retained. No Git repository was initialised or published by this step. Official
documents and the student-test policy discrepancy are described in
`../docs/REQUIREMENTS_REVIEW.md`. The generated EDA summary is an AI-assisted
working record; review it and understand the implementation before any university
submission. Steps 1–12 are complete; see final deployment references.

## Step 1 retrospective: measured data and implementation

All 80 training and 10 student-test trips were inspected: 360 files, 227,577 GPS
observations, 2,024,174 per-trip node rows, 4,061,945 directed arc rows and 34,813
reference entries. These network totals do not count globally unique roads.

| Partition | GPS observations | Nodes | Arcs | Reference entries |
|---|---:|---:|---:|---:|
| train | 208,417 | 1,933,485 | 3,876,904 | 31,336 |
| student_test | 19,160 | 90,689 | 185,041 | 3,477 |

EDA checked missing/invalid coordinates, duplicate records/coordinates, timestamp
ordering/intervals, gaps, distances, speeds/jumps, geographic extent, network
connectivity/components/isolates and reference validity/continuity. All source
files parsed; missing GPS cells, invalid coordinates, backwards intervals,
invalid endpoints and invalid reference arc IDs were all zero. This does not
prove physical ground truth is correct.

Measured issues (train/student-test): 1,216/44 gaps >5 s; 1,524/8 speed flags
>180 km/h; 4/0 displacements >500 m; 13/1 disconnected reference pairs;
1,775/128 duplicate directed endpoint pairs; 572/6 repeated coordinate records.
The maximum gap is 153,064 s at train/00000090 observation 534→535. Weak-component
counts range from 1 to 421. No cleaning or route rewriting was performed.

The five EDA PNGs show per-trip GPS counts, sampling intervals/trip distances,
network size/connectivity, missing/anomalous data, and GPS extent/speeds. The
original Folium map shows train/00000083 with 1,507 GPS points, 25,393 network
arcs and 143 supplied reference entries. It is a supplied-reference map, not a
prediction. Step 2 creates a separate candidate-only map.

| Step 1 module/function | Inputs → outputs / responsibility |
|---|---|
| `extraction.extract_dataset` | ZIP/destination/manifest paths → extracted count; validates members, preserves bytes, records hashes and refuses changed-file overwrites |
| `loaders.load_track/load_nodes/load_arcs/load_route` | Source path → typed DataFrame preserving row IDs; malformed/blank records raise `DataFormatError` with line numbers |
| `loaders.discover_trips` | Raw root, optional split → `TripFiles` list; rejects overlapping IDs/unrecognised partitions |
| `loaders.load_trip` | `TripFiles` → all four source tables including reference; used by EDA, not candidate processing |
| `loaders.iter_trips` | Raw root, explicit split → iterator over complete trips |
| `quality.haversine_m` | Longitude/latitude pairs in degrees → spherical metres |
| `quality.valid_coordinates` | Coordinate table → boolean validity mask |
| `quality.gps_quality` | Track and thresholds → per-trip metrics and consecutive-step table, without sorting/deleting |
| `quality.network_quality` | Nodes/arcs/reference → topology, reference continuity and anomaly positions |
| `eda.run_eda` | Raw/output paths, thresholds, optional manifest → CSVs, figures, map and metadata |
| `visuals.make_plots/make_map` | Audited tables or complete trip → original EDA figures/Folium HTML |

The original 37 tests passed before Step 2. They validate loading, zero-based IDs,
bad records, timestamps, distances, connectivity/reference checks, archive
integrity, partition separation and a small EDA fixture. Synthetic fixtures are
confined to tests; project EDA uses supplied data. Passing tests verifies those
behaviours, not ML accuracy. Step 2 preserves the original test/source/output
bytes. A full Step 1 rerun in ignored `.cache/step1_verification/` reproduced all
eight EDA CSV tables exactly, without overwriting original evidence.

## Step 2: directed roads and metric geometry

`RoadNetwork(nodes, arcs, ...)` in `road_network.py` retains every original arc
in an arc-indexed table with directed endpoints, geodesic `arc_length_m`,
`projected_length_m`, initial geodesic `arc_bearing_deg` (clockwise from true
north), and `geometry_status`. `arc(arc_id)` performs original-ID lookup. Parallel,
coincident and reversed arcs keep separate IDs. An optional `to_multidigraph()`
uses source arc IDs as edge keys and retains valid zero-length connectors at
zero cost. Invalid endpoints/coordinates remain explicitly recorded.

The efficient primary representation is the table plus metric node coordinates
and LineStrings, avoiding millions of Python graph-edge objects. Each trip has
its own local ellipsoidal azimuthal equidistant CRS, centred on circular mean
longitude and mean latitude of valid network nodes. WGS84/EPSG:4326 is an explicit
assumption because the source omits a datum. `MetricProjection.xy(lon, lat)`
returns metre coordinates; `lonlat(x, y)` inverts them. PyProj always uses
`always_xy=True`; Folium markers use latitude then longitude. No source coordinate
is modified and no single local CRS is applied to the worldwide collection.

Networks extending more than 300 km from their centre fail clearly; observations
outside the bound are retained as unmatched `projection_out_of_extent`. Every
valid arc's projected length is checked against PyProj Geod, with relative error
limited to 0.005. The 106 actual zero-length segments stay in road tables and
`geometry_issues.csv` but are excluded from the candidate index. No endpoint or
coordinate-invalid source arcs were found.

`geodesic_point_segment_m` independently minimises WGS84 distance along each
sampled short endpoint-to-endpoint geodesic, including endpoints. The runner
checks rank-1 candidates at up to eight uniformly spaced source observations per
trip where matched. Tolerance is 0.1 m + 0.5% of independent distance. These checks
are separate from arc-length checks and never use reference labels.

## Step 2: indexing and candidate selection

`build_spatial_index()` lazily builds one Shapely STRtree per trip and returns the
same tree thereafter. It indexes complete projected segments, not midpoints. A
separate `index_arc_ids` array maps dense tree positions to original source IDs,
including geometrically identical arcs. Later runs rebuild trees instead of
loading version-fragile pickles; derived arc metrics, CRS and source hashes are
persisted for later reuse.

`generate_candidates(network, track, dataset_partition, trip_id, config)` yields
candidate/status DataFrame batches. It projects a batch once, queries bounding
boxes, then enforces exact point-to-segment distance ≤50 m. Only an empty primary
result triggers the bounded 150 m fallback. Candidates sort by distance, then
original arc ID for exact ties, and are capped at eight per observation. Every
candidate retains its original observation/arc/node IDs. Closest points are
inverse-transformed; `position_along_arc` is normalized directed segment fraction
(0=from-node, 1=to-node), not inferred vehicle position or ground truth.

EVERY source observation has a status row, including no-candidate cases. Status is
`matched_radius`, `matched_fallback`, `unmatched`, `invalid_coordinates`, or
`projection_out_of_extent`. Timestamps, previous observation IDs, elapsed time and
gap flags cross batch boundaries unchanged. Step 1 GPS gap/speed/jump flags join
after geometric selection. They do not remove observations or change candidates.
The runner never reads `.route`; reference discontinuities remain untouched.

| `CandidateConfig` parameter | Default | Meaning |
|---|---:|---|
| `search_radius_m` | 50 | Inclusive primary radius, metres |
| `max_candidates` | 8 | Retained rows per observation |
| `fallback_radius_m` | 150 | Bounded fallback only after empty primary; `null` disables |
| `batch_size` | 1024 | Observations per query/output batch |
| `max_projection_extent_m` | 300000 | Local projection/observation guard |
| `max_projection_scale_error` | 0.005 | Maximum relative projected/geodesic arc-length error |
| `min_segment_m` | 0.000001 | Minimum projected length for spatial indexing |

`config/candidates.json` supplies defaults; invalid configuration fails clearly.
The radii are practical engineering bounds, supported by an independent
label-free, unbounded-nearest diagnostic on 32 evenly spaced GPS observations per
training trip. The 2,560 samples have median nearest distance 4.388 m, p95 16.173 m,
p99 77.578 m; 98.4375% are within 50 m and 99.96094% within 150 m. This equally
weights trips and differs from full observation-weighted coverage. Student-test
labels did not set parameters. The cap bounds storage/later state-space size but
truncates some sets; reference candidate recall remains unknown until Step 3.

## Stable schema and actual candidate example

`CANDIDATE_SCHEMA` uses string partition/trip/source, int64 observation/arc/node
IDs, int32 rank, boolean fallback and float64 geometry/timestamp values:

```text
dataset_partition, trip_id, gps_observation_index, timestamp_s,
gps_longitude, gps_latitude, candidate_arc_id, from_node_id, to_node_id,
distance_to_arc_m, candidate_rank, arc_length_m, arc_bearing_deg,
projected_longitude, projected_latitude, position_along_arc,
candidate_source, fallback_used, search_radius_used_m
```

Actual train/00000083 observation 0: timestamp 1 s; original GPS
`(37.818644,55.707001)`:

| Rank | Arc ID | From→to | Distance (m) | Directed fraction | Source |
|---:|---:|---|---:|---:|---|
| 1 | 4783 | 2369→2368 | 0.123860505 | 0.749220884 | radius |
| 2 | 4781 | 2368→2369 | 0.123860505 | 0.250779116 | radius |

Both project to about `(37.818642788,55.707001877)` while preserving direction/ID.
They are geometric alternatives, not labels. Tiny distance differences between
reversed geometries are floating-point roundoff; sorting uses exact distances.
`OBSERVATION_SCHEMA` adds original observation/time information, primary/unlimited/
retained counts, truncation/fallback flags, status, bounding-box hit counts and
Step 1 diagnostic flags. See `candidate_schema.json` for all field types.

## New modules, functions and executable workflow

| Module/function | Inputs → outputs |
|---|---|
| `projection.MetricProjection.from_nodes/xy/lonlat` | Nodes/coordinates → reusable guarded metric transforms and CRS metadata |
| `road_network.RoadNetwork` | Typed nodes/arcs, guards → arc table, geometry array, lazy index; `summary()` returns network metrics |
| `candidates.CandidateConfig` | JSON-compatible parameters → validated immutable configuration |
| `candidates.generate_candidates` | Network/track/explicit partition/trip/config → candidate and observation-status batches |
| `step2.process_trip` | TripFiles, processed path, config, optional GPS flags → streamed Parquet, metadata, small metrics/samples |
| `step2.run_step2` | Configurable raw/output/processed paths and selection → selected-run reports/plots/map/metadata |
| `step2.validate_candidate_batch` | Network/batch/config → ID, endpoint, radius, rank and fraction assertions before persistence |
| `step2.MemoryMonitor` | Processing context → process RSS sampled every 20 ms, including native libraries |
| `candidate_visuals.candidate_map/candidate_plots` | Actual generated data → separate candidate-only Folium map and two PNGs |
| `scripts/summarize_step2.py` | Output/processed paths and actual JUnit → technical summary and small geometry/count/status CSVs |
| `scripts/verify_step2.py` | Source/generated paths → full consistency, real brute force, preservation/determinism checks and verification JSON |

From `PathSnap\laptop1_backend`, install updated requirements, then:

```powershell
# Selected training trip, with separate exploratory output folders.
.\.venv\Scripts\python.exe scripts\generate_candidates.py --partition train --trip 00000083 --output outputs\step2\single_trip --processed data\processed\single_trip

# All training trips or all student-test trips independently.
.\.venv\Scripts\python.exe scripts\generate_candidates.py --partition train --output outputs\step2\train_only --processed data\processed\train_only
.\.venv\Scripts\python.exe scripts\generate_candidates.py --partition student_test --output outputs\step2\student_only --processed data\processed\student_only

# Canonical full run, complete tests, summary and verification.
.\.venv\Scripts\python.exe scripts\run_step2.py
.\.venv\Scripts\python.exe -m pytest -q --junitxml=outputs\step2\pytest-results.xml
.\.venv\Scripts\python.exe scripts\summarize_step2.py
.\.venv\Scripts\python.exe scripts\verify_step2.py

# Rebuild roads/index only; keep this separate from complete candidate outputs.
.\.venv\Scripts\python.exe scripts\build_road_network.py --partition train --trip 00000083 --output outputs\step2\network_check --processed data\processed\network_check

# Configurable radius, cap, fallback and batches.
.\.venv\Scripts\python.exe scripts\generate_candidates.py --partition train --trip 00000083 --radius 30 --max-candidates 6 --no-fallback --batch-size 512 --output outputs\step2\alternative --processed data\processed\alternative
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_step2.ps1
```

`--raw`, `--processed`, `--output`, `--config` configure paths. `--fallback-radius`
sets the bound; `--no-map` skips rendering; `--trip` can repeat. Defaults resolve
from source locations without usernames. Use distinct directories for alternative
configurations/scopes. The Windows script runs processing; tests, summaries and
verification are explicit commands so evidence reflects actual execution.

## Step 2 outputs, performance and tests

`data/processed/step2/<partition>/<trip_id>/` contains:

| File | Purpose |
|---|---|
| `road_arcs.parquet` | Every original arc with computed metrics and exclusion status |
| `candidates.parquet` | Multiple typed geometric candidates per observation; no labels |
| `observations.parquet` | Every source observation, counts, time/status/diagnostic flags |
| `unmatched.parquet` | Explicit zero-candidate observations; typed empty file when none |
| `network.json` | Source hashes, CRS, configuration and measured metrics |
| `status.json` | complete/failed/network_only state, input/config identity and counts |

Only `state="complete"` candidate trips should enter later stages. Writers append
bounded batches using Zstandard compression and finalize files after successful
closure. Re-execution rebuilds indexes/replaces generated outputs and never
changes raw sources. No enormous full-candidate CSV is created. Empty candidate
files retain their schema. One network is held at a time and released between
trips; aggregate reporting retains only distances/counts rather than all networks.

`outputs/step2/` contains network/candidate/partition summaries, count/status
distributions, geometry issues, actual candidate samples, projection/geodesic
checks, training geometry diagnostics, run/schema/map metadata, failures, two
PNG figures, candidate HTML, actual tests, preservation/determinism evidence,
verification and `STEP2_SUMMARY.md`. Processed datasets stay only on Laptop 1
and are excluded from Git. Reports are small enough to retain as evidence.

The candidate map shows all original GPS points on the representative training
trip, a 1 km local road window, and candidates for the selected observation plus
up to ten neighbours on each side. Selected segments and closest-point connectors
use separate colours/tooltips. Rendering limits do not change generated records.
There is no reference route, ML prediction or reconstructed route on this map.

<!-- STEP2_RESULTS_START -->
| Partition | Trips | GPS rows | Candidate rows | Coverage | Unmatched | Fallback observations |
|---|---:|---:|---:|---:|---:|---:|
| train | 80 | 208,417 | 1,227,056 | 99.904998% | 198 | 18,032 |
| student_test | 10 | 19,160 | 113,639 | 100.000000% | 0 | 0 |

All 90 trips succeeded; 0 failures. Geometric coverage is not label recall or prediction accuracy. All 227,577 source observations have status records. The 198 unmatched rows are train/00000002 (191) and train/00000043 (7).

Measured processing-pipeline wall time: **120.700 s**; summed trip time 99.095 s; STRtree build time 1.984 s; candidate-generation time 10.827 s. Maximum sampled process RSS during trip processing: **426.969 MiB**. Separate tests, summary regeneration and verification are excluded. These are execution measurements, not comparative benchmarks.

Independent geodesic checks: **720/720 passed**, maximum absolute point-to-segment discrepancy 0.000211032511 m. All valid arc lengths were also checked; maximum relative error 9.43605594e-06. The index excludes 106 zero-length segments while preserving them in road tables and the optional graph.

**Tests: 71 passed, 0 failed, 0 errors, 0 skipped (35.61 s).** All original 37 tests remain unchanged; the complete Step 1 rerun reproduces all eight EDA CSV tables. Saved candidate/status outputs are independently verified against source IDs and the real-data brute-force oracle; see `outputs/step2/verification.json`.

All retained candidate distances (median / p95): train **16.831 / 118.963 m**; student test **14.234 / 43.483 m**. Nearest retained distances (median / p95; excludes unmatched): train **4.966 / 83.020 m**; student test **4.760 / 16.003 m**. The cap truncated 96,077 observation candidate sets; Step 3 must assess true reference candidate misses after alignment.
<!-- STEP2_RESULTS_END -->

New dependencies: Shapely 2.2.0, PyProj 3.8.0, PyArrow 26.0.1, psutil 7.2.2.
PyProj raises the combined backend minimum to Python 3.12; execution used 3.14.0.
Step 1 source modules/tests remain unchanged. Passing geometry tests does not
establish prediction accuracy; no model exists yet.

## Limitations, progress and Step 3

Distance cannot resolve direction, parallel-road ambiguity, GPS noise or bad
references. Top-eight truncation may remove a relevant arc even when retrieval
coverage is high. Endpoint segments approximate curved roads; the datum is
assumed. Projection guards intentionally reject unsuitable wide unseen networks.
Zero-length connectors and large time gaps need later temporal/routing care.

Step 3 must design defensible observation/reference alignment, evaluate candidate
misses and construct justified features, with training-only fitting and trip-aware
validation. Supplied GPS/reference sequences differ in length and references have
14 continuity breaks. Never zip them into labels or treat nearest roads as truth.

- [x] Step 1: setup, parsing, EDA, reference map and original 37 tests.
- [x] Step 2: roads, projections, STRtree, label-free candidates, persisted
  statuses, diagnostics, tests and documentation.
- [x] Step 3 alignment and ML feature preparation; see the current stage reference.
- [x] Steps 4–5: baseline models and detailed evaluation.
- [ ] Later optimisation, HMM/Viterbi, API/UI, integration, packages/report.

Changelog — 2026-11-18: preserved Step 1 evidence and expanded its explanation;
added Step 2 road/candidate processing, Parquet data, diagnostics, tests and viva
documentation. Continuous README maintenance is retained in `../AGENTS.md`.

<!-- STEP3_CURRENT -->

## Historical Step 3 completion record

Earlier Step 1/2 descriptions above are historical. Step 3 now adds non-destructive cleaning, training-only ordered-reference supervision, recall and reusable inference features. Model training and all subsequent application stages remain pending.

- [Step 1 detailed reference](../docs/steps/STEP_01_README.md)
- [Step 2 detailed reference](../docs/steps/STEP_02_README.md)
- [Step 3 detailed reference](../docs/steps/STEP_03_README.md)

Actual Step 3 results: 90 trips; 140,392 eligible training alignments; conditional candidate recall 96.831016%; 135,943 positive and 605,573 negative candidate targets. 83 tests, 0 failures, 0 errors, 0 skipped. Student-test features remain unlabelled; frontend remains unimplemented.

- [x] Step 3 cleaning/alignment/features/tests/documentation.
- [x] Step 4 five baseline models and trip-grouped cross-validation; see current reference.

Changelog: 2026-11-18 — implemented Step 3 and created separate retrospective Step 1/2 and current Step 3 references. No new dependencies. Use the Step 3 reference for commands, function contracts, complete feature list and limitations.


Measurement note: Step 3 uses WGS84 ellipsoidal Geod displacements and flags 1,541 speed steps above 180 km/h; Step 1 used spherical haversine and flagged 1,532. The nine extra threshold crossings arise from the distance convention, not altered GPS observations. Cleaning retention is unchanged. Feature and label Parquet schemas were checked across all trips. Step 3 contains 741,516 eligible labelled training candidate rows (135,943 positive; 605,573 negative); 485,540 other training candidate targets remain null/excluded. Student-test candidate features have no label table.

<!-- STEP4_CURRENT -->

## Historical Step 4 completion record

[Step 4 detailed reference](../docs/steps/STEP_04_README.md). Earlier stage status statements above describe historical boundaries. Five baseline pipelines completed five-fold trip-held-out CV (64 train / 16 held-out trips per fold) and separate full-data fits. Only eligible training data was fitted; student-test labels were not used. Preliminary leader: hist_gradient_boosting. 101 tests, 0 failures, 0 errors, 0 skipped. Models/step4 and outputs/step4 on Laptop 1 contain actual artifacts/results. Laptop 2 remains a documentation-only HTTP UI placeholder with no backend/ML dependencies.

Commands, model explanations, full feature list, function contracts, metrics, model limitations and reproduction details are in the linked step reference. New backend dependencies: Scikit-learn, Joblib, threadpoolctl. No new frontend dependencies.

- [x] Step 4 models/CV/artifacts/verification/documentation.
- [x] Step 5 detailed evaluation; see current reference. Optimisation, decoder, API/UI and final ZIPs remain pending.

Changelog — 2026-11-18: completed Step 4 baseline training and preserved Steps 1–3 evidence. Step 5 requires explicit approval.

<!-- STEP5_CURRENT -->

## Historical Step 5 completion record

[Step 5 detailed reference](../docs/steps/STEP_05_README.md). Earlier stage boundaries above are historical. All five saved Step 4 pipelines were evaluated without refitting on 113,639 candidate rows / 19,160 student-test observations. There are 15,121 qualified alignments, 14,547 reference-present groups and 574 candidate misses. Reference labels are evaluation-only and remain separate from predictions and Step 3 training data.

Promising untuned Step 6 model: **random_forest**; backup **hist_gradient_boosting**. The difference is small; see actual Macro F1, Top-1, log loss, CV variability, continuity and timing in the stage reference. Evaluation outputs are in laptop1_backend/outputs/step5/. 115 tests, 0 failures, 0 errors, 0 skipped. No dependencies changed.

- [x] Step 5 evaluation/verification/plots/documentation.
- [x] Step 6 Bayesian optimisation; see current reference. Decoder, API/UI and deployment remain pending. Laptop 2 remains an unimplemented HTTP UI placeholder and contains no reference labels/models/backend dependencies.

Changelog — 2026-11-18: completed Step 5 using frozen Step 3 alignment and Step 4 models/OOF evidence; added reusable evaluation/continuity modules, scripts, tests, plots and reference. No models were trained. Step 6 requires explicit approval.

<!-- STEP6_CURRENT -->

## Historical Step 6 completion record

[Step 6 detailed reference](../docs/steps/STEP_06_README.md). Primary classifier: **random_forest**; backup: **hist_gradient_boosting**, chosen from training-only five-fold CV after a bounded TPE search. Student-test results were a post-selection report, not an optimisation objective or selection input. Original Steps 1–5 artifacts remain unchanged.

Completed trials: random_forest: 5; hist_gradient_boosting: 6. Actual comparisons, parameters, timings and limitations are in the step reference and laptop1_backend/outputs/step6/. Two full-training pipelines and feature/configuration metadata are under laptop1_backend/models/step6/. 126 tests; 0 failures; 0 errors; 0 skipped. Optuna is the only new direct dependency; Laptop 2 dependencies are unchanged and the frontend remains unimplemented.

- [x] Step 6 optimisation/CV/full fits/verification/documentation.
- [x] Step 7 HMM/Viterbi; see current reference. API/UI, route visualisation and deployment remain pending.

Changelog — 2026-11-18: added bounded persistent Optuna studies, tuned-model comparison and training-CV classifier selection. Earlier step references remain intact. Step 7 requires explicit approval.

<!-- STEP7_CURRENT -->

## Historical Step 7 completion record

[Step 7 detailed reference](../docs/steps/STEP_07_README.md). The frozen tuned Random Forest supplies emissions to directed, bounded shortest-path transitions and deterministic log-space Viterbi. 10 student-test trips / 19160 observations decoded into 277 explicit segments. Qualified candidate-present Top-1: classifier 85.179075%; Viterbi 87.145116% (+1.966041 percentage points). See the detailed report for candidate-miss coverage, breaks, unreachable pairs, per-trip results and limitations. No student references chose decoder parameters; no model was retrained. Steps 1–6 artifacts and step READMEs remain unchanged.

New modules: hmm.py, routing.py, transitions.py, viterbi.py and route_evaluation.py. Scripts: decode_step7.py, report_step7.py, verify_step7.py and document_step7.py; exact PowerShell commands and input/output contracts are in the linked reference. Outputs are under laptop1_backend/outputs/step7/, including selected candidates, intermediate route arcs, topology diagnostics, runtime, three static geographic examples, verification and STEP7_SUMMARY.md. No dependencies added; Laptop 2 remains a documentation-only placeholder.

Full suite: 150 tests, 0 failures, 0 errors, 0 skipped. Decoding 169.794 s; peak sampled process RSS 371.676 MiB.

- [x] Step 7 HMM/Viterbi, directed route reconstruction, evaluation, verification and documentation.
- [x] Step 8 saved-output Folium visualisation; see current reference. API/UI, two-laptop integration and deployment remain pending.

Changelog — 2026-11-18: completed Step 7 with the frozen classifier and existing feature/network/probability artifacts. Stop here; Step 8 requires explicit approval.

<!-- STEP8_CURRENT -->

## Historical Step 8 completion record

[Step 8 detailed reference](../docs/steps/STEP_08_README.md). Ten interactive student-test Folium maps cover 19,160 observations and 277 separate saved route segments. Layers show GPS/candidates, frozen RF selections, Viterbi-selected arcs, partial-arc reconstructed routes/intermediates, optional evaluation-only references, gaps/unreachable markers, direction and bounded context. Before/After/Compare controls and the relative trip index work without a Python server; local libraries are bundled and OSM tiles need internet.

Laptop 1 commands (from laptop1_backend):

```powershell
.\.venv\Scripts\python.exe scripts\visualize_step8.py
Start-Process .\outputs\step8\index.html
```

Examples: 00000009 improvement (+6.667 pp), 00000056 candidate ambiguity, 00000076 gaps and 00000065 decline (−3.306 pp). Metrics are unchanged Step 7 measurements. New map_visualization.py and visualize_step8.py/preview_step8.py/verify_step8.py/document_step8.py expose layer/HTML builders, generation, live browser verification, integrity checks and reporting; contracts and complete commands are in the reference. Outputs/step8 holds maps, local assets, GeoJSON, metadata, relative index/examples, previews, summary and verification. No dependencies added.

17 focused tests; full suite **167 passed**, zero failures/errors/skips. All ten maps and index opened in Chrome; popups/toggles/reference isolation worked. Verification checked 147,396 coordinate pairs and 745 unchanged protected files. A second run reused all ten maps. No production classifiers, optimiser or decoder jobs were rerun; earlier per-step READMEs remain intact.

- [x] Step 8 interactive visualisations, portability, examples, tests, verification and documentation.
- [x] Step 9 FastAPI; see current reference. Streamlit, two-laptop integration and deployment remain pending. Laptop 2 remains a documentation-only placeholder.

Changelog — 2026-11-18: completed Step 8 using saved Step 7 data. Stop here; Step 9 requires explicit authorisation.

<!-- STEP9_CURRENT -->

## Historical Step 9 completion record

[Step 9 detailed reference](../docs/steps/STEP_09_README.md). Laptop 1 now runs FastAPI with startup-loaded frozen RF, saved Step 7 decoder settings, demo responses/maps from Steps 7–8 and new-trajectory inference with explicit directed road context. Required endpoints: GET /health, GET /trips, GET /trips/{trip_id}, POST /predict and GET /trips/{trip_id}/map. Swagger: /docs; contracts: /openapi.json. Supporting /maps and /map-assets keep existing HTML navigation/assets working without rewriting saved files.

From laptop1_backend:

```powershell
.\.venv\Scripts\python.exe scripts\serve_api.py --host 0.0.0.0 --port 8000
```

Host/port are configurable; PATHSNAP_CORS_ORIGINS sets browser origins. Laptop 2 will later use http://<LAPTOP1_HOST>:8000, with network/firewall access; no fixed device IP is stored. No frontend or two-laptop integration has been implemented/tested. New modules: api.py, api_service.py, api_schemas.py; serve_api.py, verify_step9.py and document_step9.py provide startup, real HTTP checks and reporting. New direct backend dependencies: FastAPI, Pydantic, Uvicorn, HTTPX; Laptop 2 dependencies unchanged.

25 focused API tests; **192 full-suite tests passed**, zero failures/errors/skips (one upstream TestClient deprecation warning). Real loopback HTTP verified a saved 262-observation trip/map and a new five-point, twelve-node, ten-arc request: five matched observations, one segment; no invented accuracy. Model loaded once. All 850 protected files unchanged; no earlier training/optimisation/route/map jobs rerun. Schemas, sample requests/responses, server log and verification are in outputs/step9/. The temporary verification server was stopped.

Requests are limited to 500 GPS points, 5,000 nodes, 10,000 arcs and 2 MiB. Missing context/invalid input is rejected; disconnected/missing-candidate segments stay explicit. New requests never accept references and return metrics=null; saved demos use existing qualified evaluation metrics. Detailed contracts, limitations and all PowerShell test/connection commands are in the reference.

- [x] Step 9 FastAPI, schemas, bounded inference, demo reuse, CORS/logging, tests, real HTTP verification and documentation.
- [ ] Streamlit/frontend, real two-laptop integration and deployment packaging; not started.

Changelog — 2026-11-18: completed Step 9 on Laptop 1 and preserved Steps 1–8 artifacts/references. Stop here; further stages require explicit authorisation.

<!-- STEP10_CURRENT -->

## Historical Step 10 completion record

[Step 10 detailed reference](../docs/steps/STEP_10_README.md). Laptop 2 now has a standalone Streamlit app and Requests HTTP client. It supports a session-persistent backend URL (default http://localhost:8000), readiness/catalog, saved trip metrics, embedded existing maps and a JSON prediction form. Returned geometry/diagnostics are displayed without computing routes or new metrics. Reference data remains evaluation-only. No backend imports, project datasets or trained models are required by the frontend.

Frontend launch, from laptop2_frontend:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Install its own requirements.txt in its own environment. In the sidebar, enter http://<LAPTOP1_HOST>:8000 and click Connect / refresh; localhost refers to the machine running the frontend. New files: app.py, api_client.py, example_request.json, .streamlit/config.toml, tests/test_client.py, tests/test_app.py and scripts/verify_backend.py; frontend requirements/lock and .env.example are updated. Full setup/function/API details are in the linked reference.

**15 frontend tests passed**. Real local HTTP/AppTest loaded ten trip choices and the saved 262-observation demo/map; Streamlit server health passed. No live prediction was submitted, so the backend decoder was not rerun. 868 protected files remain unchanged. Prior backend results remain 192 tests passed (not rerun in Step 10). Temporary local services are stopped. Tests and verification reports are under laptop2_frontend/outputs/step10/ and laptop1_backend/outputs/step10/.

- [x] Step 10 standalone HTTP-only Streamlit UI, focused tests, local cached-demo verification and documentation.
- [x] Steps 11–12 integration evidence and final packaging; see the final record below.

Changelog — 2026-11-18: completed Step 10. Stop here; real two-laptop integration and final ZIP packaging require separate authorisation.

<!-- FINAL_STEPS_11_12 -->

## Current status: Steps 11–12 completed

[Step 11 integration record](../docs/steps/STEP_11_README.md) documents the owner-reported successful two-laptop connectivity, live POST, saved-map/Before/After/Compare, invalid-input recovery and observed timeout handling. These physical checks were not independently repeated; exact logs/timings were not supplied.

[Step 12 deployment reference](../docs/steps/STEP_12_README.md) records the separate allowlisted Laptop1_Backend.zip and Laptop2_Frontend.zip under PathSnap/releases/, setup/run instructions, manifests/checksums and actual extracted-package smoke results. Both archives passed CRC/hash/import checks; the extracted API served all ten demo JSONs/maps, a tiny live prediction and local assets, and the extracted frontend imported, reached the API over HTTP and started Streamlit. Existing installed dependencies were used; a clean-machine install was not tested.

Historical test runs remain 192 backend and 15 frontend tests passed, not a newly rerun combined suite. All 887 protected files are unchanged. See FINAL_PROJECT_SUMMARY.md at the project root for five-model CV/student comparisons, actual Optuna gains/degradation, RF + Viterbi results and exact commands. No training, Optuna, old-trip reconstruction or map generation repeated. No working prediction logic changed.

- Laptop1_Backend.zip: 12.097 MiB; 159 files; SHA-256 in releases/checksums.sha256.
- Laptop2_Frontend.zip: 0.012 MiB; 12 files; SHA-256 in releases/checksums.sha256.

- [x] Step 11 attributed physical integration record.
- [x] Step 12 separate deployment archives, validation, minimal extracted smoke checks, final summary and documentation.

Use each extracted archive README: Python 3.14 → create .venv → install requirements-lock.txt → backend scripts/serve_api.py (port 8000) / frontend streamlit run app.py. Enter Laptop 1’s reachable address in Backend URL. No virtual environments/caches/training-only data ship in the archives.

Changelog — 2026-11-18: completed final integration documentation and deployment packaging. Stop after Step 12.
