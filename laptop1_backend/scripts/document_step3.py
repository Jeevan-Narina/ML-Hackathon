"""Build stage documentation from actual persisted results, without rerunning EDA."""
from pathlib import Path
import json,sys,xml.etree.ElementTree as ET
import pandas as pd
B=Path(__file__).resolve().parents[1];ROOT=B.parent;O=B/'outputs/step3'
sys.path.insert(0,str(B/'src'))
from pathsnap.features import FEATURES

def main():
    r=json.loads((O/'results.json').read_text());meta=json.loads((O/'run_metadata.json').read_text())
    table=pd.read_csv(O/'trip_summary.csv',dtype={'trip_id':str}); audit=pd.read_csv(O/'cleaning_audit.csv')
    suites=list(ET.parse(O/'pytest-results.xml').getroot().iter('testsuite'))
    test={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    steps=ROOT/'docs/steps';steps.mkdir(exist_ok=True)
    common='''
## Installation and execution environment

PowerShell starts at `ML hackathon`; then `cd .\\PathSnap\\laptop1_backend`.
Use the existing `.venv` (tested Python 3.14.0; minimum 3.12 for combined backend).
Install with `.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`.
Raw data, derived tables and ML operations belong on Laptop 1. Laptop 2 remains
an independent future HTTP UI. No model, HMM decoder, API, UI or ZIP is created here.
'''
    descriptions={1:('Workspace setup, source parsing and EDA',
      'Preserve supplied bytes, establish actual schema and inspect quality before modelling.',
      'extraction.extract_dataset: ZIP and paths -> verified raw files/SHA-256 manifest; loaders.load_*: source path -> typed indexed DataFrame; quality.gps_quality/network_quality: tables -> metrics; eda.run_eda: paths/config -> CSVs, plots and map.',
      'Safe ZIP extraction -> typed whitespace readers -> haversine GPS metrics -> NetworkX directed multigraph diagnostics -> Matplotlib and Folium.',
      'scripts\\extract_dataset.py\n.\\.venv\\Scripts\\python.exe scripts\\run_eda.py',
      'outputs/eda: nine CSV files, five PNGs, original reference map, EDA_SUMMARY.md and metadata.',
      '80 train / 10 student-test trips; 227,577 GPS, 2,024,174 node and 4,061,945 arc rows. 1,260 gaps >5 s; 1,532 speed flags; four jumps; 14 reference breaks. No missing files or invalid references. Original 37 tests passed; full Step 1 rerun reproduced eight EDA CSV tables during Step 2.'),
      2:('Directed roads, metric indexing and geometric candidates',
      'Bound GPS search cost and retain multiple plausible directed alternatives without using labels.',
      'MetricProjection.from_nodes/xy/lonlat: coordinates -> checked local metric transforms; RoadNetwork: nodes/arcs -> indexed directed arc tables; generate_candidates: network/track/partition/config -> candidate/status batches; step2.run_step2 -> streamed outputs.',
      'Local WGS84 AEQD -> geodesic length checks -> complete-segment STRtree bbox search -> exact distance filter -> 50 m primary/150 m fallback -> distance rank -> top eight.',
      'scripts\\run_step2.py\n.\\.venv\\Scripts\\python.exe -m pytest -q --junitxml=outputs\\step2\\pytest-results.xml\n.\\.venv\\Scripts\\python.exe scripts\\summarize_step2.py\n.\\.venv\\Scripts\\python.exe scripts\\verify_step2.py',
      'data/processed/step2/<partition>/<trip>: road_arcs, candidates, observations, unmatched Parquet; network/status JSON. outputs/step2: reports, PNGs, candidate map, tests and verification.',
      'All 90 trips; 1,227,056 train and 113,639 student-test candidate rows; 198 unmatched training observations; 96,077 capped sets; 106 degenerate segments excluded from indexing but retained as graph connectors. 71 tests passed; 720 geodesic checks; 50 real brute-force checks; 180 repeated-run Parquet files byte-identical. Pipeline 120.700 s; sampled trip-process RSS 426.969 MiB.')}
    for number,(title,purpose,functions,method,commands,outputs,results) in descriptions.items():
        text=f'''# Step {number:02}: {title}

## Objective and role
{purpose}

## Methodology and execution flow
{method}

## Libraries and tools
Python/NumPy/Pandas implement typed tables and numeric operations; Matplotlib plots,
Folium maps, pytest verifies behaviours. Step 1 uses NetworkX connectivity and haversine.
Step 2 additionally uses PyProj geodesics/transforms, Shapely STRtree, PyArrow Parquet,
and psutil process memory measurements. Source IDs are zero-based and trip-local.

## Files, functions and contracts
{functions}
Files are under laptop1_backend/src/pathsnap/, scripts/, tests/, config/ and outputs/.
See laptop1_backend/README.md for individual function inputs/outputs and complete file inventory.
The three main READMEs and docs/REQUIREMENTS_REVIEW.md document responsibilities/policies.

## Exact commands
```powershell
.\\.venv\\Scripts\\python.exe {commands}
```

## Outputs
{outputs}

## Actual results and verification
{results}
These are historical executed stage results; current full-suite results are recorded in Step 3.
Stage 1/2 tests and data products were reused rather than regenerated for documentation.

## Issues, assumptions and limitations
Raw files remain unchanged. Noise/gaps are recorded, not blindly removed. Routes are
ordered arc occurrences, not pointwise labels; network totals sum per-trip networks.
The WGS84 datum is assumed, not specified by the source. Endpoint segments approximate
road curves. Candidate coverage does not establish label recall. Student-test references
must not tune candidate settings. Evaluator data was not supplied.

## Completion and next-step connection
Complete. Step 1 supplies validated source tables to Step 2. Step 2 supplies label-free
candidate/status records to Step 3 alignment and reusable inference-time features.
No later model or decoder was implemented by these steps.
{common}
'''
        (steps/f'STEP_{number:02}_README.md').write_text(text,encoding='utf-8')
    methods={'invalid_coordinates':'Flag nonfinite/out-of-range coordinates; retain raw values and exclude unreliable supervision',
      'duplicate_record':'Flag exact duplicate longitude/latitude/time records; retain but exclude supervision',
      'bad_timestamp':'Flag nonfinite or non-increasing seconds; start temporal segment; exclude supervision',
      'large_gap':'Start new temporal segment above five seconds; retain observations and invalidate cross-gap heading',
      'speed_outlier':'Flag >180 km/h; retain observation/features, exclude supervision and split time context',
      'gps_jump':'Flag >500 m displacement; retain, segment and exclude supervision',
      'degenerate_geometry':'Retain arc IDs in road table; use existing Step 2 geometric exclusion; zero-cost connectors remain graph edges',
      'unmatched_candidates':'Retain explicit observation status; no fabricated candidate or positive target',
      'disconnected_reference_pairs':'Keep supplied route unchanged; constrain alignment to connected components and mark infeasible sequence restarts'}
    lines=['# Step 03: cleaning, reference alignment and candidate features','',
      '## Objective and why this step is needed','Prepare honest candidate-level binary supervision while preserving noisy observations and separating labels from inference features. No model is trained.','',
      '## Data Cleaning Methods — Method, Purpose and Results','',
      '| Cleaning method | What it does | Why it is used | Python technique/library | Records affected | Final handling |',
      '|---|---|---|---|---:|---|']
    for method,g in audit.groupby('method'):
        lines.append(f'| {method} | {methods.get(method,method)} | Prevent unreliable supervision/context | Pandas/NumPy quality masks; PyProj geometry | {int(g.affected.sum())} | Retained, raw values unchanged; flags/exclusions are separate |')
    lines += ['',f"All {int(table.observations.sum()):,} observations retained; zero raw values modified. Training supervision eligibility is separate from cleaning retention.", '',
      '## Ordered training-reference alignment algorithm',
      'Reference states retain route position, original arc ID, directed closest-point fraction and connected component. STRtree retrieves reference segments within 75 m independently of Step 2 candidates. Emission cost is distance in metres + 20 times heading disagreement/180 when causal heading exists. Dynamic programming constrains component equality, nondecreasing reference occurrence, and progress with at most 5 m jitter tolerance. Progress advance is bounded by 500 m + 50 m/s times elapsed seconds; a capped displacement/progress penalty discourages gratuitous skips. It is reference-based supervision construction, not inference HMM/Viterbi.',
      'Temporal boundaries use >5 s gaps, invalid time/coordinates and speed/jump flags. Unmatched observations stay explicit. Infeasible transitions split the optimisation and exclude the first restarted assignment. A final safety pass excludes backward reference occurrences across temporal resets and heading contradictions >90 degrees. Repeated arc occurrences remain distinct states.',
      'Forward and backward constrained path costs provide a best-path margin against alternative reference occurrence assignments. Margin <2 cost units is ambiguous and excluded. This is a heuristic cost margin, not calibrated probability. Quality-flagged observations are excluded. No available reference segment within 75 m is unmatched; no segment is forced into supervision. Valid remains qualified route-derived supervision, not independently verified physical truth.', '',
      '## Actual results', '```json',json.dumps(r,indent=2),'```',
      f"Processing: {meta['seconds']:.3f} s plus finalisation {r['finalization_seconds']:.3f} s. All {len(table)} trips completed. See trip_summary.csv for actual per-trip timing and label counts.", '',
      '## Candidate recall and classification policy',
      'Recall denominator contains only valid eligible training alignments. Positives match that aligned arc; other candidates in the same eligible observation are negatives. If the aligned arc is absent, every candidate target for that observation is null and excluded: no all-negative pseudo-example is created. Ambiguous/unmatched/excluded alignments receive null targets. Candidate miss causes compare reference geometry distance with the actually used Step 2 radius; within-bound missing arcs are attributed to top-eight truncation. Other misses include bounded search and fallback policy. Defaults were not changed or references inserted. Recall is conditional on confident, near-reference alignments and can be optimistic for the entire noisy dataset.', '',
      '## Complete inference feature list', '\n'.join('- '+f for f in FEATURES), '',
      'Identifiers remain separate metadata and must not enter model inputs. Heading uses only the preceding observation with displacement >=5 m and valid temporal context. Bearings are represented with sine/cosine; heading difference is circular absolute degrees. Endpoint distances are along the projected straight segment. Degrees count directed/parallel arcs. No future GPS point, reference ID, route membership, aligned position or target enters feature construction. Undefined headings/speeds remain NaN with a heading_undefined flag; future imputation/scaling must fit on training folds only. No fitted preprocessing statistics are learned in Step 3.', '',
      '## Files, function contracts and execution flow',
      '- preprocessing.clean_observations(track) -> raw-preserving observations plus quality flags, causal measurements and temporal segment IDs.',
      '- alignment.align_training(network, cleaned, route) -> per-observation assignment/status/margin/closest point. enforce_consistency excludes cross-reset order and direction contradictions.',
      '- features.engineer_features(candidates, cleaned, network) -> 4 identifier columns and 25 label-independent feature columns.',
      '- features.supervision(candidates, alignment, partition) -> separate nullable Int8 binary labels and conditional recall table; rejects student_test.',
      '- features.trip_splits(ids) -> deterministic five validation folds of complete sorted training trips; 16 trips per fold.',
      '- scripts/run_step3.py reads existing candidates, verifies source hashes, processes one trip at a time and never reads student-test routes.',
      '- scripts/finalize_step3.py audits safety exclusions, enriches recall cause/feature statistics, validates labels and creates final diagnostics.',
      '- scripts/document_step3.py generates these stage documents from actual persisted outputs/tests.', '',
      '## Exact PowerShell commands (backend directory)', '```powershell',
      '.\\.venv\\Scripts\\python.exe scripts\\run_step3.py',
      '.\\.venv\\Scripts\\python.exe scripts\\finalize_step3.py',
      '.\\.venv\\Scripts\\python.exe -m pytest -q --junitxml=outputs\\step3\\pytest-results.xml',
      '.\\.venv\\Scripts\\python.exe scripts\\document_step3.py','```','',
      '## Outputs and tools',
      'No new dependencies. Python, NumPy/Pandas, PyProj, Shapely STRtree, PyArrow Parquet, Matplotlib, Folium and pytest are reused.',
      'data/processed/step3/<partition>/<trip>/observations.parquet and features.parquet exist for both partitions. Training alone additionally has alignment.parquet, labels.parquet and recall.parquet. Shared trip_folds.csv, feature_schema.json and preprocessing.json define grouping, features and parameters. outputs/step3 contains trip_summary.csv, cleaning_audit.csv, recall_causes.csv, feature_statistics.csv, results/run metadata, failures.csv, alignment_and_recall.png, feature_distributions.png, training_alignment.html, test results and STEP3_SUMMARY.md.', '',
      '## Tests and verification',json.dumps(test),
      'The original 71 tests remain intact. New tests cover raw preservation, duplicates, gaps, invalid coordinates/jumps, monotonic/repeated/disconnected reference alignment, ambiguity, unmatched observations, feature causality, labels/recall, train/test isolation and deterministic trip folds/alignment. Finalisation checks no labels exist for student-test trips, label groups have exactly one positive when eligible, feature schemas exclude label fields and eligible route occurrence order is monotonic.', '',
      '## Limitations, completion and Step 4 readiness',
      'All requested trips processed; no source repairs or candidate search changes. Heuristic reference alignment is not infallible and excludes uncertain observations. Across disconnected references/temporal resets, conservative exclusions reduce coverage. The radius/margins/transition bounds are documented engineering choices, not statistically fitted or calibrated confidence. Uniform trip folds are reproducible but not stratified; imbalance must be examined during Step 4. Undefined features require fold-local preprocessing. No ML models, decoder, optimisation, API/UI or deployment ZIPs exist. Step 4 is ready to consume explicit feature columns with separately joined eligible labels and trip groups, subject to approval.',
      common]
    text='\n\n'.join(lines)+'\n';(steps/'STEP_03_README.md').write_text(text,encoding='utf-8');(O/'STEP3_SUMMARY.md').write_text(text,encoding='utf-8')
    for path,prefix in [(ROOT/'README.md','docs/steps/'),(B/'README.md','../docs/steps/'),(ROOT/'laptop2_frontend/README.md','../docs/steps/')]:
        old=path.read_text(encoding='utf-8');marker='\n<!-- STEP3_CURRENT -->'
        if marker in old:old=old.split(marker)[0]
        addition=marker+'\n\n## Current status: Step 3 completed\n\nEarlier Step 1/2 descriptions above are historical. Step 3 now adds non-destructive cleaning, training-only ordered-reference supervision, recall and reusable inference features. Model training and all subsequent application stages remain pending.\n\n'
        addition+='\n'.join(f'- [Step {i} detailed reference]({prefix}STEP_{i:02}_README.md)' for i in [1,2,3])
        addition+=f"\n\nActual Step 3 results: {r['successful_trips']} trips; {int(r['valid_aligned']):,} eligible training alignments; conditional candidate recall {r['candidate_recall']:.6%}; {int(r['positive']):,} positive and {int(r['negative']):,} negative candidate targets. {test['tests']} tests, {test['failures']} failures, {test['errors']} errors, {test['skipped']} skipped. Student-test features remain unlabelled; frontend remains unimplemented.\n\n- [x] Step 3 cleaning/alignment/features/tests/documentation.\n- [ ] Step 4 five models: not started.\n\nChangelog: 2026-11-18 — implemented Step 3 and created separate retrospective Step 1/2 and current Step 3 references. No new dependencies. Use the Step 3 reference for commands, function contracts, complete feature list and limitations.\n"
        path.write_text(old+addition,encoding='utf-8')
    print('Step 1/2/3 references, summary and three main READMEs updated from measured results.')
if __name__=='__main__':main()
