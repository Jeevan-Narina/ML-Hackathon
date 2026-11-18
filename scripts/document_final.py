"""Document owner-reported integration and measured deployment packaging."""
from pathlib import Path
import csv,json,argparse
ROOT=Path(__file__).resolve().parents[1];BACK=ROOT/'laptop1_backend';FRONT=ROOT/'laptop2_frontend'


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def prepare():
    integration='''# Step 11: Two-laptop integration record

## Scope and source of evidence

The project owner reports that the actual two-laptop tests succeeded. This record
captures that supplied evidence; the coding agent did not repeat the physical LAN
tests or independently observe the two devices. No prediction logic was changed.
This is separate from the recorded Step 9/10 local automated/loopback checks.

| Check | Actual reported result | Evidence limit |
|---|---|---|
| Laptop 2 → Laptop 1 backend connectivity | Succeeded, owner-reported | Actual addresses, adapter/firewall configuration and timings were not supplied |
| Live POST /predict | Succeeded, owner-reported | Request size, payload, exact response and latency were not supplied; no new accuracy claimed |
| Saved-trip map rendering on Laptop 2 | Succeeded, owner-reported | Specific trip IDs/all-ten remote coverage were not supplied |
| Before / After / Compare controls | Succeeded, owner-reported | No new map or route benchmark inferred |
| Invalid-input handling and recovery | Succeeded, owner-reported | Exact invalid payloads/status sequences were not supplied |
| Backend timeout handling | Observed, owner-reported | No controlled timeout threshold, duration, cancellation or retry guarantee established |

The supplied statement is the integration acceptance evidence for these functional
checks. It does not establish exhaustive network/load/security tests, all ten map
controls on both devices, sustained performance or an independent road-truth score.
No dated logs/screenshots, software versions on the second physical device, packet
captures or user test timings were supplied, so none are invented here.

## Architecture and intended operation

Laptop 2 runs Streamlit/Requests → HTTP → Laptop 1 FastAPI. Laptop 1 owns the frozen
tuned RF, exact 25-feature preprocessing, candidate retrieval, directed routing and
Viterbi. Saved demos reuse Steps 7–8; fresh predictions require explicit GPS plus
directed nodes/arcs. Metrics are returned only for legitimate saved evaluation
references. Existing student-demo maps label their references as evaluation-only;
hidden evaluator labels must remain server-side.

Use Laptop 1's reachable address in the frontend sidebar, keep the FastAPI process
running and confirm private-network firewall/peer access. Do not use Laptop 2's
localhost for Laptop 1. Browser map assets and Streamlit's Python HTTP client both
need access. Server-side Requests is not browser CORS; the iframe loads directly
from the backend. A timeout does not prove server cancellation, so check readiness
before resubmitting. No automatic POST retry or hard inference cancellation was added.

## Existing independently recorded checks

Step 9 recorded 192 backend tests passing, plus a real local HTTP saved-demo/map
request and a five-point live prediction. Step 10 recorded 15 frontend tests and
GET-only cached-demo HTTP/AppTest/Streamlit startup verification. Those results
remain preserved and are not claimed as a new combined test run. See the unchanged
STEP_09_README.md and STEP_10_README.md for exact commands/contracts and limitations.

No new two-laptop deployment logic, optional feature, model fit, optimisation,
map regeneration or expensive evaluation was performed to write this record.
The next step packages the already working components without changing inference.

Changelog — 2026-11-18: recorded the owner's successful physical two-laptop checks
and timeout observation, explicitly separating supplied evidence from agent-run tests.
'''
    (ROOT/'docs/steps/STEP_11_README.md').write_text(integration,encoding='utf-8')
    cv=rows(BACK/'outputs/step4/model_comparison.csv');student={r['model_name']:r for r in rows(BACK/'outputs/step5/student_test_comparison.csv')}
    table=['| Baseline model | CV candidate accuracy | CV Macro F1 | CV observation Top-1 | CV log loss | Student Top-1 | Student Macro F1 | Student log loss |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in cv:
        s=student[r['model_name']]
        table.append(f"| {r['model_name']} | {float(r['accuracy_mean']):.3%} | {float(r['macro_f1_mean']):.6f} | {float(r['top1_accuracy_mean']):.3%} | {float(r['log_loss_mean']):.6f} | {float(s['top1_accuracy']):.3%} | {float(s['macro_f1']):.6f} | {float(s['log_loss']):.6f} |")
    text='''# PathSnap final project summary

## Delivered architecture

Laptop 1: FastAPI, the frozen tuned Random Forest, learned 25-feature pipeline,
directed candidate generation, bounded shortest paths and log-space HMM/Viterbi.
Laptop 2: standalone Streamlit/Requests using HTTP only. No project models/data or
backend code is needed in the frontend. All ten student demos reuse saved outputs;
new requests supply GPS and explicit directed road context. Inference is CPU-only.

## Actual five-model baseline comparison

Values are saved results, not newly rerun experiments. CV uses five trip-grouped
folds of 64 fitting / 16 validation trips from the 80 training trips. Candidate
classification accuracy differs from per-GPS observation Top-1. Macro F1/log loss
are candidate-level metrics. Student Top-1 is conditional on qualified labels and
the correct candidate being present; it is not full-route physical accuracy.

'''+ '\n'.join(table)+'''

## Bayesian optimisation and final classifier

Seeded persistent Optuna TPE completed five RF trials and six HGB trials. Initial
search used saved training fold 0, followed by selected-configuration five-fold CV.
RF Top-1 improved 80.545592% → 81.050104% (+0.504512 percentage points); Macro F1
0.879261 → 0.883923 and log loss 0.192647 → 0.176211. HGB tuning declined from
80.791805% to 80.052770% (−0.739034 pp), so no improvement is invented for it.
RF is primary; HGB remains a research comparison, not a deployed backup binary.
These overlapping search/CV scores are selection evidence, not an independent test.

RF parameters: 36 trees, max_depth=12, min_samples_split=2, min_samples_leaf=20,
max_features=0.5, max_samples=0.4. Frozen student-test RF candidate accuracy is
94.841294%, binary F1 0.851287, Macro F1 0.910040 and log loss 0.142557.

## Random Forest + Viterbi student-test results

Ten trips / 19,160 raw GPS observations; 15,121 qualified reference alignments,
14,547 qualified candidate-present observations and 574 candidate misses.

| Metric | Tuned RF only | RF + Viterbi |
|---|---:|---:|
| Qualified candidate-present Top-1 | 85.179075% | 87.145116% |
| Correct eligible selections | 12,391 | 12,677 |
| Accuracy including qualified candidate misses | 81.945639% | 83.837048% |
| Direct continuity on assessable adjacent pairs | 86.642939% | 94.509578% |
| Topologically unreachable adjacent selections | 31 | 2 |

The measured Top-1 gain is +1.966041 pp (286 additional correct selections).
The saved reconstruction has 277 segments and 7,387 route-arc occurrences.
223 infeasible adjacent pairs remain across explicit breaks; no infeasible link is
retained inside a segment. There are 44 temporal gaps. Viterbi has no computed
posterior log loss/Macro F1; classifier score quality is not a route posterior.
Nine trips improved and 00000065 declined by 3.306 pp. Candidate misses, heuristic
reference alignment and straight endpoint geometry remain limitations.

## Actual testing and integration status

Historical agent-run results: 192 backend tests and 15 frontend tests passed in
their respective recorded runs. They were not rerun as an imagined combined suite.
Step 9 verified real local HTTP live prediction; Step 10 verified local HTTP demo
loading and Streamlit startup. The owner now reports successful physical two-laptop
connectivity, live POST /predict, saved-map rendering, Before/After/Compare controls,
invalid-input handling and recovery, with timeout handling observed. These physical
checks are user-reported, not independently repeated or timed by the packaging agent.
Exact payloads/trips/status logs, throughput, security/load behaviour and cancellation
guarantees were not supplied. STEP_11_README.md records the evidence boundaries.

## Deployment commands

Install 64-bit CPython 3.14, extract each ZIP, and open PowerShell in its folder.
No virtual environment, Python interpreter or offline wheelhouse is included.

Laptop 1 (Laptop1_Backend):

```powershell
py -3.14 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt
.\\.venv\\Scripts\\python.exe scripts\\serve_api.py --host 0.0.0.0 --port 8000
```

Laptop 2 (Laptop2_Frontend):

```powershell
py -3.14 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt
.\\.venv\\Scripts\\python.exe -m streamlit run app.py
```

In Backend URL enter http://<LAPTOP1_HOST>:8000 and click Connect / refresh.
0.0.0.0 is the backend bind address, not a client URL. Find the reachable address
using ipconfig. Check private-network Python firewall permission and Wi-Fi peer
access; do not disable the firewall. See each archive README for troubleshooting.

## Operational limits and package verification

New requests: at most 500 GPS points, 5,000 nodes, 10,000 arcs and 2 MiB. One live
prediction runs at a time; busy requests return 503. Metrics are null for new inputs
without evaluation references. Timeouts do not guarantee cancellation or safe retry.
There is no production authentication/TLS/load validation. OSM tiles and Swagger
CDN resources need internet; map libraries/route overlays are bundled locally.

Training/Optuna/data extraction/map regeneration is not part of deployment.
Archive checks and the new small extracted-package smoke run are reported separately
in releases/packaging_verification.json and docs/steps/STEP_12_README.md. Smoke tests
use existing installed dependency environments; a clean machine/offline install is
not claimed. The complete development workspace and previous stage evidence are
preserved; deployment ZIPs intentionally omit research-only data/scripts/models.
'''
    (ROOT/'FINAL_PROJECT_SUMMARY.md').write_text(text,encoding='utf-8')
    print('Step 11 evidence record and final results summary prepared.')


def finish():
    report=json.loads((ROOT/'releases/packaging_verification.json').read_text());assert report['status']=='passed'
    archives=report['archives'];smoke=report['smoke']
    table='| Archive | Bytes | MiB | Files | SHA-256 |\n|---|---:|---:|---:|---|\n'+'\n'.join(f"| {r['name']} | {r['bytes']} | {r['bytes']/2**20:.3f} | {r['files']} | {r['sha256']} |" for r in archives)
    text='''# Step 12: Final portable deployment packages

## Objective and architecture

Separate Laptop 1 and Laptop 2 deployment ZIPs preserve the existing prediction
logic, selected model, ten saved demonstrations and HTTP-only client separation.
The complete research workspace remains unchanged apart from the explicitly
requested current READMEs and new final documentation/release files.
No model fitting, Optuna, old-trip decoding, map generation or expensive evaluation
was repeated. A two-point live prediction was run only as an extracted-package
inference smoke check, without training or accuracy claims.

## Actual archives and integrity

'''+table+'''

Archives are under releases/. Each has a named top-level directory, README.md,
setup.ps1, run.ps1, pinned runtime requirements/lock, PROJECT_SUMMARY.md and a
MANIFEST.json of per-file hashes/sizes. CRC and extracted file SHA-256/size checks
passed. releases/checksums.sha256 records whole-ZIP hashes. No digital signatures
are provided; compare hashes against a trusted copy of these results.

ZIP timestamps/order are deterministic; unchanged inputs produce reproducible
archives. The build uses an explicit allowlist rather than archiving a working
directory. Source files are copied byte-for-byte; generated setup documentation
and deployment-only dependency lists do not modify original working files.

## Laptop 1 contents and required data

The API's relative-import closure is included unchanged, with scripts/serve_api.py,
the frozen random_forest_tuned.joblib, original artifact manifest and 25-feature
schema, and the frozen Step 7 decoder configuration. modeling.py is required to
unpickle FiniteFeatures and provide existing scoring; alignment.py remains an
import dependency of the existing evaluation module. These modules are not
rewritten or executed as training/alignment jobs. Training entry scripts/data and
other model binaries are excluded. The original artifact manifest still contains
an HGB row; that binary is intentionally absent and no HGB runtime is claimed.

All ten demonstrations retain observations.parquet, the six output files named by
each Step 7 completion stamp plus that stamp, saved per-trip metrics, Step 8 HTML,
full GeoJSON sidecars and their metadata hashes, index/summary, example link pages
and local JS/CSS/image/licence assets. The apparently unused probabilities and
transition_diagnostics files are necessary because ApiService.demo verifies every
recorded completion checksum. Raw networks, training features/labels, Step 1–6
intermediates, OOF predictions, CV fit artifacts, Optuna databases, preview images,
temporary output/logs, caches and virtual environments are not shipped.

Fresh inference does not require a global road dataset: POST /predict requires
its explicit GPS/nodes/arcs context. Saved demos need no raw road graph because
their verified route geometry is already saved. Existing student reference overlays
remain clearly labelled evaluation-only; hidden evaluator labels are not packaged.

## Laptop 2 contents

Only app.py, api_client.py, example_request.json, frontend requirements/lock,
.env.example and .streamlit/config.toml are included with setup/run/readme/summary
and manifest. No backend import, project dataset, model, saved map or test output
is copied to this ZIP. Runtime dependencies are Streamlit/Requests plus their own
transitives. The example request is synthetic contract data, not a trip dataset.

## Windows setup and exact launch commands

Install 64-bit CPython 3.14 with the Python launcher. Extract each ZIP before use.
In PowerShell inside each corresponding extracted directory:

```powershell
py -3.14 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt
```

Laptop 1:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\serve_api.py --host 0.0.0.0 --port 8000
```

Laptop 2:

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app.py
```

Optional setup.ps1/run.ps1 helpers execute the same commands. If a Windows script
policy blocks a helper, use the direct commands without changing that policy.
Fresh installs need internet/package-index access or cached compatible wheels;
no interpreter, virtual environment or offline wheelhouse is included.

Keep FastAPI running. In the frontend sidebar enter http://<LAPTOP1_HOST>:8000 and
Connect / refresh. Use ipconfig to find Laptop 1's reachable address; 0.0.0.0 is
only a listening interface and localhost on Laptop 2 refers to Laptop 2. Default
port is 8000; serve_api.py supports --host/--port or PATHSNAP_HOST/PATHSNAP_PORT.
PATHSNAP_BACKEND_URL can initialize the frontend. PATHSNAP_CORS_ORIGINS accepts
explicit browser origins if needed; server-side Requests is not browser fetch CORS.

Check /health on Laptop 1 first, then from Laptop 2. Windows Firewall must allow
the backend Python/private-network port, and Wi-Fi isolation/VPN/proxy routing must
permit peer traffic. Do not disable the firewall. Both the browser (map iframe)
and Streamlit's Python process must reach the configured address. OSM/Swagger CDN
access may be needed; a blank tile background does not imply absent route overlays.
Timeouts do not prove cancellation; check status before retrying a POST.

## New package verification: actually executed

ZIP paths were checked against traversal/absolute paths and disallowed cache,
environment, training and Optuna entries. Each member passed CRC, then every
extracted byte hash/size matched MANIFEST.json. The tuned RF checksum/schema and
all ten saved demos also passed the existing API startup/checksum logic.

Isolated Python subprocesses imported the API/model classes from the extracted
source directory rather than the workspace package; all pinned backend runtime
dependency versions were checked. The frontend imported Streamlit/Requests and
its own extracted api_client. These tests reused the already installed backend
and frontend dependency environments; a fresh virtual-environment installation
on another Windows device was not executed or claimed.

The extracted API served healthy model/decoder state (one model load), all ten
demo JSON responses covering 19,160 GPS observations, all ten HTML maps, local
map assets, map index and Swagger. A new two-observation, one-arc synthetic-context
POST /predict succeeded with no accuracy metrics. The extracted frontend client
used HTTP to the extracted API and loaded a saved demo; extracted Streamlit web
startup/health passed. This is a minimal local package smoke test, not a repeat of
the physical two-laptop tests or all browser controls. All temporary processes stopped.

```json
'''+json.dumps(smoke,indent=2)+'''
```

Protected original files unchanged: '''+str(report['protected_files_unchanged'])+'''. No earlier per-step README,
model/data/route/map or test-result artifacts were overwritten. Previous 192 backend
and 15 frontend test runs remain recorded; no new full-suite run is implied. The
owner-reported physical checks are separately documented in STEP_11_README.md.

## Reproduction and generated files

From the existing full development workspace (using installed environments):

```powershell
.\\PathSnap\\laptop1_backend\\.venv\\Scripts\\python.exe .\\PathSnap\\scripts\\document_final.py --prepare
.\\PathSnap\\laptop1_backend\\.venv\\Scripts\\python.exe .\\PathSnap\\scripts\\package_step12.py
.\\PathSnap\\laptop1_backend\\.venv\\Scripts\\python.exe .\\PathSnap\\scripts\\document_final.py --finish
```

package_step12.py collects allowlisted files and installed runtime dependency
closure, writes sorted deterministic ZIP members, safely extracts into a new
releases/verification/<run>/ folder, validates manifests and starts/stops local
smoke services. Each run preserves prior work and does not recursively delete an
existing checkout. The verification extraction is not included in either ZIP.
document_final.py writes Step 11's attributed evidence and actual-results summary,
then records measured archive verification and updates the three main READMEs.

Outputs: releases/Laptop1_Backend.zip, Laptop2_Frontend.zip, checksums.sha256,
packaging_verification.json, protected_artifacts.json and non-shipped verification
extraction/logs. FINAL_PROJECT_SUMMARY.md contains the five-model comparison,
Optuna outcomes, final RF/Viterbi student results, actual test status and commands.

## Limitations and stop condition

Deployment needs Python 3.14 and compatible pinned packages; clean-machine/offline
setup is unverified. No hardware-specific latency/load guarantee, production auth,
TLS setup, hard cancellation deadline or automatic safe prediction retry was added.
Existing API input/output limits and reference/candidate/straight-road approximations
remain. Only legitimate supplied student demo references are visible in those maps;
reserved evaluator data is not included or processed. Package smoke checks do not
establish new road accuracy or exhaustive real-network/browser behaviour.

Steps 11–12 are complete within those evidence limits. No prediction logic or
optional feature was changed. Stop after Step 12.

Changelog — 2026-11-18: created two allowlisted portable runtime ZIPs, manifests,
integrity/import/extracted smoke evidence, deployment instructions and final summary.
'''
    (ROOT/'docs/steps/STEP_12_README.md').write_text(text,encoding='utf-8')
    (ROOT/'releases/STEP12_SUMMARY.md').write_text(text,encoding='utf-8')
    for path,prefix in [(ROOT/'README.md','docs/steps/'),(BACK/'README.md','../docs/steps/'),(FRONT/'README.md','../docs/steps/')]:
        old=path.read_text(encoding='utf-8');marker='\n<!-- FINAL_STEPS_11_12 -->'
        if marker in old:old=old.split(marker)[0]
        old=old.replace('## Current status: Step 10 completed','## Historical Step 10 completion record')
        old=old.replace('**Steps 1–10 are implemented:**','**Steps 1–12 are complete:**')
        old=old.replace('real two-laptop integration remains pending.','physical two-laptop checks are owner-reported successful; deployment ZIPs are complete.')
        old=old.replace('Steps 1–10 are complete; Step 11 integration requires explicit approval.','Steps 1–12 are complete; see final deployment references.')
        old=old.replace('A live prediction was not submitted in Step 10. Real two-laptop networking, browser iframe checks on Laptop 2 and final packaging remain Step 11 work. No ZIP was created.',
            'Step 10 did not submit a live prediction. The owner now reports successful physical two-laptop checks, and separate deployment ZIPs are complete; see the final Steps 11–12 record below.')
        old=old.replace('- [ ] Step 11 real two-laptop integration, end-to-end UI prediction verification and packaging; not started.',
            '- [x] Steps 11–12 integration evidence and final packaging; see the final record below.')
        addition=f'\n\n## Current status: Steps 11–12 completed\n\n[Step 11 integration record]({prefix}STEP_11_README.md) documents the owner-reported successful two-laptop connectivity, live POST, saved-map/Before/After/Compare, invalid-input recovery and observed timeout handling. These physical checks were not independently repeated; exact logs/timings were not supplied.\n\n[Step 12 deployment reference]({prefix}STEP_12_README.md) records the separate allowlisted Laptop1_Backend.zip and Laptop2_Frontend.zip under PathSnap/releases/, setup/run instructions, manifests/checksums and actual extracted-package smoke results. Both archives passed CRC/hash/import checks; the extracted API served all ten demo JSONs/maps, a tiny live prediction and local assets, and the extracted frontend imported, reached the API over HTTP and started Streamlit. Existing installed dependencies were used; a clean-machine install was not tested.\n\nHistorical test runs remain 192 backend and 15 frontend tests passed, not a newly rerun combined suite. All {report["protected_files_unchanged"]} protected files are unchanged. See FINAL_PROJECT_SUMMARY.md at the project root for five-model CV/student comparisons, actual Optuna gains/degradation, RF + Viterbi results and exact commands. No training, Optuna, old-trip reconstruction or map generation repeated. No working prediction logic changed.\n\n'
        addition+='\n'.join(f'- {r["name"]}: {r["bytes"]/2**20:.3f} MiB; {r["files"]} files; SHA-256 in releases/checksums.sha256.' for r in archives)
        addition+='\n\n- [x] Step 11 attributed physical integration record.\n- [x] Step 12 separate deployment archives, validation, minimal extracted smoke checks, final summary and documentation.\n\nUse each extracted archive README: Python 3.14 → create .venv → install requirements-lock.txt → backend scripts/serve_api.py (port 8000) / frontend streamlit run app.py. Enter Laptop 1’s reachable address in Backend URL. No virtual environments/caches/training-only data ship in the archives.\n\nChangelog — 2026-11-18: completed final integration documentation and deployment packaging. Stop after Step 12.\n'
        path.write_text(old+marker+addition,encoding='utf-8')
    print('Step 12 measured package report and all three main READMEs updated.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--finish',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    if args.finish:finish()
