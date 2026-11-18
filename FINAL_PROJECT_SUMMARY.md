# PathSnap final project summary

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

| Baseline model | CV candidate accuracy | CV Macro F1 | CV observation Top-1 | CV log loss | Student Top-1 | Student Macro F1 | Student log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| hist_gradient_boosting | 93.250% | 0.881931 | 80.792% | 0.178984 | 84.684% | 0.908340 | 0.145056 |
| random_forest | 93.177% | 0.879261 | 80.546% | 0.192647 | 84.856% | 0.908383 | 0.162496 |
| decision_tree | 92.557% | 0.872538 | 79.919% | 0.393645 | 84.244% | 0.904033 | 0.236594 |
| logistic_regression | 91.029% | 0.846455 | 79.713% | 0.234818 | 84.574% | 0.874769 | 0.197699 |
| gaussian_nb | 90.827% | 0.846121 | 79.508% | 0.485153 | 84.127% | 0.858687 | 0.487614 |

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
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts\serve_api.py --host 0.0.0.0 --port 8000
```

Laptop 2 (Laptop2_Frontend):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
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
