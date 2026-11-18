# PathSnap — GPS Map Matching and Route Reconstruction

**Complete project README | Steps 01–12**

PathSnap reconstructs plausible road routes from noisy GPS trajectories using spatial candidate generation, machine-learning road selection, and directed-network sequence decoding. The project also includes a FastAPI backend, a Streamlit interface and independently deployable packages for two laptops.

> **Status:** Implementation Steps 01–12 completed. The original 10-trip *student-testing* dataset was evaluated. An additional, separately supplied `evaluator_test.zip` has **not** yet been evaluated; results below must not be described as final evaluator-test performance.

---

## 1. Project objective

GPS coordinates do not always lie exactly on a road because of device noise, sampling gaps, speed anomalies and ambiguity between nearby roads. Selecting the nearest road to each point independently can produce disconnected or implausible paths.

PathSnap addresses this with four linked capabilities:

1. **Candidate discovery:** find plausible directed road arcs near each GPS observation.
2. **Machine-learning scoring:** rank candidate arcs using spatial, direction, movement, topology and quality features.
3. **Sequence-aware reconstruction:** use an HMM-style log-score model and the Viterbi algorithm to select coherent sequences subject to directed-graph feasibility and movement constraints.
4. **Interactive presentation:** serve routes and comparisons through a FastAPI backend and a Streamlit frontend running on the same or separate laptops.

### End-to-end workflow

```text
Original student data (.track, .nodes, .arcs, .route)
                         |
                         v
Step 01  Dataset inspection and EDA
                         |
                         v
Step 02  Road spatial index and candidate arc generation
                         |
                         v
Step 03  Quality flags, reference alignment and 25 features
                         |
                         v
Step 04  Grouped-CV training of five candidate classifiers
                         |
                         v
Step 05  Student-testing evaluation and feature analysis
                         |
                         v
Step 06  Optuna tuning of Random Forest and HistGradientBoosting
                         |
                         v
Step 07  Tuned RF probabilities + HMM/Viterbi directed route decoding
                         |
                         v
Step 08  Interactive Folium before/after/compare maps
                         |
                         v
Step 09  FastAPI serving, saved trips and live prediction
                         |
                         v
Step 10  Independent Streamlit frontend
                         |
                         v
Step 11  Real two-laptop connectivity and interaction test
                         |
                         v
Step 12  Portable backend/frontend ZIP packaging and validation
```

**Reference separation:** `.route` data supplies references for supervised training/evaluation where valid, but the live decoder does not read reference route IDs as prediction inputs. References are consulted only when evaluating completed predictions.

---

## 2. Dataset and file types

The original student-provided data consists of **80 training trips** and **10 student-testing trips**.

| Input type | Description | Usage |
|---|---|---|
| `.track` | Ordered GPS observations with coordinates and timestamps | Observed movement |
| `.nodes` | Road-network node identifiers and positions | Graph vertices |
| `.arcs` | Directed road arcs linking node IDs | Candidate roads and connectivity |
| `.route` | Ordered reference arc IDs | Reference alignment and offline evaluation, not inference |

| Original split | Trips | GPS observations | Role |
|---|---:|---:|---|
| Training | 80 | 208,417 | Feature engineering, model training and trip-grouped CV |
| Student-testing | 10 | 19,160 | Saved evaluation, route decoding and demonstrations |
| **Total** | **90** | **227,577** | Original student dataset |

A **separate evaluator ZIP** was supplied later. It contains 10 additional trips, each with `.track`, `.nodes`, `.arcs`, and `.route` files; evaluation of those trips remains outstanding. Do not merge them into the training set or use them for hyperparameter tuning.

**A GPS observation is not equivalent to one reference route arc.** Reference trajectories must be aligned to candidate observations using an explicit procedure, and ambiguous or unalignable observations must be tracked rather than assumed correct.

---

## 3. Implementation stages

### Step 01 — Exploratory data analysis

- Loaded the supplied road and GPS file formats and inspected trip counts, observation counts, coordinate/timestamp structure and road-network connectivity inputs.
- Preserved the original ZIP/data rather than modifying the source files.
- Distinguished observed GPS points from route arc sequences and confirmed the training/student-testing split.

**Output:** reproducible dataset summaries, inspection artifacts and Step 01 documentation.

### Step 02 — Spatial candidate generation

- Built a **Shapely STRtree** spatial index over road arcs.
- Searched within **50 m** of each GPS point, with a **150 m fallback** where necessary.
- Retained at most **8 candidate arcs** per observation.
- Produced approximately **1,227,056 training** candidate rows and **113,639 student-test** candidate rows.

**Output:** per-observation candidate arc records, distance/rank information and spatial-index results. Candidate misses are retained as a meaningful evaluation limitation.

### Step 03 — Data quality, alignment and features

- Flagged suspicious observations instead of silently dropping original GPS data: timestamp gaps, extreme speeds, large jumps, duplicates, invalid coordinates and missing/undefined heading.
- Recorded **1,260 gaps over 5 s**, **1,541 speed outliers over 180 km/h**, **4 GPS jumps over 500 m** and **198 unmatched** observations in the project's data-quality assessment.
- Used ordered reference alignment to derive supervised candidate labels only for eligible observations. Recorded **741,516 eligible training candidate rows**, comprising **135,943 positive** and **605,573 negative** records; excluded ambiguous/missing-reference rows from supervised labels.
- Produced **25 model features** in six groups:

| Feature group | Count | Examples |
|---|---:|---|
| Spatial | 6 | Point-to-arc distance, arc length, candidate rank, fractional arc position, endpoint distances |
| Directional | 5 | Road and GPS bearing sine/cosine, heading difference |
| Temporal/movement | 3 | Speed, elapsed time, displacement |
| Topology | 2 | From-node out-degree and to-node in-degree |
| Quality flags | 7 | Invalid coordinates, duplicate, bad timestamp, large gap, speed outlier, GPS jump, undefined heading |
| Candidate context | 2 | Fallback used, candidate count |

**Output:** cleaned-but-traceable observation metadata, aligned eligible labels, fixed feature schema and candidate feature tables. The full ordered feature schema is saved in `laptop1_backend/models/step4/feature_schema.json` in the development project.

### Step 04 — Five-model training and comparison

Trained and compared five binary candidate classifiers:

1. Logistic Regression — linear baseline.
2. Gaussian Naive Bayes — probabilistic baseline.
3. Decision Tree — interpretable non-linear baseline.
4. Random Forest — bagged tree ensemble.
5. Histogram Gradient Boosting — boosted tree ensemble.

Used **5-fold trip-grouped cross-validation** so observations from a particular training trip are not split between a fold's training and validation subsets. Preprocessing, such as imputation/scaling where applicable, was fitted within the relevant training fold rather than on validation observations.

**Output:** trained baseline models, grouped-CV comparisons, rankings, F1/accuracy metrics, Top-1 road selection and supporting charts.

### Step 05 — Student-testing evaluation

- Ran the frozen baseline models on the **10 student-testing trips**.
- Reported both per-candidate **binary classification accuracy** and **Top-1 road-choice accuracy**, which measure different things.
- Examined candidate availability, selected-road continuity, feature effects, prediction timing and per-trip variation.
- Found that the baseline Random Forest had the best student-test Top-1 among the five baselines.

**Output:** `outputs/step5/student_test_comparison.csv`, per-trip results, candidate-miss reports and feature analysis.

### Step 06 — Optuna hyperparameter optimisation

- Used **Optuna's seeded TPE search** on the **training-trip cross-validation** objective rather than tuning against student-testing reference labels.
- Tuned only the strongest two candidates, **Random Forest** and **Histogram Gradient Boosting**, to conserve computation.
- Completed 5 RF trials and 6 HGB trials.
- Selected the tuned Random Forest for sequence decoding.

**Output:** `models/step6/random_forest_tuned.joblib`, the tuned boosting artifact, Optuna study summaries and both CV and student-test before/after tables.

### Step 07 — HMM-style scoring, Viterbi and route reconstruction

- Converted tuned Random Forest candidate probabilities into observation-specific log emission scores.
- Constructed a **directed road graph**, respecting arc direction and original road IDs.
- Scored candidate transitions using road-network travel distance, observed displacement and bounded movement feasibility.
- Applied **log-space Viterbi dynamic programming** to select a coherent candidate arc at each usable GPS point.
- Inserted real shortest-path **intermediate arcs** where selected candidate roads needed a legal network connection.
- Preserved explicit segment breaks for time gaps, unreachable or otherwise infeasible transitions; never fabricated road connections across a break.
- Recorded selected candidates, transition routes and complete arc sequences separately.

**Output:** `outputs/step7/classifier_vs_viterbi.csv`, `per_trip_comparison.csv`, detailed per-trip inference files and `STEP7_SUMMARY.md`.

### Step 08 — Interactive map visualisation

- Generated **10 Folium HTML maps**, one for each original student-testing trip.
- Supported **Before**, **After**, and **Compare** visualisations of GPS observations, classifier selections, reconstructed routes, intermediate arcs and gaps.
- Produced `outputs/step8/index.html` as a starting point for saved demos.

**Output:** saved interactive route maps and map index.

### Step 09 — FastAPI inference backend

- Loaded the trained pipeline and route decoder for repeated requests without retraining.
- Exposed health, saved-trip, saved-map and **live inference** endpoints.
- Implemented a reference-free request path using supplied GPS observations and road-network topology.
- Validated request limits and errors and exercised a small end-to-end live example.

Key endpoints used in the project include:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Backend health |
| `GET` | `/docs` | OpenAPI/Swagger interface |
| `GET` | `/trips` | Available saved demonstrations |
| `GET` | `/trips/{trip_id}` | Saved prediction information |
| `GET` | `/trips/{trip_id}/map` | Saved interactive map |
| `POST` | `/predict` | New reference-free prediction from supplied GPS/road network |

For an exact live-prediction JSON schema, use the supplied `example_request.json` file and `/docs`; do not invent arc/node field names.

### Step 10 — Independent Streamlit frontend

- Built a separate **Streamlit** application using HTTP requests to the backend, without importing the backend model directly.
- Added backend connection/health checks, saved-trip browsing, map presentation, live JSON prediction and understandable error feedback.
- Included `example_request.json` for a reproducible demo.

**Output:** `laptop2_frontend/app.py`, `api_client.py`, frontend requirements, tests and usage instructions.

### Step 11 — Real two-laptop integration

- Connected the Streamlit interface on Laptop 2 to the FastAPI service on Laptop 1 using a local Wi-Fi/hotspot IP.
- Verified backend health, saved trip/map display and an actual live `POST /predict` call from the frontend.
- Checked invalid JSON/error handling and the frontend's response when the backend was stopped or unavailable.
- Documented the observed integration rather than claiming unperformed tests.

**Output:** `docs/steps/STEP_11_README.md` in the development project.

### Step 12 — Deployment packaging and final checks

Created two self-contained **source-and-artifact** deployment ZIPs, excluding virtual environments and unnecessary training/research files:

| Deployment package | Reported size | Contents |
|---|---:|---|
| `releases/Laptop1_Backend.zip` | 12.10 MiB | FastAPI inference source, tuned RF, network/inference essentials, saved 10 demo responses and map assets |
| `releases/Laptop2_Frontend.zip` | 12.7 KiB | Streamlit UI, API client, demo JSON and dependency/setup files |

Reported checks included ZIP CRC verification, file hashes, imports/dependencies in the installed environment, saved demo/map responses, one tiny live prediction, frontend HTTP access and Streamlit startup. **887 protected files remained unchanged.** Extracted-package smoke tests passed; **installing dependencies from scratch on a fresh machine was not tested**.

**Output:** two deployable ZIP files, package verification details, `docs/steps/STEP_12_README.md` and `FINAL_PROJECT_SUMMARY.md`.

---

## 4. Results and how to interpret them

### 4.1 Baseline models on the original student-testing dataset

These were evaluated on the original **10 student-testing trips**; binary classification accuracy is calculated over candidate rows, whereas **Top-1** concerns choosing the correct arc among candidates for an eligible GPS observation.

| Baseline model | Candidate classification accuracy | Macro F1 | Qualified Top-1 accuracy |
|---|---:|---:|---:|
| Logistic Regression | 92.75% | 0.8748 | 84.57% |
| Gaussian Naive Bayes | 91.28% | 0.8587 | 84.13% |
| Decision Tree | 94.48% | 0.9040 | 84.24% |
| **Random Forest** | **94.79%** | **0.9084** | **84.86%** |
| Histogram Gradient Boosting | 94.77% | 0.9083 | 84.68% |

### 4.2 Optuna tuning

| Random Forest metric | Baseline | Tuned |
|---|---:|---:|
| **Training-trip grouped-CV Top-1** | 80.55% | **81.05%** |
| Student-test candidate classification accuracy | 94.79% | **94.84%** |
| Student-test qualified Top-1 | 84.86% | **85.18%** |
| Student-test Macro F1 | 0.9084 | **0.9100** |

The **+0.505 percentage-point** Top-1 tuning gain refers to grouped CV, not the student-test comparison. After tuning, the saved tuned RF was used as the Viterbi emission scorer.

### 4.3 Final student-test inference: tuned RF vs tuned RF + Viterbi

| Metric | Tuned RF only | Tuned RF + Viterbi |
|---|---:|---:|
| Qualified Top-1 map matching | 85.179% | **87.145%** |
| Top-1 including candidate misses | 81.946% | **83.837%** |
| Correct matches / all raw observations | 64.671% | **66.164%** |
| Direct road continuity | 86.643% | **94.510%** |
| Topologically unreachable adjacent pairs | 31 | **2** |
| Feasible adjacent transitions | 17,340 | **18,883** |
| Infeasible adjacent transitions | 1,766 | **223** |

**Evaluation coverage:** 10 student trips, **19,160 raw GPS observations**, **15,121 qualified reference alignments**, **14,547 qualified alignments with a candidate available**, and **574 qualified candidate misses**. Correct qualified candidate choices increased from **12,391** to **12,677**: **286 more correct selections** with Viterbi. The decoder produced **277 separate route segments**.

**What “overall accuracy” means:**

- **87.145%** is the final *qualified candidate-present Top-1* student-test accuracy. It is appropriate to present only with that denominator and reference qualification.
- **83.837%** includes all qualified reference alignments, counting candidate misses as errors.
- **66.164%** divides correctly chosen qualified reference matches by **all** 19,160 raw observations; this is a conservative **coverage-sensitive ratio**, not a claim that every excluded or ambiguous point was physically mismatched.
- **94.510%** measures **direct road continuity**, **not** prediction accuracy. Do not present it as 94.51% map-matching accuracy.
- These scores use reference alignment derived from the student data; they do **not** establish independently measured, full-route physical correctness or unseen evaluator-test accuracy.

Viterbi improved qualified Top-1 by **1.966 percentage points** and direct continuity by approximately **7.867 percentage points**. It improved Top-1 on 9 of the 10 student-test trips and decreased it on one. The remaining 223 infeasible adjacent pairs correspond to explicit cross-segment breaks rather than incorrectly connected within-segment routes.

### 4.4 Trip-level result files

From the **development** backend folder, display the saved results in PowerShell:

```powershell
Import-Csv .\outputs\step5\student_test_comparison.csv | Format-Table -AutoSize
Import-Csv .\outputs\step6\baseline_vs_tuned_student.csv |
    Select-Object model_name,accuracy_baseline,accuracy_tuned,top1_accuracy_baseline,top1_accuracy_tuned,macro_f1_baseline,macro_f1_tuned |
    Format-List
Import-Csv .\outputs\step7\classifier_vs_viterbi.csv | Format-List
Import-Csv .\outputs\step7\per_trip_comparison.csv | Format-Table -AutoSize
Get-Content .\outputs\step7\STEP7_SUMMARY.md
```

These are **read-only result-inspection commands**. They do not retrain or rerun the decoder.

---

## 5. Repository layout

The development project is structured approximately as follows. Exact intermediate artifact contents and paths are detailed in the original Step 01–12 documentation.

```text
PathSnap/
├── README.md                        # Project-level overview
├── FINAL_PROJECT_SUMMARY.md         # Final outcomes and experimental summary
├── AGENTS.md                        # Development/workflow instructions
├── docs/
│   └── steps/
│       ├── STEP_01_README.md
│       ├── ...
│       └── STEP_12_README.md
├── laptop1_backend/
│   ├── README.md
│   ├── src/pathsnap/                # Features, graph routing, HMM/Viterbi, API logic
│   ├── scripts/                     # Execution, verification and API startup
│   ├── models/                      # Saved baseline and tuned models
│   ├── data/                        # Development datasets/intermediates
│   ├── outputs/                     # Step results, evaluation and saved maps
│   ├── tests/
│   └── requirements-lock.txt
├── laptop2_frontend/
│   ├── README.md
│   ├── app.py                        # Streamlit UI
│   ├── api_client.py                 # HTTP calls to backend
│   ├── example_request.json         # Working live-prediction request example
│   ├── assets/
│   ├── scripts/
│   ├── tests/
│   └── requirements-lock.txt
├── releases/
│   ├── Laptop1_Backend.zip
│   └── Laptop2_Frontend.zip
└── scripts/                         # Project-level documentation/packaging utilities
```

**Development source vs deployment packages:** The development repository contains experiment artifacts and historical summaries. The compact ZIPs intentionally omit many training-only files; run evaluation-report commands from the development backend, not from a slim deployment extraction.

---

## 6. Running the application on two Windows laptops

### Prerequisites

- Windows PowerShell and **64-bit Python 3.14** (the version used in the documented extraction instructions).
- Both machines connected to the **same reachable local network** (same Wi-Fi/hotspot or an appropriately routed network).
- A working package installer/network connection to install Python requirements.
- The extracted deployment ZIPs or the corresponding development folders.

> Do not copy a `.venv` from one laptop to the other. Build a local virtual environment on each laptop. Fresh-machine installation was not verified during Step 12; if a dependency fails to install, check its Python-version compatibility and the included package instructions before changing the model or code.

### Laptop 1 — FastAPI backend

Extract `Laptop1_Backend.zip`, open PowerShell **inside its extracted backend folder** and run:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts\serve_api.py --host 0.0.0.0 --port 8000
```

Check on Laptop 1:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

Find Laptop 1's local IPv4 address with:

```powershell
ipconfig
```

Identify the **active Wi-Fi/hotspot adapter's IPv4 Address**. For example, if it is `192.168.1.10`, Laptop 2 will connect to `http://192.168.1.10:8000`. This address is an example, not a fixed PathSnap address.

### Laptop 2 — Streamlit frontend

Extract `Laptop2_Frontend.zip`, open PowerShell **inside its extracted frontend folder** and run:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the Streamlit address shown in the terminal, normally:

```text
http://localhost:8501
```

In the frontend's **Backend URL** field enter:

```text
http://<LAPTOP1_IPV4_ADDRESS>:8000
```

Then connect/refresh to confirm backend availability. A Windows firewall rule may need to permit inbound TCP port `8000` on the trusted private network. Both processes must remain running during the live demo.

### One-laptop alternative

To demo locally on one computer, run the backend and frontend in **separate terminals**, and set the frontend Backend URL to `http://127.0.0.1:8000`.

---

## 7. Live demonstration procedure

1. Start the Laptop 1 FastAPI service and confirm `/health` returns successfully.
2. Start Laptop 2's Streamlit UI and set the backend IP address.
3. Open the saved trips list and select a stored example, such as trip `00000009`.
4. Show **Before**, **After**, and **Compare** map views. Explain original GPS points, independent Random Forest road choices, Viterbi-adjusted road choices, intermediate arcs and honest segment breaks.
5. Open **New prediction**, copy the provided `example_request.json` contents into the request input and submit it.
6. Show that the backend returns a prediction and that the frontend renders the result.
7. Optionally demonstrate invalid-input feedback using `{}`. Restore the valid JSON afterwards.

**Meaning of `matched_observations`:** The saved 10-trip decoding report lists all 19,160 student GPS observations as matched, but that does **not** mean all 19,160 road choices were correct. Compare against the qualified reference-based Top-1 metrics above.

---

## 8. Tests, validation and reproducibility

- Five-model comparison used **trip-grouped cross-validation**, preserving trip-level separation.
- Optuna tuning used training-trip folds only; student-test references were used for evaluation, not the tuning objective.
- Viterbi was deterministic and replay-checked on a real student trip.
- Graph traversal tests covered road direction, shortest paths, endpoint connections, loops, gaps, unreachable arcs and bounded movement.
- Historically reported check totals included **192 passing backend tests** and **15 passing frontend tests** at the corresponding implementation stages. These figures are stage-specific rather than a new post-packaging universal test-suite run.
- The two-laptop app was physically exercised: saved maps, live inference, invalid requests and backend-unreachable feedback.
- Packaging checks reported integrity, import/startup and extracted smoke-test success. A fresh clean-environment dependency install was not tested.

**Reproducibility boundary:** Keep the original student datasets, model files, feature schema, decoder configuration and saved result reports. Do not retrain on the evaluator-test data if reporting it as an unseen holdout.

---

## 9. Known limitations

- Candidate generation may miss the correct road (574 candidate misses among qualified student-test alignments).
- Reference labels are produced through a documented alignment procedure and exclude ambiguous observations; this is not an independently surveyed road truth for every raw GPS point.
- Strongly connected-looking or directly connected road sequences can still be on the wrong physical road.
- No turn restrictions or road-specific legal speed limits are inferred when not represented in the road data.
- Time gaps, noisy jumps and impossible movement trigger segment breaks rather than manufactured continuity.
- RF candidate probabilities are used as emission scores; the HMM-style score is not claimed to be a fully calibrated generative posterior.
- Student-testing data were used during development evaluation; separate evaluator-test results must be presented on their own, with no leakage into training/tuning.
- Fresh Windows deployment installation needs a real clean-machine verification if specifically required for submission.

---

## 10. Remaining external evaluation and reporting

The original implementation and deployment stages are complete. The principal outstanding evaluation is to **run the frozen model/decoder on the separately supplied `evaluator_test.zip`**, without training or tuning on it. This requires the same preprocessing and candidate-generation path as the original trip data, followed by evaluation against its `.route` references **after** inference.

A final report/presentation should include:

1. System architecture and Step 01–12 pipeline.
2. Dataset splits and `.track`/`.nodes`/`.arcs`/`.route` definitions.
3. Feature engineering and data-quality/label-alignment methodology.
4. Five-model comparison, grouped-CV design and Optuna experiments.
5. Tuned RF vs Viterbi Top-1 and route-continuity results, with proper denominators.
6. Screenshots of saved Folium before/after/compare maps and the two-laptop UI.
7. API and deployment instructions, integration evidence and validation summary.
8. Limitations, transparent accuracy terminology and **a separate evaluator-test results section once actually run**.

---

## 11. Suggested presentation summary

> PathSnap is a GPS map-matching and route-reconstruction system. We process GPS traces and a directed road network, generate nearby road candidates, engineer 25 features and compare five ML classifiers using grouped cross-validation. We tune the leading models using Optuna and combine the tuned Random Forest with an HMM-style Viterbi decoder to produce feasible road sequences. On 10 original student-testing trips (19,160 observations), the final system achieved **87.15% qualified Top-1 accuracy** and **94.51% direct road continuity**. We demonstrate the output through a FastAPI backend and an independent Streamlit frontend operating across two laptops. Results on the separately supplied evaluator-test trips are pending.

---

## 12. Documentation note

This file is a **single consolidated project overview** assembled from the project details and actual results shared during development. It is not a word-for-word merge of all twelve `docs/steps/STEP_XX_README.md` files, which were not supplied for direct inspection here. Those original detailed stage files may be retained as supporting technical evidence even if this consolidated README becomes the main project entry point.
