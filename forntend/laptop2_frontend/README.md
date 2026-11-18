# Laptop 2: standalone Streamlit frontend

Runs independently using HTTP to Laptop 1. No backend code, datasets, model files or copied maps are needed. This folder is the runnable Step 10 frontend; earlier records below describe historical stage boundaries.

## Install and run on Windows

From the workspace (or use your copied frontend directory as the current folder):

```powershell
cd .\PathSnap\laptop2_frontend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Use the Streamlit URL printed in the terminal, normally http://localhost:8501. In **Backend URL**, replace localhost with Laptop 1's reachable address, for example http://<LAPTOP1_HOST>:8000, then click **Connect / refresh**. Laptop 1 must run its Step 9 FastAPI service. This initial URL can also come from PATHSNAP_BACKEND_URL; .env.example is documentation only, not automatically loaded.

Select **Saved trips → Load trip** for actual statistics/metrics and the existing interactive map. **New prediction** accepts GPS plus directed nodes/arcs JSON and displays the returned GeoJSON/diagnostics. Only backend-provided reference metrics are shown. POST submission is explicit and not repeated on ordinary UI reruns.

Copy app.py, api_client.py, example_request.json, requirements.txt, optional requirements-lock.txt, .streamlit/, scripts/, tests/ and this README to Laptop 2; create a new .venv there. Do not copy an old .venv, the backend or any model/dataset. Runtime dependencies are only Streamlit and Requests (plus their transitive packages). No Folium or ML logic runs here.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

15 frontend tests passed; local HTTP verification passed for ten trip choices and a 262-observation saved demo/map. A live prediction was not submitted in Step 10. Real two-laptop networking, browser iframe checks on Laptop 2 and final packaging remain Step 11 work. No ZIP was created.

## Historical stage records

# Original frontend placeholder (historical Steps 1–9)

Steps 1–9 maintain this directory and documentation only. There is no runnable UI,
frontend configuration loader or startup script yet.

In a later approved stage, this laptop will run a Streamlit/Folium interface,
request trip choices and predictions through Laptop 1's FastAPI HTTP service,
and display returned routes and server-calculated metrics. Configure the future
client through the `PATHSNAP_BACKEND_URL` environment variable; no IP address
will be hardcoded. `.env.example` documents the variable but is not read yet.

No raw dataset, training script, model file, reference label or backend source
belongs in this folder. Laptop 2 must install and run independently using its
own requirements, configuration, Windows startup script and instructions once
implemented. A teammate will eventually receive `Laptop2_Frontend.zip`; final
deployment archives are intentionally deferred beyond Step 2.

## Why these laptops stay separate

Laptop 1 now owns parsing, EDA, metric roads, spatial indexes and geometric
candidates; it will later own ML inference/evaluation. This folder owns only
future presentation/client concerns. It does not need raw training data, Parquet
candidates, model files or backend source. Laptop 1 dependencies were not added
to this requirements file.

The planned HTTP workflow is: request permitted trip choices, send a selected
trip ID, receive predicted geometry and server-calculated metrics, then display
them using Streamlit/Folium. Endpoint names, payloads and authentication are not
designed or implemented yet. Ground-truth labels stay server-side and do not
enter frontend prediction inputs.

## Planned prerequisites and configuration

A future runnable frontend will need compatible Python, its own virtual
environment/dependencies, a reachable Laptop 1 and a running FastAPI service.
Backend binding/port/firewall instructions will be documented when the server
exists. Set `PATHSNAP_BACKEND_URL` to the reachable backend address; no IP will
be hardcoded. `.env.example` is documentation and is not read by a loader yet.
There is currently no frontend installation/start command or operational API.
Do not install backend requirements here.

## Progress and changelog

- [x] Steps 1–2: separate folders, requirements placeholder, URL-variable and
  architecture/HTTP documentation.
- [ ] Future stage: implement Streamlit/Folium HTTP client/UI, document exact
  setup/startup commands and test actual two-laptop operation.
- [ ] Final stage: build the independently distributable frontend ZIP.

Changelog — 2026-11-18: retained the Step 1 placeholder and expanded Step 2
separation, communication, prerequisites and status documentation. Frontend
execution has not been implemented or tested. Further work requires approval.

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

## Current status: Step 10 completed

[Step 10 detailed reference](../docs/steps/STEP_10_README.md). Laptop 2 now has a standalone Streamlit app and Requests HTTP client. It supports a session-persistent backend URL (default http://localhost:8000), readiness/catalog, saved trip metrics, embedded existing maps and a JSON prediction form. Returned geometry/diagnostics are displayed without computing routes or new metrics. Reference data remains evaluation-only. No backend imports, project datasets or trained models are required by the frontend.

Frontend launch, from laptop2_frontend:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Install its own requirements.txt in its own environment. In the sidebar, enter http://<LAPTOP1_HOST>:8000 and click Connect / refresh; localhost refers to the machine running the frontend. New files: app.py, api_client.py, example_request.json, .streamlit/config.toml, tests/test_client.py, tests/test_app.py and scripts/verify_backend.py; frontend requirements/lock and .env.example are updated. Full setup/function/API details are in the linked reference.

**15 frontend tests passed**. Real local HTTP/AppTest loaded ten trip choices and the saved 262-observation demo/map; Streamlit server health passed. No live prediction was submitted, so the backend decoder was not rerun. 868 protected files remain unchanged. Prior backend results remain 192 tests passed (not rerun in Step 10). Temporary local services are stopped. Tests and verification reports are under laptop2_frontend/outputs/step10/ and laptop1_backend/outputs/step10/.

- [x] Step 10 standalone HTTP-only Streamlit UI, focused tests, local cached-demo verification and documentation.
- [ ] Step 11 real two-laptop integration, end-to-end UI prediction verification and packaging; not started.

Changelog — 2026-11-18: completed Step 10. Stop here; real two-laptop integration and final ZIP packaging require separate authorisation.
