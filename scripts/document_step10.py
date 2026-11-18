"""Update Step 10 references and main READMEs from frontend verification evidence."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];FRONT=ROOT/'laptop2_frontend';BACK=ROOT/'laptop1_backend'


def main():
    tests=json.loads((FRONT/'outputs/step10/test_summary.json').read_text())
    http=json.loads((FRONT/'outputs/step10/backend_verification.json').read_text())
    verified=json.loads((BACK/'outputs/step10/verification.json').read_text())
    assert http['status']==verified['status']=='passed'
    setup=r'''cd .\PathSnap\laptop2_frontend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py'''
    commands=r'''# From laptop2_frontend, test without a backend or model/data files:
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# Optional initial URL; sidebar Connect / refresh can override it:
$env:PATHSNAP_BACKEND_URL = 'http://LAPTOP1_HOST:8000'
.\.venv\Scripts\python.exe -m streamlit run app.py
# Against an already running backend, GET/cache-only verification:
.\.venv\Scripts\python.exe scripts\verify_backend.py --backend-url http://localhost:8000'''
    reference=rf'''# Step 10: Standalone Streamlit frontend on Laptop 2

## Objective and completed scope

A small Streamlit app in laptop2_frontend communicates with the existing Laptop 1 FastAPI API exclusively through HTTP. It shows readiness, available trips, saved statistics/metrics, existing interactive maps and a JSON prediction form. It never imports pathsnap/backend modules, reads backend files, loads a model, creates candidates/features, computes accuracy, reconstructs routes or generates maps. Only Step 10 is implemented; real two-laptop integration and final ZIP packaging remain for Step 11.

## Architecture and isolation

Laptop 2 runs app.py → frontend api_client.py → HTTP → Laptop 1. All datasets, RF pipelines, inference, HMM/Viterbi and saved map assets stay on Laptop 1. Demo maps are embedded by backend URL in st.iframe, not copied/regenerated or loaded from a local backend directory. The browser loads the existing GET /trips/{{trip_id}}/map and its backend-served assets, preserving its controls/layers. A direct map link is also available.

The only direct frontend dependencies are Streamlit 1.67.0 and Requests 2.34.2. Tests use standard-library unittest/mock and Streamlit's included AppTest. No Folium, Scikit-learn, Joblib, FastAPI, backend schemas or project datasets/models were copied into the frontend. Streamlit installs its own transitive rendering/server dependencies; these are recorded in requirements-lock.txt and are separate from the backend environment. Tested with Python 3.14; use a compatible 64-bit Python with the pinned Streamlit wheel set (Python 3.12+ recommended).

## Windows installation and launch

```powershell
{setup}
```

Open the local URL printed by Streamlit (normally http://localhost:8501). Laptop 1 must separately run its existing Step 9 server:

```powershell
# On Laptop 1, from laptop1_backend:
.\.venv\Scripts\python.exe scripts\serve_api.py --host 0.0.0.0 --port 8000
```

To move the frontend, copy app.py, api_client.py, example_request.json, requirements.txt, optional requirements-lock.txt, .streamlit/, tests/, scripts/, README.md and .env.example to Laptop 2. Recreate .venv there; do not copy an existing Windows virtual environment. No parent PathSnap directory, backend source, model, processed dataset or saved HTML file is required by the application. Verification/test outputs are optional reports and contain no copied GPS/route datasets. Packaging a ZIP was not performed.

## Backend URL and session behaviour

The default is http://localhost:8000, optionally initialized from PATHSNAP_BACKEND_URL. In the sidebar, enter http://<LAPTOP1_HOST>:8000 using Laptop 1's reachable address and click **Connect / refresh**. localhost on Laptop 2 refers to Laptop 2, so replace it for a real separate-machine connection. No device IP is hardcoded. The URL is stored in st.session_state['backend_url']; the text input uses backend_url_input. HTTP(S) only, no URL credentials/query/fragment; optional reverse-proxy path prefixes are retained. .env.example documents the variable but no dotenv loader is used.

Connect fetches GET /health and GET /trips. A reachable degraded backend can still expose saved trips, while Predict remains disabled until model/decoder are ready. The app does not poll continuously. Changing/reconnecting the backend clears health/catalog and stale trip, map and prediction results, so an earlier backend's result is not shown under the new URL. Session state is per Streamlit session, not a shared global result cache.

## Essential UI flow

1. Connect in the sidebar. Read backend connection/model/decoder status.
2. In **Saved trips**, select an ID returned by GET /trips and click **Load trip**. GET /trips/{{trip_id}} supplies all statistics, qualification counts, metrics, selections, route geometry and diagnostics.
3. Read the supplied metric table. Top-1/direct continuity values remain backend values (fractions); no new accuracy, delta or route score is calculated. Reference-based metrics are explicitly conditional on eligible labels/candidate availability.
4. Use the embedded map or open it in a new tab. All existing Before/After controls and layers come from the backend HTML. Reference layers are evaluation-only, not prediction inputs. The current API's reference-containing maps are legitimate supplied student-demo maps, not hidden evaluator data.
5. In **New prediction**, edit the JSON example and click **Predict**. The form calls POST /predict only on submission, never on unrelated reruns. Input must contain GPS and explicit road_network nodes/arcs. The backend remains the authoritative Pydantic validator.
6. Inspect returned GeoJSON, matched/segment counts, explicit breaks and diagnostics. Classifier/Viterbi selections show up to 200 rows, labelled accordingly. Geometry is displayed as the returned GeoJSON rather than building another map or inventing connections. Without metrics, the UI explicitly says accuracy is unavailable. RF candidate scores are not displayed as route posterior confidence.

The example_request.json is a small synthetic two-observation/two-node/one-directed-arc contract illustration, not copied trip data or a measured result. The form uses the exact gps, longitude, latitude, timestamp_s, road_network, node_id, arc_id, from_node_id and to_node_id fields from Step 9. Optional observation_id passes through unchanged. Extra/reference fields are not silently removed; the backend rejects them. Existing backend limits remain 500 GPS observations, 5,000 nodes, 10,000 arcs and 2 MiB. Local parsing rejects invalid JSON, nonfinite NaN/Infinity and oversized text; it does not duplicate inference or full Pydantic semantics.

## HTTP client, contracts and error handling

ApiClient uses Requests with a 3.05-second connection timeout and a 60-second read timeout. No automatic POST retry is performed, because a timed-out server may still be working. Methods are health(), trips(), trip(id), map_url(id) and predict(payload). The eight-digit trip ID guard prevents path-like IDs. trip() requires a saved_demo response with the requested trip ID; predict() requires source=live and metrics=null.

parse_health, parse_trips and parse_route check the actual fields consumed by the UI: readiness booleans/counts, allowlisted trip IDs, GPS/selections, GeoJSON FeatureCollections, diagnostics, confidence_kind and nullable backend metrics. They preserve returned values rather than synthesize missing ones. These lightweight consumer checks are not a copied backend schema implementation; full request and response definitions remain the existing FastAPI OpenAPI contracts on Laptop 1.

Friendly ClientError messages cover connection failures, timeout, invalid URL/JSON, malformed response, backend validation errors, unavailable/busy service and non-JSON HTTP errors. Backend error details use its code/message/location fields. Errors are shown with st.error; old prediction results are cleared before a failed submission. The iframe loads independently in the user's browser; if a map fails, its direct link and connection caption help diagnose browser-to-backend reachability. No extra map proxy or frontend HTTP API was added.

## Source files and function contracts

| File / function | Inputs → outputs / purpose |
|---|---|
| app.py: main | Session/sidebar/forms + HTTP responses → simple saved-trip and prediction UI |
| app.py: clear_results | Current session → clear old backend/result state on reconnect |
| app.py: show_result | Actual RouteResponse dict → counts, available metrics, GeoJSON, diagnostics, breaks and selected-candidate tables |
| api_client.py: normalize_url | User URL → validated HTTP(S) backend base URL |
| api_client.py: parse_prediction_json | Editor text → unchanged JSON request object or friendly local error |
| api_client.py: parse_health/parse_trips/parse_route | Actual API JSON → consumer-validated original values, never fabricated metrics |
| api_client.py: ApiClient | Configured URL + optional test transport → HTTP endpoint calls and structured ClientError |
| example_request.json | Small synthetic request example → initial editable prediction form |
| .streamlit/config.toml | Streamlit settings → usage telemetry disabled |
| requirements.txt / requirements-lock.txt | Two direct frontend dependencies / tested complete frontend environment |
| tests/test_client.py | Mock HTTP/errors/schema cases → client tests without network/backend/model/data |
| tests/test_app.py | AppTest + mocked HTTP client → URL/session/connection/trip/iframe/form/error flow checks |
| scripts/verify_backend.py | Backend URL → real cached-demo HTTP + AppTest checks and metadata-only verification JSON |
| ../scripts/verify_step10_local.py | Workspace-only launcher → temporary loopback backend/frontend smoke check and protected-artifact validation; not needed on Laptop 2 |
| ../scripts/document_step10.py | Actual stage evidence → this reference, summary and three current READMEs; not needed on Laptop 2 |

## Tests and actual verification

```powershell
{commands}
```

From the full workspace, local smoke verification can be reproduced with:

```powershell
.\PathSnap\laptop2_frontend\.venv\Scripts\python.exe .\PathSnap\scripts\verify_step10_local.py
```

This development-only launcher starts separate temporary FastAPI and Streamlit processes on available loopback ports, verifies the Streamlit web-server health endpoint, uses the real standalone HTTP client and AppTest to connect/load a saved demo and confirm the iframe URL, and stops both processes. It sends only GET/cache requests to the live backend. It does not submit a prediction, regenerate data/maps or perform physical two-laptop integration.

Actual frontend test result: **{tests['tests']} passed**, {tests['failures']} failures and {tests['errors']} errors. Coverage includes URL normalization/path prefixes; exact GET/POST paths and unchanged payload; timeouts/connection/non-JSON/validation/busy failures; degraded readiness; malformed/mismatched responses; trip-ID path rejection; JSON size/finite-number checks; preserving available metrics; source=live without invented metrics; connect/load/iframe rendering; mocked prediction submission only once; friendly UI errors; and clearing stale results on backend change. AppTest emitted its standard missing ScriptRunContext notice in the bare test runner; the UI itself had no execution exceptions. The current st.iframe API is used, avoiding the deprecated components.v1.iframe path.

Real local HTTP verification: {http['catalog_trips']} available trips, loaded trip {http['loaded_trip']} with {http['gps_observations']} observations, map HTTP {http['map_http_status']}, correct iframe URL and successful connect/load through AppTest. Both services stopped afterward. The new-prediction client/form path was tested with mocked HTTP responses; a new prediction was deliberately not sent to the real backend in Step 10, avoiding a decoder rerun. Step 9 already contains the independent backend live-prediction verification. No new accuracy results are claimed.

Protected prior files: **{verified['protected_files_unchanged']} unchanged**, including model/data files, Step 7 routes, Step 8 maps, backend source/dependencies/results and Steps 1–9 per-step READMEs. The earlier **192 backend tests were not rerun** because the backend is unchanged and this task avoids decoder/training work; their previous results are retained. This stage reports the 15 executed frontend tests separately, not a newly executed combined 207-test run.

Evidence: laptop2_frontend/outputs/step10/test-results.txt, test_summary.json and backend_verification.json. Workspace-only service/protection logs and verification.json are in laptop1_backend/outputs/step10/. No raw GPS, response routes, reference data, HTML map copies or model files are persisted in the frontend verification reports.

## Limitations and Step 11 requirements

The map browser and Streamlit Python process must both reach the configured backend. OSM tiles require internet. Use compatible HTTP/HTTPS origins; browser mixed-content/frame policies and a real Laptop 2 map load were not tested here. Server-side Requests calls are not subject to browser CORS, but the actual Step 9 network bind, Windows private-network firewall and Wi-Fi peer reachability still need validation on the two laptops. No firewall settings were changed automatically.

Step 11 should validate the real network address/port on both laptops, server lifecycle and recovery, embedded-map assets/controls in Laptop 2's browser, a real prediction submission from that UI, timeouts/busy/unavailable behaviour, and evaluator reference-isolation rules. The current demo maps contain evaluation-only student references; hidden evaluator labels must remain server-side. No authentication/public deployment setup, automatic prediction retry, additional maps, route calculations, frontend ML, packaging or ZIPs were implemented.

Changelog — 2026-11-18: added standalone Streamlit/Requests frontend, HTTP-only client, simple session/forms/map embedding, 15 focused tests, cached-demo local smoke verification and documentation. Stop after Step 10.
'''
    (ROOT/'docs/steps/STEP_10_README.md').write_text(reference,encoding='utf-8')
    (FRONT/'outputs/step10/STEP10_SUMMARY.md').write_text(reference,encoding='utf-8')
    status='\n\n'+rf'''## Current status: Step 10 completed

[Step 10 detailed reference]({{prefix}}STEP_10_README.md). Laptop 2 now has a standalone Streamlit app and Requests HTTP client. It supports a session-persistent backend URL (default http://localhost:8000), readiness/catalog, saved trip metrics, embedded existing maps and a JSON prediction form. Returned geometry/diagnostics are displayed without computing routes or new metrics. Reference data remains evaluation-only. No backend imports, project datasets or trained models are required by the frontend.

Frontend launch, from laptop2_frontend:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Install its own requirements.txt in its own environment. In the sidebar, enter http://<LAPTOP1_HOST>:8000 and click Connect / refresh; localhost refers to the machine running the frontend. New files: app.py, api_client.py, example_request.json, .streamlit/config.toml, tests/test_client.py, tests/test_app.py and scripts/verify_backend.py; frontend requirements/lock and .env.example are updated. Full setup/function/API details are in the linked reference.

**15 frontend tests passed**. Real local HTTP/AppTest loaded ten trip choices and the saved 262-observation demo/map; Streamlit server health passed. No live prediction was submitted, so the backend decoder was not rerun. {verified['protected_files_unchanged']} protected files remain unchanged. Prior backend results remain 192 tests passed (not rerun in Step 10). Temporary local services are stopped. Tests and verification reports are under laptop2_frontend/outputs/step10/ and laptop1_backend/outputs/step10/.

- [x] Step 10 standalone HTTP-only Streamlit UI, focused tests, local cached-demo verification and documentation.
- [ ] Step 11 real two-laptop integration, end-to-end UI prediction verification and packaging; not started.

Changelog — 2026-11-18: completed Step 10. Stop here; real two-laptop integration and final ZIP packaging require separate authorisation.
'''
    for path,prefix in [(ROOT/'README.md','docs/steps/'),(BACK/'README.md','../docs/steps/'),(FRONT/'README.md','../docs/steps/')]:
        old=path.read_text(encoding='utf-8');marker='\n<!-- STEP10_CURRENT -->'
        if marker in old:old=old.split(marker)[0]
        old=old.replace('## Current status: Step 9 completed','## Historical Step 9 completion record')
        if path!=FRONT/'README.md':
            old=old.replace('**Steps 1–9 are implemented:**','**Steps 1–10 are implemented:**')
            old=old.replace('frontend and two-laptop integration remain pending.','the standalone Streamlit frontend is implemented; real two-laptop integration remains pending.')
            old=old.replace('frontend and integration remain pending.','Step 10 adds the standalone Streamlit frontend; real two-laptop integration remains pending.')
            old=old.replace('Steps 1–9 are complete; the frontend stage requires explicit approval.','Steps 1–10 are complete; Step 11 integration requires explicit approval.')
        else:
            old=old.replace('# Laptop 2: frontend placeholder','# Original frontend placeholder (historical Steps 1–9)',1)
            quick=rf'''# Laptop 2: standalone Streamlit frontend

Runs independently using HTTP to Laptop 1. No backend code, datasets, model files or copied maps are needed. This folder is the runnable Step 10 frontend; earlier records below describe historical stage boundaries.

## Install and run on Windows

From the workspace (or use your copied frontend directory as the current folder):

```powershell
{setup}
```

Use the Streamlit URL printed in the terminal, normally http://localhost:8501. In **Backend URL**, replace localhost with Laptop 1's reachable address, for example http://<LAPTOP1_HOST>:8000, then click **Connect / refresh**. Laptop 1 must run its Step 9 FastAPI service. This initial URL can also come from PATHSNAP_BACKEND_URL; .env.example is documentation only, not automatically loaded.

Select **Saved trips → Load trip** for actual statistics/metrics and the existing interactive map. **New prediction** accepts GPS plus directed nodes/arcs JSON and displays the returned GeoJSON/diagnostics. Only backend-provided reference metrics are shown. POST submission is explicit and not repeated on ordinary UI reruns.

Copy app.py, api_client.py, example_request.json, requirements.txt, optional requirements-lock.txt, .streamlit/, scripts/, tests/ and this README to Laptop 2; create a new .venv there. Do not copy an old .venv, the backend or any model/dataset. Runtime dependencies are only Streamlit and Requests (plus their transitive packages). No Folium or ML logic runs here.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

15 frontend tests passed; local HTTP verification passed for ten trip choices and a 262-observation saved demo/map. A live prediction was not submitted in Step 10. Real two-laptop networking, browser iframe checks on Laptop 2 and final packaging remain Step 11 work. No ZIP was created.

## Historical stage records

'''
            if not old.startswith('# Laptop 2: standalone Streamlit frontend'):old=quick+old
        path.write_text(old+marker+status.replace('{prefix}',prefix),encoding='utf-8')
    print('Step 10 reference, summary and all three main READMEs updated.')


if __name__=='__main__':main()
