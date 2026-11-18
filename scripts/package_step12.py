"""Allowlist deployment files, build reproducible ZIPs and validate extracted copies.

No training, optimisation, map generation or benchmark evaluation is performed.
"""
from pathlib import Path,PurePosixPath
import ast,csv,hashlib,importlib.metadata as metadata,json,os,socket,subprocess,sys,time,urllib.request,zipfile
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
ROOT=Path(__file__).resolve().parents[1];BACK=ROOT/'laptop1_backend';FRONT=ROOT/'laptop2_frontend';OUT=ROOT/'releases'
RUNTIME=['numpy','pandas','networkx','folium','shapely','pyproj','pyarrow','scikit-learn','joblib','threadpoolctl','fastapi','pydantic','uvicorn']


def digest(data):return hashlib.sha256(data).hexdigest()
def file_hash(path):return digest(path.read_bytes())
def write_json(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def runtime_modules():
    needed={'api','__init__'};pending=['api']
    while pending:
        name=pending.pop();tree=ast.parse((BACK/'src/pathsnap'/f'{name}.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.level==1 and node.module:
                module=node.module.split('.')[0]
                if module not in needed:needed.add(module);pending.append(module)
    return sorted(needed)


def runtime_lock():
    needed=set();pending=list(RUNTIME)
    while pending:
        name=canonicalize_name(pending.pop())
        if name in needed:continue
        needed.add(name)
        for value in metadata.requires(name) or []:
            req=Requirement(value)
            if req.marker is None or req.marker.evaluate({'extra':''}):pending.append(req.name)
    return '\n'.join(f'{name}=={metadata.version(name)}' for name in sorted(needed))+'\n'


def deployment_readme(kind):
    common=r'''Install 64-bit CPython 3.14 with the Windows Python launcher. Extract the ZIP;
do not run inside the archive. Open PowerShell in the extracted folder.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

This installs pinned dependencies; internet/package-index access is needed unless
wheels are already cached. No virtual environment or Python interpreter is shipped.
Use the exact Python/package versions for the saved Scikit-learn pipeline.
Optional setup.ps1/run.ps1 helpers duplicate these commands. If local script policy
blocks a helper, use the direct commands above/below; no policy change is required.
'''
    if kind=='Laptop1_Backend':
        detail=r'''
```powershell
.\.venv\Scripts\python.exe scripts\serve_api.py --host 0.0.0.0 --port 8000
```

Keep the terminal running. Open http://localhost:8000/health, /docs or /maps.
The model/config loads once. All ten supplied student demos use saved results.
New POST /predict requires GPS plus explicit directed nodes/arcs; no labels.
Endpoints: GET /health, /trips, /trips/{id}, /trips/{id}/map; POST /predict.
The default server limits are 500 GPS, 5,000 nodes, 10,000 arcs and 2 MiB;
one live prediction runs at a time. Busy requests return 503; invalid JSON/context
returns 422 and oversized bodies return 413. New predictions have metrics=null.

Use ipconfig to find Laptop 1's reachable IPv4 address/hostname. 0.0.0.0 is a bind
address, not the URL for Laptop 2. Host/port can also be set with PATHSNAP_HOST and
PATHSNAP_PORT. If browser CORS is needed, set PATHSNAP_CORS_ORIGINS to explicit
origins before starting (default http://localhost:8501). Streamlit's server-side
HTTP calls and ordinary map iframe navigation do not require browser fetch CORS.

The RF model, its required preprocessing class/schema, frozen decoder configuration,
all ten demo observations/routes/metrics, maps, GeoJSON and local map assets are
included. Training data/entry points, baseline/HGB model binaries and raw networks
are intentionally omitted. modeling.py is needed by the serialized preprocessing;
alignment.py is retained because an existing inference import references it.
Neither training nor reference alignment is run by the API. The original artifact
manifest also describes HGB; only the primary RF binary is deployed.

Saved student-demo maps contain clearly labelled evaluation-only references. They
are not hidden evaluator data and must not be reused to expose hidden test labels.
Map tiles/Swagger CDN assets need internet; local route overlays remain available
without OSM background tiles. Maps use the supplied straight endpoint geometry.
'''
    else:
        detail=r'''
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the Streamlit URL shown, normally http://localhost:8501. In Backend URL enter
http://<LAPTOP1_HOST>:8000 and click Connect / refresh. Replace the placeholder with
Laptop 1's reachable IP/hostname. localhost on Laptop 2 refers to Laptop 2 itself.
The sidebar stores the URL for the current session. Optional initial setting:

```powershell
$env:PATHSNAP_BACKEND_URL = 'http://LAPTOP1_HOST:8000'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Load saved trips for actual server metrics and the existing map. Use the JSON form
for a new GPS/network request. Its example is synthetic contract data, not a measured
trip. The UI displays returned geometry/diagnostics and does not infer accuracy.
No backend code, project data, trained model or copied map is required on Laptop 2.
Do not install Laptop 1 requirements in this folder.
'''
    return '# '+kind.replace('_',' ')+' deployment\n\n'+common+detail+r'''
## Network/firewall troubleshooting

Both laptops must be on a reachable LAN without Wi-Fi client isolation. Start the
backend first; check /health locally, then from Laptop 2 using Laptop 1's address.
Use the same configured port in the sidebar. Windows Firewall must allow the Python
backend on the intended private network; do not disable the firewall. Streamlit's
process and the browser must both reach Laptop 1 (the iframe loads directly there).
Check VPN/proxy settings and browser mixed HTTP/HTTPS content restrictions if only
the map fails. OSM tile outages do not imply missing route data.

Timeouts do not guarantee server cancellation; check status before resubmitting.
There is no automatic POST retry. This is a LAN demonstration deployment without
production authentication/TLS or a hard inference cancellation deadline.

## Integrity and scope

MANIFEST.json records every packaged file's SHA-256/size (excluding itself).
External checksums.sha256 authenticates the distributed ZIP files against the
published release hashes. CRC/file/import/extracted smoke checks are recorded in
the workspace releases/packaging_verification.json, not claimed as a clean-machine
offline installation test. Existing installed dependency environments are used for
the extracted smoke test. Fresh installation on another Windows machine may require
network access; no wheelhouse is included. See PROJECT_SUMMARY.md for actual model
results and the distinction between prior automated tests, owner-reported physical
two-laptop checks and this packaging run.
'''


def add_generated(files,kind):
    files['README.md']=deployment_readme(kind).encode()
    files['PROJECT_SUMMARY.md']=(ROOT/'FINAL_PROJECT_SUMMARY.md').read_bytes()
    files['setup.ps1']=br'''$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
py -3.14 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.14 virtual environment creation failed.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
'''
    if kind=='Laptop1_Backend':
        files['run.ps1']=br'''param([string]$HostName = '0.0.0.0', [int]$Port = 8000)
Set-Location -LiteralPath $PSScriptRoot
& .\.venv\Scripts\python.exe scripts\serve_api.py --host $HostName --port $Port
exit $LASTEXITCODE
'''
    else:
        files['run.ps1']=br'''Set-Location -LiteralPath $PSScriptRoot
& .\.venv\Scripts\python.exe -m streamlit run app.py
exit $LASTEXITCODE
'''


def collect():
    backend={};frontend={}
    def add(files,base,path):
        if not path.is_file():raise FileNotFoundError(path)
        files[path.relative_to(base).as_posix()]=path.read_bytes()
    modules=runtime_modules()
    for name in modules:add(backend,BACK,BACK/'src/pathsnap'/f'{name}.py')
    for rel in ['scripts/serve_api.py','models/step6/random_forest_tuned.joblib','models/step6/artifact_manifest.csv','models/step6/feature_schema.json',
                'outputs/step7/configuration.json','outputs/step7/per_trip_comparison.csv','outputs/step8/index.html','outputs/step8/summary.csv']:
        add(backend,BACK,BACK/rel)
    trips=sorted((BACK/'outputs/step7/trips').iterdir());assert len(trips)==10
    for folder in trips:
        trip=folder.name;stamp=json.loads((folder/'complete.json').read_text())
        add(backend,BACK,folder/'complete.json')
        for name,value in stamp['outputs_sha256'].items():
            assert file_hash(folder/name)==value;add(backend,BACK,folder/name)
        for path in [BACK/'data/processed/step3/student_test'/trip/'observations.parquet',BACK/'outputs/step8/maps'/f'{trip}.html',
                     BACK/'outputs/step8/data'/f'{trip}.json',BACK/'outputs/step8/metadata'/f'{trip}.json']:add(backend,BACK,path)
    for subdir in ['assets','examples']:
        for path in sorted((BACK/'outputs/step8'/subdir).rglob('*')):
            if path.is_file():add(backend,BACK,path)
    backend['requirements.txt']=('\n'.join(f'{name}=={metadata.version(name)}' for name in RUNTIME)+'\n').encode()
    backend['requirements-lock.txt']=runtime_lock().encode()
    for rel in ['app.py','api_client.py','example_request.json','requirements.txt','requirements-lock.txt','.env.example','.streamlit/config.toml']:
        add(frontend,FRONT,FRONT/rel)
    add_generated(backend,'Laptop1_Backend');add_generated(frontend,'Laptop2_Frontend')
    return {'Laptop1_Backend':backend,'Laptop2_Frontend':frontend},modules


def build(kind,files):
    manifest={name:dict(sha256=digest(data),bytes=len(data)) for name,data in sorted(files.items())}
    contents={**files,'MANIFEST.json':json.dumps(manifest,indent=2).encode()}
    path=OUT/f'{kind}.zip';temporary=path.with_suffix('.partial')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in sorted(contents.items()):
            info=zipfile.ZipInfo(kind+'/'+name,date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            archive.writestr(info,data)
    temporary.replace(path)
    return path


def extract_and_validate(path,directory,kind):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            p=PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name
            assert all(part not in ['.venv','__pycache__','pytest_temp','.git'] for part in p.parts)
            assert not any(part=='train' for part in p.parts) and 'optuna' not in name.lower()
            target=(directory/name).resolve();assert target.is_relative_to(directory.resolve())
        archive.extractall(directory)
    root=directory/kind;manifest=json.loads((root/'MANIFEST.json').read_text())
    assert len(manifest)+1==len([p for p in root.rglob('*') if p.is_file()])
    for name,entry in manifest.items():assert file_hash(root/name)==entry['sha256'] and (root/name).stat().st_size==entry['bytes']
    return root,manifest


def free_port():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def smoke(back,front,log_dir):
    backend_python=BACK/'.venv/Scripts/python.exe';frontend_python=FRONT/'.venv/Scripts/python.exe'
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1'
    code="import sys,pathlib,json;sys.path.insert(0,'src');from pathsnap.api import app;import pathsnap.api;from packaging.requirements import Requirement;import importlib.metadata as m;assert pathlib.Path(pathsnap.api.__file__).resolve().is_relative_to(pathlib.Path('.').resolve());rows=[Requirement(s) for s in pathlib.Path('requirements-lock.txt').read_text().splitlines() if s];assert all(m.version(r.name) in r.specifier for r in rows);print(json.dumps({'runtime_dependencies_verified':len(rows),'source_is_extracted':True}))"
    imports=subprocess.run([str(backend_python),'-I','-c',code],cwd=back,env=env,capture_output=True,text=True,timeout=45)
    assert imports.returncode==0,imports.stderr
    frontend_imports=subprocess.run([str(frontend_python),'-I','-c',"import sys,pathlib;sys.path.insert(0,str(pathlib.Path('.').resolve()));import streamlit,requests,api_client;assert pathlib.Path(api_client.__file__).resolve().is_relative_to(pathlib.Path('.').resolve());print('standalone frontend imports passed')"],cwd=front,env=env,capture_output=True,text=True,timeout=30)
    assert frontend_imports.returncode==0,frontend_imports.stderr
    processes=[];logs=[];opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def start(command,cwd,name):
        log=(log_dir/name).open('w',encoding='utf-8');logs.append(log)
        p=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0);processes.append(p);return p
    def get(url):
        with opener.open(url,timeout=45) as response:return response.read()
    def ready(url,p):
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            if p.poll() is not None:raise RuntimeError('Extracted server exited; inspect smoke logs')
            try:return get(url)
            except OSError:time.sleep(.2)
        raise TimeoutError('Extracted server readiness timeout')
    try:
        port=free_port();base=f'http://127.0.0.1:{port}'
        server=start([str(backend_python),str(back/'scripts/serve_api.py'),'--host','127.0.0.1','--port',str(port)],back,'backend_smoke.log')
        health=json.loads(ready(base+'/health',server));assert health['model_ready'] and health['decoder_ready'] and health['model_loads']==1
        trips=json.loads(get(base+'/trips'))['trips'];assert len(trips)==10
        observations=0
        for trip in trips:
            result=json.loads(get(base+'/trips/'+trip['trip_id']));assert result['source']=='saved_demo';observations+=len(result['gps'])
            html=get(base+'/trips/'+trip['trip_id']+'/map').decode();assert '/map-assets/leaflet.js' in html and 'window.pathsnapReady=true' in html
        assert observations==19160 and get(base+'/map-assets/leaflet.js')
        assert b'/trips/00000065/map' in get(base+'/maps') and get(base+'/docs')
        request=json.loads((front/'example_request.json').read_text())
        data=urllib.request.Request(base+'/predict',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'},method='POST')
        live=json.loads(get(data));assert live['source']=='live' and live['metrics'] is None and live['diagnostics']['observations']==2
        assert live['diagnostics']['matched_observations']==2 and live['viterbi_route']['features']
        # Standalone extracted client reaches extracted API; it reads no backend files.
        client_code="import sys,pathlib,json;sys.path.insert(0,str(pathlib.Path('.').resolve()));from api_client import ApiClient;c=ApiClient(sys.argv[1]);assert len(c.trips())==10;assert c.trip('00000052')['source']=='saved_demo';print('extracted frontend HTTP passed')"
        client=subprocess.run([str(frontend_python),'-I','-c',client_code,base],cwd=front,env=env,capture_output=True,text=True,timeout=45)
        assert client.returncode==0,client.stderr
        ui_port=free_port();ui=start([str(frontend_python),'-m','streamlit','run','app.py','--server.headless=true','--server.address=127.0.0.1',f'--server.port={ui_port}'],front,'frontend_smoke.log')
        assert ready(f'http://127.0.0.1:{ui_port}/_stcore/health',ui)
        result=dict(status='passed',dependency_import_check=json.loads(imports.stdout),frontend_imports=True,model_loaded_once=True,
            cached_demo_jsons=10,cached_demo_maps=10,demo_gps_observations=observations,local_map_assets=True,swagger=True,
            tiny_live_prediction=dict(observations=2,matched=live['diagnostics']['matched_observations'],segments=live['diagnostics']['segments'],metrics=None),
            extracted_frontend_http=True,extracted_streamlit_health=True,clean_venv_install_performed=False)
    finally:
        for p in reversed(processes):
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
        for log in logs:log.close()
    result['temporary_services_stopped']=all(p.poll() is not None for p in processes)
    return result


def main():
    OUT.mkdir(exist_ok=True);protection=OUT/'protected_artifacts.json'
    if protection.exists():protected=json.loads(protection.read_text())
    else:
        protected=json.loads((BACK/'outputs/step10/protected_artifacts.json').read_text())
        paths=[*list((BACK/'outputs/step10').rglob('*')),*list((FRONT/'outputs/step10').rglob('*')),
            *list((ROOT/'docs/steps').glob('STEP_10_README.md')),*list(FRONT.glob('*.py')),
            FRONT/'requirements.txt',FRONT/'requirements-lock.txt',FRONT/'example_request.json',FRONT/'.streamlit/config.toml',
            *list((FRONT/'tests').glob('*.py')),*list((FRONT/'scripts').glob('*.py'))]
        for p in paths:
            if p.is_file():protected[p.relative_to(ROOT).as_posix()]=file_hash(p)
        write_json(protection,protected)
    files,modules=collect();records=[];extract_root=OUT/'verification'/str(time.time_ns());extract_root.mkdir(parents=True)
    extracted={}
    for kind,content in files.items():
        path=build(kind,content);root,manifest=extract_and_validate(path,extract_root,kind);extracted[kind]=root
        records.append(dict(name=path.name,bytes=path.stat().st_size,sha256=file_hash(path),files=len(manifest)+1,uncompressed_bytes=sum(v['bytes'] for v in manifest.values()),crc_passed=True,manifest_passed=True))
        print(f'Built and extracted {path.name}: {path.stat().st_size/2**20:.2f} MiB, {len(manifest)+1} files',flush=True)
    status=smoke(extracted['Laptop1_Backend'],extracted['Laptop2_Frontend'],extract_root)
    for relative,value in protected.items():assert file_hash(ROOT/relative)==value,relative
    # Exclude runtime-created __pycache__ from final ZIPs: they exist only in smoke extraction.
    report=dict(status='passed',archives=records,runtime_source_modules=modules,protected_files_unchanged=len(protected),
        smoke=status,extraction_directory=extract_root.relative_to(ROOT).as_posix(),
        physical_two_laptop_results='User-reported successful; not repeated by this packaging script')
    write_json(OUT/'packaging_verification.json',report)
    (OUT/'checksums.sha256').write_text('\n'.join(r['sha256']+'  '+r['name'] for r in records)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
