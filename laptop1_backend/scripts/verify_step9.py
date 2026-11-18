"""Start a temporary loopback API, verify real HTTP requests, then stop it."""
from pathlib import Path
import argparse,sys,json,time,socket,subprocess,os,hashlib
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
import httpx
import pandas as pd
from pathsnap.loaders import load_nodes
from pathsnap.api_schemas import PredictRequest,RouteResponse


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def supported_request():
    trip='00000052';root=B/'data/processed/step3/student_test'/trip
    obs=pd.read_parquet(root/'observations.parquet').iloc[:5]
    candidates=pd.read_parquet(B/'data/processed/step2/student_test'/trip/'candidates.parquet')
    arcs=pd.read_parquet(B/'data/processed/step2/student_test'/trip/'road_arcs.parquet')
    ids=candidates.loc[candidates.gps_observation_index.isin(obs.index),'candidate_arc_id'].unique()
    arcs=arcs.loc[arcs.arc_id.isin(ids),['arc_id','from_node_id','to_node_id']]
    nodes=load_nodes(B/'data/raw/student_test'/trip/f'{trip}.nodes').loc[sorted(set(arcs[['from_node_id','to_node_id']].to_numpy().ravel()))]
    gps=obs[['longitude','latitude','timestamp_s']].copy();gps.insert(0,'observation_id',obs.index)
    data=dict(gps=json.loads(gps.to_json(orient='records')),
        road_network=dict(nodes=json.loads(nodes.reset_index().to_json(orient='records')),arcs=json.loads(arcs.to_json(orient='records'))))
    PredictRequest.model_validate(data)
    return data


def verify(output):
    output.mkdir(parents=True,exist_ok=True)
    protected_path=output/'protected_artifacts.json'
    if not protected_path.exists():
        protected=json.loads((B/'outputs/step8/protected_artifacts.json').read_text())
        paths=[*list((B/'outputs/step8').rglob('*')),*list((B.parent/'docs/steps').glob('STEP_0[1-8]_README.md')),
            *[p for p in (B/'src/pathsnap').glob('*.py') if not p.name.startswith('api')]]
        for p in paths:
            if p.is_file():protected[p.relative_to(B.parent).as_posix()]=digest(p)
        protected_path.write_text(json.dumps(protected,indent=2))
    protected=json.loads(protected_path.read_text())
    request=supported_request();(output/'sample_predict_request.json').write_text(json.dumps(request,indent=2))
    with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    log=(output/'server_verification.log').open('w',encoding='utf-8')
    command=[sys.executable,str(B/'scripts/serve_api.py'),'--host','127.0.0.1','--port',str(port)]
    proc=subprocess.Popen(command,cwd=B,stdout=log,stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    started=time.perf_counter();calls=[]
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=60,trust_env=False) as client:
            deadline=time.monotonic()+45
            while True:
                if proc.poll() is not None:raise RuntimeError('Verification server stopped before readiness; see server_verification.log')
                try:
                    health=client.get('/health')
                    if health.status_code==200:break
                except httpx.TransportError:pass
                if time.monotonic()>deadline:raise TimeoutError('API did not become ready')
                time.sleep(.2)
            def call(method,path,**kwargs):
                tick=time.perf_counter();response=client.request(method,path,**kwargs)
                calls.append(dict(method=method,path=path,status=response.status_code,seconds=time.perf_counter()-tick))
                return response
            assert health.json()['model_loads']==1
            trips=call('GET','/trips');assert trips.status_code==200 and len(trips.json()['trips'])==10
            demo=call('GET','/trips/00000052');assert demo.status_code==200
            demo_data=RouteResponse.model_validate(demo.json());assert len(demo_data.gps)==262 and demo_data.metrics
            (output/'demo_response.json').write_text(json.dumps(demo.json(),indent=2))
            map_response=call('GET','/trips/00000052/map');assert map_response.status_code==200 and '/map-assets/leaflet.js' in map_response.text
            assert call('GET','/map-assets/leaflet.js').status_code==200
            assert call('GET','/maps').status_code==200
            assert call('GET','/docs').status_code==200
            schema=call('GET','/openapi.json');assert schema.status_code==200
            (output/'openapi.json').write_text(json.dumps(schema.json(),indent=2))
            prediction=call('POST','/predict',json=request);assert prediction.status_code==200,prediction.text
            result=RouteResponse.model_validate(prediction.json())
            assert result.source=='live' and result.metrics is None and result.diagnostics.observations==5
            assert result.diagnostics.matched_observations==5 and result.viterbi_route.features
            allowed={a['arc_id'] for a in request['road_network']['arcs']}
            assert all(f.properties['arc_id'] in allowed for f in result.viterbi_route.features)
            (output/'live_predict_response.json').write_text(json.dumps(prediction.json(),indent=2))
            assert call('POST','/predict',json={'gps':request['gps']}).status_code==422
            assert call('GET','/health').json()['model_loads']==1
            print('Real HTTP demo, map/assets, docs, schemas and new prediction passed.',flush=True)
    finally:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        log.close()
    for relative,value in protected.items():assert digest(B.parent/relative)==value,relative
    (output/'request_schema.json').write_text(json.dumps(PredictRequest.model_json_schema(),indent=2))
    (output/'response_schema.json').write_text(json.dumps(RouteResponse.model_json_schema(),indent=2))
    report=dict(status='passed',transport='real loopback HTTP to a temporary Uvicorn process',port=port,
        demonstration_trips=10,verified_demo_trip='00000052',demo_gps_observations=len(demo_data.gps),
        live_gps_observations=result.diagnostics.observations,live_nodes=len(request['road_network']['nodes']),
        live_arcs=len(request['road_network']['arcs']),live_matched=result.diagnostics.matched_observations,
        live_segments=result.diagnostics.segments,live_route_arc_occurrences=len(result.viterbi_route.features),
        live_metrics_available=False,model_loads=1,server_stopped=proc.poll() is not None,
        protected_files_unchanged=len(protected),prior_models_and_step7_step8_outputs_preserved=True,
        verification_seconds=time.perf_counter()-started,laptop2_connection_tested=False,calls=calls)
    (output/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=B/'outputs/step9');a=p.parse_args();verify(a.output)
