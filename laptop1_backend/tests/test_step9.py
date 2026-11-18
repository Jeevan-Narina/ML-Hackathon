import copy,json,hashlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from pathsnap.api import create_app
from pathsnap.api_schemas import MAX_BODY_BYTES,MAX_GPS

B=Path(__file__).resolve().parents[1]


def supported_request():
    return dict(gps=[dict(observation_id=10+i,longitude=77.0001+i*.0001,latitude=12.,timestamp_s=float(i)) for i in range(3)],
        road_network=dict(nodes=[dict(node_id=100,longitude=77.,latitude=12.),dict(node_id=200,longitude=77.001,latitude=12.)],
            arcs=[dict(arc_id=900,from_node_id=100,to_node_id=200),dict(arc_id=901,from_node_id=200,to_node_id=100)]))


@pytest.fixture(scope='module')
def client():
    with TestClient(create_app(cors_origins=['http://laptop2.test:8501'])) as c:yield c


def test_health_model_loaded_once_and_decoder_ready(client):
    first=client.get('/health');second=client.get('/health')
    assert first.status_code==200 and first.json()['model_ready'] and first.json()['decoder_ready']
    assert first.json()['model_loads']==second.json()['model_loads']==1
    assert 'x-request-id' in first.headers and 'C:' not in first.text


def test_trip_list_is_saved_demo_allowlist(client):
    trips=client.get('/trips').json()['trips']
    assert len(trips)==10 and sum(t['observations'] for t in trips)==19160
    assert all(t['map_available'] for t in trips)


def test_demo_reuses_saved_outputs_without_inference(client,monkeypatch):
    import pathsnap.api_service as service
    def forbidden(*a,**k):raise AssertionError('Demo endpoints must not run inference or decoding')
    monkeypatch.setattr(service,'decode',forbidden);monkeypatch.setattr(service,'predict_frame',forbidden)
    r=client.get('/trips/00000065');assert r.status_code==200
    data=r.json();assert data['source']=='saved_demo' and len(data['gps'])==768
    assert data['diagnostics']['segments']==14 and data['metrics']['viterbi']['top1_accuracy']<data['metrics']['classifier']['top1_accuracy']
    assert 'reference' not in json.dumps(data['viterbi_route']) and 'aligned_arc_id' not in r.text


@pytest.mark.parametrize('url',['/trips/99999999','/trips/abcdefgh','/trips/%2e%2e/map','/map-assets/%2e%2e/summary.csv'])
def test_unknown_and_path_like_ids_do_not_expose_files(client,url):
    r=client.get(url);assert r.status_code in [404,422] and 'C:\\Users' not in r.text


def test_existing_map_and_local_assets_work_under_http(client):
    path=B/'outputs/step8/maps/00000065.html';before=hashlib.sha256(path.read_bytes()).hexdigest()
    r=client.get('/trips/00000065/map');assert r.status_code==200 and r.headers['content-type'].startswith('text/html')
    assert '/map-assets/leaflet.js' in r.text and 'href="/maps"' in r.text and '../assets/' not in r.text
    assert client.get('/map-assets/leaflet.js').status_code==200
    assert client.get('/maps').status_code==200 and '/trips/00000065/map' in client.get('/maps').text
    assert client.get('/maps/summary.csv').status_code==200
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def test_real_frozen_model_prediction_and_geojson(client,monkeypatch):
    from sklearn.pipeline import Pipeline
    monkeypatch.setattr(Pipeline,'fit',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Never fit')))
    r=client.post('/predict',json=supported_request());assert r.status_code==200,r.text
    data=r.json();assert data['source']=='live' and data['metrics'] is None
    assert data['diagnostics']['observations']==3 and data['diagnostics']['matched_observations']==3
    assert [p['observation_index'] for p in data['gps']]==[10,11,12]
    assert data['viterbi_route']['features'] and data['classifier_predictions']
    assert all(0<=p['positive_probability']<=1 for p in data['viterbi_predictions'])
    for f in data['viterbi_route']['features']:
        coords=f['geometry']['coordinates'];first=coords if f['geometry']['type']=='Point' else coords[0]
        assert 77<=first[0]<=77.001 and first[1]==12


def test_prediction_disconnected_sections_never_joined(client):
    p=dict(gps=[dict(longitude=77.00005,latitude=12,timestamp_s=0),dict(longitude=77.00205,latitude=12,timestamp_s=5)],
      road_network=dict(nodes=[dict(node_id=i,longitude=lon,latitude=12) for i,lon in enumerate([77.,77.0001,77.002,77.0021])],
        arcs=[dict(arc_id=50,from_node_id=0,to_node_id=1),dict(arc_id=60,from_node_id=2,to_node_id=3)]))
    r=client.post('/predict',json=p);assert r.status_code==200,r.text
    d=r.json();assert d['diagnostics']['segments']==2 and d['transitions']==[]
    assert any(b['reason']=='no_feasible_transition' for b in d['segment_breaks'])
    assert len(d['viterbi_route']['features'])==2 and all(f['geometry']['type']=='Point' for f in d['viterbi_route']['features'])


def test_prediction_no_candidates_returns_unmatched_not_fake_accuracy(client):
    p=supported_request()
    for row in p['gps']:row['longitude']+=.1
    r=client.post('/predict',json=p);assert r.status_code==200,r.text
    d=r.json();assert d['diagnostics']['unmatched_observations']==3 and d['viterbi_route']['features']==[] and d['metrics'] is None


def test_temporal_gap_and_implicit_observation_ids(client):
    p=supported_request()
    for row in p['gps']:row.pop('observation_id')
    p['gps'][-1]['timestamp_s']=20
    r=client.post('/predict',json=p);assert r.status_code==200,r.text
    assert [x['observation_index'] for x in r.json()['gps']]==[0,1,2]
    assert any(b['reason']=='temporal_gap' for b in r.json()['segment_breaks'])


@pytest.mark.parametrize('change',[
    lambda p:p.pop('road_network'),
    lambda p:p['gps'][0].update(latitude=91),
    lambda p:p['gps'][1].update(timestamp_s=0),
    lambda p:p['road_network']['arcs'][0].update(from_node_id=999),
    lambda p:p['road_network']['nodes'].append(copy.deepcopy(p['road_network']['nodes'][0])),
    lambda p:p.update(reference_route=[900]),
    lambda p:p['gps'][0].update(target=1),
    lambda p:p['gps'][0].update(longitude=float('nan')),
])
def test_invalid_inputs_and_reference_fields_are_rejected(client,change):
    payload=supported_request();change(payload)
    r=client.post('/predict',content=json.dumps(payload),headers={'Content-Type':'application/json'})
    assert r.status_code==422 and r.json()['error']['code']=='validation_error'
    assert all('input' not in detail for detail in r.json()['error']['details'])


def test_raw_body_and_observation_limits(client):
    r=client.post('/predict',content=b'x'*(MAX_BODY_BYTES+1));assert r.status_code==413
    p=supported_request();p['gps']=[dict(longitude=77.,latitude=12.,timestamp_s=i) for i in range(MAX_GPS+1)]
    assert client.post('/predict',json=p).status_code==422
    assert client.post('/predict',content=b'not JSON',headers={'Content-Type':'application/json'}).status_code==422


def test_chunked_body_limit_without_content_length(client):
    content=iter([b'x'*(MAX_BODY_BYTES//2),b'x'*(MAX_BODY_BYTES//2+1)])
    assert client.post('/predict',content=content,headers={'Content-Type':'application/json'}).status_code==413


def test_live_inference_has_backpressure(client):
    lock=client.app.state.service.predict_lock;lock.acquire()
    try:
        r=client.post('/predict',json=supported_request());assert r.status_code==503 and r.headers['retry-after']=='1'
    finally:lock.release()


def test_cors_allowlist_and_openapi_contract(client):
    r=client.options('/predict',headers={'Origin':'http://laptop2.test:8501','Access-Control-Request-Method':'POST'})
    assert r.headers['access-control-allow-origin']=='http://laptop2.test:8501'
    r=client.options('/predict',headers={'Origin':'http://unlisted.test','Access-Control-Request-Method':'POST'})
    assert 'access-control-allow-origin' not in r.headers
    schema=client.get('/openapi.json').json()
    assert '/predict' in schema['paths'] and 'PredictRequest' in schema['components']['schemas'] and 'RouteResponse' in schema['components']['schemas']
    assert schema['paths']['/health']['get']['responses']['503']['content']['application/json']['schema']['$ref'].endswith('/HealthResponse')
    assert client.get('/docs').status_code==200


def test_missing_server_artifacts_fail_safely(tmp_path):
    with TestClient(create_app(backend=tmp_path)) as c:
        health=c.get('/health');assert health.status_code==503 and not health.json()['model_ready']
        assert str(tmp_path) not in health.text
        assert c.post('/predict',json=supported_request()).status_code==503
        c.app.state.service.trip_rows['00000000']={}
        assert c.get('/trips/00000000').status_code==503
        assert c.get('/trips/00000000/map').status_code==503
