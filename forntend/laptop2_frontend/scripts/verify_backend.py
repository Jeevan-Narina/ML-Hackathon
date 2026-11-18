"""Verify the standalone frontend against a running API over HTTP only.

Reads saved demonstrations; intentionally does not submit a live prediction or
persist GPS, routes, HTML maps, models or backend datasets in the frontend.
"""
from pathlib import Path
import argparse,json,sys,time
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
import requests
from api_client import ApiClient


def verify(base_url,output):
    start=time.perf_counter();client=ApiClient(base_url)
    health=client.health();trips=client.trips();assert trips,'No demo trips available'
    trip=min(trips,key=lambda row:row['observations'])
    result=client.trip(trip['trip_id']);assert result['source']=='saved_demo'
    response=requests.get(client.map_url(trip['trip_id']),timeout=(3.05,60));response.raise_for_status()
    assert 'text/html' in response.headers['content-type'] and 'window.pathsnapReady=true' in response.text
    assert '/map-assets/leaflet.js' in response.text
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(B/'app.py')).run(timeout=30)
    app.text_input(key='backend_url_input').set_value(client.base_url)
    next(b for b in app.button if b.label=='Connect / refresh').click().run(timeout=30)
    assert not app.exception
    app.selectbox[0].select(trip['trip_id']).run(timeout=30)
    next(b for b in app.button if b.label=='Load trip').click().run(timeout=30)
    assert not app.exception and app.get('iframe')[0].proto.src==client.map_url(trip['trip_id'])
    report=dict(status='passed',communication='HTTP only',backend_ready=health['model_ready'] and health['decoder_ready'],
        catalog_trips=len(trips),loaded_trip=trip['trip_id'],gps_observations=result['diagnostics']['observations'],
        map_http_status=response.status_code,ui_connect_and_load_passed=True,iframe_url_correct=True,
        live_post_submitted=False,backend_inference_or_decoder_rerun=False,backend_data_persisted_in_frontend=False,
        two_laptop_network_tested=False,elapsed_seconds=time.perf_counter()-start)
    output.mkdir(parents=True,exist_ok=True);(output/'backend_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--backend-url',default='http://localhost:8000');parser.add_argument('--output',type=Path,default=B/'outputs/step10');args=parser.parse_args();verify(args.backend_url,args.output)
