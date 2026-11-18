"""Independent saved-map integrity checks; never imports a classifier or decoder."""
from pathlib import Path
import sys,json,hashlib,re,argparse
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
import numpy as np
import pandas as pd
from pathsnap.loaders import load_nodes,load_route
from pathsnap.quality import valid_coordinates
from pathsnap.map_visualization import LAYER_NAMES,render_map


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.scripts=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag in ['a','link'] and 'href' in attrs:self.links.append(attrs['href'])
        if tag=='script' and 'src' in attrs:self.scripts.append(attrs['src'])


def check_links(path,root):
    parser=Links();parser.feed(path.read_text(encoding='utf-8'));parser.close()
    for value in parser.links+parser.scripts:
        if not value or value.startswith('#') or urlsplit(value).scheme:continue
        resolved=(path.parent/unquote(value.split('#')[0])).resolve()
        assert resolved.is_relative_to(root.resolve()) and resolved.is_file(),(path,value)
    return parser


def verify(output):
    protected=json.loads((output/'protected_artifacts.json').read_text())
    for relative,value in protected.items():assert digest(B.parent/relative)==value,relative
    summary=pd.read_csv(output/'summary.csv',dtype={'trip_id':str});metrics=pd.read_csv(B/'outputs/step7/per_trip_comparison.csv',dtype={'trip_id':str})
    assert len(summary)==10 and summary.trip_id.nunique()==10
    assets=json.loads((output/'assets/manifest.json').read_text())
    for relative,info in assets.items():assert digest(output/'assets'/relative)==info['sha256']
    parser=check_links(output/'index.html',output);assert len([p for p in parser.links if p.startswith('maps/')])==10
    checked_coordinates=0;route_count=0;route_segments=0
    for row in summary.itertuples():
        trip=row.trip_id;html_path=output/row.map_href;meta=json.loads((output/'metadata'/f'{trip}.json').read_text())
        data_path=output/'data'/f'{trip}.json';payload=json.loads(data_path.read_text());layers=payload['layers'];display=payload['metadata']
        assert digest(html_path)==meta['html_sha256'] and digest(data_path)==meta['data_sha256']
        assert set(layers)==set(LAYER_NAMES)
        for relative,value in meta['identity']['inputs'].items():assert digest(B/relative)==value
        html=html_path.read_text(encoding='utf-8');parser=check_links(html_path,output)
        assert set(parser.scripts)=={'../assets/leaflet.js','../assets/jquery.min.js'}
        assert all(name in html or json.dumps(name)[1:-1] in html for name in LAYER_NAMES.values())
        assert 'https://tile.openstreetmap.org/' in html and 'OpenStreetMap' in html
        for key,layer in layers.items():
            assert layer['type']=='FeatureCollection' and len({f['id'] for f in layer['features']})==len(layer['features'])
            assert len(layer['features'])==display['layer_counts'][key]
            for f in layer['features']:
                coords=np.asarray(f['geometry']['coordinates'],dtype=float).reshape(-1,2)
                assert np.isfinite(coords).all() and (abs(coords[:,0])<=180).all() and (abs(coords[:,1])<=90).all()
                checked_coordinates+=len(coords)
        obs=pd.read_parquet(B/'data/processed/step3/student_test'/trip/'observations.parquet');valid=obs.loc[valid_coordinates(obs)]
        nodes=load_nodes(B/'data/raw/student_test'/trip/f'{trip}.nodes');arcs=pd.read_parquet(B/'data/processed/step2/student_test'/trip/'road_arcs.parquet').set_index('arc_id')
        folder=B/'outputs/step7/trips'/trip;decoded=pd.read_parquet(folder/'decoded.parquet');saved=pd.read_parquet(folder/'route_arcs.parquet')
        features=pd.read_parquet(B/'data/processed/step3/student_test'/trip/'features.parquet')
        assert [f['properties']['observation_index'] for f in layers['gps']['features']]==valid.index.tolist()
        np.testing.assert_array_equal([f['geometry']['coordinates'] for f in layers['gps']['features']],valid[['longitude','latitude']].to_numpy())
        for endpoint,idx in zip(layers['endpoints']['features'],[valid.index[0],valid.index[-1]]):
            assert endpoint['properties']['observation_index']==idx
            np.testing.assert_array_equal(endpoint['geometry']['coordinates'],valid.loc[idx,['longitude','latitude']])
        assert {f['properties']['arc_id'] for f in layers['candidates']['features']}==set(features.candidate_arc_id)
        for variant,key,file in [('classifier','classifier','classifier_selected'),('viterbi','selected','decoded')]:
            winners=pd.read_parquet(folder/f'{file}.parquet')
            assert {f['properties']['arc_id'] for f in layers[key]['features']}==set(winners.candidate_arc_id.dropna())
        route=layers['route']['features']
        assert [(f['properties']['segment_id'],f['properties']['route_order'],f['properties']['arc_id'],f['properties']['role']) for f in route]==list(saved[['segment_id','route_order','arc_id','role']].itertuples(index=False,name=None))
        assert [f['id'] for f in layers['intermediate']['features']]==[f['id'] for f in route if f['properties']['role']=='intermediate']
        for f in route:
            p=f['properties'];a=arcs.loc[p['arc_id']];xy=nodes.loc[[int(a.from_node_id),int(a.to_node_id)],['longitude','latitude']].to_numpy()
            chosen=decoded.loc[decoded.segment_id.eq(p['segment_id'])];segment=saved.loc[saved.segment_id.eq(p['segment_id'])]
            lo=float(chosen.position_along_arc.iloc[0]) if p['route_order']==0 else 0.
            hi=float(chosen.position_along_arc.iloc[-1]) if p['route_order']==len(segment)-1 else 1.
            assert p['start_fraction']==lo and p['end_fraction']==hi and 0<=lo<=hi<=1
            expected=np.asarray([xy[0]+lo*(xy[1]-xy[0]),xy[0]+hi*(xy[1]-xy[0])])
            if f['geometry']['type']=='Point':expected=expected[0]
            np.testing.assert_allclose(f['geometry']['coordinates'],expected,rtol=0,atol=1e-12)
        for segment,group in saved.groupby('segment_id'):
            ids=group.arc_id.tolist()
            for left,right in zip(ids,ids[1:]):assert arcs.loc[left,'to_node_id']==arcs.loc[right,'from_node_id']
        for key,reason in [('gaps','temporal_gap'),('disconnected','no_feasible_transition')]:
            expected=decoded.loc[decoded.break_reason.eq(reason),'gps_observation_index'].tolist()
            assert [f['properties']['observation_index'] for f in layers[key]['features']]==expected
            assert all(f['geometry']['type']=='Point' for f in layers[key]['features'])
        assert all(f['geometry']['type']=='Point' for f in layers['unreachable']['features'])
        ref=load_route(B/'data/raw/student_test'/trip/f'{trip}.route')
        assert [f['properties']['arc_id'] for f in layers['reference']['features']]==ref.arc_id.tolist()
        for variant,prefix in [('classifier','classifier'),('viterbi','viterbi')]:
            source=metrics.loc[(metrics.trip_id==trip)&(metrics.variant==variant)].iloc[0]
            assert np.isclose(getattr(row,prefix+'_top1'),source.top1_accuracy)
            assert np.isclose(getattr(row,prefix+'_continuity'),source.direct_continuity)
        assert row.observations==len(obs) and row.decoded_segments==decoded.segment_id.nunique()
        trip_metrics={name:metrics.loc[(metrics.trip_id==trip)&(metrics.variant==name)].iloc[0].to_dict() for name in ['classifier','viterbi']}
        # Real-data HTML regeneration only; no feature/ML/decoder processing.
        if trip=='00000065':assert render_map(trip,layers,display,trip_metrics)==html
        route_count+=len(route);route_segments+=row.decoded_segments
    examples=json.loads((output/'examples.json').read_text());assert any(e['trip_id']=='00000065' for e in examples)
    for e in examples:check_links(output/'examples'/f'{e["trip_id"]}.html',output)
    browser=json.loads((output/'browser_verification.json').read_text());assert browser['status']=='passed' and browser['maps_opened']==10
    result=dict(status='passed',trip_maps=10,observations=int(summary.observations.sum()),route_arc_occurrences=route_count,
        separate_route_segments=int(route_segments),coordinate_pairs_checked=checked_coordinates,
        protected_files_unchanged=len(protected),local_assets_verified=len(assets),relative_links_valid=True,
        reference_is_evaluation_overlay_only=True,no_classifier_or_decoder_execution=True,
        reproducible_real_map='00000065',browser_maps_opened=browser['maps_opened'])
    (output/'validation_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=B/'outputs/step8');a=p.parse_args();verify(a.output)
