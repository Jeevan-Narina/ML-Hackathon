import copy
import json
from html.parser import HTMLParser
import numpy as np
import pandas as pd
import pytest
from pathsnap.features import IDENTIFIERS,FEATURES
from pathsnap.map_visualization import (arc_feature,point_feature,reconstructed_features,build_prediction_layers,
    reference_features,render_map,stable_html,LAYER_NAMES)


@pytest.fixture
def inputs():
    nodes=pd.DataFrame({'longitude':77+np.arange(6)*.001,'latitude':[12.]*6})
    arcs=pd.DataFrame({'from_node_id':[0,1,2,4],'to_node_id':[1,2,3,5]},index=[10,15,20,30])
    obs=pd.DataFrame({'longitude':[77.0002,77.0028,77.0042,77.0044],'latitude':[12.]*4,
        'timestamp_s':[0.,1.,10.,11.],'temporal_segment':[0,0,1,1]})
    rows=[]
    for oid,aid in enumerate([10,20,30,30]):
        row={k:0. for k in FEATURES};row.update(dict(zip(IDENTIFIERS,['student_test','00000001',oid,aid])))
        row.update(candidate_rank=1,distance_to_arc_m=2.,heading_difference_deg=3.,position_along_arc=.4)
        rows.append(row)
    features=pd.DataFrame(rows,columns=IDENTIFIERS+FEATURES)
    probabilities=features[IDENTIFIERS].assign(positive_probability=.8)
    decoded=pd.DataFrame({'gps_observation_index':range(4),'candidate_arc_id':pd.array([10,20,30,30],dtype='Int64'),
        'position_along_arc':[.25,.75,.2,.4],'segment_id':pd.array([0,0,1,1],dtype='Int64'),
        'status':['matched']*4,'break_reason':['trip_start','','temporal_gap','']})
    classifier=decoded[['gps_observation_index','candidate_arc_id','position_along_arc']].copy()
    routes=pd.DataFrame({'segment_id':[0,0,0,1],'route_order':[0,1,2,0],'arc_id':[10,15,20,30],
        'role':['selected_start','intermediate','selected_end','selected_start']})
    pairs=pd.DataFrame({'assessable':[True,False,True],'topologically_reachable':[True,False,True],
        'from_observation':[0,1,2],'to_observation':[1,2,3]})
    return dict(observations=obs,features=features,probabilities=probabilities,classifier=classifier,decoded=decoded,
        routes=routes,transitions=pd.DataFrame(),pair_diagnostics={'classifier':pairs,'viterbi':pairs},nodes=nodes,arcs=arcs)


def test_geographic_order_direction_and_partial_arc(inputs):
    f=arc_feature(inputs['arcs'],inputs['nodes'],10,'fixture',.25,.75)
    np.testing.assert_allclose(f['geometry']['coordinates'],[[77.00025,12],[77.00075,12]])
    assert f['properties']['from_node_id']==0 and f['properties']['to_node_id']==1


@pytest.mark.parametrize('lo,hi',[(-.1,.5),(.8,.2),(0,1.1),(np.nan,.5)])
def test_invalid_directed_fraction_rejected(inputs,lo,hi):
    with pytest.raises(ValueError):arc_feature(inputs['arcs'],inputs['nodes'],10,'x',lo,hi)


def test_reconstructed_sequence_intermediates_clipping_and_boundaries(inputs):
    f=reconstructed_features(inputs['routes'],inputs['decoded'],inputs['arcs'],inputs['nodes'])
    assert [x['properties']['arc_id'] for x in f]==[10,15,20,30]
    assert [x['properties']['segment_id'] for x in f]==[0,0,0,1]
    assert f[0]['properties']['start_fraction']==.25
    assert f[2]['properties']['end_fraction']==.75
    assert f[3]['properties']['start_fraction']==.2 and f[3]['properties']['end_fraction']==.4
    assert f[1]['properties']['role']=='intermediate'
    # No geographic chord from the end of arc 20 to disconnected arc 30.
    assert all(len(x['geometry']['coordinates'])==2 for x in f)


def test_disconnected_saved_segment_rejected(inputs):
    r=inputs['routes'].copy();r.loc[1,'arc_id']=30
    with pytest.raises(ValueError,match='Disconnected'):reconstructed_features(r,inputs['decoded'],inputs['arcs'],inputs['nodes'])


def test_loop_occurrences_are_preserved(inputs):
    a=inputs['arcs'].copy();a.loc[99]=[3,0]
    r=pd.DataFrame({'segment_id':[0]*5,'route_order':range(5),'arc_id':[10,15,20,99,10],'role':['selected_start','intermediate','intermediate','intermediate','selected_end']})
    d=inputs['decoded'].iloc[:2].copy();d['candidate_arc_id']=[10,10];d['position_along_arc']=[.8,.2]
    result=reconstructed_features(r,d,a,inputs['nodes'])
    assert [f['properties']['arc_id'] for f in result]==[10,15,20,99,10]
    assert result[0]['properties']['start_fraction']==.8 and result[-1]['properties']['end_fraction']==.2


def test_single_selected_position_is_point_not_full_arc(inputs):
    r=inputs['routes'].iloc[[0]].copy();d=inputs['decoded'].iloc[[0]]
    f=reconstructed_features(r,d,inputs['arcs'],inputs['nodes'])[0]
    assert f['geometry']['type']=='Point' and f['geometry']['coordinates']==[77.00025,12.]


def test_layers_popups_start_end_and_temporal_gaps(inputs):
    layers,meta=build_prediction_layers(**inputs)
    assert set(layers)==set(LAYER_NAMES)-{'reference'}
    assert meta['observations']==4 and meta['route_occurrences']==4
    assert layers['endpoints']['features'][0]['geometry']['coordinates']==[77.0002,12]
    assert layers['endpoints']['features'][-1]['properties']['label']=='End'
    assert len(layers['trace']['features'])==2 and len(layers['gaps']['features'])==1
    assert layers['gaps']['features'][0]['geometry']['type']=='Point'
    assert 'RF p(+)' in layers['gps']['features'][0]['properties']['detail']
    assert layers['candidates']['features'][0]['properties']['heading_disagreement_deg']=='3.00'
    assert len(layers['intermediate']['features'])==1


def test_missing_candidates_keep_gps_and_unmatched_marker(inputs):
    inputs['features']=inputs['features'].iloc[:-1];inputs['probabilities']=inputs['probabilities'].iloc[:-1]
    inputs['decoded'].loc[3,['candidate_arc_id','segment_id']]=pd.NA
    inputs['decoded'].loc[3,['status','break_reason']]=['unmatched','no_usable_candidates']
    inputs['classifier']=inputs['classifier'].iloc[:-1]
    layers,meta=build_prediction_layers(**inputs)
    assert len(layers['gps']['features'])==4 and layers['gps']['features'][-1]['properties']['candidate_count']==0
    assert len(layers['unmatched']['features'])==1


def test_invalid_gps_is_counted_not_fabricated(inputs):
    inputs['observations'].loc[1,'longitude']=np.nan
    layers,meta=build_prediction_layers(**inputs)
    assert meta['invalid_gps']==1 and len(layers['gps']['features'])==3
    with pytest.raises(ValueError):point_feature('bad',181,0)


def test_directed_unreachable_marker_has_no_connector(inputs):
    inputs['pair_diagnostics']['classifier'].loc[0,'topologically_reachable']=False
    layers,_=build_prediction_layers(**inputs)
    assert len(layers['unreachable']['features'])==2 # shared fixture diagnostic used by both variants
    assert all(f['geometry']['type']=='Point' for f in layers['unreachable']['features'])


def test_reference_overlay_is_independent_and_prediction_layers_unchanged(inputs):
    before=copy.deepcopy(inputs);layers,_=build_prediction_layers(**inputs);saved=copy.deepcopy(layers)
    reference=reference_features(pd.DataFrame({'arc_id':[30,10,30]}),inputs['nodes'],inputs['arcs'])
    assert [f['properties']['arc_id'] for f in reference['features']]==[30,10,30]
    assert all('evaluation only' in f['properties']['detail'] for f in reference['features'])
    assert layers==saved and build_prediction_layers(**before)[0]==saved
    assert 'reference' not in layers


def test_probability_identity_mismatch_rejected(inputs):
    inputs['probabilities'].loc[0,'candidate_arc_id']=999
    with pytest.raises(ValueError,match='identity'):build_prediction_layers(**inputs)


def test_html_is_reproducible_and_uses_local_assets(inputs):
    layers,meta=build_prediction_layers(**inputs)
    layers['reference']=reference_features(pd.DataFrame({'arc_id':[10]}),inputs['nodes'],inputs['arcs'])
    metrics={k:dict(top1_accuracy=.8,direct_continuity=.9) for k in ['classifier','viterbi']}
    first=render_map('00000001',layers,meta,metrics);second=render_map('00000001',layers,meta,metrics)
    assert first==second
    assert '../assets/leaflet.js' in first and '../assets/jquery.min.js' in first and '../assets/leaflet.css' in first
    assert 'cdn.jsdelivr' not in first and 'https://tile.openstreetmap.org/' in first
    assert 'Before · RF' in first and 'After · Viterbi' in first and 'window.pathsnapReady=true' in first
    assert first.rfind('var pathsnapLayers=')>first.rfind('.fitBounds(')
    assert '../index.html' in first and all(name in first or json.dumps(name)[1:-1] in first for name in LAYER_NAMES.values())
    HTMLParser().feed(first)


def test_canonical_ids_remain_consistent():
    assert stable_html('map_'+'a'*32+' = map_'+'a'*32)==stable_html('map_'+'b'*32+' = map_'+'b'*32)
