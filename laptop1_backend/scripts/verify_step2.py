"""Independently validate saved Step 2 outputs against original source files."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import shapely

BACKEND=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BACKEND/'src'))
from pathsnap.candidates import CANDIDATE_SCHEMA, OBSERVATION_SCHEMA, CandidateConfig
from pathsnap.eda import verify_manifest
from pathsnap.loaders import discover_trips, load_arcs, load_nodes, load_track
from pathsnap.road_network import RoadNetwork


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(raw, output, processed):
    started=time.perf_counter()
    meta=json.loads((output/'run_metadata.json').read_text())
    assert meta['failed_trips']==0 and not meta['network_only'], 'Run has failures or is network-only'
    config=CandidateConfig(**meta['configuration'])
    summary=pd.read_csv(output/'candidate_summary.csv',dtype={'trip_id':str})
    expected={(r.dataset_partition,r.trip_id):r for r in summary.itertuples()}
    assert len(summary)==meta['successful_trips']==meta['requested_trips']
    total_observations=total_candidates=total_unmatched=0
    for files in discover_trips(raw):
        if (files.split,files.trip_id) not in expected:
            continue
        directory=processed/files.split/files.trip_id
        state=json.loads((directory/'status.json').read_text())
        assert state['state']=='complete'
        assert state['configuration']==meta['configuration']
        for kind,value in state['source_sha256'].items():
            assert digest(files.path(kind))==value, f'Source changed: {files.split}/{files.trip_id}/{kind}'
        c=pd.read_parquet(directory/'candidates.parquet')
        o=pd.read_parquet(directory/'observations.parquet')
        u=pd.read_parquet(directory/'unmatched.parquet')
        arcs,track=load_arcs(files.path('arcs')),load_track(files.path('track'))
        assert pq.read_schema(directory/'candidates.parquet').remove_metadata()==CANDIDATE_SCHEMA
        assert pq.read_schema(directory/'observations.parquet').remove_metadata()==OBSERVATION_SCHEMA
        assert o.gps_observation_index.tolist()==track.index.tolist()
        assert len(c)==int(o.candidate_count.sum()) and len(u)==int((o.candidate_count==0).sum())
        assert o.candidate_count.between(0,config.max_candidates).all()
        assert c.dataset_partition.eq(files.split).all() and c.trip_id.eq(files.trip_id).all()
        assert c.candidate_arc_id.isin(arcs.index).all()
        mapped=arcs.loc[c.candidate_arc_id]
        np.testing.assert_array_equal(mapped.from_node_id,c.from_node_id)
        np.testing.assert_array_equal(mapped.to_node_id,c.to_node_id)
        original=track.loc[c.gps_observation_index]
        np.testing.assert_allclose(original.longitude,c.gps_longitude,rtol=0,atol=0,equal_nan=True)
        np.testing.assert_allclose(original.latitude,c.gps_latitude,rtol=0,atol=0,equal_nan=True)
        np.testing.assert_allclose(original.timestamp_s,c.timestamp_s,rtol=0,atol=0,equal_nan=True)
        assert c.position_along_arc.between(0,1).all()
        assert (c.distance_to_arc_m>=0).all() and (c.distance_to_arc_m<=c.search_radius_used_m).all()
        assert not c.duplicated(['gps_observation_index','candidate_arc_id']).any()
        assert (c.groupby('gps_observation_index').distance_to_arc_m.diff().dropna()>=-1e-9).all()
        assert (c.groupby('gps_observation_index').candidate_rank.apply(list)==
                c.groupby('gps_observation_index').size().apply(lambda n:list(range(1,n+1)))).all()
        actual_counts=c.groupby('gps_observation_index').size().reindex(track.index,fill_value=0)
        np.testing.assert_array_equal(actual_counts,o.candidate_count)
        candidate_obs=o.set_index('gps_observation_index').loc[c.gps_observation_index]
        np.testing.assert_array_equal(candidate_obs.fallback_used,c.fallback_used)
        assert candidate_obs.loc[c.fallback_used.to_numpy(),'primary_candidate_count'].eq(0).all()
        assert not set(c.columns)&{'label','ground_truth','reference_arc_id','route_position'}
        dt=np.r_[np.nan,np.diff(track.timestamp_s)]
        np.testing.assert_allclose(dt,o.dt_previous_s,equal_nan=True)
        row=expected[(files.split,files.trip_id)]
        assert row.gps_observations==len(o) and row.candidate_rows==len(c) and row.unmatched_observations==len(u)
        total_observations+=len(o); total_candidates+=len(c); total_unmatched+=len(u)
    # Real-data brute-force oracle includes a trip with fallback and unmatched rows.
    brute_observations=0
    for partition,trip_id in [('train','00000002'),('train','00000083'),('student_test','00000000')]:
        if (partition,trip_id) not in expected:
            continue
        files=next(f for f in discover_trips(raw,partition) if f.trip_id==trip_id)
        track=load_track(files.path('track'))
        network=RoadNetwork(load_nodes(files.path('nodes')),load_arcs(files.path('arcs')),
            max_extent_m=config.max_projection_extent_m,max_scale_error=config.max_projection_scale_error,min_segment_m=config.min_segment_m)
        directory=processed/partition/trip_id
        o=pd.read_parquet(directory/'observations.parquet')
        c=pd.read_parquet(directory/'candidates.parquet')
        ids=set(np.linspace(0,len(track)-1,min(16,len(track)),dtype=int))
        for status in ['matched_radius','matched_fallback','unmatched']:
            rows=o.loc[o.status==status]
            if len(rows): ids.add(int(rows.iloc[0].gps_observation_index))
        for observation_id in sorted(ids):
            original=track.loc[observation_id]
            point=shapely.Point(*network.projection.xy(original.longitude,original.latitude))
            distances=shapely.distance(point,network.geometries)  # All segments; deliberately no STRtree.
            keep=distances<=config.search_radius_m
            if not keep.any() and config.fallback_radius_m is not None:
                keep=distances<=config.fallback_radius_m
            positions=np.flatnonzero(keep)
            order=np.lexsort((network.index_arc_ids[positions],distances[positions]))[:config.max_candidates]
            expected_ids=network.index_arc_ids[positions[order]].tolist()
            actual=c.loc[c.gps_observation_index==observation_id,'candidate_arc_id'].tolist()
            assert actual==expected_ids, f'Brute-force mismatch: {partition}/{trip_id}/{observation_id}'
            brute_observations+=1
    checks=pd.read_csv(output/'geodesic_candidate_checks.csv')
    assert checks.passed.all()
    for name in ['candidate_distributions.png','candidate_coverage_and_runtime.png']:
        with Image.open(output/name) as picture: picture.verify()
    if meta['map_generated']:
        m=json.loads((output/'candidate_map_metadata.json').read_text())
        html=(output/'candidate_training_trip.html').read_text(encoding='utf-8')
        assert m['dataset_partition']=='train' and not m['reference_route_used']
        assert html.count('L.circleMarker(')==m['gps_points_displayed']+m['selected_candidate_rows']
        assert 'Closest point on arc' in html and 'No ML prediction' in html
    preserved=json.loads((output/'step1_preservation.json').read_text()) if (output/'step1_preservation.json').exists() else {}
    for name,value in preserved.items():
        assert digest(BACKEND/name)==value, f'Step 1 artifact altered: {name}'
    rerun=BACKEND/'.cache/step1_verification'
    if rerun.exists():
        for path in (BACKEND/'outputs/eda').glob('*.csv'):
            if path.name!='source_manifest.csv':
                pd.testing.assert_frame_equal(pd.read_csv(path),pd.read_csv(rerun/path.name))
    verify_manifest(raw,BACKEND/'outputs/eda/source_manifest.csv')
    baseline=output/'determinism_baseline.json'
    hashes=json.loads(baseline.read_text()) if baseline.exists() else {}
    for relative,value in hashes.items():
        assert digest(processed/relative)==value, f'Real-data determinism changed: {relative}'
    for name,value in meta['source_code_sha256'].items():
        assert digest(BACKEND/'src/pathsnap'/name)==value, f'Code differs from executed version: {name}'
    frontend=BACKEND.parent/'laptop2_frontend'
    assert not list(frontend.rglob('*.parquet')) and not list(frontend.rglob('*.track')) and not list(frontend.rglob('*.py'))
    suites=list(ET.parse(output/'pytest-results.xml').getroot().iter('testsuite'))
    tests={key:sum(int(s.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
    assert tests['failures']==tests['errors']==tests['skipped']==0
    result={'status':'passed','trips_verified':len(expected),'observations_verified':total_observations,
            'candidate_rows_verified':total_candidates,'unmatched_observations':total_unmatched,
            'real_brute_force_observations':brute_observations,'independent_geodesic_checks':len(checks),
            'step1_artifact_hashes_unchanged':len(preserved),'deterministic_candidate_and_status_files':len(hashes),
            'tests':tests,'runtime_seconds':time.perf_counter()-started}
    (output/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=BACKEND/'data/raw')
    parser.add_argument('--output',type=Path,default=BACKEND/'outputs/step2')
    parser.add_argument('--processed',type=Path,default=BACKEND/'data/processed/step2')
    args=parser.parse_args()
    verify(args.raw,args.output,args.processed)
