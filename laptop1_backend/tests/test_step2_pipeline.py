import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from pathsnap.candidates import CANDIDATE_SCHEMA, OBSERVATION_SCHEMA, CandidateConfig
from pathsnap.step2 import load_step1_flags, run_step2


def make_raw(tmp_path):
    raw=tmp_path/'raw'
    for split,trip_id in [('train','00000001'),('student_test','00000007')]:
        d=raw/split/trip_id
        d.mkdir(parents=True)
        for ext,text in {'nodes':'8 49\n8.001 49\n','arcs':'0 1\n1 0\n',
                         'track':'8.0005 49.00001 1\n8.0005 49.00001 153065\n8.01 49.01 153066\n'}.items():
            (d/f'{trip_id}.{ext}').write_text(text)
        # Intentionally no .route file: candidate processing must not depend on it.
    return raw


def test_pipeline_typed_parquet_unmatched_flags_and_no_reference_dependency(tmp_path):
    raw=make_raw(tmp_path)
    flags=tmp_path/'anomalies.csv'
    pd.DataFrame([{'split':'train','trip_id':'00000001','category':'gap_gt_5s','record_id':1},
                  {'split':'train','trip_id':'00000001','category':'speed_outlier','record_id':0},
                  {'split':'train','trip_id':'00000001','category':'route_disconnected_pairs','record_id':2}]).to_csv(flags,index=False)
    output,processed=tmp_path/'out',tmp_path/'derived'
    meta=run_step2(raw,output,processed,CandidateConfig(batch_size=1),step1_anomalies=flags)
    assert meta['successful_trips']==2 and meta['failed_trips']==0
    assert not meta['candidate_decisions_use_reference_routes']
    for split,trip_id in [('train','00000001'),('student_test','00000007')]:
        d=processed/split/trip_id
        c=pd.read_parquet(d/'candidates.parquet')
        o=pd.read_parquet(d/'observations.parquet')
        u=pd.read_parquet(d/'unmatched.parquet')
        assert pq.read_schema(d/'candidates.parquet').remove_metadata()==CANDIDATE_SCHEMA
        assert pq.read_schema(d/'observations.parquet').remove_metadata()==OBSERVATION_SCHEMA
        assert len(o)==3 and o.candidate_count.tolist()==[2,2,0] and len(u)==1
        assert o.dt_previous_s.iloc[1]==153064
        assert set(c.candidate_arc_id)=={0,1} and c.dataset_partition.eq(split).all()
        assert not set(c.columns)&{'label','ground_truth','reference_arc_id','route_position'}
    training=pd.read_parquet(processed/'train/00000001/observations.parquet')
    assert training.step1_gap_flag.tolist()==[False,True,False]
    assert training.step1_speed_flag.tolist()==[True,False,False]
    sample=pd.read_csv(output/'training_geometry_sample.csv')
    assert sample.dataset_partition.eq('train').all()
    html=(output/'candidate_training_trip.html').read_text(encoding='utf-8')
    assert 'No ML prediction' in html and 'Closest point on arc' in html
    assert json.loads((output/'candidate_map_metadata.json').read_text())['reference_route_used'] is False
    # Changing/adding route labels cannot change candidates, even through the runner.
    (raw/'train/00000001/00000001.route').write_text('999999\n-1\n')
    before=pd.read_parquet(processed/'train/00000001/candidates.parquet')
    run_step2(raw,output,processed,CandidateConfig(batch_size=2),partitions=('train',),make_map=False)
    pd.testing.assert_frame_equal(before,pd.read_parquet(processed/'train/00000001/candidates.parquet'))


def test_pipeline_empty_candidate_file_keeps_schema_and_records_observation(tmp_path):
    raw=make_raw(tmp_path)
    (raw/'train/00000001/00000001.track').write_text('8.1 49.1 1\n')
    meta=run_step2(raw,tmp_path/'out',tmp_path/'derived',CandidateConfig(fallback_radius_m=None),
                   partitions=('train',),make_map=False)
    assert meta['successful_trips']==1
    d=tmp_path/'derived/train/00000001'
    assert pd.read_parquet(d/'candidates.parquet').empty
    assert len(pd.read_parquet(d/'unmatched.parquet'))==1
    assert pq.read_schema(d/'candidates.parquet').remove_metadata()==CANDIDATE_SCHEMA


def test_missing_requested_trip_does_not_silently_process_something_else(tmp_path):
    raw=make_raw(tmp_path)
    with pytest.raises(ValueError,match='requested trip'):
        run_step2(raw,tmp_path/'out',tmp_path/'derived',CandidateConfig(),trip_ids=['00000099'])


def test_output_cannot_be_written_inside_original_raw_data(tmp_path):
    raw=make_raw(tmp_path)
    with pytest.raises(ValueError,match='raw source'):
        run_step2(raw,raw/'reports',tmp_path/'derived',CandidateConfig())


def test_failed_trip_has_explicit_status_and_does_not_claim_success(tmp_path):
    raw=make_raw(tmp_path)
    (raw/'train/00000001/00000001.nodes').write_text('invalid\n')
    meta=run_step2(raw,tmp_path/'out',tmp_path/'derived',CandidateConfig(),partitions=('train',),make_map=False)
    assert meta['successful_trips']==0 and meta['failed_trips']==1
    status=json.loads((tmp_path/'derived/train/00000001/status.json').read_text())
    assert status['state']=='failed'
    assert len(pd.read_csv(tmp_path/'out/failures.csv'))==1
