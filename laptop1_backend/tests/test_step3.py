import numpy as np
import pandas as pd
import pytest
from pathsnap.preprocessing import clean_observations
from pathsnap.alignment import align_training
from pathsnap.features import engineer_features,supervision,trip_splits,FEATURES
from pathsnap.road_network import RoadNetwork
from pathsnap.candidates import generate_candidates

def fixture():
    nodes=pd.DataFrame({'longitude':[8,8.001,8.002,8.003],'latitude':[49]*4})
    arcs=pd.DataFrame({'from_node_id':[0,1,2],'to_node_id':[1,2,3]})
    net=RoadNetwork(nodes,arcs)
    track=pd.DataFrame({'longitude':[8.0002,8.0004,8.0012,8.0014,8.0022],'latitude':[49]*5,'timestamp_s':[1.,2.,3.,4.,5.]})
    return net,track,pd.DataFrame({'arc_id':[0,1,2]})

def test_clean_preserves_raw_and_flags_duplicates():
    _,t,_=fixture();t=pd.concat([t,t.iloc[[-1]]],ignore_index=True); before=t.copy()
    c=clean_observations(t)
    pd.testing.assert_frame_equal(c[t.columns],before)
    assert c.duplicate_record.iloc[-1] and c.bad_timestamp.iloc[-1]
    assert not c.supervision_quality_ok.iloc[-1]

def test_gap_segmentation_and_undefined_heading():
    _,t,_=fixture(); t.loc[2:,'timestamp_s']+=100
    c=clean_observations(t);assert c.large_gap.sum()==1
    assert c.temporal_segment.iloc[2]>c.temporal_segment.iloc[1]
    assert np.isnan(c.gps_heading_deg.iloc[2])

def test_invalid_and_jump_retained():
    _,t,_=fixture();t.loc[1,'longitude']=np.nan;t.loc[3,'longitude']=9
    c=clean_observations(t);assert len(c)==5 and c.invalid_coordinates.sum()==1 and c.gps_jump.any()

def test_alignment_monotonic_and_directed():
    net,t,r=fixture();c=clean_observations(t);a=align_training(net,c,r)
    assert a.route_position.dropna().is_monotonic_increasing
    assert a.aligned_arc_id.tolist()==[0,0,1,1,2]
    assert a.supervised_eligible.any()

def test_repeated_occurrences_ambiguous():
    net,t,_=fixture();r=pd.DataFrame({'arc_id':[0,0]})
    a=align_training(net,clean_observations(t.iloc[:1]),r)
    assert a.alignment_status.iloc[0]=='ambiguous'

def test_distant_observation_not_forced():
    net,t,r=fixture();t.loc[:,'latitude']=50
    a=align_training(net,clean_observations(t),r);assert a.alignment_status.eq('unmatched').all()

def test_disconnected_reference_explicit_restart():
    net,t,_=fixture();r=pd.DataFrame({'arc_id':[0,2]})
    a=align_training(net,clean_observations(t.iloc[[0,4]]),r)
    assert a.reference_component.nunique()==2 and a.sequence_restart.iloc[-1]
    assert not a.supervised_eligible.iloc[-1]

def test_features_causal_and_label_free():
    net,t,_=fixture();c=next(generate_candidates(net,t,'train','00000001'))[0]
    f=engineer_features(c,clean_observations(t),net)
    assert list(f.columns[4:])==FEATURES
    assert 'aligned_arc_id' not in f and 'target' not in f
    assert np.allclose(f.distance_from_node_m+f.distance_to_node_m,net.arcs.loc[c.candidate_arc_id].projected_length_m)
    changed=t.copy();changed.iloc[-1,0]+=1
    f2=engineer_features(c,clean_observations(changed),net)
    pd.testing.assert_frame_equal(f.iloc[c.gps_observation_index.to_numpy()<4],f2.iloc[c.gps_observation_index.to_numpy()<4])

def test_labels_and_recall_no_all_negative_missing_positive():
    net,t,r=fixture();c=next(generate_candidates(net,t,'train','00000001'))[0]
    a=align_training(net,clean_observations(t),r);labels,recall=supervision(c,a,'train')
    assert labels.target.eq(1).sum()>0 and recall.candidate_present.all()
    c=c.loc[c.candidate_arc_id!=0]; labels,recall=supervision(c,a,'train')
    assert not labels.loc[labels.gps_observation_index==0,'training_eligible'].any()
    assert not recall.loc[[0],'candidate_present'].any()

def test_student_labels_prohibited():
    with pytest.raises(ValueError,match='training-only'):supervision(None,None,'student_test')

def test_deterministic_trip_folds():
    ids=[f'{i:08}' for i in range(80)];a=trip_splits(ids);b=trip_splits(ids[::-1])
    pd.testing.assert_frame_equal(a,b);assert a.trip_id.nunique()==80 and a.validation_fold.value_counts().eq(16).all()

def test_alignment_deterministic_and_input_unchanged():
    net,t,r=fixture();c=clean_observations(t);before=c.copy()
    pd.testing.assert_frame_equal(align_training(net,c,r),align_training(net,c,r))
    pd.testing.assert_frame_equal(c,before)
