import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from pathsnap.features import FEATURES,IDENTIFIERS
from pathsnap.modeling import make_pipeline
from pathsnap.evaluation import predict_frame,evaluation_alignment,qualify_evaluation,score_evaluation,top1
from pathsnap.continuity import DirectedReachability,continuity_diagnostics,summarise_continuity

def candidate_fixture():
    return pd.DataFrame(dict(dataset_partition=['student_test']*4,trip_id=['00000007']*4,
        gps_observation_index=[0,0,1,1],candidate_arc_id=[0,1,0,1]))

def alignment_fixture():
    return pd.DataFrame(dict(supervised_eligible=[True,True],aligned_arc_id=pd.array([0,99],dtype='Int64'),alignment_status=['valid','valid']))

def test_candidate_miss_retained_in_denominator():
    c=candidate_fixture();labels,obs=qualify_evaluation(c,alignment_fixture())
    assert labels.evaluation_target.tolist()[:2]==[1,0]
    assert labels.evaluation_target.iloc[2:].isna().all()
    pred=c.assign(positive_probability=[.9,.1,.8,.2]);r=score_evaluation(pred,labels,obs,2)
    assert r['top1_accuracy']==1 and r['top1_including_candidate_misses']==.5
    assert r['candidate_miss_rate']==.5 and r['classification_rows']==2
    assert r['macro_f1']==1 and r['prediction_coverage']==1

def test_ambiguous_not_in_qualified_denominator():
    a=alignment_fixture();a.loc[1,['supervised_eligible','alignment_status']]=[False,'ambiguous']
    labels,obs=qualify_evaluation(candidate_fixture(),a)
    r=score_evaluation(candidate_fixture().assign(positive_probability=[.1,.9,.8,.2]),labels,obs,2)
    assert r['qualified_alignments']==1 and r['candidate_miss_rate']==0 and r['top1_accuracy']==0

def test_no_present_candidate_gives_undefined_conditional_metric():
    c=candidate_fixture().iloc[2:];a=alignment_fixture()
    labels,obs=qualify_evaluation(c,a)
    r=score_evaluation(c.assign(positive_probability=.5),labels,obs,2)
    assert r['top1_accuracy'] is None and r['top1_including_candidate_misses']==0

def test_prediction_identity_mismatch_rejected():
    c=candidate_fixture();labels,obs=qualify_evaluation(c,alignment_fixture())
    with pytest.raises(ValueError,match='identity'):score_evaluation(c.iloc[:3].assign(positive_probability=.5),labels,obs,2)

def test_top1_tie_break_and_order():
    p=candidate_fixture().assign(positive_probability=.5).iloc[::-1]
    assert top1(p).gps_observation_index.tolist()==[0,1]
    assert top1(p).candidate_arc_id.tolist()==[0,0]

def test_prediction_reload_never_fits_and_rejects_labels(tmp_path,monkeypatch):
    rng=np.random.default_rng(42);X=pd.DataFrame(rng.normal(size=(20,len(FEATURES))),columns=FEATURES)
    pipeline=make_pipeline('gaussian_nb');pipeline.fit(X,np.tile([0,1],10))
    path=tmp_path/'pipeline.joblib';joblib.dump(pipeline,path);pipeline=joblib.load(path)
    f=pd.concat([pd.concat([candidate_fixture()]*5,ignore_index=True),X],axis=1)
    monkeypatch.setattr(Pipeline,'fit',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Refit forbidden')))
    p=predict_frame(pipeline,f);assert list(p.columns)==IDENTIFIERS+['positive_probability']
    assert np.isfinite(p.positive_probability).all()
    with pytest.raises(ValueError):predict_frame(pipeline,f.assign(reference_arc_id=1))

def test_alignment_uses_identical_step3_method(monkeypatch):
    import pathsnap.evaluation as module
    calls=[];sentinel=object()
    monkeypatch.setattr(module,'align_training',lambda n,c,r,**kwargs:calls.append(kwargs) or sentinel)
    config=dict(alignment_radius_m=75,ambiguity_cost_margin=2,heading_penalty_m=20,backward_progress_tolerance_m=5,gap_s=5,speed_kmh=180,jump_m=500)
    assert evaluation_alignment(None,None,None,config) is sentinel
    assert calls==[dict(radius_m=75,ambiguity_margin=2)]
    with pytest.raises(ValueError):evaluation_alignment(None,None,None,{**config,'alignment_radius_m':150})

def continuity_fixture():
    nodes=pd.DataFrame(index=range(7))
    arcs=pd.DataFrame({'from_node_id':[0,1,2,4,5],'to_node_id':[1,2,3,5,6]})
    return arcs,DirectedReachability(nodes,arcs)

@pytest.mark.parametrize('arc_ids,expected', [([0,0],'same_arc'),([0,1],'directly_connected'),([0,2],'intermediate_required'),([0,3],'unreachable')])
def test_continuity_categories(arc_ids,expected):
    arcs,reach=continuity_fixture();w=pd.DataFrame({'gps_observation_index':[0,1],'candidate_arc_id':arc_ids})
    obs=pd.DataFrame({'timestamp_s':[1,2]});d=continuity_diagnostics(w,obs,arcs,reach)
    assert d.category.tolist()==[expected]
    assert summarise_continuity(d)['assessable_pairs']==1

def test_gaps_missing_predictions_and_invalid_time_not_bridged():
    arcs,reach=continuity_fixture();w=pd.DataFrame({'gps_observation_index':[0,1,3],'candidate_arc_id':[0,1,2]})
    obs=pd.DataFrame({'timestamp_s':[1,9,10,10]});d=continuity_diagnostics(w,obs,arcs,reach)
    assert d.category.tolist()==['temporal_gap','missing_prediction','invalid_timestamp']
    assert summarise_continuity(d)['direct_continuity'] is None

def test_same_arc_backward_projection_explicitly_flagged():
    arcs,reach=continuity_fixture();w=pd.DataFrame({'gps_observation_index':[0,1],'candidate_arc_id':[0,0],'position_along_arc':[.9,.1]})
    d=continuity_diagnostics(w,pd.DataFrame({'timestamp_s':[1,2]}),arcs,reach)
    assert summarise_continuity(d)['same_arc_backward_fraction']==1

def test_saved_report_consistency_and_cv_preservation():
    output=Path(__file__).resolve().parents[1]/'outputs/step5'
    student=pd.read_csv(output/'student_test_comparison.csv');per_trip=pd.read_csv(output/'per_trip_metrics.csv')
    compared=pd.read_csv(output/'cv_vs_student.csv').set_index('model_name')
    cv=pd.read_csv(output.parent/'step4/model_comparison.csv').set_index('model_name')
    for row in student.itertuples():
        trip=per_trip.loc[per_trip.model_name==row.model_name]
        assert len(trip)==10 and trip.raw_observations.sum()==row.raw_observations
        assert trip.candidate_misses.sum()==row.candidate_misses
        positive_f1=2*row.tp/(2*row.tp+row.fp+row.fn)
        negative_f1=2*row.tn/(2*row.tn+row.fp+row.fn)
        assert row.f1==pytest.approx(positive_f1)
        assert row.macro_f1==pytest.approx((positive_f1+negative_f1)/2)
        assert row.top1_accuracy==pytest.approx(np.average(trip.top1_accuracy,weights=trip.candidate_present_alignments))
        assert compared.loc[row.model_name,'macro_f1_mean']==pytest.approx(cv.loc[row.model_name,'macro_f1_mean'])
