from itertools import product
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from pathsnap.routing import DirectedRouter
from pathsnap.hmm import DecoderConfig, log_emissions, emission_states, OBSERVATION_COLUMNS, STATE_COLUMNS
from pathsnap.transitions import transition
from pathsnap.viterbi import decode
from pathsnap.features import FEATURES, IDENTIFIERS


def roads(rows=None, **kwargs):
    rows = rows or [(10,0,1,100),(20,1,2,50),(30,2,0,50),(40,3,4,100)]
    a=pd.DataFrame(rows,columns=['arc_id','from_node_id','to_node_id','arc_length_m']).set_index('arc_id')
    return DirectedRouter(a,**kwargs)


def observations(times=(0,1,2), distances=None):
    return pd.DataFrame(dict(longitude=[0.]*len(times),latitude=[0.]*len(times),timestamp_s=times,
        displacement_m=distances or [np.nan]+[10.]*(len(times)-1)),columns=OBSERVATION_COLUMNS)


def states(rows):
    return pd.DataFrame(rows,columns=STATE_COLUMNS)


def test_partial_directed_arc_and_intermediate_path():
    r=roads()
    assert r.route(10,.2,10,.8).distance_m==pytest.approx(60)
    path=r.route(10,.8,30,.2)
    assert path.arcs==(10,20,30) and path.distance_m==pytest.approx(80)
    assert r.route(10,.8,20,.2).distance_m==pytest.approx(30)


def test_backward_same_arc_requires_real_loop_and_preserves_repeated_arc():
    r=roads(); path=r.route(10,.8,10,.2)
    assert path.arcs==(10,20,30,10) and path.distance_m==pytest.approx(140)
    one_way=roads([(10,0,1,100)])
    assert one_way.route(10,.8,10,.2).status=='directed_unreachable'


def test_disconnected_opposite_direction_and_bounded_failure_are_distinct():
    r=roads(cutoff_m=30)
    assert r.route(10,.9,40,.1).status=='directed_unreachable'
    assert r.route(10,.9,30,.1).status=='distance_bound'
    limited=roads(max_settled=1)
    assert limited.between_nodes(0,2).status=='search_budget'


def test_parallel_arcs_zero_connectors_and_cache_are_deterministic():
    r=roads([(9,0,1,0),(5,0,1,0),(11,1,2,10),(12,2,0,20)],cache_size=1)
    assert r.between_nodes(0,2).arcs==(5,11)
    assert r.between_nodes(0,2).arcs==(5,11) and r.stats['cache_hits']==1
    r.between_nodes(1,0);assert len(r.cache)==1


def test_invalid_arc_and_fraction_fail_explicitly():
    assert roads().route(999,.1,10,.2).status=='unknown_arc'
    assert roads().route(10,np.nan,10,.2).status=='invalid_position'


def test_emissions_zero_tiny_and_invalid_values():
    assert np.isfinite(log_emissions([0,1,1e-300])).all()
    assert np.exp(log_emissions([.2,.6])).sum()==pytest.approx(1)
    for values in [[np.nan,.2],[-.1,.5],[1.1],[]]:
        with pytest.raises(ValueError):log_emissions(values)


def test_viterbi_uses_global_path_and_matches_exhaustive_optimum():
    obs=observations();s=states([(i,a,.1+.1*i,p) for i,ps in enumerate([(.4,.6),(.9,.1),(.8,.2)]) for a,p in zip([10,40],ps)])
    result,links,path,diag=decode(obs,s,roads())
    assert result.candidate_arc_id.tolist()==[10,10,10]
    scores=[]
    groups=[g.sort_values('candidate_arc_id') for _,g in s.groupby('gps_observation_index')]
    for choices in product(range(2),repeat=3):
        score=-np.log(2)+sum(log_emissions(g.positive_probability)[j] for g,j in zip(groups,choices))
        for i in range(1,3):score+=transition(roads(),groups[i-1].iloc[choices[i-1]],groups[i].iloc[choices[i]],1,10,DecoderConfig())[0]
        scores.append(score)
    assert result.prefix_log_score.iloc[-1]==pytest.approx(max(scores))
    assert len(links)==2 and path.arc_id.tolist()==[10]
    assert diag.feasible_pairs.tolist()==[2,2]


def test_ties_use_original_arc_id_and_repeat_identically():
    obs=observations();s=states([(i,a,.1+.1*i,.5) for i in range(3) for a in [40,10]])
    a=decode(obs,s,roads())[0];b=decode(obs,s.sample(frac=1,random_state=3),roads())[0]
    pd.testing.assert_frame_equal(a,b);assert a.candidate_arc_id.tolist()==[10,10,10]


@pytest.mark.parametrize('times,reason', [((0,1,20),'temporal_gap'),((0,1,1),'invalid_timestamp_interval')])
def test_temporal_boundaries(times,reason):
    result,links,*_=decode(observations(times),states([(i,10,.1*i,.8) for i in range(3)]),roads())
    assert result.segment_id.tolist()==[0,0,1] and result.break_reason.iloc[-1]==reason
    assert len(links)==1


def test_missing_candidates_and_invalid_observations_do_not_bridge():
    result,links,path,_=decode(observations(),states([(0,10,.1,.8),(2,10,.3,.8)]),roads())
    assert result.status.tolist()==['matched','unmatched','matched'] and len(links)==0
    assert path.segment_id.tolist()==[0,1]
    obs=observations();obs.loc[1,'longitude']=np.nan
    result=decode(obs,states([(i,10,.1*i,.8) for i in range(3)]),roads())[0]
    assert result.break_reason.iloc[1]=='invalid_observation'


def test_infeasible_transition_restarts_and_never_invents_connection():
    result,links,path,diag=decode(observations((0,1)),states([(0,10,.1,.8),(1,40,.1,.9)]),roads())
    assert result.segment_id.tolist()==[0,1] and result.break_reason.iloc[1]=='no_feasible_transition'
    assert links.empty and path.arc_id.tolist()==[10,40]
    assert diag.directed_unreachable.iloc[0]==1


def test_reconstructed_intermediate_arcs_and_loop_occurrences():
    obs=observations((0,3),[np.nan,140.])
    result,links,path,_=decode(obs,states([(0,10,.8,.8),(1,10,.2,.8)]),roads())
    assert path.arc_id.tolist()==[10,20,30,10] and links.intermediate_arcs.iloc[0]==2
    assert result.segment_id.tolist()==[0,0]


def test_movement_bounds_and_numerical_stability():
    result,links,*_=decode(observations((0,.1),[np.nan,1]),states([(0,10,0,0),(1,10,1,1e-300)]),roads())
    assert links.empty and result.break_reason.iloc[-1]=='no_feasible_transition'
    assert np.isfinite(result.prefix_log_score).all()
    result=decode(observations((0,1),[np.nan,1000]),states([(0,10,.1,.8),(1,10,.2,.8)]),roads())[0]
    assert result.break_reason.iloc[-1]=='gps_jump'


def test_empty_states_and_unknown_candidates_have_explicit_unmatched_rows():
    result=decode(observations(),states([]),roads())[0]
    assert len(result)==3 and result.status.eq('unmatched').all()
    result=decode(observations((0,)),states([(0,999,.5,.9)]),roads())[0]
    assert result.status.iloc[0]=='unmatched'


def test_reference_leakage_and_identity_mismatch_rejected():
    f=pd.DataFrame([['student_test','fixture',0,10]+[0.]*25],columns=IDENTIFIERS+FEATURES)
    p=f[IDENTIFIERS].assign(positive_probability=.5)
    assert len(emission_states(f,p))==1
    for frame in [f.assign(target=1),f.assign(aligned_arc_id=10)]:
        with pytest.raises(ValueError):emission_states(frame,p)
    with pytest.raises(ValueError):emission_states(f,p.assign(target=1))
    with pytest.raises(ValueError):emission_states(f,p.assign(candidate_arc_id=40))
    with pytest.raises(ValueError):decode(observations().assign(reference_arc=10),states([]),roads())


def test_evaluation_keeps_unmatched_and_candidate_misses_in_denominators():
    from pathsnap.route_evaluation import selected_accuracy
    selected=pd.DataFrame({'gps_observation_index':[0,1,2],'candidate_arc_id':pd.array([10,None,10],dtype='Int64')})
    labels=pd.DataFrame({'gps_observation_index':[0,1,2],'candidate_arc_id':[10,10,10],
        'evaluation_target':pd.array([1,1,None],dtype='Int8')})
    alignment=pd.DataFrame({'supervised_eligible':[True,True,True], 'reference_candidate_present':[True,True,False]})
    result=selected_accuracy(selected,labels,alignment)
    assert result['top1_accuracy']==.5 and result['top1_including_candidate_misses']==pytest.approx(1/3)
    assert result['correct']==1 and result['candidate_misses']==1


def test_shortest_routes_match_networkx_lengths_and_direction():
    import networkx as nx
    r=roads([(1,0,1,10),(2,1,2,20),(3,0,2,50),(4,2,3,5),(5,3,1,8)])
    g=nx.DiGraph()
    for _,(u,v,length) in r.arcs.items():g.add_edge(u,v,weight=length)
    for u,v in product(range(4),repeat=2):
        route=r.between_nodes(u,v)
        if nx.has_path(g,u,v):assert route.distance_m==nx.shortest_path_length(g,u,v,weight='weight')
        else:assert route.status=='directed_unreachable'


def test_opposite_arcs_require_finishing_start_arc_before_turn():
    r=roads([(10,0,1,100),(11,1,0,100)])
    route=r.route(10,.2,11,.3)
    assert route.arcs==(10,11) and route.distance_m==pytest.approx(110)


@pytest.mark.parametrize('kwargs',[{'beta_m':0},{'gap_s':np.inf},{'probability_floor':0},{'cache_sources':0}])
def test_invalid_decoder_configuration(kwargs):
    with pytest.raises(ValueError):DecoderConfig(**kwargs)


def test_frozen_pipeline_reloads_and_predicts_without_fitting(monkeypatch):
    from pathlib import Path
    import joblib
    from sklearn.pipeline import Pipeline
    from threadpoolctl import threadpool_limits
    from pathsnap.evaluation import predict_frame
    monkeypatch.setattr(Pipeline,'fit',lambda *a,**k: (_ for _ in ()).throw(AssertionError('Must never fit')))
    root=Path(__file__).resolve().parents[1]
    path=root/'models/step6/random_forest_tuned.joblib'
    if not path.exists():pytest.skip('Full-fit artifact absent in source-only checkout')
    f=pd.read_parquet(root/'data/processed/step3/student_test/00000000/features.parquet').iloc[:20]
    a,b=joblib.load(path),joblib.load(path)
    with threadpool_limits(limits=2):
        x=predict_frame(a,f);y=predict_frame(b,f)
    np.testing.assert_allclose(x.positive_probability,y.positive_probability,rtol=1e-13,atol=1e-15)
    assert len(emission_states(f,x))==20
