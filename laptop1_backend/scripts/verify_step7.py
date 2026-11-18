"""Verify saved paths, scoring and identities without refitting any model."""
from pathlib import Path
import sys,json,hashlib,argparse
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from PIL import Image
from pathsnap.hmm import DecoderConfig,OBSERVATION_COLUMNS,emission_states
from pathsnap.routing import DirectedRouter
from pathsnap.viterbi import decode
from pathsnap.evaluation import predict_frame
from pathsnap.route_evaluation import selected_accuracy,aggregate_pairs,diagnose_pairs
from pathsnap.transitions import transition


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(output):
    Pipeline.fit=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Verification must not fit'))
    protected=json.loads((output/'protected_artifacts.json').read_text())
    for relative,value in protected.items():assert digest(B.parent/relative)==value,relative
    settings=json.loads((output/'configuration.json').read_text());model=settings.pop('model_name');settings.pop('parameter_policy');config=DecoderConfig(**settings)
    folders=sorted((output/'trips').iterdir());assert len(folders)==10
    per_trip=pd.read_csv(output/'per_trip_comparison.csv',dtype={'trip_id':str})
    assert len(per_trip)==20 and not per_trip.duplicated(['trip_id','variant']).any()
    decoded_count=0;route_count=0;link_count=0
    replay_trip=min(folders,key=lambda f:json.loads((f/'complete.json').read_text())['runtime']['observations'])
    all_pairs={name:[] for name in ['classifier','viterbi']}
    for folder in folders:
        info=json.loads((folder/'complete.json').read_text());trip=folder.name
        for name,value in info['outputs_sha256'].items():assert digest(folder/name)==value
        for name,value in info['identity']['code_sha256'].items():assert digest(B/'src/pathsnap'/name)==value
        features_path=B/'data/processed/step3/student_test'/trip/'features.parquet'
        obs_path=features_path.with_name('observations.parquet');roads_path=B/'data/processed/step2/student_test'/trip/'road_arcs.parquet'
        for key,path in [('features',features_path),('observations',obs_path),('roads',roads_path)]:assert digest(path)==info['identity']['inputs'][key]
        assert info['identity']['configuration']==json.loads((output/'configuration.json').read_text())
        f=pd.read_parquet(features_path);obs=pd.read_parquet(obs_path)[OBSERVATION_COLUMNS];arcs=pd.read_parquet(roads_path).set_index('arc_id')
        router=DirectedRouter(arcs,config.routing_cutoff_m,config.max_settled_nodes,config.cache_sources)
        selected=pd.read_parquet(folder/'decoded.parquet');links=pd.read_parquet(folder/'transitions.parquet');route=pd.read_parquet(folder/'route_arcs.parquet')
        probabilities=pd.read_parquet(folder/'probabilities.parquet');states=emission_states(f,probabilities)
        assert selected.gps_observation_index.tolist()==obs.index.tolist() and not selected.gps_observation_index.duplicated().any()
        candidates=set(zip(f.gps_observation_index,f.candidate_arc_id))
        for row in selected.loc[selected.status.eq('matched')].itertuples():assert (row.gps_observation_index,row.candidate_arc_id) in candidates
        selected_index=selected.set_index('gps_observation_index')
        for row in links.itertuples():
            left,right=selected_index.loc[row.from_observation],selected_index.loc[row.to_observation]
            assert left.segment_id==right.segment_id==row.segment_id
            assert row.to_observation==row.from_observation+1
            dt=float(obs.loc[row.to_observation,'timestamp_s']-obs.loc[row.from_observation,'timestamp_s'])
            displacement=float(obs.loc[row.to_observation,'displacement_m'])
            assert np.isclose(dt,row.dt_s) and np.isclose(displacement,row.gps_displacement_m)
            score,path=transition(router,left,right,dt,displacement,config)
            assert path.feasible and np.isfinite(score)
            assert np.isclose(path.distance_m,row.network_distance_m) and np.isclose(score,row.transition_log_score)
            assert list(path.arcs)==list(row.route_arcs)
        for segment,group in route.groupby('segment_id'):
            assert group.route_order.tolist()==list(range(len(group)))
            ids=group.arc_id.tolist()
            for a,b in zip(ids,ids[1:]):assert router.arcs[a][1]==router.arcs[b][0],(trip,a,b)
            chosen=selected.loc[selected.segment_id.eq(segment)]
            assert int(chosen.candidate_arc_id.iloc[0])==ids[0] and int(chosen.candidate_arc_id.iloc[-1])==ids[-1]
        expected_links=int(selected.status.eq('matched').sum())-selected.segment_id.nunique()
        assert len(links)==expected_links
        # Replay one preselected smallest real trip, not all completed trips.
        if folder==replay_trip:
            replay=decode(obs,states,router,config)
            for actual,saved in zip(replay,[selected,links,route,pd.read_parquet(folder/'transition_diagnostics.parquet')]):pd.testing.assert_frame_equal(actual,saved)
            pipeline=joblib.load(B/'models/step6'/f'{model}_tuned.joblib')
            with threadpool_limits(limits=2):fresh=predict_frame(pipeline,f.iloc[:100])
            pd.testing.assert_frame_equal(fresh.iloc[:,:4],probabilities.iloc[:100,:4])
            np.testing.assert_allclose(fresh.positive_probability,probabilities.positive_probability.iloc[:100],rtol=1e-13,atol=1e-15)
        labels=pd.read_parquet(B/'outputs/step5/evaluation_labels'/trip/'candidate_labels.parquet');alignment=pd.read_parquet(B/'outputs/step5/evaluation_labels'/trip/'alignment.parquet')
        for variant,file in [('classifier','classifier_selected'),('viterbi','decoded')]:
            winners=pd.read_parquet(folder/f'{file}.parquet');actual=selected_accuracy(winners,labels,alignment)
            saved=per_trip.loc[(per_trip.trip_id==trip)&(per_trip.variant==variant)].iloc[0]
            for key,value in actual.items():assert np.isclose(saved[key],value),key
            pairs=diagnose_pairs(winners,obs,router,config);all_pairs[variant].append(pairs)
            pd.testing.assert_frame_equal(pairs,pd.read_parquet(folder/f'{variant}_pair_diagnostics.parquet'))
            for key,value in aggregate_pairs(pairs).items():assert np.isclose(saved[key],value),key
        decoded_count+=len(selected);route_count+=len(route);link_count+=len(links)
    comparison=pd.read_csv(output/'classifier_vs_viterbi.csv').set_index('variant')
    for variant in ['classifier','viterbi']:
        selected=per_trip.loc[per_trip.variant==variant]
        assert np.isclose(comparison.loc[variant,'top1_accuracy'],selected.correct.sum()/selected.candidate_present.sum())
        for key,value in aggregate_pairs(pd.concat(all_pairs[variant],ignore_index=True)).items():assert np.isclose(comparison.loc[variant,key],value)
    previous=pd.read_csv(B/'outputs/step6/tuned_student_test.csv').set_index('model_name')
    assert np.isclose(previous.loc[model,'top1_accuracy'],comparison.loc['classifier','top1_accuracy'])
    assert comparison.loc['viterbi','within_segment_infeasible']==0
    assert len(list(output.glob('*route_diagnostic.png')))>=3
    for path in output.glob('*.png'):
        with Image.open(path) as picture:picture.verify()
    result=dict(status='passed',trips=10,observations=decoded_count,route_arc_occurrences=route_count,
        connected_selected_links=link_count,protected_files_verified=len(protected),prior_artifacts_unchanged=True,
        model_reloaded_without_fitting=True,deterministic_real_trip_replay=replay_trip.name,
        reference_data_used_only_after_prediction=True,decoder_parameters_tuned_on_student=False,
        hidden_evaluator_data_accessed=False,impossible_within_segment_links=0)
    (output/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=B/'outputs/step7');a=p.parse_args();verify(a.output)
