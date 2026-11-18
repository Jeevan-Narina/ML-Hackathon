"""Predict first, build evaluation-only references second, compare without fitting."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse
os.environ.setdefault('OMP_NUM_THREADS','2')
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from pathsnap.modeling import MODELS,FEATURES,IDENTIFIERS
from pathsnap.evaluation import predict_frame,evaluation_alignment,qualify_evaluation,score_evaluation,top1
from pathsnap.continuity import DirectedReachability,continuity_diagnostics,summarise_continuity
from pathsnap.loaders import load_nodes,load_arcs,load_route
from pathsnap.road_network import RoadNetwork

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run(output):
    start=time.perf_counter();output.mkdir(parents=True,exist_ok=True)
    root=B/'data/processed/step3';source=B/'data/processed/step2';models=B/'models/step4'
    trips=sorted((root/'student_test').iterdir());assert len(trips)==10
    manifest=pd.read_csv(models/'artifact_manifest.csv').set_index('model_name')
    assert set(manifest.index)==set(MODELS)
    assert json.loads((B/'outputs/step4/verification.json').read_text())['status']=='passed'
    assert len(list((B/'outputs/step4/oof').glob('*/*.parquet')))==25
    protected={p.relative_to(B).as_posix():digest(p) for p in [*models.glob('*'),*root.glob('student_test/*/*.parquet'),*source.glob('student_test/*/*.parquet'),*list((B/'outputs/step4/oof').glob('*/*.parquet')),B/'outputs/step4/model_comparison.csv'] if p.is_file()}
    identities={};timings=[]
    # Phase 1 deliberately has no reference-route input. All five models score first.
    for name in MODELS:
        path=models/f'{name}.joblib';assert digest(path)==manifest.loc[name,'sha256']
        tick=time.perf_counter();pipeline=joblib.load(path);load_seconds=time.perf_counter()-tick
        for trip in trips:
            feature_path=trip/'features.parquet';f=pd.read_parquet(feature_path)
            dest=output/'predictions'/name;dest.mkdir(parents=True,exist_ok=True)
            pred_path=dest/f'{trip.name}.parquet';stamp=dest/f'{trip.name}.json'
            identity=dict(model_sha256=digest(path),features_sha256=digest(feature_path),features=FEATURES)
            if pred_path.exists() and stamp.exists() and json.loads(stamp.read_text())['identity']==identity:
                info=json.loads(stamp.read_text());p=pd.read_parquet(pred_path)
                pd.testing.assert_frame_equal(p[IDENTIFIERS],f[IDENTIFIERS]);assert np.isfinite(p.positive_probability).all()
            else:
                tick=time.perf_counter()
                with threadpool_limits(limits=2):p=predict_frame(pipeline,f)
                seconds=time.perf_counter()-tick;p.to_parquet(pred_path,index=False)
                info=dict(identity=identity,prediction_seconds=seconds,rows=len(f),model_load_seconds=load_seconds)
                stamp.write_text(json.dumps(info,indent=2))
            obs=pd.read_parquet(trip/'observations.parquet')
            timings.append(dict(model_name=name,trip_id=trip.name,candidate_rows=len(f),observations=len(obs),
                prediction_seconds=info['prediction_seconds'],milliseconds_per_observation=info['prediction_seconds']/len(obs)*1000,model_load_seconds=load_seconds))
        identities[name]=digest(path)
        print('Predicted without labels:',name,flush=True)
    pd.DataFrame(timings).to_csv(output/'inference_times.csv',index=False)
    config=json.loads((root/'preprocessing.json').read_text());alignment_rows=[]
    assert digest(B/'src/pathsnap/alignment.py')==config['source_sha256']['alignment.py'],'Frozen Step 3 alignment code changed'
    for trip in trips:
        raw=B/'data/raw/student_test'/trip.name;dest=output/'evaluation_labels'/trip.name;dest.mkdir(parents=True,exist_ok=True)
        identity=dict(route_sha256=digest(raw/f'{trip.name}.route'),observations_sha256=digest(trip/'observations.parquet'),
                      nodes_sha256=digest(raw/f'{trip.name}.nodes'),arcs_sha256=digest(raw/f'{trip.name}.arcs'),
                      candidates_sha256=digest(source/'student_test'/trip.name/'candidates.parquet'),alignment_code_sha256=digest(B/'src/pathsnap/alignment.py'),configuration=config)
        if not (dest/'identity.json').exists() or json.loads((dest/'identity.json').read_text())!=identity:
            net=RoadNetwork(load_nodes(raw/f'{trip.name}.nodes'),load_arcs(raw/f'{trip.name}.arcs'))
            obs=pd.read_parquet(trip/'observations.parquet');route=load_route(raw/f'{trip.name}.route')
            a=evaluation_alignment(net,obs,route,config)
            candidates=pd.read_parquet(source/'student_test'/trip.name/'candidates.parquet')
            labels,a=qualify_evaluation(candidates,a)
            a.to_parquet(dest/'alignment.parquet');labels.to_parquet(dest/'candidate_labels.parquet',index=False)
            (dest/'identity.json').write_text(json.dumps(identity,indent=2))
        a=pd.read_parquet(dest/'alignment.parquet')
        alignment_rows.append(dict(trip_id=trip.name,observations=len(a),qualified_alignments=int(a.supervised_eligible.sum()),
           candidate_present=int(a.reference_candidate_present.sum()),candidate_misses=int((a.supervised_eligible&~a.reference_candidate_present).sum()),
           ambiguous=int(a.alignment_status.eq('ambiguous').sum()),unmatched_alignment=int(a.alignment_status.eq('unmatched').sum()),excluded=int(a.alignment_status.eq('excluded').sum())))
        print('Evaluation-only alignment:',trip.name,flush=True)
    pd.DataFrame(alignment_rows).to_csv(output/'candidate_miss_report.csv',index=False)
    # Phase 3 joins separate labels only for scoring. Prediction files never gain targets.
    summaries=[];per_trip=[]
    for name in MODELS:
        all_pred=[];all_labels=[];all_obs=[]
        for trip in trips:
            pred=pd.read_parquet(output/'predictions'/name/f'{trip.name}.parquet')
            labels=pd.read_parquet(output/'evaluation_labels'/trip.name/'candidate_labels.parquet')
            obs=pd.read_parquet(output/'evaluation_labels'/trip.name/'alignment.parquet')
            per_trip.append(dict(model_name=name,trip_id=trip.name,**score_evaluation(pred,labels,obs,len(obs))))
            all_pred.append(pred);all_labels.append(labels);all_obs.append(obs)
        totals=score_evaluation(pd.concat(all_pred),pd.concat(all_labels),pd.concat(all_obs),sum(len(a) for a in all_obs))
        times=pd.DataFrame(timings);t=times.loc[times.model_name==name]
        summaries.append(dict(model_name=name,**totals,inference_seconds=float(t.prediction_seconds.sum()),
          inference_ms_per_observation=float(t.prediction_seconds.sum()/t.observations.sum()*1000),
          mean_trip_inference_seconds=float(t.prediction_seconds.mean()),artifact_bytes=(models/f'{name}.joblib').stat().st_size))
    student=pd.DataFrame(summaries);student.to_csv(output/'student_test_comparison.csv',index=False)
    pd.DataFrame(per_trip).to_csv(output/'per_trip_metrics.csv',index=False)
    cv=pd.read_csv(B/'outputs/step4/model_comparison.csv');student.merge(cv,on='model_name',validate='one_to_one').to_csv(output/'cv_vs_student.csv',index=False)
    # Build small winner tables once from existing OOF predictions; no CV refits.
    cv_winners={name:top1(pd.concat([pd.read_parquet(p) for p in sorted((B/'outputs/step4/oof'/name).glob('*.parquet'))])) for name in MODELS}
    continuity=[]
    for partition in ['student_test','train']:
        for trip in sorted((root/partition).iterdir()):
            if not trip.is_dir():continue
            raw=B/'data/raw'/partition/trip.name;nodes=load_nodes(raw/f'{trip.name}.nodes');arcs=load_arcs(raw/f'{trip.name}.arcs')
            reach=DirectedReachability(nodes,arcs);observations=pd.read_parquet(trip/'observations.parquet')
            candidates=pd.read_parquet(source/partition/trip.name/'candidates.parquet',columns=['gps_observation_index','candidate_arc_id','position_along_arc'])
            for name in MODELS:
                winners=top1(pd.read_parquet(output/'predictions'/name/f'{trip.name}.parquet')) if partition=='student_test' else cv_winners[name].loc[cv_winners[name].trip_id==trip.name]
                winners=winners.merge(candidates,on=['gps_observation_index','candidate_arc_id'],validate='one_to_one')
                transitions=continuity_diagnostics(winners,observations,arcs,reach)
                continuity.append(dict(dataset_partition=partition,model_name=name,trip_id=trip.name,**summarise_continuity(transitions)))
                dest=output/'continuity_pairs'/partition/name;dest.mkdir(parents=True,exist_ok=True)
                transitions.to_parquet(dest/f'{trip.name}.parquet',index=False)
            print('Connectivity diagnostics:',partition,trip.name,flush=True)
    pd.DataFrame(continuity).to_csv(output/'continuity_by_trip.csv',index=False)
    for path,value in protected.items():assert digest(B/path)==value,'Source/model changed during evaluation'
    (output/'run_metadata.json').write_text(json.dumps(dict(runtime_seconds=time.perf_counter()-start,protected_sha256=protected,
       model_sha256=identities,alignment_configuration=config,phases=['feature-only predictions','evaluation-only alignment','scoring and topology diagnostics'],
       refitted_models=False,student_trips=10,models=MODELS,cv_winner_scope='Only Step 4 eligible candidate-present observations; missing neighbours excluded from continuity',
       source_code_sha256={p.name:digest(p) for p in (B/'src/pathsnap').glob('*.py')}),indent=2))
    print('Step 5 evaluation finished',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=B/'outputs/step5');args=parser.parse_args();run(args.output)
