"""Train five baseline pipelines with genuine trip-held-out validation."""
from pathlib import Path
import sys,os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,json,time,hashlib,importlib.metadata,warnings
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from pathsnap.modeling import MODELS,FEATURES,IDENTIFIERS,load_training,make_pipeline,observation_weights,positive_probability,metrics
from pathsnap.step2 import MemoryMonitor
B=Path(__file__).resolve().parents[1]

def run(root,output,artifacts,resume=False):
    started=time.perf_counter();output.mkdir(parents=True,exist_ok=True);artifacts.mkdir(parents=True,exist_ok=True)
    data,folds,counts=load_training(root);print(json.dumps(counts),flush=True)
    (output/'dataset_validation.json').write_text(json.dumps(counts,indent=2))
    X=data[FEATURES]; fold_rows=[];trip_rows=[];fit_rows=[];split_rows=[];config={}
    config_path=artifacts/'model_configurations.json' if (artifacts/'model_configurations.json').exists() else output/'cv_configurations.json'
    old_configs=json.loads(config_path.read_text()) if resume else {}
    existing_manifest=pd.read_csv(artifacts/'artifact_manifest.csv').set_index('model_name') if resume and (artifacts/'artifact_manifest.csv').exists() else pd.DataFrame()
    cached=pd.read_csv(output/'fold_metrics.csv') if resume else pd.DataFrame()
    if resume:
        for relative,value in json.loads((output/'preservation.json').read_text()).items():
            assert hashlib.sha256((B/relative).read_bytes()).hexdigest()==value,'Resume inputs changed'
    for fold in range(5):
        train=data.fold_id!=fold;valid=~train
        assert set(data.loc[train,'trip_id']).isdisjoint(data.loc[valid,'trip_id'])
        split_rows.append(dict(fold_id=fold,training_trips=data.loc[train,'trip_id'].nunique(),validation_trips=data.loc[valid,'trip_id'].nunique(),
           training_rows=int(train.sum()),validation_rows=int(valid.sum()),training_observations=data.loc[train].groupby(['trip_id','gps_observation_index']).ngroups,
           validation_observations=data.loc[valid].groupby(['trip_id','gps_observation_index']).ngroups))
    pd.DataFrame(split_rows).to_csv(output/'fold_integrity.csv',index=False)
    for name in MODELS:
        config[name]=make_pipeline(name).named_steps['classifier'].get_params()
        model_dir=output/'oof'/name;model_dir.mkdir(parents=True,exist_ok=True)
        for fold in range(5):
            train=data.fold_id!=fold;valid=~train;pipeline=make_pipeline(name)
            prior=cached.loc[(cached.model_name==name)&(cached.fold_id==fold)] if len(cached) else pd.DataFrame()
            if len(prior):
                assert old_configs[name]==config[name],f'Refusing changed configuration for cached {name}'
                pred=pd.read_parquet(model_dir/f'fold_{fold}.parquet')
                pd.testing.assert_frame_equal(pred[IDENTIFIERS+['target','fold_id']].reset_index(drop=True),data.loc[valid,IDENTIFIERS+['target','fold_id']].reset_index(drop=True))
                fold_rows.extend(prior.to_dict('records'))
                for trip,g in pred.groupby('trip_id'):trip_rows.append(dict(model_name=name,trip_id=trip,fold_id=fold,**metrics(g)))
                print(f'{name}: verified/reused completed held-out fold {fold}',flush=True)
                continue
            print(f'{name}: fold {fold}, 64 train / 16 validation trips',flush=True)
            tick=time.perf_counter()
            with MemoryMonitor() as mem,threadpool_limits(limits=2),warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                pipeline.fit(X.loc[train],data.loc[train,'target'],classifier__sample_weight=observation_weights(data.loc[train]))
                seconds=time.perf_counter()-tick;t=time.perf_counter();prob=positive_probability(pipeline,X.loc[valid]);infer=time.perf_counter()-t
            pred=data.loc[valid,IDENTIFIERS+['target','fold_id']].copy();pred['model_name']=name;pred['positive_probability']=prob
            pred.to_parquet(model_dir/f'fold_{fold}.parquet',index=False)
            row=dict(model_name=name,fold_id=fold,training_seconds=seconds,prediction_seconds=infer,peak_rss_mib=mem.peak/2**20,
                     training_rows=int(train.sum()),validation_rows=int(valid.sum()),warnings=' | '.join(str(w.message) for w in caught),**metrics(pred))
            fold_rows.append(row);pd.DataFrame(fold_rows).to_csv(output/'fold_metrics.csv',index=False)
            for trip,g in pred.groupby('trip_id'):trip_rows.append(dict(model_name=name,trip_id=trip,fold_id=fold,**metrics(g)))
            print(f'  {seconds:.2f}s; Top-1 {row["top1_accuracy"]:.5f}; F1 {row["f1"]:.5f}',flush=True)
        final=make_pipeline(name);tick=time.perf_counter();path=artifacts/f'{name}.joblib'
        reuse_final=False
        if resume and name in existing_manifest.index and path.exists() and old_configs.get(name)==config[name]:
            try:
                assert hashlib.sha256(path.read_bytes()).hexdigest()==existing_manifest.loc[name,'sha256']
                loaded=joblib.load(path)
                assert loaded.named_steps['classifier'].get_params()==config[name]
                positive_probability(loaded,X.iloc[:100])
                final=loaded;reuse_final=True
            except (AssertionError,ValueError,OSError):pass
        if reuse_final:
            value=existing_manifest.loc[name,'full_fit_seconds'];fit_seconds=float(value) if pd.notna(value) else None
            print(f'{name}: verified/reused final full-training pipeline',flush=True)
        else:
            with threadpool_limits(limits=2):final.fit(X,data.target,classifier__sample_weight=observation_weights(data))
            fit_seconds=time.perf_counter()-tick;joblib.dump(final,path,compress=3)
        reloaded=joblib.load(path);np.testing.assert_allclose(positive_probability(final,X.iloc[:100]),positive_probability(reloaded,X.iloc[:100]),rtol=1e-13,atol=1e-15)
        fit_rows.append(dict(model_name=name,full_fit_seconds=fit_seconds,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        names=final.named_steps['imputer'].get_feature_names_out(FEATURES);clf=final.named_steps['classifier']
        values=clf.coef_[0] if hasattr(clf,'coef_') else clf.feature_importances_ if hasattr(clf,'feature_importances_') else None
        if values is not None:pd.DataFrame({'feature':names,'value':values}).to_csv(output/f'{name}_feature_effects.csv',index=False)
    pd.DataFrame(trip_rows).to_csv(output/'per_trip_metrics.csv',index=False);pd.DataFrame(fit_rows).to_csv(artifacts/'artifact_manifest.csv',index=False)
    (artifacts/'model_configurations.json').write_text(json.dumps(config,indent=2))
    (output/'cv_configurations.json').write_text(json.dumps(config,indent=2))
    (artifacts/'feature_schema.json').write_text(json.dumps({'ordered_features':FEATURES,'positive_class':1,'excluded_metadata':IDENTIFIERS},indent=2))
    metadata=dict(counts=counts,total_seconds=time.perf_counter()-started,resumed=resume,seed=42,threads=2,
      versions={n:importlib.metadata.version(n) for n in ['scikit-learn','joblib','numpy','pandas','scipy','threadpoolctl']},
      preprocessing='Finite-to-NaN, median imputation with missing indicators and retained all-missing columns; scaling LR/GNB; all learned inside each fit',
      weighting='Inverse candidate count within each eligible observation, normalised to mean 1 on fitting rows only; identical across all models',
      early_stopping='HGB early_stopping=False avoids an internal random candidate-row validation split',
      scope='OOF metrics only from held-out training trips; full fits separate; student-test data never opened',
      inputs={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('train/*/*.parquet')},
      folds_sha256=hashlib.sha256((root/'trip_folds.csv').read_bytes()).hexdigest())
    for directory in [output,artifacts]:(directory/'training_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('ALL FIVE MODELS COMPLETE',metadata['total_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=B/'data/processed/step3');p.add_argument('--output',type=Path,default=B/'outputs/step4');p.add_argument('--models',type=Path,default=B/'models/step4');p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.data,a.output,a.models,a.resume)
