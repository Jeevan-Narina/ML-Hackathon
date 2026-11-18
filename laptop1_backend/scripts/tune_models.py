"""Resumable, bounded Bayesian optimisation of RF/HGB using training trips only."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse,importlib.metadata
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from pathsnap.optimization import TUNED_MODELS,SEARCH_SPACES,run_study,tuned_pipeline,check_training_partition
from pathsnap.modeling import load_training,FEATURES,IDENTIFIERS,observation_weights,positive_probability,metrics
from pathsnap.step2 import MemoryMonitor

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint(frame):return hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest()
def predict_fold(data,X,model,parameters,fold,threads):
    train=data.fold_id!=fold;validation=~train
    assert set(data.loc[train,'trip_id']).isdisjoint(data.loc[validation,'trip_id'])
    pipeline=tuned_pipeline(model,parameters)
    if model=='random_forest':pipeline.set_params(classifier__n_jobs=threads)
    tick=time.perf_counter()
    with MemoryMonitor() as mem,threadpool_limits(limits=threads):
        pipeline.fit(X.loc[train],data.loc[train,'target'],classifier__sample_weight=observation_weights(data.loc[train]))
        fit_seconds=time.perf_counter()-tick;tick=time.perf_counter();prob=positive_probability(pipeline,X.loc[validation]);inference=time.perf_counter()-tick
    predictions=data.loc[validation,IDENTIFIERS+['target','fold_id']].copy();predictions['model_name']=model;predictions['positive_probability']=prob
    row=dict(model_name=model,fold_id=fold,training_rows=int(train.sum()),validation_rows=int(validation.sum()),training_trips=data.loc[train,'trip_id'].nunique(),
       validation_trips=data.loc[validation,'trip_id'].nunique(),training_seconds=fit_seconds,prediction_seconds=inference,peak_rss_mib=mem.peak/2**20,**metrics(predictions))
    return predictions,row

def run(root,output,models,config):
    started=time.perf_counter();output.mkdir(parents=True,exist_ok=True);models.mkdir(parents=True,exist_ok=True)
    search=config['search_folds']
    if not search or len(search)!=len(set(search)) or not set(search)<=set(range(5)):raise ValueError('Invalid search-fold subset')
    if config['threads'] not in [1,2]:raise ValueError('Use 1 or 2 threads on this CPU')
    data,folds,counts=load_training(root);check_training_partition(data);X=data[FEATURES]
    identity=dict(data_sha256=fingerprint(data[IDENTIFIERS+FEATURES+['target','fold_id']]),features=FEATURES,search_folds=search,seed=config['seed'],threads=config['threads'],
      spaces=SEARCH_SPACES,modeling_code=digest(B/'src/pathsnap/modeling.py'),optimization_code=digest(B/'src/pathsnap/optimization.py'),
      sklearn=importlib.metadata.version('scikit-learn'),optuna=importlib.metadata.version('optuna'))
    identity_path=output/'input_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('Step 6 input identity changed; use a new output directory')
    identity_path.write_text(json.dumps(identity,indent=2))
    protection=output/'protected_artifacts.json'
    if not protection.exists():
        files=[*list((B/'models/step4').glob('*')),*list((B/'outputs/step4').glob('*.csv')),*list((B/'outputs/step4/oof').glob('*/*.parquet')),
               *list((B/'outputs/step5').glob('*.csv')),*list(root.glob('train/*/*.parquet')),root/'trip_folds.csv']
        protection.write_text(json.dumps({p.relative_to(B).as_posix():digest(p) for p in files if p.is_file()},indent=2))
    for relative,value in json.loads(protection.read_text()).items():assert digest(B/relative)==value,'Prior artifact changed'
    all_cv=[];best_configs={};artifact_rows=[];study_rows=[]
    for name in TUNED_MODELS:
        def objective(trial,parameters):
            values=[];rows=[];folder=output/'trials'/name/f'trial_{trial.number}';folder.mkdir(parents=True,exist_ok=True)
            for fold in search:
                prediction,row=predict_fold(data,X,name,parameters,fold,config['threads'])
                prediction.to_parquet(folder/f'fold_{fold}.parquet',index=False);rows.append(row);values.append(row['top1_accuracy'])
                pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
                trial.set_user_attr('fold_metrics',rows)
            trial.set_user_attr('parameters',parameters)
            return np.mean(values)
        tick=time.perf_counter();study=run_study(name,output/'studies'/f'{name}.sqlite',identity,objective,
            config['trials_per_model'],config['timeout_seconds_per_model'],config['seed'])
        best=study.best_trial;parameters=best.params;best_configs[name]=parameters
        (output/f'{name}_best_parameters.json').write_text(json.dumps(dict(trial_number=best.number,objective=best.value,parameters=parameters),indent=2))
        study_rows.append(dict(model_name=name,completed_trials=sum(t.state.name=='COMPLETE' for t in study.trials),attempted_trials=len(study.trials),best_trial=best.number,best_objective=best.value,study_invocation_seconds=time.perf_counter()-tick,
                               recorded_trial_seconds=sum(t.user_attrs.get('elapsed_seconds',0) for t in study.trials)))
        folder=output/'cv'/name;folder.mkdir(parents=True,exist_ok=True)
        for fold in range(5):
            path=folder/f'fold_{fold}.parquet';stamp=folder/f'fold_{fold}.json';stamp_identity=dict(parameters=parameters,input_identity=identity,fold=fold)
            if path.exists() and stamp.exists() and json.loads(stamp.read_text())['identity']==stamp_identity:
                prediction=pd.read_parquet(path);row=json.loads(stamp.read_text())['metrics']
                pd.testing.assert_frame_equal(prediction[IDENTIFIERS+['target','fold_id']].reset_index(drop=True),data.loc[data.fold_id==fold,IDENTIFIERS+['target','fold_id']].reset_index(drop=True))
            elif fold in search:
                prediction=pd.read_parquet(output/'trials'/name/f'trial_{best.number}'/f'fold_{fold}.parquet')
                row=next(r for r in best.user_attrs['fold_metrics'] if r['fold_id']==fold)
            else:prediction,row=predict_fold(data,X,name,parameters,fold,config['threads'])
            prediction.to_parquet(path,index=False);stamp.write_text(json.dumps(dict(identity=stamp_identity,metrics=row),indent=2))
            all_cv.append(row);pd.DataFrame(all_cv).to_csv(output/'tuned_fold_metrics.csv',index=False)
            print(f'{name} selected configuration fold {fold}: {row["top1_accuracy"]:.6f}',flush=True)
        path=models/f'{name}_tuned.joblib';stamp=models/f'{name}_identity.json';final_identity=dict(parameters=parameters,input_identity=identity)
        if path.exists() and stamp.exists() and json.loads(stamp.read_text())['identity']==final_identity:
            final=joblib.load(path);info=json.loads(stamp.read_text());assert digest(path)==info['sha256']
        else:
            final=tuned_pipeline(name,parameters)
            if name=='random_forest':final.set_params(classifier__n_jobs=config['threads'])
            tick=time.perf_counter()
            with threadpool_limits(limits=config['threads']):final.fit(X,data.target,classifier__sample_weight=observation_weights(data))
            fit_seconds=time.perf_counter()-tick;joblib.dump(final,path,compress=3)
            info=dict(identity=final_identity,training_seconds=fit_seconds,sha256=digest(path));stamp.write_text(json.dumps(info,indent=2))
        with threadpool_limits(limits=config['threads']):np.testing.assert_allclose(positive_probability(final,X.iloc[:100]),positive_probability(joblib.load(path),X.iloc[:100]),rtol=1e-13,atol=1e-15)
        artifact_rows.append(dict(model_name=name,path=path.name,sha256=digest(path),bytes=path.stat().st_size,full_fit_seconds=info['training_seconds']))
    pd.DataFrame(study_rows).to_csv(output/'study_summary.csv',index=False);pd.DataFrame(artifact_rows).to_csv(models/'artifact_manifest.csv',index=False)
    (models/'best_parameters.json').write_text(json.dumps(best_configs,indent=2));(models/'feature_schema.json').write_text(json.dumps(dict(ordered_features=FEATURES,positive_class=1),indent=2))
    pd.DataFrame(all_cv).to_csv(output/'tuned_fold_metrics.csv',index=False)
    fields=['accuracy','precision','recall','f1','macro_f1','roc_auc','log_loss','top1_accuracy','mean_reciprocal_rank','prediction_seconds','training_seconds']
    comparison=pd.DataFrame(all_cv).groupby('model_name')[fields].agg(['mean','std']);comparison.columns=['_'.join(c) for c in comparison.columns];comparison=comparison.reset_index()
    baseline=pd.read_csv(B/'outputs/step4/model_comparison.csv');comparison=comparison.merge(baseline,on='model_name',suffixes=('_tuned','_baseline'),validate='one_to_one')
    comparison['top1_improvement_percentage_points']=(comparison.top1_accuracy_mean_tuned-comparison.top1_accuracy_mean_baseline)*100
    comparison.to_csv(output/'baseline_vs_tuned.csv',index=False)
    ranked=comparison.sort_values(['top1_accuracy_mean_tuned','log_loss_mean_tuned','top1_accuracy_std_tuned'],ascending=[False,True,True])
    selection=dict(primary=ranked.iloc[0].model_name,backup=ranked.iloc[1].model_name,criterion='Training-only five-fold CV Top-1 primary; log loss and fold stability break exact ties',
        independent_test=False,student_test_used_for_selection=False)
    (models/'classifier_selection.json').write_text(json.dumps(selection,indent=2));(output/'classifier_selection.json').write_text(json.dumps(selection,indent=2))
    metadata=dict(configuration=config,counts=counts,runtime_seconds=time.perf_counter()-started,versions={n:importlib.metadata.version(n) for n in ['optuna','scikit-learn','joblib','numpy','pandas','sqlalchemy']},
                  selection_cv_warning='Five-fold selected-configuration scores overlap the search folds and are model-selection evidence, not independent held-out test estimates',
                  threads=config['threads'],student_data_used_in_objective=False,search_space=SEARCH_SPACES)
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2));(models/'training_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Step 6 training/optimisation complete:',selection,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=B/'data/processed/step3');p.add_argument('--output',type=Path,default=B/'outputs/step6');p.add_argument('--models',type=Path,default=B/'models/step6');p.add_argument('--config',type=Path,default=B/'config/optimization.json');p.add_argument('--trials',type=int);p.add_argument('--timeout',type=float);a=p.parse_args()
    cfg=json.loads(a.config.read_text());
    if a.trials is not None:cfg['trials_per_model']=a.trials
    if a.timeout is not None:cfg['timeout_seconds_per_model']=a.timeout
    run(a.data,a.output,a.models,cfg)
