import numpy as np
import pandas as pd
import pytest
import joblib
from threadpoolctl import threadpool_limits
from pathsnap.modeling import (FEATURES,MODELS,IDENTIFIERS,FiniteFeatures,make_pipeline,positive_probability,
    validate_folds,observation_weights,ranking,metrics,load_training)

def fixture():
    rng=np.random.default_rng(7);X=pd.DataFrame(rng.normal(size=(60,len(FEATURES))),columns=FEATURES)
    X.loc[:5,'gps_heading_sin']=np.nan;X.loc[0,'speed_kmh']=np.inf
    y=np.tile([0,1],30);return X,y

@pytest.mark.parametrize('name',MODELS)
def test_all_five_pipelines_missing_values_positive_probability_and_reload(name,tmp_path):
    X,y=fixture();p=make_pipeline(name)
    with threadpool_limits(limits=2):p.fit(X,y,classifier__sample_weight=np.ones(len(y)))
    prob=positive_probability(p,X);assert prob.shape==(60,) and np.isfinite(prob).all()
    path=tmp_path/'pipeline.joblib';joblib.dump(p,path);loaded=joblib.load(path)
    np.testing.assert_allclose(prob,positive_probability(loaded,X),rtol=1e-13,atol=1e-15)
    assert list(loaded.named_steps['finite'].feature_names_in_)==FEATURES

@pytest.mark.parametrize('extra',['target','aligned_arc_id','trip_id','reference_component'])
def test_reference_labels_and_identifiers_rejected(extra):
    X,_=fixture();X[extra]=1
    with pytest.raises(ValueError,match='features'):FiniteFeatures().fit(X)

def test_feature_order_required():
    X,_=fixture()
    with pytest.raises(ValueError):FiniteFeatures().fit(X[FEATURES[::-1]])

def test_fold_local_imputer_and_scaler_do_not_see_validation():
    X,y=fixture();X.loc[:39,'distance_to_arc_m']=2;X.loc[40:,'distance_to_arc_m']=10000
    p=make_pipeline('logistic_regression')
    with threadpool_limits(limits=2):p.fit(X.iloc[:40],y[:40])
    assert p.named_steps['imputer'].statistics_[0]==2
    assert p.named_steps['scale'].mean_[0]==2
    before=p.named_steps['imputer'].statistics_.copy();positive_probability(p,X.iloc[40:])
    np.testing.assert_array_equal(before,p.named_steps['imputer'].statistics_)

def test_trip_folds_no_observation_leakage():
    f=pd.DataFrame({'trip_id':[f'{i:08}' for i in range(5)],'validation_fold':range(5)})
    rows=pd.DataFrame({'trip_id':np.repeat(f.trip_id,4),'gps_observation_index':np.tile([0,0,1,1],5)})
    mapped=validate_folds(rows,f)
    assert mapped.groupby(rows.trip_id).nunique().eq(1).all()
    with pytest.raises(ValueError):validate_folds(rows,f.iloc[:-1])

def test_student_test_targets_rejected(tmp_path):
    d=tmp_path/'train/00000001';d.mkdir(parents=True)
    pd.DataFrame({'trip_id':['00000001'],'validation_fold':[0]}).to_csv(tmp_path/'trip_folds.csv',index=False)
    X,_=fixture();f=X.iloc[:1].copy()
    for col,value in [('dataset_partition','student_test'),('trip_id','00000001'),('gps_observation_index',0),('candidate_arc_id',0)]:f[col]=value
    f[IDENTIFIERS+FEATURES].to_parquet(d/'features.parquet',index=False)
    labels=f[IDENTIFIERS].assign(target=1,training_eligible=True);labels.to_parquet(d/'labels.parquet',index=False)
    with pytest.raises(ValueError,match='leakage'):load_training(tmp_path)

def test_ranking_metrics_tie_break_and_candidate_accuracy_distinction():
    p=pd.DataFrame({'trip_id':['a']*4,'gps_observation_index':[0,0,1,1],'candidate_arc_id':[1,2,3,4],
       'target':[1,0,0,1],'positive_probability':[.9,.2,.5,.5]})
    m=metrics(p)
    assert m['top1_accuracy']==.5 and m['mean_reciprocal_rank']==.75
    assert m['accuracy']==.75 and m['prediction_coverage']==1
    assert m['tp']==2 and m['fp']==1 and m['tn']==1 and m['fn']==0
    assert m['log_loss']>0 and 0<=m['roc_auc']<=1

def test_single_class_auc_undefined_and_invalid_ranking_rejected():
    p=pd.DataFrame({'trip_id':['a'],'gps_observation_index':[0],'candidate_arc_id':[1],'target':[1],'positive_probability':[.5]})
    assert metrics(p)['roc_auc'] is None
    p.target=0
    with pytest.raises(ValueError):ranking(p)

def test_equal_total_weight_per_observation():
    rows=pd.DataFrame({'trip_id':['a']*5,'gps_observation_index':[0,0,1,1,1],'target':[1,0,1,0,0]})
    weights=observation_weights(rows)
    assert np.mean(weights)==pytest.approx(1)
    assert weights[:2].sum()==pytest.approx(weights[2:].sum())

def test_reproducible_configs_and_hgb_no_random_internal_split():
    for name in MODELS:assert make_pipeline(name).named_steps['classifier'].get_params()==make_pipeline(name).named_steps['classifier'].get_params()
    assert not make_pipeline('hist_gradient_boosting').named_steps['classifier'].early_stopping

def test_resume_reuses_completed_folds_and_final_pipelines_without_fitting(tmp_path,monkeypatch):
    import importlib.util,json,hashlib
    from pathlib import Path
    from sklearn.pipeline import Pipeline
    script=Path(__file__).resolve().parents[1]/'scripts/train_models.py'
    spec=importlib.util.spec_from_file_location('step4_resume_test',script);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    root=tmp_path/'data';out=tmp_path/'out';models=tmp_path/'models'
    for d in [root,out,models]:d.mkdir()
    rng=np.random.default_rng(42);data=pd.DataFrame(rng.normal(size=(160,len(FEATURES))),columns=FEATURES)
    data['dataset_partition']='train';data['trip_id']=np.repeat([f'{i:08}' for i in range(80)],2)
    data['gps_observation_index']=0;data['candidate_arc_id']=np.tile([0,1],80);data['target']=np.tile([0,1],80)
    folds=pd.DataFrame({'trip_id':[f'{i:08}' for i in range(80)],'validation_fold':np.arange(80)%5})
    folds.to_csv(root/'trip_folds.csv',index=False);data['fold_id']=validate_folds(data,folds)
    configurations={};manifest=[];cached=[]
    for name in MODELS:
        pipeline=make_pipeline(name)
        with threadpool_limits(limits=2):pipeline.fit(data[FEATURES],data.target)
        configurations[name]=pipeline.named_steps['classifier'].get_params()
        path=models/f'{name}.joblib';joblib.dump(pipeline,path)
        manifest.append(dict(model_name=name,full_fit_seconds=1.,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        directory=out/'oof'/name;directory.mkdir(parents=True)
        for fold in range(5):
            pred=data.loc[data.fold_id==fold,IDENTIFIERS+['target','fold_id']].copy()
            pred['model_name']=name;pred['positive_probability']=.2+.6*pred.target
            pred.to_parquet(directory/f'fold_{fold}.parquet',index=False);cached.append(dict(model_name=name,fold_id=fold))
    (models/'model_configurations.json').write_text(json.dumps(configurations));pd.DataFrame(manifest).to_csv(models/'artifact_manifest.csv',index=False)
    pd.DataFrame(cached).to_csv(out/'fold_metrics.csv',index=False);(out/'preservation.json').write_text('{}')
    monkeypatch.setattr(runner,'load_training',lambda _: (data,folds,{'eligible_rows':160}))
    def no_fit(*args,**kwargs):raise AssertionError('Completed valid pipelines must not be retrained')
    monkeypatch.setattr(Pipeline,'fit',no_fit)
    before={p.name:p.read_bytes() for p in models.glob('*.joblib')}
    runner.run(root,out,models,resume=True)
    assert before=={p.name:p.read_bytes() for p in models.glob('*.joblib')}
    assert len(pd.read_csv(out/'fold_metrics.csv'))==25
