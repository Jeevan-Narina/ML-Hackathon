import json
import numpy as np
import pandas as pd
import pytest
import joblib
import optuna
from threadpoolctl import threadpool_limits
from pathsnap.optimization import (SEARCH_SPACES,suggest_parameters,tuned_pipeline,check_training_partition,run_study)
from pathsnap.modeling import FEATURES,positive_probability,metrics

@pytest.mark.parametrize('model',['random_forest','hist_gradient_boosting'])
def test_valid_deterministic_parameters_pipeline_and_reload(model,tmp_path):
    params={k: (np.sqrt(v['low']*v['high']) if isinstance(v,dict) else v[0]) for k,v in SEARCH_SPACES[model].items()}
    a=suggest_parameters(optuna.trial.FixedTrial(params),model);b=suggest_parameters(optuna.trial.FixedTrial(params),model)
    assert a==b
    rng=np.random.default_rng(42);X=pd.DataFrame(rng.normal(size=(60,len(FEATURES))),columns=FEATURES);X.iloc[0,0]=np.nan
    p=tuned_pipeline(model,a)
    with threadpool_limits(limits=2):p.fit(X,np.tile([0,1],30),classifier__sample_weight=np.ones(60))
    path=tmp_path/'p.joblib';joblib.dump(p,path)
    np.testing.assert_allclose(positive_probability(p,X),positive_probability(joblib.load(path),X),rtol=1e-13,atol=1e-15)
    assert p.named_steps['classifier'].random_state==42
    if model=='hist_gradient_boosting':assert not p.named_steps['classifier'].early_stopping

def test_sqlite_resume_never_repeats_completed_trials_and_matches_serial_run(tmp_path):
    calls=[]
    def objective(trial,params):calls.append(trial.number);return params['n_estimators']/100+params['max_depth']/1000
    identity={'training':'fixture','search_folds':[0]}
    resumed=tmp_path/'resumed.sqlite'
    run_study('random_forest',resumed,identity,objective,trials=3,timeout_seconds=60)
    run_study('random_forest',resumed,identity,objective,trials=6,timeout_seconds=60)
    assert calls==list(range(6))
    before=len(calls);s=run_study('random_forest',resumed,identity,objective,trials=6,timeout_seconds=60)
    assert len(calls)==before and len(s.trials)==6
    uninterrupted=run_study('random_forest',tmp_path/'serial.sqlite',identity,lambda t,p:p['n_estimators']/100+p['max_depth']/1000,trials=6,timeout_seconds=60)
    assert [t.params for t in s.trials]==[t.params for t in uninterrupted.trials]

def test_changed_study_identity_rejected(tmp_path):
    path=tmp_path/'study.sqlite'
    run_study('random_forest',path,{'data':'a'},lambda t,p:.5,trials=1)
    with pytest.raises(ValueError,match='identity changed'):
        run_study('random_forest',path,{'data':'student_test'},lambda t,p:.5,trials=2)

def test_abandoned_running_trial_fails_without_repeating_completed_work(tmp_path):
    path=tmp_path/'abandoned.sqlite';url='sqlite:///'+path.resolve().as_posix()
    s=optuna.create_study(study_name='random_forest',storage=url,direction='maximize');s.ask()
    resumed=run_study('random_forest',path,{},lambda t,p:.5,trials=2)
    assert resumed.trials[0].state.name=='FAIL' and resumed.trials[1].state.name=='COMPLETE'

def test_tuning_rejects_student_partition():
    frame=pd.DataFrame({'dataset_partition':['train']*5,'target':[0,1,0,1,0],'fold_id':range(5)})
    check_training_partition(frame)
    frame.loc[0,'dataset_partition']='student_test'
    with pytest.raises(ValueError,match='forbidden'):check_training_partition(frame)

@pytest.mark.parametrize('budget',[0,21,-1])
def test_invalid_budget_rejected(tmp_path,budget):
    with pytest.raises(ValueError):run_study('random_forest',tmp_path/'x.sqlite',{},lambda t,p:.5,trials=budget)

def test_objective_is_observation_top1_not_candidate_accuracy():
    p=pd.DataFrame({'trip_id':['a']*4,'gps_observation_index':[0,0,1,1],'candidate_arc_id':[0,1,0,1],
                    'target':[1,0,0,1],'positive_probability':[.49,.1,.3,.4]})
    values=metrics(p);assert values['top1_accuracy']==1 and values['accuracy']==.5

def test_reference_parameters_and_other_models_prohibited():
    with pytest.raises(ValueError):tuned_pipeline('logistic_regression',{})
    with pytest.raises(ValueError):tuned_pipeline('random_forest',{'reference_arc_id':1})
