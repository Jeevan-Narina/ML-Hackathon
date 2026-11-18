"""Training-only seeded TPE studies and unchanged fold-local classifier pipelines."""
from pathlib import Path
import time,json
import numpy as np
import optuna
from optuna.trial import TrialState
from .modeling import make_pipeline

TUNED_MODELS=('random_forest','hist_gradient_boosting')
SEARCH_SPACES={
 'random_forest':dict(n_estimators=[12,24,36],max_depth=[8,10,12],min_samples_split=[2,10,20],
                     min_samples_leaf=[20,40,80],max_features=['sqrt',.5],max_samples=[.4,.6]),
 'hist_gradient_boosting':dict(learning_rate={'low':.04,'high':.2,'log':True},max_iter=[50,75,100],max_leaf_nodes=[7,15,23],
                             max_depth=[4,6,8,None],min_samples_leaf=[20,40,80],
                             l2_regularization={'low':.01,'high':10.,'log':True},max_bins=[31,63])}

def suggest_parameters(trial,model):
    if model not in TUNED_MODELS:raise ValueError('Only RF and HGB may be optimised in Step 6')
    return {key:trial.suggest_float(key,**space) if isinstance(space,dict) else trial.suggest_categorical(key,space)
            for key,space in SEARCH_SPACES[model].items()}

def tuned_pipeline(model,parameters):
    if model not in TUNED_MODELS or set(parameters)!=set(SEARCH_SPACES[model]):raise ValueError('Invalid tuned parameter schema')
    return make_pipeline(model).set_params(**{f'classifier__{k}':v for k,v in parameters.items()})

def check_training_partition(data):
    if not data.dataset_partition.eq('train').all():raise ValueError('Student/evaluator data forbidden in optimisation')
    if not data.target.isin([0,1]).all():raise ValueError('Only eligible binary targets may enter tuning')
    if set(data.fold_id)!={0,1,2,3,4}:raise ValueError('Expected the frozen five saved trip folds')

def run_study(model,storage_path,identity,objective,trials=6,timeout_seconds=180,seed=42):
    """Total attempted-trial budget; deterministic trial-local sampler seeds.

    SQLite persists every ask/parameter/tell. A completed trial is never called
    again. Abandoned RUNNING trials become FAIL on serial-process recovery;
    failed attempts consume the total budget. No concurrent study workers.
    """
    if model not in TUNED_MODELS:raise ValueError('Unsupported tuning model')
    if not isinstance(trials,int) or not 1<=trials<=20:raise ValueError('Trial budget must be an integer in [1,20]')
    if not np.isfinite(timeout_seconds) or timeout_seconds<=0:raise ValueError('Time budget must be finite and positive')
    storage_path=Path(storage_path);storage_path.parent.mkdir(parents=True,exist_ok=True)
    url='sqlite:///'+storage_path.resolve().as_posix()
    study=optuna.create_study(study_name=model,storage=url,direction='maximize',load_if_exists=True,
                              sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=2))
    canonical=json.loads(json.dumps(identity,sort_keys=True))
    if 'identity' in study.user_attrs and study.user_attrs['identity']!=canonical:
        raise ValueError('Study input/configuration identity changed; use a separate study directory')
    study.set_user_attr('identity',canonical)
    for abandoned in study.get_trials(states=(TrialState.RUNNING,)):
        study.tell(abandoned.number,state=TrialState.FAIL)
    started=time.perf_counter()
    while len(study.trials)<trials and time.perf_counter()-started<timeout_seconds:
        number=len(study.trials)
        # Recreating a seeded sampler for each trial makes normal serial resumes
        # reproduce proposals from the same persisted trial history, without RNG pickles.
        study.sampler=optuna.samplers.TPESampler(seed=seed+number,n_startup_trials=2)
        trial=study.ask();tick=time.perf_counter()
        try:
            parameters=suggest_parameters(trial,model)
            value=float(objective(trial,parameters))
            if not np.isfinite(value):raise ValueError('Nonfinite optimisation objective')
            trial.set_user_attr('elapsed_seconds',time.perf_counter()-tick)
            study.tell(trial,value)
        except Exception as exc:
            trial.set_user_attr('error',str(exc));study.tell(trial,state=TrialState.FAIL)
            raise
        print(f'{model} trial {trial.number}: Top-1={value:.6f}, {time.perf_counter()-tick:.2f} s',flush=True)
        study.trials_dataframe().to_csv(storage_path.parent/f'{model}_trials.csv',index=False)
    if not study.get_trials(states=(TrialState.COMPLETE,)):raise RuntimeError('No successful trials; cannot select parameters')
    study.trials_dataframe().to_csv(storage_path.parent/f'{model}_trials.csv',index=False)
    return study
