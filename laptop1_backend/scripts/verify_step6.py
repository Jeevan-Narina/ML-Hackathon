"""Verify persisted studies, genuine selected-configuration CV and model artifacts."""
from pathlib import Path
import sys,json,hashlib
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
import numpy as np
import pandas as pd
import joblib,optuna
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from PIL import Image
from pathsnap.modeling import FEATURES,IDENTIFIERS,load_training,positive_probability,metrics
from pathsnap.optimization import TUNED_MODELS,check_training_partition

def verify(root,output,models):
    for relative,value in json.loads((output/'protected_artifacts.json').read_text()).items():
        assert hashlib.sha256((B/relative).read_bytes()).hexdigest()==value,relative
    data,folds,counts=load_training(root);check_training_partition(data)
    meta=json.loads((output/'run_metadata.json').read_text());assert counts==meta['counts'] and not meta['student_data_used_in_objective']
    expected=data.set_index(IDENTIFIERS)[['target','fold_id']].sort_index()
    rows=pd.read_csv(output/'tuned_fold_metrics.csv');assert len(rows)==10 and not rows.duplicated(['model_name','fold_id']).any()
    manifest=pd.read_csv(models/'artifact_manifest.csv').set_index('model_name');best=json.loads((models/'best_parameters.json').read_text());states={}
    original_fit=Pipeline.fit
    Pipeline.fit=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Verification must never refit'))
    try:
        for name in TUNED_MODELS:
            study=optuna.load_study(study_name=name,storage='sqlite:///'+(output/'studies'/f'{name}.sqlite').resolve().as_posix())
            assert study.best_params==best[name]
            complete=[t for t in study.trials if t.state.name=='COMPLETE'];states[name]=len(complete)
            for trial in complete:
                measured=[]
                for fold in meta['configuration']['search_folds']:
                    pred=pd.read_parquet(output/'trials'/name/f'trial_{trial.number}'/f'fold_{fold}.parquet')
                    pd.testing.assert_frame_equal(pred[IDENTIFIERS+['target','fold_id']].reset_index(drop=True),data.loc[data.fold_id==fold,IDENTIFIERS+['target','fold_id']].reset_index(drop=True))
                    measured.append(metrics(pred)['top1_accuracy'])
                assert np.isclose(np.mean(measured),trial.value)
            all_pred=[]
            for fold in range(5):
                pred=pd.read_parquet(output/'cv'/name/f'fold_{fold}.parquet');all_pred.append(pred)
                assert pred.fold_id.eq(fold).all() and pred.model_name.eq(name).all()
                assert set(pred.trip_id)==set(folds.loc[folds.validation_fold==fold,'trip_id'])
                actual=metrics(pred);saved=rows.loc[(rows.model_name==name)&(rows.fold_id==fold)].iloc[0]
                assert saved.training_trips==64 and saved.validation_trips==16
                for key,value in actual.items():
                    assert pd.isna(saved[key]) if value is None else np.isclose(saved[key],value),key
            pred=pd.concat(all_pred);assert len(pred)==len(data) and not pred.duplicated(IDENTIFIERS).any()
            pd.testing.assert_frame_equal(pred.set_index(IDENTIFIERS)[['target','fold_id']].sort_index(),expected)
            path=models/manifest.loc[name,'path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest.loc[name,'sha256']
            pipeline=joblib.load(path)
            for key,value in best[name].items():assert pipeline.named_steps['classifier'].get_params()[key]==value
            assert list(pipeline.named_steps['finite'].feature_names_in_)==FEATURES
            with threadpool_limits(limits=2):prob=positive_probability(pipeline,data[FEATURES].iloc[:1000])
            assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    finally:Pipeline.fit=original_fit
    comparison=pd.read_csv(output/'baseline_vs_tuned.csv');baseline=pd.read_csv(B/'outputs/step4/model_comparison.csv').set_index('model_name')
    for row in comparison.itertuples():
        actual=rows.loc[rows.model_name==row.model_name]
        for key in ['top1_accuracy','macro_f1','log_loss']:
            assert np.isclose(getattr(row,key+'_mean_tuned'),actual[key].mean())
            assert np.isclose(getattr(row,key+'_mean_baseline'),baseline.loc[row.model_name,key+'_mean'])
    selection=json.loads((models/'classifier_selection.json').read_text());ranked=comparison.sort_values(['top1_accuracy_mean_tuned','log_loss_mean_tuned','top1_accuracy_std_tuned'],ascending=[False,True,True])
    assert selection['primary']==ranked.iloc[0].model_name and not selection['student_test_used_for_selection']
    for path in output.glob('*.png'):
        with Image.open(path) as picture:picture.verify()
    result=dict(status='passed',completed_trials=states,selected_cv_folds=10,oof_rows_per_model=len(data),
                baseline_and_prior_artifacts_unchanged=True,models_reloaded_without_fitting=2,
                primary=selection['primary'],backup=selection['backup'],selection_uses_student_labels=False)
    (output/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=B/'data/processed/step3');p.add_argument('--output',type=Path,default=B/'outputs/step6');p.add_argument('--models',type=Path,default=B/'models/step6');a=p.parse_args();verify(a.data,a.output,a.models)
