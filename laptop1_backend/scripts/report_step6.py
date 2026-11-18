"""Post-selection student check and actual optimisation/CV plots; never fits."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse
os.environ.setdefault('OMP_NUM_THREADS','2')
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
import numpy as np
import pandas as pd
import joblib,optuna
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathsnap.optimization import TUNED_MODELS
from pathsnap.evaluation import predict_frame,score_evaluation

def report(output,models):
    selection=json.loads((models/'classifier_selection.json').read_text());selection_bytes=(models/'classifier_selection.json').read_bytes()
    comparison=pd.read_csv(output/'baseline_vs_tuned.csv');all_student=[]
    # Official guidelines require a final student comparison. Selection is already
    # frozen from training-only CV and is never changed by this reporting function.
    for name in TUNED_MODELS:
        pipeline=joblib.load(models/f'{name}_tuned.joblib');predictions=[];labels=[];observations=[];elapsed=0.
        for trip in sorted((B/'data/processed/step3/student_test').iterdir()):
            f=pd.read_parquet(trip/'features.parquet');folder=output/'student_predictions'/name;folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'{trip.name}.parquet';stamp=path.with_suffix('.json')
            identity={'model_sha256':hashlib.sha256((models/f'{name}_tuned.joblib').read_bytes()).hexdigest(),'features_sha256':hashlib.sha256((trip/'features.parquet').read_bytes()).hexdigest()}
            if path.exists() and stamp.exists() and json.loads(stamp.read_text())['identity']==identity:
                pred=pd.read_parquet(path);seconds=json.loads(stamp.read_text())['seconds']
            else:
                tick=time.perf_counter()
                with threadpool_limits(limits=2):pred=predict_frame(pipeline,f)
                seconds=time.perf_counter()-tick;pred.to_parquet(path,index=False);stamp.write_text(json.dumps(dict(identity=identity,seconds=seconds)))
            elapsed+=seconds;predictions.append(pred)
            labels.append(pd.read_parquet(B/'outputs/step5/evaluation_labels'/trip.name/'candidate_labels.parquet'))
            observations.append(pd.read_parquet(B/'outputs/step5/evaluation_labels'/trip.name/'alignment.parquet'))
        scored=score_evaluation(pd.concat(predictions),pd.concat(labels),pd.concat(observations),sum(len(o) for o in observations))
        all_student.append(dict(model_name=name,**scored,inference_seconds=elapsed))
    student=pd.DataFrame(all_student);student.to_csv(output/'tuned_student_test.csv',index=False)
    student.merge(pd.read_csv(B/'outputs/step5/student_test_comparison.csv'),on='model_name',suffixes=('_tuned','_baseline'),validate='one_to_one').to_csv(output/'baseline_vs_tuned_student.csv',index=False)
    assert (models/'classifier_selection.json').read_bytes()==selection_bytes,'Student report must not change selection'
    for metric,title in [('top1_accuracy','Eligible observation Top-1'),('macro_f1','Candidate Macro F1'),('log_loss','Candidate binary log loss')]:
        fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(comparison))
        for offset,variant in [(-.2,'baseline'),(.2,'tuned')]:
            ax.bar(x+offset,comparison[f'{metric}_mean_{variant}'],.4,yerr=comparison[f'{metric}_std_{variant}'],capsize=4,label=variant)
        ax.set_xticks(x,comparison.model_name);ax.set_title(title+' — model-selection CV mean ± SD');ax.legend();fig.tight_layout();fig.savefig(output/f'{metric}_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4));importance_status={}
    for ax,name in zip(axes,TUNED_MODELS):
        path=output/'studies'/f'{name}.sqlite'
        study=optuna.load_study(study_name=name,storage='sqlite:///'+path.resolve().as_posix())
        complete=[t for t in study.trials if t.state.name=='COMPLETE'];numbers=[t.number for t in complete];values=[t.value for t in complete]
        ax.plot(numbers,values,'o-',label='Trial search-fold Top-1');ax.plot(numbers,np.maximum.accumulate(values),'--',label='Best so far')
        ax.set(title=name,xlabel='Actual trial number',ylabel='Training search-fold Top-1');ax.legend()
        if len(complete)>=10 and len(set(values))>1:
            try:
                values=optuna.importance.get_param_importances(study,evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42))
                pd.DataFrame(values.items(),columns=['parameter','importance']).to_csv(output/f'{name}_parameter_importance.csv',index=False)
                impfig,impax=plt.subplots(figsize=(8,4));impax.barh(list(values),list(values.values()));impax.set_title(name+' — exploratory fANOVA importance');impfig.tight_layout();impfig.savefig(output/f'{name}_parameter_importance.png');plt.close(impfig)
                importance_status[name]='Generated exploratory fANOVA importance; small-sample caveats apply'
            except (ValueError,RuntimeError) as exc:importance_status[name]=f'Not available: {exc}'
        else:importance_status[name]=f'Not generated: {len(complete)} successful trials; require >=10 with varying objective for a minimally interpretable estimate'
    fig.tight_layout();fig.savefig(output/'optimisation_history.png',dpi=150);plt.close(fig)
    (output/'parameter_importance_status.json').write_text(json.dumps(importance_status,indent=2))
    print(comparison[['model_name','top1_accuracy_mean_baseline','top1_accuracy_mean_tuned','top1_improvement_percentage_points','macro_f1_mean_tuned','log_loss_mean_tuned']].to_string(index=False))
    print('Frozen training-CV classifier selection:',selection)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=B/'outputs/step6');p.add_argument('--models',type=Path,default=B/'models/step6');a=p.parse_args();report(a.output,a.models)
