"""Interpret saved model evaluations and generate measured Step 5 reports."""
from pathlib import Path
import os,sys,json,time,hashlib
os.environ.setdefault('OMP_NUM_THREADS','2')
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
from pathsnap.modeling import MODELS,FEATURES,IDENTIFIERS

def report(output):
    start=time.perf_counter();student=pd.read_csv(output/'student_test_comparison.csv')
    comparison=pd.read_csv(output/'cv_vs_student.csv');continuity=pd.read_csv(output/'continuity_by_trip.csv',dtype={'trip_id':str})
    count_cols=['adjacent_observation_pairs','assessable_pairs','same_arc','directly_connected','intermediate_required','unreachable','temporal_gap','invalid_timestamp','missing_prediction','same_arc_backward_fraction']
    total=continuity.groupby(['dataset_partition','model_name'])[count_cols].sum().reset_index()
    total['direct_continuity']=(total.same_arc+total.directly_connected)/total.assessable_pairs.replace(0,np.nan)
    total['topological_reachability']=(total.same_arc+total.directly_connected+total.intermediate_required)/total.assessable_pairs.replace(0,np.nan)
    total.to_csv(output/'continuity_summary.csv',index=False)
    c=total.loc[total.dataset_partition=='student_test',['model_name','direct_continuity','topological_reachability','unreachable','assessable_pairs']]
    comparison=comparison.drop(columns=[name for name in c.columns if name!='model_name'],errors='ignore').merge(c,on='model_name',validate='one_to_one')
    comparison['cv_top1_minus_student']=comparison.top1_accuracy_mean-comparison.top1_accuracy
    comparison['cv_macro_f1_minus_student']=comparison.macro_f1_mean-comparison.macro_f1
    comparison.to_csv(output/'cv_vs_student.csv',index=False)
    for field,title,file in [('top1_accuracy','Conditional Top-1 on candidate-present qualified observations','top1_comparison.png'),('macro_f1','Candidate Macro F1 on eligible reference-present groups','macro_f1_comparison.png'),('log_loss','Candidate binary log loss on eligible reference-present groups','log_loss_comparison.png')]:
        fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(comparison))
        ax.bar(x-.2,comparison[field+'_mean'],.4,yerr=comparison[field+'_std'],capsize=3,label='CV: equal-fold mean ± SD')
        ax.bar(x+.2,comparison[field],.4,label='Student test: pooled rows/groups')
        ax.set_xticks(x,comparison.model_name,rotation=20);ax.set_title(title);ax.legend();fig.tight_layout();fig.savefig(output/file,dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(18,4))
    for ax,row in zip(axes,student.itertuples()):
        cm=np.array([[row.tn,row.fp],[row.fn,row.tp]],dtype=int);ax.imshow(cm,cmap='Blues');ax.set(title=row.model_name,xlabel='Predicted class',ylabel='Reference-derived class');ax.set_xticks([0,1]);ax.set_yticks([0,1])
        for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center')
    fig.tight_layout();fig.savefig(output/'confusion_matrices.png',dpi=150);plt.close(fig)
    # Complete observation groups, deterministic equal spacing; no refitting/feature selection.
    frames=[]
    for trip in sorted((output/'evaluation_labels').iterdir()):
        f=pd.read_parquet(B/'data/processed/step3/student_test'/trip.name/'features.parquet')
        labels=pd.read_parquet(trip/'candidate_labels.parquet')
        frames.append(f.merge(labels,on=IDENTIFIERS,validate='one_to_one').query('evaluation_eligible'))
    frame=pd.concat(frames,ignore_index=True)
    groups=frame[['trip_id','gps_observation_index']].drop_duplicates().sort_values(['trip_id','gps_observation_index'])
    groups=groups.iloc[np.unique(np.linspace(0,len(groups)-1,min(1000,len(groups)),dtype=int))]
    sample=frame.merge(groups,on=['trip_id','gps_observation_index'],validate='many_to_one')
    identities=sample[IDENTIFIERS];identities.to_parquet(output/'interpretability_sample_keys.parquet',index=False)
    importance=[]
    for name in MODELS:
        path=output/f'{name}_permutation_importance.csv'
        stamp=path.with_suffix('.json')
        identity=dict(model_sha256=hashlib.sha256((B/'models/step4'/f'{name}.joblib').read_bytes()).hexdigest(),
                      sample_sha256=hashlib.sha256(pd.util.hash_pandas_object(sample[IDENTIFIERS+FEATURES+['evaluation_target']],index=False).values.tobytes()).hexdigest(),repeats=2,seed=42,scoring='f1_macro')
        if path.exists() and stamp.exists() and json.loads(stamp.read_text())==identity:result=pd.read_csv(path)
        else:
            pipeline=joblib.load(B/'models/step4'/f'{name}.joblib');tick=time.perf_counter()
            with threadpool_limits(limits=2):
                measured=permutation_importance(pipeline,sample[FEATURES],sample.evaluation_target.astype('int8'),scoring='f1_macro',n_repeats=2,random_state=42,n_jobs=1)
            result=pd.DataFrame({'model_name':name,'feature':FEATURES,'importance_mean':measured.importances_mean,'importance_std':measured.importances_std,'seconds':time.perf_counter()-tick,'sample_candidate_rows':len(sample),'sample_observations':len(groups)})
            result.to_csv(path,index=False)
            stamp.write_text(json.dumps(identity,indent=2))
        importance.append(result)
        print('Read-only feature analysis:',name,flush=True)
    importance=pd.concat(importance,ignore_index=True);importance.to_csv(output/'feature_analysis.csv',index=False)
    existing=[]
    for path in (B/'outputs/step4').glob('*_feature_effects.csv'):
        existing.append(pd.read_csv(path).assign(model_name=path.name.replace('_feature_effects.csv',''),method='scaled_coefficient' if path.name.startswith('logistic') else 'impurity_importance'))
    pd.concat(existing).to_csv(output/'reused_feature_effects.csv',index=False)
    fig,axes=plt.subplots(1,5,figsize=(22,7))
    for ax,name in zip(axes,MODELS):
        subset=importance.loc[importance.model_name==name].nlargest(8,'importance_mean').sort_values('importance_mean')
        ax.barh(subset.feature,subset.importance_mean,xerr=subset.importance_std);ax.set_title(name);ax.set_xlabel('Decrease in sample Macro F1')
    fig.tight_layout();fig.savefig(output/'permutation_importance.png',dpi=150);plt.close(fig)
    ordered=comparison.sort_values(['top1_accuracy','macro_f1','log_loss'],ascending=[False,False,True])
    choice=dict(promising_model=ordered.iloc[0].model_name,backup_model=ordered.iloc[1].model_name,
                criterion='Student-test conditional observation Top-1 primary; Macro F1 and log loss break exact ties; CV stability, timing and topology reported as trade-offs',
                top1_gap_percentage_points=float((ordered.iloc[0].top1_accuracy-ordered.iloc[1].top1_accuracy)*100),
                selection_status='Promising untuned model for Step 6, not a final tuned model',report_seconds=time.perf_counter()-start)
    (output/'preliminary_selection.json').write_text(json.dumps(choice,indent=2))
    print(ordered[['model_name','macro_f1','top1_accuracy','top1_including_candidate_misses','log_loss','inference_ms_per_observation','direct_continuity']].to_string(index=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=B/'outputs/step5');args=p.parse_args();report(args.output)
