"""Summarise existing out-of-fold results; does not train or evaluate student labels."""
from pathlib import Path
import sys,json,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]

def evaluate(output,models):
    folds=pd.read_csv(output/'fold_metrics.csv')
    if len(folds)!=25 or folds.groupby('model_name').fold_id.nunique().ne(5).any():raise ValueError('All 25 held-out fits are required')
    fields=['precision','recall','f1','macro_f1','accuracy','roc_auc','log_loss','top1_accuracy','mean_reciprocal_rank','prediction_coverage','training_seconds','prediction_seconds']
    comparison=folds.groupby('model_name')[fields].agg(['mean','std']);comparison.columns=['_'.join(c) for c in comparison.columns]
    comparison=comparison.reset_index().sort_values(['top1_accuracy_mean','mean_reciprocal_rank_mean','log_loss_mean'],ascending=[False,False,True])
    comparison.to_csv(output/'model_comparison.csv',index=False)
    names=comparison.model_name.tolist()
    for field,label,file in [('f1','Candidate positive-class F1','f1_comparison.png'),('top1_accuracy','Eligible observation Top-1 accuracy','top1_comparison.png'),('training_seconds','Training duration (seconds)','training_time.png')]:
        fig,ax=plt.subplots(figsize=(10,4));ax.bar(names,comparison[field+'_mean'],yerr=comparison[field+'_std'],capsize=4)
        ax.set_title(label+' — trip-held-out CV mean ± sample SD');ax.tick_params(axis='x',labelrotation=20)
        fig.tight_layout();fig.savefig(output/file,dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(18,4))
    for ax,name in zip(axes,names):
        g=folds.loc[folds.model_name==name];cm=np.array([[g.tn.sum(),g.fp.sum()],[g.fn.sum(),g.tp.sum()]])
        ax.imshow(cm,cmap='Blues');ax.set(title=name,xlabel='Predicted',ylabel='Actual');ax.set_xticks([0,1]);ax.set_yticks([0,1])
        for (i,j),value in np.ndenumerate(cm):ax.text(j,i,str(value),ha='center',va='center')
    fig.tight_layout();fig.savefig(output/'confusion_matrices.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(18,4))
    for ax,name in zip(axes,names):
        p=pd.concat([pd.read_parquet(f,columns=['target','positive_probability']) for f in sorted((output/'oof'/name).glob('*.parquet'))])
        for target in [0,1]:ax.hist(p.loc[p.target==target,'positive_probability'],bins=np.linspace(0,1,41),alpha=.5,label=f'target {target}')
        ax.set(title=name,xlabel='OOF positive probability',ylabel='Rows');ax.legend()
    fig.tight_layout();fig.savefig(output/'probability_distributions.png',dpi=150);plt.close(fig)
    effects=list(output.glob('*_feature_effects.csv'))
    fig,axes=plt.subplots(1,len(effects),figsize=(18,6));axes=np.atleast_1d(axes)
    for ax,path in zip(axes,effects):
        f=pd.read_csv(path);f=f.iloc[np.argsort(abs(f.value.to_numpy()))[-12:]];ax.barh(f.feature,f.value);ax.set_title(path.stem.replace('_feature_effects',''))
    fig.tight_layout();fig.savefig(output/'feature_effects.png',dpi=150);plt.close(fig)
    best=names[0];(output/'preliminary_selection.json').write_text(json.dumps({'model_name':best,'criterion':'Highest equal-fold mean eligible-observation Top-1; MRR then log loss break ties','status':'Preliminary CV baseline selection; no tuning, student-test evaluation or route decoder'},indent=2))
    print(comparison[['model_name','f1_mean','top1_accuracy_mean','log_loss_mean','training_seconds_mean']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=B/'outputs/step4');p.add_argument('--models',type=Path,default=B/'models/step4');a=p.parse_args();evaluate(a.output,a.models)
