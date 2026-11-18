"""Step 4: training-only datasets, fold-local pipelines and OOF ranking metrics."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, log_loss, confusion_matrix
from .features import FEATURES, IDENTIFIERS

MODELS=['logistic_regression','gaussian_nb','decision_tree','random_forest','hist_gradient_boosting']

class FiniteFeatures(TransformerMixin,BaseEstimator):
    """Require exactly the inference feature schema; map infinities to missing."""
    def fit(self,X,y=None):
        self.transform(X);self.feature_names_in_=np.array(FEATURES,dtype=object);self.n_features_in_=len(FEATURES)
        return self
    def transform(self,X):
        if not isinstance(X,pd.DataFrame) or list(X.columns)!=FEATURES:
            raise ValueError('Input must be a DataFrame with exactly the ordered inference features; labels/IDs forbidden')
        return X.astype('float64').replace([np.inf,-np.inf],np.nan)
    def get_feature_names_out(self,input_features=None):return np.array(FEATURES,dtype=object)

def make_pipeline(name):
    estimators={
      'logistic_regression':lambda:LogisticRegression(C=1.,solver='lbfgs',max_iter=400,random_state=42),
      'gaussian_nb':lambda:GaussianNB(var_smoothing=1e-8),
      'decision_tree':lambda:DecisionTreeClassifier(max_depth=14,min_samples_leaf=20,random_state=42),
      'random_forest':lambda:RandomForestClassifier(n_estimators=24,max_depth=10,min_samples_leaf=40,max_features='sqrt',max_samples=.5,n_jobs=2,random_state=42),
      'hist_gradient_boosting':lambda:HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,max_bins=63,learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=42)}
    if name not in estimators:raise ValueError(f'Unknown model: {name}')
    steps=[('finite',FiniteFeatures()),('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True))]
    if name in ['logistic_regression','gaussian_nb']:steps.append(('scale',StandardScaler()))
    return Pipeline(steps+[('classifier',estimators[name]())])

def validate_folds(rows,folds):
    if folds.trip_id.duplicated().any() or set(folds.validation_fold)!={0,1,2,3,4}:raise ValueError('Invalid five-fold trip metadata')
    if set(rows.trip_id)!=set(folds.trip_id):raise ValueError('Trip coverage differs from fold metadata')
    return rows.trip_id.map(folds.set_index('trip_id').validation_fold).astype('int8')

def load_training(root):
    root=Path(root);frames=[];all_rows=excluded=0
    folds=pd.read_csv(root/'trip_folds.csv',dtype={'trip_id':str})
    for directory in sorted((root/'train').iterdir()):
        if not directory.is_dir():continue
        f=pd.read_parquet(directory/'features.parquet');labels=pd.read_parquet(directory/'labels.parquet')
        if list(f.columns)!=IDENTIFIERS+FEATURES:raise ValueError('Unexpected feature columns/order')
        if not f.dataset_partition.eq('train').all() or not labels.dataset_partition.eq('train').all():raise ValueError('Student-test leakage')
        if not f.trip_id.eq(directory.name).all():raise ValueError('Trip identity mismatch')
        merged=f.merge(labels,on=IDENTIFIERS,validate='one_to_one',how='outer',indicator=True)
        if not merged['_merge'].eq('both').all():raise ValueError('Feature/label keys do not match exactly')
        eligible=merged.training_eligible & merged.target.notna()
        if (merged.training_eligible!=merged.target.notna()).any():raise ValueError('Eligibility/null-target disagreement')
        all_rows+=len(merged);excluded+=int((~eligible).sum());merged=merged.loc[eligible,IDENTIFIERS+FEATURES+['target']]
        if not merged.target.isin([0,1]).all():raise ValueError('Target must encode positive=1, negative=0')
        groups=merged.groupby(['trip_id','gps_observation_index']).target
        if not groups.sum().eq(1).all():raise ValueError('Every eligible observation must have exactly one positive')
        frames.append(merged)
    data=pd.concat(frames,ignore_index=True);data['target']=data.target.astype('int8')
    data['fold_id']=validate_folds(data,folds)
    if len(folds)!=80 or not folds.validation_fold.value_counts().eq(16).all():raise ValueError('Expected 80 trips, 16 per held-out fold')
    return data,folds,dict(all_training_candidates=all_rows,excluded_targets=excluded,eligible_rows=len(data),
       positive=int(data.target.sum()),negative=int((data.target==0).sum()),eligible_observations=data.groupby(['trip_id','gps_observation_index']).ngroups)

def observation_weights(rows):
    sizes=rows.groupby(['trip_id','gps_observation_index']).target.transform('size').to_numpy()
    w=1/sizes;return w/w.mean()

def positive_probability(pipeline,X):
    classes=list(pipeline.named_steps['classifier'].classes_)
    if 1 not in classes:raise ValueError('Fitted model lacks positive class 1')
    p=pipeline.predict_proba(X)[:,classes.index(1)]
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid probabilities')
    return p

def ranking(predictions):
    ranked=predictions.sort_values(['trip_id','gps_observation_index','positive_probability','candidate_arc_id'],ascending=[True,True,False,True]).copy()
    group=ranked.groupby(['trip_id','gps_observation_index'],sort=False)
    ranked['predicted_rank']=group.cumcount()+1
    positives=ranked.loc[ranked.target==1]
    if not group.target.sum().eq(1).all():raise ValueError('Ranking requires one supervised positive per observation')
    return dict(top1_accuracy=float(positives.predicted_rank.eq(1).mean()),mean_reciprocal_rank=float((1/positives.predicted_rank).mean()),
                prediction_coverage=float(ranked.assign(_finite=np.isfinite(ranked.positive_probability)).groupby(['trip_id','gps_observation_index'])._finite.all().mean()),observations=len(positives))

def metrics(predictions):
    y=predictions.target.to_numpy();p=predictions.positive_probability.to_numpy();hat=(p>=.5).astype(int)
    cm=confusion_matrix(y,hat,labels=[0,1])
    return dict(precision=precision_score(y,hat,zero_division=0),recall=recall_score(y,hat,zero_division=0),
      f1=f1_score(y,hat,zero_division=0),macro_f1=f1_score(y,hat,average='macro',zero_division=0),accuracy=accuracy_score(y,hat),
      roc_auc=roc_auc_score(y,p) if len(np.unique(y))==2 else None,log_loss=log_loss(y,p,labels=[0,1]),
      tn=int(cm[0,0]),fp=int(cm[0,1]),fn=int(cm[1,0]),tp=int(cm[1,1]),**ranking(predictions))
