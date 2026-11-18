"""Evaluation-only supervision and scoring; never fits a model or changes candidates."""
import numpy as np
import pandas as pd
from .features import FEATURES, IDENTIFIERS
from .modeling import positive_probability, metrics
from .alignment import align_training

def predict_frame(pipeline, features):
    if list(features.columns)!=IDENTIFIERS+FEATURES:
        raise ValueError('Inference input must contain only identifiers and the frozen inference features')
    result=features[IDENTIFIERS].copy()
    result['positive_probability']=positive_probability(pipeline,features[FEATURES])
    return result

def evaluation_alignment(network,cleaned,route,configuration):
    """Identical Step 3 algorithm/settings, used only to construct evaluation references."""
    required={'alignment_radius_m':75,'ambiguity_cost_margin':2,'heading_penalty_m':20,
              'backward_progress_tolerance_m':5,'gap_s':5,'speed_kmh':180,'jump_m':500}
    if any(configuration.get(k)!=v for k,v in required.items()):
        raise ValueError('Frozen Step 3 alignment configuration differs; do not silently change evaluation labels')
    return align_training(network,cleaned,route,radius_m=configuration['alignment_radius_m'],ambiguity_margin=configuration['ambiguity_cost_margin'])

def qualify_evaluation(candidates,alignment):
    if candidates.duplicated(IDENTIFIERS).any():raise ValueError('Duplicate candidate identity')
    a=alignment.loc[candidates.gps_observation_index]
    positive=candidates.candidate_arc_id.to_numpy()==a.aligned_arc_id.to_numpy(dtype=float,na_value=np.nan)
    present=pd.Series(positive,index=candidates.gps_observation_index).groupby(level=0).any()
    eligible=a.supervised_eligible.to_numpy()&candidates.gps_observation_index.map(present).to_numpy()
    labels=candidates[IDENTIFIERS].copy();labels['evaluation_eligible']=eligible
    labels['evaluation_target']=pd.array(np.where(eligible,positive.astype(int),np.nan),dtype='Int8')
    observations=alignment.copy()
    keys=set(zip(candidates.gps_observation_index,candidates.candidate_arc_id))
    observations['reference_candidate_present']=[bool(row.supervised_eligible) and (int(i),int(row.aligned_arc_id)) in keys
                                               for i,row in alignment.iterrows()]
    return labels,observations

def top1(predictions):
    keys=['trip_id','gps_observation_index']
    return predictions.sort_values(keys+['positive_probability','candidate_arc_id'],ascending=[True,True,False,True]).drop_duplicates(keys)

def score_evaluation(predictions,labels,observation_labels,raw_observations):
    merged=predictions.merge(labels,on=IDENTIFIERS,how='outer',validate='one_to_one',indicator=True)
    if not merged['_merge'].eq('both').all():raise ValueError('Prediction/evaluation identity mismatch')
    eligible=merged.loc[merged.evaluation_eligible].rename(columns={'evaluation_target':'target'})
    result=metrics(eligible.assign(target=eligible.target.astype('int8'))) if len(eligible) else {
        k:None for k in ['precision','recall','f1','macro_f1','accuracy','roc_auc','log_loss','tn','fp','fn','tp','top1_accuracy','mean_reciprocal_rank']}
    valid=int(observation_labels.supervised_eligible.sum());present=int(observation_labels.reference_candidate_present.sum())
    result.update(raw_observations=int(raw_observations),qualified_alignments=valid,candidate_present_alignments=present,
        candidate_misses=valid-present,candidate_miss_rate=(valid-present)/valid if valid else None,
        classification_rows=len(eligible),positive_evaluation_rows=int(eligible.target.eq(1).sum()),
        alignment_coverage=valid/raw_observations if raw_observations else None,
        prediction_coverage=predictions.groupby(['trip_id','gps_observation_index']).ngroups/raw_observations if raw_observations else None,
        top1_including_candidate_misses=result['top1_accuracy']*present/valid if valid and present else (0. if valid else None))
    return result
