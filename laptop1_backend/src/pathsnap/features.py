"""Causal, label-independent candidate features and separate supervision targets."""
import numpy as np
import pandas as pd

IDENTIFIERS=['dataset_partition','trip_id','gps_observation_index','candidate_arc_id']
FEATURES=['distance_to_arc_m','arc_length_m','arc_bearing_sin','arc_bearing_cos','gps_heading_sin','gps_heading_cos',
 'heading_difference_deg','speed_kmh','dt_previous_s','displacement_m','candidate_rank','position_along_arc',
 'distance_from_node_m','distance_to_node_m','from_out_degree','to_in_degree','invalid_coordinates','duplicate_record',
 'bad_timestamp','large_gap','speed_outlier','gps_jump','heading_undefined','fallback_used','candidate_count']

def engineer_features(candidates,cleaned,network):
    f=candidates[IDENTIFIERS+['distance_to_arc_m','arc_length_m','candidate_rank','position_along_arc','fallback_used']].copy()
    obs=cleaned.loc[candidates.gps_observation_index]
    for col in ['speed_kmh','dt_previous_s','displacement_m','invalid_coordinates','duplicate_record','bad_timestamp','large_gap','speed_outlier','gps_jump']:
        f[col]=obs[col].to_numpy()
    heading=obs.gps_heading_deg.to_numpy(); bearing=candidates.arc_bearing_deg.to_numpy()
    f['arc_bearing_sin']=np.sin(np.radians(bearing)); f['arc_bearing_cos']=np.cos(np.radians(bearing))
    f['gps_heading_sin']=np.sin(np.radians(heading)); f['gps_heading_cos']=np.cos(np.radians(heading))
    f['heading_difference_deg']=abs((bearing-heading+180)%360-180); f['heading_undefined']=~np.isfinite(heading)
    # Distances along the straight metric segment to its endpoints.
    lengths=network.arcs.loc[candidates.candidate_arc_id].projected_length_m.to_numpy()
    f['distance_from_node_m']=lengths*candidates.position_along_arc.to_numpy()
    f['distance_to_node_m']=lengths*(1-candidates.position_along_arc.to_numpy())
    valid=network.arcs.loc[network.arcs.geometry_status.isin(['valid','degenerate_segment'])]
    f['from_out_degree']=candidates.from_node_id.map(valid.from_node_id.value_counts()).fillna(0).to_numpy(dtype='int64')
    f['to_in_degree']=candidates.to_node_id.map(valid.to_node_id.value_counts()).fillna(0).to_numpy(dtype='int64')
    f['candidate_count']=candidates.groupby('gps_observation_index').candidate_arc_id.transform('size')
    return f[IDENTIFIERS+FEATURES]

def supervision(candidates,alignment,partition):
    if partition!='train': raise ValueError('Supervised labels are training-only')
    a=alignment.loc[candidates.gps_observation_index]
    positive=candidates.candidate_arc_id.to_numpy()==a.aligned_arc_id.to_numpy(dtype=float,na_value=np.nan)
    present=pd.Series(positive,index=candidates.gps_observation_index).groupby(level=0).any()
    eligible=a.supervised_eligible.to_numpy()&candidates.gps_observation_index.map(present).to_numpy()
    out=candidates[IDENTIFIERS].copy()
    out['training_eligible']=eligible
    out['target']=pd.array(np.where(eligible,positive.astype(int),np.nan),dtype='Int8')
    valid=alignment.loc[alignment.supervised_eligible]
    candidate_ids=set(zip(candidates.gps_observation_index,candidates.candidate_arc_id))
    recall=valid.copy(); recall['candidate_present']=[(i,int(row.aligned_arc_id)) in candidate_ids for i,row in valid.iterrows()]
    lookup={(int(r.gps_observation_index),int(r.candidate_arc_id)):int(r.candidate_rank) for r in candidates.itertuples()}
    recall['reference_candidate_rank']=[lookup.get((i,int(r.aligned_arc_id))) for i,r in valid.iterrows()]
    return out,recall

def trip_splits(ids,folds=5):
    return pd.DataFrame({'trip_id':sorted(ids),'validation_fold':[i%folds for i in range(len(ids))]})
