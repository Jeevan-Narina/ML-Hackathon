"""Run Step 3 using existing Step 2 outputs; never read student-test routes."""
from pathlib import Path
import sys,json,time,hashlib
import argparse
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pathsnap.loaders import discover_trips,load_nodes,load_arcs,load_track,load_route
from pathsnap.road_network import RoadNetwork
from pathsnap.preprocessing import clean_observations
from pathsnap.alignment import align_training
from pathsnap.features import engineer_features,supervision,trip_splits,FEATURES

B=Path(__file__).resolve().parents[1]
def run(raw,step2,processed,output):
    start=time.perf_counter(); output.mkdir(parents=True,exist_ok=True); processed.mkdir(parents=True,exist_ok=True)
    stats=[]; audits=[]; failures=[]; sample=None
    for files in discover_trips(raw):
        tick=time.perf_counter()
        try:
            src=step2/files.split/files.trip_id
            state=json.loads((src/'status.json').read_text()); assert state['state']=='complete'
            for ext,expected in state['source_sha256'].items(): assert hashlib.sha256(files.path(ext).read_bytes()).hexdigest()==expected
            c=pd.read_parquet(src/'candidates.parquet'); track=load_track(files.path('track'))
            net=RoadNetwork(load_nodes(files.path('nodes')),load_arcs(files.path('arcs')))
            clean=clean_observations(track); dest=processed/files.split/files.trip_id; dest.mkdir(parents=True,exist_ok=True)
            clean.to_parquet(dest/'observations.parquet')
            f=engineer_features(c,clean,net); f.to_parquet(dest/'features.parquet',index=False)
            row=dict(partition=files.split,trip_id=files.trip_id,observations=len(clean),feature_rows=len(f),retained=len(clean),modified_raw_values=0,
                     unmatched_candidates=len(clean)-c.gps_observation_index.nunique())
            for col in ['invalid_coordinates','duplicate_record','bad_timestamp','large_gap','speed_outlier','gps_jump']:
                audits.append(dict(partition=files.split,trip_id=files.trip_id,method=col,affected=int(clean[col].sum()),retained=len(clean),modified=0))
            if files.split=='train':
                route=load_route(files.path('route'))
                reference_edges=net.arcs.reindex(route.arc_id.to_numpy())
                breaks=int((reference_edges.to_node_id.to_numpy()[:-1]!=reference_edges.from_node_id.to_numpy()[1:]).sum())
                audits.append(dict(partition='train',trip_id=files.trip_id,method='disconnected_reference_pairs',affected=breaks,retained=len(route),modified=0))
                a=align_training(net,clean,route)
                labels,recall=supervision(c,a,'train')
                if len(recall):
                    geom_lookup={int(arc):i for i,arc in enumerate(net.index_arc_ids)}
                    import shapely
                    ids=recall.index.to_numpy(); raw=track.loc[ids]
                    points=shapely.points(np.column_stack(net.projection.xy(raw.longitude.to_numpy(),raw.latitude.to_numpy())))
                    geometries=net.geometries[[geom_lookup[int(arc)] for arc in recall.aligned_arc_id]]
                    recall['reference_distance_m']=shapely.distance(points,geometries)
                    obs_status=pd.read_parquet(src/'observations.parquet').set_index('gps_observation_index').loc[ids]
                    recall['fallback_used']=obs_status.fallback_used.to_numpy()
                    recall['step2_unmatched']=obs_status.candidate_count.eq(0).to_numpy()
                    cfg=state['configuration']; fallback=cfg['fallback_radius_m'] or cfg['search_radius_m']
                    bound=np.where(obs_status.primary_candidate_count.to_numpy()>0,cfg['search_radius_m'],fallback)
                    recall['miss_reason']=np.where(recall.candidate_present,'present',np.where(recall.reference_distance_m.to_numpy()<=bound,'top8_truncation','search_bound_or_fallback_policy'))
                a.to_parquet(dest/'alignment.parquet'); labels.to_parquet(dest/'labels.parquet',index=False)
                recall.to_parquet(dest/'recall.parquet')
                row.update(valid_aligned=int(a.supervised_eligible.sum()),ambiguous=int(a.alignment_status.eq('ambiguous').sum()),
                  alignment_unmatched=int(a.alignment_status.eq('unmatched').sum()),excluded=int(a.alignment_status.eq('excluded').sum()),
                  candidate_hits=int(recall.candidate_present.sum()),positive=int(labels.target.eq(1).sum()),negative=int(labels.target.eq(0).sum()))
                if files.trip_id=='00000083': sample=(track.copy(),a.copy(),c.copy(),net)
            audits.append(dict(partition=files.split,trip_id=files.trip_id,method='degenerate_geometry',affected=int(net.arcs.geometry_status.ne('valid').sum()),retained=len(net.arcs),modified=0))
            audits.append(dict(partition=files.split,trip_id=files.trip_id,method='unmatched_candidates',affected=row['unmatched_candidates'],retained=len(clean),modified=0))
            row['seconds']=time.perf_counter()-tick; stats.append(row)
            print(files.split,files.trip_id,row.get('valid_aligned','features only'),flush=True)
        except Exception as exc:
            failures.append(dict(partition=files.split,trip_id=files.trip_id,error=str(exc))); print('FAILED',files.trip_id,str(exc),flush=True)
    summary=pd.DataFrame(stats); summary.to_csv(output/'trip_summary.csv',index=False)
    pd.DataFrame(audits).to_csv(output/'cleaning_audit.csv',index=False)
    pd.DataFrame(failures,columns=['partition','trip_id','error']).to_csv(output/'failures.csv',index=False)
    trip_splits(summary.loc[summary.partition=='train','trip_id']).to_csv(processed/'trip_folds.csv',index=False)
    (processed/'feature_schema.json').write_text(json.dumps({'identifiers_excluded_from_model':['dataset_partition','trip_id','gps_observation_index','candidate_arc_id'],
      'features':FEATURES,'undefined_policy':'NaN retained with heading_undefined; fit imputers only inside future training folds',
      'causal':'Only current and preceding GPS observations; no future observations or references'},indent=2))
    config=dict(alignment_radius_m=75,ambiguity_cost_margin=2,heading_penalty_m=20,backward_progress_tolerance_m=5,gap_s=5,speed_kmh=180,jump_m=500,
                fitted_parameters=False,seconds=time.perf_counter()-start,successful_trips=len(stats),failed_trips=len(failures))
    (output/'run_metadata.json').write_text(json.dumps(config,indent=2)); (processed/'preprocessing.json').write_text(json.dumps(config,indent=2))
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    training=summary.loc[summary.partition=='train']
    fig,axes=plt.subplots(1,2,figsize=(12,4)); axes[0].bar(training.trip_id,training.valid_aligned/training.observations*100)
    axes[0].set(title='Confident reference alignment coverage',ylabel='Training observations (%)'); axes[0].tick_params(axis='x',labelrotation=90,labelsize=5)
    axes[1].hist((training.candidate_hits/training.valid_aligned).dropna(),bins=20); axes[1].set(title='Candidate recall on eligible alignment',xlabel='Recall',ylabel='Trips')
    fig.tight_layout(); fig.savefig(output/'alignment_and_recall.png'); plt.close(fig)
    if sample:
        import folium
        track,a,c,net=sample; m=folium.Map(location=track[['latitude','longitude']].mean().tolist(),zoom_start=13,prefer_canvas=True)
        gps=folium.FeatureGroup(name='Original GPS'); aligned=folium.FeatureGroup(name='Reference-derived positions'); roads=folium.FeatureGroup(name='Candidate segments',show=False)
        for i,r in track.iterrows(): folium.CircleMarker([r.latitude,r.longitude],radius=2,color='blue').add_to(gps)
        for i,r in a.dropna(subset=['closest_latitude']).iterrows():
            folium.CircleMarker([r.closest_latitude,r.closest_longitude],radius=3,color='green' if r.supervised_eligible else 'red',tooltip=f'{i}: {r.alignment_status}; route position {r.route_position}').add_to(aligned)
        for arcid in c.candidate_arc_id.unique():
            edge=net.arc(arcid); folium.PolyLine(net.nodes.loc[[edge.from_node_id,edge.to_node_id],['latitude','longitude']].values.tolist(),color='orange',weight=2).add_to(roads)
        gps.add_to(m);aligned.add_to(m);roads.add_to(m);folium.LayerControl().add_to(m)
        m.get_root().html.add_child(folium.Element('<div style="position:fixed;bottom:20px;left:20px;background:white;z-index:1000;padding:10px">Training diagnostic only. Blue: GPS; green: eligible reference positions; red: uncertain/excluded; orange: candidates. No ML predictions.</div>'))
        m.save(str(output/'training_alignment.html'))
    if failures: raise RuntimeError(f'{len(failures)} trips failed; inspect failures.csv')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,default=B/'data/raw');p.add_argument('--step2',type=Path,default=B/'data/processed/step2')
    p.add_argument('--processed',type=Path,default=B/'data/processed/step3');p.add_argument('--output',type=Path,default=B/'outputs/step3'); args=p.parse_args()
    run(args.raw,args.step2,args.processed,args.output)
