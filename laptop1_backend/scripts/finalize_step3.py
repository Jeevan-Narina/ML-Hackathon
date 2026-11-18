"""Audit persisted alignments, compute recall diagnostics, and document real results."""
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
import shapely
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pathsnap.loaders import load_nodes,load_arcs,load_route
from pathsnap.road_network import RoadNetwork
from pathsnap.alignment import enforce_consistency
from pathsnap.features import supervision,FEATURES
B=Path(__file__).resolve().parents[1]; O=B/'outputs/step3'; P=B/'data/processed/step3'; S=B/'data/processed/step2'

def main():
    start=time.perf_counter(); summary=pd.read_csv(O/'trip_summary.csv',dtype={'trip_id':str}); audits=pd.read_csv(O/'cleaning_audit.csv',dtype={'trip_id':str})
    recalls=[]; distributions=[]; extra=[]; feature_arrays=[]
    for index,row in summary.iterrows():
        d=P/row.partition/row.trip_id; src=S/row.partition/row.trip_id
        clean=pd.read_parquet(d/'observations.parquet');c=pd.read_parquet(src/'candidates.parquet')
        f=pd.read_parquet(d/'features.parquet'); assert len(f)==len(c)
        assert not set(f)&{'target','aligned_arc_id','route_position','reference_component'}
        feature_arrays.append(f[['distance_to_arc_m','heading_difference_deg']].assign(partition=row.partition))
        if row.partition!='train':
            assert not (d/'labels.parquet').exists() and not (d/'alignment.parquet').exists()
            continue
        raw=B/'data/raw'/row.partition/row.trip_id
        net=RoadNetwork(load_nodes(raw/f'{row.trip_id}.nodes'),load_arcs(raw/f'{row.trip_id}.arcs'))
        a=enforce_consistency(pd.read_parquet(d/'alignment.parquet'),clean,net);a.to_parquet(d/'alignment.parquet')
        labels,r=supervision(c,a,'train');labels.to_parquet(d/'labels.parquet',index=False)
        ids=r.index; obs=clean.loc[ids];lookup={int(arc):i for i,arc in enumerate(net.index_arc_ids)}
        points=shapely.points(np.column_stack(net.projection.xy(obs.longitude.to_numpy(),obs.latitude.to_numpy())))
        r['reference_distance_m']=shapely.distance(points,net.geometries[[lookup[int(arc)] for arc in r.aligned_arc_id]])
        status=pd.read_parquet(src/'observations.parquet').set_index('gps_observation_index').loc[ids]
        cfg=json.loads((src/'status.json').read_text())['configuration']
        bound=np.where(status.primary_candidate_count>0,cfg['search_radius_m'],cfg['fallback_radius_m'] or cfg['search_radius_m'])
        r['fallback_used']=status.fallback_used.to_numpy();r['step2_unmatched']=status.candidate_count.eq(0).to_numpy()
        r['miss_reason']=np.where(r.candidate_present,'present',np.where(r.reference_distance_m<=bound,'top8_truncation','search_bound_or_fallback_policy'))
        r.to_parquet(d/'recall.parquet');recalls.append(r.assign(trip_id=row.trip_id))
        for key,value in dict(valid_aligned=int(a.supervised_eligible.sum()),ambiguous=int(a.alignment_status.eq('ambiguous').sum()),
              alignment_unmatched=int(a.alignment_status.eq('unmatched').sum()),excluded=int(a.alignment_status.eq('excluded').sum()),
              candidate_hits=int(r.candidate_present.sum()),positive=int(labels.target.eq(1).sum()),negative=int(labels.target.eq(0).sum())).items():summary.loc[index,key]=value
        assert a.loc[a.supervised_eligible,'route_position'].is_monotonic_increasing
        assert labels.loc[labels.training_eligible].groupby('gps_observation_index').target.sum().eq(1).all()
        edges=net.arcs.reindex(load_route(raw/f'{row.trip_id}.route').arc_id.to_numpy())
        extra.append(dict(partition='train',trip_id=row.trip_id,method='disconnected_reference_pairs',affected=int((edges.to_node_id.to_numpy()[:-1]!=edges.from_node_id.to_numpy()[1:]).sum()),retained=len(edges),modified=0))
    summary.to_csv(O/'trip_summary.csv',index=False)
    # Add geometry/unmatched audit methods using already verified Step 2 metadata.
    for row in summary.itertuples():
        net=pd.read_parquet(S/row.partition/row.trip_id/'road_arcs.parquet',columns=['geometry_status'])
        for method,n in [('degenerate_geometry',int(net.geometry_status.ne('valid').sum())),('unmatched_candidates',int(row.unmatched_candidates))]:
            extra.append(dict(partition=row.partition,trip_id=row.trip_id,method=method,affected=n,retained=len(net) if method=='degenerate_geometry' else int(row.observations),modified=0))
    audits=pd.concat([audits,pd.DataFrame(extra)],ignore_index=True).drop_duplicates(['partition','trip_id','method'],keep='last')
    audits.to_csv(O/'cleaning_audit.csv',index=False)
    recall=pd.concat(recalls);recall.groupby('miss_reason').size().rename('observations').to_csv(O/'recall_causes.csv')
    feature=pd.concat(feature_arrays,ignore_index=True);feature.groupby('partition')[['distance_to_arc_m','heading_difference_deg']].describe().to_csv(O/'feature_statistics.csv')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for part,g in feature.groupby('partition'):
        axes[0].hist(g.distance_to_arc_m,bins=np.linspace(0,150,40),alpha=.5,label=part)
        axes[1].hist(g.heading_difference_deg.dropna(),bins=np.linspace(0,180,37),alpha=.5,label=part)
    for ax,title in zip(axes,['Distance feature (m)','Causal heading difference (degrees)']):ax.set_title(title);ax.legend()
    fig.tight_layout();fig.savefig(O/'feature_distributions.png');plt.close(fig)
    train=summary.loc[summary.partition=='train']; fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].hist(train.valid_aligned/train.observations,bins=20);axes[0].set_title('Eligible training alignment coverage by trip')
    axes[1].hist(train.candidate_hits/train.valid_aligned,bins=20);axes[1].set_title('Candidate recall by trip')
    fig.tight_layout();fig.savefig(O/'alignment_and_recall.png');plt.close(fig)
    # Map colours must reflect final exclusions, not preliminary alignments.
    import folium
    d=P/'train/00000083';a=pd.read_parquet(d/'alignment.parquet');obs=pd.read_parquet(d/'observations.parquet');c=pd.read_parquet(S/'train/00000083/candidates.parquet')
    m=folium.Map(location=obs[['latitude','longitude']].mean().tolist(),zoom_start=13,prefer_canvas=True)
    for i,r in obs.iterrows():folium.CircleMarker([r.latitude,r.longitude],radius=2,color='blue').add_to(m)
    for i,r in a.dropna(subset=['closest_latitude']).iterrows():folium.CircleMarker([r.closest_latitude,r.closest_longitude],radius=3,color='green' if r.supervised_eligible else 'red',tooltip=f'{i}: {r.alignment_status}').add_to(m)
    layer=folium.FeatureGroup(name='Candidate projections',show=False)
    raw=B/'data/raw/train/00000083'; nodes=load_nodes(raw/'00000083.nodes');arcs=load_arcs(raw/'00000083.arcs')
    segments=folium.FeatureGroup(name='Candidate road segments',show=False)
    for arc_id in c.candidate_arc_id.unique():
        edge=arcs.loc[arc_id];folium.PolyLine(nodes.loc[[edge.from_node_id,edge.to_node_id],['latitude','longitude']].values.tolist(),color='orange',weight=2).add_to(segments)
    segments.add_to(m)
    for r in c.itertuples():folium.CircleMarker([r.projected_latitude,r.projected_longitude],radius=1,color='orange').add_to(layer)
    layer.add_to(m);folium.LayerControl().add_to(m);m.get_root().html.add_child(folium.Element('<div style="position:fixed;bottom:20px;left:20px;z-index:1000;background:white;padding:8px">Training diagnostic, no ML prediction. Blue: raw GPS; green: eligible reference positions; red: uncertain/excluded; orange layer: candidate projections.</div>'));m.save(str(O/'training_alignment.html'))
    totals=train[['observations','valid_aligned','ambiguous','alignment_unmatched','excluded','candidate_hits','positive','negative']].sum().to_dict()
    totals.update(candidate_recall=float(recall.candidate_present.mean()),miss_causes=recall.miss_reason.value_counts().to_dict(),
                  successful_trips=len(summary),feature_rows=int(summary.feature_rows.sum()),finalization_seconds=time.perf_counter()-start)
    (O/'results.json').write_text(json.dumps(totals,indent=2))
    print(json.dumps(totals,indent=2))
if __name__=='__main__':main()
