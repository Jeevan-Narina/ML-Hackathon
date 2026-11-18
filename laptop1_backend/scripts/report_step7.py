"""Score completed immutable routes, then save static geographic diagnostics."""
from pathlib import Path
import sys,json,argparse
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathsnap.hmm import DecoderConfig,OBSERVATION_COLUMNS
from pathsnap.routing import DirectedRouter
from pathsnap.route_evaluation import selected_accuracy,diagnose_pairs,aggregate_pairs
from pathsnap.loaders import load_nodes


def example_plot(trip,folder,output):
    obs=pd.read_parquet(B/'data/processed/step3/student_test'/trip/'observations.parquet')
    nodes=load_nodes(B/'data/raw/student_test'/trip/f'{trip}.nodes')
    arcs=pd.read_parquet(B/'data/processed/step2/student_test'/trip/'road_arcs.parquet').set_index('arc_id')
    baseline=pd.read_parquet(folder/'classifier_selected.parquet');decoded=pd.read_parquet(folder/'decoded.parquet');route=pd.read_parquet(folder/'route_arcs.parquet')
    def segments(ids):
        a=arcs.loc[list(ids)]
        return np.stack([nodes.loc[a.from_node_id,['longitude','latitude']].to_numpy(),nodes.loc[a.to_node_id,['longitude','latitude']].to_numpy()],axis=1)
    fig,axes=plt.subplots(1,3,figsize=(16,6),sharex=True,sharey=True)
    xmin,xmax=obs.longitude.min(),obs.longitude.max();ymin,ymax=obs.latitude.min(),obs.latitude.max()
    margin=max(xmax-xmin,ymax-ymin)*.04+1e-5
    starts=decoded.loc[decoded.break_reason.ne('') & decoded.break_reason.ne('trip_start'),'gps_observation_index']
    for ax,title in zip(axes,['Classifier-only selected arcs','Viterbi selected candidate arcs','Full route arcs and segment breaks']):
        ax.scatter(obs.longitude,obs.latitude,s=3,color='royalblue',alpha=.35,label='Original GPS')
        ax.set(xlim=(xmin-margin,xmax+margin),ylim=(ymin-margin,ymax+margin),title=title,xlabel='Longitude (WGS84 assumption)')
        ax.set_aspect(1/max(.1,np.cos(np.radians(obs.latitude.mean()))));ax.grid(alpha=.2)
    axes[0].add_collection(LineCollection(segments(baseline.candidate_arc_id.unique()),colors='green',linewidths=1.5,label='Classifier arcs'))
    axes[1].add_collection(LineCollection(segments(decoded.candidate_arc_id.dropna().astype(int).unique()),colors='crimson',linewidths=1.5,label='Viterbi arcs'))
    intermediate=route.loc[route.role.eq('intermediate'),'arc_id'].unique()
    axes[2].add_collection(LineCollection(segments(route.arc_id.unique()),colors='crimson',linewidths=1.3,label='Reconstructed arcs'))
    if len(intermediate):axes[2].add_collection(LineCollection(segments(intermediate),colors='darkorange',linewidths=2.3,label='Intermediate arcs'))
    if len(starts):axes[2].scatter(obs.loc[starts,'longitude'],obs.loc[starts,'latitude'],marker='x',color='black',s=30,label='Segment starts / breaks')
    resets=decoded.loc[decoded.break_reason.eq('no_feasible_transition'),'gps_observation_index']
    if len(resets):axes[2].scatter(obs.loc[resets,'longitude'],obs.loc[resets,'latitude'],marker='D',facecolors='none',edgecolors='purple',s=35,label='No feasible transition: restart')
    diagnostic_path=folder/'classifier_pair_diagnostics.parquet'
    if diagnostic_path.exists():
        pairs=pd.read_parquet(diagnostic_path)
        unreachable=pairs.loc[pairs.assessable & ~pairs.topologically_reachable,'to_observation']
        if len(unreachable):axes[0].scatter(obs.loc[unreachable,'longitude'],obs.loc[unreachable,'latitude'],marker='s',facecolors='none',edgecolors='purple',s=40,label='Directed-unreachable pair')
    for ax in axes:ax.legend(fontsize=8,loc='best')
    axes[0].set_ylabel('Latitude');fig.suptitle(f'Trip {trip}: actual Step 7 diagnostics (no lines across breaks)')
    fig.tight_layout();fig.savefig(output/f'{trip}_route_diagnostic.png',dpi=160);plt.close(fig)


def report(output):
    settings=json.loads((output/'configuration.json').read_text());model=settings.pop('model_name');settings.pop('parameter_policy');config=DecoderConfig(**settings)
    folders=sorted((output/'trips').iterdir());assert len(folders)==10
    per_trip=[];pairs={name:[] for name in ['classifier','viterbi']}
    for folder in folders:
        assert (folder/'complete.json').exists(),'Decode all trips before reading evaluation labels'
        trip=folder.name;obs=pd.read_parquet(B/'data/processed/step3/student_test'/trip/'observations.parquet')[OBSERVATION_COLUMNS]
        arcs=pd.read_parquet(B/'data/processed/step2/student_test'/trip/'road_arcs.parquet').set_index('arc_id')
        router=DirectedRouter(arcs,config.routing_cutoff_m,config.max_settled_nodes,config.cache_sources)
        labels=pd.read_parquet(B/'outputs/step5/evaluation_labels'/trip/'candidate_labels.parquet')
        alignment=pd.read_parquet(B/'outputs/step5/evaluation_labels'/trip/'alignment.parquet')
        for name,file in [('classifier','classifier_selected'),('viterbi','decoded')]:
            selected=pd.read_parquet(folder/f'{file}.parquet')
            diagnostic=diagnose_pairs(selected,obs,router,config);diagnostic.to_parquet(folder/f'{name}_pair_diagnostics.parquet',index=False)
            pairs[name].append(diagnostic)
            per_trip.append(dict(trip_id=trip,variant=name,**selected_accuracy(selected,labels,alignment),**aggregate_pairs(diagnostic)))
    frame=pd.DataFrame(per_trip);frame.to_csv(output/'per_trip_comparison.csv',index=False);rows=[]
    for name in ['classifier','viterbi']:
        group=frame.loc[frame.variant==name];correct=int(group.correct.sum());present=int(group.candidate_present.sum());qualified=int(group.qualified_alignments.sum());raw=int(group.raw_observations.sum())
        rows.append(dict(variant=name,model_name=model,trips=len(group),raw_observations=raw,matched_observations=int(group.matched_observations.sum()),
            qualified_alignments=qualified,candidate_present=present,candidate_misses=qualified-present,correct=correct,
            top1_accuracy=correct/present,top1_including_candidate_misses=correct/qualified,correct_per_raw_observation=correct/raw,
            **aggregate_pairs(pd.concat(pairs[name],ignore_index=True))))
    comparison=pd.DataFrame(rows);comparison.to_csv(output/'classifier_vs_viterbi.csv',index=False)
    for folder in folders[:3]:example_plot(folder.name,folder,output)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    axes[0].bar(comparison.variant,comparison.top1_accuracy,color=['green','crimson']);axes[0].set(title='Qualified candidate-present Top-1',ylim=(0,1))
    axes[1].bar(comparison.variant,comparison.direct_continuity,color=['green','crimson']);axes[1].set(title='Direct continuity on assessable pairs',ylim=(0,1))
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print(comparison.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=B/'outputs/step7');a=p.parse_args();report(a.output)
