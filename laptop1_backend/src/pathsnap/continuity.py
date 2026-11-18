"""Classifier-only directed reachability diagnostics, without reconstructing paths."""
import numpy as np
import pandas as pd
import networkx as nx

class DirectedReachability:
    def __init__(self,nodes,arcs):
        graph=nx.DiGraph();graph.add_nodes_from(nodes.index)
        good=arcs.from_node_id.isin(nodes.index)&arcs.to_node_id.isin(nodes.index)
        graph.add_edges_from(arcs.loc[good,['from_node_id','to_node_id']].itertuples(index=False,name=None))
        self.dag=nx.condensation(graph);self.component=self.dag.graph['mapping'];self.cache={}
    def reachable(self,source,target):
        a,b=self.component[source],self.component[target]
        if a==b:return True
        if a not in self.cache:self.cache[a]=nx.descendants(self.dag,a)
        return b in self.cache[a]

def continuity_diagnostics(winners,observations,arcs,reachability):
    if winners.gps_observation_index.duplicated().any():raise ValueError('Require one predicted arc per observation')
    chosen=winners.set_index('gps_observation_index');rows=[]
    ids=observations.index.to_numpy();times=observations.timestamp_s.to_numpy()
    for k in range(1,len(ids)):
        previous,current=int(ids[k-1]),int(ids[k]);dt=float(times[k]-times[k-1]);category=''
        if not np.isfinite(dt) or dt<=0:category='invalid_timestamp'
        elif dt>5:category='temporal_gap'
        elif previous not in chosen.index or current not in chosen.index:category='missing_prediction'
        backward=False
        if not category:
            left,right=chosen.loc[previous],chosen.loc[current]
            a,b=int(left.candidate_arc_id),int(right.candidate_arc_id)
            if a not in arcs.index or b not in arcs.index:raise ValueError('Prediction refers to unknown original arc ID')
            if a==b:
                category='same_arc'
                if 'position_along_arc' in chosen:backward=bool(right.position_along_arc<left.position_along_arc-1e-9)
            elif int(arcs.loc[a,'to_node_id'])==int(arcs.loc[b,'from_node_id']):category='directly_connected'
            elif reachability.reachable(int(arcs.loc[a,'to_node_id']),int(arcs.loc[b,'from_node_id'])):category='intermediate_required'
            else:category='unreachable'
        rows.append(dict(from_observation=previous,to_observation=current,dt_s=dt,category=category,same_arc_backward_fraction=backward))
    return pd.DataFrame(rows,columns=['from_observation','to_observation','dt_s','category','same_arc_backward_fraction'])

def summarise_continuity(frame):
    counts=frame.category.value_counts().to_dict()
    eligible=sum(counts.get(k,0) for k in ['same_arc','directly_connected','intermediate_required','unreachable'])
    direct=counts.get('same_arc',0)+counts.get('directly_connected',0)
    return dict(adjacent_observation_pairs=len(frame),assessable_pairs=eligible,
        **{k:counts.get(k,0) for k in ['same_arc','directly_connected','intermediate_required','unreachable','temporal_gap','invalid_timestamp','missing_prediction']},
        same_arc_backward_fraction=int(frame.same_arc_backward_fraction.sum()),
        direct_continuity=direct/eligible if eligible else None,
        topological_reachability=(direct+counts.get('intermediate_required',0))/eligible if eligible else None)
