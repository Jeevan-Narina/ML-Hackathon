"""Training-only ordered-reference dynamic programming, never inference decoding.

States retain reference occurrences, not just arc IDs. Cost is distance in m
plus 20*(heading disagreement/180); transitions add a capped progress/displacement
penalty. Reference component, position order and directed progress are constrained.
"""
import numpy as np
import pandas as pd
import shapely

def align_training(network, cleaned, route, radius_m=75., ambiguity_margin=2.):
    edges=network.arcs.reindex(route.arc_id.to_numpy())
    valid=edges.geometry_status.eq('valid').to_numpy()
    connected=np.zeros(len(route),dtype=bool)
    if len(route)>1:
        connected[1:]=valid[:-1]&valid[1:]&(edges.to_node_id.to_numpy()[:-1]==edges.from_node_id.to_numpy()[1:])
    component=np.cumsum(~connected)-1
    offsets=np.r_[0,np.cumsum(edges.projected_length_m.fillna(0).to_numpy())[:-1]]
    positions=np.flatnonzero(valid)
    geometry_by_id={int(a):i for i,a in enumerate(network.index_arc_ids)}
    geometries=network.geometries[[geometry_by_id[int(route.iloc[p].arc_id)] for p in positions]]
    tree=shapely.STRtree(geometries)
    rows=[{'observation_id':int(i),'alignment_status':'unmatched','supervised_eligible':False,
           'aligned_arc_id':None,'route_position':None,'reference_component':None,'alignment_distance_m':np.nan,
           'alignment_margin':np.nan,'closest_longitude':np.nan,'closest_latitude':np.nan,'sequence_restart':False} for i in cleaned.index]
    episodes=[]; episode=[]; previous_segment=None
    for row_number,(obs_id,obs) in enumerate(cleaned.iterrows()):
        if previous_segment!=obs.temporal_segment:
            if episode: episodes.append(episode)
            episode=[]; previous_segment=obs.temporal_segment
        if obs.invalid_coordinates: continue
        x,y=network.projection.xy(obs.longitude,obs.latitude); point=shapely.Point(x,y)
        hits=tree.query(shapely.box(x-radius_m,y-radius_m,x+radius_m,y+radius_m))
        d=shapely.distance(point,geometries[hits]); hits=hits[d<=radius_m]; d=d[d<=radius_m]
        if not len(hits): continue
        order=np.argsort(positions[hits]); hits,d=hits[order],d[order]
        pos=positions[hits]; g=geometries[hits]
        fraction=shapely.line_locate_point(g,point,normalized=True)
        progress=offsets[pos]+fraction*edges.projected_length_m.to_numpy()[pos]
        bearing=edges.arc_bearing_deg.to_numpy()[pos]
        heading_cost=np.zeros(len(pos)) if not np.isfinite(obs.gps_heading_deg) else abs((bearing-obs.gps_heading_deg+180)%360-180)/180*20
        episode.append(dict(row=row_number,obs=obs,pos=pos,component=component[pos],progress=progress,
                            emission=d+heading_cost,distance=d,geometry=g,fraction=fraction))
    if episode: episodes.append(episode)

    def transition(a,b):
        delta=b['progress'][None,:]-a['progress'][:,None]
        allowed=(b['pos'][None,:]>=a['pos'][:,None])&(b['component'][None,:]==a['component'][:,None])&(delta>=-5)
        elapsed=b['obs'].timestamp_s-a['obs'].timestamp_s
        allowed &= delta<=500+50*max(0,elapsed if np.isfinite(elapsed) else 0)
        displacement=b['obs'].displacement_m
        penalty=np.minimum(abs(delta-(displacement if np.isfinite(displacement) else 0))/100,10)
        return np.where(allowed,penalty,np.inf)

    def solve(states,restart=False):
        if not states: return
        forward=[states[0]['emission'].copy()]; parents=[]; transitions=[]
        for k in range(1,len(states)):
            tr=transition(states[k-1],states[k]); costs=forward[-1][:,None]+tr
            f=costs.min(axis=0)+states[k]['emission']
            if not np.isfinite(f).any():
                solve(states[:k],restart); solve(states[k:],True); return
            parents.append(costs.argmin(axis=0)); transitions.append(tr); forward.append(f)
        choice=[int(forward[-1].argmin())]
        for parent in parents[::-1]: choice.append(int(parent[choice[-1]]))
        choice=choice[::-1]; backward=[None]*len(states); backward[-1]=np.zeros(len(states[-1]['pos']))
        for k in range(len(states)-2,-1,-1):
            backward[k]=(transitions[k]+states[k+1]['emission'][None,:]+backward[k+1][None,:]).min(axis=1)
        best=float(forward[-1].min())
        for k,(state,j) in enumerate(zip(states,choice)):
            scores=forward[k]+backward[k]; alternatives=np.delete(scores,j)
            margin=float(alternatives.min()-best) if len(alternatives) else np.inf
            ambiguous=margin<ambiguity_margin
            quality=bool(state['obs'].supervision_quality_ok) and not (restart and k==0)
            status='excluded' if not quality else 'ambiguous' if ambiguous else 'valid'
            pos=int(state['pos'][j]); foot=shapely.line_interpolate_point(state['geometry'][j],state['fraction'][j],normalized=True)
            lon,lat=network.projection.lonlat(foot.x,foot.y)
            rows[state['row']].update(alignment_status=status,supervised_eligible=status=='valid',
                aligned_arc_id=int(route.iloc[pos].arc_id),route_position=pos,reference_component=int(component[pos]),
                alignment_distance_m=float(state['distance'][j]),alignment_margin=margin,
                closest_longitude=lon,closest_latitude=lat,sequence_restart=restart and k==0)
    for states in episodes: solve(states)
    result=pd.DataFrame(rows).set_index('observation_id')
    for col in ['aligned_arc_id','route_position','reference_component']: result[col]=result[col].astype('Int64')
    return enforce_consistency(result,cleaned,network)

def enforce_consistency(result,cleaned,network):
    """Exclude contradictions across temporal resets; never repair supervision."""
    result=result.copy(); floor=-1
    for i,row in result.iterrows():
        if not row.supervised_eligible: continue
        heading=cleaned.loc[i,'gps_heading_deg']; bearing=network.arc(int(row.aligned_arc_id)).arc_bearing_deg
        opposite=np.isfinite(heading) and abs((bearing-heading+180)%360-180)>90
        if row.route_position<floor or opposite:
            result.loc[i,['supervised_eligible','alignment_status']]=[False,'excluded']
        else: floor=int(row.route_position)
    return result
