"""Saved-output geographic views. No model inference, decoding or reference fitting."""
from html import escape
import hashlib
import json
import math
import re
import numpy as np
import pandas as pd
import folium
from .quality import valid_coordinates
from .features import IDENTIFIERS

COLORS=dict(gps='#2563eb',candidates='#d89716',classifier='#168047',selected='#8051ce',
            route='#dc334e',intermediate='#f58a21',reference='#087e8b',gap='#7e8492',
            disconnected='#c026b6',unmatched='#111827',network='#8a98aa',direction='#99243c')
LAYER_NAMES=dict(gps='GPS observations',trace='Raw GPS trace (split at gaps)',candidates='Candidate directed roads',
    classifier='Random Forest selected roads',selected='Viterbi selected candidate roads',route='Reconstructed route (partial end arcs)',
    intermediate='Intermediate reconstructed arcs',reference='Reference route — evaluation only',
    gaps='Temporal gaps — markers only',disconnected='Infeasible transitions — markers only',
    unreachable='Directed-unreachable selections — markers only',unmatched='Unmatched observations',
    direction='Travel direction (sampled arrows)',network='Context directed roads (up to 2000)',nodes='Context nodes (up to 600)',endpoints='GPS start and end')


def number(value, digits=4):
    return f'{float(value):.{digits}f}' if pd.notna(value) and math.isfinite(float(value)) else 'unavailable'


def collection(features):return dict(type='FeatureCollection',features=features)


def point_feature(fid,longitude,latitude,**properties):
    if not np.isfinite([longitude,latitude]).all() or abs(longitude)>180 or abs(latitude)>90:
        raise ValueError('Invalid geographic point')
    return dict(type='Feature',id=str(fid),geometry=dict(type='Point',coordinates=[float(longitude),float(latitude)]),properties=properties)


def arc_feature(arcs,nodes,aid,fid,lo=0.,hi=1.,**properties):
    """Original from→to endpoint geometry; clip only at actual saved fractions."""
    if not (0<=lo<=hi<=1):raise ValueError('Invalid directed interval on arc')
    a=arcs.loc[int(aid)];endpoints=nodes.loc[[int(a.from_node_id),int(a.to_node_id)],['longitude','latitude']]
    if not valid_coordinates(endpoints).all():raise ValueError('Arc endpoint has invalid coordinates')
    start,end=endpoints.to_numpy(dtype=float);delta=end-start
    coordinates=[(start+lo*delta).tolist(),(start+hi*delta).tolist()]
    geometry=dict(type='LineString',coordinates=coordinates)
    if coordinates[0]==coordinates[1]:geometry=dict(type='Point',coordinates=coordinates[0])
    return dict(type='Feature',id=str(fid),geometry=geometry,properties=dict(arc_id=int(aid),
        from_node_id=int(a.from_node_id),to_node_id=int(a.to_node_id),start_fraction=float(lo),end_fraction=float(hi),**properties))


def reconstructed_features(routes,decoded,arcs,nodes):
    features=[]
    for segment,group in routes.groupby('segment_id',sort=True):
        group=group.sort_values('route_order');chosen=decoded.loc[decoded.segment_id.eq(segment)]
        if chosen.empty or group.route_order.tolist()!=list(range(len(group))):raise ValueError('Invalid saved segment ordering')
        if int(group.arc_id.iloc[0])!=int(chosen.candidate_arc_id.iloc[0]) or int(group.arc_id.iloc[-1])!=int(chosen.candidate_arc_id.iloc[-1]):
            raise ValueError('Route end arcs do not match saved selections')
        ids=group.arc_id.astype(int).tolist()
        for left,right in zip(ids,ids[1:]):
            if int(arcs.loc[left,'to_node_id'])!=int(arcs.loc[right,'from_node_id']):raise ValueError('Disconnected arcs inside saved segment')
        for position,row in enumerate(group.itertuples(index=False)):
            lo=float(chosen.position_along_arc.iloc[0]) if position==0 else 0.
            hi=float(chosen.position_along_arc.iloc[-1]) if position==len(group)-1 else 1.
            features.append(arc_feature(arcs,nodes,row.arc_id,f'route-{segment}-{row.route_order}',lo,hi,
                segment_id=int(segment),route_order=int(row.route_order),role=row.role,
                detail=f'Segment {segment}; occurrence {row.route_order}; {escape(row.role)}. Actual directed road geometry.'))
    return features


def _selected_features(selected,arcs,nodes,prefix):
    result=[]
    for aid,group in selected.loc[selected.candidate_arc_id.notna()].groupby('candidate_arc_id',sort=True):
        result.append(arc_feature(arcs,nodes,int(aid),f'{prefix}-{aid}',
            observations=len(group),example_observations=', '.join(map(str,group.gps_observation_index.head(6))),
            detail=f'{len(group)} saved selections on this full candidate arc; not a claim of full traversal.'))
    return result


def build_prediction_layers(observations,features,probabilities,classifier,decoded,routes,transitions,pair_diagnostics,nodes,arcs):
    """Reference-free layer builder: input tables → complete display GeoJSON layers.

    Candidate geometry is deduplicated; every observation popup retains all its
    candidates. Context only is sampled; predictions/routes are never simplified.
    """
    if probabilities.duplicated(IDENTIFIERS).any() or features.duplicated(IDENTIFIERS).any():raise ValueError('Duplicate candidate identity')
    data=features.merge(probabilities,on=IDENTIFIERS,how='outer',validate='one_to_one',indicator=True)
    if not data['_merge'].eq('both').all():raise ValueError('Feature/probability identity mismatch')
    if decoded.gps_observation_index.tolist()!=observations.index.tolist():raise ValueError('Decoded observation coverage differs')
    valid=observations.loc[valid_coordinates(observations)]
    if valid.empty:raise ValueError('No valid GPS coordinates')
    grouped={i:g.sort_values(['candidate_rank','candidate_arc_id']) for i,g in data.groupby('gps_observation_index',sort=False)}
    layers={key:collection([]) for key in LAYER_NAMES if key!='reference'}
    for oid,row in valid.iterrows():
        group=grouped.get(oid,data.iloc[:0]);entries=[]
        for c in group.itertuples(index=False):
            entries.append(f'<tr><td>{int(c.candidate_arc_id)}</td><td>{number(c.distance_to_arc_m,1)}</td><td>{number(c.heading_difference_deg,1)}</td><td>{number(c.positive_probability,4)}</td></tr>')
        detail='<table class="candidate-table"><thead><tr><th>Arc</th><th>Distance m</th><th>Heading Δ°</th><th>RF p(+)</th></tr></thead><tbody>'+''.join(entries)+'</tbody></table>' if entries else 'No retained candidates.'
        layers['gps']['features'].append(point_feature(f'gps-{oid}',row.longitude,row.latitude,
            observation_index=int(oid),timestamp_s=number(row.timestamp_s,3),candidate_count=len(group),detail=detail))
    # The raw trace uses its existing temporal segments and never bridges missing/invalid rows.
    pieces=[];current=[];previous=None
    for oid,row in observations.iterrows():
        usable=bool(valid_coordinates(observations.loc[[oid]]).iloc[0])
        boundary=previous is not None and (row.temporal_segment!=previous.temporal_segment or not np.isfinite(row.timestamp_s-previous.timestamp_s) or row.timestamp_s<=previous.timestamp_s)
        if not usable or boundary:
            if len(current)>1:pieces.append(current)
            current=[]
        if usable:current.append([float(row.longitude),float(row.latitude)])
        previous=row if usable else None
    if len(current)>1:pieces.append(current)
    for i,coords in enumerate(pieces):layers['trace']['features'].append(dict(type='Feature',id=f'trace-{i}',geometry=dict(type='LineString',coordinates=coords),properties=dict(detail='Raw GPS trace, not a road route. Split using existing temporal segments.')))
    for aid,group in data.groupby('candidate_arc_id',sort=True):
        row=group.sort_values(['distance_to_arc_m','gps_observation_index']).iloc[0]
        layers['candidates']['features'].append(arc_feature(arcs,nodes,int(aid),f'candidate-{aid}',observations=len(group),
            representative_observation=int(row.gps_observation_index),distance_m=number(row.distance_to_arc_m,2),
            heading_disagreement_deg=number(row.heading_difference_deg,2),classifier_probability=number(row.positive_probability,6),
            detail='Values show the nearest-distance retained row on this arc. Click a GPS observation for all its candidate scores.'))
    layers['classifier']=collection(_selected_features(classifier,arcs,nodes,'classifier'))
    layers['selected']=collection(_selected_features(decoded,arcs,nodes,'viterbi'))
    reconstructed=reconstructed_features(routes,decoded,arcs,nodes)
    layers['route']=collection(reconstructed)
    layers['intermediate']=collection([f for f in reconstructed if f['properties']['role']=='intermediate'])
    # Arrows are display glyphs on sampled actual route occurrences, never inferred connections.
    lines=[f for f in reconstructed if f['geometry']['type']=='LineString']
    step=max(1,math.ceil(len(lines)/160))
    for f in lines[::step]:
        start,end=np.asarray(f['geometry']['coordinates']);delta=end-start;tip=start+.7*delta
        factor=max(.1,math.cos(math.radians(tip[1])));scaled=delta*np.array([factor,1.])
        perpendicular=np.array([-scaled[1],scaled[0]])/np.array([factor,1.])
        wings=[(tip-.14*delta+.06*perpendicular).tolist(),tip.tolist(),(tip-.14*delta-.06*perpendicular).tolist()]
        layers['direction']['features'].append(dict(type='Feature',id='arrow-'+f['id'],geometry=dict(type='LineString',coordinates=wings),properties=dict(detail=f"Directed travel on arc {f['properties']['arc_id']}, segment {f['properties']['segment_id']}. Arrow is a display symbol.")))
    for row in decoded.loc[decoded.break_reason.ne('') & decoded.break_reason.ne('trip_start')].itertuples(index=False):
        oid=int(row.gps_observation_index)
        if oid not in valid.index:continue
        key='unmatched' if row.status=='unmatched' else 'gaps' if row.break_reason=='temporal_gap' else 'disconnected'
        obs=observations.loc[oid]
        layers[key]['features'].append(point_feature(f'{key}-{oid}',obs.longitude,obs.latitude,observation_index=oid,
            reason=row.break_reason,detail='Boundary marker only; no line joins across this break. No-feasible-transition is not necessarily a proof of disconnected topology.'))
    for variant,pairs in pair_diagnostics.items():
        for row in pairs.loc[pairs.assessable & ~pairs.topologically_reachable].itertuples(index=False):
            if row.to_observation not in valid.index:continue
            obs=observations.loc[row.to_observation]
            layers['unreachable']['features'].append(point_feature(f'unreachable-{variant}-{row.to_observation}',obs.longitude,obs.latitude,
                variant=variant,from_observation=int(row.from_observation),to_observation=int(row.to_observation),
                detail='Proven directed-unreachable adjacent selections. Marker only; no connector is drawn.'))
    for label,oid,color in [('Start',int(valid.index[0]),'#168047'),('End',int(valid.index[-1]),'#dc334e')]:
        row=valid.loc[oid];layers['endpoints']['features'].append(point_feature(label.lower(),row.longitude,row.latitude,
            label=label,observation_index=oid,timestamp_s=number(row.timestamp_s,3),color=color))
    # Optional background is explicitly bounded; all candidate/selected/route layers remain complete.
    low=valid[['longitude','latitude']].min().to_numpy()-.002;high=valid[['longitude','latitude']].max().to_numpy()+.002
    a=nodes.reindex(arcs.from_node_id)[['longitude','latitude']].to_numpy();b=nodes.reindex(arcs.to_node_id)[['longitude','latitude']].to_numpy()
    visible=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)&((np.maximum(a,b)>=low)&(np.minimum(a,b)<=high)).all(axis=1)
    ids=arcs.index[visible].to_numpy();total_context=len(ids)
    if len(ids)>2000:ids=ids[np.linspace(0,len(ids)-1,2000,dtype=int)]
    for aid in ids:layers['network']['features'].append(arc_feature(arcs,nodes,int(aid),f'network-{aid}',detail='Context road from original from-node to to-node; context only may be sampled.'))
    node_ids=sorted(set(arcs.loc[ids,['from_node_id','to_node_id']].to_numpy().ravel()))
    if len(node_ids)>600:node_ids=[node_ids[i] for i in np.linspace(0,len(node_ids)-1,600,dtype=int)]
    for nid in node_ids:
        n=nodes.loc[nid];layers['nodes']['features'].append(point_feature(f'node-{nid}',n.longitude,n.latitude,node_id=int(nid),detail='Original road node; bounded optional context.'))
    metadata=dict(observations=len(observations),valid_gps=len(valid),invalid_gps=len(observations)-len(valid),
        candidate_rows=len(data),unique_candidate_arcs=int(data.candidate_arc_id.nunique()),route_occurrences=len(reconstructed),
        decoded_segments=int(decoded.segment_id.nunique()),context_arcs_in_bounds=total_context,
        layer_counts={k:len(v['features']) for k,v in layers.items()},
        bounds=[[float(valid.latitude.min()),float(valid.longitude.min())],[float(valid.latitude.max()),float(valid.longitude.max())]])
    return layers,metadata


def reference_features(route,nodes,arcs):
    """Evaluation-only overlay; never used by build_prediction_layers."""
    return collection([arc_feature(arcs,nodes,int(row.arc_id),f'reference-{position}',
        route_position=int(position),detail='Supplied reference route — evaluation only, not a prediction. Arc occurrences are drawn separately; reference breaks are not bridged.')
        for position,row in route.iterrows()])


def stable_html(html):
    """Canonicalize Folium/Branca random element IDs without global RNG patches."""
    names={}
    def replace(match):
        token=match.group(0)
        if token not in names:names[token]=hashlib.sha256(f'pathsnap-map-{len(names)}'.encode()).hexdigest()[:32]
        return names[token]
    return re.sub(r'(?<=_)[0-9a-f]{32}(?=_|\b)',replace,html)


def render_map(trip,layers,metadata,metrics,index_href='../index.html'):
    """GeoJSON → standalone HTML with relative local libraries and OSM tiles."""
    if not re.fullmatch(r'\d{8}',trip):raise ValueError('Invalid trip ID')
    bounds=metadata['bounds'];center=np.mean(bounds,axis=0).tolist()
    m=folium.Map(location=center,tiles='OpenStreetMap',prefer_canvas=True,control_scale=True)
    m.default_js=[('leaflet','../assets/leaflet.js'),('jquery','../assets/jquery.min.js')]
    m.default_css=[('leaflet_css','../assets/leaflet.css')]
    groups={};defaults={'gps','route','intermediate','endpoints','gaps','disconnected','unreachable','unmatched'}
    color_keys={'trace':'gps','endpoints':'gps','gaps':'gap','unreachable':'disconnected','nodes':'network'}
    for key in LAYER_NAMES:
        if key not in layers:continue
        group=folium.FeatureGroup(name=LAYER_NAMES[key],show=key in defaults);group.add_to(m);groups[key]=group
        features=layers[key]['features']
        if not features:continue
        color=COLORS.get(color_keys.get(key,key),'#333333')
        fields=list(features[0]['properties'])
        style=dict(color=color,weight=4 if key in ['route','intermediate'] else 3 if key in ['classifier','selected','reference'] else 1.5,
                   opacity=.85,fillColor=color,fillOpacity=.65,smoothFactor=0)
        if key=='trace':style.update(dashArray='4 5',weight=1.5)
        if key=='reference':style.update(dashArray='8 5',weight=5)
        point_radius=3 if key in ['gps','nodes','route','intermediate'] else 7
        geo=folium.GeoJson(collection(features),name=LAYER_NAMES[key],control=False,smooth_factor=0,
            style_function=lambda feature,s=style:s,
            marker=folium.CircleMarker(radius=point_radius,color=color,weight=2,fill=True,fill_color=color,fill_opacity=.7),
            popup=folium.GeoJsonPopup(fields=fields,aliases=[field.replace('_',' ').title() for field in fields],localize=False,max_width=480),
            tooltip=folium.GeoJsonTooltip(fields=fields[:min(3,len(fields))],sticky=False))
        geo.add_to(group)
        if key=='endpoints':
            for f in features:
                p=f['properties'];lon,lat=f['geometry']['coordinates']
                folium.Marker([lat,lon],icon=folium.DivIcon(html=f'<div class="endpoint" style="background:{p["color"]}">{p["label"]}</div>',icon_size=(48,24),icon_anchor=(24,12)),
                    tooltip=f'{p["label"]}: observation {p["observation_index"]}; t={p["timestamp_s"]}').add_to(group)
    folium.LayerControl(collapsed=True).add_to(m);m.fit_bounds(bounds,padding=(35,35))
    classifier=metrics['classifier'];viterbi=metrics['viterbi'];delta=(viterbi['top1_accuracy']-classifier['top1_accuracy'])*100
    header=f'''<aside id="pathsnap-panel"><a href="{escape(index_href,quote=True)}">← All trips</a><div class="eyebrow">PATHSNAP / STUDENT TEST</div><h1>Trip {trip}</h1>
      <div class="metrics"><span>RF Top-1<b>{classifier['top1_accuracy']:.2%}</b></span><span>Viterbi Top-1<b>{viterbi['top1_accuracy']:.2%}</b></span></div>
      <p>{metadata['observations']:,} GPS observations · {metadata['decoded_segments']} segments<br><strong>{delta:+.3f} percentage points</strong> on qualified, candidate-present labels.</p>
      <div class="view-buttons"><button onclick="pathsnapView('before')">Before · RF</button><button onclick="pathsnapView('after')">After · Viterbi</button><button onclick="pathsnapView('both')">Compare</button></div>
      <details><summary>How to inspect</summary><p>Use the layer control at top right. Zoom in, then click a GPS point for every retained candidate and its score. Reference data is evaluation-only and starts hidden.</p><p>RF direct continuity {classifier['direct_continuity']:.2%}; Viterbi {viterbi['direct_continuity']:.2%}. Break markers remain visible in every view.</p></details></aside>'''
    legend=''.join(f'<div><i style="background:{COLORS[color]}"></i>{label}</div>' for color,label in [
        ('gps','GPS observations / dashed raw trace'),('candidates','Candidate roads'),('classifier','Random Forest selections'),('selected','Viterbi candidate selections'),
        ('route','Reconstructed road route'),('intermediate','Intermediate arcs'),('reference','Reference — evaluation only (dashed)'),('gap','Temporal gap marker'),('disconnected','Infeasible / unreachable marker'),('unmatched','Unmatched observation')])
    m.get_root().html.add_child(folium.Element(header+f'<details id="pathsnap-legend" open><summary>Map legend</summary>{legend}<p>Roads follow original directed endpoints. No line bridges a segment break. Context and direction glyphs are sampled; route arcs are complete.</p></details>'))
    css='''<style>html,body{margin:0;font-family:Arial,sans-serif}#pathsnap-panel,#pathsnap-legend{position:fixed;z-index:1000;background:#fffdf9;border:1px solid #d8dedc;border-radius:12px;box-shadow:0 4px 20px #142c2520;color:#183b35}#pathsnap-panel{left:60px;top:16px;width:300px;padding:18px}#pathsnap-panel h1{font-size:24px;margin:7px 0 12px}.eyebrow{font-size:10px;letter-spacing:1.5px;margin-top:15px;color:#60726d}a{color:#21645b}p{font-size:12px;line-height:1.6}.metrics{display:flex;gap:20px}.metrics span{font-size:11px;color:#52655f}.metrics b{display:block;font-size:23px;color:#183b35;margin-top:5px}.view-buttons{display:flex;gap:5px;margin:12px 0}button{padding:8px 7px;border-radius:5px;border:1px solid #b4c9c2;background:#edf5ef;color:#183b35;cursor:pointer;font-size:11px}button:hover{background:#dcebe1}summary{cursor:pointer;font-size:12px;font-weight:bold}#pathsnap-legend{right:14px;bottom:25px;max-width:250px;padding:12px;font-size:11px}#pathsnap-legend div{margin-top:6px}#pathsnap-legend i{display:inline-block;width:18px;height:4px;vertical-align:middle;margin-right:8px}#pathsnap-legend p{font-size:10px;margin-bottom:0}.leaflet-control-layers-expanded{max-height:65vh;overflow:auto}.leaflet-popup-content{max-height:330px;overflow:auto;font-size:12px}.leaflet-popup-content table{border-collapse:collapse}.leaflet-popup-content th,.leaflet-popup-content td{padding:4px 6px;border-bottom:1px solid #e3e9e6;text-align:left}.endpoint{color:white;font:11px Arial,sans-serif;font-weight:bold;border:2px solid white;border-radius:10px;padding:4px;text-align:center;box-shadow:0 1px 4px #555}.leaflet-tooltip{font:11px Arial,sans-serif}@media(max-width:650px){#pathsnap-panel{left:48px;width:220px;padding:10px}#pathsnap-panel h1{font-size:19px}.metrics b{font-size:18px}#pathsnap-legend{max-width:175px;font-size:10px}#pathsnap-panel details{display:none}}</style>'''
    m.get_root().header.add_child(folium.Element('{% raw %}'+css+'{% endraw %}'))
    controlled=['classifier','selected','route','intermediate','direction','reference']
    names={key:groups[key].get_name() for key in controlled if key in groups}
    object_text='{'+','.join(json.dumps(k)+':'+v for k,v in names.items())+'}'
    js=f'''var pathsnapLayers={object_text};window.pathsnapView=function(mode){{Object.keys(pathsnapLayers).forEach(function(k){{var active=mode==='before'?k==='classifier':mode==='after'?['route','intermediate'].includes(k):['classifier','route','intermediate'].includes(k);if(active){{{m.get_name()}.addLayer(pathsnapLayers[k]);}}else{{{m.get_name()}.removeLayer(pathsnapLayers[k]);}}}});}};window.pathsnapReady=true;'''
    # Root script children render after the Map/GeoJSON scripts.
    m.get_root().script.add_child(folium.Element(js))
    html=m.get_root().render()
    # Branca root script children can precede Map render; move our controls last.
    html=html.replace(js,'').replace('</html>',f'<script>{js}</script>\n</html>')
    html=html.replace('<head>','<head>\n<title>PathSnap · Trip '+trip+'</title>',1)
    return stable_html(html)
