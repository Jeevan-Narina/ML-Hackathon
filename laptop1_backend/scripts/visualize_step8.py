"""Generate resumable Folium maps from existing validated Step 7 files only."""
from pathlib import Path
import sys,json,time,hashlib,argparse,importlib.metadata,ssl,subprocess,os
from html import escape
from urllib.request import urlopen
from urllib.error import URLError
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
import pandas as pd
from pathsnap.loaders import load_nodes,load_route
from pathsnap.map_visualization import build_prediction_layers,reference_features,render_map,LAYER_NAMES

ASSET_URLS={
 'leaflet.js':'https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js',
 'leaflet.css':'https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css',
 'jquery.min.js':'https://code.jquery.com/jquery-3.7.1.min.js',
 'LEAFLET-LICENSE.txt':'https://cdn.jsdelivr.net/npm/leaflet@1.9.3/LICENSE',
 'JQUERY-LICENSE.txt':'https://cdn.jsdelivr.net/npm/jquery@3.7.1/LICENSE.txt',
 **{f'images/{name}':f'https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/images/{name}' for name in ['layers.png','layers-2x.png','marker-icon.png','marker-icon-2x.png','marker-shadow.png']}}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def ensure_assets(output):
    folder=output/'assets';folder.mkdir(parents=True,exist_ok=True);manifest=folder/'manifest.json'
    saved=json.loads(manifest.read_text()) if manifest.exists() else {}
    for name,url in ASSET_URLS.items():
        path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
        if name in saved and path.exists() and digest(path)==saved[name]['sha256']:continue
        tmp=path.with_suffix('.tmp')
        try:
            with urlopen(url,timeout=30) as response:data=response.read()
            tmp.write_bytes(data)
        except URLError as error:
            if os.name!='nt' or not isinstance(error.reason,ssl.SSLCertVerificationError):raise
            # Windows uses its own trusted certificate store; never disable TLS checks.
            quote=lambda text:"'"+str(text).replace("'","''")+"'"
            subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
                f'Invoke-WebRequest -Uri {quote(url)} -OutFile {quote(tmp.resolve())} -UseBasicParsing'],
                check=True,creationflags=subprocess.CREATE_NO_WINDOW,timeout=45)
        if not tmp.stat().st_size:raise ValueError(f'Empty map asset {url}')
        tmp.replace(path)
        saved[name]=dict(url=url,sha256=digest(path),bytes=path.stat().st_size)
        write_json(manifest,saved)
    return saved


def load_trip_layers(trip):
    folder=B/'outputs/step7/trips'/trip;root=B/'data/processed/step3/student_test'/trip
    nodes=load_nodes(B/'data/raw/student_test'/trip/f'{trip}.nodes')
    arcs=pd.read_parquet(B/'data/processed/step2/student_test'/trip/'road_arcs.parquet').set_index('arc_id')
    obs=pd.read_parquet(root/'observations.parquet');features=pd.read_parquet(root/'features.parquet')
    frames={key:pd.read_parquet(folder/f'{name}.parquet') for key,name in dict(probabilities='probabilities',classifier='classifier_selected',decoded='decoded',routes='route_arcs',transitions='transitions').items()}
    pairs={variant:pd.read_parquet(folder/f'{variant}_pair_diagnostics.parquet') for variant in ['classifier','viterbi']}
    layers,metadata=build_prediction_layers(obs,features,frames['probabilities'],frames['classifier'],frames['decoded'],frames['routes'],frames['transitions'],pairs,nodes,arcs)
    # Predictions are already complete before reading this optional evaluation overlay.
    reference_path=B/'data/raw/student_test'/trip/f'{trip}.route'
    if reference_path.exists():layers['reference']=reference_features(load_route(reference_path),nodes,arcs)
    metadata.update(reference_available='reference' in layers,reference_used_for_prediction=False,
        reference_occurrences=len(layers.get('reference',{}).get('features',[])),layer_counts={k:len(v['features']) for k,v in layers.items()})
    return layers,metadata


def choose_examples(summary):
    best=summary.sort_values(['improvement_pp','trip_id'],ascending=[False,True]).iloc[0]
    gaps=summary.sort_values(['temporal_gaps','infeasible_breaks','trip_id'],ascending=[False,False,True]).iloc[0]
    ambiguous=summary.sort_values(['ambiguous_candidate_observations','trip_id'],ascending=[False,True]).iloc[0]
    return [dict(role='Largest measured Top-1 improvement',trip_id=best.trip_id,reason=f'{best.improvement_pp:+.3f} percentage points'),
      dict(role='Dense candidate ambiguity / junction inspection',trip_id=ambiguous.trip_id,reason=f'{int(ambiguous.ambiguous_candidate_observations)} observations with multiple retained candidates; inspect junctions on the map, not a labelled intersection count'),
      dict(role='Temporal gaps and infeasible restarts',trip_id=gaps.trip_id,reason=f'{int(gaps.temporal_gaps)} temporal gaps; {int(gaps.infeasible_breaks)} infeasible restarts'),
      dict(role='Measured limitation',trip_id='00000065',reason=f"{summary.set_index('trip_id').loc['00000065','improvement_pp']:+.3f} percentage points despite improved continuity")]


def write_index(output,summary,examples):
    cards=''.join(f'<a class="example" href="examples/{e["trip_id"]}.html"><span>{escape(e["role"])}</span><h3>{e["trip_id"]}</h3><p>{escape(e["reason"])}</p></a>' for e in examples)
    rows=[]
    for r in summary.itertuples():
        rows.append(f'<tr><td><a href="{r.map_href}">{r.trip_id} ↗</a></td><td>{r.observations:,}</td><td>{r.decoded_segments}</td><td>{r.classifier_top1:.2%}</td><td>{r.viterbi_top1:.2%}</td><td class="{"gain" if r.improvement_pp>=0 else "loss"}">{r.improvement_pp:+.3f}</td><td>{r.classifier_continuity:.2%} → {r.viterbi_continuity:.2%}</td></tr>')
    pooled=pd.read_csv(B/'outputs/step7/classifier_vs_viterbi.csv').set_index('variant')
    html='''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PathSnap · Route comparison atlas</title>
    <style>*{box-sizing:border-box}body{margin:0;background:#f5f4ef;color:#163b34;font:15px Arial,sans-serif}header{background:#183e35;color:#f9fbf6;padding:32px max(6vw,20px)}header .brand{font-size:12px;letter-spacing:3px;color:#b7d2c3}h1{font-size:clamp(30px,4vw,48px);letter-spacing:-1px;margin:14px 0}header p{color:#c6d9cf;max-width:730px;line-height:1.7}main{max-width:1200px;margin:auto;padding:30px 24px 50px}.stats{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:30px}.stat{background:#fffdf8;border:1px solid #dee4da;padding:18px 25px;border-radius:12px;min-width:180px}.stat b{display:block;font-size:27px;margin-top:7px}.stat span{font-size:12px;color:#64766d}h2{margin-top:32px;font-size:22px}.examples{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}.example{background:#fffdf8;border:1px solid #dce4d9;padding:20px;border-radius:12px;text-decoration:none;color:inherit}.example:hover{border-color:#438970}.example span{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:#5b7165}.example h3{font-size:22px;margin:12px 0}.example p,.note{font-size:13px;line-height:1.6;color:#62756b}.table-wrap{overflow:auto;background:white;border:1px solid #dce4d9;border-radius:12px}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{text-align:left;padding:15px 16px;border-bottom:1px solid #e9ede5;font-size:13px}th{background:#edf1e9;font-size:11px;text-transform:uppercase;letter-spacing:.5px}a{color:#206854}.gain{color:#247849}.loss{color:#b14c31}.notice{margin-top:24px;padding:18px;background:#e8eee3;border-radius:10px;font-size:13px;line-height:1.7}footer{margin-top:30px;font-size:12px;color:#68796f}</style></head><body><header><div class="brand">PATHSNAP / GEOGRAPHIC ATLAS</div><h1>From noisy points to road routes.</h1><p>Compare the frozen Random Forest selections with HMM/Viterbi reconstruction. Explore real GPS observations, directed candidate roads, intermediate arcs and explicit route breaks across all ten student-test trips.</p></header><main>'''
    html+=f'<div class="stats"><div class="stat"><span>Trips / GPS observations</span><b>{len(summary)} / {int(summary.observations.sum()):,}</b></div><div class="stat"><span>Classifier → Viterbi Top-1</span><b>{pooled.loc["classifier","top1_accuracy"]:.2%} → {pooled.loc["viterbi","top1_accuracy"]:.2%}</b></div><div class="stat"><span>Separate decoded segments</span><b>{int(summary.decoded_segments.sum())}</b></div></div>'
    html+='<h2>Start with an example</h2><div class="examples">'+cards+'</div><h2>All student-test trips</h2><p class="note">Top-1 is conditional on qualified reference labels and candidate availability. Δ is in percentage points. Continuity compares legal direct adjacency; gaps remain explicit.</p><div class="table-wrap"><table><thead><tr><th>Trip / map</th><th>GPS points</th><th>Segments</th><th>RF Top-1</th><th>Viterbi Top-1</th><th>Δ pp</th><th>Direct continuity · before → after</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
    html+='<div class="notice"><strong>Reading the maps.</strong> Use Before, After or Compare, then enable optional layers. Click GPS points for candidate distances, heading disagreement and classifier scores. Reference roads are evaluation-only and initially hidden. No route line connects separate decoded segments.<br><strong>Portable viewing.</strong> Keep assets/, maps/ and examples/ with this index. Local map libraries are included; OpenStreetMap tiles need internet. These files open directly in a browser without a Python server.</div><footer>Step 8 · saved Step 7 results · no model training or decoding · <a href="summary.csv">Download summary CSV</a></footer></main></body></html>'
    (output/'index.html').write_text(html,encoding='utf-8')


def run(output,force=False):
    output.mkdir(parents=True,exist_ok=True)
    for name in ['maps','data','metadata','examples']:(output/name).mkdir(exist_ok=True)
    assets=ensure_assets(output);started=time.perf_counter();source=B/'outputs/step7'
    assert json.loads((source/'verification.json').read_text())['status']=='passed'
    protection=output/'protected_artifacts.json'
    if not protection.exists():
        prior=json.loads((source/'protected_artifacts.json').read_text())
        paths={ROOT_REL:hashvalue for ROOT_REL,hashvalue in prior.items()}
        for p in [*source.rglob('*'),*list((B.parent/'docs/steps').glob('STEP_0[1-7]_README.md'))]:
            if p.is_file():paths[p.relative_to(B.parent).as_posix()]=digest(p)
        write_json(protection,paths)
    # All Step 7 artifacts are verified before maps are rendered; no decoder imports.
    metrics=pd.read_csv(source/'per_trip_comparison.csv',dtype={'trip_id':str})
    code_sha=digest(B/'src/pathsnap/map_visualization.py');versions={k:importlib.metadata.version(k) for k in ['folium','branca','pandas','numpy']}
    rows=[];timings=[]
    for folder in sorted((source/'trips').iterdir()):
        trip=folder.name;stamp=json.loads((folder/'complete.json').read_text())
        for name,value in stamp['outputs_sha256'].items():assert digest(folder/name)==value,f'Step 7 output invalid: {trip}/{name}'
        inputs=[*folder.glob('*.parquet'),B/'data/processed/step3/student_test'/trip/'features.parquet',B/'data/processed/step3/student_test'/trip/'observations.parquet',
                B/'data/processed/step2/student_test'/trip/'road_arcs.parquet',B/'data/raw/student_test'/trip/f'{trip}.nodes',source/'per_trip_comparison.csv',source/'configuration.json']
        ref=B/'data/raw/student_test'/trip/f'{trip}.route'
        if ref.exists():inputs.append(ref)
        identity=dict(inputs={p.relative_to(B).as_posix():digest(p) for p in inputs},code_sha256=code_sha,renderer_sha256=digest(Path(__file__)),versions=versions,assets=assets)
        meta_path=output/'metadata'/f'{trip}.json';html_path=output/'maps'/f'{trip}.html';data_path=output/'data'/f'{trip}.json'
        tick=time.perf_counter();reused=False
        if not force and meta_path.exists() and html_path.exists() and data_path.exists():
            meta=json.loads(meta_path.read_text())
            reused=meta['identity']==identity and digest(html_path)==meta['html_sha256'] and digest(data_path)==meta['data_sha256']
        if not reused:
            layers,metadata=load_trip_layers(trip)
            trip_metrics={name:metrics.loc[(metrics.trip_id==trip)&(metrics.variant==name)].iloc[0].to_dict() for name in ['classifier','viterbi']}
            html=render_map(trip,layers,metadata,trip_metrics);tmp=html_path.with_suffix('.tmp');tmp.write_text(html,encoding='utf-8');tmp.replace(html_path)
            write_json(data_path,dict(trip_id=trip,layers=layers,metadata=metadata))
            meta=dict(trip_id=trip,identity=identity,display=metadata,html_sha256=digest(html_path),data_sha256=digest(data_path),html_bytes=html_path.stat().st_size,generation_seconds=time.perf_counter()-tick)
            write_json(meta_path,meta)
        classifier=metrics.loc[(metrics.trip_id==trip)&(metrics.variant=='classifier')].iloc[0];viterbi=metrics.loc[(metrics.trip_id==trip)&(metrics.variant=='viterbi')].iloc[0]
        decoded=pd.read_parquet(folder/'decoded.parquet');features=pd.read_parquet(B/'data/processed/step3/student_test'/trip/'features.parquet',columns=['gps_observation_index'])
        rows.append(dict(trip_id=trip,map_href=f'maps/{trip}.html',observations=meta['display']['observations'],decoded_segments=meta['display']['decoded_segments'],
            classifier_top1=classifier.top1_accuracy,viterbi_top1=viterbi.top1_accuracy,improvement_pp=(viterbi.top1_accuracy-classifier.top1_accuracy)*100,
            classifier_continuity=classifier.direct_continuity,viterbi_continuity=viterbi.direct_continuity,
            temporal_gaps=int(decoded.break_reason.eq('temporal_gap').sum()),infeasible_breaks=int(decoded.break_reason.eq('no_feasible_transition').sum()),
            ambiguous_candidate_observations=int(features.groupby('gps_observation_index').size().gt(1).sum()),html_bytes=meta['html_bytes']))
        timings.append(dict(trip_id=trip,reused=reused,invocation_seconds=time.perf_counter()-tick,original_generation_seconds=meta['generation_seconds']))
        print(('Reused map: ' if reused else 'Generated map: ')+trip+f' ({meta["html_bytes"]/2**20:.2f} MiB)',flush=True)
    summary=pd.DataFrame(rows);summary.to_csv(output/'summary.csv',index=False);examples=choose_examples(summary);write_json(output/'examples.json',examples)
    for e in examples:
        html=f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta http-equiv="refresh" content="0; url=../maps/{e["trip_id"]}.html"><title>PathSnap example {e["trip_id"]}</title></head><body><p>{escape(e["role"])}: {escape(e["reason"])}</p><a href="../maps/{e["trip_id"]}.html">Open interactive trip map</a></body></html>'
        (output/'examples'/f'{e["trip_id"]}.html').write_text(html,encoding='utf-8')
    write_index(output,summary,examples);pd.DataFrame(timings).to_csv(output/'runtime_by_trip.csv',index=False)
    write_json(output/'run_metadata.json',dict(trips=len(rows),observations=int(summary.observations.sum()),invocation_seconds=time.perf_counter()-started,
        versions=versions,reused_maps=sum(t['reused'] for t in timings),models_retrained=False,decoder_rerun=False,reference_in_prediction=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=B/'outputs/step8');p.add_argument('--force',action='store_true',help='Rebuild maps only; never run ML/decoding');a=p.parse_args();run(a.output,a.force)
