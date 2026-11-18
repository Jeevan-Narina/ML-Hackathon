"""Frozen-model, label-free, resumable per-trip route reconstruction only."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse,importlib.metadata
os.environ.setdefault('OMP_NUM_THREADS','2')
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from pathsnap.hmm import DecoderConfig, emission_states, OBSERVATION_COLUMNS
from pathsnap.routing import DirectedRouter
from pathsnap.viterbi import decode
from pathsnap.evaluation import predict_frame,top1
from pathsnap.step2 import MemoryMonitor


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)


def run(output,model='random_forest',configuration=None):
    config=DecoderConfig(**(configuration or {}));output.mkdir(parents=True,exist_ok=True)
    config_path=output/'configuration.json';settings=dict(model_name=model,**config.to_dict(),parameter_policy='Fixed engineering defaults before student evaluation; no parameter search')
    if config_path.exists() and json.loads(config_path.read_text())!=settings:
        raise ValueError('Frozen decoder configuration changed; use a separate output directory')
    atomic_json(config_path,settings)
    if model not in ['random_forest','hist_gradient_boosting']:raise ValueError('Only frozen Step 6 pipelines supported')
    model_path=B/'models/step6'/f'{model}_tuned.joblib'
    expected=pd.read_csv(B/'models/step6/artifact_manifest.csv').set_index('model_name').loc[model,'sha256']
    assert digest(model_path)==expected
    pipeline=joblib.load(model_path)
    # Explicitly disable refitting throughout this inference-only entry point.
    Pipeline.fit=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Step 7 must never fit'))
    protected_path=output/'protected_artifacts.json'
    if not protected_path.exists():
        paths=[B/r for r in json.loads((B/'outputs/step6/protected_artifacts.json').read_text())]
        paths += [*list((B/'models/step6').glob('*')),*list((B/'outputs/step6').glob('*.csv')),
                  *list((B/'outputs/step6/student_predictions').glob('*/*')),
                  *list((B/'evaluator_workspace/processed/step2/student_test').glob('*/*.parquet')),
                  *list((B/'evaluator_workspace/processed/step3/student_test').glob('*/*.parquet')),
                  *list((B/'outputs/step5/evaluation_labels').glob('*/*.parquet')),
                  *list((B.parent/'docs/steps').glob('STEP_0[1-6]_README.md')),
                  *list((B/'data/raw/student_test').glob('*/*'))]
        atomic_json(protected_path,{p.relative_to(B.parent).as_posix():digest(p) for p in paths if p.is_file()})
    for rel,value in json.loads(protected_path.read_text()).items():assert digest(B.parent/rel)==value,rel
    code={p.name:digest(p) for p in [B/'src/pathsnap'/f'{name}.py' for name in ['routing','hmm','transitions','viterbi']]}
    trips=sorted((B/'evaluator_workspace/processed/step3/student_test').iterdir());runtimes=[];started=time.perf_counter()
    for trip in trips:
        folder=output/'trips'/trip.name;folder.mkdir(parents=True,exist_ok=True)
        paths=dict(features=trip/'features.parquet',observations=trip/'observations.parquet',
                   roads=B/'evaluator_workspace/processed/step2/student_test'/trip.name/'road_arcs.parquet')
        identity=dict(inputs={k:digest(p) for k,p in paths.items()},model_sha256=expected,configuration=settings,code_sha256=code,
                      versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','networkx','scikit-learn']})
        stamp=folder/'complete.json'
        if stamp.exists():
            info=json.loads(stamp.read_text())
            if info['identity']!=identity:raise ValueError('Trip identity changed; use a separate output directory')
            if all((folder/p).exists() and digest(folder/p)==v for p,v in info['outputs_sha256'].items()):
                runtimes.append(dict(trip_id=trip.name,reused=True,**info['runtime']))
                print('Reused verified trip:',trip.name,flush=True);continue
        tick=time.perf_counter()
        features=pd.read_parquet(paths['features']);observations=pd.read_parquet(paths['observations'])[OBSERVATION_COLUMNS]
        arcs=pd.read_parquet(paths['roads']).set_index('arc_id')
        cached=B/'evaluator_workspace/cache/student_predictions'/model/f'{trip.name}.parquet';cache_identity=cached.with_suffix('.json')
        ptick=time.perf_counter();reused_probability=False
        if cached.exists() and cache_identity.exists():
            cache=json.loads(cache_identity.read_text())['identity']
            if cache==dict(model_sha256=expected,features_sha256=identity['inputs']['features']):
                probabilities=pd.read_parquet(cached);reused_probability=True
        if not reused_probability:
            with threadpool_limits(limits=2):probabilities=predict_frame(pipeline,features)
        probability_seconds=time.perf_counter()-ptick
        states=emission_states(features,probabilities)
        with MemoryMonitor() as memory:
            rtick=time.perf_counter();router=DirectedRouter(arcs,config.routing_cutoff_m,config.max_settled_nodes,config.cache_sources)
            graph_seconds=time.perf_counter()-rtick;dtick=time.perf_counter()
            selected,links,route,diagnostics=decode(observations,states,router,config)
            decode_seconds=time.perf_counter()-dtick
        classifier=top1(probabilities).merge(features[['gps_observation_index','candidate_arc_id','position_along_arc']],on=['gps_observation_index','candidate_arc_id'],validate='one_to_one')
        frames=dict(probabilities=probabilities,classifier_selected=classifier,decoded=selected,
                    transitions=links,route_arcs=route,transition_diagnostics=diagnostics)
        for name,frame in frames.items():
            path=folder/f'{name}.parquet';tmp=path.with_suffix('.tmp');frame.to_parquet(tmp,index=False);tmp.replace(path)
        runtime=dict(observations=len(observations),candidate_rows=len(features),decoded=int(selected.status.eq('matched').sum()),
            segments=int(selected.segment_id.nunique()),unmatched=int(selected.status.eq('unmatched').sum()),
            graph_seconds=graph_seconds,decode_seconds=decode_seconds,probability_seconds=probability_seconds,
            total_seconds=time.perf_counter()-tick,peak_process_rss_mib=memory.peak/2**20,
            probabilities_reused=reused_probability,**router.stats)
        atomic_json(stamp,dict(identity=identity,runtime=runtime,outputs_sha256={f'{name}.parquet':digest(folder/f'{name}.parquet') for name in frames}))
        runtimes.append(dict(trip_id=trip.name,reused=False,**runtime))
        pd.DataFrame(runtimes).to_csv(output/'runtime_by_trip.csv',index=False)
        print(f'Decoded {trip.name}: {runtime["decoded"]}/{len(observations)} observations, {runtime["segments"]} segments, {decode_seconds:.2f}s',flush=True)
    pd.DataFrame(runtimes).to_csv(output/'runtime_by_trip.csv',index=False)
    atomic_json(output/'run_metadata.json',dict(invocation_seconds=time.perf_counter()-started,trips=len(trips),
       model_sha256=expected,code_sha256=code,refitted_models=False,reference_inputs_used=False,configuration=settings))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=B/'evaluator_workspace/outputs/step7');p.add_argument('--model',choices=['random_forest','hist_gradient_boosting'],default='random_forest');p.add_argument('--config',type=Path)
    a=p.parse_args();run(a.output,a.model,json.loads(a.config.read_text()) if a.config else None)

