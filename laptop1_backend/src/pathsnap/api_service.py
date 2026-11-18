"""Inference-only service; demo artifacts are read, never recomputed."""
from pathlib import Path
import hashlib,json,threading,time,logging,re
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from .api_schemas import RouteResponse,MAX_GPS
from .features import FEATURES,IDENTIFIERS,engineer_features
from .candidates import CandidateConfig,generate_candidates
from .preprocessing import clean_observations
from .road_network import RoadNetwork
from .evaluation import predict_frame,top1
from .hmm import DecoderConfig,emission_states,OBSERVATION_COLUMNS
from .routing import DirectedRouter
from .viterbi import decode
from .map_visualization import arc_feature,reconstructed_features,collection

LOGGER=logging.getLogger('pathsnap.api')


def records(frame):return json.loads(frame.to_json(orient='records'))


def selections(frame):
    fields=['gps_observation_index','candidate_arc_id','position_along_arc','positive_probability','segment_id','status','break_reason']
    return records(frame[[k for k in fields if k in frame]])


def gps_records(observations):
    frame=observations[['longitude','latitude','timestamp_s']].copy()
    frame.insert(0,'observation_index',observations.index)
    return records(frame)


def summary(decoded,candidates,routes,status_counts=None,routing_stats=None):
    return dict(observations=len(decoded),candidate_rows=candidates,matched_observations=int(decoded.status.eq('matched').sum()),
        unmatched_observations=int(decoded.status.eq('unmatched').sum()),segments=int(decoded.segment_id.nunique()),
        infeasible_restarts=int(decoded.break_reason.eq('no_feasible_transition').sum()),temporal_gaps=int(decoded.break_reason.eq('temporal_gap').sum()),
        route_arc_occurrences=len(routes),candidate_status_counts=status_counts or {},routing_statistics=routing_stats or {})


def breaks(decoded):
    frame=decoded.loc[decoded.break_reason.ne(''),['gps_observation_index','segment_id','break_reason']]
    return records(frame.rename(columns=dict(gps_observation_index='observation_index',break_reason='reason')))


class PredictionBusy(Exception):pass


class ApiService:
    def __init__(self,backend):
        self.backend=Path(backend);self.pipeline=None;self.decoder_config=None;self.readiness_error=None
        self.model_loads=0;self.predict_lock=threading.Lock();self.demo_cache={};self.demo_metrics=None;self.trip_rows={}
        try:
            settings=json.loads((self.backend/'outputs/step7/configuration.json').read_text())
            self.decoder_config=DecoderConfig(**{k:v for k,v in settings.items() if k in DecoderConfig.__dataclass_fields__})
        except Exception:
            self.readiness_error='decoder_configuration_unavailable';LOGGER.exception('Decoder startup validation failed')
        try:
            path=self.backend/'models/step6/random_forest_tuned.joblib'
            manifest=pd.read_csv(self.backend/'models/step6/artifact_manifest.csv').set_index('model_name')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest.loc['random_forest','sha256']:raise ValueError('Model checksum mismatch')
            schema=json.loads((self.backend/'models/step6/feature_schema.json').read_text())
            if schema['ordered_features']!=FEATURES:raise ValueError('Model feature order mismatch')
            model=joblib.load(path)
            if list(model.named_steps['finite'].feature_names_in_)!=FEATURES:raise ValueError('Pipeline feature order mismatch')
            self.pipeline=model;self.model_loads=1
        except Exception:
            self.readiness_error='model_unavailable';LOGGER.exception('Model startup validation failed')
        try:
            data=pd.read_csv(self.backend/'outputs/step8/summary.csv',dtype={'trip_id':str})
            self.demo_metrics=pd.read_csv(self.backend/'outputs/step7/per_trip_comparison.csv',dtype={'trip_id':str})
            for row in data.to_dict(orient='records'):
                trip=row['trip_id']
                if re.fullmatch(r'\d{8}',trip) and (self.backend/'outputs/step7/trips'/trip/'complete.json').is_file():self.trip_rows[trip]=row
        except (OSError,ValueError):LOGGER.exception('Demonstration catalog unavailable')
        LOGGER.info('Startup ready model=%s decoder=%s demo_trips=%d',self.pipeline is not None,self.decoder_config is not None,len(self.trip_rows))

    def demo(self,trip):
        if trip in self.demo_cache:return self.demo_cache[trip]
        b=self.backend;folder=b/'outputs/step7/trips'/trip
        stamp=json.loads((folder/'complete.json').read_text())
        for name,expected in stamp['outputs_sha256'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:raise ValueError('Saved demo checksum mismatch')
        obs=pd.read_parquet(b/'data/processed/step3/student_test'/trip/'observations.parquet')
        classifier=pd.read_parquet(folder/'classifier_selected.parquet');decoded=pd.read_parquet(folder/'decoded.parquet')
        transitions=pd.read_parquet(folder/'transitions.parquet');routes=pd.read_parquet(folder/'route_arcs.parquet')
        data_path=b/'outputs/step8/data'/f'{trip}.json'
        display_stamp=json.loads((b/'outputs/step8/metadata'/f'{trip}.json').read_text())
        if hashlib.sha256(data_path.read_bytes()).hexdigest()!=display_stamp['data_sha256']:raise ValueError('Saved display checksum mismatch')
        payload=json.loads(data_path.read_text())
        metrics={}
        fields=['top1_accuracy','qualified_alignments','candidate_present','candidate_misses','direct_continuity','topologically_unreachable']
        for variant in ['classifier','viterbi']:
            rows=self.demo_metrics.loc[(self.demo_metrics.trip_id==trip)&(self.demo_metrics.variant==variant)]
            if len(rows)==1:metrics[variant]=json.loads(rows[fields].to_json(orient='records'))[0]
        response=RouteResponse(source='saved_demo',trip_id=trip,gps=gps_records(obs),classifier_predictions=selections(classifier),
            viterbi_predictions=selections(decoded),classifier_roads=payload['layers']['classifier'],viterbi_route=payload['layers']['route'],
            segment_breaks=breaks(decoded),transitions=records(transitions),metrics=metrics or None,
            diagnostics=summary(decoded,stamp['runtime']['candidate_rows'],routes),timings_seconds=dict(saved_decode=stamp['runtime']['decode_seconds']))
        self.demo_cache[trip]=response
        return response

    def predict(self,request):
        if not self.predict_lock.acquire(blocking=False):raise PredictionBusy()
        try:return self._predict(request)
        finally:self.predict_lock.release()

    def _predict(self,request):
        started=time.perf_counter();points=request.gps
        obs=pd.DataFrame([dict(longitude=p.longitude,latitude=p.latitude,timestamp_s=p.timestamp_s) for p in points],
            index=pd.Index([p.observation_id if p.observation_id is not None else i for i,p in enumerate(points)],name='observation_id'))
        nodes=pd.DataFrame([p.model_dump() for p in request.road_network.nodes]).set_index('node_id')
        arcs=pd.DataFrame([a.model_dump() for a in request.road_network.arcs]).set_index('arc_id')
        config=CandidateConfig(batch_size=64)
        network=RoadNetwork(nodes,arcs)
        if not len(network.geometries):raise ValueError('Road context has no usable nondegenerate arcs')
        cleaned=clean_observations(obs)
        batches=list(generate_candidates(network,obs,'student_test','request',config))
        candidates=pd.concat([a for a,_ in batches],ignore_index=True);statuses=pd.concat([s for _,s in batches],ignore_index=True)
        features=engineer_features(candidates,cleaned,network);preprocessing_seconds=time.perf_counter()-started
        tick=time.perf_counter()
        if len(features):
            with threadpool_limits(limits=2):probabilities=predict_frame(self.pipeline,features)
        else:probabilities=features[IDENTIFIERS].assign(positive_probability=pd.Series(dtype='float64'))
        model_seconds=time.perf_counter()-tick
        states=emission_states(features,probabilities);cfg=self.decoder_config
        router=DirectedRouter(network.arcs,cfg.routing_cutoff_m,cfg.max_settled_nodes,cfg.cache_sources)
        tick=time.perf_counter();decoded,links,routes,diagnostics=decode(cleaned[OBSERVATION_COLUMNS],states,router,cfg)
        decoder_seconds=time.perf_counter()-tick
        if len(routes)>50000:raise ValueError('Reconstructed output exceeds route budget; split the trajectory')
        classifier=top1(probabilities).merge(features[['gps_observation_index','candidate_arc_id','position_along_arc']],on=['gps_observation_index','candidate_arc_id'],validate='one_to_one')
        roads=[arc_feature(network.arcs,nodes,int(aid),f'classifier-{aid}',detail='Frozen RF selected full candidate arc') for aid in sorted(classifier.candidate_arc_id.unique())]
        geometry=reconstructed_features(routes,decoded,network.arcs,nodes)
        response=RouteResponse(source='live',gps=gps_records(cleaned),classifier_predictions=selections(classifier),viterbi_predictions=selections(decoded),
            classifier_roads=collection(roads),viterbi_route=collection(geometry),segment_breaks=breaks(decoded),transitions=records(links),metrics=None,
            diagnostics=summary(decoded,len(candidates),routes,statuses.status.value_counts().astype(int).to_dict(),router.stats),
            timings_seconds=dict(preprocessing=preprocessing_seconds,classifier=model_seconds,decoder=decoder_seconds,total=time.perf_counter()-started))
        LOGGER.info('Prediction observations=%d candidates=%d segments=%d unmatched=%d seconds=%.3f',len(obs),len(candidates),response.diagnostics.segments,response.diagnostics.unmatched_observations,response.timings_seconds['total'])
        return response
