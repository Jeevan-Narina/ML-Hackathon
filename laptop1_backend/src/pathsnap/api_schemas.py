"""Public API contracts; reference labels and server paths are never inputs."""
from typing import Annotated,Any,Literal
import math
from pydantic import BaseModel,ConfigDict,Field,model_validator

MAX_GPS=500
MAX_NODES=5000
MAX_ARCS=10000
MAX_BODY_BYTES=2*1024*1024
Id=Annotated[int,Field(strict=True,ge=0,le=2**31-1)]
Longitude=Annotated[float,Field(ge=-180,le=180)]
Latitude=Annotated[float,Field(ge=-90,le=90)]


class Contract(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False,strict=True)


class ObservationInput(Contract):
    observation_id:Id|None=None
    longitude:Longitude
    latitude:Latitude
    timestamp_s:float


class RoadNodeInput(Contract):
    node_id:Id
    longitude:Longitude
    latitude:Latitude


class RoadArcInput(Contract):
    arc_id:Id
    from_node_id:Id
    to_node_id:Id


class RoadContext(Contract):
    nodes:list[RoadNodeInput]=Field(min_length=2,max_length=MAX_NODES)
    arcs:list[RoadArcInput]=Field(min_length=1,max_length=MAX_ARCS)

    @model_validator(mode='after')
    def check_ids(self):
        ids={n.node_id for n in self.nodes}
        if len(ids)!=len(self.nodes):raise ValueError('Road node IDs must be unique')
        if len({a.arc_id for a in self.arcs})!=len(self.arcs):raise ValueError('Road arc IDs must be unique')
        if any(a.from_node_id not in ids or a.to_node_id not in ids for a in self.arcs):raise ValueError('Every arc endpoint must refer to a supplied node')
        return self


class PredictRequest(Contract):
    gps:list[ObservationInput]=Field(min_length=2,max_length=MAX_GPS)
    road_network:RoadContext

    @model_validator(mode='after')
    def check_trajectory(self):
        ids=[p.observation_id for p in self.gps]
        if any(i is None for i in ids) and not all(i is None for i in ids):raise ValueError('Provide all observation IDs or omit all of them')
        if ids[0] is not None and len(set(ids))!=len(ids):raise ValueError('Observation IDs must be unique')
        if any(b.timestamp_s<=a.timestamp_s for a,b in zip(self.gps,self.gps[1:])):raise ValueError('GPS timestamps must increase in supplied sequence order')
        return self


class GPSOutput(Contract):
    observation_index:int
    longitude:float|None
    latitude:float|None
    timestamp_s:float|None


class CandidateSelection(Contract):
    gps_observation_index:int
    candidate_arc_id:int|None
    position_along_arc:float|None=None
    positive_probability:Annotated[float,Field(ge=0,le=1)]|None=None
    segment_id:int|None=None
    status:Literal['matched','unmatched']='matched'
    break_reason:str=''


class Geometry(Contract):
    type:Literal['Point','LineString']
    coordinates:list[Any]


class GeoFeature(Contract):
    type:Literal['Feature']='Feature'
    id:str
    geometry:Geometry
    properties:dict[str,Any]


class GeoCollection(Contract):
    type:Literal['FeatureCollection']='FeatureCollection'
    features:list[GeoFeature]=Field(default_factory=list)


class SegmentBreak(Contract):
    observation_index:int
    segment_id:int|None
    reason:str


class TransitionOutput(Contract):
    from_observation:int
    to_observation:int
    from_arc:int
    to_arc:int
    segment_id:int
    dt_s:float
    gps_displacement_m:float
    network_distance_m:float
    transition_log_score:float
    intermediate_arcs:int
    route_arcs:list[int]


class Diagnostics(Contract):
    observations:int
    candidate_rows:int
    matched_observations:int
    unmatched_observations:int
    segments:int
    infeasible_restarts:int
    temporal_gaps:int
    route_arc_occurrences:int
    candidate_status_counts:dict[str,int]=Field(default_factory=dict)
    routing_statistics:dict[str,int]=Field(default_factory=dict)


class EvaluationMetrics(Contract):
    top1_accuracy:float|None
    qualified_alignments:int
    candidate_present:int
    candidate_misses:int
    direct_continuity:float|None
    topologically_unreachable:int
    evaluation_scope:str='Existing qualified student-test reference labels and candidate availability; not independent full-route truth'


class RouteResponse(Contract):
    source:Literal['saved_demo','live']
    trip_id:str|None=None
    model_name:Literal['random_forest_tuned']='random_forest_tuned'
    gps:list[GPSOutput]
    classifier_predictions:list[CandidateSelection]
    viterbi_predictions:list[CandidateSelection]
    classifier_roads:GeoCollection
    viterbi_route:GeoCollection
    segment_breaks:list[SegmentBreak]
    transitions:list[TransitionOutput]
    diagnostics:Diagnostics
    confidence_kind:str='positive_probability is an independent binary RF score, not a Viterbi posterior or calibrated route confidence'
    metrics:dict[str,EvaluationMetrics]|None=None
    timings_seconds:dict[str,float]=Field(default_factory=dict)


class HealthResponse(Contract):
    status:Literal['ok','degraded']
    server_ready:bool
    model_ready:bool
    decoder_ready:bool
    demonstration_trips:int
    model_loads:int
    readiness_error:str|None=None


class TripInfo(Contract):
    trip_id:str
    observations:int
    decoded_segments:int
    map_available:bool


class TripList(Contract):
    trips:list[TripInfo]


class APIError(Contract):
    code:str
    message:str
    details:list[dict[str,Any]]=Field(default_factory=list)


class ErrorResponse(Contract):
    error:APIError
