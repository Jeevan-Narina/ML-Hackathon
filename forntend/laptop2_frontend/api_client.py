"""Frontend-only HTTP client. No backend imports, filesystem datasets or ML."""
import json,math,re
from urllib.parse import urlsplit,urlunsplit
import requests


class ClientError(Exception):
    def __init__(self,message,code='client_error',status=None):
        super().__init__(message);self.code=code;self.status=status


def normalize_url(value):
    try:
        url=urlsplit(value.strip());port=url.port
        if url.scheme not in ['http','https'] or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError()
        if port is not None and not 1<=port<=65535:raise ValueError()
        if any(c.isspace() for c in value) or '\\' in value:raise ValueError()
        return urlunsplit((url.scheme,url.netloc,url.path.rstrip('/'),'',''))
    except (ValueError,AttributeError):
        raise ClientError('Enter a backend URL such as http://localhost:8000 or http://LAPTOP1_HOST:8000.','invalid_url')


def parse_prediction_json(text):
    if len(text.encode('utf-8'))>2*1024*1024:raise ClientError('The JSON exceeds the backend’s 2 MiB limit.','invalid_request')
    def invalid_constant(value):raise ValueError('Nonfinite number')
    try:data=json.loads(text,parse_constant=invalid_constant)
    except (ValueError,TypeError):raise ClientError('Enter valid JSON with finite numbers; NaN and Infinity are not supported.','invalid_json')
    if not isinstance(data,dict) or not isinstance(data.get('gps'),list) or not isinstance(data.get('road_network'),dict):
        raise ClientError('JSON must contain a gps list and a road_network object with nodes and arcs.','invalid_request')
    return data


def require(condition):
    if not condition:raise ClientError('The backend returned an unexpected response. Check that it is the PathSnap Step 9 API.','invalid_response')


def parse_health(data):
    require(isinstance(data,dict) and data.get('status') in ['ok','degraded'])
    require(all(type(data.get(k)) is bool for k in ['server_ready','model_ready','decoder_ready']))
    require(type(data.get('demonstration_trips')) is int and type(data.get('model_loads')) is int)
    return data


def parse_trips(data):
    require(isinstance(data,dict) and isinstance(data.get('trips'),list))
    seen=set()
    for trip in data['trips']:
        require(isinstance(trip,dict) and isinstance(trip.get('trip_id'),str) and bool(re.fullmatch(r'[0-9]{8}',trip['trip_id'])))
        require(trip['trip_id'] not in seen);seen.add(trip['trip_id'])
        require(all(type(trip.get(k)) is int and trip[k]>=0 for k in ['observations','decoded_segments']))
        require(type(trip.get('map_available')) is bool)
    return data['trips']


def parse_route(data):
    require(isinstance(data,dict) and data.get('source') in ['saved_demo','live'] and data.get('model_name')=='random_forest_tuned')
    for field in ['gps','classifier_predictions','viterbi_predictions','segment_breaks','transitions']:require(isinstance(data.get(field),list))
    for field in ['classifier_roads','viterbi_route']:
        value=data.get(field);require(isinstance(value,dict) and value.get('type')=='FeatureCollection' and isinstance(value.get('features'),list))
        for feature in value['features']:
            require(isinstance(feature,dict) and feature.get('type')=='Feature' and isinstance(feature.get('properties'),dict))
            geometry=feature.get('geometry');require(isinstance(geometry,dict) and geometry.get('type') in ['Point','LineString'] and isinstance(geometry.get('coordinates'),list))
    diagnostics=data.get('diagnostics');require(isinstance(diagnostics,dict))
    require(all(type(diagnostics.get(k)) is int and diagnostics[k]>=0 for k in ['observations','candidate_rows','matched_observations','unmatched_observations','segments','infeasible_restarts','temporal_gaps','route_arc_occurrences']))
    require(isinstance(data.get('confidence_kind'),str))
    require('metrics' in data)
    if data['source']=='saved_demo':require(isinstance(data.get('trip_id'),str) and bool(re.fullmatch(r'[0-9]{8}',data['trip_id'])))
    metrics=data.get('metrics');require(metrics is None or isinstance(metrics,dict))
    if data['source']=='live':require(metrics is None)
    if metrics:
        for value in metrics.values():
            require(isinstance(value,dict))
            for name in ['top1_accuracy','direct_continuity']:
                p=value.get(name);require(p is None or (type(p) in [int,float] and math.isfinite(p) and 0<=p<=1))
            require(all(type(value.get(k)) is int for k in ['qualified_alignments','candidate_present','candidate_misses','topologically_unreachable']))
    for row in data['gps']:require(isinstance(row,dict) and type(row.get('observation_index')) is int and all(k in row for k in ['longitude','latitude','timestamp_s']))
    for field in ['classifier_predictions','viterbi_predictions']:
        for row in data[field]:
            require(isinstance(row,dict) and type(row.get('gps_observation_index')) is int and 'candidate_arc_id' in row)
            p=row.get('positive_probability');require(p is None or (type(p) in [int,float] and math.isfinite(p) and 0<=p<=1))
    return data


class ApiClient:
    def __init__(self,base_url,transport=None,timeout=(3.05,60)):
        self.base_url=normalize_url(base_url);self.transport=transport or requests.request;self.timeout=timeout

    def request(self,method,path,payload=None,health=False):
        try:
            response=self.transport(method,self.base_url+'/'+path.lstrip('/'),json=payload,timeout=self.timeout)
        except requests.Timeout:
            raise ClientError('The backend timed out. It may still be processing a prediction; check its status before retrying.','timeout')
        except requests.ConnectionError:
            raise ClientError('Cannot connect to Laptop 1. Check the backend URL, that FastAPI is running, and network access.','connection_error')
        except requests.RequestException:
            raise ClientError('The HTTP request failed. Check the backend URL and connection.','http_error')
        try:data=response.json()
        except (ValueError,requests.exceptions.JSONDecodeError):
            raise ClientError(f'The backend returned a non-JSON response (HTTP {response.status_code}).','invalid_response',response.status_code)
        if health and response.status_code==503 and isinstance(data,dict) and data.get('status')=='degraded':return data
        if not 200<=response.status_code<300:
            error=data.get('error',{}) if isinstance(data,dict) else {}
            if not isinstance(error,dict):error={}
            message=error.get('message') or f'Backend request failed (HTTP {response.status_code}).'
            detail=error.get('details',[])
            if isinstance(detail,list):
                for row in detail[:3]:
                    if isinstance(row,dict):message+=' '+'.'.join(map(str,row.get('location',[])))+': '+str(row.get('message',''))
            raise ClientError(message,str(error.get('code','http_error')),response.status_code)
        return data

    def health(self):return parse_health(self.request('GET','/health',health=True))
    def trips(self):return parse_trips(self.request('GET','/trips'))
    @staticmethod
    def trip_id(value):
        if not isinstance(value,str) or not re.fullmatch(r'[0-9]{8}',value):raise ClientError('Choose a valid trip from the backend list.','invalid_trip')
        return value
    def trip(self,trip_id):
        trip_id=self.trip_id(trip_id);data=parse_route(self.request('GET',f'/trips/{trip_id}'))
        require(data['source']=='saved_demo' and data['trip_id']==trip_id)
        return data
    def map_url(self,trip_id):return self.base_url+f'/trips/{self.trip_id(trip_id)}/map'
    def predict(self,payload):
        data=parse_route(self.request('POST','/predict',payload));require(data['source']=='live');return data
