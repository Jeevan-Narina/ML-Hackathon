import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import requests
from api_client import ApiClient,ClientError,normalize_url,parse_health,parse_trips,parse_route,parse_prediction_json


def route(source='saved_demo'):
    return dict(source=source,trip_id='00000065' if source=='saved_demo' else None,model_name='random_forest_tuned',
        gps=[dict(observation_index=0,longitude=77.,latitude=12.,timestamp_s=0.)],
        classifier_predictions=[dict(gps_observation_index=0,candidate_arc_id=900,positive_probability=.8)],
        viterbi_predictions=[dict(gps_observation_index=0,candidate_arc_id=900,positive_probability=.8)],
        classifier_roads=dict(type='FeatureCollection',features=[]),viterbi_route=dict(type='FeatureCollection',features=[]),
        segment_breaks=[],transitions=[],diagnostics=dict(observations=1,candidate_rows=1,matched_observations=1,
            unmatched_observations=0,segments=1,infeasible_restarts=0,temporal_gaps=0,route_arc_occurrences=0),
        confidence_kind='Independent binary classifier scores, not route confidence.',metrics=None,timings_seconds={})


def health(ready=True):
    return dict(status='ok' if ready else 'degraded',server_ready=True,model_ready=ready,decoder_ready=ready,demonstration_trips=1,model_loads=int(ready))


def catalog():return {'trips':[dict(trip_id='00000065',observations=1,decoded_segments=1,map_available=True)]}


def response(data,status=200):
    r=Mock();r.status_code=status;r.json.return_value=data
    return r


class HttpClientTests(unittest.TestCase):
    def test_urls_and_base_prefix(self):
        self.assertEqual(normalize_url('http://localhost:8000/'),'http://localhost:8000')
        self.assertEqual(ApiClient('https://laptop1.test/api').map_url('00000065'),'https://laptop1.test/api/trips/00000065/map')
        for url in ['file:///etc/passwd','localhost:8000','http://user:pass@host','http://host?x=1','http://host/#fragment','http://host:0','http://bad host','http://host:wrong']:
            with self.subTest(url=url),self.assertRaises(ClientError):normalize_url(url)

    def test_exact_endpoint_paths_and_post_payload(self):
        transport=Mock(side_effect=[response(health()),response(catalog()),response(route()),response(route('live'))])
        c=ApiClient('http://laptop1.test:8000',transport=transport)
        self.assertTrue(c.health()['model_ready']);self.assertEqual(c.trips()[0]['trip_id'],'00000065');c.trip('00000065')
        payload=json.loads((Path(__file__).resolve().parents[1]/'example_request.json').read_text());self.assertIsNone(c.predict(payload)['metrics'])
        self.assertEqual([call.args[:2] for call in transport.call_args_list],[
            ('GET','http://laptop1.test:8000/health'),('GET','http://laptop1.test:8000/trips'),
            ('GET','http://laptop1.test:8000/trips/00000065'),('POST','http://laptop1.test:8000/predict')])
        self.assertEqual(transport.call_args.kwargs['json'],payload);self.assertEqual(transport.call_args.kwargs['timeout'],(3.05,60))

    def test_timeout_connection_and_other_http_failures(self):
        for exc,code in [(requests.Timeout(),'timeout'),(requests.ConnectionError(),'connection_error'),(requests.RequestException(),'http_error')]:
            with self.subTest(code=code):
                with self.assertRaises(ClientError) as ctx:ApiClient('http://localhost',transport=Mock(side_effect=exc)).health()
                self.assertEqual(ctx.exception.code,code)

    def test_degraded_health_is_parsed(self):
        c=ApiClient('http://localhost',transport=Mock(return_value=response(health(False),503)))
        self.assertEqual(c.health()['status'],'degraded')

    def test_validation_and_busy_errors_are_friendly(self):
        for code,status in [('validation_error',422),('prediction_busy',503)]:
            data={'error':{'code':code,'message':'Check request','details':[dict(location=['body','road_network'],message='Field required')]}}
            with self.assertRaises(ClientError) as ctx:ApiClient('http://localhost',transport=Mock(return_value=response(data,status))).predict({})
            self.assertEqual(ctx.exception.status,status);self.assertIn('body.road_network',str(ctx.exception));self.assertEqual(ctx.exception.code,code)

    def test_non_json_and_nonstandard_errors(self):
        r=response({});r.json.side_effect=ValueError('Bad JSON')
        with self.assertRaises(ClientError):ApiClient('http://localhost',transport=Mock(return_value=r)).health()
        for data in [None,{'error':'unavailable'}]:
            with self.assertRaises(ClientError):ApiClient('http://localhost',transport=Mock(return_value=response(data,502))).health()

    def test_trip_path_injection_is_rejected_before_http(self):
        transport=Mock();c=ApiClient('http://localhost',transport=transport)
        for value in ['../models','00000065/map','abcdefgh',None]:
            with self.assertRaises(ClientError):c.trip(value)
        transport.assert_not_called()

    def test_json_editor_preserves_payload_and_rejects_invalid_numbers(self):
        text=(Path(__file__).resolve().parents[1]/'example_request.json').read_text()
        self.assertEqual(parse_prediction_json(text),json.loads(text))
        for text in ['not json','[]','{"gps": [], "road_network": NaN}','{}','x'*(2*1024*1024+1)]:
            with self.assertRaises(ClientError):parse_prediction_json(text)

    def test_unexpected_health_and_catalog_are_rejected(self):
        for value in [{},[],{'status':'ok','server_ready':'yes'},health()|{'model_ready':1}]:
            with self.assertRaises(ClientError):parse_health(value)
        for value in [{}, {'trips':[{}]},catalog()|{'trips':[dict(trip_id='../bad',observations=1,decoded_segments=1,map_available=True)]}]:
            with self.assertRaises(ClientError):parse_trips(value)

    def test_route_parsing_does_not_invent_metrics(self):
        d=route('live');self.assertIsNone(parse_route(d)['metrics'])
        for key,value in [('metrics',{'classifier':{}}),('viterbi_route',{}),('diagnostics',{}),('model_name','wrong')]:
            with self.subTest(key=key),self.assertRaises(ClientError):parse_route(d|{key:value})
        d=route();d.pop('metrics')
        with self.assertRaises(ClientError):parse_route(d)

    def test_saved_evaluation_metrics_are_preserved(self):
        d=route();m=dict(top1_accuracy=.8,direct_continuity=.9,qualified_alignments=10,candidate_present=10,candidate_misses=0,topologically_unreachable=0)
        d['metrics']={'classifier':m};self.assertIs(parse_route(d),d);self.assertEqual(d['metrics']['classifier']['top1_accuracy'],.8)
        d=copy.deepcopy(d);d['metrics']['classifier']['top1_accuracy']=float('nan')
        with self.assertRaises(ClientError):parse_route(d)

    def test_wrong_trip_or_prediction_source_is_rejected(self):
        c=ApiClient('http://localhost',transport=Mock(return_value=response(route()|{'trip_id':'00000052'})))
        with self.assertRaises(ClientError):c.trip('00000065')
        c=ApiClient('http://localhost',transport=Mock(return_value=response(route())))
        with self.assertRaises(ClientError):c.predict({})


if __name__=='__main__':unittest.main()
