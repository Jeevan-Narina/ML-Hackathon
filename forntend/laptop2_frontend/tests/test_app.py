import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from test_client import route,health,catalog
from api_client import ClientError

APP=Path(__file__).resolve().parents[1]/'app.py'


class AppTests(unittest.TestCase):
    def test_connect_load_trip_embed_and_predict_only_on_submit(self):
        with patch('api_client.ApiClient') as cls:
            c=cls.return_value;c.health.return_value=health();c.trips.return_value=catalog()['trips'];c.trip.return_value=route();c.map_url.return_value='http://laptop1.test:8000/trips/00000065/map';c.predict.return_value=route('live')
            at=AppTest.from_file(str(APP)).run(timeout=20)
            self.assertFalse(at.exception);self.assertEqual(at.session_state['backend_url'],'http://localhost:8000');c.health.assert_not_called()
            at.text_input(key='backend_url_input').set_value('http://laptop1.test:8000')
            next(b for b in at.button if b.label=='Connect / refresh').click().run(timeout=20)
            self.assertFalse(at.exception);self.assertEqual(at.session_state['backend_url'],'http://laptop1.test:8000');c.health.assert_called_once()
            next(b for b in at.button if b.label=='Load trip').click().run(timeout=20)
            self.assertFalse(at.exception);c.trip.assert_called_once_with('00000065')
            self.assertEqual(at.get('iframe')[0].proto.src,'http://laptop1.test:8000/trips/00000065/map')
            next(b for b in at.button if b.label=='Predict').click().run(timeout=20)
            self.assertFalse(at.exception);c.predict.assert_called_once();self.assertIn('road_network',c.predict.call_args.args[0])
            at.run(timeout=20);c.predict.assert_called_once()

    def test_connection_failure_is_shown_without_crash(self):
        with patch('api_client.ApiClient') as cls:
            cls.return_value.health.side_effect=ClientError('Cannot connect to Laptop 1.','connection_error')
            at=AppTest.from_file(str(APP)).run(timeout=20);next(b for b in at.button if b.label=='Connect / refresh').click().run(timeout=20)
            self.assertFalse(at.exception);self.assertTrue(any('Cannot connect' in e.value for e in at.error))

    def test_backend_change_clears_stale_results(self):
        with patch('api_client.ApiClient') as cls:
            c=cls.return_value;c.health.return_value=health();c.trips.return_value=catalog()['trips'];c.trip.return_value=route();c.map_url.return_value='http://localhost:8000/trips/00000065/map'
            at=AppTest.from_file(str(APP)).run(timeout=20);next(b for b in at.button if b.label=='Connect / refresh').click().run(timeout=20)
            next(b for b in at.button if b.label=='Load trip').click().run(timeout=20)
            self.assertIn('trip_result',at.session_state)
            c.health.side_effect=ClientError('Cannot connect','connection_error')
            at.text_input(key='backend_url_input').set_value('http://other.test:8000');next(b for b in at.button if b.label=='Connect / refresh').click().run(timeout=20)
            self.assertFalse(at.exception);self.assertNotIn('trip_result',at.session_state);self.assertFalse(at.get('iframe'))


if __name__=='__main__':unittest.main()
