"""Standalone Laptop 2 UI. All data/model/map access goes through HTTP."""
from pathlib import Path
import os
import streamlit as st
from api_client import ApiClient,ClientError,normalize_url,parse_prediction_json


def clear_results():
    for key in ['health','trips','trip_result','trip_map_url','prediction_result']:
        st.session_state.pop(key,None)


def show_result(result,key):
    d=result['diagnostics'];a,b,c=st.columns(3)
    a.metric('GPS observations',d['observations']);b.metric('Matched observations',d['matched_observations']);c.metric('Decoded segments',d['segments'])
    metrics=result['metrics']
    if metrics:
        st.subheader('Available evaluation metrics')
        rows=[]
        for name,m in metrics.items():
            rows.append({'variant':name,**{field:m[field] for field in ['top1_accuracy','direct_continuity','qualified_alignments','candidate_present','candidate_misses','topologically_unreachable']}})
        st.dataframe(rows,hide_index=True)
        st.caption('Accuracy is conditional on qualified reference labels and candidate availability. Reference routes are evaluation-only.')
    else:st.info('No evaluation metrics are available for this prediction. No accuracy is inferred from the route.')
    with st.expander('Reconstructed route and diagnostics',expanded=result['source']=='live'):
        st.write('Reconstructed route (returned GeoJSON)');st.json(result['viterbi_route'],expanded=False)
        st.write('Diagnostics');st.json(d)
        if result['segment_breaks']:st.write('Segment breaks');st.dataframe(result['segment_breaks'],hide_index=True)
        st.caption(result['confidence_kind'])
        st.write('Viterbi-selected candidates (up to 200 rows)');st.dataframe(result['viterbi_predictions'][:200],hide_index=True)
        st.write('Classifier-only selections (up to 200 rows)');st.dataframe(result['classifier_predictions'][:200],hide_index=True)


def main():
    st.set_page_config(page_title='PathSnap',layout='wide')
    st.title('PathSnap')
    st.caption('Compare saved road reconstructions or submit GPS with its directed road network to Laptop 1.')
    st.session_state.setdefault('backend_url',os.environ.get('PATHSNAP_BACKEND_URL','http://localhost:8000'))
    with st.sidebar:
        st.header('Backend connection')
        with st.form('connection'):
            address=st.text_input('Backend URL',value=st.session_state['backend_url'],key='backend_url_input')
            connect=st.form_submit_button('Connect / refresh')
        st.caption('On Laptop 2, replace localhost with Laptop 1’s reachable network address.')
    if connect:
        clear_results()
        try:
            st.session_state['backend_url']=normalize_url(address)
            client=ApiClient(st.session_state['backend_url'])
            with st.spinner('Connecting to backend…'):
                st.session_state['health']=client.health();st.session_state['trips']=client.trips()
        except ClientError as error:st.error(str(error))
    health=st.session_state.get('health')
    with st.sidebar:
        if not health:st.info('Not connected. Enter the backend URL and connect.')
        elif health['model_ready'] and health['decoder_ready']:st.success('Connected · model and decoder ready')
        else:st.warning('Backend reachable; live model or decoder is not ready. Saved trips may still be available.')
    client=ApiClient(st.session_state['backend_url']) if health else None
    saved,predict=st.tabs(['Saved trips','New prediction'])
    with saved:
        trips=st.session_state.get('trips',[])
        if not trips:st.info('Connect to a backend with available demonstration trips.')
        else:
            selected=st.selectbox('Demonstration trip',[t['trip_id'] for t in trips])
            if st.button('Load trip'):
                st.session_state.pop('trip_result',None);st.session_state.pop('trip_map_url',None)
                try:
                    with st.spinner('Loading saved trip…'):result=client.trip(selected)
                    st.session_state['trip_result']=result
                    if next(t for t in trips if t['trip_id']==selected)['map_available']:st.session_state['trip_map_url']=client.map_url(selected)
                except ClientError as error:st.error(str(error))
            result=st.session_state.get('trip_result')
            if result:
                st.subheader('Trip '+result['trip_id']);show_result(result,'trip')
                url=st.session_state.get('trip_map_url')
                if url:
                    st.link_button('Open interactive map in a new tab',url)
                    st.iframe(url,height=760)
                    st.caption('Existing backend map: use its Before/After buttons and layer control. The browser must also reach the backend URL; OpenStreetMap tiles require internet.')
                else:st.warning('No saved interactive map is available for this trip.')
    with predict:
        st.subheader('New GPS trajectory')
        st.caption('Provide GPS and explicit road_network nodes/arcs. Up to 500 GPS points, 5,000 nodes, 10,000 arcs and 2 MiB. The backend validates the full contract; no reference labels are accepted.')
        example=Path(__file__).with_name('example_request.json').read_text(encoding='utf-8')
        with st.form('new_prediction'):
            text=st.text_area('Prediction request JSON',value=example,height=330,key='prediction_json')
            submitted=st.form_submit_button('Predict',disabled=not bool(health and health['model_ready'] and health['decoder_ready']))
        if submitted:
            st.session_state.pop('prediction_result',None)
            try:
                payload=parse_prediction_json(text)
                with st.spinner('Laptop 1 is reconstructing the route…'):st.session_state['prediction_result']=client.predict(payload)
            except ClientError as error:st.error(str(error))
        result=st.session_state.get('prediction_result')
        if result:show_result(result,'prediction')


if __name__=='__main__':main()
