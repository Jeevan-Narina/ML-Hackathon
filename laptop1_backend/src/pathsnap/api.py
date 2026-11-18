"""Laptop 1 FastAPI application: cached demonstrations and bounded inference."""
from contextlib import asynccontextmanager
from pathlib import Path as FilePath
from typing import Annotated
import json,logging,os,re,time,uuid
from fastapi import FastAPI,HTTPException,Path,Request,Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse,HTMLResponse,FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from .api_schemas import (PredictRequest,RouteResponse,HealthResponse,TripList,TripInfo,ErrorResponse,MAX_BODY_BYTES)
from .api_service import ApiService,PredictionBusy

B=FilePath(__file__).resolve().parents[2]
LOGGER=logging.getLogger('pathsnap.api')
TripId=Annotated[str,Path(pattern=r'^\d{8}$',description='An ID from GET /trips; never a filesystem path')]


def error(status,code,message,details=None,headers=None):
    return JSONResponse(status_code=status,content={'error':dict(code=code,message=message,details=details or [])},headers=headers)


class RequestBounds:
    """Bound actual streamed POST bytes before JSON validation; log no GPS bodies."""
    def __init__(self,app,max_bytes=MAX_BODY_BYTES):self.app=app;self.max_bytes=max_bytes
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        started=time.perf_counter();request_id=uuid.uuid4().hex;status=500
        async def observed(message):
            nonlocal status
            if message['type']=='http.response.start':
                status=message['status'];message.setdefault('headers',[]).append((b'x-request-id',request_id.encode()))
            await send(message)
        if scope['method']=='POST':
            headers=dict(scope['headers']);length=headers.get(b'content-length')
            if length is not None:
                try:
                    if int(length)<0:raise ValueError()
                    too_large=int(length)>self.max_bytes
                except ValueError:return await error(400,'invalid_length','Invalid Content-Length')(scope,receive,observed)
                if too_large:return await error(413,'payload_too_large','Request exceeds the 2 MiB body limit')(scope,receive,observed)
            chunks=[];size=0
            while True:
                message=await receive()
                if message['type']=='http.disconnect':return
                block=message.get('body',b'');size+=len(block)
                if size>self.max_bytes:return await error(413,'payload_too_large','Request exceeds the 2 MiB body limit')(scope,receive,observed)
                chunks.append(block)
                if not message.get('more_body',False):break
            body=b''.join(chunks);del chunks;sent=False
            original_receive=receive
            async def replay():
                nonlocal sent
                if not sent:sent=True;return dict(type='http.request',body=body,more_body=False)
                return await original_receive()
            receive=replay
        try:await self.app(scope,receive,observed)
        finally:
            route=getattr(scope.get('route'),'path','unmatched')
            LOGGER.info('request_id=%s method=%s route=%s status=%s seconds=%.3f',request_id,scope['method'],route,status,time.perf_counter()-started)


def create_app(backend=B,cors_origins=None):
    backend=FilePath(backend)
    @asynccontextmanager
    async def lifespan(application):
        application.state.service=ApiService(backend)
        yield
        application.state.service.demo_cache.clear()
        application.state.service.pipeline=None
    app=FastAPI(title='PathSnap Laptop 1 API',version='0.9.0',lifespan=lifespan,
        description='Saved student demonstrations and reference-free RF + HMM/Viterbi inference. Coordinates in GeoJSON are longitude, latitude.',
        responses={code:{'model':ErrorResponse} for code in [400,404,413,422,500,503]})
    app.add_middleware(RequestBounds)
    origins=cors_origins if cors_origins is not None else [o.strip() for o in os.environ.get('PATHSNAP_CORS_ORIGINS','http://localhost:8501').split(',') if o.strip()]
    app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=False,allow_methods=['GET','POST','OPTIONS'],allow_headers=['Content-Type'],expose_headers=['X-Request-ID'])
    app.mount('/map-assets',StaticFiles(directory=backend/'outputs/step8/assets',check_dir=False),name='map_assets')

    @app.exception_handler(RequestValidationError)
    async def invalid(request,exc):
        details=[dict(location=list(e['loc']),message=e['msg'],type=e['type']) for e in exc.errors()[:20]]
        return error(422,'validation_error','Request validation failed',details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request,exc):
        detail=exc.detail if isinstance(exc.detail,dict) else dict(code='http_error',message='Request could not be served')
        return error(exc.status_code,detail.get('code','http_error'),detail.get('message','Request could not be served'),headers=exc.headers)

    @app.exception_handler(Exception)
    async def unexpected(request,exc):
        LOGGER.exception('Unhandled API failure')
        return error(500,'internal_error','Internal processing failure; consult the server request log')

    def get_service(request):
        service=getattr(request.app.state,'service',None)
        if service is None:raise HTTPException(503,detail=dict(code='not_ready',message='Server startup is not complete'))
        return service

    def demo_service(request,trip_id):
        service=get_service(request)
        if trip_id not in service.trip_rows:raise HTTPException(404,detail=dict(code='trip_not_found',message='Unknown demonstration trip'))
        return service

    @app.get('/health',response_model=HealthResponse,responses={503:{'model':HealthResponse,'description':'Server responsive; model or decoder unavailable'}})
    def health(request:Request,response:Response):
        s=get_service(request);ready=s.pipeline is not None and s.decoder_config is not None
        if not ready:response.status_code=503
        return dict(status='ok' if ready else 'degraded',server_ready=True,model_ready=s.pipeline is not None,
            decoder_ready=s.decoder_config is not None,demonstration_trips=len(s.trip_rows),model_loads=s.model_loads,readiness_error=s.readiness_error)

    @app.get('/trips',response_model=TripList)
    def trips(request:Request):
        s=get_service(request)
        return dict(trips=[TripInfo(trip_id=trip,observations=int(row['observations']),decoded_segments=int(row['decoded_segments']),
            map_available=(backend/'outputs/step8/maps'/f'{trip}.html').is_file()) for trip,row in sorted(s.trip_rows.items())])

    @app.get('/trips/{trip_id}',response_model=RouteResponse)
    def trip_detail(trip_id:TripId,request:Request):
        s=demo_service(request,trip_id)
        try:return s.demo(trip_id)
        except (OSError,ValueError,KeyError):
            LOGGER.exception('Saved demonstration unavailable')
            raise HTTPException(503,detail=dict(code='demo_unavailable',message='Saved demonstration artifacts are unavailable or invalid'))

    @app.post('/predict',response_model=RouteResponse)
    def predict(payload:PredictRequest,request:Request):
        s=get_service(request)
        if s.pipeline is None or s.decoder_config is None:raise HTTPException(503,detail=dict(code='not_ready',message='Model or decoder is unavailable'))
        try:return s.predict(payload)
        except PredictionBusy:raise HTTPException(503,detail=dict(code='prediction_busy',message='Another prediction is running; retry shortly'),headers={'Retry-After':'1'})
        except ValueError:
            LOGGER.exception('Prediction context rejected')
            raise HTTPException(422,detail=dict(code='unsupported_context',message='Road geometry or reconstructed output exceeds supported constraints; check context or split the trajectory'))

    def prefix(request):return request.scope.get('root_path','').rstrip('/')

    @app.get('/trips/{trip_id}/map',response_class=HTMLResponse)
    def trip_map(trip_id:TripId,request:Request):
        demo_service(request,trip_id)
        try:html=(backend/'outputs/step8/maps'/f'{trip_id}.html').read_text(encoding='utf-8')
        except OSError:raise HTTPException(503,detail=dict(code='map_unavailable',message='Saved map is unavailable'))
        html=html.replace('../assets/',prefix(request)+'/map-assets/').replace('../index.html',prefix(request)+'/maps')
        return HTMLResponse(html,headers={'X-PathSnap-Reference-Overlay':'evaluation-only-student-demo'})

    @app.get('/maps',response_class=HTMLResponse,include_in_schema=False)
    def map_index(request:Request):
        try:html=(backend/'outputs/step8/index.html').read_text(encoding='utf-8')
        except OSError:raise HTTPException(503,detail=dict(code='map_unavailable',message='Saved map index is unavailable'))
        html=re.sub(r'href="(?:maps|examples)/(\d{8})\.html"',lambda m:f'href="{prefix(request)}/trips/{m[1]}/map"',html)
        return HTMLResponse(html.replace('href="summary.csv"',f'href="{prefix(request)}/maps/summary.csv"'))

    @app.get('/maps/summary.csv',include_in_schema=False)
    def map_summary():
        path=backend/'outputs/step8/summary.csv'
        if not path.is_file():raise HTTPException(503,detail=dict(code='map_unavailable',message='Saved summary unavailable'))
        return FileResponse(path,media_type='text/csv',filename='pathsnap_demo_summary.csv')

    return app


app=create_app()
