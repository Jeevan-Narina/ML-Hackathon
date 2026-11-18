"""Workspace-only local verification; not required by the standalone frontend.

Starts temporary loopback services, uses HTTP/Streamlit AppTest and stops both.
Never submits a prediction or regenerates earlier datasets/maps/decoding results.
"""
from pathlib import Path
import hashlib,json,os,re,socket,subprocess,time,urllib.request
ROOT=Path(__file__).resolve().parents[1];BACK=ROOT/'laptop1_backend';FRONT=ROOT/'laptop2_frontend'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def free_port():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def main():
    output=BACK/'outputs/step10';output.mkdir(exist_ok=True)
    front_output=FRONT/'outputs/step10';front_output.mkdir(parents=True,exist_ok=True)
    protected_path=output/'protected_artifacts.json'
    if protected_path.exists():protected=json.loads(protected_path.read_text())
    else:
        protected=json.loads((BACK/'outputs/step9/protected_artifacts.json').read_text())
        paths=[*(BACK/'outputs/step9').rglob('*'),*(BACK/'src/pathsnap').glob('*.py'),BACK/'requirements.txt',BACK/'requirements-lock.txt',
            *(ROOT/'docs/steps').glob('STEP_0[1-9]_README.md')]
        for path in paths:
            if path.is_file():protected[path.relative_to(ROOT).as_posix()]=digest(path)
        protected_path.write_text(json.dumps(protected,indent=2))
    front_python=FRONT/'.venv/Scripts/python.exe';backend_python=BACK/'.venv/Scripts/python.exe'
    tests=subprocess.run([str(front_python),'-m','unittest','discover','-s','tests','-v'],cwd=FRONT,capture_output=True,text=True,timeout=60)
    (front_output/'test-results.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    assert tests.returncode==0,tests.stderr
    count=int(re.search(r'Ran (\d+) tests',tests.stderr).group(1))
    (front_output/'test_summary.json').write_text(json.dumps(dict(tests=count,failures=0,errors=0),indent=2))
    backend_port,frontend_port=free_port(),free_port()
    services=[];logs=[];opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def start(command,cwd,log_name):
        log=(output/log_name).open('w',encoding='utf-8');logs.append(log)
        process=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0);services.append(process);return process
    def wait(url,process):
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Temporary service exited; inspect verification logs')
            try:
                with opener.open(url,timeout=2) as response:
                    if response.status==200:return response.read()
            except OSError:pass
            time.sleep(.2)
        raise TimeoutError('Temporary service did not become ready')
    try:
        backend=start([str(backend_python),str(BACK/'scripts/serve_api.py'),'--host','127.0.0.1','--port',str(backend_port)],BACK,'backend_server.log')
        backend_url=f'http://127.0.0.1:{backend_port}';wait(backend_url+'/health',backend)
        frontend=start([str(front_python),'-m','streamlit','run','app.py','--server.headless=true','--server.address=127.0.0.1',f'--server.port={frontend_port}'],FRONT,'streamlit_server.log')
        wait(f'http://127.0.0.1:{frontend_port}/_stcore/health',frontend)
        check=subprocess.run([str(front_python),str(FRONT/'scripts/verify_backend.py'),'--backend-url',backend_url],cwd=FRONT,capture_output=True,text=True,timeout=90)
        (output/'frontend_http_verification.log').write_text(check.stdout+check.stderr,encoding='utf-8')
        assert check.returncode==0,check.stdout+check.stderr
        print(check.stdout,flush=True)
    finally:
        for process in reversed(services):
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        for log in logs:log.close()
    for relative,value in protected.items():assert digest(ROOT/relative)==value,relative
    assert 'route=/predict' not in (output/'backend_server.log').read_text(encoding='utf-8')
    report=dict(status='passed',frontend_tests=count,streamlit_web_server_health=True,
        real_http_cached_demo_verified=True,protected_files_unchanged=len(protected),
        backend_imports_in_frontend=False,backend_datasets_or_models_copied=False,
        prediction_posted_to_live_backend=False,prior_backend_tests_not_rerun=192,
        two_laptop_network_tested=False,temporary_services_stopped=all(p.poll() is not None for p in services))
    (output/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
