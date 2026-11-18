"""Run the single-worker Laptop 1 API; no classifier fitting or demo regeneration."""
from pathlib import Path
import argparse,os,sys,logging
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default=os.environ.get('PATHSNAP_HOST','0.0.0.0'))
    p.add_argument('--port',type=int,default=int(os.environ.get('PATHSNAP_PORT','8000')))
    a=p.parse_args()
    if not 1<=a.port<=65535:p.error('Port must be between 1 and 65535')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s')
    import uvicorn
    uvicorn.run('pathsnap.api:app',host=a.host,port=a.port,workers=1,limit_concurrency=8,timeout_keep_alive=5)
