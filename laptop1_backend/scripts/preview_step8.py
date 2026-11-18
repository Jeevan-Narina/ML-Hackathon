"""Open local maps in headless Chrome/Edge; check live layers and save previews.

This is browser validation, not an API or a web server. Map tiles are deliberately
blocked for the offline-library check. Regular map files retain online OSM tiles.
"""
from pathlib import Path
import argparse,subprocess,json,re,html,os,time
B=Path(__file__).resolve().parents[1]

HARNESS=r'''<script>
window.addEventListener('load',function(){setTimeout(function(){
let result={ready:window.pathsnapReady===true,errors:window.pathsnapErrors||[]};
try{
 const name=Object.keys(window).find(k=>/^map_[0-9a-f]+$/.test(k)&&window[k] instanceof L.Map),map=window[name];
 result.layerControls=document.querySelectorAll('.leaflet-control-layers-selector').length;
 result.canvas=document.querySelectorAll('canvas').length;
 result.referenceInitiallyHidden=!map.hasLayer(pathsnapLayers.reference);
 pathsnapView('before');result.before=map.hasLayer(pathsnapLayers.classifier)&&!map.hasLayer(pathsnapLayers.route);
 pathsnapView('after');result.after=!map.hasLayer(pathsnapLayers.classifier)&&map.hasLayer(pathsnapLayers.route);
 pathsnapView('both');result.compare=map.hasLayer(pathsnapLayers.classifier)&&map.hasLayer(pathsnapLayers.route);
 const geo=Object.keys(window).filter(k=>/^geo_json_[0-9a-f]+$/.test(k)).map(k=>window[k]);
 const gps=geo.find(g=>g instanceof L.GeoJSON&&g.getLayers().some(l=>l.feature&&l.feature.properties.observation_index!==undefined&&l.feature.properties.candidate_count!==undefined));
 result.gpsFeatures=gps.getLayers().length;
 const first=gps.getLayers()[0];first.fire('click',{latlng:first.getLatLng()},true);result.popupText=map._popup?map._popup.getElement().textContent.slice(0,1200):'';result.popup=result.popupText.includes('Timestamp');map.closePopup();
 const layer=pathsnapLayers.reference;map.addLayer(layer);result.referenceCanEnable=map.hasLayer(layer);map.removeLayer(layer);
 pathsnapView('after');
}catch(error){result.error=String(error)}
const node=document.createElement('pre');node.id='pathsnap-browser-result';node.style.display='none';node.textContent=JSON.stringify(result);document.body.appendChild(node);
},100);});</script>'''


def preview(output):
    candidates=[Path(os.environ.get('PROGRAMFILES','C:/Program Files'))/'Google/Chrome/Application/chrome.exe',
        Path(os.environ.get('PROGRAMFILES(X86)','C:/Program Files (x86)'))/'Microsoft/Edge/Application/msedge.exe']
    browser=next((p for p in candidates if p.exists()),None)
    if browser is None:raise RuntimeError('Chrome or Edge is needed for browser verification')
    dest=output/'previews';dest.mkdir(exist_ok=True);profile=B/'pytest_temp/step8_browser_profile';profile.mkdir(parents=True,exist_ok=True)
    results=[]
    for path in sorted((output/'maps').glob('*.html')):
        probe=dest/'probe.html';source=path.read_text(encoding='utf-8')
        source=source.replace('<head>','<head><script>window.pathsnapErrors=[];window.addEventListener("error",function(e){if(e.message)window.pathsnapErrors.push(e.message);});</script>',1)
        probe.write_text(source.replace('</html>',HARNESS+'</html>'),encoding='utf-8')
        screenshot=dest/f'{path.stem}.png' if path.stem in ['00000009','00000065'] else None
        command=[str(browser),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
          '--allow-file-access-from-files','--no-proxy-server','--host-resolver-rules=MAP tile.openstreetmap.org ~NOTFOUND',
          f'--user-data-dir={profile.resolve()}','--window-size=1440,1000','--virtual-time-budget=1500','--dump-dom']
        if screenshot:command.append(f'--screenshot={screenshot.resolve()}')
        command.append(probe.resolve().as_uri())
        proc=subprocess.run(command,capture_output=True,timeout=45,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        dom=proc.stdout.decode('utf-8',errors='replace');match=re.search(r'<pre id="pathsnap-browser-result"[^>]*>(.*?)</pre>',dom,re.S)
        (dest/f'{path.stem}_browser.log').write_text(proc.stderr.decode('utf-8',errors='replace'),encoding='utf-8')
        if not match:raise AssertionError(f'Browser probe did not complete: {path.name}; see preview log')
        result=json.loads(html.unescape(match.group(1)));result['trip_id']=path.stem
        (dest/f'{path.stem}_probe_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        for flag in ['ready','referenceInitiallyHidden','before','after','compare','popup','referenceCanEnable']:assert result.get(flag),result
        assert result['layerControls']==17 and result['canvas']>0 and not result.get('errors') and not result.get('error'),result
        assert result['gpsFeatures']==json.loads((output/'metadata'/f'{path.stem}.json').read_text())['display']['valid_gps']
        results.append(result);print('Browser passed:',path.stem,flush=True)
    # A separate native browser load of the index confirms its DOM and links.
    index_command=[str(browser),'--headless=new','--disable-gpu','--no-first-run',f'--user-data-dir={profile.resolve()}',
        '--window-size=1440,1100',f'--screenshot={(dest/"index.png").resolve()}','--dump-dom',(output/'index.html').resolve().as_uri()]
    index=subprocess.run(index_command,capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    index_dom=index.stdout.decode('utf-8',errors='replace');assert index.returncode==0 and 'From noisy points to road routes.' in index_dom and index_dom.count('maps/')>=10
    report=dict(status='passed',browser=str(browser),maps_opened=len(results),index_opened=True,
        local_assets_work_without_tiles=True,tiles_blocked_during_probe=True,trip_results=results)
    (output/'browser_verification.json').write_text(json.dumps(report,indent=2));probe.unlink(missing_ok=True)
    print('Browser verified all maps and index; previews saved.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=B/'outputs/step8');a=p.parse_args();preview(a.output)
