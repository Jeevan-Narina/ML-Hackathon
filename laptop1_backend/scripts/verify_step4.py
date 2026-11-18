"""Verify OOF identity/coverage, source preservation and reloadable final pipelines."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pathsnap.modeling import load_training,FEATURES,IDENTIFIERS,MODELS,positive_probability,metrics
B=Path(__file__).resolve().parents[1]

def verify(root,output,artifacts):
    data,folds,counts=load_training(root);meta=json.loads((output/'training_metadata.json').read_text())
    assert counts==meta['counts']
    for relative,value in meta['inputs'].items():assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==value
    assert hashlib.sha256((root/'trip_folds.csv').read_bytes()).hexdigest()==meta['folds_sha256']
    if (output/'preservation.json').exists():
        for path,value in json.loads((output/'preservation.json').read_text()).items():assert hashlib.sha256((B/path).read_bytes()).hexdigest()==value,path
    manifest=pd.read_csv(artifacts/'artifact_manifest.csv');results=[]
    configs=json.loads((artifacts/'model_configurations.json').read_text())
    expected=data.set_index(IDENTIFIERS)[['target','fold_id']].sort_index()
    fold_metrics=pd.read_csv(output/'fold_metrics.csv')
    for name in MODELS:
        preds=pd.concat([pd.read_parquet(p) for p in sorted((output/'oof'/name).glob('*.parquet'))],ignore_index=True)
        assert len(preds)==len(data) and not preds.duplicated(IDENTIFIERS).any()
        pd.testing.assert_frame_equal(preds.set_index(IDENTIFIERS)[['target','fold_id']].sort_index(),expected)
        assert preds.model_name.eq(name).all() and np.isfinite(preds.positive_probability).all()
        for fold,g in preds.groupby('fold_id'):
            assert set(g.trip_id)==set(folds.loc[folds.validation_fold==fold,'trip_id'])
            row=fold_metrics.loc[(fold_metrics.model_name==name)&(fold_metrics.fold_id==fold)].iloc[0]
            actual=metrics(g)
            for metric,value in actual.items():
                if value is None:assert pd.isna(row[metric])
                else:assert np.isclose(value,row[metric]),f'{name}/{fold}: metric disagreement for {metric}'
        path=artifacts/f'{name}.joblib';checksum=manifest.set_index('model_name').loc[name,'sha256']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==checksum
        p=joblib.load(path)
        assert p.named_steps['classifier'].get_params()==configs[name]
        with threadpool_limits(limits=2):prob=positive_probability(p,data[FEATURES].iloc[:1000])
        assert len(prob)==1000 and list(p.named_steps['finite'].feature_names_in_)==FEATURES
        results.append({'model_name':name,'oof_rows':len(preds),'final_pipeline_reload':'passed'})
    for path in output.glob('*.png'):
        with Image.open(path) as im:im.verify()
    integrity=pd.read_csv(output/'fold_integrity.csv');assert integrity.training_trips.eq(64).all() and integrity.validation_trips.eq(16).all()
    result={'status':'passed','models':results,'folds':5,'training_trips_per_fold':64,'validation_trips_per_fold':16,'counts':counts,'all_input_hashes_preserved':True}
    (output/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=B/'data/processed/step3');p.add_argument('--output',type=Path,default=B/'outputs/step4');p.add_argument('--models',type=Path,default=B/'models/step4');a=p.parse_args();verify(a.data,a.output,a.models)
