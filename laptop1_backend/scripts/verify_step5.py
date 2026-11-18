"""Verify evaluation identity, scoring, continuity and immutable model artifacts."""
from pathlib import Path
import sys,json,hashlib
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from PIL import Image
from threadpoolctl import threadpool_limits
from pathsnap.evaluation import predict_frame,score_evaluation,qualify_evaluation
from pathsnap.modeling import MODELS,IDENTIFIERS

def verify(output):
    meta=json.loads((output/'run_metadata.json').read_text())
    for relative,value in meta['protected_sha256'].items():assert hashlib.sha256((B/relative).read_bytes()).hexdigest()==value,relative
    models=B/'models/step4';metrics=pd.read_csv(output/'student_test_comparison.csv').set_index('model_name')
    assert set(metrics.index)==set(MODELS)
    trips=sorted((B/'data/processed/step3/student_test').iterdir());observations=candidate_rows=qualified=misses=0
    for trip in trips:
        a=pd.read_parquet(output/'evaluation_labels'/trip.name/'alignment.parquet')
        labels=pd.read_parquet(output/'evaluation_labels'/trip.name/'candidate_labels.parquet')
        c=pd.read_parquet(B/'data/processed/step2/student_test'/trip.name/'candidates.parquet')
        expected,obs=qualify_evaluation(c,a)
        pd.testing.assert_frame_equal(expected,labels)
        assert a.index.tolist()==pd.read_parquet(trip/'observations.parquet').index.tolist()
        assert a.loc[a.supervised_eligible,'route_position'].is_monotonic_increasing
        observations+=len(a);candidate_rows+=len(c);qualified+=int(a.supervised_eligible.sum());misses+=int((a.supervised_eligible&~a.reference_candidate_present).sum())
    assert observations==19160 and candidate_rows==113639
    original_fit=Pipeline.fit
    def forbidden(*args,**kwargs):raise AssertionError('Evaluation attempted to fit a pipeline')
    Pipeline.fit=forbidden
    try:
        for name in MODELS:
            pipeline=joblib.load(models/f'{name}.joblib');preds=[];labels=[];aligned=[]
            for trip in trips:
                pred=pd.read_parquet(output/'predictions'/name/f'{trip.name}.parquet')
                assert list(pred.columns)==IDENTIFIERS+['positive_probability']
                f=pd.read_parquet(trip/'features.parquet');pd.testing.assert_frame_equal(f[IDENTIFIERS],pred[IDENTIFIERS])
                with threadpool_limits(limits=2):fresh=predict_frame(pipeline,f)
                np.testing.assert_allclose(fresh.positive_probability,pred.positive_probability,rtol=1e-13,atol=1e-15)
                assert pred.positive_probability.between(0,1).all() and not pred.duplicated(IDENTIFIERS).any()
                preds.append(pred);labels.append(pd.read_parquet(output/'evaluation_labels'/trip.name/'candidate_labels.parquet'));aligned.append(pd.read_parquet(output/'evaluation_labels'/trip.name/'alignment.parquet'))
            actual=score_evaluation(pd.concat(preds),pd.concat(labels),pd.concat(aligned),observations)
            for key,value in actual.items():
                if value is None:assert pd.isna(metrics.loc[name,key])
                else:assert np.isclose(value,metrics.loc[name,key]),f'{name}: {key}'
    finally:Pipeline.fit=original_fit
    cont=pd.read_csv(output/'continuity_by_trip.csv',dtype={'trip_id':str});assert len(cont)==450
    categories=['same_arc','directly_connected','intermediate_required','unreachable','temporal_gap','invalid_timestamp','missing_prediction']
    assert cont[categories].sum(axis=1).eq(cont.adjacent_observation_pairs).all()
    assert cont[['same_arc','directly_connected','intermediate_required','unreachable']].sum(axis=1).eq(cont.assessable_pairs).all()
    for partition in ['student_test','train']:
        for name in MODELS:
            group=cont.loc[(cont.dataset_partition==partition)&(cont.model_name==name)]
            for row in group.itertuples():
                pairs=pd.read_parquet(output/'continuity_pairs'/partition/name/f'{row.trip_id}.parquet')
                assert len(pairs)==row.adjacent_observation_pairs
                for cat in categories:assert int(pairs.category.eq(cat).sum())==getattr(row,cat)
    comparison=pd.read_csv(output/'cv_vs_student.csv');cv=pd.read_csv(B/'outputs/step4/model_comparison.csv').set_index('model_name')
    for row in comparison.itertuples():
        assert np.isclose(row.top1_accuracy_mean,cv.loc[row.model_name,'top1_accuracy_mean'])
        assert np.isclose(row.macro_f1,metrics.loc[row.model_name,'macro_f1'])
    importance=pd.read_csv(output/'feature_analysis.csv');assert len(importance)==125 and importance.groupby('model_name').feature.nunique().eq(25).all()
    for path in output.glob('*.png'):
        with Image.open(path) as image:image.verify()
    result=dict(status='passed',models=5,student_trips=10,prediction_rows_per_model=candidate_rows,raw_observations=observations,
                qualified_alignments=qualified,candidate_misses=misses,continuity_trip_model_rows=len(cont),
                pipeline_fit_calls=0,model_and_source_hashes_unchanged=True,all_saved_predictions_replayed_without_refit=True)
    (output/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=B/'outputs/step5');a=p.parse_args();verify(a.output)
