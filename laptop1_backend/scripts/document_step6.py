"""Document measured Bayesian optimisation, selection and recovery behaviour."""
from pathlib import Path
import json,argparse,xml.etree.ElementTree as ET
import pandas as pd
B=Path(__file__).resolve().parents[1];ROOT=B.parent

def document(output,models):
    meta=json.loads((output/'run_metadata.json').read_text());selection=json.loads((models/'classifier_selection.json').read_text())
    coverage=json.loads((B/'outputs/step3/results.json').read_text())
    comparison=pd.read_csv(output/'baseline_vs_tuned.csv');folds=pd.read_csv(output/'tuned_fold_metrics.csv');studies=pd.read_csv(output/'study_summary.csv')
    params=json.loads((models/'best_parameters.json').read_text());manifest=pd.read_csv(models/'artifact_manifest.csv')
    student=pd.read_csv(output/'baseline_vs_tuned_student.csv');importance=json.loads((output/'parameter_importance_status.json').read_text())
    verified=json.loads((output/'verification.json').read_text());suites=list(ET.parse(output/'pytest-results.xml').getroot().iter('testsuite'))
    tests={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    lines=['# Step 06: Bayesian hyperparameter optimisation and classifier selection','',
      '## Objective, purpose and boundary',
      'Tune the two promising probabilistic classifiers without rerunning data processing or retraining the five baselines. Step 5 favoured RF on development Top-1 by only 0.172 percentage points, while HGB had better probability quality and CV. Neither model was assumed to win. The result is a selected classifier for the upcoming decoder stage; no HMM/Viterbi, route visualisation, API/UI or deployment ZIP is implemented.', '',
      '## Bayesian optimisation and Optuna',
      'Optuna persists parameter/objective observations in SQLite. Seeded Tree-structured Parzen Estimator (TPE) proposals use densities of promising versus less promising parameter settings to choose further candidates. The first two trials are seeded startup exploration; subsequent proposals use completed study history. This is adaptive search, not exhaustive grid search or a guarantee of a global optimum.',
      'The sampler is seeded with base_seed + trial_number for each serial trial. Given identical complete history/library versions this reproduces proposals across normal stop/resume boundaries without serialising an RNG pickle. SQLite writes asks, parameters, attributes and completed values; completed trials are never evaluated again. A RUNNING trial abandoned by a killed process is marked FAIL on recovery and consumes an attempted-trial slot. That interrupted attempt can change subsequent trial numbering/proposals relative to a never-interrupted run; reproducibility does not mean interrupted and uninterrupted traces are identical in that case.', '',
      '## Data integrity, objective and preprocessing',json.dumps(meta['counts'],indent=2),
      'Only the existing 80 training trips enter the optimiser. Original feature/label identifiers join one-to-one and the 25-feature order is unchanged. Null/excluded targets remain excluded. There is exactly one positive per eligible candidate-present observation. check_training_partition rejects student/evaluator rows; all optimisation feature matrices use the fixed feature allowlist and never include route IDs or aligned labels.',
      f"The original training population has {int(coverage['observations'])} raw GPS observations: {int(coverage['valid_aligned'])} valid reference-derived alignments, {int(coverage['ambiguous'])} ambiguous, {int(coverage['alignment_unmatched'])} unalignable and {int(coverage['excluded'])} excluded. Of valid alignments, {int(coverage['candidate_hits'])} have their reference candidate and {int(coverage['valid_aligned']-coverage['candidate_hits'])} are candidate misses. Conditional candidate recall remains {coverage['candidate_recall']:.6%}; optimisation does not repair these omissions or improve end-to-end coverage by definition.",
      'The objective is the equal mean of held-out observation-level Top-1 across the configured search-fold subset. Rank positive-class probability descending and original arc ID ascending for exact ties. Supporting accuracy, precision/recall, binary/Macro F1, ROC-AUC, MRR, log loss and confusion counts are saved, not used as substitutes for Top-1. Candidate accuracy has candidate-row denominator; Top-1/MRR use eligible reference-present observation groups. Candidate recall and end-to-end/raw coverage are unchanged data limitations, not improved by tuning.',
      'Every trial/fold creates a fresh Step 4-compatible Scikit-learn Pipeline: finite-to-missing schema guard, training-fold median imputation/missing indicators, classifier. Both tuned tree families need no scaler. Inverse candidate-count observation weights, normalised to mean one on fitting rows, apply to classifier fitting; imputer statistics remain ordinary training-row medians. Validation rows never fit transformations. HGB early_stopping=False avoids an internal random-row validation split. The base estimator seed remains 42.', '',
      '## CPU budget and rationale',json.dumps(meta['configuration'],indent=2),
      'The default is at most six attempted trials per model (configurable from 1 to 20), a 180-second soft per-model search budget, one serial worker and two native/estimator threads. The default search uses saved fold 0: 64 fitting trips and 16 held-out trips. This is deliberately limited initial exploration on the supplied CPU; final selected configurations are assessed on all five saved folds. Search folds may be configured to a larger subset before starting a distinct study. Data/eligibility/fold assignments are never changed.',
      'The clock is checked between trials, so a fit already in progress is allowed to finish/checkpoint and can overrun the soft limit. Full five-fold selected-configuration evaluation and final fitting are required additional work outside that search budget. A later invocation has a fresh time allowance but the same total attempted-trial target; it only adds missing attempts up to that target. Raising --trials can deliberately extend a study. Do not launch two runners against the same study directory.', '',
      '## Search spaces and actual best parameters',
      'RF: n_estimators bounds forest cost; max_depth and min_samples_leaf constrain complexity; min_samples_split constrains node eligibility; max_features controls random feature availability; max_samples controls bootstrap size. Some split-size options are dominated by the minimum-leaf constraint and can yield operationally equivalent configurations. Repeated objective values are genuine results, not fabricated improvements.',
      'HGB: learning_rate and max_iter trade boosting step size against rounds; max_leaf_nodes/max_depth/min_samples_leaf constrain tree complexity; l2_regularization shrinks leaf values; max_bins bounds histogram resolution and cost. Learning rate and L2 are sampled on logarithmic scales. All combinations are valid; no GPU dependency is used.', '```json',json.dumps(meta['search_space'],indent=2),'```',
      'Actual selected parameters:', '```json',json.dumps(params,indent=2),'```', '',
      '## Actual studies and trial histories','',
      '| Model | Successful trials | Attempted trials | Best trial | Search-fold objective | Recorded trial seconds |',
      '|---|---:|---:|---:|---:|---:|']
    for r in studies.itertuples():lines.append(f'| {r.model_name} | {r.completed_trials} | {r.attempted_trials} | {r.best_trial} | {r.best_objective:.6f} | {r.recorded_trial_seconds:.3f} |')
    lines += ['',
      'SQLite studies, trial-history CSVs, per-trial metrics and held-out predictions are saved incrementally. Best search-fold scores are not five-fold scores. On a selected configuration, an already computed matching search-fold prediction can be reused after identity checks; the remaining saved folds are fitted/evaluated. Final full-training pipelines are fitted separately, and their predictions are never substituted for CV.', '',
      '## Matched baseline versus tuned five-fold selection results','',
      '| Model | Variant | Top-1 ± SD | Macro F1 | Precision | Recall | ROC-AUC | MRR | Log loss | Mean fit seconds |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparison.itertuples():
        for variant in ['baseline','tuned']:
            get=lambda field:getattr(row,f'{field}_mean_{variant}')
            lines.append(f'| {row.model_name} | {variant} | {get("top1_accuracy"):.6f} ± {getattr(row,"top1_accuracy_std_"+variant):.6f} | {get("macro_f1"):.6f} | {get("precision"):.6f} | {get("recall"):.6f} | {get("roc_auc"):.6f} | {get("mean_reciprocal_rank"):.6f} | {get("log_loss"):.6f} | {get("training_seconds"):.3f} |')
    for row in comparison.itertuples():
        direction='improvement' if row.top1_improvement_percentage_points>0 else 'degradation' if row.top1_improvement_percentage_points<0 else 'no change'
        lines += ['',f'{row.model_name}: {direction} of {row.top1_improvement_percentage_points:+.6f} percentage points in equal-fold mean Top-1 relative to its baseline. Other metrics must be read separately; a Top-1 change does not guarantee better calibration.']
    lines += ['',
      'The same five saved folds give 64 fitting and 16 validation trips each. Means equally weight folds; SD uses ddof=1. Search influenced at least fold 0, and five-fold evaluation is used for selection, so these are optimistic model-selection estimates, not a new independent held-out test. No nested CV claim is made. Reference-derived labels, excluded ambiguities and candidate misses preserve the limitations of Steps 3–5.', '',
      '## Selected classifier and backup',
      f"Primary: **{selection['primary']}**. Backup: **{selection['backup']}**. Choice prioritises training-only five-fold Top-1; log loss and fold stability break exact ties. See inference durations and artifact size for deployment trade-offs. If the difference is small, this does not establish statistical superiority. Both provide positive-class probabilities compatible with future HMM emissions; they are independent binary candidate scores, not route probabilities.",
      'The original Step 4 models remain unchanged and available. “Selected classifier” identifies the current untuned-versus-tuned experiment decision; it does not claim route reconstruction or independently verified physical truth. If tuning degrades baseline CV, the degradation is explicitly shown above and must be considered before deployment.', '',
      '## One post-selection student-test comparison',
      'Official Models and Training Guidelines §7 also asks for a tuned student-test comparison. After freezing the training-CV choice, report_step6.py scores each tuned full-fit pipeline once on unchanged student inference features, then joins the existing Step 5 evaluation-only labels. No reference alignment is regenerated, no model/scaler/imputer is refitted and no student result changes the saved selection. This is development evidence, not an untouched evaluator test.', '',
      '| Model | Baseline Top-1 | Tuned Top-1 | Baseline Macro F1 | Tuned Macro F1 | Baseline log loss | Tuned log loss |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for r in student.itertuples():lines.append(f'| {r.model_name} | {r.top1_accuracy_baseline:.6f} | {r.top1_accuracy_tuned:.6f} | {r.macro_f1_baseline:.6f} | {r.macro_f1_tuned:.6f} | {r.log_loss_baseline:.6f} | {r.log_loss_tuned:.6f} |')
    lines += ['',
      'Student denominators remain frozen: 19,160 raw observations, 15,121 qualified alignments, 14,547 qualified reference-present groups and 574 candidate misses. Conditional Top-1 does not count all raw observations; miss-aware values and coverage remain in tuned_student_test.csv. This check is not used for objective/search-space/primary-model selection.', '',
      '## Runtime, memory and importance',
      f"Latest runner invocation elapsed {meta['runtime_seconds']:.3f} seconds. Search-time totals are in study_summary.csv; selected-CV fit times sum to {folds.training_seconds.sum():.3f} seconds, including any reused best search-fold fits. Do not sum trial and CV time without subtracting reused fits. Maximum sampled process RSS in selected CV fits: {folds.peak_rss_mib.max():.3f} MiB. Baseline timings were measured under different system load; no comparative speedup benchmark is claimed. A resumed invocation elapsed time excludes earlier invocations while persisted per-trial/per-fold timings remain available.",
      'Parameter-importance generation status:',json.dumps(importance,indent=2),
      'The report requires at least ten successful trials with varying objective values for exploratory fANOVA parameter importance. Smaller studies retain genuine histories/parameters but do not fabricate an importance chart. Even at ten trials, parameter importance is exploratory and affected by correlated or equivalent settings.', '',
      '## Python modules, libraries and function contracts',
      '- optimization.SEARCH_SPACES/suggest_parameters(trial, model) -> valid bounded RF/HGB parameters; other model families are rejected.',
      '- optimization.tuned_pipeline(model, parameters) -> unchanged feature/imputer/classifier pipeline with only the specified classifier hyperparameters changed.',
      '- optimization.check_training_partition(data) -> rejects student/evaluator partitions, missing eligibility and invalid folds/targets.',
      '- optimization.run_study(model, SQLite path, identity, objective, budgets, seed) -> persistent TPE study; identity mismatches fail, completed trials are reused, abandoned running trials fail explicitly.',
      '- scripts/tune_models.py predict_fold -> a fresh fold-local fit, genuine held-out probabilities and metrics; run -> studies, selected five-fold evaluation, full fits and training-CV selection.',
      '- scripts/report_step6.py -> frozen-selection student check, matched comparison and optimisation-history plots; no fitting.',
      '- scripts/verify_step6.py -> trial objective replay, all OOF identities/folds/metrics, protected-artifact hashes, classifier/schema/reload checks without fitting.',
      '- scripts/document_step6.py -> this measured reference, STEP6_SUMMARY.md and all three main README updates.',
      'Optuna TPE and SQLAlchemy/SQLite provide search/persistence; Scikit-learn/Joblib/threadpoolctl reuse model pipelines and bounded CPU fitting; Pandas/NumPy/Parquet handle data and predictions; Matplotlib plots actual results; psutil samples process RSS through the existing helper; pytest verifies behaviour. The only new direct dependency is Optuna, pinned in requirements.txt; complete versions are in requirements-lock.txt and training metadata.', '',
      '## Outputs and saved artifacts',
      'outputs/step6/studies/<model>.sqlite and <model>_trials.csv retain all attempts. trials/<model>/trial_<number>/ holds genuine search-fold predictions and metrics. cv/<model>/fold_<n>.parquet and JSON identity/metric records hold selected-configuration five-fold results. Additional outputs: tuned_fold_metrics.csv, study_summary.csv, *_best_parameters.json, baseline_vs_tuned.csv, classifier_selection.json, input_identity.json, protected_artifacts.json, run_metadata.json, parameter_importance_status.json, tuned_student_test.csv, baseline_vs_tuned_student.csv and separate student_predictions/. Plots: optimisation_history.png, top1_accuracy_comparison.png, macro_f1_comparison.png, log_loss_comparison.png; parameter-importance PNG/CSV only when supported. Verification and actual test evidence accompany STEP6_SUMMARY.md.',
      'models/step6/ contains random_forest_tuned.joblib and hist_gradient_boosting_tuned.joblib, identity/checksum JSONs, artifact_manifest.csv, best_parameters.json, exact feature_schema.json, classifier_selection.json and training_metadata.json. Pipelines retain learned preprocessing. Step 4 models and Steps 1–5 data/reports are never overwritten.','```json',manifest.to_json(orient='records',indent=2),'```','',
      '## Recovery and exact Windows PowerShell commands','```powershell',
      'cd .\\PathSnap\\laptop1_backend',
      '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt',
      '.\\.venv\\Scripts\\python.exe scripts\\tune_models.py',
      '.\\.venv\\Scripts\\python.exe scripts\\report_step6.py',
      '.\\.venv\\Scripts\\python.exe scripts\\verify_step6.py',
      '.\\.venv\\Scripts\\python.exe -m pytest -q --junitxml=outputs\\step6\\pytest-results.xml',
      '.\\.venv\\Scripts\\python.exe scripts\\document_step6.py','```',
      'Rerun tune_models.py with the same paths/configuration to resume; no special flag is needed. Completed SQLite trials are never rerun. Matching selected-fold predictions and final pipelines are reused. --trials 10 --timeout 300 deliberately extends the target/time allowance. Change search spaces, seeds, folds, feature/code/input identities only in a separate output directory. --config, --data, --output and --models configure paths. Keep the same environment/library versions. One runner per study directory is required; do not start concurrent optimisers. Budget changes do not silently change partitions or feature definitions.', '',
      '## Tests and verification',json.dumps(tests),json.dumps(verified),
      'New tests cover valid/reproducible configurations, probability and model save/reload, training-only partitions, Top-1 versus candidate accuracy, forbidden reference parameters, SQLite recovery, reuse of completed trials, abandoned RUNNING attempts, budget validation and equality of normally resumed versus uninterrupted seeded suggestions. Earlier 115 tests remain intact. Independent verification checks all recorded successful trial objectives, ten selected CV folds, held-out identities/targets, all metrics, unchanged baseline artifacts and reload inference with fitting disabled. No model is retrained by verification.', '',
      '## Limitations and Step 7 readiness',
      'The small trial budget and single search fold favour time efficiency over exhaustive/stable tuning. Search results can depend on that fold, especially with trip/region variability. Full five-fold selection overlaps search data and is not independent. Heuristic labels and candidate-miss/exclusion policies can make conditional performance optimistic. No improvement is presumed; numerical changes and degradations are shown explicitly. Student data is development data and is not used to steer optimisation. No hidden evaluator data is accessed.',
      'The selected primary/backup pipelines, exact feature order, probabilities, parameters and comparison evidence are ready for the separately approved Step 7 HMM/Viterbi work. Neither decoder, route visualisation nor FastAPI/Streamlit/deployment has been started. Some parameter-importance charts may be unavailable because the budget is intentionally small; this is documented rather than fabricated.', '',
      'Changelog — 2026-11-18: added Step 6 training-only Bayesian studies, resumable checkpoints, selected-configuration CV/full fits, frozen-choice reporting, verification, tests and documentation. AI-assisted results remain subject to team review and event disclosure requirements.','']
    tradeoffs=['# Classifier selection report','',f"Primary: **{selection['primary']}**; backup: **{selection['backup']}**. Selected using training CV before student reporting.",'',
      '| Model | CV Top-1 | Top-1 SD | Log loss | Mean fold inference (s) | Artifact bytes | Full fit (s) |',
      '|---|---:|---:|---:|---:|---:|---:|']
    runtime=[]
    for r in comparison.itertuples():
        artifact=manifest.loc[manifest.model_name==r.model_name].iloc[0]
        study=studies.loc[studies.model_name==r.model_name].iloc[0]
        subset=folds.loc[folds.model_name==r.model_name]
        tradeoffs.append(f'| {r.model_name} | {r.top1_accuracy_mean_tuned:.6f} | {r.top1_accuracy_std_tuned:.6f} | {r.log_loss_mean_tuned:.6f} | {r.prediction_seconds_mean_tuned:.6f} | {artifact["bytes"]} | {artifact.full_fit_seconds:.3f} |')
        runtime.append(dict(model_name=r.model_name,search_trial_seconds=study.recorded_trial_seconds,selected_cv_fit_seconds_including_reuse=subset.training_seconds.sum(),cv_inference_seconds=subset.prediction_seconds.sum(),full_fit_seconds=artifact.full_fit_seconds,peak_selected_cv_rss_mib=subset.peak_rss_mib.max()))
    tradeoffs += ['', 'Random Forest has the higher tuned five-fold mean Top-1 and lower log loss. HGB is smaller and faster at inference in this run. Fold variability is substantial; this small non-nested experiment does not establish statistical superiority. Both provide binary candidate probabilities suitable for future emission construction; neither implements sequence inference.']
    (output/'model_selection_report.md').write_text('\n'.join(tradeoffs),encoding='utf-8')
    pd.DataFrame(runtime).to_csv(output/'runtime_resource_summary.csv',index=False)
    lines += ['', '## Measured selection trade-offs', *tradeoffs[2:]]
    # Markdown tables and fenced blocks require consecutive lines.
    rendered=[];in_code=False
    for line in lines:
        previous=rendered[-1] if rendered else ''
        contiguous=in_code or (line.startswith('|') and previous.startswith('|'))
        if not line and not in_code:continue
        if rendered and not contiguous:rendered.append('')
        rendered.append(line)
        if line.startswith('```'):in_code=not in_code
    text='\n'.join(rendered);(ROOT/'docs/steps/STEP_06_README.md').write_text(text,encoding='utf-8');(output/'STEP6_SUMMARY.md').write_text(text,encoding='utf-8')
    for path,prefix in [(ROOT/'README.md','docs/steps/'),(B/'README.md','../docs/steps/'),(ROOT/'laptop2_frontend/README.md','../docs/steps/')]:
        old=path.read_text(encoding='utf-8');marker='\n<!-- STEP6_CURRENT -->'
        if marker in old:old=old.split(marker)[0]
        old=old.replace('**Steps 1–5 are implemented:**','**Steps 1–6 are implemented:**').replace('Optimisation, decoding and the distributed application remain pending.','Bayesian classifier optimisation is now complete. Decoding and the distributed application remain pending.')
        old=old.replace('(Steps 1–5)','(Steps 1–6)').replace('Steps 1–5 maintain this directory and documentation only.','Steps 1–6 maintain this directory and documentation only.')
        old=old.replace('## Current status: Step 5 completed','## Historical Step 5 completion record').replace('stop before Step 6','Step 6 Bayesian optimisation; stop before Step 7')
        old=old.replace('thirteen direct dependencies;','fourteen direct dependencies;')
        old=old.replace('Steps 1–5 are complete; Step 6 requires explicit approval.','Steps 1–6 are complete; Step 7 requires explicit approval.')
        old=old.replace('- [ ] Step 6 Bayesian optimisation: not started.','- [x] Step 6 Bayesian optimisation; see current reference.')
        addition=marker+f"\n\n## Current status: Step 6 completed\n\n[Step 6 detailed reference]({prefix}STEP_06_README.md). Primary classifier: **{selection['primary']}**; backup: **{selection['backup']}**, chosen from training-only five-fold CV after a bounded TPE search. Student-test results were a post-selection report, not an optimisation objective or selection input. Original Steps 1–5 artifacts remain unchanged.\n\nCompleted trials: "+'; '.join(f'{r.model_name}: {r.completed_trials}' for r in studies.itertuples())+f". Actual comparisons, parameters, timings and limitations are in the step reference and laptop1_backend/outputs/step6/. Two full-training pipelines and feature/configuration metadata are under laptop1_backend/models/step6/. {tests['tests']} tests; {tests['failures']} failures; {tests['errors']} errors; {tests['skipped']} skipped. Optuna is the only new direct dependency; Laptop 2 dependencies are unchanged and the frontend remains unimplemented.\n\n- [x] Step 6 optimisation/CV/full fits/verification/documentation.\n- [ ] Step 7 HMM/Viterbi; not started. API/UI, route visualisation and deployment remain pending.\n\nChangelog — 2026-11-18: added bounded persistent Optuna studies, tuned-model comparison and training-CV classifier selection. Earlier step references remain intact. Step 7 requires explicit approval.\n"
        path.write_text(old+addition,encoding='utf-8')
    print('Step 6 reference, summary and three main READMEs updated from actual evidence.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=B/'outputs/step6');p.add_argument('--models',type=Path,default=B/'models/step6');a=p.parse_args();document(a.output,a.models)
