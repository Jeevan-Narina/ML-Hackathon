from pathlib import Path
import sys
import pandas as pd
import numpy as np

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / "src"))

from pathsnap.evaluation import (
    evaluation_alignment,
    qualify_evaluation,
    score_evaluation,
)
from pathsnap.loaders import load_nodes, load_arcs, load_route
from pathsnap.road_network import RoadNetwork
import json

W = B / "evaluator_workspace"
raw = W / "raw/student_test"
step2 = W / "processed/step2/student_test"
step3 = W / "processed/step3/student_test"
step7 = W / "outputs/step7/trips"
out = W / "outputs/evaluation"
out.mkdir(parents=True, exist_ok=True)

config = json.loads(
    (W / "processed/step3/preprocessing.json").read_text()
)

results = []
all_scores = {"random_forest": [], "viterbi": []}

for trip in sorted(step3.iterdir()):
    if not trip.is_dir():
        continue

    tid = trip.name
    r = raw / tid
    pred_dir = step7 / tid

    net = RoadNetwork(
        load_nodes(r / f"{tid}.nodes"),
        load_arcs(r / f"{tid}.arcs"),
    )

    observations = pd.read_parquet(
        trip / "observations.parquet"
    )

    route = load_route(r / f"{tid}.route")

    alignment = evaluation_alignment(
        net, observations, route, config
    )

    candidates = pd.read_parquet(
        step2 / tid / "candidates.parquet"
    )

    labels, alignment = qualify_evaluation(
        candidates, alignment
    )

    probabilities = pd.read_parquet(
        pred_dir / "probabilities.parquet"
    )

    decoded = pd.read_parquet(
        pred_dir / "decoded.parquet"
    )

    rf_score = score_evaluation(
        probabilities, labels, alignment, len(alignment)
    )

    selected = decoded[
        ["gps_observation_index", "candidate_arc_id"]
    ].dropna()

    viterbi_probs = probabilities.copy()
    viterbi_probs = viterbi_probs.merge(
        selected.assign(_selected=1),
        on=["gps_observation_index", "candidate_arc_id"],
        how="left",
        validate="many_to_one",
    )

    viterbi_probs["positive_probability"] = (
        viterbi_probs["_selected"].fillna(0).astype(float)
    )
    viterbi_probs = viterbi_probs.drop(columns="_selected")

    vit_score = score_evaluation(
        viterbi_probs, labels, alignment, len(alignment)
    )

    results.append({
        "trip_id": tid,
        "observations": len(observations),
        "rf_top1_accuracy": rf_score.get("top1_accuracy"),
        "viterbi_top1_accuracy": vit_score.get("top1_accuracy"),
        "rf_metrics": rf_score,
        "viterbi_metrics": vit_score,
    })

    all_scores["random_forest"].append(
        (probabilities, labels, alignment)
    )
    all_scores["viterbi"].append(
        (viterbi_probs, labels, alignment)
    )

    print(f"Evaluated {tid}", flush=True)

summary = {}

for model, frames in all_scores.items():
    summary[model] = score_evaluation(
        pd.concat([x[0] for x in frames], ignore_index=True),
        pd.concat([x[1] for x in frames], ignore_index=True),
        pd.concat([x[2] for x in frames], ignore_index=True),
        sum(len(x[2]) for x in frames),
    )

rows = []
for result in results:
    row = {
        "trip_id": result["trip_id"],
        "observations": result["observations"],
    }
    for model, prefix in [
        ("rf_metrics", "rf"),
        ("viterbi_metrics", "viterbi"),
    ]:
        row.update({
            f"{prefix}_{k}": v
            for k, v in result[model].items()
        })
    rows.append(row)

pd.DataFrame(rows).to_csv(
    out / "per_trip_metrics.csv", index=False
)

pd.DataFrame([
    {"model": model, **metrics}
    for model, metrics in summary.items()
]).to_csv(out / "overall_metrics.csv", index=False)

print("\n========== EVALUATOR OVERALL RESULTS ==========")
print(pd.read_csv(out / "overall_metrics.csv").to_string(index=False))

print("\n========== PER-TRIP RESULTS ==========")
print(pd.read_csv(out / "per_trip_metrics.csv").to_string(index=False))

print("\nResults saved to:", out)
