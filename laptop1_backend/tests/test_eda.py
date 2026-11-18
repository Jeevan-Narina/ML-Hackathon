import json

import pandas as pd

from pathsnap.eda import run_eda
from pathsnap.quality import Thresholds


def test_full_audit_reports_broken_files_and_cross_split_duplicates(tmp_path):
    """Tiny synthetic fixtures are confined to tests, never used for project EDA."""
    raw = tmp_path / "raw"
    for split, trip_id in [("train", "00000001"), ("student_test", "00000002"), ("train", "00000003")]:
        directory = raw / split / trip_id
        directory.mkdir(parents=True)
        source = {"track": "8 49 1\n8.001 49 2\n", "nodes": "8 49\n8.001 49\n",
                  "arcs": "0 1\n1 0\n", "route": "0\n1\n"}
        if trip_id == "00000003":
            source["nodes"] = "8 49\n\n8.001 49\n"
            del source["route"]
        for ext, text in source.items():
            (directory / f"{trip_id}.{ext}").write_text(text)
    result = run_eda(raw, tmp_path / "out", Thresholds())
    output = tmp_path / "out"
    assert not result["complete_analysis"]
    assert result["structural_errors"] == 3  # Malformed record, missing file, leaked duplicate track.
    assert result["gps_trips_analyzed"] == 3 and result["network_trips_analyzed"] == 2
    inventory = pd.read_csv(output / "dataset_inventory.csv", dtype={"trip_id": str})
    broken = inventory.set_index("trip_id").loc["00000003"]
    assert broken.missing_files == "route" and broken.malformed_records == 1
    assert broken.network_nodes == 3  # Source row count; malformed line was not dropped/reindexed.
    anomalies = pd.read_csv(output / "anomalies.csv")
    malformed = anomalies[anomalies.category == "malformed_record"].iloc[0]
    assert malformed.record_id == 2  # Source file line number.
    metadata = json.loads((output / "map_metadata.json").read_text())
    assert metadata["split"] == "train" and metadata["trip_id"] == "00000001"
    html = (output / "representative_training_trip.html").read_text(encoding="utf-8")
    assert "[49.0, 8.0]" in html  # Folium markers use latitude, longitude.
    assert "No model predictions" in html
    assert len(list(output.glob("*.png"))) == 5
    assert all((output / name).is_file() for name in result["outputs"])
