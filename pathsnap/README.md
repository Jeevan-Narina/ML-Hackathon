# PathSnap

Map-matching of noisy GPS trajectories to a road network.

## Setup
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Data
Copy the trip folders from the provided "Student folder" into:
- data/train/          (80 trips, e.g. data/train/00000048/00000048.track ...)
- data/student_test/   (10 trips)
Raw data is git-ignored.

## Step 1: candidate generation on one trip
    python src/candidates_one_trip.py data/train/00000048 50 maps
Opens-able output: maps/trip_candidates.html

## Layout
- src/ source code   - models/ saved models   - api/ FastAPI backend
- ui/ evaluator UI   - results/ CSVs and logs   - maps/ Folium maps
- notebooks/ exploration
