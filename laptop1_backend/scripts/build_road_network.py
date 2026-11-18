"""Build/rebuild per-trip network tables and in-memory spatial indexes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.step2 import main

if __name__ == "__main__":
    main(network_only=True)
