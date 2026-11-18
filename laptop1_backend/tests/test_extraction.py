from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from pathsnap.extraction import extract_dataset


def test_extraction_idempotent_preserves_bytes_and_refuses_modified_files(tmp_path):
    archive = tmp_path / "source.zip"
    raw = tmp_path / "raw"
    manifest = tmp_path / "manifest.csv"
    with ZipFile(archive, "w") as z:
        z.writestr("Student folder/train/00000001/00000001.track", b"8\t49\t1\r\n")
    assert extract_dataset(archive, raw, manifest) == 1
    target = raw / "train/00000001/00000001.track"
    assert target.read_bytes() == b"8\t49\t1\r\n"
    assert extract_dataset(archive, raw, manifest) == 1
    assert pd.read_csv(manifest).iloc[0]["bytes"] == 8  # Exact source bytes, including CRLF.
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        extract_dataset(archive, raw, manifest)
    assert target.read_bytes() == b"changed"


@pytest.mark.parametrize("name", ["../outside.track", "/absolute.track", "C:/outside.track",
                                       "Student folder/train/../../outside.track", "Student folder\\train\\x.track",
                                       "Student folder/evaluator_test/00000001/00000001.track"])
def test_unsafe_or_unexpected_zip_members_rejected_before_writes(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("Student folder/train/00000001/00000001.track", "8 49 1\n")
        z.writestr(name, "bad")
    with pytest.raises(ValueError):
        extract_dataset(archive, tmp_path / "raw", tmp_path / "manifest.csv")
    assert not (tmp_path / "raw").exists()
