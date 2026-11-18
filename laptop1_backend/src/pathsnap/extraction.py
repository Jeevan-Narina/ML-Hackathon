"""Archive validation and extraction. Existing differing files are never overwritten."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path, PurePosixPath
import re
import stat
from zipfile import ZipFile


def extract_dataset(archive: Path, destination: Path, manifest: Path) -> int:
    destination = Path(destination).resolve()
    entries = []
    seen = set()
    with ZipFile(archive) as source:
        # Validate the entire member list before writing any data.
        for info in source.infolist():
            name = info.filename
            parts = PurePosixPath(name).parts
            if ("\\" in name or ":" in name or ".." in parts
                    or not parts or parts[0] != "Student folder"
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f"Unsafe archive member: {name}")
            if info.is_dir():
                continue
            if (len(parts) != 4 or parts[1] not in {"train", "student_test"}
                    or not re.fullmatch(r"\d{8}", parts[2])
                    or parts[3] not in {f"{parts[2]}.{ext}" for ext in ("track", "nodes", "arcs", "route")}):
                raise ValueError(f"Unexpected dataset member: {name}")
            relative = Path(*parts[1:])
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination) or str(relative).casefold() in seen:
                raise ValueError(f"Unsafe or duplicate destination: {name}")
            seen.add(str(relative).casefold())
            entries.append((info, relative, target))
        # Preflight existing data before extraction, including repeat invocations.
        for info, _, target in entries:
            if target.exists() and target.read_bytes() != source.read(info):
                raise ValueError(f"Refusing to overwrite changed source file: {target}")
        rows = []
        for info, relative, target in entries:
            content = source.read(info)  # ZipFile checks CRC.
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open("xb") as output:
                    output.write(content)
            digest = hashlib.sha256(content).hexdigest()
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Post-extraction hash mismatch: {relative}")
            rows.append({"archive_member": info.filename, "relative_path": relative.as_posix(),
                         "bytes": len(content), "sha256": digest})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["archive_member", "relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
