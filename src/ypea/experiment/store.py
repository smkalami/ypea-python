"""Files of an experiment: one JSON file per run, plus ``metadata.json``.

Layout of the output folder::

    metadata.json
    runs/<algorithm>/<problem>/run_<i>.json

Folder names are the labels with characters that Windows forbids replaced by ``_``. The
real names are inside the files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def safe_name(name: str) -> str:
    """A folder name for a label or problem name, valid on Windows, macOS and Linux."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name)).strip().rstrip(".")
    return s or "_"


def run_path(output, label, problem, run) -> Path:
    return Path(output) / "runs" / safe_name(label) / safe_name(problem) / f"run_{run}.json"


def name_hash(name) -> int:
    """A stable 32-bit number for a name (Python's ``hash`` changes between processes)."""
    return int.from_bytes(hashlib.sha256(str(name).encode("utf-8")).digest()[:4], "little")


def key_hash(data) -> str:
    """A stable hash of plain data (JSON with sorted keys, SHA-256)."""
    text = json.dumps(data, sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_json(path, data):
    """Write JSON atomically: a temporary file in the same folder, then ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=repr)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_run(path, key=None):
    """The run stored at ``path``, or None if it is missing, unreadable or has another key."""
    try:
        rec = read_json(path)
    except (OSError, ValueError):
        return None
    if key is not None and rec.get("key") != key:
        return None
    return rec


def load(output):
    """Metadata and run records of an output folder, in the order of the metadata."""
    output = Path(output)
    meta = read_json(output / "metadata.json")
    records = []
    for label, by_problem in meta["keys"].items():
        for problem, keys in by_problem.items():
            # only runs made with the settings in the metadata
            for i, key in enumerate(keys):
                rec = read_run(run_path(output, label, problem, i), key)
                if rec is not None:
                    # the per iteration history can be many megabytes per run, and reports use only the trace
                    rec["result"].pop("history", None)
                    records.append(rec)
    return meta, records
