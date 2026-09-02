"""Result provenance manifest.

Every result JSON must carry a provenance header so a number can always be
traced back to code, data split, and checkpoint. ``provenance`` builds the
header; ``wrap`` attaches it without dropping the mandatory ``rows``.
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone


def sha256_file(path, chunk=1 << 20):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def config_hash(*objs):
    """Stable hash of the deciding config values passed in."""
    h = hashlib.sha256()
    for o in objs:
        h.update(repr(o).encode())
    return h.hexdigest()[:16]


def provenance(exp_id, arm, checkpoint=None, config_hash_value=None,
               split_id=None, protocol="n240-v1", seed=None):
    return {
        "exp_id": exp_id,
        "arm": arm,
        "checkpoint": checkpoint,
        "ckpt_sha256": sha256_file(checkpoint),
        "config_hash": config_hash_value,
        "split_id": split_id,
        "protocol": protocol,
        "seed": seed,
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_result(path, header, result):
    """Write a result JSON with provenance header, retaining ``rows``."""
    payload = dict(header)
    payload.update(result)
    if "rows" not in payload:
        raise ValueError("result must retain the per-conversion 'rows' field")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_pointer(exp_dir, result_path):
    """Write results.pointer.json next to an experiment so the link survives
    clones without symlink privileges (Windows fallback)."""
    ptr = {
        "result": os.path.relpath(result_path, exp_dir).replace("\\", "/"),
        "sha256": sha256_file(result_path),
    }
    with open(os.path.join(exp_dir, "results.pointer.json"), "w", encoding="utf-8") as f:
        json.dump(ptr, f, indent=2)
    return ptr
