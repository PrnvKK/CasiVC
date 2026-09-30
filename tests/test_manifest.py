import json

import pytest

from vc.eval import manifest


def test_write_result_requires_rows(tmp_path):
    with pytest.raises(ValueError):
        manifest.write_result(str(tmp_path / "x.json"), {"exp_id": "EXP-1"}, {"margin": 1.0})


def test_write_result_roundtrip(tmp_path):
    p = tmp_path / "r.json"
    manifest.write_result(str(p), {"exp_id": "EXP-1", "arm": "vq"},
                          {"margin": -0.1, "rows": [{"target": "A"}]})
    d = json.loads(p.read_text())
    assert d["exp_id"] == "EXP-1"
    assert d["rows"] == [{"target": "A"}]


def test_provenance_has_required_keys():
    header = manifest.provenance("EXP-1", "arm", checkpoint=None, seed=0)
    for k in ("git_commit", "config_hash", "ckpt_sha256", "split_id",
              "protocol", "seed", "created_utc"):
        assert k in header


def test_write_pointer(tmp_path):
    r = tmp_path / "results.json"
    manifest.write_result(str(r), {"exp_id": "E"}, {"margin": 0.0, "rows": []})
    ptr = manifest.write_pointer(str(tmp_path), str(r))
    assert ptr["sha256"]
