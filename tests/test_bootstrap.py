from vc.eval.bootstrap import paired


def _targets(vals):
    return {f"spk{i:02d}": [v] for i, v in enumerate(vals)}


def test_paired_positive_shift():
    a = _targets([0.0] * 16)
    b = _targets([0.05] * 16)
    mean, lo, hi = paired(a, b, "b-a")
    assert mean > 0 and lo > 0 and hi > 0


def test_paired_null_ci_contains_zero():
    a = _targets([0.0, 0.2] * 8)
    mean, lo, hi = paired(a, a, "a-a")
    assert mean == 0 and lo <= 0 <= hi
