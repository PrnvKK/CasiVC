"""Paired target-level bootstrap over saved arm results.

Two entry points:
  * ``load`` / ``paired`` / ``summary`` — margin-level comparison over
    ``per_target`` means (identical eval pairs, 4000 resamples by target).
  * ``target_means`` — component metrics (sim_tgt/sim_src/content_l1) when the
    result JSON retains per-conversion ``rows``.

Result JSONs MUST retain ``rows`` so the second path stays possible.
"""

import json
import random
import statistics as st
from collections import defaultdict


def load(p):
    """Load a results JSON. Supports 'file.json:key' for multi-arm files
    (e.g. a file holding both 'v0' and 'v1')."""
    if ":" in p:
        p, key = p.split(":", 1)
        d = json.load(open(p))
        if "per_target" in d:
            return d["per_target"], d
        return d[key]["per_target"], d[key]
    d = json.load(open(p))
    if "per_target" in d:
        return d["per_target"], d
    k = list(d.keys())[0]
    return d[k]["per_target"], d[k]


def paired(a, b, name, boot=4000):
    """Paired mean difference b - a with target-level bootstrap CI."""
    common = sorted(set(a) & set(b))
    td = [st.mean(b[t]) - st.mean(a[t]) for t in common]
    rng = random.Random(0)
    boot_means = []
    for _ in range(boot):
        s = [td[rng.randrange(len(td))] for _ in range(len(td))]
        boot_means.append(st.mean(s))
    boot_means.sort()
    lo, hi = boot_means[int(0.025 * len(boot_means))], boot_means[int(0.975 * len(boot_means))]
    print(f"{name}: mean {st.mean(td):+.4f} CI [{lo:+.4f},{hi:+.4f}] "
          f"wins {sum(1 for x in td if x > 0)}/{len(td)}")
    return st.mean(td), lo, hi


def summary(name, v):
    print(f"{name:10} margin {v['margin']:+.4f} CI [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
          f"src {v['sim_src']:.4f} tgt {v['sim_tgt']:.4f} L1 {v['content_l1']:.4f}")


def target_means(data, metric):
    """Per-target-speaker mean of a component metric from retained ``rows``."""
    grouped = defaultdict(list)
    for row in data["rows"]:
        grouped[row["target"]].append(float(row[metric]))
    return {speaker: st.mean(values) for speaker, values in grouped.items()}
