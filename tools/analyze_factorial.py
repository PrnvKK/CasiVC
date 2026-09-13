# analyze_factorial.py
"""
Paired target-level bootstrap over saved arm results (n=240 protocol).

Compares arms on identical eval pairs: report mean paired difference,
95% percentile-bootstrap CI (4000 resamples over target speakers),
and number of target speakers where the difference is positive.

Usage:
  python analyze_factorial.py                          # canonical 4-arm table
  python analyze_factorial.py --a X.json --b Y.json    # any single pairing
  python analyze_factorial.py --labels a,b,c --files f1,f2,f3   # table only
"""
import argparse
import json
import random
import statistics as st


def load(p):
    """Load a results JSON. Supports 'file.json:key' for multi-arm files
    (e.g. v1_true_ecapa_results.json holds both 'v0' and 'v1')."""
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


def paired(a, b, name):
    """Paired mean difference b - a with target-level bootstrap CI."""
    common = sorted(set(a) & set(b))
    td = [st.mean(b[t]) - st.mean(a[t]) for t in common]
    rng = random.Random(0)
    boot = []
    for _ in range(4000):
        s = [td[rng.randrange(len(td))] for _ in range(len(td))]
        boot.append(st.mean(s))
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    print(f"{name}: mean {st.mean(td):+.4f} CI [{lo:+.4f},{hi:+.4f}] "
          f"wins {sum(1 for x in td if x > 0)}/{len(td)}")
    return st.mean(td), lo, hi


def summary(name, v):
    print(f"{name:6} margin {v['margin']:+.4f} CI [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
          f"src {v['sim_src']:.4f} tgt {v['sim_tgt']:.4f} L1 {v['content_l1']:.4f}")


CANONICAL = [
    ("V0", "eval/EXP-001-local-content-suppression/v0_full_eval.json"),
    ("VQ", "eval/EXP-001-local-content-suppression/cf_vq_fixed_n240.json"),
    ("VQ+id", "eval/EXP-002-local-identity-supervision/cf_vq_fixed_id_n240.json"),
]
# Frozen evidence lives under eval/ (never edited). Falsified-pilot and legacy
# JSONs live in _archive/results/. See eval/INDEX.md for the full map.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="",
                    help="comma-separated arm names for the summary table")
    ap.add_argument("--files", default="",
                    help="comma-separated result JSONs (same order as --labels)")
    ap.add_argument("--a", default="", help="file A for a single paired test")
    ap.add_argument("--b", default="", help="file B for a single paired test")
    args = ap.parse_args()

    if args.a and args.b:
        ta, _ = load(args.a)
        tb, _ = load(args.b)
        paired(ta, tb, f"{args.b} - {args.a}")
        return

    if args.labels and args.files:
        labels = args.labels.split(",")
        files = args.files.split(",")
        arms = list(zip(labels, files))
    else:
        arms = CANONICAL

    loaded = []
    for name, f in arms:
        pt, s = load(f)
        summary(name, s)
        loaded.append((name, pt))
    print()
    for i in range(1, len(loaded)):
        paired(loaded[i - 1][1], loaded[i][1],
               f"{loaded[i][0]} - {loaded[i - 1][0]}")


if __name__ == "__main__":
    main()
