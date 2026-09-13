"""Paired bootstrap of per-target component metrics stored in result rows."""
import argparse
import json
import random
import statistics as st
from collections import defaultdict


def target_means(data, metric):
    grouped = defaultdict(list)
    for row in data["rows"]:
        grouped[row["target"]].append(float(row[metric]))
    return {speaker: st.mean(values) for speaker, values in grouped.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--metrics", default="margin,sim_tgt,sim_src,content_l1")
    parser.add_argument("--boot", type=int, default=4000)
    args = parser.parse_args()
    a = json.load(open(args.a, encoding="utf-8"))
    b = json.load(open(args.b, encoding="utf-8"))
    rng = random.Random(0)
    for metric in args.metrics.split(","):
        ma, mb = target_means(a, metric), target_means(b, metric)
        speakers = sorted(set(ma) & set(mb))
        differences = [mb[s] - ma[s] for s in speakers]
        boot = sorted(st.mean([differences[rng.randrange(len(differences))]
                               for _ in differences]) for _ in range(args.boot))
        lo = boot[int(0.025 * args.boot)]
        hi = boot[int(0.975 * args.boot)]
        wins = sum(d > 0 for d in differences)
        print(f"{metric}: {st.mean(differences):+.4f} CI [{lo:+.4f},{hi:+.4f}] "
              f"wins {wins}/{len(speakers)}")


if __name__ == "__main__":
    main()
