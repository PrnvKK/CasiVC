"""Trivial control baseline: 'conversion' = vocode the source's own GT mel.

No speaker transformation at all. Establishes the paper's lower control: the
source-copy margin should be <= 0 (output stays the source speaker), and it
anchors what "no conversion" scores.

  python baselines/source_copy.py --cache_dir cache/heavy_cache
"""
import os
import sys
import argparse
import random
import statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
from tqdm import tqdm

import vc.data.compat
from vc.data.splits import group_cache, load_item, make_split
from vc.vocoder.inference import load_vocoder
from vc.eval.ecapa import load_ecapa, ecapa_emb, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache/heavy_cache")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--num_val_spk", type=int, default=16)
    ap.add_argument("--n_src", type=int, default=4)
    ap.add_argument("--n_content", type=int, default=2)
    ap.add_argument("--n_ref", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    by_spk = group_cache(args.cache_dir)
    _, val_spks = make_split(by_spk, args.split_seed, args.num_val_spk)
    val = {s: [load_item(f) for f in by_spk[s]] for s in val_spks}
    vocoder = load_vocoder(None, device=args.device)
    enc = load_ecapa(args.device)

    ref = {}
    with torch.no_grad():
        for s in val_spks:
            wav = vocoder(val[s][0]["mel"].unsqueeze(0).to(args.device)).squeeze(0)
            ref[s] = ecapa_emb(enc, wav)

    srcs = sorted(val_spks)[:args.n_src]
    per_target = {}
    with torch.no_grad():
        for T in tqdm(sorted(val_spks), desc="source-copy", unit="target"):
            ms = []
            for S in srcs:
                if S == T:
                    continue
                for c in val[S][:args.n_content]:
                    wav = vocoder(c["mel"].unsqueeze(0).to(args.device)).squeeze(0)
                    e = ecapa_emb(enc, wav)
                    ms.append(cos(e, ref[T]) - cos(e, ref[S]))
            per_target[T] = ms

    allm = [m for v in per_target.values() for m in v]
    tmeans = [st.mean(v) for v in per_target.values()]
    rng = random.Random(0)
    boot = sorted(st.mean([tmeans[rng.randrange(len(tmeans))] for _ in tmeans])
                  for _ in range(4000))
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    print(f"[source-copy] margin {st.mean(allm):+.4f} [95% CI {lo:+.4f},{hi:+.4f}] "
          f"pos={sum(m > 0 for m in allm)}/{len(allm)} n_targets={len(tmeans)}")
    print("=> trivial lower control: no conversion should keep the source speaker.")


if __name__ == "__main__":
    main()
