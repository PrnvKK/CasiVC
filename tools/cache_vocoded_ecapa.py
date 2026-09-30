# cache_vocoded_ecapa.py
"""
Precompute the EXACT evaluation-domain identity target for W4a:

    target(utt) = ECAPA( HiFi-GAN( log-mel(utt) ) )

so a mel-domain teacher can be trained against the same quantity the benchmark
actually measures (instead of ECAPA of raw audio, which lives in a different
domain). Mels are cropped to a fixed number of frames for speed; ECAPA is
length-invariant.

Saves cache/vocoded_ecapa.pt = {utt_id: tensor[192]} plus the split used.
"""

import os
import sys
import glob
import time
import random
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

import vc.data.compat
from vc.vocoder.inference import load_vocoder
from vc.eval.ecapa import load_ecapa
from vc.data.splits import group_cache, make_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache/heavy_cache")
    ap.add_argument("--out", default="cache/vocoded_ecapa.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_val_spk", type=int, default=16)
    ap.add_argument("--per_spk_train", type=int, default=50)
    ap.add_argument("--per_spk_val", type=int, default=20)
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    by_spk = group_cache(args.cache_dir)
    train_spks, val_spks = make_split(by_spk, args.seed, args.num_val_spk)

    # deterministic per-speaker selection
    rng = random.Random(args.seed)
    selection = []
    for s in train_spks:
        files = by_spk[s][:]
        rng.shuffle(files)
        selection += files[: args.per_spk_train]
    for s in val_spks:
        files = by_spk[s][:]
        rng.shuffle(files)
        selection += files[: args.per_spk_val]
    print(f"[voca] speakers train={len(train_spks)} val={len(val_spks)} | utts={len(selection)}")

    vocoder = load_vocoder(None, device=args.device)
    enc = load_ecapa(args.device)

    # load mels + ids, sort by length for efficient batching
    items = []
    for f in selection:
        d = torch.load(f, map_location="cpu")
        m = d["gt_mel"].float()
        T = min(args.crop, m.shape[1])
        items.append((d["utt_id"], m[:, :T]))
    items.sort(key=lambda x: x[1].shape[1])

    out = {}
    t0 = time.time()
    pbar = tqdm(range(0, len(items), args.batch), desc="vocoded-ecapa", unit="batch")
    with torch.no_grad():
        for i in pbar:
            grp = items[i:i + args.batch]
            L = max(m.shape[1] for _, m in grp)
            batch = torch.stack([F.pad(m, (0, L - m.shape[1])) for _, m in grp]).to(args.device)
            wavs = vocoder(batch)  # [B, T_wav]
            if wavs.dim() == 3:
                wavs = wavs.squeeze(1)
            hop = 256
            for j, (uid, m) in enumerate(grp):
                tl = m.shape[1] * hop
                w = wavs[j, :tl].unsqueeze(0)
                e = enc.encode_batch(w)
                out[uid] = F.normalize(e.reshape(-1, e.shape[-1]).squeeze(0), dim=-1).cpu()
            pbar.set_postfix(saved=len(out))

    torch.save({"targets": out, "train_spks": train_spks, "val_spks": val_spks,
                "crop": args.crop}, args.out)
    print(f"[voca] saved {len(out)} targets -> {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
