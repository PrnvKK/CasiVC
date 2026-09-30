"""Cache loading, speaker-disjoint splits, and batch padding.

Extracted from the original ``exp_identifiability.py`` shared-utilities section.
"""

import os
import glob
import random
from collections import defaultdict

import torch


def group_cache(cache_dir):
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    by_spk = defaultdict(list)
    for f in files:
        by_spk[os.path.basename(f).split("_")[0]].append(f)
    return by_spk


def load_item(path):
    d = torch.load(path, map_location="cpu")
    # content/mel stay at their on-disk dtype (fp16) to cut in-RAM pool size ~2.5x;
    # pad_content/pad_mel allocate fp32 and upcast losslessly (cache IS fp16).
    return {
        "content": d["content_feats"],
        "speaker": d["speaker_feats"].float().reshape(192),
        "mel": d["gt_mel"],
        "id": d["utt_id"],
        "spk": os.path.basename(path).split("_")[0],
        "path": path,
    }


def make_split(by_spk, seed=0, num_val_spk=16):
    speakers = sorted(by_spk.keys())
    rng = random.Random(seed)
    rng.shuffle(speakers)
    val_spks = sorted(speakers[:num_val_spk])
    train_spks = sorted(speakers[num_val_spk:])
    return train_spks, val_spks


def pad_content(items):
    maxT = max(it["content"].shape[0] for it in items)
    C = torch.zeros(len(items), maxT, 768)
    L = torch.zeros(len(items))
    for i, it in enumerate(items):
        T = it["content"].shape[0]
        C[i, :T] = it["content"]
        L[i] = T
    return C, L


def pad_mel(mels):
    maxT = max(m.shape[1] for m in mels)
    M = torch.zeros(len(mels), 80, maxT)
    L = torch.zeros(len(mels))
    for i, m in enumerate(mels):
        T = m.shape[1]
        M[i, :, :T] = m
        L[i] = T
    return M, L
