"""Content quantization: K-means codebook fit/apply for HuBERT features.

Extracted from the original ``exp_content_factorial.py`` so that the training
recipe and the quantization live in different modules.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def kmeans_torch(X, K, iters=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(X.size(0), generator=g)[:K]].clone()
    pbar = tqdm(range(iters), desc="kmeans", unit="iter", leave=False)
    for it in pbar:
        a = torch.cdist(X, C).argmin(dim=1)
        onehot = F.one_hot(a, K).float()
        cnt = onehot.sum(0)
        dead = (cnt == 0).nonzero(as_tuple=True)[0]
        if len(dead):
            # re-seed empty centroids from random data points (a zero-vector
            # centroid would attract low-energy frames and erase their content)
            C[dead] = X[torch.randint(X.size(0), (len(dead),), generator=g)]
            pbar.set_postfix(dead=len(dead))
            continue
        C = (onehot.t() @ X) / cnt[:, None]
    pbar.close()
    a = torch.cdist(X, C).argmin(dim=1)
    n_dead = int((F.one_hot(a, K).float().sum(0) == 0).sum())
    print(f"[kmeans] K={K} iters={iters} dead_clusters={n_dead}")
    return C


def build_codebook(items, K, iters, seed, max_frames, mean_norm, device="cpu"):
    """Fit K-means in the SAME space used at quantization time.

    With mean_norm=True, every frame is pooled from per-utterance mean-centered
    features, matching apply_content(). With mean_norm=False (legacy), raw frames
    are pooled — that mismatched space is kept only for reproducing pre-fix arms.
    On CUDA the fit runs on the GPU (CPU cdist is minutes on a 2-vCPU Colab).
    """
    g = torch.Generator().manual_seed(seed)
    # Subsample per-utterance BEFORE concatenating: cat of the full pool was
    # ~8 GB fp32 on train-clean-100 and OOM-killed Colab. Proportional
    # per-speaker-free stratified sampling keeps the matrix at ~max_frames.
    totals = [it["content"].shape[0] for it in items]
    total = sum(totals)
    chunks = []
    for it, T in zip(items, totals):
        c = it["content"].float()
        if mean_norm:
            c = c - c.mean(dim=0, keepdim=True)
        if total > max_frames:
            n_i = max(1, round(max_frames * T / total))
            if n_i < T:
                c = c[torch.randperm(T, generator=g)[:n_i]]
        chunks.append(c)
    X = torch.cat(chunks, dim=0)
    if X.shape[0] > max_frames:
        X = X[torch.randperm(X.shape[0], generator=g)[:max_frames]]
    if device.startswith("cuda"):
        X = X.to(device)
    C = kmeans_torch(X, K, iters=iters, seed=seed)
    return C.cpu()


def apply_content(items, mode, codebook, mean_norm, device="cpu"):
    cb = None
    if mode == "vq" and codebook is not None and device.startswith("cuda"):
        cb = codebook.to(device)
    for it in items:
        out_dtype = it["content"].dtype
        c = it["content"].float()
        if mean_norm:
            c = c - c.mean(dim=0, keepdim=True)
        if mode == "vq":
            if cb is not None:
                a = torch.cdist(c.to(device), cb).argmin(dim=1)
                c = cb[a].cpu()
            else:
                a = torch.cdist(c, codebook).argmin(dim=1)
                c = codebook[a]
        it["content"] = c.to(out_dtype)
