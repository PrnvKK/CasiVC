"""Evaluation protocol: held-out-speaker margin with target-level bootstrap.

The canonical benchmark. Deterministic given (n_src, n_content, n_ref), so arms
are comparable only at identical settings.
"""

import random
import statistics as st

import torch
from tqdm import tqdm

from vc.losses.masked import masked_l1
from vc.eval.ecapa import ecapa_emb, cos


def evaluate_arm(model, val_items_by_spk, vocoder, enc, ref_cache, n_src=4, n_content=2, n_ref=2,
                 n_targets=0):
    """Speaker-level margin on held-out speakers. Returns per-target margins."""
    model.eval()
    targets = sorted(val_items_by_spk.keys())
    if n_targets:
        targets = targets[:n_targets]
    srcs = sorted(val_items_by_spk.keys())[:n_src]
    per_target = {}
    rows = []
    pbar = tqdm(targets, desc="eval", unit="target", leave=False)
    with torch.no_grad():
        for T in pbar:
            tm = []
            refs = val_items_by_spk[T][:n_ref]
            for Sspk in srcs:
                if Sspk == T:
                    continue
                contents = val_items_by_spk[Sspk][:n_content]
                for r in refs:
                    S_emb = r["speaker"].unsqueeze(0)
                    for c in contents:
                        C = c["content"].unsqueeze(0)
                        ml = torch.tensor([c["mel"].shape[1]]).float()
                        cl = torch.tensor([c["content"].shape[0]]).float()
                        pred, _, _ = model(precomputed_content_feats=C,
                                           precomputed_speaker_feats=S_emb,
                                           target_lengths=ml, content_lengths=cl)
                        Tl = c["mel"].shape[1]
                        cl1 = masked_l1(pred[:, :, :Tl], c["mel"].unsqueeze(0), ml).item()
                        wav = vocoder(pred).squeeze(0)  # stays on the eval device
                        e = ecapa_emb(enc, wav)
                        sim_t = cos(e, ref_cache[T])
                        sim_s = cos(e, ref_cache[Sspk])
                        m = sim_t - sim_s
                        tm.append(m)
                        rows.append((T, Sspk, sim_s, sim_t, m, cl1))
            per_target[T] = tm
    allm = [x for v in per_target.values() for x in v]
    mean_m = st.mean(allm)
    mean_st = st.mean([r[3] for r in rows])
    mean_ss = st.mean([r[2] for r in rows])
    # bootstrap by target speaker
    tmeans = [st.mean(v) for v in per_target.values()]
    rng = random.Random(0)
    boot = []
    for _ in range(2000):
        sample = [tmeans[rng.randrange(len(tmeans))] for _ in range(len(tmeans))]
        boot.append(st.mean(sample))
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
    pos = sum(1 for m in allm if m > 0)
    mean_cl1 = st.mean([r[5] for r in rows])
    print(f"  margin={mean_m:+.4f} [95% CI {lo:+.4f},{hi:+.4f}] | sim_src={mean_ss:.4f} sim_tgt={mean_st:.4f} | content_L1={mean_cl1:.4f} | pos={pos}/{len(allm)} | n_targets={len(tmeans)}")
    return {"margin": mean_m, "ci": [lo, hi], "sim_src": mean_ss, "sim_tgt": mean_st,
            "content_l1": mean_cl1, "pos": pos, "n": len(allm), "per_target": per_target, "rows": rows}
