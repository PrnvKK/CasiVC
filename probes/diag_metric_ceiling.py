import sys
sys.stdout.reconfigure(encoding="utf-8")
import torch, glob, os
import torch.nn.functional as F
import vc.data.compat
from vc.data.splits import group_cache, load_item, make_split
from vc.vocoder.inference import load_vocoder
from vc.eval.ecapa import load_ecapa, ecapa_emb, cos

by_spk = group_cache("cache/heavy_cache")
train_spks, val_spks = make_split(by_spk, 0, 16)
vocoder = load_vocoder(None, device="cpu")
enc = load_ecapa("cpu")

sims = []
with torch.no_grad():
    for s in val_spks:
        items = [load_item(f) for f in by_spk[s]]
        ref = ecapa_emb(enc, vocoder(items[0]["mel"].unsqueeze(0)).squeeze(0).cpu())
        for other in items[1:3]:
            e = ecapa_emb(enc, vocoder(other["mel"].unsqueeze(0)).squeeze(0).cpu())
            sims.append(cos(e, ref))

print(f"cross-utterance same-speaker ECAPA cosine (16 held-out spk, {len(sims)} pairs)")
print(f"  mean {sum(sims)/len(sims):.4f}  min {min(sims):.4f}  max {max(sims):.4f}")
print(f"  => protocol ceiling for sim_tgt (perfect conversion, different-ref): ~{sum(sims)/len(sims):.2f}")
