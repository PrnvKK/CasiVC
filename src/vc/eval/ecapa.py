# src/vc/eval/ecapa.py
"""
Fast real-metric margin probe: ECAPA speaker similarity of converted audio to
source vs target. Uses the frozen HiFi-GAN vocoder to obtain waveforms.

margin = cos(ECAPA(vocoder(pred)), ECAPA(target_wav))
       - cos(ECAPA(vocoder(pred)), ECAPA(source_wav))
> 0  => target speaker wins (real conversion)
< 0  => source leakage dominates

Small N, CPU only. Meant as a fast go/no-go metric for training changes.
"""

import os
import sys
import glob
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F

import vc.data.compat
from vc.config import AudioConfig, ModelConfig, TrainingConfig
from vc.model import HubertVCModel
from vc.vocoder.inference import load_vocoder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_ecapa(device="cpu"):
    from speechbrain.inference.classifiers import EncoderClassifier
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/speechbrain_spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    enc.eval()
    return enc


def ecapa_emb(enc, wav):
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    with torch.no_grad():
        e = enc.encode_batch(wav)
    return F.normalize(e.reshape(-1, e.shape[-1]), dim=-1)


def cos(a, b):
    return float(F.cosine_similarity(a, b, dim=-1).mean().item())


def load_cache(cache_dir, min_per_spk=2, max_per_spk=3):
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    by_spk = defaultdict(list)
    for f in files:
        by_spk[os.path.basename(f).split("_")[0]].append(f)
    return {k: v[:max_per_spk] for k, v in by_spk.items() if len(v) >= min_per_spk}


def run(checkpoint, cache_dir, num_speakers, num_content, device="cpu", speakers=None):
    model = HubertVCModel(AudioConfig(), ModelConfig(), TrainingConfig(), load_encoders=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()
    print(f"[eval] {checkpoint} (epoch {ckpt.get('epoch','?')})")

    vocoder = load_vocoder(None, device=device)
    enc = load_ecapa(device)

    by_spk = load_cache(cache_dir)
    if speakers:
        speakers = [str(s) for s in speakers]
        speakers = [s for s in speakers if s in by_spk]
    else:
        speakers = sorted(by_spk.keys())[:num_speakers]
    items = {}
    for s in speakers:
        d = torch.load(by_spk[s][0], map_location=device)
        items[s] = {"spk": d["speaker_feats"].float().reshape(1, 192), "mel": d["gt_mel"].float()}

    rows = []
    with torch.no_grad():
        for A in speakers:
            for item_f in by_spk[A][:num_content]:
                d = torch.load(item_f, map_location=device)
                C = d["content_feats"].float().unsqueeze(0)
                M_i = d["gt_mel"].float().unsqueeze(0)
                S_A = d["speaker_feats"].float().reshape(1, 192)
                wav_src = vocoder(M_i).squeeze(0).cpu()
                e_src = ecapa_emb(enc, wav_src)
                for B in speakers:
                    if B == A:
                        continue
                    S_B = items[B]["spk"]
                    M_j = items[B]["mel"].unsqueeze(0)
                    wav_tgt = vocoder(M_j).squeeze(0).cpu()
                    e_tgt = ecapa_emb(enc, wav_tgt)

                    pred, _, _ = model(precomputed_content_feats=C,
                                       precomputed_speaker_feats=S_B, gt_mels=M_i)
                    wav_conv = vocoder(pred).squeeze(0).cpu()
                    e_conv = ecapa_emb(enc, wav_conv)

                    sim_t = cos(e_conv, e_tgt)
                    sim_s = cos(e_conv, e_src)
                    rows.append((A, B, d["utt_id"][:16], sim_s, sim_t, sim_t - sim_s))

    print(f"\n{'src':>6} {'tgt':>6} {'utt':>17} {'sim_src':>9} {'sim_tgt':>9} {'margin':>9}")
    print("-" * 62)
    for r in rows:
        print(f"{r[0]:>6} {r[1]:>6} {r[2]:>17} {r[3]:>9.4f} {r[4]:>9.4f} {r[5]:>+9.4f}")
    import statistics as st
    ms = [r[5] for r in rows]
    ss = [r[3] for r in rows]
    ts = [r[4] for r in rows]
    print("-" * 62)
    print(f"MEAN sim_source {st.mean(ss):.4f} | sim_target {st.mean(ts):.4f} | margin {st.mean(ms):+.4f}")
    print(f"Positive margins: {sum(1 for m in ms if m>0)}/{len(ms)}")
    return ms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best_heavy.ckpt")
    ap.add_argument("--cache_dir", default="cache/heavy_cache")
    ap.add_argument("--num_speakers", type=int, default=3)
    ap.add_argument("--num_content", type=int, default=1)
    ap.add_argument("--speakers", type=str, default=None, help="Comma-separated speaker IDs")
    args = ap.parse_args()
    spks = args.speakers.split(",") if args.speakers else None
    run(args.checkpoint, args.cache_dir, args.num_speakers, args.num_content, speakers=spks)
