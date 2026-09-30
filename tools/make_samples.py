# make_samples.py
"""
Vocode held-out conversions from one or more checkpoints for auditioning.

For each of the first `--n_pairs` conversions of the deterministic n=240 eval
enumeration (same order as evaluate_arm), vocodes:
  - the source ground-truth mel            (content ground truth)
  - each checkpoint's converted output     (named via --labels/--inits)
  - the target reference mel               (what the conversion should sound like)

Also reports an intelligibility proxy per arm:
  content_corr  = Pearson corr(pred mel, source GT mel) over valid frames
  content_L1    = masked mel L1 (same as the eval metric)

Usage:
  python make_samples.py --inits checkpoints/ident_valid.ckpt,checkpoints/cf_vq_fixed_id.ckpt `
      --labels v0,vq_id --out_dir samples --n_pairs 4
"""
import os
import sys
import argparse
import json

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from vc.config import AudioConfig, ModelConfig, TrainingConfig
from vc.model import HubertVCModel
from vc.vocoder.inference import load_vocoder
from vc.losses.masked import masked_l1
from vc.experiments.content_factorial import load_split, prepare_content

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

A_CFG = AudioConfig()
M_CFG = ModelConfig()
T_CFG = TrainingConfig()


def save_wav(path, wav, sr):
    """wav: [1, T] or [T] float tensor -> wav file via soundfile."""
    data = wav.detach().cpu().numpy().squeeze()
    sf.write(path, data, sr)


def masked_pearson(pred, tgt, length):
    """Frame-wise Pearson correlation between pred and GT mel over valid frames."""
    if tgt.dim() == 2:
        tgt = tgt.unsqueeze(0)
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    L = min(int(pred.shape[-1]), int(tgt.shape[-1]))
    p = pred[0, :, :L] - pred[0, :, :L].mean(dim=1, keepdim=True)
    t = tgt[0, :, :L] - tgt[0, :, :L].mean(dim=1, keepdim=True)
    num = (p * t).sum()
    den = (p.pow(2).sum().sqrt() * t.pow(2).sum().sqrt()).clamp(min=1e-8)
    return (num / den).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", required=True, help="comma-separated checkpoint paths")
    ap.add_argument("--labels", required=True, help="comma-separated arm names")
    ap.add_argument("--out_dir", default="samples")
    ap.add_argument("--cache_dir", default="cache/heavy_cache")
    ap.add_argument("--content", default="", help="override content mode (else from ckpt)")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--num_val_spk", type=int, default=16)
    ap.add_argument("--n_pairs", type=int, default=4,
                    help="number of (target, source, content, ref) tuples to vocode")
    ap.add_argument("--n_src", type=int, default=4)
    ap.add_argument("--n_ref", type=int, default=2)
    ap.add_argument("--metrics_json", default="")
    ap.add_argument("--smooth", type=int, default=0,
                    help="moving-average window (frames) over quantized content "
                         "at inference (0 = off; test of quantization-artifact fix)")
    args = ap.parse_args()

    labels = args.labels.split(",")
    paths = args.inits.split(",")
    assert len(labels) == len(paths), "--labels and --inits must match"
    os.makedirs(args.out_dir, exist_ok=True)

    load_ns = argparse.Namespace(cache_dir=args.cache_dir,
                                 split_seed=args.split_seed,
                                 num_val_spk=args.num_val_spk)
    _, _, _, _, val_by_spk = load_split(load_ns)

    vocoder = load_vocoder(None, device="cpu")
    srcs = sorted(val_by_spk.keys())[:args.n_src]

    def build_tuples(vb):
        """Deterministic enumeration mirroring evaluate_arm, over PREPARED items."""
        tuples = []
        for T in sorted(vb.keys()):
            refs = vb[T][:args.n_ref]
            for S in srcs:
                if S == T:
                    continue
                for r in refs:
                    tuples.append((T, S, vb[S][0], r))
        return tuples[:args.n_pairs]

    print(f"[samples] {len(build_tuples(val_by_spk))} conversion tuples from {len(srcs)} sources")

    metrics = {}
    for name, path in zip(labels, paths):
        ck = torch.load(path, map_location="cpu")
        content = args.content or ck.get("content", "continuous")
        model = HubertVCModel(A_CFG, M_CFG, T_CFG, load_encoders=False)
        model.load_state_dict(ck["model_state"], strict=False)
        model.eval()
        _, _, _, train_items, val_by_spk_c = load_split(load_ns)
        prepare_content(argparse.Namespace(
            content=content, mean_norm=bool(ck.get("mean_norm")),
            legacy_codebook=False, codebook=256, km_iters=25, km_frames=80000,
            split_seed=args.split_seed), train_items, val_by_spk_c, ck.get("codebook"),
            ck.get("codebook_normed"))

        if args.smooth > 1:
            k = args.smooth
            pad = (k - 1) // 2
            for s in val_by_spk_c:
                for it in val_by_spk_c[s]:
                    c = it["content"].t().unsqueeze(0)              # [1, 768, T]
                    c = F.pad(c, (pad, pad), mode="replicate")
                    it["content"] = F.avg_pool1d(c, k, stride=1).squeeze(0).t()
            print(f"[samples] smoothing quantized content with window {k}")

        tuples = build_tuples(val_by_spk_c)
        rows = []
        for k, (T, S, c_item, r) in enumerate(tuples):
            tag = f"T{T}_S{S}_{k}"
            c = c_item["content"].unsqueeze(0)
            ml = torch.tensor([c_item["mel"].shape[1]]).float()
            cl = torch.tensor([c_item["content"].shape[0]]).float()
            L = int(c_item["mel"].shape[1])
            with torch.no_grad():
                pred, _, _ = model(precomputed_content_feats=c,
                                   precomputed_speaker_feats=r["speaker"].unsqueeze(0),
                                   target_lengths=ml, content_lengths=cl)
                l1 = masked_l1(pred[:, :, :L], c_item["mel"].unsqueeze(0), ml).item()
                corr = masked_pearson(pred, c_item["mel"], ml)
                save_wav(os.path.join(args.out_dir, f"{tag}_{name}.wav"),
                         vocoder(pred), A_CFG.sample_rate)
            rows.append({"tag": tag, "l1": l1, "corr": corr})
        metrics[name] = {"mean_l1": sum(r["l1"] for r in rows) / len(rows),
                         "mean_corr": sum(r["corr"] for r in rows) / len(rows),
                         "rows": rows}
        print(f"[samples] {name}: mean_L1={metrics[name]['mean_l1']:.4f} "
              f"mean_corr={metrics[name]['mean_corr']:.4f}")

    # content ground truth + target reference audio (once, same for all arms)
    for k, (T, S, c_item, r) in enumerate(tuples):
        tag = f"T{T}_S{S}_{k}"
        with torch.no_grad():
            save_wav(os.path.join(args.out_dir, f"{tag}_srcGT.wav"),
                     vocoder(c_item["mel"].unsqueeze(0)), A_CFG.sample_rate)
            save_wav(os.path.join(args.out_dir, f"{tag}_tgt_ref.wav"),
                     vocoder(r["mel"].unsqueeze(0)), A_CFG.sample_rate)

    if args.metrics_json:
        with open(args.metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)
    print(f"[samples] saved wavs -> {args.out_dir}/")


if __name__ == "__main__":
    main()
