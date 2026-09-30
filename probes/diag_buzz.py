# diag_buzz.py
"""
Numeric localization of conversion artifacts (buzz/twang).

For the same deterministic audition tuples as make_samples.py, computes per
checkpoint:
  - per-band masked L1 vs GT mel (low / mid / high thirds of the 80 mel bins)
  - temporal jitter: mean |second difference| along time (pred vs GT) —
    high-frequency frame-to-frame instability reads as buzz
  - global masked L1 and mel Pearson corr (cross-check vs make_samples)

Buzz that is arm-independent points at the decoder/resampler, not content.

Usage:
  python diag_buzz.py --inits checkpoints/ident_valid.ckpt,checkpoints/cf_km_id.ckpt `
      --labels v0,km_id
"""
import os
import sys
import argparse

import torch

from vc.config import AudioConfig, ModelConfig, TrainingConfig
from vc.model import HubertVCModel
from vc.losses.masked import masked_l1
from vc.experiments.content_factorial import load_split, prepare_content

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

A_CFG = AudioConfig()
M_CFG = ModelConfig()
T_CFG = TrainingConfig()
NMEL = 80


def band_l1(pred, tgt, lo, hi):
    L = int(tgt.shape[-1])
    lens = torch.tensor([L]).float()
    return masked_l1(pred[:, lo:hi, :L], tgt[:, lo:hi, :], lens).item()


def temporal_jitter(mel, L):
    """mean |2nd difference| along time, per frame, over valid frames."""
    x = mel[:, :, :L]
    d2 = x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]
    return d2.abs().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--num_val_spk", type=int, default=16)
    ap.add_argument("--n_pairs", type=int, default=4)
    ap.add_argument("--n_src", type=int, default=4)
    ap.add_argument("--n_ref", type=int, default=2)
    ap.add_argument("--cache_dir", default="cache/heavy_cache")
    args = ap.parse_args()

    labels = args.labels.split(",")
    paths = args.inits.split(",")
    assert len(labels) == len(paths)

    load_ns = argparse.Namespace(cache_dir=args.cache_dir,
                                 split_seed=args.split_seed,
                                 num_val_spk=args.num_val_spk)
    _, _, _, _, val_by_spk = load_split(load_ns)
    srcs = sorted(val_by_spk.keys())[:args.n_src]

    def build_tuples(vb):
        tuples = []
        for T in sorted(vb.keys()):
            for r in vb[T][:args.n_ref]:
                for S in srcs:
                    if S == T:
                        continue
                    tuples.append((T, S, vb[S][0], r))
        return tuples[:args.n_pairs]

    print(f"{'arm':10} {'L1':>7} {'L1low':>7} {'L1mid':>7} {'L1hi':>7} "
          f"{'jit':>7} {'GTjit':>7}")
    results = {}
    for name, path in zip(labels, paths):
        ck = torch.load(path, map_location="cpu")
        content = ck.get("content", "continuous")
        model = HubertVCModel(A_CFG, M_CFG, T_CFG, load_encoders=False)
        model.load_state_dict(ck["model_state"], strict=False)
        model.eval()
        _, _, _, train_items, val_by_spk_c = load_split(load_ns)
        prepare_content(argparse.Namespace(
            content=content, mean_norm=bool(ck.get("mean_norm")),
            legacy_codebook=False, codebook=256, km_iters=25, km_frames=80000,
            split_seed=args.split_seed), train_items, val_by_spk_c, ck.get("codebook"),
            ck.get("codebook_normed"))
        tuples = build_tuples(val_by_spk_c)

        agg = {"l1": [], "lo": [], "mid": [], "hi": [], "jit": [], "gtjit": []}
        with torch.no_grad():
            for T, S, c_item, r in tuples:
                c = c_item["content"].unsqueeze(0)
                ml = torch.tensor([c_item["mel"].shape[1]]).float()
                cl = torch.tensor([c_item["content"].shape[0]]).float()
                pred, _, _ = model(precomputed_content_feats=c,
                                   precomputed_speaker_feats=r["speaker"].unsqueeze(0),
                                   target_lengths=ml, content_lengths=cl)
                gt = c_item["mel"].unsqueeze(0)
                L = int(gt.shape[2])
                agg["l1"].append(masked_l1(pred[:, :, :L], gt, ml).item())
                agg["lo"].append(band_l1(pred, gt, 0, 27))
                agg["mid"].append(band_l1(pred, gt, 27, 54))
                agg["hi"].append(band_l1(pred, gt, 54, NMEL))
                agg["jit"].append(temporal_jitter(pred, L))
                agg["gtjit"].append(temporal_jitter(gt, L))
        m = {k: sum(v) / len(v) for k, v in agg.items()}
        results[name] = m
        print(f"{name:10} {m['l1']:7.4f} {m['lo']:7.4f} {m['mid']:7.4f} "
              f"{m['hi']:7.4f} {m['jit']:7.4f} {m['gtjit']:7.4f}")
    return results


if __name__ == "__main__":
    main()

