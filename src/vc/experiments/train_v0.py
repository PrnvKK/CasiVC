"""V0 trainer: the objective-identifiability A/B baseline.

Trains the sub-500k architecture from scratch on a speaker-disjoint split under
the repaired validity recipe (``valid`` arm). The retired mel-teacher ``id`` arm
is a hard error — use ``vc.experiments.content_factorial --stage id`` for true
vocoder→ECAPA supervision.

Entry point: ``tools/train_v0.py``.
"""

import os
import sys
import time
import json
import random
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

import vc.data.compat
from vc.config import AudioConfig, ModelConfig, TrainingConfig
from vc.model import HubertVCModel
from vc.vocoder.inference import load_vocoder
from vc.eval.ecapa import load_ecapa, ecapa_emb
from vc.data.splits import group_cache, load_item, make_split, pad_content, pad_mel
from vc.losses.masked import masked_l1, masked_spec_conv
from vc.eval.protocol import evaluate_arm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

A_CFG = AudioConfig()
M_CFG = ModelConfig()
T_CFG = TrainingConfig()


def train_arm(arm, train_items, by_spk_train, seed, epochs, batch_size, landmark=None, teacher=None,
              w_id=20.0):
    random.seed(seed)
    torch.manual_seed(seed)
    snames = sorted(by_spk_train.keys())
    rng = random.Random(seed)
    # map speaker -> train item indices
    spk_to_idx = defaultdict(list)
    for i, it in enumerate(train_items):
        spk_to_idx[it["spk"]].append(i)

    model = HubertVCModel(A_CFG, M_CFG, T_CFG, load_encoders=False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    n = len(train_items)
    print(f"[{arm}] train utts={n} speakers={len(snames)} epochs={epochs}")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        order = list(range(n))
        rng.shuffle(order)
        acc, nb = 0.0, 0
        for s in range(0, n, batch_size):
            idxs = order[s:s + batch_size]
            if len(idxs) < 2:
                continue
            items = [train_items[i] for i in idxs]
            C, clens = pad_content(items)
            M, mlens = pad_mel([it["mel"] for it in items])

            # conditioning speaker embeddings
            if arm in ("ref", "id"):
                S = []
                for it in items:
                    same = spk_to_idx[it["spk"]]
                    j = same[rng.randrange(len(same))]
                    S.append(train_items[j]["speaker"])
                S = torch.stack(S)
            else:
                S = torch.stack([it["speaker"] for it in items])

            if arm == "base":
                pred, _, _ = model(precomputed_content_feats=C, precomputed_speaker_feats=S, gt_mels=M)
                loss = 45.0 * F.l1_loss(pred, M) + 2.0 * (
                    torch.norm(M - pred, p="fro") / (torch.norm(M, p="fro") + 1e-6))
            else:
                pred, _, _ = model(precomputed_content_feats=C, precomputed_speaker_feats=S,
                                   target_lengths=mlens, content_lengths=clens)
                loss = 45.0 * masked_l1(pred, M, mlens) + 2.0 * masked_spec_conv(pred, M, mlens)

            if arm == "id" and teacher is not None:
                # cross-speaker identity pairs
                cross_idx = []
                for it in items:
                    for _ in range(6):
                        k = rng.randrange(n)
                        if train_items[k]["spk"] != it["spk"]:
                            cross_idx.append(k)
                            break
                citems = [train_items[k] for k in cross_idx]
                Sx = torch.stack([it["speaker"] for it in citems])
                pred_x, _, _ = model(precomputed_content_feats=C, precomputed_speaker_feats=Sx,
                                     target_lengths=mlens, content_lengths=clens)
                # teacher-space margin: pred must be closer to cross-speaker than source speaker
                with torch.no_grad():
                    Mx, Lx = pad_mel([it["mel"] for it in citems])
                    t_tgt = F.normalize(teacher(Mx, Lx), dim=-1)
                    t_src = F.normalize(teacher(M, mlens), dim=-1)
                t_pred = F.normalize(teacher(pred_x, mlens), dim=-1)
                cos_tgt = F.cosine_similarity(t_pred, t_tgt, dim=-1)
                cos_src = F.cosine_similarity(t_pred, t_src, dim=-1)
                loss_id = (1.0 - cos_tgt).mean() + F.relu(cos_src - cos_tgt + 0.2).mean()
                loss = loss + w_id * loss_id

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc += loss.item()
            nb += 1
        sched.step()
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[{arm}] ep {ep:3d}/{epochs} loss={acc/max(1,nb):.4f} ({time.time()-t0:.0f}s)")
    ckpt = os.path.join("checkpoints", f"ident_{arm}.ckpt")
    torch.save({"model_state": model.state_dict(), "arm": arm, "epochs": epochs}, ckpt)
    print(f"[{arm}] saved -> {ckpt} ({time.time()-t0:.0f}s)")
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache/heavy_cache")
    ap.add_argument("--arms", default="base,valid,ref,id")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_val_spk", type=int, default=16)
    ap.add_argument("--w_id", type=float, default=20.0)
    ap.add_argument("--teacher", default="checkpoints/mel_ecapa_teacher.ckpt")
    ap.add_argument("--out", default="eval/V0/identifiability_results.json")
    ap.add_argument("--n_targets", type=int, default=0, help="0 = all held-out targets")
    ap.add_argument("--n_src", type=int, default=4)
    ap.add_argument("--n_content", type=int, default=2)
    ap.add_argument("--n_ref", type=int, default=2)
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    by_spk = group_cache(args.cache_dir)
    train_spks, val_spks = make_split(by_spk, args.seed, args.num_val_spk)
    print(f"[split] train={len(train_spks)} val={len(val_spks)}")
    print(f"[split] val speakers={val_spks}")

    train_items = [load_item(f) for s in train_spks for f in by_spk[s]]
    by_spk_train = {s: by_spk[s] for s in train_spks}
    val_items_by_spk = {s: [load_item(f) for f in by_spk[s]] for s in val_spks}

    teacher = None
    if "id" in arms:
        # The mel->ECAPA teacher arm was REJECTED (kill_log K-001). The teacher
        # script is archived in _archive/scripts/. Use content_factorial --stage id
        # for true vocoder->ECAPA identity supervision instead.
        raise SystemExit(
            "[train_v0] the mel->ECAPA teacher 'id' arm is retired (kill_log K-001). "
            "Use tools/content_factorial.py --stage id.")

    results = {}
    models = {}
    for arm in arms:
        models[arm] = train_arm(arm, train_items, by_spk_train, args.seed, args.epochs,
                                args.batch_size, landmark=None, teacher=teacher, w_id=args.w_id)

    if args.skip_eval:
        print("\n[skip_eval] training done; skipping benchmark eval.")
        return

    # Evaluation setup
    print("\n[eval] loading vocoder + ECAPA...")
    vocoder = load_vocoder(None, device="cpu")
    enc = load_ecapa("cpu")
    ref_cache = {}
    with torch.no_grad():
        for s in val_spks:
            d = val_items_by_spk[s][0]
            wav = vocoder(d["mel"].unsqueeze(0)).squeeze(0).cpu()
            ref_cache[s] = ecapa_emb(enc, wav)

    print("\n=== IDENTIFIABILITY A/B (held-out speakers) ===")
    for arm in arms:
        model = models[arm][0]
        print(f"[{arm}]")
        results[arm] = evaluate_arm(model, val_items_by_spk, vocoder, enc, ref_cache,
                                    n_src=args.n_src, n_content=args.n_content,
                                    n_ref=args.n_ref, n_targets=args.n_targets)

    with open(args.out, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in results.items()}, f, indent=2)
    print(f"\n[eval] saved -> {args.out}")


if __name__ == "__main__":
    main()
