# src/vc/experiments/content_factorial.py
"""
Clean 2x2 factorial for the content-suppression lever.

    content  in {continuous, vq}     (vq = per-utterance mean-norm + K-means)
    id_loss  in {0, 1}               (true frozen HiFi-GAN -> ECAPA supervision)

ALL arms use the REPAIRED training path:
  - per-sample target mel lengths  (no batch-max warp)
  - per-sample content lengths     (no source-side batch-max warp)
  - masked mel L1 + spec-conv      (no zero-padding objective)
Conditioning is identical to V0 (ECAPA tokens + cross-attn; no frames_only).

Stage 'train' : train a content representation from scratch.
Stage 'id'    : fine-tune a trained checkpoint with true ECAPA identity loss.

Usage (via tools/content_factorial.py):
  python tools/content_factorial.py --stage train --content continuous --out checkpoints/cf_cont.ckpt
  python tools/content_factorial.py --stage train --content vq         --out checkpoints/cf_vq.ckpt
  python tools/content_factorial.py --stage id --content vq --init checkpoints/cf_vq.ckpt --out checkpoints/cf_vq_id.ckpt
"""

import os
import sys
import json
import time
import random
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import vc.data.compat
from vc.config import AudioConfig, ModelConfig, TrainingConfig
from vc.model import HubertVCModel
from vc.vocoder.inference import load_vocoder
from vc.eval.ecapa import load_ecapa, ecapa_emb
from vc.content.quantize import build_codebook, apply_content
from vc.data.splits import group_cache, load_item, pad_content, pad_mel, make_split
from vc.losses.masked import masked_l1, masked_spec_conv
from vc.eval.protocol import evaluate_arm
from vc.eval import manifest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

A_CFG = AudioConfig()
M_CFG = ModelConfig()
T_CFG = TrainingConfig()


def load_split(args):
    by_spk = group_cache(args.cache_dir)
    train_spks, val_spks = make_split(by_spk, args.split_seed, args.num_val_spk)
    train_items = [load_item(f) for s in train_spks for f in by_spk[s]]
    val_by_spk = {s: [load_item(f) for f in by_spk[s]] for s in val_spks}
    return by_spk, train_spks, val_spks, train_items, val_by_spk


def prepare_content(args, train_items, val_by_spk, codebook=None, ck_normed=None):
    device = getattr(args, "device", "cpu")
    if args.content == "vq":
        if codebook is None:
            print(f"[cf] building K-means codebook (mean_norm={args.mean_norm}, "
                  f"legacy={args.legacy_codebook}, device={device})...")
            codebook = build_codebook(train_items, args.codebook, args.km_iters,
                                      args.split_seed, args.km_frames, args.mean_norm,
                                      device=device)
        elif ck_normed is not None:
            fit_normed = bool(ck_normed)
            req_normed = bool(args.mean_norm and not args.legacy_codebook)
            if fit_normed != req_normed:
                raise RuntimeError(
                    f"[cf] codebook space mismatch: ckpt fit on "
                    f"{'mean-normed' if fit_normed else 'RAW'} frames but requested "
                    f"{'mean-normed' if req_normed else 'raw'} quantization. "
                    f"Pass --legacy_codebook only to reproduce legacy arms.")
        elif ck_normed is None and args.content == "vq":
            print("[cf][WARN] ckpt has no codebook_normed flag (pre-§7b legacy save); "
                  "space consistency not verified")
        apply_content(train_items, "vq", codebook, args.mean_norm, device=device)
        for s in val_by_spk:
            apply_content(val_by_spk[s], "vq", codebook, args.mean_norm, device=device)
    elif args.content == "continuous":
        if args.mean_norm:
            apply_content(train_items, "continuous", None, True)
            for s in val_by_spk:
                apply_content(val_by_spk[s], "continuous", None, True)
    return codebook


def save_ckpt(path, model, opt, sched, codebook, args, **extra):
    state = {"model_state": model.state_dict(), "codebook": codebook,
             "content": args.content, "mean_norm": args.mean_norm,
             "codebook_normed": bool(args.mean_norm and not args.legacy_codebook)}
    if opt is not None:
        state["opt_state"] = opt.state_dict()
        state["sched_state"] = sched.state_dict() if sched is not None else None
        state["rng"] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    state.update(extra)
    torch.save(state, path)


def restore_rng(ck):
    r = ck.get("rng")
    if r:
        random.setstate(r["python"])
        torch.set_rng_state(r["torch"])


def load_model_state(model, state):
    """strict load; on failure report the key diff instead of silently
    dropping weights (a partial init would confound everything downstream)."""
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        msd = model.state_dict()
        missing = [k for k in msd if k not in state]
        unexpected = [k for k in state if k not in msd]
        raise RuntimeError(
            f"checkpoint key mismatch: {len(missing)} missing "
            f"(e.g. {missing[:5]}), {len(unexpected)} unexpected "
            f"(e.g. {unexpected[:5]})")


def stage_train(args):
    by_spk, train_spks, val_spks, train_items, val_by_spk = load_split(args)
    print(f"[cf:train] content={args.content} mean_norm={args.mean_norm} "
          f"train={len(train_items)} val_spks={len(val_spks)}")
    codebook = prepare_content(args, train_items, val_by_spk)

    random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    model = HubertVCModel(A_CFG, M_CFG, T_CFG, load_encoders=False).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    rng = random.Random(args.train_seed)
    start_ep = 1
    if args.resume and os.path.exists(args.out):
        ck = torch.load(args.out, map_location="cpu")
        if "opt_state" not in ck:
            print("[cf:train][WARN] checkpoint has no optimizer state; "
                  "resume will restart training dynamics (use --save_every next time)")
        model.load_state_dict(ck["model_state"])
        if "opt_state" in ck:
            opt.load_state_dict(ck["opt_state"])
        if ck.get("sched_state"):
            sched.load_state_dict(ck["sched_state"])
        restore_rng(ck)
        start_ep = ck.get("epoch", 0) + 1
        print(f"[cf:train] resumed from {args.out} at epoch {start_ep}/{args.epochs}")
    n = len(train_items)
    t0 = time.time()
    for ep in tqdm(range(start_ep, args.epochs + 1), desc=f"cf:train:{args.content}", unit="ep"):
        model.train()
        order = list(range(n)); rng.shuffle(order)
        acc, nb = 0.0, 0
        pbar = tqdm(range(0, n, args.batch_size), desc=f"  ep{ep}", leave=False, unit="batch")
        for s in pbar:
            idxs = order[s:s + args.batch_size]
            if len(idxs) < 2:
                continue
            items = [train_items[i] for i in idxs]
            C, clens = pad_content(items)
            M, mlens = pad_mel([it["mel"] for it in items])
            S = torch.stack([it["speaker"] for it in items])
            pred, _, _ = model(precomputed_content_feats=C, precomputed_speaker_feats=S,
                               target_lengths=mlens, content_lengths=clens)
            loss = 45.0 * masked_l1(pred, M, mlens) + 2.0 * masked_spec_conv(pred, M, mlens)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc += loss.item(); nb += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            print(f"[cf:train:{args.content}] ep {ep:3d}/{args.epochs} "
                  f"loss={acc/max(1,nb):.4f} ({time.time()-t0:.0f}s)")
        if args.save_every > 0 and (ep % args.save_every == 0 or ep == args.epochs):
            save_ckpt(args.out, model, opt, sched, codebook, args, epoch=ep)
            print(f"[cf:train] checkpoint -> {args.out} (ep {ep})")
    save_ckpt(args.out, model, opt, sched, codebook, args, epoch=args.epochs)
    print(f"[cf:train] saved -> {args.out}")
    return model, val_by_spk, codebook


def stage_id(args):

    def _ecapa_of_mel(vocoder, enc, mel):
        """ECAPA of vocoded mel [1, 80, L] WITH gradient through the
        vocoder+ECAPA graph (no no_grad — this carries L_id's gradient).
        encode_batch returns [B, 1, 192]; normalized to [B, 192]."""
        wav = vocoder(mel.to(args.device))
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        e = enc.encode_batch(wav)
        return F.normalize(e.reshape(-1, e.shape[-1]), dim=-1)

    def get_src_emb(utt_id, mel):
        """Full-length ECAPA(vocoder(gt_mel)) of a SOURCE utterance, detached
        and cached (used only by the anti-source term)."""
        if utt_id not in src_emb:
            src_emb[utt_id] = _ecapa_of_mel(vocoder, enc, mel.unsqueeze(0).float()
                                            ).detach().squeeze(0)
        return src_emb[utt_id]

    def get_full_tgt_emb(utt_id, mel):
        """Full-length supervised target embedding (overrides the crop-160
        cache when --crop 0 so BOTH sides of L_id are full-length)."""
        if utt_id not in full_tgt_emb:
            full_tgt_emb[utt_id] = _ecapa_of_mel(vocoder, enc, mel.unsqueeze(0).float()
                                                 ).detach().squeeze(0)
        return full_tgt_emb[utt_id]

    by_spk, train_spks, val_spks, train_items, val_by_spk = load_split(args)
    ck = torch.load(args.init, map_location="cpu")
    codebook = ck.get("codebook", None)
    print(f"[cf:id] content={args.content} init={args.init}")
    codebook = prepare_content(args, train_items, val_by_spk, codebook,
                               ck.get("codebook_normed"))

    tk = None
    vocoder = None
    enc = None
    cached_pool = []
    src_emb = {}
    full_tgt_emb = {}
    if args.w_id > 0:
        tk = torch.load(args.target_file, map_location="cpu")
        targets = tk["targets"]
        cached_pool = [it for it in train_items if it["id"] in targets]

        vocoder = load_vocoder(None, device=args.device)
        enc = load_ecapa(args.device)
        for p in list(vocoder.parameters()) + list(enc.parameters()):
            p.requires_grad = False
    else:
        print("[cf:id] w_id=0 -> steps-only control (identity loss disabled)")

    model = HubertVCModel(A_CFG, M_CFG, T_CFG, load_encoders=False).to(args.device)
    load_model_state(model, ck["model_state"])
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.id_lr, weight_decay=1e-4)
    rng = random.Random(args.train_seed)
    start_step = 1
    if args.resume and os.path.exists(args.out):
        rck = torch.load(args.out, map_location="cpu")
        if "opt_state" not in rck:
            print("[cf:id][WARN] resume checkpoint has no optimizer state; "
                  "restart the id stage from scratch instead")
        model.load_state_dict(rck["model_state"])
        if "opt_state" in rck:
            opt.load_state_dict(rck["opt_state"])
        restore_rng(rck)
        start_step = rck.get("step", 0) + 1
        print(f"[cf:id] resumed from {args.out} at step {start_step}/{args.steps}")
    n = len(train_items)
    if args.w_id > 0 and args.crop == 0:
        print(f"[cf:id] crop=0: full-length BOTH-sides L_id "
              f"(+ neg margin term w_neg={args.w_neg}, shift m={args.neg_margin})")
    t0 = time.time()
    pbar = tqdm(range(start_step, args.steps + 1), desc=f"cf:id:{args.content}", unit="step")
    per_sample_id = (args.w_id > 0 and args.crop == 0)
    for step in pbar:
        idxs = [rng.randrange(n) for _ in range(args.batch_size)]
        items = [train_items[i] for i in idxs]
        C, clens = pad_content(items)
        M, mlens = pad_mel([it["mel"] for it in items])
        S = torch.stack([it["speaker"] for it in items])
        pred, _, _ = model(precomputed_content_feats=C, precomputed_speaker_feats=S,
                           target_lengths=mlens, content_lengths=clens)
        loss_self = 45.0 * masked_l1(pred, M, mlens) + 2.0 * masked_spec_conv(pred, M, mlens)
        loss_id_val = 0.0
        neg_val = 0.0
        if args.w_id > 0:
            spk_ids = [it["spk"] for it in items]
            cidx = []
            for si in spk_ids:
                for _ in range(8):
                    j = rng.randrange(len(cached_pool))
                    if cached_pool[j]["spk"] != si:
                        cidx.append(j); break
            citems = [cached_pool[j] for j in cidx]
            Sx = torch.stack([it["speaker"] for it in citems])
            # supervised length: crop-64 (frozen recipe), crop-128, or full-length
            # (crop=0, the Phase-2 mode). With full-length the model output is
            # supervised at its per-sample full length AND the target side is
            # ECAPA(vocoder(full mel)) — the crop-160 target cache would cap what
            # full-length supervision can teach.
            if per_sample_id:
                # crop=0: backward PER SAMPLE so only ONE full-length vocoder
                # grad graph is alive at a time (a batched backward keeps all
                # B graphs live and OOMs a 15 GB Colab GPU).
                opt.zero_grad()
                loss_self.backward()  # batched self-recon; frees its graph
                B_id = len(citems)
                id_sum = neg_sum = 0.0
                for bi, (it, c) in enumerate(zip(items, citems)):
                    pred_x, _, _ = model(precomputed_content_feats=C[bi:bi + 1],
                                         precomputed_speaker_feats=Sx[bi:bi + 1],
                                         target_lengths=mlens[bi:bi + 1],
                                         content_lengths=clens[bi:bi + 1])
                    L = min(c["mel"].shape[1], pred_x.shape[-1])
                    e_x = _ecapa_of_mel(vocoder, enc, pred_x[0][:, :L].unsqueeze(0))
                    e_j = get_full_tgt_emb(c["id"], c["mel"]).to(e_x.device).unsqueeze(0)
                    id_i = (1.0 - F.cosine_similarity(e_x, e_j, dim=-1)).mean()
                    total_i = args.w_id * id_i
                    if args.w_neg > 0:
                        # anti-source margin term: push the converted audio's
                        # embedding away from the SOURCE utterance's own
                        # embedding (relu hinge below cosine = neg_margin).
                        e_src = get_src_emb(it["id"], it["mel"]).to(e_x.device).unsqueeze(0)
                        neg_i = torch.relu(
                            F.cosine_similarity(e_x, e_src, dim=-1)
                            - args.neg_margin).mean()
                        total_i = total_i + args.w_neg * neg_i
                        neg_sum += neg_i.item() / B_id
                    # single backward per sample: the e_x graph is freed after
                    # this, so id and neg terms must share one backward pass
                    (total_i / B_id).backward()
                    id_sum += id_i.item() / B_id
                    del pred_x, e_x, id_i, total_i  # free the vocoder graph now
                loss_id_val = id_sum
                neg_val = neg_sum
            else:
                e_j = torch.stack([targets[it["id"]] for it in citems]).to(args.device)
                wav_x = vocoder(pred_x[:, :, :args.crop])
                e_xs = F.normalize(enc.encode_batch(wav_x).reshape(len(citems), -1),
                                   dim=-1)
                loss_id = (1.0 - F.cosine_similarity(e_xs, e_j, dim=-1)).mean()
                loss = loss_self + args.w_id * loss_id
                loss_id_val = loss_id.item()
                if args.w_neg > 0:
                    # anti-source margin term: push the converted audio's embedding
                    # away from the SOURCE utterance's own embedding (relu hinge
                    # below cosine = neg_margin).
                    cos_src_all = []
                    for c, e_x in zip(citems, e_xs):
                        e_src = get_src_emb(c["id"], c["mel"]).to(e_x.device)
                        cos_src_all.append(
                            F.cosine_similarity(e_x.unsqueeze(0), e_src.unsqueeze(0),
                                                dim=-1))
                    loss_neg = torch.relu(torch.cat(cos_src_all, dim=0)
                                          - args.neg_margin).mean()
                    loss = loss + args.w_neg * loss_neg
                    neg_val = loss_neg.item()
        if per_sample_id:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if step % 10 == 0:
            pbar.set_postfix(loss_self=f"{loss_self.item():.2f}",
                             loss_id=f"{loss_id_val:.3f}", neg=f"{neg_val:.3f}")
        if args.save_every > 0 and step % args.save_every == 0:
            save_ckpt(args.out, model, opt, None, codebook, args, step=step)
    save_ckpt(args.out, model, opt, None, codebook, args, step=args.steps)
    print(f"[cf:id] saved -> {args.out}")
    return model, val_by_spk, codebook


def stage_eval(args):
    by_spk, train_spks, val_spks, train_items, val_by_spk = load_split(args)
    ck = torch.load(args.init, map_location="cpu")
    codebook = ck.get("codebook", None)
    print(f"[cf:eval] content={args.content} init={args.init}")
    prepare_content(args, train_items, val_by_spk, codebook, ck.get("codebook_normed"))
    model = HubertVCModel(A_CFG, M_CFG, T_CFG, load_encoders=False).to(args.device)
    load_model_state(model, ck["model_state"])
    return model, val_by_spk, codebook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "id", "eval"], required=True)
    ap.add_argument("--content", choices=["continuous", "vq"], default="continuous")
    ap.add_argument("--cache_dir", default="cache/heavy_cache")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--num_val_spk", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--mean_norm", action="store_true",
                    help="per-utterance mean-norm of content (default off; on for vq arm)")
    ap.add_argument("--codebook", type=int, default=256)
    ap.add_argument("--km_iters", type=int, default=25)
    ap.add_argument("--km_frames", type=int, default=80000)
    ap.add_argument("--legacy_codebook", action="store_true",
                    help="fit K-means on RAW frames (pre-fix behavior; reproduces the "
                         "confounded legacy arms)")
    ap.add_argument("--resume", action="store_true",
                    help="resume train/id stage from the checkpoint at --out "
                         "(requires optimizer state; step checkpoints carry it)")
    ap.add_argument("--init", default="")
    ap.add_argument("--target_file", default="cache/vocoded_ecapa.pt")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--id_lr", type=float, default=5e-5)
    ap.add_argument("--w_id", type=float, default=20.0)
    ap.add_argument("--crop", type=int, default=64,
                    help="id-loss supervision length: 64 = frozen recipe crop, "
                         "0 = full-length BOTH sides (Phase-2)")
    ap.add_argument("--w_neg", type=float, default=0.0,
                    help="weight of the anti-source relu hinge term (0 = off)")
    ap.add_argument("--neg_margin", type=float, default=0.1,
                    help="hinge shift m for the anti-source term "
                         "(gradient flows while cos(pred,src) > m)")
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--out", default="checkpoints/cf.ckpt")
    ap.add_argument("--n_src", type=int, default=4)
    ap.add_argument("--n_content", type=int, default=2)
    ap.add_argument("--n_ref", type=int, default=2)
    ap.add_argument("--results", default="")
    ap.add_argument("--exp_id", default="adhoc",
                    help="experiment id for the provenance header of --results")
    ap.add_argument("--device", default="cpu",
                    help="cpu/cuda; training, vocoder and ECAPA all run here")
    args = ap.parse_args()

    if args.stage == "train":
        model, val_by_spk, codebook = stage_train(args)
    elif args.stage == "id":
        model, val_by_spk, codebook = stage_id(args)
    else:
        model, val_by_spk, codebook = stage_eval(args)

    if args.results:
        vocoder = load_vocoder(None, device=args.device)
        enc = load_ecapa(args.device)
        ref_cache = {}
        with torch.no_grad():
            for s in val_by_spk:
                wav = vocoder(val_by_spk[s][0]["mel"].unsqueeze(0).to(args.device)).squeeze(0)
                ref_cache[s] = ecapa_emb(enc, wav)
        print(f"[cf] eval content={args.content} stage={args.stage}")
        res = evaluate_arm(model, val_by_spk, vocoder, enc, ref_cache,
                           n_src=args.n_src, n_content=args.n_content, n_ref=args.n_ref)
        # Provenance header; retain `rows` (required for paired component CIs).
        header = manifest.provenance(
            exp_id=args.exp_id, arm=f"cf:{args.content}:{args.stage}",
            checkpoint=args.init or args.out, seed=args.train_seed,
            split_id=f"split_seed{args.split_seed}_val{args.num_val_spk}",
            config_hash_value=manifest.config_hash(
                args.content, args.mean_norm, args.codebook, args.n_content,
                args.n_ref, args.n_src, args.split_seed, args.train_seed))
        manifest.write_result(args.results, header, res)
        print(f"[cf] saved -> {args.results}")


if __name__ == "__main__":
    main()
