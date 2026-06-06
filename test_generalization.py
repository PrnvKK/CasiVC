import os
import torch
import torchaudio
import argparse
from pathlib import Path

from config import AudioConfig, ModelConfig, TrainingConfig
from data.audio_utils import extract_mel_spectrogram, load_audio, split_utterance_for_training
from models.hubertvc_model import HubertVCModel
from inference import load_vocoder

def test_generalization(checkpoint_path: str, output_dir: str):
    print("="*60)
    print("🚀 RUNNING 2-UTTERANCE GENERALIZATION TEST")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    
    print("\n[1] Loading Vocoder...")
    vocoder = load_vocoder(None, device=str(device))
    vocoder.eval()
    
    # ── Hardcoded paths provided by the user ─────────────────────
    utt_path_A = "/content/LibriTTS/dev-clean/2428/83705/2428_83705_000000_000001.wav"
    utt_path_B = "/content/LibriTTS/dev-clean/1988/148538/1988_148538_000002_000000.wav"
    spk_A = "2428 (man)"
    spk_B = "1988 (woman)"
    
    print(f"\n[2] Loading Specific Utterances...")
    print(f"   Utterance A: {Path(utt_path_A).name}  (speaker {spk_A})")
    print(f"   Utterance B: {Path(utt_path_B).name}  (speaker {spk_B})")

    # Load full audio for both
    full_audio_A = load_audio(utt_path_A, sample_rate=audio_cfg.sample_rate).to(device)
    full_audio_B = load_audio(utt_path_B, sample_rate=audio_cfg.sample_rate).to(device)

    # Deterministic split (using the exact same logic as training)
    ref_A, content_A = split_utterance_for_training(
        full_audio_A, ref_length_range=(1.0, 2.0),
        sample_rate=audio_cfg.sample_rate, min_content_length=0.5, deterministic=True
    )
    ref_B, content_B = split_utterance_for_training(
        full_audio_B, ref_length_range=(1.0, 2.0),
        sample_rate=audio_cfg.sample_rate, min_content_length=0.5, deterministic=True
    )
    print(f"   A: ref={ref_A.shape[0]/audio_cfg.sample_rate:.2f}s, content={content_A.shape[0]/audio_cfg.sample_rate:.2f}s")
    print(f"   B: ref={ref_B.shape[0]/audio_cfg.sample_rate:.2f}s, content={content_B.shape[0]/audio_cfg.sample_rate:.2f}s")

    # ── Test 1: Raw ground truth waveforms ────────────────────────
    torchaudio.save(os.path.join(output_dir, "01_raw_content_A.wav"),
                    content_A.cpu().unsqueeze(0), audio_cfg.sample_rate)
    torchaudio.save(os.path.join(output_dir, "02_raw_content_B.wav"),
                    content_B.cpu().unsqueeze(0), audio_cfg.sample_rate)
    print("\n✅ Saved raw content A and B")

    # ── Test 2: Vocoder ceiling (GT mel → vocoded) ─────────────────
    print("\n[3] Testing Vocoder Ceiling...")
    gt_mel_A = extract_mel_spectrogram(content_A, sample_rate=audio_cfg.sample_rate)  # (80, T)
    gt_mel_B = extract_mel_spectrogram(content_B, sample_rate=audio_cfg.sample_rate)
    with torch.no_grad():
        vocoded_A = vocoder(gt_mel_A.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu()
        vocoded_B = vocoder(gt_mel_B.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu()
    torchaudio.save(os.path.join(output_dir, "03_vocoded_A.wav"), vocoded_A.unsqueeze(0), audio_cfg.sample_rate)
    torchaudio.save(os.path.join(output_dir, "04_vocoded_B.wav"), vocoded_B.unsqueeze(0), audio_cfg.sample_rate)
    print("✅ Saved vocoded GT mels for A and B")

    # ── Load model ─────────────────────────────────────────────────
    print(f"\n[4] Loading Model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}. Skipping model tests.")
        return
    model = HubertVCModel(audio_cfg, model_cfg, train_cfg).to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    print("✅ Model loaded.")

    # ═══════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 2: mel_proj weight channel utilization (split paths)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🔬 DIAGNOSTIC 2: mel_proj weight channel utilization (split paths)")
    print("="*70)
    with torch.no_grad():
        w_content = model.decoder.mel_proj_content.weight
        w_speaker = model.decoder.mel_proj_speaker.weight
        content_in_ch = w_content.shape[1]
        speaker_in_ch = w_speaker.shape[1]
        speaker_start = content_in_ch
        speaker_end = content_in_ch + speaker_in_ch - 1
        
        # Content path
        wc_norm = w_content.squeeze(-1).norm(p=2, dim=0)  # [80]
        print(f"  [CONTENT PATH] ch 0-{content_in_ch-1}:  mean L2={wc_norm.mean():.6f}")
        
        # Speaker path
        ws_norm = w_speaker.squeeze(-1).norm(p=2, dim=0)
        print(f"  [SPEAKER PATH] ch {speaker_start}-{speaker_end}: mean L2={ws_norm.mean():.6f}")
        
        # Per-output-channel breakdown for speaker path
        ws_to_hi = w_speaker.squeeze(-1).norm(p=2, dim=1)  # [80]
        n_active = (ws_to_hi > 0.01).sum().item()
        print(f"  Speaker path output channels with |w_s| > 0.01: {n_active}/80")
        print(f"  [VERDICT] ", end="")
        if ws_norm.mean() < 0.005:
            print(f"Speaker path channels {speaker_start}-{speaker_end} are DEAD. CE gradient may be insufficient.")
        else:
            print(f"Speaker path channels {speaker_start}-{speaker_end} are ACTIVE. Explicit partition working.")
    print("="*70)

    # ── Speaker Space Diagnostic ─────────────────────────────────────────
    print("\n[SPEAKER SPACE DIAGNOSTIC]")
    with torch.no_grad():
        spk_A = model.mel_encoder(ref_A.unsqueeze(0))  # [1, 1, 96]
        spk_B = model.mel_encoder(ref_B.unsqueeze(0))  # [1, 1, 96]

        vec_A = spk_A.view(1, -1)  # flatten to [1, 96]
        vec_B = spk_B.view(1, -1)

        cos_sim = torch.nn.functional.cosine_similarity(vec_A, vec_B, dim=-1)
        l2_dist = torch.norm(vec_A - vec_B, p=2)
        diff = (vec_A - vec_B).abs()
        top5 = diff.squeeze().topk(5)

        print(f"  Cosine similarity (Man vs Woman): {cos_sim.item():.4f}")
        print(f"  L2 distance       (Man vs Woman): {l2_dist.item():.4f}")
        print(f"  Top-5 most different dims values:  {[round(v, 4) for v in top5.values.tolist()]}")
        print(f"  Top-5 most different dims indices:  {top5.indices.tolist()}")
        print(f"  [VERDICT] ", end="")
        if cos_sim.item() > 0.90:
            print("Speaker space is COLLAPSED. Tanh is compressing Man/Woman into the same region.")
        elif cos_sim.item() > 0.50:
            print("Partial separation. Speaker info is weak but present.")
        else:
            print("Speaker space is WELL-SEPARATED. Bottleneck is downstream fusion/decoder training signal.")

        # ── Per-token diversity audit ─────────────────────────────────────
        print("\n[PER-TOKEN DIVERSITY AUDIT]")
        for speaker_label, spk_feats in [("A (Man)", spk_A), ("B (Woman)", spk_B)]:
            tokens = spk_feats[0]  # [8, 96]
            print(f"  Speaker {speaker_label}:")
            token_l2 = torch.norm(tokens, dim=-1)  # [8]
            token_means = tokens.mean(dim=-1)       # [8]
            token_stds  = tokens.std(dim=-1)        # [8]
            print(f"    Token L2 norms:  [{', '.join(f'{v:.3f}' for v in token_l2.tolist())}]")
            print(f"    Token means:     [{', '.join(f'{v:.3f}' for v in token_means.tolist())}]")
            print(f"    Token stds:      [{', '.join(f'{v:.3f}' for v in token_stds.tolist())}]")
            # Inter-token cosine similarity (lower = more diverse)
            tokens_norm = torch.nn.functional.normalize(tokens, dim=-1)  # [8, 96]
            cos_matrix = tokens_norm @ tokens_norm.T  # [8, 8]
            off_diag = cos_matrix[~torch.eye(8, dtype=torch.bool, device=tokens.device)]
            print(f"    Inter-token cosine: mean={off_diag.mean():.4f}, max={off_diag.max():.4f} (lower=more diverse)")
            # Effective rank (should now be >> 8/96)
            _, S, _ = torch.svd(tokens)
            ev = (S**2) / (S**2).sum()
            rank_95 = (ev.cumsum(0) < 0.95).sum().item() + 1
            print(f"    Effective rank (95% var): {rank_95}/8  |  Singular values: [{', '.join(f'{v:.2f}' for v in S.tolist())}]")
        # ─────────────────────────────────────────────────────────────────
    print()

    # ─────────────────────────────────────────────────────────────────────

    # ── Interpolation Probe ───────────────────────────────────────────
    print("[INTERPOLATION PROBE]")
    with torch.no_grad():
        spk_A = model.mel_encoder([ref_A.to(device)]) # [1, 1, 96]
        spk_B = model.mel_encoder([ref_B.to(device)]) # [1, 1, 96]
        blended_spk = 0.5 * spk_A + 0.5 * spk_B  # 50/50 blend of Man + Woman
        
        # Override speaker features in model forward pass
        pred_blend_A, _, _ = model(
            precomputed_speaker_feats=blended_spk,
            content_audio=[content_A.to(device)]
        )
        wave_blend = vocoder(pred_blend_A).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "09_blend_A_spk50.wav"), 
                        wave_blend.unsqueeze(0), audio_cfg.sample_rate)
        print(f"  Blended mel mean: {pred_blend_A.mean():.4f}")
        print("✅ Saved 09_blend_A_spk50.wav  (Man content + 50% Man / 50% Woman voice)")
    print()
    # ──────────────────────────────────────────────────────────────────

    # ── Suppress per-call debug prints after diagnostic probe ────────
    model._verbose = False
    model.cross_attn._verbose = False
    model.decoder._verbose = False
    model.decoder.speaker_film._verbose = False
    model.decoder.block3_id_film._verbose = False
    model.decoder.adapter_speaker_film._verbose = False
    model.decoder.mel_speaker_film._verbose = False
    for blk in model.decoder.blocks:
        blk._verbose = False
    model.mel_encoder._verbose = False
    # ─────────────────────────────────────────────────────────────────

    with torch.no_grad():
        # ── Test 3: Self-reconstruction A → A ─────────────────────
        print("\n[5] Self-reconstruction: A content + A voice...")
        pred_mel_AA, _, _ = model(ref_A.unsqueeze(0), [content_A])
        wave_AA = vocoder(pred_mel_AA).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "05_self_recon_A.wav"), wave_AA.unsqueeze(0), audio_cfg.sample_rate)
        print(f"   Pred mel: {pred_mel_AA.shape}, mean={pred_mel_AA.mean():.4f}")
        print("✅ Saved A→A")

        # ── Test 4: Self-reconstruction B → B ─────────────────────
        print("\n[6] Self-reconstruction: B content + B voice...")
        pred_mel_BB, _, _ = model(ref_B.unsqueeze(0), [content_B])
        wave_BB = vocoder(pred_mel_BB).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "06_self_recon_B.wav"), wave_BB.unsqueeze(0), audio_cfg.sample_rate)
        print(f"   Pred mel: {pred_mel_BB.shape}, mean={pred_mel_BB.mean():.4f}")
        print("✅ Saved B→B")

        # ── Test 5: Cross-conversion A content → B voice (Man speaking as Woman) ──────────
        print("\n[7] Cross-conversion: A content + B voice (Man speaking as Woman)...")
        pred_mel_AB, _, _ = model(ref_B.unsqueeze(0), [content_A])
        wave_AB = vocoder(pred_mel_AB).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "07_cross_AtoB.wav"), wave_AB.unsqueeze(0), audio_cfg.sample_rate)
        print(f"   Pred mel: {pred_mel_AB.shape}, mean={pred_mel_AB.mean():.4f}")
        print("✅ Saved A→B")

        # ── Test 6: Cross-conversion B content → A voice (Woman speaking as Man) ──────────
        print("\n[8] Cross-conversion: B content + A voice (Woman speaking as Man)...")
        pred_mel_BA, _, _ = model(ref_A.unsqueeze(0), [content_B])
        wave_BA = vocoder(pred_mel_BA).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "08_cross_BtoA.wav"), wave_BA.unsqueeze(0), audio_cfg.sample_rate)
        print(f"   Pred mel: {pred_mel_BA.shape}, mean={pred_mel_BA.mean():.4f}")
        print("\u2705 Saved B\u2192A")

        # ═══════════════════════════════════════════════════════════════
        # 🔬 SPK_SIM: Speaker similarity from ECAPA embeddings on audio
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "="*70)
        print("🔬 SPK_SIM: ECAPA speaker similarity on vocoded waveforms")
        print("="*70)
        # Extract ECAPA embeddings from vocoded waveforms
        def get_spk_emb(wave_1d):
            w = wave_1d.unsqueeze(0).to(device)
            return model.mel_encoder.extract_speaker_features(w, apply_projection=False)  # [1, 192]
        
        with torch.no_grad():
            emb_A = get_spk_emb(wave_AA).squeeze()     # self-recon A→A
            emb_B = get_spk_emb(wave_BB).squeeze()     # self-recon B→B  
            emb_AB = get_spk_emb(wave_AB).squeeze()    # cross A→B (target: B)
            emb_BA = get_spk_emb(wave_BA).squeeze()    # cross B→A (target: A)
            
            # Reference embeddings from original content audio (vocoded)
            emb_gt_A = model.mel_encoder.extract_speaker_features(
                content_A.unsqueeze(0).to(device), apply_projection=False
            ).squeeze()
            emb_gt_B = model.mel_encoder.extract_speaker_features(
                content_B.unsqueeze(0).to(device), apply_projection=False
            ).squeeze()
            emb_gt_voc_A = get_spk_emb(vocoded_A).squeeze()
            emb_gt_voc_B = get_spk_emb(vocoded_B).squeeze()
             
            def cos_sim(a, b):
                return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()

            ceiling_A = cos_sim(emb_gt_voc_A, emb_gt_A)
            ceiling_B = cos_sim(emb_gt_voc_B, emb_gt_B)
            avg_ceiling = 0.5 * (ceiling_A + ceiling_B)

            print(f"\n  {'VOCODER CEILING':<30} {'Cos Sim':>8}  {'Use':<8}")
            print(f"  {'-'*30} {'-'*8}  {'-'*8}")
            print(f"  {'GT-vocoded A vs raw GT_A':<30} {ceiling_A:>8.4f}  {'ceiling':>8}")
            print(f"  {'GT-vocoded B vs raw GT_B':<30} {ceiling_B:>8.4f}  {'ceiling':>8}")
            print(f"  {'Average vocoder ceiling':<30} {avg_ceiling:>8.4f}  {'anchor':>8}")
             
            print(f"\n  {'Comparison':<30} {'Cos Sim':>8}  {'Target':<8}")
            print(f"  {'─'*30} {'─'*8}  {'─'*8}")
            
            aa = cos_sim(emb_A, emb_gt_A)
            bb = cos_sim(emb_B, emb_gt_B)
            print(f"  {'A→A vs GT_A (self-recon)':<30} {aa:>8.4f}  {'>0.85':>8}")
            print(f"  {'B→B vs GT_B (self-recon)':<30} {bb:>8.4f}  {'>0.85':>8}")
            
            ab = cos_sim(emb_AB, emb_gt_B)
            ba = cos_sim(emb_BA, emb_gt_A)
            print(f"  {'A→B vs GT_B (conversion)':<30} {ab:>8.4f}  {'>0.60':>8}")
            print(f"  {'B→A vs GT_A (conversion)':<30} {ba:>8.4f}  {'>0.60':>8}")
            
            leak_ab = cos_sim(emb_AB, emb_gt_A)
            leak_ba = cos_sim(emb_BA, emb_gt_B)
            print(f"  {'A→B vs GT_A (source leak)':<30} {leak_ab:>8.4f}  {'<0.30':>8}")
            print(f"  {'B→A vs GT_B (source leak)':<30} {leak_ba:>8.4f}  {'<0.30':>8}")
            
            def pct_of_ceiling(score, ceiling):
                if abs(ceiling) < 1e-6:
                    return float("nan")
                return 100.0 * score / ceiling

            print(f"\n  {'Ceiling-normalized':<30} {'% Ceiling':>10}")
            print(f"  {'-'*30} {'-'*10}")
            print(f"  {'A->A / ceiling A':<30} {pct_of_ceiling(aa, ceiling_A):>9.1f}%")
            print(f"  {'B->B / ceiling B':<30} {pct_of_ceiling(bb, ceiling_B):>9.1f}%")
            print(f"  {'A->B / ceiling B':<30} {pct_of_ceiling(ab, ceiling_B):>9.1f}%")
            print(f"  {'B->A / ceiling A':<30} {pct_of_ceiling(ba, ceiling_A):>9.1f}%")

            def match_band_stats(base_mel, target_mel, band_slice):
                patched = base_mel.clone()
                target_band = target_mel[band_slice, :].unsqueeze(0)
                target_band = torch.nn.functional.interpolate(
                    target_band, size=base_mel.shape[-1],
                    mode="linear", align_corners=False,
                ).squeeze(0)
                base_band = patched[0, band_slice, :]
                base_mean = base_band.mean(dim=-1, keepdim=True)
                base_std = base_band.std(dim=-1, keepdim=True).clamp_min(1e-6)
                target_mean = target_band.mean(dim=-1, keepdim=True)
                target_std = target_band.std(dim=-1, keepdim=True).clamp_min(1e-6)
                patched[0, band_slice, :] = (base_band - base_mean) / base_std * target_std + target_mean
                return patched

            def speaker_sim_from_mel(mel, target_emb):
                wav = vocoder(mel.to(device)).squeeze(0).squeeze(0).cpu()
                emb = get_spk_emb(wav).squeeze()
                return cos_sim(emb, target_emb)

            print("\n  BAND-STAT SPK_SIM RESCUE")
            print("  Patches one mel band group to target-speaker mean/std, then vocodes.")
            print(f"  {'Band':<9} {'A->B sim':>9} {'dAB':>8} {'B->A sim':>9} {'dBA':>8} {'avg d':>8}")
            print(f"  {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")
            band_rescues = []
            for start in range(0, 80, 10):
                band_slice = slice(start, start + 10)
                patched_ab = match_band_stats(pred_mel_AB.detach(), gt_mel_B.to(device), band_slice)
                patched_ba = match_band_stats(pred_mel_BA.detach(), gt_mel_A.to(device), band_slice)
                sim_ab = speaker_sim_from_mel(patched_ab, emb_gt_B)
                sim_ba = speaker_sim_from_mel(patched_ba, emb_gt_A)
                delta_ab = sim_ab - ab
                delta_ba = sim_ba - ba
                avg_delta = 0.5 * (delta_ab + delta_ba)
                band_rescues.append((avg_delta, start))
                print(f"  {start:02d}-{start+9:<4d} {sim_ab:>9.4f} {delta_ab:>+8.4f} {sim_ba:>9.4f} {delta_ba:>+8.4f} {avg_delta:>+8.4f}")

            band_rescues.sort(reverse=True)
            print("  Top rescue bands:", ", ".join(
                f"{start:02d}-{start+9} (avg d={avg_delta:+.4f})"
                for avg_delta, start in band_rescues[:3]
            ))

            # If conversions are better than self-recons (unlikely but possible)
            if ab > aa: print(f"  ⚠️  A→B MORE similar to B than A→A is to A — over-conversion?")
            if ba > bb: print(f"  ⚠️  B→A MORE similar to A than B→B is to B — over-conversion?")
            
            print(f"\n  [VERDICT] ", end="")
            if ab > 0.60 and leak_ab < 0.30 and ba > 0.60 and leak_ba < 0.30:
                print("✅ Voice conversion working — high target sim, low source leak")
            elif ab < 0.40 or ba < 0.40:
                print("❌ Voice conversion FAILED — output not matching target speaker")
            elif leak_ab > 0.40 or leak_ba > 0.40:
                print("❌ Source speaker LEAKING — model outputs source identity")
            else:
                print("⚠️  Partial conversion — check individual scores above")
        print("="*70)

    # ═══════════════════════════════════════════════════════════════════
    # CONTENT FIDELITY AUDIT — measures intelligibility, not speaker ID
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🔬 CONTENT FIDELITY AUDIT")
    print("="*70)

    def per_band_corr(pred, gt):
        """Pearson r per mel band: [80] vector of correlations."""
        if pred.shape[-1] < 4:
            return torch.zeros(80, device=pred.device)
        p = pred.float(); g = gt.float()
        pm = p.mean(dim=-1, keepdim=True); gm = g.mean(dim=-1, keepdim=True)
        pc = p - pm; gc = g - gm
        cov = (pc * gc).sum(dim=-1)
        denom = (pc.norm(dim=-1) * gc.norm(dim=-1)) + 1e-8
        return cov / denom

    def energy_envelope(mel):
        """Frame energy: mean(abs(mel)) over 80 bands -> [T]."""
        return mel.float().abs().mean(dim=1)

    def band_group_l1(pred, gt):
        """L1 per band group: low(0-40), mid(40-60), high(60-80)."""
        groups = {"low (0-39)": slice(0,40), "mid (40-59)": slice(40,60), "high (60-79)": slice(60,80)}
        return {k: (pred[:, s] - gt[:, s]).abs().mean().item() for k, s in groups.items()}

    def spectral_flatness(mel):
        """Per-frame peak-to-median ratio (exp-log), then averaged.
           > 3.0 = harmonic/peaked, < 1.5 = noise-like/flat."""
        eps = 1e-8
        peak = mel.float().max(dim=1).values
        med = mel.float().median(dim=1).values
        return (peak / (med.abs() + eps)).mean().item()

    with torch.no_grad():
        ml = min(gt_mel_A.shape[-1], pred_mel_AA.shape[-1], pred_mel_AB.shape[-1])
        gt_A_mel = gt_mel_A[:, :ml].to(device)
        pred_AA_mel = pred_mel_AA[:, :, :ml].to(device)
        pred_AB_mel = pred_mel_AB[:, :, :ml].to(device)

        # ---------------------------------------------------------------
        # 1. Per-band correlation
        # ---------------------------------------------------------------
        corr_AA = per_band_corr(pred_AA_mel.squeeze(0), gt_A_mel)  # [80]
        corr_AB = per_band_corr(pred_AB_mel.squeeze(0), gt_A_mel)  # [80]

        print(f"\n  ── BAND CORRELATION TABLE ──")
        print(f"  {'BAND':<8}", end="")
        for s in range(0, 80, 10):
            print(f"{s:>3d}..{s+9:<3d}  ", end="")
        print(f"  {'MEAN':>7}")
        print(f"  {'A→A':<8}", end="")
        for s in range(0, 80, 10):
            v = corr_AA[s:s+10].mean().item()
            print(f" {v:+.3f}   ", end="")
        print(f"  {corr_AA.mean().item():+.4f}")
        print(f"  {'A→B':<8}", end="")
        for s in range(0, 80, 10):
            v = corr_AB[s:s+10].mean().item()
            print(f" {v:+.3f}   ", end="")
        print(f"  {corr_AB.mean().item():+.4f}")

        corr_drop = corr_AA.mean().item() - corr_AB.mean().item()
        print(f"  Δ(A→A - A→B) = {corr_drop:+.4f}  {'✅ content preserved' if corr_drop < 0.10 else '❌ speaker delta corrupting content'}")

        # ---------------------------------------------------------------
        # 2. Energy envelope correlation
        # ---------------------------------------------------------------
        env_AA = energy_envelope(pred_AA_mel.squeeze(0))
        env_gt = energy_envelope(gt_A_mel)
        energy_corr_AA = torch.nn.functional.cosine_similarity(
            (env_AA - env_AA.mean()).unsqueeze(0),
            (env_gt - env_gt.mean()).unsqueeze(0), dim=-1).item()
        env_AB = energy_envelope(pred_AB_mel.squeeze(0))
        energy_corr_AB = torch.nn.functional.cosine_similarity(
            (env_AB - env_AB.mean()).unsqueeze(0),
            (env_gt - env_gt.mean()).unsqueeze(0), dim=-1).item()

        print(f"\n  ── ENERGY ENVELOPE ──")
        print(f"  Corr(A→A, GT): {energy_corr_AA:+.4f}  {'✅' if energy_corr_AA > 0.4 else '❌ flat/misaligned'}")
        print(f"  Corr(A→B, GT): {energy_corr_AB:+.4f}  {'✅' if energy_corr_AB > 0.4 else '❌ flat/misaligned'}")

        # ---------------------------------------------------------------
        # 3. Delta (frame-diff) correlation
        # ---------------------------------------------------------------
        da_AA = pred_AA_mel.squeeze(0)[:, 1:] - pred_AA_mel.squeeze(0)[:, :-1]
        da_gt = gt_A_mel[:, 1:] - gt_A_mel[:, :-1]
        delta_corr_AA = per_band_corr(da_AA, da_gt).mean().item()
        da_AB = pred_AB_mel.squeeze(0)[:, 1:] - pred_AB_mel.squeeze(0)[:, :-1]
        delta_corr_AB = per_band_corr(da_AB, da_gt).mean().item()

        print(f"\n  ── ARTICULATION (Δ correlation) ──")
        print(f"  Δ-corr(A→A, GT): {delta_corr_AA:+.4f}  {'✅ consonants preserved' if delta_corr_AA > 0.2 else '❌ transitions lost'}")
        print(f"  Δ-corr(A→B, GT): {delta_corr_AB:+.4f}  {'✅ consonants preserved' if delta_corr_AB > 0.2 else '❌ transitions lost'}")

        # ---------------------------------------------------------------
        # 4. Band-group L1
        # ---------------------------------------------------------------
        l1_groups_AA = band_group_l1(pred_AA_mel, gt_A_mel.unsqueeze(0))
        print(f"\n  ── L1 PER BAND GROUP ──")
        for k in ["low (0-39)", "mid (40-59)", "high (60-79)"]:
            v = l1_groups_AA[k]
            marker = "✅" if v < 0.6 else ("⚠️ " if v < 1.0 else "❌")
            print(f"  A→A {k}: L1={v:.4f}  {marker}")

        # ---------------------------------------------------------------
        # 5. Cross content loss ratio
        # ---------------------------------------------------------------
        l1_AA_full = (pred_AA_mel - gt_A_mel.unsqueeze(0)).abs().mean().item()
        l1_AB_full = (pred_AB_mel - gt_A_mel.unsqueeze(0)).abs().mean().item()
        loss_ratio = l1_AB_full / (l1_AA_full + 1e-8)
        print(f"\n  ── CROSS-PAIR CONTENT INTERFERENCE ──")
        print(f"  L1(A→A, GT): {l1_AA_full:.4f}  |  L1(A→B, GT): {l1_AB_full:.4f}")
        print(f"  Ratio = {loss_ratio:.3f}x  {'✅ delta ~neutral' if loss_ratio < 1.3 else ('⚠️  mild interference' if loss_ratio < 2.0 else '❌ delta replacing content')}")

        # ---------------------------------------------------------------
        # 6. High-frequency energy ratio (noise detection)
        # ---------------------------------------------------------------
        def hf_ratio(mel):
            lo = mel[:, :20, :].std(dim=-1).mean().item()
            hi = mel[:, 60:, :].std(dim=-1).mean().item()
            return hi / (lo + 1e-8)
        hfr_AA = hf_ratio(pred_AA_mel)
        hfr_GT = hf_ratio(gt_A_mel.unsqueeze(0))
        print(f"\n  ── NOISE & ARTIFACT ──")
        print(f"  HF ratio (GT):     {hfr_GT:.3f}  (reference)")
        print(f"  HF ratio (A→A):    {hfr_AA:.3f}  {'✅' if hfr_AA < hfr_GT * 1.5 else '❌ broadband HF noise'}")

        # ---------------------------------------------------------------
        # 7. Spectral flatness
        # ---------------------------------------------------------------
        sf_AA = spectral_flatness(pred_AA_mel.squeeze(0))
        sf_GT = spectral_flatness(gt_A_mel)
        print(f"  Spec peak/med (GT):  {sf_GT:.2f}  (ref; <1.5 = noise-like, >3 = harmonic)")
        print(f"  Spec peak/med (A→A): {sf_AA:.2f}  {'✅' if sf_AA > 1.8 else '❌ noise-like output'}")

        # ---------------------------------------------------------------
        # 8. Silence fraction
        # ---------------------------------------------------------------
        threshold = gt_A_mel.mean().item() - 2.0 * gt_A_mel.std().item()
        sil_AA = (pred_AA_mel.mean(dim=1) < threshold).float().mean().item() * 100
        sil_GT = (gt_A_mel < threshold).float().mean().item() * 100
        print(f"  Silence%% (GT):     {sil_GT:.1f}%  (reference)")
        print(f"  Silence%% (A→A):    {sil_AA:.1f}%  {'✅' if abs(sil_AA - sil_GT) < 15 else '❌ collapsed or empty frames'}")

        # ---------------------------------------------------------------
        # 9. Out-of-range fraction
        # ---------------------------------------------------------------
        p5_gt = gt_A_mel.quantile(0.05).item()
        p95_gt = gt_A_mel.quantile(0.95).item()
        oob_AA = ((pred_AA_mel < p5_gt) | (pred_AA_mel > p95_gt)).float().mean().item() * 100
        print(f"  Out-of-range%% (A→A): {oob_AA:.1f}%  {'✅' if oob_AA < 25 else '❌ hallucinated mel values'}")

        # ---------------------------------------------------------------
        # Verdict
        # ---------------------------------------------------------------
        issues = []
        if corr_AA.mean().item() < 0.15: issues.append("band correlation < 0.15 (unintelligible)")
        if energy_corr_AA < 0.20: issues.append("energy envelope flat/misaligned")
        if delta_corr_AA < 0.10: issues.append("articulation transitions lost")
        if hfr_AA > hfr_GT * 2.0: issues.append("broadband HF noise")
        if sf_AA < 1.5: issues.append("noise-like spectrum (peak/med < 1.5)")
        if abs(sil_AA - sil_GT) > 20: issues.append("silence mismatch")
        if oob_AA > 40: issues.append("hallucinated values")

        print(f"\n  [CONTENT VERDICT] ", end="")
        if not issues:
            print("✅ Content intelligibility signals healthy")
        else:
            print(f"❌ {len(issues)} content quality issue(s):")
            for iss in issues:
                print(f"     • {iss}")
    print("="*70)

    # ═══════════════════════════════════════════════════════════════════
    # STAGE-DELTA AUDIT: A-voice vs B-voice divergence at each pipeline stage
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🔬 STAGE-DELTA AUDIT (A-voice vs B-voice divergence per stage)")
    print("="*70)
    with torch.no_grad():
        cont_A_t = model.hubert([content_A.to(device)])
        cont_A_t = model.hubert_proj(cont_A_t)
        cont_A_t = model.info_bottleneck(cont_A_t)
        spk_A_t = model.mel_encoder([ref_A.to(device)])
        spk_B_t = model.mel_encoder([ref_B.to(device)])

        fused_AA = model.cross_attn(cont_A_t, spk_A_t)
        fused_AB = model.cross_attn(cont_A_t, spk_B_t)

        tgt_len = gt_mel_A.shape[-1]
        resampled_AA = model.temporal_resampler(fused_AA, target_length=tgt_len)
        resampled_AB = model.temporal_resampler(fused_AB, target_length=tgt_len)

        adapter_AA = model.decoder.adapter(resampled_AA.transpose(1, 2))
        adapter_AB = model.decoder.adapter(resampled_AB.transpose(1, 2))

        # ── Adapter-entry speaker FiLM ──
        adapter_film_AA = model.decoder.adapter_speaker_film(adapter_AA, spk_A_t)
        adapter_film_AB = model.decoder.adapter_speaker_film(adapter_AB, spk_B_t)

        x_AA, x_AB = adapter_film_AA, adapter_film_AB
        for i, blk in enumerate(model.decoder.blocks):
            if i == 3:
                # ── Block3 decomposed: identity vs body vs sum ──
                # Replicate MobileNetBlock.forward logic to capture each branch
                identity_AA, identity_AB = x_AA, x_AB
                # upsampling (none for block3)
                if blk.upsample_first is not None:
                    x_AA = blk.upsample_first(x_AA)
                    x_AB = blk.upsample_first(x_AB)
                    identity_AA = blk.upsample_first(identity_AA)
                    identity_AB = blk.upsample_first(identity_AB)
                # body
                body_AA = blk.block(x_AA)
                body_AB = blk.block(x_AB)
                # identity projection
                if blk.residual_proj is not None:
                    id_proj_AA = blk.residual_proj(identity_AA)
                    id_proj_AB = blk.residual_proj(identity_AB)
                else:
                    id_proj_AA = identity_AA
                    id_proj_AB = identity_AB
                # store raw id_proj BEFORE FiLM (the erasure point)
                b3_id_AA, b3_id_AB = id_proj_AA.clone(), id_proj_AB.clone()
                # ── INJECT: block3_id_film at residual_proj output ──
                id_proj_AA = model.decoder.block3_id_film(id_proj_AA, spk_A_t)
                id_proj_AB = model.decoder.block3_id_film(id_proj_AB, spk_B_t)
                # store id_proj AFTER FiLM (speaker-recovered)
                b3_id_film_AA, b3_id_film_AB = id_proj_AA.clone(), id_proj_AB.clone()
                # final sum
                sum_AA = blk.residual_identity_scale * id_proj_AA + blk.residual_scale * body_AA
                sum_AB = blk.residual_identity_scale * id_proj_AB + blk.residual_scale * body_AB
                # store for stage audit
                b3_body_AA, b3_body_AB = body_AA, body_AB
                block3_AA, block3_AB = sum_AA, sum_AB
                x_AA, x_AB = sum_AA, sum_AB
            else:
                x_AA, x_AB = blk(x_AA), blk(x_AB)
            if i == 0:
                block0_AA, block0_AB = x_AA.clone(), x_AB.clone()
            if i == 2:
                block2_AA, block2_AB = x_AA.clone(), x_AB.clone()

        # ── Speaker FiLM: re-inject speaker identity before mel_proj ──
        film_AA = model.decoder.speaker_film(block3_AA, spk_A_t)
        film_AB = model.decoder.speaker_film(block3_AB, spk_B_t)

        # ═══════════════════════════════════════════════════════════════
        # DIAGNOSTIC 1: Per-channel speaker divergence at spk_film output
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "-"*60)
        print("🔬 DIAGNOSTIC 1: Per-channel speaker divergence at spk_film")
        print("-"*60)
        chan_diff = (film_AA - film_AB).abs().mean(dim=-1).squeeze(0)  # [96]
        _, rank_order = chan_diff.sort(descending=True)
        low_diff = chan_diff[:64]   # channels 0-63 (content path)
        high_diff = chan_diff[64:]  # channels 64-95 (speaker path)
        print(f"  Mean divergence ch 0-63:  {low_diff.mean():.6f}")
        print(f"  Mean divergence ch 64-95: {high_diff.mean():.6f}")
        print(f"  Ratio (64-95 / 0-63):     {high_diff.mean()/(low_diff.mean()+1e-8):.4f}")
        top16 = set(rank_order[:16].tolist())
        in_high = sum(1 for i in top16 if i >= 64)
        print(f"  Top-16 channels: {in_high}/16 are in range 64-95")
        print(f"  Top-16 channel indices (sorted): {sorted(top16)}")
        print(f"  Top-10 per-channel divergences:")
        for rank_idx, ch_idx in enumerate(rank_order[:10].tolist()):
            zone = " <- SPEAKER PATH (64-95)" if ch_idx >= 64 else ""
            print(f"    #{rank_idx+1}: ch {ch_idx:3d}  div={chan_diff[ch_idx]:.6f}{zone}")
        print(f"  [VERDICT] ", end="")
        if in_high >= 8:
            print("Speaker info CONCENTRATED in ch 64-95. Speaker path is active; judge usefulness by SPK_SIM and band rescue.")
        elif in_high >= 3:
            print("Speaker info MIXED across channel groups. Use downstream SPK_SIM/band rescue before changing training.")
        else:
            print("Speaker info mostly in ch 0-63. Speaker path may be under-used.")

        # ═══════════════════════════════════════════════════════════════
        # DIAGNOSTIC 4: Channel 64-95 speaker-path contribution
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "-"*60)
        print("🔬 DIAGNOSTIC 4: Channel 64-95 speaker-path contribution")
        print("-"*60)
        # Normal path cent_cos (content path only for apples-to-apples comparison)
        mel_norm_AA = model.decoder.mel_proj_content(film_AA[:, :64, :])
        mel_norm_AB = model.decoder.mel_proj_content(film_AB[:, :64, :])
        aa_c_n = mel_norm_AA.flatten() - mel_norm_AA.flatten().mean()
        ab_c_n = mel_norm_AB.flatten() - mel_norm_AB.flatten().mean()
        cos_norm = torch.nn.functional.cosine_similarity(aa_c_n, ab_c_n, dim=0).item()
        
        print(f"  Normal    mel_proj_content cent_cos: {cos_norm:.4f} (Isolated!)")

        # Also check speaker path contribution (reads dynamic temporal features now)
        spk_pooled_AA = film_AA[:, 64:, :]  # [B, 32, T]
        spk_pooled_AB = film_AB[:, 64:, :]
        mel_spk_base_d4_AA = model.decoder.mel_proj_speaker(spk_pooled_AA)
        mel_spk_base_d4_AB = model.decoder.mel_proj_speaker(spk_pooled_AB)
        # Apply content-gated modulation (match decoder forward)
        fe_d4_AA = mel_norm_AA.detach().abs().mean(dim=1, keepdim=True)
        fe_d4_AB = mel_norm_AB.detach().abs().mean(dim=1, keepdim=True)
        en_d4_AA = fe_d4_AA / (fe_d4_AA.mean(dim=-1, keepdim=True) + 1e-6)
        en_d4_AB = fe_d4_AB / (fe_d4_AB.mean(dim=-1, keepdim=True) + 1e-6)
        mel_spk_AA_d4 = mel_spk_base_d4_AA * en_d4_AA.clamp(0, 3.0)
        mel_spk_AB_d4 = mel_spk_base_d4_AB * en_d4_AB.clamp(0, 3.0)
        aa_c_s = mel_spk_AA_d4.flatten() - mel_spk_AA_d4.flatten().mean()
        ab_c_s = mel_spk_AB_d4.flatten() - mel_spk_AB_d4.flatten().mean()
        cos_spk = torch.nn.functional.cosine_similarity(aa_c_s, ab_c_s, dim=0).item()
        print(f"  Speaker path alone    cent_cos: {cos_spk:.4f}")
        spk_delta_std = mel_spk_AA_d4.std().item()
        print(f"  Speaker path delta std: {spk_delta_std:.4f} (>0.05 = active)")
        print(f"  [VERDICT] ", end="")
        if spk_delta_std < 0.05:
            print("Channels 64-95 contribute NEGLIGIBLE speaker info at mel_proj. Functionally dead.")
        else:
            print("Channels 64-95 carry ACTIVE dynamic speaker info. Explicit isolation working.")

        # ── Split mel_proj: content + speaker paths ──
        mel_proj_raw_AA = model.decoder.mel_proj_content(film_AA[:, :64, :])  # content path only
        mel_proj_raw_AB = model.decoder.mel_proj_content(film_AB[:, :64, :])
        # Speaker path now reads dynamic features
        spk_feat_AA = film_AA[:, 64:, :]
        spk_feat_AB = film_AB[:, 64:, :]
        mel_spk_base_AA = model.decoder.mel_proj_speaker(spk_feat_AA)
        mel_spk_base_AB = model.decoder.mel_proj_speaker(spk_feat_AB)

        # ── Content-gated modulation (match decoder forward exactly) ──
        frame_energy_AA = mel_proj_raw_AA.detach().abs().mean(dim=1, keepdim=True)
        frame_energy_AB = mel_proj_raw_AB.detach().abs().mean(dim=1, keepdim=True)
        energy_norm_AA = frame_energy_AA / (frame_energy_AA.mean(dim=-1, keepdim=True) + 1e-6)
        energy_norm_AB = frame_energy_AB / (frame_energy_AB.mean(dim=-1, keepdim=True) + 1e-6)
        gate_AA = energy_norm_AA.clamp(0, 3.0)
        gate_AB = energy_norm_AB.clamp(0, 3.0)
        mel_spk_AA = mel_spk_base_AA * gate_AA
        mel_spk_AB = mel_spk_base_AB * gate_AB

        # ── Magnitude cap: uses model's learnable delta_cap ──────
        delta_cap = torch.nn.functional.softplus(model.decoder.raw_delta_cap) + 0.05
        spk_std_AA = mel_spk_AA.flatten(1).std(dim=-1).view(-1, 1, 1).detach()
        cont_std_AA = mel_proj_raw_AA.detach().flatten(1).std(dim=-1).view(-1, 1, 1)
        spk_scale_AA = (delta_cap * cont_std_AA / (spk_std_AA + 1e-6)).clamp(max=1.0)
        mel_spk_AA = mel_spk_AA * spk_scale_AA

        spk_std_AB = mel_spk_AB.flatten(1).std(dim=-1).view(-1, 1, 1).detach()
        cont_std_AB = mel_proj_raw_AB.detach().flatten(1).std(dim=-1).view(-1, 1, 1)
        spk_scale_AB = (delta_cap * cont_std_AB / (spk_std_AB + 1e-6)).clamp(max=1.0)
        mel_spk_AB = mel_spk_AB * spk_scale_AB

        # ── mel_scaled: after unconditioned out_scale (content path only) ──
        out_scale = torch.nn.functional.softplus(model.decoder.raw_out_scale) + 1.5
        out_scale_AA = out_scale
        out_scale_AB = out_scale

        mel_band_mean_AA = mel_proj_raw_AA.mean(dim=-1, keepdim=True)
        mel_band_mean_AB = mel_proj_raw_AB.mean(dim=-1, keepdim=True)
        mel_centered_AA = mel_proj_raw_AA - mel_band_mean_AA
        mel_centered_AB = mel_proj_raw_AB - mel_band_mean_AB
        mel_scaled_AA = mel_centered_AA * out_scale_AA + mel_band_mean_AA + model.decoder.out_bias
        mel_scaled_AB = mel_centered_AB * out_scale_AB + mel_band_mean_AB + model.decoder.out_bias

        # ── spk_film_mel: after out_scale + speaker delta (content + speaker paths) ──
        spk_film_AA = mel_scaled_AA.detach() + mel_spk_AA
        spk_film_AB = mel_scaled_AB.detach() + mel_spk_AB

        # ── mel_spk_film: speaker-conditioned bias AFTER out_scale ──
        mel_affine_AA = model.decoder.mel_speaker_film(spk_film_AA.detach(), spk_A_t)
        mel_affine_AB = model.decoder.mel_speaker_film(spk_film_AB.detach(), spk_B_t)

        # ── mel_scaled_final: after clamp (final model output) ──
        mel_final_AA = torch.clamp(mel_affine_AA, min=-11.5, max=2.0)
        mel_final_AB = torch.clamp(mel_affine_AB, min=-11.5, max=2.0)

        stages = [
            ("cross_attn", fused_AA, fused_AB),
            ("resampler", resampled_AA, resampled_AB),
            ("adapter", adapter_AA, adapter_AB),
            ("adapter_film", adapter_film_AA, adapter_film_AB),
            ("block0", block0_AA, block0_AB),
            ("block2", block2_AA, block2_AB),
            ("b3_identity", b3_id_AA, b3_id_AB),        # raw residual_proj output (the erasure point)
            ("b3_id_film", b3_id_film_AA, b3_id_film_AB), # after Block3IdentityFiLM (speaker recovered)
            ("b3_body", b3_body_AA, b3_body_AB),
            ("block3_sum", block3_AA, block3_AB),
            ("spk_film", film_AA, film_AB),
            ("mel_content", mel_proj_raw_AA, mel_proj_raw_AB),    # content path
            ("mel_speaker", mel_spk_AA, mel_spk_AB),               # speaker path
            ("mel_scaled", mel_scaled_AA, mel_scaled_AB),          # post variance scaling
            ("mel_spk_film", mel_affine_AA, mel_affine_AB),      # final output before clamp
            ("mel_scaled_final", mel_final_AA, mel_final_AB),      # after clamp
        ]
        print(f"  {'Stage':<17} {'L1 diff':>9} {'cos sim':>9} {'cent cos':>9} {'σ(A)':>8} {'σ(B)':>8} {'σ ratio':>8}")
        for name, aa, ab in stages:
            l1 = (aa - ab).abs().mean().item()
            cos = torch.nn.functional.cosine_similarity(aa.flatten(), ab.flatten(), dim=0).item()
            # Centered cosine: subtract per-tensor mean before cosine, removing DC/bias contamination
            aa_c = aa.flatten() - aa.flatten().mean()
            ab_c = ab.flatten() - ab.flatten().mean()
            cos_cent = torch.nn.functional.cosine_similarity(aa_c, ab_c, dim=0).item()
            sa, sb = aa.std().item(), ab.std().item()
            ratio = sb / (sa + 1e-8)
            print(f"  {name:<17} {l1:>9.4f} {cos:>9.4f} {cos_cent:>9.4f} {sa:>8.4f} {sb:>8.4f} {ratio:>8.3f}")
    print("="*70)
    # ── δ_ratio: speaker delta magnitude vs content signal ───
    delta_ratio = mel_spk_AA.std().item() / (mel_proj_raw_AA.std().item() + 1e-8)
    print(f"\n  δ_ratio = σ_speaker / σ_content = {delta_ratio:.3f}")
    if delta_ratio > 0.5:
        print(f"  ❌ Speaker delta >50% of content — REPLACING content, not converting")
    elif delta_ratio > 0.2:
        print(f"  ⚠️  Speaker delta 20-50% of content — significant but may be OK")
    else:
        print(f"  ✅ Speaker delta ≤20% of content — subtle timbre, content preserved")
    print("="*70)
    # ═══════════════════════════════════════════════════════════════════

    # ── Re-enable verbose for forced-gamma diagnostic ────────────────
    model._verbose = True
    model.cross_attn._verbose = True
    model.decoder._verbose = True
    model.decoder.speaker_film._verbose = True
    model.decoder.block3_id_film._verbose = True
    model.decoder.adapter_speaker_film._verbose = True
    model.decoder.mel_speaker_film._verbose = True
    for blk in model.decoder.blocks:
        blk._verbose = True
    # ─────────────────────────────────────────────────────────────────

    # \u2500\u2500 FORCED-GAMMA DIAGNOSTIC \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # Bypasses the mapping network. Injects constant gamma to answer:
    # "Can decoder blocks compute given strong FiLM, or are they dead?"
    #
    # Read [Block] After block: std= lines in the decoder prints above.
    #   Block 3 body std > 0.25  \u2192  blocks ALIVE, mapping net is timid.
    #                                Fix: raise raw_film_scale to 0.5.
    #   Block 3 body std < 0.20  \u2192  block weights structurally dead.
    #                                Fix: reduce residual suppression 0.6\u21920.35.
    print("\n" + "="*70)
    print("\U0001f52c FORCED-GAMMA DIAGNOSTIC (mapping network bypassed)")
    print("="*70)
    print("  Runs A\u2192A with constant gamma injected. Beta=0. Pure residual-scale test.")
    print("  Watch [Block] After block: std= in decoder prints above each result.\n")
    with torch.no_grad():
        for forced_val in [0.5, 1.0]:
            print(f"  --- gamma={forced_val}  (residual scaled by {1+forced_val:.1f}x) ---")
            model.cross_attn.force_gamma = forced_val
            pred_forced, _, _ = model(ref_A.unsqueeze(0), [content_A])
            print(f"  Forced mel: mean={pred_forced.mean():.4f},  \u03c3={pred_forced.std():.4f}")
            print()
    model.cross_attn.force_gamma = None   # always reset
    print("  [force_gamma reset \u2192 None, normal mode restored]")
    print("="*70)
    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    # ═══════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 3: mel_spk_affine gradient decomposition (L1 loss)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🔬 DIAGNOSTIC 3: mel_spk_affine gradient decomposition (L1 loss)")
    print("="*70)

    # Suppress verbose prints during gradient pass
    saved_verbose = {}
    for name in ['_verbose', 'cross_attn', 'decoder', 'speaker_film',
                 'block3_id_film', 'adapter_speaker_film', 'mel_speaker_film']:
        if name == '_verbose':
            saved_verbose[name] = model._verbose
            model._verbose = False
        elif name == 'cross_attn':
            saved_verbose[name] = model.cross_attn._verbose
            model.cross_attn._verbose = False
        elif name == 'decoder':
            saved_verbose[name] = model.decoder._verbose
            model.decoder._verbose = False
            for blk in model.decoder.blocks:
                blk._verbose = False
        elif name == 'speaker_film':
            saved_verbose[name] = model.decoder.speaker_film._verbose
            model.decoder.speaker_film._verbose = False
        elif name == 'block3_id_film':
            saved_verbose[name] = model.decoder.block3_id_film._verbose
            model.decoder.block3_id_film._verbose = False
        elif name == 'adapter_speaker_film':
            saved_verbose[name] = model.decoder.adapter_speaker_film._verbose
            model.decoder.adapter_speaker_film._verbose = False
        elif name == 'mel_speaker_film':
            saved_verbose[name] = model.decoder.mel_speaker_film._verbose
            model.decoder.mel_speaker_film._verbose = False

    aff = model.decoder.mel_speaker_film

    # Freeze all model parameters, enable grad only on mel_spk_film
    for p in model.parameters():
        p.requires_grad = False
    for p in aff.parameters():
        p.requires_grad = True

    # Forward: A→A self-reconstruction with grad tracking
    pred_AA_grad, _, _ = model(ref_A.unsqueeze(0), [content_A])
    gt_mel_grad = extract_mel_spectrogram(content_A, sample_rate=audio_cfg.sample_rate).unsqueeze(0).to(device)

    # Trim to min length
    min_len = min(pred_AA_grad.shape[-1], gt_mel_grad.shape[-1])
    pred_AA_grad = pred_AA_grad[:, :, :min_len]
    gt_mel_grad = gt_mel_grad[:, :, :min_len]

    # L1 loss + backward
    l1_loss = torch.nn.functional.l1_loss(pred_AA_grad, gt_mel_grad)
    l1_loss.backward()

    # Log gradient norms
    mlp0 = aff.mlp[0]   # Linear(96, 96)
    mlp2 = aff.mlp[2]   # Linear(96, 160)
    print(f"  L1 loss value: {l1_loss.item():.6f}")
    print(f"  --- Gradient norms (L2) ---")
    if mlp0.weight.grad is not None:
        print(f"  mlp[0].weight (96→96):       {mlp0.weight.grad.norm():.6f}")
    else:
        print(f"  mlp[0].weight (96→96):       NO GRADIENT")
    if mlp2.weight.grad is not None:
        print(f"  mlp[2].weight (96→160):       {mlp2.weight.grad.norm():.6f}")
    else:
        print(f"  mlp[2].weight (96→160):       NO GRADIENT")
    if mlp2.bias.grad is not None:
        print(f"  mlp[2].bias (160):            {mlp2.bias.grad.norm():.6f}")
    
    if hasattr(aff, 'raw_film_scale') and aff.raw_film_scale.grad is not None:
        ds_grad_val = aff.raw_film_scale.grad.item()
        print(f"  raw_film_scale grad:        {ds_grad_val:+.6f}")
        print(f"  [VERDICT] ", end="")
        if ds_grad_val < -0.0001:
            print("L1 PUSHES film_scale DOWN. L1 actively suppresses speaker envelope shifts.")
        elif ds_grad_val > 0.0001:
            print("L1 pushes film_scale UP. L1 actually wants more spectral shift.")
        else:
            print("L1 gradient on film_scale is NEAR ZERO.")
    else:
        print(f"  raw_film_scale grad:        NO GRADIENT (or missing)")
        print(f"  [VERDICT] Cannot determine.")

    # Cleanup
    model.zero_grad()
    for p in aff.parameters():
        p.requires_grad = False
    # Restore model to eval-appropriate state (params stay frozen)
    del pred_AA_grad, gt_mel_grad, l1_loss
    print("="*70)

    print("\n" + "="*60)
    print("\U0001f4cb LISTENING GUIDE:")
    print("="*60)
    print("  01_raw_content_A.wav   \u2192 Ground truth Man (what was said)")
    print("  02_raw_content_B.wav   \u2192 Ground truth Woman (what was said)")
    print("  05_self_recon_A.wav    \u2192 Model: Man content + Man voice")
    print("  06_self_recon_B.wav    \u2192 Model: Woman content + Woman voice")
    print("  07_cross_AtoB.wav      \u2192 Model: Man's WORDS in Woman's VOICE \u2190 Key Test!")
    print("  08_cross_BtoA.wav      \u2192 Model: Woman's WORDS in Man's VOICE \u2190 Key Test!")
    print("="*60)


    # ── Mel variance summary: compare pred vs matching GT, not fixed 2.5 ─
    print("\n[MEL VARIANCE SUMMARY]  (pred vs matching GT)")
    gt_A_std  = gt_mel_A.std().item()
    gt_B_std  = gt_mel_B.std().item()
    aa_std    = pred_mel_AA.std().item()
    bb_std    = pred_mel_BB.std().item()
    ab_std    = pred_mel_AB.std().item()
    ba_std    = pred_mel_BA.std().item()
    print(f"  GT A σ={gt_A_std:.4f}  |  GT B σ={gt_B_std:.4f}")
    print(f"  A→A σ={aa_std:.4f}  |  B→B σ={bb_std:.4f}  |  A→B σ={ab_std:.4f}  |  B→A σ={ba_std:.4f}")
    deficit_aa = ((gt_A_std - aa_std) / gt_A_std) * 100
    deficit_bb = ((gt_B_std - bb_std) / gt_B_std) * 100
    print(f"  Variance deficit (A→A vs GT A={gt_A_std:.4f}): {deficit_aa:.1f}%  {'✅ OK' if deficit_aa < 15 else '❌ Compressed'}")
    print(f"  Variance deficit (B→B vs GT B={gt_B_std:.4f}): {deficit_bb:.1f}%  {'✅ OK' if deficit_bb < 15 else '❌ Compressed'}")
    # ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt", help="Path to your trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="generalization_test", help="Folder to save the output audio files")
    args = parser.parse_args()
    test_generalization(args.checkpoint, args.output_dir)
