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
    # DIAGNOSTIC 2: mel_proj weight channel utilization
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🔬 DIAGNOSTIC 2: mel_proj weight channel utilization")
    print("="*70)
    with torch.no_grad():
        w = model.decoder.mel_proj.weight  # [80, 96, 1]
        w_norm = w.squeeze(-1).norm(p=2, dim=0)  # [96] per-input-channel L2
        low_chans = w_norm[:80]   # channels 0-79 (identity-init)
        high_chans = w_norm[80:]  # channels 80-95 (zero-init)
        print(f"  Input channels 0-79:  mean L2={low_chans.mean():.6f}, max={low_chans.max():.6f}, min={low_chans.min():.6f}")
        print(f"  Input channels 80-95: mean L2={high_chans.mean():.6f}, max={high_chans.max():.6f}, min={high_chans.min():.6f}")
        ratio = high_chans.mean() / (low_chans.mean() + 1e-8)
        print(f"  Ratio (80-95 / 0-79): {ratio:.4f}")
        # Per-output-channel breakdown: how much weight goes to 80-95 vs 0-79
        w_mat = w.squeeze(-1)  # [80, 96]
        w_to_lo = w_mat[:, :80].norm(p=2, dim=1)  # [80] per output chan
        w_to_hi = w_mat[:, 80:].norm(p=2, dim=1)  # [80]
        print(f"  Per-output-channel weight to ch 0-79:  mean={w_to_lo.mean():.6f}, max={w_to_lo.max():.6f}")
        print(f"  Per-output-channel weight to ch 80-95: mean={w_to_hi.mean():.6f}, max={w_to_hi.max():.6f}")
        # How many output channels have meaningful high-channel utilization?
        n_active = (w_to_hi > 0.01).sum().item()
        print(f"  Output channels with |w_hi| > 0.01: {n_active}/80")
        print(f"  [VERDICT] ", end="")
        if w_to_hi.max() < 0.005:
            print("Channels 80-95 are DEAD. Gradient starvation via identity shortcut confirmed.")
        elif ratio < 0.05:
            print("Channels 80-95 are SEVERELY STARVED. Near-zero recruitment.")
        elif ratio < 0.2:
            print("Channels 80-95 are WEAK. Partial recruitment, gradient competition with identity path.")
        else:
            print("Channels 80-95 are ACTIVE. Erosion has a different cause.")
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
            print("Speaker space is WELL-SEPARATED. Problem is training task (no cross-pair training).")

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
    model.decoder.mel_speaker_affine._verbose = False
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
        low_diff = chan_diff[:80]   # channels 0-79 (identity path)
        high_diff = chan_diff[80:]  # channels 80-95 (zero-init path)
        print(f"  Mean divergence ch 0-79:  {low_diff.mean():.6f}")
        print(f"  Mean divergence ch 80-95: {high_diff.mean():.6f}")
        print(f"  Ratio (80-95 / 0-79):     {high_diff.mean()/(low_diff.mean()+1e-8):.4f}")
        top16 = set(rank_order[:16].tolist())
        in_high = sum(1 for i in top16 if i >= 80)
        print(f"  Top-16 channels: {in_high}/16 are in range 80-95")
        print(f"  Top-16 channel indices (sorted): {sorted(top16)}")
        print(f"  Top-10 per-channel divergences:")
        for rank_idx, ch_idx in enumerate(rank_order[:10].tolist()):
            zone = " ← ZERO-INIT (80-95)" if ch_idx >= 80 else ""
            print(f"    #{rank_idx+1}: ch {ch_idx:3d}  div={chan_diff[ch_idx]:.6f}{zone}")
        print(f"  [VERDICT] ", end="")
        if in_high >= 8:
            print("Speaker info CONCENTRATED in ch 80-95. Identity-init mel_proj is structurally wrong.")
        elif in_high >= 3:
            print("Speaker info MIXED across channel groups. mel_proj partially usable but erodes high channels.")
        else:
            print("Speaker info in ch 0-79. mel_proj identity path should preserve it. Erosion has different cause.")

        # ═══════════════════════════════════════════════════════════════
        # DIAGNOSTIC 4: Channel 80-95 ablation at mel_proj input
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "-"*60)
        print("🔬 DIAGNOSTIC 4: Channel 80-95 ablation at mel_proj input")
        print("-"*60)
        # Normal path cent_cos
        mel_norm_AA = model.decoder.mel_proj(film_AA)
        mel_norm_AB = model.decoder.mel_proj(film_AB)
        aa_c_n = mel_norm_AA.flatten() - mel_norm_AA.flatten().mean()
        ab_c_n = mel_norm_AB.flatten() - mel_norm_AB.flatten().mean()
        cos_norm = torch.nn.functional.cosine_similarity(aa_c_n, ab_c_n, dim=0).item()
        # Ablated: zero out channels 80-95 before mel_proj
        film_AA_abl = film_AA.clone()
        film_AB_abl = film_AB.clone()
        film_AA_abl[:, 80:, :] = 0.0
        film_AB_abl[:, 80:, :] = 0.0
        mel_abl_AA = model.decoder.mel_proj(film_AA_abl)
        mel_abl_AB = model.decoder.mel_proj(film_AB_abl)
        aa_c_a = mel_abl_AA.flatten() - mel_abl_AA.flatten().mean()
        ab_c_a = mel_abl_AB.flatten() - mel_abl_AB.flatten().mean()
        cos_abl = torch.nn.functional.cosine_similarity(aa_c_a, ab_c_a, dim=0).item()
        print(f"  Normal    mel_proj_raw cent_cos: {cos_norm:.4f}")
        print(f"  Ablated   mel_proj_raw cent_cos: {cos_abl:.4f}")
        delta = cos_abl - cos_norm
        print(f"  Delta (ablated - normal):         {delta:+.4f}")
        print(f"  [VERDICT] ", end="")
        if abs(delta) < 0.02:
            print("Channels 80-95 contribute NEGLIGIBLE speaker info at mel_proj. Functionally dead.")
        elif delta > 0.05:
            print("Channels 80-95 carry speaker info — ablation IMPROVES separation (suggests noise from dead channels).")
        elif delta < -0.02:
            print("Channels 80-95 carry speaker info — ablation WORSENS separation. mel_proj has partially learned to use them.")
        else:
            print(f"Channels 80-95 have MARGINAL contribution (|Δ|={abs(delta):.3f}).")

        # ── Monolithic mel_proj: Conv1d(96→80) ──
        mel_proj_raw_AA = model.decoder.mel_proj(film_AA)
        mel_proj_raw_AB = model.decoder.mel_proj(film_AB)

        # ── mel_scaled: after unconditioned out_scale ──
        out_scale = torch.nn.functional.softplus(model.decoder.raw_out_scale) + 1.5  # [1, 80, 1]
        out_scale_AA = out_scale
        out_scale_AB = out_scale

        mel_band_mean_AA = mel_proj_raw_AA.mean(dim=-1, keepdim=True)
        mel_band_mean_AB = mel_proj_raw_AB.mean(dim=-1, keepdim=True)
        mel_centered_AA = mel_proj_raw_AA - mel_band_mean_AA
        mel_centered_AB = mel_proj_raw_AB - mel_band_mean_AB
        mel_scaled_AA = mel_centered_AA * out_scale_AA + mel_band_mean_AA + model.decoder.out_bias
        mel_scaled_AB = mel_centered_AB * out_scale_AB + mel_band_mean_AB + model.decoder.out_bias

        # ── mel_spk_affine: speaker-conditioned bias AFTER out_scale ──
        mel_affine_AA = model.decoder.mel_speaker_affine(mel_scaled_AA, spk_A_t)
        mel_affine_AB = model.decoder.mel_speaker_affine(mel_scaled_AB, spk_B_t)

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
            ("mel_proj_raw", mel_proj_raw_AA, mel_proj_raw_AB),    # monolithic Conv1d(96→80)
            ("mel_scaled", mel_scaled_AA, mel_scaled_AB),            # after out_scale (pre-spkr-bias)
            ("mel_spk_affine", mel_affine_AA, mel_affine_AB),      # after speaker bias
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
    # ═══════════════════════════════════════════════════════════════════

    # ── Re-enable verbose for forced-gamma diagnostic ────────────────
    model._verbose = True
    model.cross_attn._verbose = True
    model.decoder._verbose = True
    model.decoder.speaker_film._verbose = True
    model.decoder.block3_id_film._verbose = True
    model.decoder.adapter_speaker_film._verbose = True
    model.decoder.mel_speaker_affine._verbose = True
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
                 'block3_id_film', 'adapter_speaker_film', 'mel_speaker_affine']:
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
        elif name == 'mel_speaker_affine':
            saved_verbose[name] = model.decoder.mel_speaker_affine._verbose
            model.decoder.mel_speaker_affine._verbose = False

    aff = model.decoder.mel_speaker_affine

    # Freeze all model parameters, enable grad only on mel_spk_affine
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
    mlp2 = aff.mlp[2]   # Linear(96, 80)
    print(f"  L1 loss value: {l1_loss.item():.6f}")
    print(f"  --- Gradient norms (L2) ---")
    if mlp0.weight.grad is not None:
        print(f"  mlp[0].weight (96→96):       {mlp0.weight.grad.norm():.6f}")
    else:
        print(f"  mlp[0].weight (96→96):       NO GRADIENT")
    if mlp2.weight.grad is not None:
        print(f"  mlp[2].weight (96→80):       {mlp2.weight.grad.norm():.6f}")
    else:
        print(f"  mlp[2].weight (96→80):       NO GRADIENT")
    if mlp2.bias.grad is not None:
        print(f"  mlp[2].bias (80):            {mlp2.bias.grad.norm():.6f}")
    
    if hasattr(aff, 'raw_delta_scale') and aff.raw_delta_scale.grad is not None:
        ds_grad_val = aff.raw_delta_scale.grad.item()
        print(f"  raw_delta_scale grad:        {ds_grad_val:+.6f}")
        print(f"  [VERDICT] ", end="")
        if ds_grad_val < -0.0001:
            print("L1 PUSHES delta_scale DOWN. L1 actively suppresses speaker envelope shifts.")
        elif ds_grad_val > 0.0001:
            print("L1 pushes delta_scale UP. L1 actually wants more spectral shift.")
        else:
            print("L1 gradient on delta_scale is NEAR ZERO.")
    else:
        print(f"  raw_delta_scale grad:        NO GRADIENT (or missing)")
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
