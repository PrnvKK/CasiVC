import os
import torch
import torch.nn.functional as F
import torchaudio
import argparse
from pathlib import Path

from config import AudioConfig, ModelConfig, TrainingConfig
from data.audio_utils import extract_mel_spectrogram, load_audio, split_utterance_for_training
from models.hubertvc_model import HubertVCModel
from inference import load_vocoder


def _safe_cos(a, b):
    a_c = a.flatten() - a.flatten().mean()
    b_c = b.flatten() - b.flatten().mean()
    return float(F.cosine_similarity(a_c, b_c, dim=0))


def _compute_spk_sim(waveform, ref_waveform, ecapa_model, device):
    """ECAPA cosine similarity between two waveforms."""
    wav_a = waveform.to(device)
    wav_b = ref_waveform.to(device)
    if wav_a.dim() == 1:
        wav_a = wav_a.unsqueeze(0)
    if wav_b.dim() == 1:
        wav_b = wav_b.unsqueeze(0)
    with torch.no_grad():
        emb_a = ecapa_model.extract_speaker_features(wav_a, apply_projection=False)
        emb_b = ecapa_model.extract_speaker_features(wav_b, apply_projection=False)
    return float(F.cosine_similarity(emb_a.flatten(), emb_b.flatten(), dim=0))


def _compute_content_sim(waveform, ref_waveform, hubert_model, hubert_proj, bottleneck, device):
    """HuBERT content cosine between two waveforms."""
    wav_a = waveform.to(device)
    wav_b = ref_waveform.to(device)
    if wav_a.dim() == 1:
        wav_a = wav_a.unsqueeze(0)
    if wav_b.dim() == 1:
        wav_b = wav_b.unsqueeze(0)
    with torch.no_grad():
        feats_a = hubert_model([wav_a.squeeze(0)])
        feats_a = bottleneck(hubert_proj(feats_a))
        feats_b = hubert_model([wav_b.squeeze(0)])
        feats_b = bottleneck(hubert_proj(feats_b))
    T = min(feats_a.shape[1], feats_b.shape[1])
    return float(F.cosine_similarity(feats_a[0, :T].flatten(), feats_b[0, :T].flatten(), dim=0))


def _compute_tilt_mae(pred, gt, device):
    T = min(pred.shape[-1], gt.shape[-1])
    p_mean = pred.squeeze(0)[:, :T].mean(dim=-1)
    g_mean = gt.to(device)[:, :T].mean(dim=-1)
    err = (p_mean - g_mean).abs()
    return err.mean().item(), err[:40].mean().item(), err[40:].mean().item()


def test_generalization(checkpoint_path, output_dir,
                        full_audit=False, forced_gamma_audit=False,
                        grad_audit=False, verbose=False):
    print("=" * 62)
    print("  CASIVC GENERALIZATION DIAGNOSTIC v2")
    print("=" * 62)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    # ---- [0] SYSTEM ----------------------------------------------------------
    print("\n---- [0] SYSTEM ----")
    print(f"  Device: {device}")
    print("  [1] Loading Vocoder...")
    vocoder = load_vocoder(None, device=str(device))
    vocoder.eval()

    # ---- Load utterance paths ------------------------------------------------
    utt_path_A = "/content/LibriTTS/dev-clean/2428/83705/2428_83705_000000_000001.wav"
    utt_path_B = "/content/LibriTTS/dev-clean/1988/148538/1988_148538_000002_000000.wav"
    spk_A_label = "A (Man)"
    spk_B_label = "B (Woman)"

    print(f"\n  [2] Loading Utterances...")
    print(f"   A: {Path(utt_path_A).name}  ({spk_A_label})")
    print(f"   B: {Path(utt_path_B).name}  ({spk_B_label})")

    full_audio_A = load_audio(utt_path_A, sample_rate=audio_cfg.sample_rate).to(device)
    full_audio_B = load_audio(utt_path_B, sample_rate=audio_cfg.sample_rate).to(device)

    ref_A, content_A = split_utterance_for_training(
        full_audio_A, ref_length_range=(1.0, 2.0),
        sample_rate=audio_cfg.sample_rate, min_content_length=0.5, deterministic=True)
    ref_B, content_B = split_utterance_for_training(
        full_audio_B, ref_length_range=(1.0, 2.0),
        sample_rate=audio_cfg.sample_rate, min_content_length=0.5, deterministic=True)

    print(f"   A: ref={ref_A.shape[0]/audio_cfg.sample_rate:.2f}s, content={content_A.shape[0]/audio_cfg.sample_rate:.2f}s")
    print(f"   B: ref={ref_B.shape[0]/audio_cfg.sample_rate:.2f}s, content={content_B.shape[0]/audio_cfg.sample_rate:.2f}s")

    # Save raw content
    torchaudio.save(os.path.join(output_dir, "01_raw_content_A.wav"),
                    content_A.cpu().unsqueeze(0), audio_cfg.sample_rate)
    torchaudio.save(os.path.join(output_dir, "02_raw_content_B.wav"),
                    content_B.cpu().unsqueeze(0), audio_cfg.sample_rate)

    # Vocoder ceiling
    print("\n  [3] Vocoder Ceiling...")
    gt_mel_A = extract_mel_spectrogram(content_A, sample_rate=audio_cfg.sample_rate)
    gt_mel_B = extract_mel_spectrogram(content_B, sample_rate=audio_cfg.sample_rate)
    with torch.no_grad():
        vocoded_A = vocoder(gt_mel_A.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu()
        vocoded_B = vocoder(gt_mel_B.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu()
    torchaudio.save(os.path.join(output_dir, "03_vocoded_A.wav"), vocoded_A.unsqueeze(0), audio_cfg.sample_rate)
    torchaudio.save(os.path.join(output_dir, "04_vocoded_B.wav"), vocoded_B.unsqueeze(0), audio_cfg.sample_rate)
    print("  Saved vocoded GT mels.")

    # ---- Load model ----------------------------------------------------------
    print(f"\n  [4] Loading Model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"  CHECKPOINT NOT FOUND: {checkpoint_path}. Aborting.")
        return
    model = HubertVCModel(audio_cfg, model_cfg, train_cfg).to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.decoder.spk_out_scale.disabled = True

    # Quiet model by default; --verbose re-enables below
    model._verbose = False

    # Set verbose flags
    if verbose:
        model._verbose = True
        model.cross_attn._verbose = True
        model.decoder._verbose = True
        model.decoder.speaker_film._verbose = True
        model.decoder.block3_id_film._verbose = True
        model.decoder.adapter_speaker_film._verbose = True
        model.mel_encoder._verbose = True
        model.decoder.spk_out_scale._verbose = True
        model.decoder.mel_speaker_affine._verbose = True
        for blk in model.decoder.blocks:
            blk._verbose = True

    not_found_msg = ""
    print(f"  Model loaded.{not_found_msg}")

    # ---- [1] ECAPA SPEAKER ENCODER -------------------------------------------
    print("\n" + "-" * 62)
    print("---- [1] ECAPA SPEAKER ENCODER ----")
    print("-" * 62)

    with torch.no_grad():
        spk_A_raw = model.mel_encoder.extract_speaker_features(ref_A.unsqueeze(0).to(device))
        spk_B_raw = model.mel_encoder.extract_speaker_features(ref_B.unsqueeze(0).to(device))

        vec_A = spk_A_raw[0].flatten()
        vec_B = spk_B_raw[0].flatten()
        cos_ab = float(F.cosine_similarity(vec_A, vec_B, dim=0))
        l2_ab = float((vec_A - vec_B).norm(p=2))

        inter = F.normalize(spk_A_raw[0], dim=-1)
        cos_mat = inter @ inter.T
        off_diag = cos_mat[~torch.eye(spk_A_raw.shape[1], dtype=torch.bool, device=device)]
        inter_tok_A = float(off_diag.mean())

        inter = F.normalize(spk_B_raw[0], dim=-1)
        cos_mat = inter @ inter.T
        off_diag = cos_mat[~torch.eye(spk_B_raw.shape[1], dtype=torch.bool, device=device)]
        inter_tok_B = float(off_diag.mean())

        print(f"  A<->B:  cos={cos_ab:.4f}  L2={l2_ab:.2f}  "
              f"inter-tok_cos: A={inter_tok_A:.3f} B={inter_tok_B:.3f}")

    # ---- Audio outputs -------------------------------------------------------
    print("\n  Generating conversion outputs...")
    with torch.no_grad():
        pred_mel_AA, _, _ = model(ref_A.unsqueeze(0), [content_A])
        wave_AA = vocoder(pred_mel_AA).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "05_self_recon_A.wav"), wave_AA.unsqueeze(0), audio_cfg.sample_rate)

        pred_mel_BB, _, _ = model(ref_B.unsqueeze(0), [content_B])
        wave_BB = vocoder(pred_mel_BB).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "06_self_recon_B.wav"), wave_BB.unsqueeze(0), audio_cfg.sample_rate)

        pred_mel_AB, _, _ = model(ref_B.unsqueeze(0), [content_A])
        wave_AB = vocoder(pred_mel_AB).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "07_cross_AtoB.wav"), wave_AB.unsqueeze(0), audio_cfg.sample_rate)

        pred_mel_BA, _, _ = model(ref_A.unsqueeze(0), [content_B])
        wave_BA = vocoder(pred_mel_BA).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "08_cross_BtoA.wav"), wave_BA.unsqueeze(0), audio_cfg.sample_rate)

        blended_spk = 0.5 * spk_A_raw + 0.5 * spk_B_raw
        pred_blend, _, _ = model(precomputed_speaker_feats=blended_spk, content_audio=[content_A.to(device)])
        wave_blend = vocoder(pred_blend).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "09_blend_A_spk50.wav"), wave_blend.unsqueeze(0), audio_cfg.sample_rate)

    print("  All outputs saved.")

    # ---- [2] CONVERSION QUALITY ----------------------------------------------
    print("\n" + "-" * 62)
    print("---- [2] CONVERSION QUALITY ----")
    print("-" * 62)

    gt_A_std = gt_mel_A.std().item()
    gt_B_std = gt_mel_B.std().item()

    with torch.no_grad():
        spk_A_ecapa = model.mel_encoder.extract_speaker_features(ref_A.unsqueeze(0).to(device), apply_projection=False)
        spk_B_ecapa = model.mel_encoder.extract_speaker_features(ref_B.unsqueeze(0).to(device), apply_projection=False)

        spk_sim = {}
        for label, wave, target_ref, source_ref in [
            ("A->A", wave_AA, ref_A, ref_A),
            ("B->B", wave_BB, ref_B, ref_B),
            ("A->B", wave_AB, ref_B, ref_A),
            ("B->A", wave_BA, ref_A, ref_B),
        ]:
            wav = wave.to(device)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            emb_out = model.mel_encoder.extract_speaker_features(wav, apply_projection=False)
            spk_sim[(label, 'tgt')] = float(F.cosine_similarity(
                emb_out.flatten(), spk_A_ecapa.flatten() if target_ref is ref_A else spk_B_ecapa.flatten(), dim=0))
            spk_sim[(label, 'src')] = float(F.cosine_similarity(
                emb_out.flatten(), spk_A_ecapa.flatten() if source_ref is ref_A else spk_B_ecapa.flatten(), dim=0))

        content_cos = {}
        for label, wave, source_content in [
            ("A->A", wave_AA, content_A),
            ("B->B", wave_BB, content_B),
            ("A->B", wave_AB, content_A),
            ("B->A", wave_BA, content_B),
        ]:
            content_cos[label] = _compute_content_sim(
                wave, source_content, model.hubert, model.hubert_proj,
                model.info_bottleneck, device)

    mel_stats = {
        "A->A": (pred_mel_AA.mean().item(), pred_mel_AA.std().item()),
        "B->B": (pred_mel_BB.mean().item(), pred_mel_BB.std().item()),
        "A->B": (pred_mel_AB.mean().item(), pred_mel_AB.std().item()),
        "B->A": (pred_mel_BA.mean().item(), pred_mel_BA.std().item()),
    }

    print(f"  {'':<8} {'mel_mean':>8} {'mel_std':>8} {'SPK_SIM(tgt)':>13} {'SPK_SIM(src)':>13} {'Content_cos':>11}")
    for label in ["A->A", "B->B", "A->B", "B->A"]:
        mu, sd = mel_stats[label]
        tgt = spk_sim.get((label, 'tgt'), float('nan'))
        src = spk_sim.get((label, 'src'), float('nan'))
        cc = content_cos.get(label, float('nan'))
        print(f"  {label:<8} {mu:>8.3f} {sd:>8.3f} {tgt:>13.4f} {src:>13.4f} {cc:>11.4f}")
    print(f"  {'GT A':<8} {'-':>8} {gt_A_std:>8.3f}")
    print(f"  {'GT B':<8} {'-':>8} {gt_B_std:>8.3f}")

    # ---- [3] MEL QUALITY -----------------------------------------------------
    print("\n" + "-" * 62)
    print("---- [3] MEL QUALITY & OUTPUT HEAD DIAGNOSTICS ----")
    print("-" * 62)

    deficit_aa = ((gt_A_std - mel_stats["A->A"][1]) / gt_A_std) * 100
    deficit_bb = ((gt_B_std - mel_stats["B->B"][1]) / gt_B_std) * 100
    print(f"  Variance deficit: A->A={deficit_aa:.1f}%  "
          f"B->B={deficit_bb:.1f}%")

    # Output head diagnostics: mel_proj -> out_scale cent_cos erosion
    with torch.no_grad():
        cont_A_t = model.hubert([content_A.to(device)])
        cont_A_t = model.hubert_proj(cont_A_t)
        cont_A_t = model.info_bottleneck(cont_A_t)
        cont_B_t = model.hubert([content_B.to(device)])
        cont_B_t = model.hubert_proj(cont_B_t)
        cont_B_t = model.info_bottleneck(cont_B_t)

        spk_A_t = model.mel_encoder([ref_A.to(device)])
        spk_B_t = model.mel_encoder([ref_B.to(device)])

        # Full pipeline to get intermediates for A and B voice
        fused_AA = model.cross_attn(cont_A_t, spk_A_t)
        fused_AB = model.cross_attn(cont_A_t, spk_B_t)
        fused_BB = model.cross_attn(cont_B_t, spk_B_t)
        fused_BA = model.cross_attn(cont_B_t, spk_A_t)

        tgt_len_A = gt_mel_A.shape[-1]
        tgt_len_B = gt_mel_B.shape[-1]

        # Run decoder for A-voice vs B-voice paths
        def _decoder_run(fused, speaker, tgt_len):
            r = model.temporal_resampler(fused, target_length=tgt_len)
            a = model.decoder.adapter(r.transpose(1, 2))
            a = model.decoder.adapter_speaker_film(a, speaker)
            x = a
            for i, blk in enumerate(model.decoder.blocks):
                if i == 3:
                    identity = x
                    if blk.upsample_first is not None:
                        x = blk.upsample_first(x)
                        identity = blk.upsample_first(identity)
                    body = blk.block(x)
                    id_proj = blk.residual_proj(identity) if blk.residual_proj is not None else identity
                    id_proj = model.decoder.block3_id_film(id_proj, speaker)
                    x = blk.residual_identity_scale * id_proj + blk.residual_scale * body
                else:
                    x = blk(x)
            x = model.decoder.speaker_film(x, speaker)
            mel_raw = model.decoder.mel_proj(x)
            mel_raw = model.decoder.mel_speaker_affine(mel_raw, speaker)
            band_mean = mel_raw.mean(dim=-1, keepdim=True)
            mel_centered = mel_raw - band_mean
            base_scale = F.softplus(model.decoder.raw_out_scale) + 1.5
            out_scale = model.decoder.spk_out_scale(base_scale, speaker)
            mel_scaled = mel_centered * out_scale + band_mean + model.decoder.out_bias
            mel_final = torch.clamp(mel_scaled, min=-11.5, max=2.0)
            return mel_raw, mel_scaled, mel_final

        mel_raw_AA, mel_scaled_AA, mel_final_AA = _decoder_run(fused_AA, spk_A_t, tgt_len_A)
        mel_raw_AB, mel_scaled_AB, mel_final_AB = _decoder_run(fused_AB, spk_B_t, tgt_len_A)
        mel_raw_BB, mel_scaled_BB, mel_final_BB = _decoder_run(fused_BB, spk_B_t, tgt_len_B)
        mel_raw_BA, mel_scaled_BA, mel_final_BA = _decoder_run(fused_BA, spk_A_t, tgt_len_B)

        # Output head diagnostics (A vs B voice conditioning)
        raw_aa_cos = _safe_cos(mel_raw_AA, mel_raw_AB)
        raw_bb_cos = _safe_cos(mel_raw_BB, mel_raw_BA)
        raw_cent_cos = (raw_aa_cos + raw_bb_cos) / 2
        scl_aa_cos = _safe_cos(mel_scaled_AA, mel_scaled_AB)
        scl_bb_cos = _safe_cos(mel_scaled_BB, mel_scaled_BA)
        scl_cent_cos = (scl_aa_cos + scl_bb_cos) / 2

        std_raw_A = mel_raw_AA.std(dim=-1).squeeze(0)
        std_raw_B = mel_raw_BB.std(dim=-1).squeeze(0)
        std_scl_A = mel_scaled_AA.std(dim=-1).squeeze(0)
        std_scl_B = mel_scaled_BB.std(dim=-1).squeeze(0)
        ratio_raw = float((std_raw_A / (std_raw_B + 1e-8)).mean())
        ratio_scl = float((std_scl_A / (std_scl_B + 1e-8)).mean())

        tilt_raw_AA = _compute_tilt_mae(mel_raw_AA, gt_mel_A, device)
        tilt_scl_AA = _compute_tilt_mae(mel_scaled_AA, gt_mel_A, device)
        tilt_raw_BB = _compute_tilt_mae(mel_raw_BB, gt_mel_B, device)
        tilt_scl_BB = _compute_tilt_mae(mel_scaled_BB, gt_mel_B, device)

        print(f"\n  [OUTPUT HEAD DIAGNOSTICS]")
        print(f"  {'':<28} {'before out_scale':>18} {'after out_scale':>18}")
        print(f"  {'A/B std ratio':<28} {ratio_raw:>18.4f} {ratio_scl:>18.4f}")
        print(f"  {'Tilt MAE A->A (All/Lo/Hi)':<28} "
              f"{tilt_raw_AA[0]:>6.3f}/{tilt_raw_AA[1]:>5.3f}/{tilt_raw_AA[2]:>5.3f}"
              f"  {tilt_scl_AA[0]:>6.3f}/{tilt_scl_AA[1]:>5.3f}/{tilt_scl_AA[2]:>5.3f}")
        print(f"  {'Tilt MAE B->B (All/Lo/Hi)':<28} "
              f"{tilt_raw_BB[0]:>6.3f}/{tilt_raw_BB[1]:>5.3f}/{tilt_raw_BB[2]:>5.3f}"
              f"  {tilt_scl_BB[0]:>6.3f}/{tilt_scl_BB[1]:>5.3f}/{tilt_scl_BB[2]:>5.3f}")
        print(f"  {'cent_cos A<->B':<28} {raw_cent_cos:>18.4f} {scl_cent_cos:>18.4f}")

    # ---- [4] ARCHITECTURE BOTTLENECK -----------------------------------------
    print("\n" + "=" * 62)
    print("---- [4] ARCHITECTURE BOTTLENECK: STAGE-DELTA ----")
    print("=" * 62)
    print("  Lower cent_cos = better A/B separation.")

    with torch.no_grad():
        # Build full stages for A-voice vs B-voice
        resampled_AA = model.temporal_resampler(fused_AA, target_length=tgt_len_A)
        resampled_AB = model.temporal_resampler(fused_AB, target_length=tgt_len_A)

        adapter_AA = model.decoder.adapter(resampled_AA.transpose(1, 2))
        adapter_AB = model.decoder.adapter(resampled_AB.transpose(1, 2))
        adapter_film_AA = model.decoder.adapter_speaker_film(adapter_AA, spk_A_t)
        adapter_film_AB = model.decoder.adapter_speaker_film(adapter_AB, spk_B_t)

        x_AA, x_AB = adapter_film_AA, adapter_film_AB
        block_outputs_AA = []
        block_outputs_AB = []
        b3_body_AA = b3_body_AB = b3_id_AA = b3_id_AB = None
        b3_id_film_AA = b3_id_film_AB = None

        for i, blk in enumerate(model.decoder.blocks):
            if i == 3:
                identity_AA, identity_AB = x_AA, x_AB
                if blk.upsample_first is not None:
                    x_AA = blk.upsample_first(x_AA)
                    x_AB = blk.upsample_first(x_AB)
                    identity_AA = blk.upsample_first(identity_AA)
                    identity_AB = blk.upsample_first(identity_AB)
                body_AA = blk.block(x_AA)
                body_AB = blk.block(x_AB)
                b3_body_AA, b3_body_AB = body_AA.clone(), body_AB.clone()
                id_proj_AA = blk.residual_proj(identity_AA) if blk.residual_proj is not None else identity_AA
                id_proj_AB = blk.residual_proj(identity_AB) if blk.residual_proj is not None else identity_AB
                b3_id_AA, b3_id_AB = id_proj_AA.clone(), id_proj_AB.clone()
                id_proj_AA = model.decoder.block3_id_film(id_proj_AA, spk_A_t)
                id_proj_AB = model.decoder.block3_id_film(id_proj_AB, spk_B_t)
                b3_id_film_AA, b3_id_film_AB = id_proj_AA.clone(), id_proj_AB.clone()
                sum_AA = blk.residual_identity_scale * id_proj_AA + blk.residual_scale * body_AA
                sum_AB = blk.residual_identity_scale * id_proj_AB + blk.residual_scale * body_AB
                x_AA, x_AB = sum_AA, sum_AB
            else:
                x_AA, x_AB = blk(x_AA), blk(x_AB)
            block_outputs_AA.append(x_AA)
            block_outputs_AB.append(x_AB)

        film_AA = model.decoder.speaker_film(block_outputs_AA[-1], spk_A_t)
        film_AB = model.decoder.speaker_film(block_outputs_AB[-1], spk_B_t)

        stages_all = [
            ("cross_attn",     fused_AA,       fused_AB),
            ("resampler",      resampled_AA,   resampled_AB),
            ("adapter",        adapter_AA,     adapter_AB),
            ("adapter_film",   adapter_film_AA, adapter_film_AB),
            ("block0",         block_outputs_AA[0], block_outputs_AB[0]),
            ("block2",         block_outputs_AA[2], block_outputs_AB[2]),
            ("b3_identity",    b3_id_AA,       b3_id_AB),
            ("b3_id_film",     b3_id_film_AA,  b3_id_film_AB),
            ("b3_body",        b3_body_AA,     b3_body_AB),
            ("block3_sum",     block_outputs_AA[3], block_outputs_AB[3]),
            ("spk_film",       film_AA,        film_AB),
            ("mel+affine",    mel_raw_AA,     mel_raw_AB),
            ("mel_scaled",     mel_scaled_AA,  mel_scaled_AB),
            ("mel_final",      mel_final_AA,   mel_final_AB),
        ]

        default_rows = ["cross_attn", "block2", "block3_sum", "spk_film", "mel+affine", "mel_scaled"]
        rows_to_show = stages_all if full_audit else [s for s in stages_all if s[0] in default_rows]

        prev_cos = None
        print(f"  {'Stage':<15} {'cent_cos':>9} {'Δcent_cos':>10} {'σ(A)':>8} {'σ(B)':>8}")
        for name, aa, ab in rows_to_show:
            cc = _safe_cos(aa, ab)
            sa, sb = float(aa.std()), float(ab.std())
            if prev_cos is not None:
                d = cc - prev_cos
                print(f"  {name:<15} {cc:>9.4f} {d:>+10.4f} {sa:>8.3f} {sb:>8.3f}")
            else:
                print(f"  {name:<15} {cc:>9.4f} {'--':>10} {sa:>8.3f} {sb:>8.3f}")
            prev_cos = cc

    # Cross-attention key numbers
    with torch.no_grad():
        actual_temp = F.softplus(model.cross_attn.raw_temperature) + 0.01
        print(f"\n  [CROSS-ATTN]  T={actual_temp.item():.3f}")

    # ---- [5] OPTIONAL AUDITS ------------------------------------------------
    if forced_gamma_audit:
        print("\n" + "=" * 62)
        print("---- [5a] FORCED-GAMMA AUDIT ----")
        print("=" * 62)
        print(f"  {'gamma':>6} {'cross_σ':>9} {'adapter_σ':>10} {'block3_σ':>9} {'mel_σ':>9}")
        with torch.no_grad():
            for gv in [0.5, 1.0]:
                model.cross_attn.force_gamma = gv
                pred_f, _, _ = model(ref_A.unsqueeze(0), [content_A])
                print(f"  {gv:>6.1f} {float(pred_f.std()):>9.3f} {'--':>10} {'--':>9} {float(pred_f.std()):>9.3f}")
            model.cross_attn.force_gamma = None
        print("  force_gamma reset to None.")

    if full_audit:
        print("\n" + "=" * 62)
        print("---- [5b] FULL STAGE TABLE ----")
        print("=" * 62)
        print(f"  {'Stage':<15} {'cent_cos':>9} {'L1 diff':>9} {'σ(A)':>8} {'σ(B)':>8}")
        for name, aa, ab in stages_all:
            cc = _safe_cos(aa, ab)
            l1 = float((aa - ab).abs().mean())
            sa, sb = float(aa.std()), float(ab.std())
            print(f"  {name:<15} {cc:>9.4f} {l1:>9.4f} {sa:>8.3f} {sb:>8.3f}")

    if grad_audit:
        print("\n" + "=" * 62)
        print("---- [5c] GRADIENT AUDIT (1 training step) ----")
        print("=" * 62)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=0)
        opt.zero_grad()
        pred, loss_dict, _ = model(
            ref_audio=ref_A.unsqueeze(0),
            content_audio=[content_A],
            gt_mels=gt_mel_A.unsqueeze(0).to(device),
            compute_losses=True,
            return_aux=False)
        loss_dict["mel"].backward()

        modules_to_check = [
            "mel_encoder.projection", "hubert_proj", "info_bottleneck",
            "cross_attn.content_proj", "cross_attn.mapping_network",
            "decoder.adapter", "decoder.adapter_speaker_film",
            "decoder.blocks.0", "decoder.blocks.1", "decoder.blocks.2",
            "decoder.block3_id_film", "decoder.speaker_film",
            "decoder.mel_proj", "decoder.mel_speaker_affine", "decoder.raw_out_scale",
        ]
        print(f"  {'Module':<32} {'|grad| mean':>12} {'|grad| max':>12}")
        for mod_name in modules_to_check:
            total_mean = 0.0
            total_max = 0.0
            count = 0
            for name, param in model.named_parameters():
                if mod_name in name and param.requires_grad and param.grad is not None:
                    total_mean += param.grad.abs().mean().item()
                    total_max = max(total_max, param.grad.abs().max().item())
                    count += 1
            if count > 0:
                print(f"  {mod_name:<32} {total_mean/count:>12.2e} {total_max:>12.2e}")
            else:
                print(f"  {mod_name:<32} {'NO GRAD':>12} {'--':>12}")
        model.eval()

    # ---- Summary -------------------------------------------------------------
    print("\n" + "=" * 62)
    print("  LISTENING GUIDE:")
    print("=" * 62)
    print("  01_raw_content_A.wav  -> Ground truth Man (what was said)")
    print("  02_raw_content_B.wav  -> Ground truth Woman (what was said)")
    print("  05_self_recon_A.wav   -> Model: Man content + Man voice")
    print("  06_self_recon_B.wav   -> Model: Woman content + Woman voice")
    print("  07_cross_AtoB.wav     -> Model: Man WORDS in Woman VOICE  <- KEY")
    print("  08_cross_BtoA.wav     -> Model: Woman WORDS in Man VOICE  <- KEY")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--output_dir", type=str, default="generalization_test")
    parser.add_argument("--full-audit", action="store_true",
                        help="Show full 14-row stage table + channel diagnostics")
    parser.add_argument("--forced-gamma-audit", action="store_true",
                        help="Run forced-gamma decoder responsiveness test")
    parser.add_argument("--grad-audit", action="store_true",
                        help="Run 1 training step for per-module gradient norms")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable model-internal per-call debug prints")
    args = parser.parse_args()
    test_generalization(
        args.checkpoint, args.output_dir,
        full_audit=args.full_audit,
        forced_gamma_audit=args.forced_gamma_audit,
        grad_audit=args.grad_audit,
        verbose=args.verbose)
