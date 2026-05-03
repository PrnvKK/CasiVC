import os
import torch
import torchaudio
import argparse
from pathlib import Path

from config import AudioConfig, ModelConfig, TrainingConfig
from data.audio_utils import extract_mel_spectrogram, load_audio, split_utterance_for_training
from models.hubertvc_model import HubertVCModel
from inference import load_vocoder, load_speaker_encoder

# ---------------------------------------------------------------------------
#  Diagnostic helpers
# ---------------------------------------------------------------------------
def compute_band_metric(pred_mel, target_mel):
    """Per-band mean/std L1 distance (timbre bias metric)."""
    pred_mean = pred_mel.mean(dim=-1)   # [n_mels]
    tgt_mean = target_mel.mean(dim=-1)
    pred_std = pred_mel.std(dim=-1)
    tgt_std = target_mel.std(dim=-1)
    mean_l1 = torch.nn.functional.l1_loss(pred_mean, tgt_mean).item()
    std_l1 = torch.nn.functional.l1_loss(pred_std, tgt_std).item()
    return mean_l1, std_l1

def compute_spk_sim(waveform, ref_waveform, speaker_encoder):
    """Cosine similarity between speaker embeddings (Resemblyzer)."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if ref_waveform.dim() == 1:
        ref_waveform = ref_waveform.unsqueeze(0)
    emb_pred = speaker_encoder.encode_batch(waveform)
    emb_ref = speaker_encoder.encode_batch(ref_waveform)
    emb_pred = torch.nn.functional.normalize(emb_pred.squeeze(1), p=2, dim=1)
    emb_ref = torch.nn.functional.normalize(emb_ref.squeeze(1), p=2, dim=1)
    return torch.nn.functional.cosine_similarity(emb_pred, emb_ref, dim=-1).item()

def extract_aux_metrics(aux):
    """Extract scalar diagnostics from model aux dict."""
    cross_attn = aux.get("cross_attn", {}) if aux else {}
    return {
        "block2_std": aux.get("block2_std", float('nan')) if aux else float('nan'),
        "film_gamma_ts": cross_attn.get("gamma_temporal_std", torch.tensor(float('nan'))).item()
                         if cross_attn.get("gamma_temporal_std") is not None else float('nan'),
        "ca_entropy": cross_attn.get("entropy", torch.tensor(float('nan'))).item()
                      if cross_attn.get("entropy") is not None else float('nan'),
        "ca_temperature": cross_attn.get("temperature", torch.tensor(float('nan'))).item()
                          if cross_attn.get("temperature") is not None else float('nan'),
    }

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
    
    # ── Load speaker encoder for SPK_SIM ─────────────────────────────
    spk_encoder = load_speaker_encoder(None, device=str(device))
    spk_encoder.eval()
    print("✅ Speaker encoder loaded for SPK_SIM.")
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

    # ── Results accumulator for output_store.txt ────────────────────
    results = {}

    with torch.no_grad():
        # ── Test 3: Self-reconstruction A → A ─────────────────────
        print("\n[5] Self-reconstruction: A content + A voice...")
        pred_mel_AA, _, aux_AA = model(ref_A.unsqueeze(0), [content_A], return_aux=True, return_bottleneck=True)
        wave_AA = vocoder(pred_mel_AA).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "05_self_recon_A.wav"), wave_AA.unsqueeze(0), audio_cfg.sample_rate)
        # Diagnostics
        m = extract_aux_metrics(aux_AA)
        m["mel_output_std"] = pred_mel_AA.std().item()
        gt_mel_AA = extract_mel_spectrogram(content_A, sample_rate=audio_cfg.sample_rate)
        m["gt_mel_std"] = gt_mel_AA.std().item()
        m["spk_sim"] = compute_spk_sim(wave_AA.to(device), content_A.to(device), spk_encoder)
        m["band_mean_l1"], m["band_std_l1"] = compute_band_metric(pred_mel_AA.squeeze(0).cpu(), gt_mel_AA.cpu())
        results["self_A"] = m
        print(f"   Pred mel: {pred_mel_AA.shape}, mean={pred_mel_AA.mean():.4f}")
        print(f"   Metrics: block2_std={m['block2_std']:.3f} mel_std={m['mel_output_std']:.3f} "
              f"spk_sim={m['spk_sim']:.3f} band_mean_l1={m['band_mean_l1']:.4f}")
        print("✅ Saved A→A")

        # ── Test 4: Self-reconstruction B → B ─────────────────────
        print("\n[6] Self-reconstruction: B content + B voice...")
        pred_mel_BB, _, aux_BB = model(ref_B.unsqueeze(0), [content_B], return_aux=True, return_bottleneck=True)
        wave_BB = vocoder(pred_mel_BB).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "06_self_recon_B.wav"), wave_BB.unsqueeze(0), audio_cfg.sample_rate)
        # Diagnostics
        m = extract_aux_metrics(aux_BB)
        m["mel_output_std"] = pred_mel_BB.std().item()
        gt_mel_BB = extract_mel_spectrogram(content_B, sample_rate=audio_cfg.sample_rate)
        m["gt_mel_std"] = gt_mel_BB.std().item()
        m["spk_sim"] = compute_spk_sim(wave_BB.to(device), content_B.to(device), spk_encoder)
        m["band_mean_l1"], m["band_std_l1"] = compute_band_metric(pred_mel_BB.squeeze(0).cpu(), gt_mel_BB.cpu())
        results["self_B"] = m
        print(f"   Pred mel: {pred_mel_BB.shape}, mean={pred_mel_BB.mean():.4f}")
        print(f"   Metrics: block2_std={m['block2_std']:.3f} mel_std={m['mel_output_std']:.3f} "
              f"spk_sim={m['spk_sim']:.3f} band_mean_l1={m['band_mean_l1']:.4f}")
        print("✅ Saved B→B")

        # ── Test 5: Cross-conversion A content → B voice (Man speaking as Woman) ──────────
        print("\n[7] Cross-conversion: A content + B voice (Man speaking as Woman)...")
        pred_mel_AB, _, aux_AB = model(ref_B.unsqueeze(0), [content_A], return_aux=True, return_bottleneck=True)
        wave_AB = vocoder(pred_mel_AB).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "07_cross_AtoB.wav"), wave_AB.unsqueeze(0), audio_cfg.sample_rate)
        # Diagnostics — compare against TARGET speaker (B)
        m = extract_aux_metrics(aux_AB)
        m["mel_output_std"] = pred_mel_AB.std().item()
        m["spk_sim"] = compute_spk_sim(wave_AB.to(device), ref_B.to(device), spk_encoder)
        gt_mel_B = extract_mel_spectrogram(content_B, sample_rate=audio_cfg.sample_rate)
        m["band_mean_l1"], m["band_std_l1"] = compute_band_metric(pred_mel_AB.squeeze(0).cpu(), gt_mel_B.cpu())
        results["cross_AtoB"] = m
        print(f"   Pred mel: {pred_mel_AB.shape}, mean={pred_mel_AB.mean():.4f}")
        print(f"   Metrics: block2_std={m['block2_std']:.3f} mel_std={m['mel_output_std']:.3f} "
              f"spk_sim={m['spk_sim']:.3f} band_mean_l1={m['band_mean_l1']:.4f}")
        print("✅ Saved A→B")

        # ── Test 6: Cross-conversion B content → A voice (Woman speaking as Man) ──────────
        print("\n[8] Cross-conversion: B content + A voice (Woman speaking as Man)...")
        pred_mel_BA, _, aux_BA = model(ref_A.unsqueeze(0), [content_B], return_aux=True, return_bottleneck=True)
        wave_BA = vocoder(pred_mel_BA).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "08_cross_BtoA.wav"), wave_BA.unsqueeze(0), audio_cfg.sample_rate)
        # Diagnostics — compare against TARGET speaker (A)
        m = extract_aux_metrics(aux_BA)
        m["mel_output_std"] = pred_mel_BA.std().item()
        m["spk_sim"] = compute_spk_sim(wave_BA.to(device), ref_A.to(device), spk_encoder)
        gt_mel_A = extract_mel_spectrogram(content_A, sample_rate=audio_cfg.sample_rate)
        m["band_mean_l1"], m["band_std_l1"] = compute_band_metric(pred_mel_BA.squeeze(0).cpu(), gt_mel_A.cpu())
        results["cross_BtoA"] = m
        print(f"   Pred mel: {pred_mel_BA.shape}, mean={pred_mel_BA.mean():.4f}")
        print(f"   Metrics: block2_std={m['block2_std']:.3f} mel_std={m['mel_output_std']:.3f} "
              f"spk_sim={m['spk_sim']:.3f} band_mean_l1={m['band_mean_l1']:.4f}")
        print("✅ Saved B→A")

    print("\n" + "="*60)
    print("📋 LISTENING GUIDE:")
    print("="*60)
    print("  01_raw_content_A.wav   → Ground truth Man (what was said)")
    print("  02_raw_content_B.wav   → Ground truth Woman (what was said)")
    print("  05_self_recon_A.wav    → Model: Man content + Man voice")
    print("  06_self_recon_B.wav    → Model: Woman content + Woman voice")
    print("  07_cross_AtoB.wav      → Model: Man's WORDS in Woman's VOICE ← Key Test!")
    print("  08_cross_BtoA.wav      → Model: Woman's WORDS in Man's VOICE ← Key Test!")
    print("="*60)

    # ── Write diagnostics to output_store.txt ──────────────────────
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_store.txt")
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  CasiVC Generalization Test — Diagnostic Metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Speaker A: {spk_A}\n")
        f.write(f"Speaker B: {spk_B}\n\n")
        
        f.write("─" * 60 + "\n")
        f.write("  TARGET RANGES (for interpretation)\n")
        f.write("─" * 60 + "\n")
        f.write("  block2_std:         0.9 – 1.2  (noise-free decoder)\n")
        f.write("  mel_output_std:     ≥ 1.8      (clear audio, proper spectral variance)\n")
        f.write("  film_gamma_ts:      > 0.01     (per-frame FiLM modulation active)\n")
        f.write("  ca_entropy:         1.80 – 1.92 (cross-attention not collapsed)\n")
        f.write("  ca_temperature:     ~ 0.49     (attention sharpness)\n")
        f.write("  spk_sim:            ≥ 0.75     (target speaker identity preserved)\n")
        f.write("  band_mean_l1:       lower = better timbre match\n")
        f.write("  band_std_l1:        lower = better timbre match\n\n")

        for name, m in results.items():
            f.write("─" * 60 + "\n")
            f.write(f"  {name}\n")
            f.write("─" * 60 + "\n")
            for key in ["block2_std", "mel_output_std", "film_gamma_ts", "ca_entropy",
                        "ca_temperature", "spk_sim", "band_mean_l1", "band_std_l1"]:
                val = m.get(key, float('nan'))
                f.write(f"  {key:20s} = {val:.4f}\n")
            # Also show gt_mel_std for self-recon
            if "gt_mel_std" in m:
                f.write(f"  {'gt_mel_std':20s} = {m['gt_mel_std']:.4f}\n")
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("  End of diagnostics\n")
        f.write("=" * 60 + "\n")
    print(f"\n📊 Diagnostics written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt", help="Path to your trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="generalization_test", help="Folder to save the output audio files")
    args = parser.parse_args()
    test_generalization(args.checkpoint, args.output_dir)
