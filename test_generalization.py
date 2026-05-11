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
    print(f"   GT mel A: mean={gt_mel_A.mean():.4f}, std={gt_mel_A.std():.4f}")
    print(f"   GT mel B: mean={gt_mel_B.mean():.4f}, std={gt_mel_B.std():.4f}")

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

    # Initialize beta cache for per-speaker FiLM analysis
    beta_cache = {}

    # ── Interpolation Probe ──────────────────────────────────────────
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
        beta_cache['blend'] = model.cross_attn._cached_beta.clone()
        wave_blend = vocoder(pred_blend_A).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "09_blend_A_spk50.wav"), 
                        wave_blend.unsqueeze(0), audio_cfg.sample_rate)
        print(f"  Blended mel mean: {pred_blend_A.mean():.4f}, std={pred_blend_A.std():.4f}")
        print("✅ Saved 09_blend_A_spk50.wav  (Man content + 50% Man / 50% Woman voice)")
    print()

    # ================================================================
    # DIAGNOSTIC A: Information Bottleneck Speaker Leakage Probe
    # ================================================================
    print("\n" + "="*60)
    print("[DIAGNOSTIC A] INFORMATION BOTTLENECK 32D SPEAKER LEAKAGE")
    print("="*60)
    with torch.no_grad():
        # Run both speakers' audio through HuBERT → hubert_proj → bottleneck[:3]
        hubert_A = model.hubert([content_A.to(device)])      # [1, T_A, 768]
        hubert_B = model.hubert([content_B.to(device)])      # [1, T_B, 768]
        proj_A = model.hubert_proj(hubert_A)                  # [1, T_A, 96]
        proj_B = model.hubert_proj(hubert_B)                  # [1, T_B, 96]
        # Extract 32D chokepoint: info_bottleneck[:3] = Linear→LayerNorm→GELU
        bn_32_A = model.info_bottleneck[:3](proj_A).squeeze(0)  # [T_A, 32]
        bn_32_B = model.info_bottleneck[:3](proj_B).squeeze(0)  # [T_B, 32]

        mean_A = bn_32_A.mean(dim=0)  # [32]
        mean_B = bn_32_B.mean(dim=0)  # [32]

        cos_bn = torch.nn.functional.cosine_similarity(
            mean_A.unsqueeze(0), mean_B.unsqueeze(0)
        ).item()
        l2_bn = torch.norm(mean_A - mean_B, p=2).item()

        print(f"  Bottleneck frames A: {bn_32_A.shape[0]}, B: {bn_32_B.shape[0]}")
        print(f"  Mean-A vector: μ={mean_A.mean():.4f}, σ={mean_A.std():.4f}")
        print(f"  Mean-B vector: μ={mean_B.mean():.4f}, σ={mean_B.std():.4f}")
        print(f"  Cosine similarity (mean-A vs mean-B): {cos_bn:.4f}")
        print(f"  L2 distance   (mean-A vs mean-B):    {l2_bn:.4f}")

        # Nearest-centroid per-frame classification
        centroids = torch.stack([mean_A, mean_B])          # [2, 32]
        all_feats = torch.cat([bn_32_A, bn_32_B])           # [T_A+T_B, 32]
        labels = torch.cat([
            torch.zeros(bn_32_A.shape[0]),
            torch.ones(bn_32_B.shape[0])
        ]).to(device)
        dists = torch.cdist(all_feats, centroids)           # [T_A+T_B, 2]
        preds = dists.argmin(dim=1)
        acc = (preds == labels).float().mean().item()

        print(f"  Nearest-centroid frame accuracy: {acc:.4f} (chance=0.50)")
        if acc > 0.70:
            print(f"  ⚠️  SEVERE LEAK: Bottleneck 32D strongly retains speaker identity (>70%)")
        elif acc > 0.60:
            print(f"  ⚠️  MODERATE LEAK: Source-speaker info survives bottleneck (>60%)")
        elif acc > 0.55:
            print(f"  ⚠️  MILD: Weak speaker signal in bottleneck (55-60%)")
        else:
            print(f"  ✅ Bottleneck effectively scrubs speaker-discriminable content (≤55%)")
    print("="*60)

    # ──────────────────────────────────────────────────────────────────

    with torch.no_grad():
        # ── Test 3: Self-reconstruction A → A ─────────────────────
        print("\n[5] Self-reconstruction: A content + A voice...")
        pred_mel_AA, _, _ = model(ref_A.unsqueeze(0), [content_A])
        beta_cache['AA'] = model.cross_attn._cached_beta.clone()
        wave_AA = vocoder(pred_mel_AA).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "05_self_recon_A.wav"), wave_AA.unsqueeze(0), audio_cfg.sample_rate)
        print(f"   Pred mel: {pred_mel_AA.shape}, mean={pred_mel_AA.mean():.4f}, std={pred_mel_AA.std():.4f}")
        print("✅ Saved A→A")

        # ── Test 4: Self-reconstruction B → B ─────────────────────
        print("\n[6] Self-reconstruction: B content + B voice...")
        pred_mel_BB, _, _ = model(ref_B.unsqueeze(0), [content_B])
        beta_cache['BB'] = model.cross_attn._cached_beta.clone()
        wave_BB = vocoder(pred_mel_BB).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "06_self_recon_B.wav"), wave_BB.unsqueeze(0), audio_cfg.sample_rate)
        print(f"   Pred mel: {pred_mel_BB.shape}, mean={pred_mel_BB.mean():.4f}, std={pred_mel_BB.std():.4f}")
        print("✅ Saved B→B")

        # ── Test 5: Cross-conversion A content → B voice (Man speaking as Woman) ──────────
        print("\n[7] Cross-conversion: A content + B voice (Man speaking as Woman)...")
        pred_mel_AB, _, _ = model(ref_B.unsqueeze(0), [content_A])
        beta_cache['AB'] = model.cross_attn._cached_beta.clone()
        wave_AB = vocoder(pred_mel_AB).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "07_cross_AtoB.wav"), wave_AB.unsqueeze(0), audio_cfg.sample_rate)
        print(f"   Pred mel: {pred_mel_AB.shape}, mean={pred_mel_AB.mean():.4f}, std={pred_mel_AB.std():.4f}")
        print("✅ Saved A→B")

        # ── Test 6: Cross-conversion B content → A voice (Woman speaking as Man) ──────────
        print("\n[8] Cross-conversion: B content + A voice (Woman speaking as Man)...")
        pred_mel_BA, _, _ = model(ref_A.unsqueeze(0), [content_B])
        beta_cache['BA'] = model.cross_attn._cached_beta.clone()
        wave_BA = vocoder(pred_mel_BA).squeeze(0).squeeze(0).cpu()
        torchaudio.save(os.path.join(output_dir, "08_cross_BtoA.wav"), wave_BA.unsqueeze(0), audio_cfg.sample_rate)
        print(f"   Pred mel: {pred_mel_BA.shape}, mean={pred_mel_BA.mean():.4f}, std={pred_mel_BA.std():.4f}")
        print("✅ Saved B→A")

    # ================================================================
    # DIAGNOSTIC B: FiLM Beta Per-Speaker Analysis
    # ================================================================
    print("\n" + "="*60)
    print("[DIAGNOSTIC B] FiLM BETA PER-SPEAKER ANALYSIS")
    print("="*60)
    for key in ['AA', 'BB', 'AB', 'BA', 'blend']:
        if key in beta_cache:
            b = beta_cache[key].squeeze(0)  # [T, 96]
            ch_mean = b.mean(dim=0)         # [96] per-channel temporal mean
            ch_std  = b.std(dim=0)          # [96] per-channel temporal std
            print(f"  beta_{key:5s}: ch-mean μ={ch_mean.mean():.4f}, σ={ch_mean.std():.4f}, "
                  f"ch-std μ={ch_std.mean():.4f}, |μ|max={ch_mean.abs().max():.4f}")

    # Compare A-speaker beta (AA) vs B-speaker beta (BB) channel-wise
    if 'AA' in beta_cache and 'BB' in beta_cache:
        ch_mean_AA = beta_cache['AA'].squeeze(0).mean(dim=0)  # [96]
        ch_mean_BB = beta_cache['BB'].squeeze(0).mean(dim=0)  # [96]
        beta_diff = (ch_mean_AA - ch_mean_BB).abs()
        top5_diff = beta_diff.topk(5)
        print(f"\n  Self-recon beta channel-mean Δ (|A − B|):")
        print(f"    mean Δ = {beta_diff.mean():.4f}, max = {beta_diff.max():.4f}")
        print(f"    Top-5 differing channels: {top5_diff.indices.tolist()}")
        print(f"    Values: {[f'{v:.4f}' for v in top5_diff.values.tolist()]}")
        if beta_diff.mean() < 0.05:
            print(f"  ⚠️  WARNING: Beta channel-means nearly identical across speakers (Δ<0.05)")
            print(f"  → FiLM beta is NOT encoding per-speaker DC offset / spectral envelope!")
            print(f"  → This explains why cross-conversion mel-mean follows source speaker.")
        elif beta_diff.mean() < 0.15:
            print(f"  ⚠️  MILD: Beta carries weak speaker signal (0.05≤Δ<0.15)")
        else:
            print(f"  ✅ Beta channel-means differ meaningfully across speakers (Δ≥0.15)")

    # Cross-conversion: does beta change when target speaker changes?
    if 'AB' in beta_cache and 'AA' in beta_cache:
        ch_mean_AB = beta_cache['AB'].squeeze(0).mean(dim=0)
        ch_mean_AA = beta_cache['AA'].squeeze(0).mean(dim=0)
        delta_same_content = (ch_mean_AB - ch_mean_AA).abs()
        print(f"\n  Cross-pair beta shift (A→B minus A→A, same content):")
        print(f"    mean |Δβ| = {delta_same_content.mean():.4f}, max = {delta_same_content.max():.4f}")
        if delta_same_content.mean() < 0.02:
            print(f"  ⚠️  Beta barely moves when target speaker changes (same content)!")
            print(f"  → FiLM conditioning is being OVERRIDDEN by content-path signal.")

    # attended_features std from cross-attention logs (summary reference)
    print(f"\n  [REFERENCE] See cross_attn logs above for attended_features std per condition.")
    print(f"  Key metric: attended_features std should differ between self and cross conditions.")
    print("="*60)

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
    
    print("\n" + "="*60)
    print("📊 VARIANCE SUMMARY (GT vs Pred)")
    print("="*60)
    print(f"  GT mel A:           mean={gt_mel_A.mean():.4f}, std={gt_mel_A.std():.4f}")
    print(f"  GT mel B:           mean={gt_mel_B.mean():.4f}, std={gt_mel_B.std():.4f}")
    print(f"  A→A pred:           mean={pred_mel_AA.mean():.4f}, std={pred_mel_AA.std():.4f}  "
          f"(std ratio={pred_mel_AA.std().item()/gt_mel_A.std().item():.3f})")
    print(f"  B→B pred:           mean={pred_mel_BB.mean():.4f}, std={pred_mel_BB.std():.4f}  "
          f"(std ratio={pred_mel_BB.std().item()/gt_mel_B.std().item():.3f})")
    print(f"  A→B pred:           mean={pred_mel_AB.mean():.4f}, std={pred_mel_AB.std():.4f}  "
          f"(std ratio={pred_mel_AB.std().item()/gt_mel_B.std().item():.3f})")
    print(f"  B→A pred:           mean={pred_mel_BA.mean():.4f}, std={pred_mel_BA.std():.4f}  "
          f"(std ratio={pred_mel_BA.std().item()/gt_mel_A.std().item():.3f})")
    print(f"  Blend (A+50%):      mean={pred_blend_A.mean():.4f}, std={pred_blend_A.std():.4f}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt", help="Path to your trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="generalization_test", help="Folder to save the output audio files")
    args = parser.parse_args()
    test_generalization(args.checkpoint, args.output_dir)
