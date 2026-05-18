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


    # ── Mel variance summary (key out_scale health check) ─────────────
    print("\n[MEL VARIANCE SUMMARY]  (target σ ≈ 2.5 for HiFi-GAN)")
    gt_A_std  = gt_mel_A.std().item()
    gt_B_std  = gt_mel_B.std().item()
    aa_std    = pred_mel_AA.std().item()
    bb_std    = pred_mel_BB.std().item()
    ab_std    = pred_mel_AB.std().item()
    ba_std    = pred_mel_BA.std().item()
    print(f"  GT A σ={gt_A_std:.4f}  |  GT B σ={gt_B_std:.4f}")
    print(f"  A→A σ={aa_std:.4f}  |  B→B σ={bb_std:.4f}  |  A→B σ={ab_std:.4f}  |  B→A σ={ba_std:.4f}")
    deficit = ((2.5 - aa_std) / 2.5) * 100
    print(f"  Variance deficit (A→A vs 2.5): {deficit:.1f}%  {'✅ OK' if deficit < 15 else '❌ Compressed'}")
    # ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt", help="Path to your trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="generalization_test", help="Folder to save the output audio files")
    args = parser.parse_args()
    test_generalization(args.checkpoint, args.output_dir)
