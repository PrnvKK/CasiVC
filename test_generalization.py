import os
import json
import torch
import torchaudio
import argparse
import random
from pathlib import Path
from tqdm import tqdm

from config import AudioConfig, ModelConfig, TrainingConfig
from data.audio_utils import extract_mel_spectrogram, load_audio, split_utterance_for_training
from models.hubertvc_model import HubertVCModel
from inference import load_vocoder

# Try to load SpeechBrain for SPK_SIM
try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    print("\n[WARNING] SpeechBrain not installed. SPK_SIM metric will be skipped.")
    print("Run: pip install speechbrain\n")

def build_or_load_manifest(dataset_root: str, num_utterances: int = 50, seed: int = 42, manifest_path: str = "eval_manifest.json") -> list:
    """
    Deterministically selects num_utterances from the test-clean split (or dev-clean) and saves them.
    If the manifest already exists, loads it to ensure reproducibility across runs.
    """
    if os.path.exists(manifest_path):
        print(f"Loading existing evaluation manifest from {manifest_path}...")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        return manifest

    print(f"Building new deterministic evaluation manifest with {num_utterances} utterances...")
    
    # Check test-clean, fallback to dev-clean
    test_dir = os.path.join(dataset_root, "test-clean")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(dataset_root, "dev-clean")
        
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Neither test-clean nor dev-clean found in {dataset_root}")

    all_wavs = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.wav'):
                all_wavs.append(os.path.join(root, file))
                
    if not all_wavs:
        raise ValueError(f"No .wav files found in {test_dir}")

    # Deterministic sort and seed
    all_wavs.sort()
    rng = random.Random(seed)
    
    # Group by speaker to ensure diversity
    speaker_dict = {}
    for wav in all_wavs:
        # LibriTTS format: speaker/chapter/wav
        parts = Path(wav).parts
        speaker_id = parts[-3] if len(parts) >= 3 else "unknown"
        if speaker_id not in speaker_dict:
            speaker_dict[speaker_id] = []
        speaker_dict[speaker_id].append(wav)
        
    for spk in speaker_dict:
        rng.shuffle(speaker_dict[spk])
        
    selected_wavs = []
    speakers = sorted(list(speaker_dict.keys()))
    rng.shuffle(speakers)
    
    # Round robin selection
    while len(selected_wavs) < num_utterances and speakers:
        for spk in list(speakers):
            if speaker_dict[spk]:
                selected_wavs.append(speaker_dict[spk].pop(0))
                if len(selected_wavs) >= num_utterances:
                    break
            else:
                speakers.remove(spk)

    # Save manifest
    with open(manifest_path, 'w') as f:
        json.dump(selected_wavs, f, indent=2)
        
    print(f"Saved manifest to {manifest_path}")
    return selected_wavs

def calculate_spk_sim(encoder, wave1, wave2, device="cpu"):
    """Calculate cosine similarity between two waveforms using SpeechBrain ECAPA-TDNN."""
    if not SPEECHBRAIN_AVAILABLE or encoder is None:
        return 0.0
    with torch.no_grad():
        emb1 = encoder.encode_batch(wave1.to(device))
        emb2 = encoder.encode_batch(wave2.to(device))
        # embeddings are (B, 1, 192)
        sim = torch.nn.functional.cosine_similarity(emb1.squeeze(1), emb2.squeeze(1), dim=-1)
        return sim.item()

def run_evaluation(checkpoint_path: str, output_dir: str, smoke_test: bool = False, zero_speaker_delta: bool = False):
    print("="*70)
    print("🚀 RUNNING CASIVC GENERALIZATION EVALUATION")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    
    # ── Load Subsystems ──────────────────────────
    print("\n[1] Loading Vocoder & Metrics...")
    vocoder = load_vocoder(None, device=str(device))
    vocoder.eval()
    
    spk_encoder = None
    if SPEECHBRAIN_AVAILABLE:
        try:
            spk_encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": str(device)}
            )
            spk_encoder.eval()
            print("✅ Loaded ECAPA-TDNN for SPK_SIM metric.")
        except Exception as e:
            print(f"⚠️ Failed to load ECAPA-TDNN: {e}")
    
    print(f"\n[2] Loading Model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}. Exiting.")
        return
        
    model = HubertVCModel(audio_cfg, model_cfg, train_cfg).to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    try:
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model loaded successfully (Strict Mode Disabled).")
    except RuntimeError as e:
        print(f"\n❌ [ERROR] Loading failed!")
        print(e)
        print("\nPlease run a fresh training session from scratch.")
        return

    # ── Step 0 Diagnostic: zero the speaker delta branch ──────────
    # Patches speaker_delta_proj.forward to return zeros. The rest of
    # the forward (speaker_film, out_bias, clamp, vocoder) runs
    # unchanged — isolates the output-level speaker delta's contribution
    # to the disturbance. Compare audio and metrics against the normal
    # run to determine if the buzz is speaker-side or content-side.
    if zero_speaker_delta:
        def _zeroed_delta(x, speaker_feats):
            B, C, T = x.shape
            return torch.zeros(B, 80, T, device=x.device, dtype=x.dtype)
        model.decoder.speaker_delta_proj.forward = _zeroed_delta
        print("\n" + "="*60)
        print("⚠️  DIAGNOSTIC MODE: speaker_delta_proj ZEROED")
        print("   Measuring content path through full forward (no output delta).")
        print("="*60)

    # ── Manifest Setup ────────────────────────────
    print("\n[3] Preparing Data...")
    if smoke_test:
        print("💨 SMOKE TEST MODE: Running 2 fixed utterances.")
        manifest = [
            "/content/LibriTTS/dev-clean/2428/83705/2428_83705_000000_000001.wav",
            "/content/LibriTTS/dev-clean/1988/148538/1988_148538_000002_000000.wav"
        ]
    else:
        manifest = build_or_load_manifest("/content/LibriTTS", num_utterances=50, seed=42)
        
    # Validation
    valid_manifest = [p for p in manifest if os.path.exists(p)]
    if not valid_manifest:
        print("❌ No valid audio files found in manifest. Check your dataset paths.")
        return
        
    # We will create pairs: utterance[i] as content, utterance[(i+1)%N] as speaker reference
    # shuffled ref uses offset 2 so content/target/shuffled are three different files
    N = len(valid_manifest)
    shuffled_offset = 2 if N >= 3 else 1
    
    results = []

    print(f"\n[4] Running Evaluation on {N} Pairs...")

    avg_l1_self = 0.0
    avg_spk_sim_self = 0.0
    avg_spk_sim_cross = 0.0
    avg_spk_sim_ceiling = 0.0
    avg_spk_sim_shuf_to_shuf = 0.0
    avg_spk_sim_shuf_to_target = 0.0
    avg_mel_speaker_mag = 0.0
    avg_mel_speaker_rms_ratio = 0.0  # RMS(mel_speaker) / RMS(content_mel)
    avg_content_l1_self = 0.0         # L1 on prebias_mel (content-only) vs GT
    evaluated_pairs = 0

    # ── Step 1 audit (Action Plan Step 1) ───────────────────────────────
    # Cross-attention behaviour on the CROSS path (where source leakage lives):
    #   attn_entropy  : head-avg attention entropy per content frame.
    #                    low  → few speaker tokens dominate (sharp fusion)
    #                    high → diffuse (uniform ≈ log(num_tokens))
    #   attn_max      : mean per-frame max attention weight (sharpness proxy).
    #   cos_fused_src : cos(fused, content_feats_pre_attn) — residual pass-through.
    #                    high → cross_attn acts as identity-ish content relay.
    #   cos_fused_tgt : cos(fused, pooled speaker_tokens) — speaker fusion strength.
    #                    low  → speaker info barely enters fused features.
    #   sim_content_only_self : ECAPA SPK_SIM of vocoded prebias_mel on the
    #                            SELF path vs the source/own raw audio.
    #                            == Self SPK_SIM with mel_speaker removed:
    #                            isolates whether identity survives the decoder
    #                            without the (near-dead) output delta.
    avg_attn_entropy = 0.0
    avg_attn_max = 0.0
    avg_cos_fused_src = 0.0
    avg_cos_fused_tgt = 0.0
    avg_cos_attn_tgt = 0.0  # cos(attn_pre_film, spk_pooled) — CE probe readout
    avg_sim_content_only_self = 0.0  # only accumulates when SPK_SIM available
    n_attn_samples = 0  # only pairs with attention weights present
    n_sim_content_only = 0
    
    with torch.no_grad():
        for i in tqdm(range(N), desc="Evaluating Pairs"):
            content_path = valid_manifest[i]
            spk_path = valid_manifest[(i+1) % N]
            shuffled_path = valid_manifest[(i + shuffled_offset) % N]
            
            # Load
            content_audio = load_audio(content_path, sample_rate=audio_cfg.sample_rate).to(device)
            spk_audio = load_audio(spk_path, sample_rate=audio_cfg.sample_rate).to(device)
            shuffled_audio = load_audio(shuffled_path, sample_rate=audio_cfg.sample_rate).to(device)
            
            # Use FULL utterances for evaluation
            content_seg = content_audio
            spk_ref = spk_audio
            content_ref = content_audio
            shuffled_ref = shuffled_audio
            
            gt_mel_content = extract_mel_spectrogram(content_seg, sample_rate=audio_cfg.sample_rate).to(device)
            
            # ── 1. Self Reconstruction (Content A, Voice A) ──
            pred_mel_self, _, aux_self = model(
                content_ref.unsqueeze(0), [content_seg],
                return_bottleneck=True, return_aux=True
            )
            pred_mel_self_sq = pred_mel_self.squeeze(0)  # [80, T] for L1
            
            min_len = min(pred_mel_self_sq.size(-1), gt_mel_content.size(-1))
            l1_self = torch.nn.functional.l1_loss(
                pred_mel_self_sq[:, :min_len], 
                gt_mel_content[:, :min_len]
            ).item()
            avg_l1_self += l1_self
            
            # ── 2. Cross Conversion (Content A, Voice B) ──
            pred_mel_cross, _, aux_cross = model(
                spk_ref.unsqueeze(0), [content_seg],
                return_bottleneck=True, return_aux=True
            )
            
            # mel_speaker magnitude (measure whether speaker delta branch is active)
            if aux_cross is not None and "mel_speaker" in aux_cross:
                mel_spk = aux_cross["mel_speaker"]
                avg_mel_speaker_mag += mel_spk.abs().mean().item()
                # RMS ratio: how large is speaker delta relative to content?
                if "prebias_mel" in aux_cross:
                    content_mel = aux_cross["prebias_mel"]
                    rms_spk = mel_spk.pow(2).mean().sqrt().item()
                    rms_content = content_mel.pow(2).mean().sqrt().item()
                    avg_mel_speaker_rms_ratio += rms_spk / (rms_content + 1e-8)
            
            # ── 3. Shuffled Reference (Content A, Voice C) ──
            pred_mel_shuffled, _, _ = model(
                shuffled_ref.unsqueeze(0), [content_seg],
                return_bottleneck=True, return_aux=True
            )

            # ── Step 1 audit: cross-attention behaviour on the CROSS path ──
            # Computed under no_grad; purely diagnostic, no model changes.
            if aux_cross is not None and "attention_weights" in aux_cross \
                    and aux_cross["attention_weights"] is not None:
                aw = aux_cross["attention_weights"]              # [1, T, K]
                # entropy over speaker-token keys, averaged across content frames
                ent_per_frame = -(aw * (aw + 1e-9).log()).sum(dim=-1)  # [1, T]
                avg_attn_entropy += ent_per_frame.mean().item()
                # sharpness: per-frame max attention probability, averaged
                avg_attn_max += aw.max(dim=-1).values.mean().item()
                n_attn_samples += 1

            if aux_cross is not None \
                    and "fused_features" in aux_cross \
                    and "content_feats_pre_attn" in aux_cross \
                    and "speaker_tokens" in aux_cross:
                fused = aux_cross["fused_features"]               # [1, T, 96]
                cf = aux_cross["content_feats_pre_attn"]          # [1, T, 96]
                st = aux_cross["speaker_tokens"]                  # [1, K, 96]
                pooled_spk = st.mean(dim=1, keepdim=True)         # [1, 1, 96]
                pooled_spk_b = pooled_spk.expand(-1, fused.size(1), -1)  # [1, T, 96]
                cos_fc = torch.nn.functional.cosine_similarity(fused, cf, dim=-1)         # [1, T]
                cos_fs = torch.nn.functional.cosine_similarity(fused, pooled_spk_b, dim=-1)  # [1, T]
                avg_cos_fused_src += cos_fc.mean().item()
                avg_cos_fused_tgt += cos_fs.mean().item()
            
            # ── Step 1 audit: cos(attn_pre_film, spk_pooled) — CE probe readout ──
            if aux_cross is not None and "attn_feats_pre_film" in aux_cross:
                af = aux_cross["attn_feats_pre_film"]            # [1, T, 96]
                st = aux_cross["speaker_tokens"]                  # [1, K, 96]
                pooled_spk = st.mean(dim=1, keepdim=True)        # [1, 1, 96]
                pooled_spk_b = pooled_spk.expand(-1, af.size(1), -1)
                cos_afs = torch.nn.functional.cosine_similarity(af, pooled_spk_b, dim=-1)
                avg_cos_attn_tgt += cos_afs.mean().item()
            
            # ── Vocoder & SPK_SIM ──
            if spk_encoder is not None:
                # Target Speaker Ceiling (GT Mel of speaker reference -> Vocoded)
                gt_mel_spk = extract_mel_spectrogram(spk_ref, sample_rate=audio_cfg.sample_rate)
                wav_ceiling = vocoder(gt_mel_spk.unsqueeze(0).to(device)).squeeze(0).cpu()
                
                # Self-recon vocoded
                wav_self = vocoder(pred_mel_self).squeeze(0).cpu()
                wav_content_raw = content_audio.cpu().unsqueeze(0)
                
                # Cross-conversion vocoded
                wav_cross = vocoder(pred_mel_cross).squeeze(0).cpu()
                wav_target_raw = spk_ref.cpu().unsqueeze(0)
                
                # Shuffled-conversion vocoded
                wav_shuffled = vocoder(pred_mel_shuffled).squeeze(0).cpu()
                wav_shuffled_raw = shuffled_audio.cpu().unsqueeze(0)
                
                # Ceiling SPK SIM (Vocoded GT vs Raw)
                sim_ceil = calculate_spk_sim(spk_encoder, wav_ceiling, wav_target_raw, device)
                avg_spk_sim_ceiling += sim_ceil
                
                # Self SPK SIM (Self-recon vs Content Raw)
                sim_self = calculate_spk_sim(spk_encoder, wav_self, wav_content_raw, device)
                avg_spk_sim_self += sim_self

                # Step 1 audit: content-only SPK_SIM — vocoded prebias_mel
                # on the SELF path (no speaker delta) vs the own raw audio.
                # If this ≈ Self SPK_SIM, mel_speaker really is dead and the
                # identity erosion is happening upstream of the output delta.
                if aux_self is not None and "prebias_mel" in aux_self:
                    pred_content_mel_self = aux_self["prebias_mel"].to(device)
                    wav_content_only = vocoder(pred_content_mel_self).squeeze(0).cpu()
                    sim_content_only = calculate_spk_sim(
                        spk_encoder, wav_content_only, wav_content_raw, device
                    )
                    avg_sim_content_only_self += sim_content_only
                    n_sim_content_only += 1
                
                # Model SPK SIM (Cross vs Target Raw)
                sim_cross = calculate_spk_sim(spk_encoder, wav_cross, wav_target_raw, device)
                avg_spk_sim_cross += sim_cross
                
                # SPK SIM (Shuffled vs Shuffled Raw) — does the model track the shuffled ref?
                sim_shuf_to_shuf = calculate_spk_sim(spk_encoder, wav_shuffled, wav_shuffled_raw, device)
                avg_spk_sim_shuf_to_shuf += sim_shuf_to_shuf
                
                # SPK SIM (Shuffled vs Original Target Raw) — should be low (no tonic)
                sim_shuf_to_target = calculate_spk_sim(spk_encoder, wav_shuffled, wav_target_raw, device)
                avg_spk_sim_shuf_to_target += sim_shuf_to_target
            
            # ── Three-way ablation: content-only L1 (prebias_mel, no speaker delta) ──
            if aux_self is not None and "prebias_mel" in aux_self:
                content_mel_self = aux_self["prebias_mel"].squeeze(0)  # [80, T]
                min_len_c = min(content_mel_self.size(-1), gt_mel_content.size(-1))
                l1_content = torch.nn.functional.l1_loss(
                    content_mel_self[:, :min_len_c],
                    gt_mel_content[:, :min_len_c]
                ).item()
                avg_content_l1_self += l1_content
                
            # Save first 5 pairs for manual listening
            if i < 5:
                torchaudio.save(os.path.join(output_dir, f"pair_{i}_source_content.wav"), content_seg.cpu().unsqueeze(0), audio_cfg.sample_rate)
                torchaudio.save(os.path.join(output_dir, f"pair_{i}_target_voice.wav"), spk_ref.cpu().unsqueeze(0), audio_cfg.sample_rate)
                torchaudio.save(os.path.join(output_dir, f"pair_{i}_shuffled_voice.wav"), shuffled_ref.cpu().unsqueeze(0), audio_cfg.sample_rate)
                
                wav_self_save = vocoder(pred_mel_self).squeeze(0).cpu()
                torchaudio.save(os.path.join(output_dir, f"pair_{i}_pred_self.wav"), wav_self_save, audio_cfg.sample_rate)
                
                # Three-way ablation: content-only path (prebias_mel, no speaker delta)
                if aux_self is not None and "prebias_mel" in aux_self:
                    content_mel_self = aux_self["prebias_mel"]  # [1, 80, T] or [80, T]
                    if content_mel_self.dim() == 2:
                        content_mel_self = content_mel_self.unsqueeze(0)
                    wav_content = vocoder(content_mel_self.to(device)).squeeze(0).cpu()
                    torchaudio.save(os.path.join(output_dir, f"pair_{i}_pred_content_only.wav"), wav_content, audio_cfg.sample_rate)
                
                wav_cross_save = vocoder(pred_mel_cross).squeeze(0).cpu()
                torchaudio.save(os.path.join(output_dir, f"pair_{i}_pred_cross.wav"), wav_cross_save, audio_cfg.sample_rate)
                
                wav_shuf_save = vocoder(pred_mel_shuffled).squeeze(0).cpu()
                torchaudio.save(os.path.join(output_dir, f"pair_{i}_pred_shuffled.wav"), wav_shuf_save, audio_cfg.sample_rate)

            evaluated_pairs += 1

    # ── Summary Report ─────────────────────────────
    if evaluated_pairs == 0:
        print("\n❌ No pairs were long enough to evaluate. Please check your dataset.")
        return
        
    avg_l1_self /= evaluated_pairs
    avg_spk_sim_self /= evaluated_pairs
    avg_spk_sim_cross /= evaluated_pairs
    avg_spk_sim_ceiling /= evaluated_pairs
    avg_spk_sim_shuf_to_shuf /= evaluated_pairs
    avg_spk_sim_shuf_to_target /= evaluated_pairs
    avg_mel_speaker_mag /= evaluated_pairs
    avg_mel_speaker_rms_ratio /= evaluated_pairs
    avg_content_l1_self /= evaluated_pairs

    # Step 1 audit averages (guard against zero denominators)
    if n_attn_samples > 0:
        avg_attn_entropy /= n_attn_samples
        avg_attn_max /= n_attn_samples
    else:
        avg_attn_entropy = 0.0
        avg_attn_max = 0.0
    if evaluated_pairs > 0:
        avg_cos_fused_src /= evaluated_pairs
        avg_cos_fused_tgt /= evaluated_pairs
    else:
        avg_cos_fused_src = 0.0
        avg_cos_fused_tgt = 0.0
    if n_sim_content_only > 0:
        avg_sim_content_only_self /= n_sim_content_only
    else:
        avg_sim_content_only_self = 0.0
    
    print("\n" + "="*60)
    print("📊 EVALUATION SCOREBOARD")
    print("="*60)
    print(f"  Pairs Evaluated:          {evaluated_pairs} (out of {len(valid_manifest)})")
    print(f"  Self-Recon Mel L1:        {avg_l1_self:.4f} (Lower is better)")
    if avg_content_l1_self > 0:
        print(f"  Content-Only L1 (prebias):{avg_content_l1_self:.4f} (vs full {avg_l1_self:.4f} — gap = speaker-delta impact)")
    if spk_encoder is not None:
        print(f"  Self SPK_SIM:             {avg_spk_sim_self:.4f} (Target > 0.70, upper ~0.92)")
        print(f"  Cross SPK_SIM:            {avg_spk_sim_cross:.4f} (Target > 0.35)")
        print(f"  Vocoder Ceiling SIM:      {avg_spk_sim_ceiling:.4f} (Upper bound for vocoded audio)")
        print(f"  Shuffled→Shuffled SIM:    {avg_spk_sim_shuf_to_shuf:.4f}")
        print(f"  Shuffled→Target SIM:      {avg_spk_sim_shuf_to_target:.4f}")
        print(f"  Delta (shuf→shuf minus shuf→target):  {avg_spk_sim_shuf_to_shuf - avg_spk_sim_shuf_to_target:.4f} (should be > 0.05)")
        if avg_mel_speaker_mag > 0:
            print(f"  mel_speaker |mean|:       {avg_mel_speaker_mag:.4f} (should be > 0.01)")
            print(f"  RMS(mel_speaker)/RMS(content): {avg_mel_speaker_rms_ratio:.4f} (<0.05=negligible, >0.15=dominant)")
        else:
            print(f"  mel_speaker |mean|:       [not available]")
    else:
        print("  SPK_SIM:                   [Skipped - SpeechBrain missing]")

    # ── Step 1 audit scoreboard (Action Plan Step 1) ───────────────────
    import math as _math
    print("-"*60)
    print("🔎 STEP 1 AUDIT — cross-attention fusion strength")
    print("-"*60)
    aux_aw = aux_cross.get("attention_weights") if aux_cross is not None else None
    n_tokens = int(aux_aw.shape[-1]) if aux_aw is not None else 0
    print(f"  Attn Entropy (cross):     {avg_attn_entropy:.4f}"
          + (f"  (uniform ≈ {_math.log(n_tokens):.2f} over {n_tokens} tokens)"
             if n_tokens > 0 else "  (n/a)"))
    print(f"  Attn Max Weight (cross):  {avg_attn_max:.4f}  (1.0 = single-token; "
          f"1/{n_tokens if n_tokens else '?'} = uniform)")
    print(f"  cos(fused, content_pre):  {avg_cos_fused_src:.4f}  → high = residual pass-through")
    print(f"  cos(fused, spk_pooled):   {avg_cos_fused_tgt:.4f}  → high = speaker fusion active")
    if n_attn_samples > 0:
        attn_avg = avg_cos_attn_tgt / n_attn_samples
    else:
        attn_avg = 0.0
    print(f"  cos(attn_pre_film, spk_pooled):  {attn_avg:.4f}  → primary readout for CE probe success")
    if spk_encoder is not None and n_sim_content_only > 0:
        print("-"*60)
        print("🔎 STEP 1 AUDIT — identity preservation without output delta")
        print("-"*60)
        print(f"  Content-Only SPK_SIM:     {avg_sim_content_only_self:.4f}  "
              f"(vs Self SPK_SIM {avg_spk_sim_self:.4f} — Δ = "
              f"{avg_spk_sim_self - avg_sim_content_only_self:.4f})")
    print("="*60)
    print(f"Sample audio saved to {output_dir}/")
    print("Listen to pair_0_pred_self.wav to verify basic intelligibility.")
    print("Compare pair_0_pred_content_only.wav vs pair_0_pred_self.wav to isolate noise source.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CasiVC Generalization")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt", help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="generalization_outputs", help="Output directory for audio")
    parser.add_argument("--smoke", action="store_true", help="Run quick 2-utterance smoke test instead of full 50-pair manifest")
    parser.add_argument("--zero_speaker_delta", action="store_true", help="Zero the speaker delta branch to isolate disturbance source (Step 0 diagnostic)")
    args = parser.parse_args()
    
    run_evaluation(args.checkpoint, args.output_dir, args.smoke, args.zero_speaker_delta)
