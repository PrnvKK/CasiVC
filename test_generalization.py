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

def run_evaluation(checkpoint_path: str, output_dir: str, smoke_test: bool = False):
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
    results = []
    
    print(f"\n[4] Running Evaluation on {len(valid_manifest)} Pairs...")
    
    avg_l1_self = 0.0
    avg_spk_sim_cross = 0.0
    avg_spk_sim_ceiling = 0.0
    evaluated_pairs = 0
    
    with torch.no_grad():
        for i in tqdm(range(len(valid_manifest)), desc="Evaluating Pairs"):
            content_path = valid_manifest[i]
            spk_path = valid_manifest[(i+1) % len(valid_manifest)]
            
            # Load
            content_audio = load_audio(content_path, sample_rate=audio_cfg.sample_rate).to(device)
            spk_audio = load_audio(spk_path, sample_rate=audio_cfg.sample_rate).to(device)
            
            # Use FULL utterances for evaluation instead of chopping them into tiny fragments
            content_seg = content_audio
            spk_ref = spk_audio
            content_ref = content_audio
            
            gt_mel_content = extract_mel_spectrogram(content_seg, sample_rate=audio_cfg.sample_rate).to(device)
            
            # -- Self Reconstruction (Content A, Voice A) --
            pred_mel_self, _, _ = model(content_ref.unsqueeze(0), [content_seg])
            pred_mel_self = pred_mel_self.squeeze(0)
            
            min_len = min(pred_mel_self.size(-1), gt_mel_content.size(-1))
            l1_self = torch.nn.functional.l1_loss(
                pred_mel_self[:, :min_len], 
                gt_mel_content[:, :min_len]
            ).item()
            avg_l1_self += l1_self
            
            # -- Cross Conversion (Content A, Voice B) --
            pred_mel_cross, _, _ = model(spk_ref.unsqueeze(0), [content_seg])
            
            # -- Vocoder & SPK_SIM --
            if spk_encoder is not None:
                # Target Speaker Ceiling (GT Mel of speaker reference -> Vocoded)
                gt_mel_spk = extract_mel_spectrogram(spk_ref, sample_rate=audio_cfg.sample_rate)
                wav_ceiling = vocoder(gt_mel_spk.unsqueeze(0).to(device)).squeeze(0).cpu()
                
                # Cross-conversion output
                wav_cross = vocoder(pred_mel_cross).squeeze(0).cpu()
                
                # Raw target reference (Ground Truth)
                wav_target_raw = spk_ref.cpu().unsqueeze(0)
                
                # Ceiling SPK SIM (Vocoded vs Raw)
                sim_ceil = calculate_spk_sim(spk_encoder, wav_ceiling, wav_target_raw, device)
                avg_spk_sim_ceiling += sim_ceil
                
                # Model SPK SIM (Cross vs Raw)
                sim_cross = calculate_spk_sim(spk_encoder, wav_cross, wav_target_raw, device)
                avg_spk_sim_cross += sim_cross
                
            # Save first 5 pairs for manual listening
            if i < 5:
                torchaudio.save(os.path.join(output_dir, f"pair_{i}_source_content.wav"), content_seg.cpu().unsqueeze(0), audio_cfg.sample_rate)
                torchaudio.save(os.path.join(output_dir, f"pair_{i}_target_voice.wav"), spk_ref.cpu().unsqueeze(0), audio_cfg.sample_rate)
                
                wav_self = vocoder(pred_mel_self).squeeze(0).cpu()
                torchaudio.save(os.path.join(output_dir, f"pair_{i}_pred_self.wav"), wav_self, audio_cfg.sample_rate)
                
                wav_cross = vocoder(pred_mel_cross).squeeze(0).cpu()
                torchaudio.save(os.path.join(output_dir, f"pair_{i}_pred_cross.wav"), wav_cross, audio_cfg.sample_rate)

                
            evaluated_pairs += 1

    # ── Summary Report ─────────────────────────────
    if evaluated_pairs == 0:
        print("\n❌ No pairs were long enough to evaluate. Please check your dataset.")
        return
        
    avg_l1_self /= evaluated_pairs
    avg_spk_sim_cross /= evaluated_pairs
    avg_spk_sim_ceiling /= evaluated_pairs
    
    print("\n" + "="*60)
    print("📊 EVALUATION SCOREBOARD")
    print("="*60)
    print(f"  Pairs Evaluated:       {evaluated_pairs} (out of {len(valid_manifest)})")
    print(f"  Self-Recon Mel L1:     {avg_l1_self:.4f} (Lower is better)")
    if spk_encoder is not None:
        print(f"  Cross SPK_SIM:         {avg_spk_sim_cross:.4f} (Target > 0.35)")
        print(f"  Vocoder Ceiling SIM:   {avg_spk_sim_ceiling:.4f} (Upper bound for vocoded audio)")
    else:
        print("  SPK_SIM:               [Skipped - SpeechBrain missing]")
    print("="*60)
    print(f"Sample audio saved to {output_dir}/")
    print("Listen to pair_0_pred_self.wav to verify basic intelligibility.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CasiVC Generalization")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt", help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="generalization_outputs", help="Output directory for audio")
    parser.add_argument("--smoke", action="store_true", help="Run quick 2-utterance smoke test instead of full 50-pair manifest")
    args = parser.parse_args()
    
    run_evaluation(args.checkpoint, args.output_dir, args.smoke)
