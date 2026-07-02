import os
import json
import torch
import torchaudio
import argparse
import random
import soundfile as sf
import numpy as np
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
        # S21 SpeakerDeltaProj.forward signature is (x, speaker_feats,
        # content_speaker_feats=None) -> (delta, gate, cos_value). The patch
        # must match so the 3-tuple unpack at the call site doesn't crash.
        # Delta force-zeroed; gate/cos returned as ones/zeros (delta is gone,
        # so gate diagnostics are moot in this mode).
        def _zeroed_delta(x, speaker_feats, content_speaker_feats=None):
            B, C, T = x.shape
            delta = torch.zeros(B, 80, T, device=x.device, dtype=x.dtype)
            gate = torch.ones(B, device=x.device, dtype=x.dtype)
            cos_value = torch.zeros(B, device=x.device, dtype=x.dtype)
            return delta, gate, cos_value
        model.decoder.speaker_delta_proj.forward = _zeroed_delta
        print("\n" + "="*60)
        print("⚠️  DIAGNOSTIC MODE: speaker_delta_proj ZEROED (global, all paths)")
        print("   Measuring content path through full forward (no output delta).")
        print("   NOTE: in-run Cross Content-Only SPK_SIM (prebias_mel) supersedes")
        print("   this for the S22 decision — this flag is a heavier audio-inspection hammer.")
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
    avg_speaker_delta_gate = 0.0       # S21: G2 gate on cross path (should be ≥0.3)
    avg_speaker_delta_cos = 0.0         # S21: cos between content and target pooled tokens
    n_gate_samples = 0
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
    avg_sim_content_only_cross = 0.0  # S22 check: cross content-only (delta removed)
    n_attn_samples = 0  # only pairs with attention weights present
    # ── Decoder trace audit: cos(stage_pooled, spk_pooled) at each 96-dim stage ──
    decoder_trace = {}  # name → accumulated cos sum
    decoder_trace_counts = {}
    n_sim_content_only = 0
    n_sim_content_only_cross = 0  # S22 check: cross content-only pairs
    
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
            
            # ── S21: raw content speaker ECAPA (B,192) for G2 gate ──
            content_spk_raw = model.mel_encoder.extract_speaker_features(
                content_seg.unsqueeze(0), apply_projection=False
            )[0]  # [192]

            # ── 1. Self Reconstruction (Content A, Voice A) ──
            pred_mel_self, _, aux_self = model(
                content_ref.unsqueeze(0), [content_seg],
                return_bottleneck=True, return_aux=True,
                precomputed_content_speaker_feats=content_spk_raw.unsqueeze(0)  # [1, 192]
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
                return_bottleneck=True, return_aux=True,
                precomputed_content_speaker_feats=content_spk_raw.unsqueeze(0)  # [1, 192]
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
            
            # S21: collect G2 gate/cos from cross path
            if aux_cross is not None and "speaker_delta_gate" in aux_cross:
                avg_speaker_delta_gate += aux_cross["speaker_delta_gate"].mean().item()
                avg_speaker_delta_cos += aux_cross["speaker_delta_cos"].mean().item()
                n_gate_samples += 1
            
            # ── 3. Shuffled Reference (Content A, Voice C) ──
            pred_mel_shuffled, _, _ = model(
                shuffled_ref.unsqueeze(0), [content_seg],
                return_bottleneck=True, return_aux=True,
                precomputed_content_speaker_feats=content_spk_raw.unsqueeze(0)  # [1, 192]
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

            # ── Decoder trace: cos(stage_pooled, spk_pooled) at each 96-dim stage ──
            if aux_cross is not None and "speaker_tokens" in aux_cross:
                sp = aux_cross["speaker_tokens"].mean(dim=1).squeeze(0)  # [96]
                stages = {}
                if "resampled_features" in aux_cross:
                    stages["resampled"] = aux_cross["resampled_features"].squeeze(0).mean(dim=0)
                if "post_adapter" in aux_cross:
                    stages["post_adapter"] = aux_cross["post_adapter"].squeeze(0).mean(dim=1)
                if aux_cross.get("decoder_intermediates"):
                    for j, feat in enumerate(aux_cross["decoder_intermediates"]):
                        stages[f"block{j}"] = feat.squeeze(0).mean(dim=1)  # [B,C,T]→[C]→pool
                for name, pooled in stages.items():
                    if pooled.shape[0] == sp.shape[0]:  # same dim for cos
                        cos_val = torch.nn.functional.cosine_similarity(pooled.unsqueeze(0), sp.unsqueeze(0)).item()
                        decoder_trace.setdefault(name, 0.0)
                        decoder_trace[name] += cos_val
                        decoder_trace_counts[name] = decoder_trace_counts.get(name, 0) + 1

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

                # S22 check: Cross content-only SPK_SIM — vocoded prebias_mel on the
                # CROSS path (content_mel with NO speaker delta) vs TARGET speaker raw.
                # Δ from Cross SPK_SIM = speaker delta's contribution to cross identity.
                #   Δ ≈ 0 → delta decorative at output → S22 (pre-adapter x) wrong lever.
                #   Δ > 0 → delta load-bearing      → S22 (pre-adapter x) well-motivated.
                if aux_cross is not None and "prebias_mel" in aux_cross:
                    pred_content_mel_cross = aux_cross["prebias_mel"].to(device)
                    wav_cross_content_only = vocoder(pred_content_mel_cross).squeeze(0).cpu()
                    sim_content_only_cross = calculate_spk_sim(
                        spk_encoder, wav_cross_content_only, wav_target_raw, device
                    )
                    avg_sim_content_only_cross += sim_content_only_cross
                    n_sim_content_only_cross += 1
                
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
                def _save_audio(filepath, tensor_1d, sr):
                    data = tensor_1d.detach().cpu().float().numpy()
                    sf.write(filepath, data, sr)
                    if i == 0:
                        print(f"  [SAVE] {filepath}  ({data.shape[0] / sr:.1f}s)")

                _save_audio(os.path.join(output_dir, f"pair_{i}_source_content.wav"), content_seg, audio_cfg.sample_rate)
                _save_audio(os.path.join(output_dir, f"pair_{i}_target_voice.wav"), spk_ref, audio_cfg.sample_rate)
                _save_audio(os.path.join(output_dir, f"pair_{i}_shuffled_voice.wav"), shuffled_ref, audio_cfg.sample_rate)

                wav_self_save = vocoder(pred_mel_self).squeeze(0).cpu()
                _save_audio(os.path.join(output_dir, f"pair_{i}_pred_self.wav"), wav_self_save, audio_cfg.sample_rate)

                # Three-way ablation: content-only path (prebias_mel, no speaker delta)
                if aux_self is not None and "prebias_mel" in aux_self:
                    content_mel_self = aux_self["prebias_mel"]  # [1, 80, T] or [80, T]
                    if content_mel_self.dim() == 2:
                        content_mel_self = content_mel_self.unsqueeze(0)
                    wav_content = vocoder(content_mel_self.to(device)).squeeze(0).cpu()
                    _save_audio(os.path.join(output_dir, f"pair_{i}_pred_content_only.wav"), wav_content, audio_cfg.sample_rate)

                wav_cross_save = vocoder(pred_mel_cross).squeeze(0).cpu()
                _save_audio(os.path.join(output_dir, f"pair_{i}_pred_cross.wav"), wav_cross_save, audio_cfg.sample_rate)

                wav_shuf_save = vocoder(pred_mel_shuffled).squeeze(0).cpu()
                _save_audio(os.path.join(output_dir, f"pair_{i}_pred_shuffled.wav"), wav_shuf_save, audio_cfg.sample_rate)

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

    if n_gate_samples > 0:
        avg_speaker_delta_gate /= n_gate_samples
        avg_speaker_delta_cos /= n_gate_samples

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
    if n_sim_content_only_cross > 0:
        avg_sim_content_only_cross /= n_sim_content_only_cross
    else:
        avg_sim_content_only_cross = 0.0
    
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
        if n_sim_content_only_cross > 0:
            print(f"  Cross Content-Only SPK_SIM: {avg_sim_content_only_cross:.4f}  (delta Δ = {avg_spk_sim_cross - avg_sim_content_only_cross:.4f} — S22 decision)")
        print(f"  Vocoder Ceiling SIM:      {avg_spk_sim_ceiling:.4f} (Upper bound for vocoded audio)")
        print(f"  Shuffled→Shuffled SIM:    {avg_spk_sim_shuf_to_shuf:.4f}")
        print(f"  Shuffled→Target SIM:      {avg_spk_sim_shuf_to_target:.4f}")
        print(f"  Delta (shuf→shuf minus shuf→target):  {avg_spk_sim_shuf_to_shuf - avg_spk_sim_shuf_to_target:.4f} (should be > 0.05)")
        if avg_mel_speaker_mag > 0:
            print(f"  mel_speaker |mean|:       {avg_mel_speaker_mag:.4f} (should be > 0.01)")
            print(f"  RMS(mel_speaker)/RMS(content): {avg_mel_speaker_rms_ratio:.4f} (<0.05=negligible, >0.15=dominant)")
        else:
            print(f"  mel_speaker |mean|:       [not available]")
        print(f"  ── S21 G2 Gate Diagnostics ──")
        if n_gate_samples > 0:
            print(f"  speaker_delta_gate (cross): {avg_speaker_delta_gate:.4f} (target ≥ 0.3)")
            print(f"  speaker_delta_cos (cross):  {avg_speaker_delta_cos:.4f} (target ≤ 0.7)")
        else:
            print(f"  speaker_delta_gate/cos:     [not available — check aux dict availability]")
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
    # ── Decoder trace print ──
    if decoder_trace:
        print("  ── DECODER TRACE: cos(stage_pooled, spk_pooled) ──")
        for name in ["resampled", "post_adapter", "block0", "block1", "block2", "block3"]:
            if name in decoder_trace:
                val = decoder_trace[name] / max(decoder_trace_counts.get(name, 1), 1)
                print(f"    {name:14s}: {val:+.4f}")
    if spk_encoder is not None and n_sim_content_only > 0:
        print("-"*60)
        print("🔎 STEP 1 AUDIT — identity preservation without output delta")
        print("-"*60)
        print(f"  Content-Only SPK_SIM:     {avg_sim_content_only_self:.4f}  "
              f"(vs Self SPK_SIM {avg_spk_sim_self:.4f} — Δ = "
              f"{avg_spk_sim_self - avg_sim_content_only_self:.4f})")
    if spk_encoder is not None and n_sim_content_only_cross > 0:
        print("-"*60)
        print("🔎 S22 DECISION — cross identity with/without speaker delta")
        print("-"*60)
        print(f"  Cross Content-Only SPK_SIM: {avg_sim_content_only_cross:.4f}  "
              f"(vs Cross SPK_SIM {avg_spk_sim_cross:.4f} — Δ = "
              f"{avg_spk_sim_cross - avg_sim_content_only_cross:.4f})")
        print(f"  → Δ = speaker delta's contribution to cross identity (decisive for S22):")
        print(f"     Δ ≈ 0  → delta decorative at output → S22 (pre-adapter x) WRONG lever")
        print(f"     Δ > 0  → delta load-bearing         → S22 (pre-adapter x) well-motivated")
    print("="*60)
    print(f"Sample audio saved to {output_dir}/")
    print("Listen to pair_0_pred_self.wav to verify basic intelligibility.")
    print("Compare pair_0_pred_content_only.wav vs pair_0_pred_self.wav to isolate noise source.")


def run_delta_scale_sweep(checkpoint_path: str, output_dir: str, smoke_test: bool = False,
                          s_values=None, num_pairs: int = 50):
    """Eval-only sweep: scale SpeakerDeltaProj's output delta by s and measure
    Cross/Self SPK_SIM. Decoder combines as variance_mel.detach() + mel_speaker
    (mobilenet_decoder.py:865), so scaling the delta at the source yields
    pred_mel = clamp(content_mel + out_bias + s*mel_speaker). No training, no
    model code change — forward is patched per s and restored in finally.

    Decision rule (best s by Cross SPK_SIM):
      s~0    -> delta harmful; retire SpeakerDeltaProj, tune upstream alpha_bias/cross-attn
      s<0    -> delta has speaker info but wrong sign/basis (flippable)
      0<s<1  -> delta over-amplified; dial down (raw_delta_scale / CE weight)
      s>1    -> delta needs more power -> S22 (richer x) or alpha-reduction viable
    """
    if s_values is None:
        s_values = [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5]

    print("=" * 70)
    print("🎚️  DELTA SCALE SWEEP — SpeakerDeltaProj output scaling (eval-only)")
    print("=" * 70)
    print(f"s_values: {s_values}  | pairs: {num_pairs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    # ── Load vocoder + SPK_SIM encoder + model (mirrors run_evaluation) ──
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
    if spk_encoder is None:
        print("❌ SPK_SIM requires SpeechBrain ECAPA-TDNN. Aborting sweep.")
        return

    print(f"\n[2] Loading Model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}. Exiting.")
        return
    model = HubertVCModel(audio_cfg, model_cfg, train_cfg).to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict, strict=False)
    print("✅ Model loaded successfully (Strict Mode Disabled).")

    # Suppress per-call verbose prints (sweep makes hundreds of calls; keep output clean)
    model._verbose = False
    model.decoder._verbose = False
    model.cross_attn._verbose = False
    model.decoder.speaker_delta_proj._verbose = False

    # ── Manifest ──
    print("\n[3] Preparing Data...")
    if smoke_test:
        print("💨 SMOKE TEST MODE: Running 2 fixed utterances.")
        manifest = [
            "/content/LibriTTS/dev-clean/2428/83705/2428_83705_000000_000001.wav",
            "/content/LibriTTS/dev-clean/1988/148538/1988_148538_000002_000000.wav"
        ]
    else:
        manifest = build_or_load_manifest("/content/LibriTTS", num_utterances=num_pairs, seed=42)
    valid_manifest = [p for p in manifest if os.path.exists(p)]
    if not valid_manifest:
        print("❌ No valid audio files found in manifest. Check your dataset paths.")
        return
    N = min(len(valid_manifest), num_pairs)
    valid_manifest = valid_manifest[:N]
    print(f"Using {N} pairs (content[i] -> target[(i+1)%N]).")

    # ── Per-s accumulators ──
    cross_sims = {s: 0.0 for s in s_values}
    self_sims = {s: 0.0 for s in s_values}
    self_l1s = {s: 0.0 for s in s_values}
    counts = {s: 0 for s in s_values}

    # Save the bound class-method forward; patch per s, restore in finally.
    sdp = model.decoder.speaker_delta_proj
    _orig_forward = sdp.forward

    def _make_scaled(scalar, orig):
        def _scaled(x, speaker_feats, content_speaker_feats=None):
            delta, gate, cos = orig(x, speaker_feats, content_speaker_feats)
            return delta * scalar, gate, cos  # real gate/cos preserved; only delta scaled
        return _scaled

    print(f"\n[4] Sweeping {len(s_values)} scale values x {N} pairs...")
    try:
        with torch.no_grad():
            for i in tqdm(range(N), desc="Sweep pairs"):
                content_path = valid_manifest[i]
                spk_path = valid_manifest[(i + 1) % N]

                content_audio = load_audio(content_path, sample_rate=audio_cfg.sample_rate).to(device)
                spk_ref = load_audio(spk_path, sample_rate=audio_cfg.sample_rate).to(device)
                content_seg = content_audio
                content_ref = content_audio

                gt_mel_content = extract_mel_spectrogram(content_seg, sample_rate=audio_cfg.sample_rate).to(device)

                # Raw content speaker ECAPA (B,192) for the G2 gate
                content_spk_raw = model.mel_encoder.extract_speaker_features(
                    content_seg.unsqueeze(0), apply_projection=False
                )[0]  # [192]

                wav_content_raw = content_audio.cpu().unsqueeze(0)
                wav_target_raw = spk_ref.cpu().unsqueeze(0)

                for s in s_values:
                    sdp.forward = _make_scaled(s, _orig_forward)

                    # Cross conversion (Content A, Voice B) with delta scaled by s
                    pred_mel_cross, _, _ = model(
                        spk_ref.unsqueeze(0), [content_seg],
                        return_bottleneck=True, return_aux=True,
                        precomputed_content_speaker_feats=content_spk_raw.unsqueeze(0)
                    )
                    wav_cross = vocoder(pred_mel_cross).squeeze(0).cpu()
                    cross_sims[s] += calculate_spk_sim(spk_encoder, wav_cross, wav_target_raw, device)

                    # Self reconstruction (Content A, Voice A) — gate~0 => delta~0 => s-invariant
                    pred_mel_self, _, _ = model(
                        content_ref.unsqueeze(0), [content_seg],
                        return_bottleneck=True, return_aux=True,
                        precomputed_content_speaker_feats=content_spk_raw.unsqueeze(0)
                    )
                    wav_self = vocoder(pred_mel_self).squeeze(0).cpu()
                    self_sims[s] += calculate_spk_sim(spk_encoder, wav_self, wav_content_raw, device)

                    # Self L1 (reconstruction quality, should be ~flat across s)
                    pred_sq = pred_mel_self.squeeze(0)
                    min_len = min(pred_sq.size(-1), gt_mel_content.size(-1))
                    self_l1s[s] += torch.nn.functional.l1_loss(
                        pred_sq[:, :min_len], gt_mel_content[:, :min_len]
                    ).item()
                    counts[s] += 1
    finally:
        sdp.forward = _orig_forward  # restore original forward

    # ── Average + report ──
    rows = []
    for s in s_values:
        c = max(counts[s], 1)
        rows.append((s, cross_sims[s] / c, self_sims[s] / c, self_l1s[s] / c))

    best = max(rows, key=lambda r: r[1])  # best by Cross SPK_SIM
    s0_cross = next((r[1] for r in rows if r[0] == 0.0), None)

    print("\n" + "=" * 60)
    print("📊 DELTA SCALE SWEEP RESULTS")
    print("=" * 60)
    print(f"{'s':>6} {'Cross SPK_SIM':>16} {'Self SPK_SIM':>16} {'Self L1':>10}")
    print("-" * 52)
    for s, cs, ss, l1 in rows:
        marker = "  <- best cross" if s == best[0] else ""
        print(f"{s:>6.2f} {cs:>16.4f} {ss:>16.4f} {l1:>10.4f}{marker}")
    print("-" * 52)
    print(f"Best s (cross) = {best[0]:.2f}  ->  Cross SPK_SIM = {best[1]:.4f}")
    if s0_cross is not None:
        print(f"s=0 Cross SPK_SIM = {s0_cross:.4f}  (prebias ablation was 0.1940; close = out_bias SPK_SIM-neutral, confirmed)")
    print("\n→ Decision rule (best s by Cross SPK_SIM):")
    print("   best s ~ 0    -> delta harmful; retire SpeakerDeltaProj, tune upstream alpha_bias/cross-attn")
    print("   best s < 0    -> delta has speaker info but wrong sign/basis (flippable)")
    print("   0 < best < 1  -> delta over-amplified; dial down (raw_delta_scale / CE weight)")
    print("   best s > 1    -> delta needs more power -> S22 (richer x) or alpha-reduction viable")
    print("=" * 60)


def run_alpha_bias_sweep(checkpoint_path: str, output_dir: str, smoke_test: bool = False,
                         k_values=None, num_pairs: int = 50):
    """Eval-only sweep: scale the detached speaker-bias injection in cross-attention
    (cross_attention.py:380, 487 — the alpha_bias * spk_bias term) by k and measure
    Cross/Self SPK_SIM. The bias injection is the most direct knob on how much
    target-speaker signal enters fused_features upstream. k=1.0 is the trained point
    (should reproduce Cross SPK_SIM ~0.1816). No training; _bias_scale_k defaults to
    1.0 so normal training/eval is byte-identical.

    Decision rule (best k by Cross SPK_SIM):
      Cross rises monotonically with k -> entry-limited; upstream injection is the lever
      Cross flat / drops at high k     -> crush-limited; decoder crushes the extra (S19)
      Cross peaks then declines        -> mild entry-limit + overshoot at high k
    NOTE: scales only the additive bias injection, not the attention K/V fusion.
    """
    if k_values is None:
        k_values = [0.0, 0.5, 1.0, 2.0, 4.0]

    print("=" * 70)
    print("🎚️  ALPHA_BIAS SCALE SWEEP — upstream speaker-bias injection (eval-only)")
    print("=" * 70)
    print(f"k_values: {k_values}  | pairs: {num_pairs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    # ── Load vocoder + SPK_SIM encoder + model (mirrors run_evaluation) ──
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
    if spk_encoder is None:
        print("❌ SPK_SIM requires SpeechBrain ECAPA-TDNN. Aborting sweep.")
        return

    print(f"\n[2] Loading Model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}. Exiting.")
        return
    model = HubertVCModel(audio_cfg, model_cfg, train_cfg).to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict, strict=False)
    print("✅ Model loaded successfully (Strict Mode Disabled).")

    # Suppress per-call verbose prints (sweep makes hundreds of calls)
    model._verbose = False
    model.decoder._verbose = False
    model.cross_attn._verbose = False
    model.decoder.speaker_delta_proj._verbose = False

    # ── Manifest ──
    print("\n[3] Preparing Data...")
    if smoke_test:
        print("💨 SMOKE TEST MODE: Running 2 fixed utterances.")
        manifest = [
            "/content/LibriTTS/dev-clean/2428/83705/2428_83705_000000_000001.wav",
            "/content/LibriTTS/dev-clean/1988/148538/1988_148538_000002_000000.wav"
        ]
    else:
        manifest = build_or_load_manifest("/content/LibriTTS", num_utterances=num_pairs, seed=42)
    valid_manifest = [p for p in manifest if os.path.exists(p)]
    if not valid_manifest:
        print("❌ No valid audio files found in manifest. Check your dataset paths.")
        return
    N = min(len(valid_manifest), num_pairs)
    valid_manifest = valid_manifest[:N]
    print(f"Using {N} pairs (content[i] -> target[(i+1)%N]).")

    # ── Per-k accumulators ──
    cross_sims = {k: 0.0 for k in k_values}
    self_sims = {k: 0.0 for k in k_values}
    counts = {k: 0 for k in k_values}

    print(f"\n[4] Sweeping {len(k_values)} scale values x {N} pairs...")
    try:
        with torch.no_grad():
            for i in tqdm(range(N), desc="Sweep pairs"):
                content_path = valid_manifest[i]
                spk_path = valid_manifest[(i + 1) % N]

                content_audio = load_audio(content_path, sample_rate=audio_cfg.sample_rate).to(device)
                spk_ref = load_audio(spk_path, sample_rate=audio_cfg.sample_rate).to(device)
                content_seg = content_audio
                content_ref = content_audio

                content_spk_raw = model.mel_encoder.extract_speaker_features(
                    content_seg.unsqueeze(0), apply_projection=False
                )[0]  # [192]

                wav_content_raw = content_audio.cpu().unsqueeze(0)
                wav_target_raw = spk_ref.cpu().unsqueeze(0)

                for k in k_values:
                    model.cross_attn._bias_scale_k = k

                    # Cross conversion (Content A, Voice B) with bias injection scaled by k
                    pred_mel_cross, _, _ = model(
                        spk_ref.unsqueeze(0), [content_seg],
                        return_bottleneck=True, return_aux=True,
                        precomputed_content_speaker_feats=content_spk_raw.unsqueeze(0)
                    )
                    wav_cross = vocoder(pred_mel_cross).squeeze(0).cpu()
                    cross_sims[k] += calculate_spk_sim(spk_encoder, wav_cross, wav_target_raw, device)

                    # Self reconstruction — bias is self-speaker here, so it reinforces
                    # self identity (expect self SPK_SIM to rise slightly with k).
                    pred_mel_self, _, _ = model(
                        content_ref.unsqueeze(0), [content_seg],
                        return_bottleneck=True, return_aux=True,
                        precomputed_content_speaker_feats=content_spk_raw.unsqueeze(0)
                    )
                    wav_self = vocoder(pred_mel_self).squeeze(0).cpu()
                    self_sims[k] += calculate_spk_sim(spk_encoder, wav_self, wav_content_raw, device)

                    counts[k] += 1
    finally:
        model.cross_attn._bias_scale_k = 1.0  # restore

    # ── Average + report ──
    rows = []
    for k in k_values:
        c = max(counts[k], 1)
        rows.append((k, cross_sims[k] / c, self_sims[k] / c))

    best = max(rows, key=lambda r: r[1])  # best by Cross SPK_SIM
    k1_cross = next((r[1] for r in rows if r[0] == 1.0), None)

    print("\n" + "=" * 60)
    print("📊 ALPHA_BIAS SCALE SWEEP RESULTS")
    print("=" * 60)
    print(f"{'k':>6} {'Cross SPK_SIM':>16} {'Self SPK_SIM':>16}")
    print("-" * 42)
    for k, cs, ss in rows:
        marker = "  <- best cross" if k == best[0] else ""
        print(f"{k:>6.2f} {cs:>16.4f} {ss:>16.4f}{marker}")
    print("-" * 42)
    print(f"Best k (cross) = {best[0]:.2f}  ->  Cross SPK_SIM = {best[1]:.4f}")
    if k1_cross is not None:
        print(f"k=1.0 Cross SPK_SIM = {k1_cross:.4f}  (trained point; should match E30 Cross ~0.1816)")
    print("\n→ Decision rule (best k by Cross SPK_SIM):")
    print("   Cross rises monotonically with k -> ENTRY-LIMITED; upstream injection is the lever (strengthen alpha_bias / cross-attn)")
    print("   Cross flat / drops at high k     -> CRUSH-LIMITED; decoder crushes the extra (S19) -> decoder is the ceiling")
    print("   Cross peaks then declines        -> mild entry-limit + overshoot at high k")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CasiVC Generalization")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt", help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="generalization_outputs", help="Output directory for audio")
    parser.add_argument("--smoke", action="store_true", help="Run quick 2-utterance smoke test instead of full 50-pair manifest")
    parser.add_argument("--zero_speaker_delta", action="store_true", help="Zero the speaker delta branch to isolate disturbance source (Step 0 diagnostic)")
    parser.add_argument("--delta_sweep", action="store_true", help="Eval-only sweep: scale mel_speaker by s in {-1..1.5}, measure Cross/Self SPK_SIM. Decisive for S22 motivation.")
    parser.add_argument("--s_values", type=str, default=None, help="Comma-separated scale factors, e.g. '-1,-0.5,0,0.25,0.5,1,1.5' (default)")
    parser.add_argument("--num_pairs", type=int, default=50, help="Number of pairs for the sweep (default 50, comparable to the E30 scoreboard)")
    parser.add_argument("--alpha_bias_sweep", action="store_true", help="Eval-only sweep: scale the upstream speaker-bias injection (cross_attn alpha_bias) by k in {0..4}. Tests entry- vs crush-limited.")
    parser.add_argument("--k_values", type=str, default=None, help="Comma-separated bias scale factors, e.g. '0,0.5,1,2,4' (default)")
    args = parser.parse_args()
    
    if args.delta_sweep:
        s_vals = None
        if args.s_values:
            s_vals = [float(v.strip()) for v in args.s_values.split(",")]
        run_delta_scale_sweep(args.checkpoint, args.output_dir, args.smoke, s_vals, num_pairs=args.num_pairs)
    elif args.alpha_bias_sweep:
        k_vals = None
        if args.k_values:
            k_vals = [float(v.strip()) for v in args.k_values.split(",")]
        run_alpha_bias_sweep(args.checkpoint, args.output_dir, args.smoke, k_vals, num_pairs=args.num_pairs)
    else:
        run_evaluation(args.checkpoint, args.output_dir, args.smoke, args.zero_speaker_delta)
