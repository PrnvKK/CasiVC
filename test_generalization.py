import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.pretrained import EncoderClassifier

from config import AudioConfig, ModelConfig, TrainingConfig, DataConfig
from data.audio_utils import extract_mel_spectrogram, load_audio, split_utterance_for_training
from models.hubertvc_model import HubertVCModel
from inference import load_vocoder
from data.dataset import VoiceConversionDataset
import numpy as np
import random

BANDS = {
    "low": (0, 20),
    "mid": (20, 60),
    "high": (60, 80),
}


def _load_eval_speaker_encoder(device: torch.device):
    print("[SPK_EVAL] Loading independent SpeechBrain ECAPA...")
    spk_eval = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
        savedir="pretrained_models/spkrec-ecapa-voxceleb-eval",
    )
    spk_eval.eval()
    return spk_eval


def _extract_eval_embedding(spk_eval, wave: torch.Tensor, device: torch.device) -> torch.Tensor:
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    wave = wave.detach().to(device=device, dtype=torch.float32)
    with torch.no_grad():
        emb = spk_eval.encode_batch(wave)
    if emb.dim() == 3:
        emb = emb.squeeze(1)
    return F.normalize(emb, p=2, dim=-1)


def _band_signature(mel: torch.Tensor):
    """Per-band scalar mean/std signature used for diagnostic distances."""
    if mel.dim() == 3:
        mel = mel.squeeze(0)
    if mel.dim() != 2:
        raise ValueError(f"Expected mel shape (80, T), got {mel.shape}")

    signature = {}
    for name, (lo, hi) in BANDS.items():
        band = mel[lo:hi, :]
        per_bin_mean = band.mean(dim=-1)
        per_bin_std = band.std(dim=-1)
        signature[name] = {
            "mean": per_bin_mean.mean().item(),
            "std": per_bin_std.mean().item(),
        }
    return signature


def _band_distance(sig_a, sig_b):
    dist = 0.0
    for band in ("low", "mid", "high"):
        dist += abs(sig_a[band]["mean"] - sig_b[band]["mean"])
        dist += abs(sig_a[band]["std"] - sig_b[band]["std"])
    return dist


def _print_cross_attention_debug(model: HubertVCModel, tag: str):
    debug = getattr(model.cross_attn, "last_debug", None)
    if not debug:
        print(f"[ATTN_DEBUG:{tag}] No attention debug payload available.")
        return

    token_usage = debug.get("token_usage") or []
    token_usage_str = ", ".join([f"{v:.4f}" for v in token_usage])

    print(
        f"[ATTN_DEBUG:{tag}] entropy={debug.get('attention_entropy'):.4f} "
        f"| attn_std={debug.get('attention_std'):.4f} "
        f"| num_tokens={debug.get('num_tokens')}"
    )
    print(f"[ATTN_DEBUG:{tag}] token_usage=[{token_usage_str}]")

    film_mean = debug.get("film_scale_mean")
    if film_mean is not None:
        print(
            f"[ATTN_DEBUG:{tag}] FiLM scale p10/p50/p90="
            f"{debug.get('film_scale_p10'):.4f}/"
            f"{debug.get('film_scale_p50'):.4f}/"
            f"{debug.get('film_scale_p90'):.4f} "
            f"(mean={debug.get('film_scale_mean'):.4f}, std={debug.get('film_scale_std'):.4f})"
        )


def _print_band_metric(tag: str, pred_mel: torch.Tensor, src_sig, tgt_sig):
    pred_sig = _band_signature(pred_mel.detach().cpu())
    d_pred_tgt = _band_distance(pred_sig, tgt_sig)
    d_pred_src = _band_distance(pred_sig, src_sig)
    metric = d_pred_tgt - d_pred_src
    verdict = "towards target" if metric < 0 else "towards source"
    print(
        f"[BAND_METRIC:{tag}] D(pred,tgt)={d_pred_tgt:.4f} "
        f"D(pred,src)={d_pred_src:.4f} metric={metric:.4f} ({verdict})"
    )


def _print_speaker_similarity(
    spk_eval,
    converted_wave: torch.Tensor,
    src_ref_wave: torch.Tensor,
    tgt_ref_wave: torch.Tensor,
    tag: str,
    device: torch.device,
    sample_rate: float,
):
    with torch.no_grad():
        emb_conv = _extract_eval_embedding(spk_eval, converted_wave, device)
        emb_src = _extract_eval_embedding(spk_eval, src_ref_wave, device)
        emb_tgt = _extract_eval_embedding(spk_eval, tgt_ref_wave, device)

        cos_tgt = F.cosine_similarity(emb_conv, emb_tgt, dim=-1).item()
        cos_src = F.cosine_similarity(emb_conv, emb_src, dim=-1).item()
        print(
            f"[SPK_SIM:{tag}] cos(conv,target)={cos_tgt:.4f} "
            f"cos(conv,source)={cos_src:.4f} "
            f"delta={cos_tgt - cos_src:.4f}"
        )

        # F0 Diagnostics
        try:
            # compute_kaldi_pitch returns [..., 2] where index 0 is pitch (Hz), index 1 is NCCF.
            conv = converted_wave.detach().cpu().float().unsqueeze(0)
            src = src_ref_wave.detach().cpu().float().unsqueeze(0)
            tgt = tgt_ref_wave.detach().cpu().float().unsqueeze(0)
            sr = float(sample_rate)
            p_conv = torchaudio.functional.compute_kaldi_pitch(conv, sr)[..., 0]
            p_src = torchaudio.functional.compute_kaldi_pitch(src, sr)[..., 0]
            p_tgt = torchaudio.functional.compute_kaldi_pitch(tgt, sr)[..., 0]

            conv_voiced = p_conv[p_conv > 0]
            src_voiced = p_src[p_src > 0]
            tgt_voiced = p_tgt[p_tgt > 0]

            f0_conv = conv_voiced.mean().item() if conv_voiced.numel() > 0 else float("nan")
            f0_src = src_voiced.mean().item() if src_voiced.numel() > 0 else float("nan")
            f0_tgt = tgt_voiced.mean().item() if tgt_voiced.numel() > 0 else float("nan")
            print(f"[PITCH:{tag}] F0(conv)={f0_conv:.1f}Hz | F0(src)={f0_src:.1f}Hz | F0(tgt)={f0_tgt:.1f}Hz")
        except Exception as e:
            print(f"[PITCH:{tag}] ERROR: {e}")


def test_generalization(checkpoint_path: str, output_dir: str):
    print("=" * 60)
    print("RUNNING 2-UTTERANCE GENERALIZATION TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    print("\n[1] Loading Vocoder...")
    vocoder = load_vocoder(None, device=str(device))
    vocoder.eval()
    spk_eval = _load_eval_speaker_encoder(device)

    utt_path_A = "/content/LibriTTS/dev-clean/2428/83705/2428_83705_000000_000001.wav"
    utt_path_B = "/content/LibriTTS/dev-clean/1988/148538/1988_148538_000002_000000.wav"
    spk_A = "2428 (man)"
    spk_B = "1988 (woman)"

    print("\n[2] Loading Specific Utterances...")
    print(f"   Utterance A: {Path(utt_path_A).name}  (speaker {spk_A})")
    print(f"   Utterance B: {Path(utt_path_B).name}  (speaker {spk_B})")

    full_audio_A = load_audio(utt_path_A, sample_rate=audio_cfg.sample_rate).to(device)
    full_audio_B = load_audio(utt_path_B, sample_rate=audio_cfg.sample_rate).to(device)

    ref_A, content_A = split_utterance_for_training(
        full_audio_A,
        ref_length_range=(1.0, 2.0),
        sample_rate=audio_cfg.sample_rate,
        min_content_length=0.5,
        deterministic=True,
    )
    ref_B, content_B = split_utterance_for_training(
        full_audio_B,
        ref_length_range=(1.0, 2.0),
        sample_rate=audio_cfg.sample_rate,
        min_content_length=0.5,
        deterministic=True,
    )

    print(f"   A: ref={ref_A.shape[0] / audio_cfg.sample_rate:.2f}s, content={content_A.shape[0] / audio_cfg.sample_rate:.2f}s")
    print(f"   B: ref={ref_B.shape[0] / audio_cfg.sample_rate:.2f}s, content={content_B.shape[0] / audio_cfg.sample_rate:.2f}s")

    torchaudio.save(
        os.path.join(output_dir, "01_raw_content_A.wav"),
        content_A.cpu().unsqueeze(0),
        audio_cfg.sample_rate,
    )
    torchaudio.save(
        os.path.join(output_dir, "02_raw_content_B.wav"),
        content_B.cpu().unsqueeze(0),
        audio_cfg.sample_rate,
    )
    print("\nSaved raw content A and B")

    print("\n[3] Testing Vocoder Ceiling...")
    gt_mel_A = extract_mel_spectrogram(content_A, sample_rate=audio_cfg.sample_rate)
    gt_mel_B = extract_mel_spectrogram(content_B, sample_rate=audio_cfg.sample_rate)
    with torch.no_grad():
        vocoded_A = vocoder(gt_mel_A.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu()
        vocoded_B = vocoder(gt_mel_B.unsqueeze(0).to(device)).squeeze(0).squeeze(0).cpu()
    torchaudio.save(os.path.join(output_dir, "03_vocoded_A.wav"), vocoded_A.unsqueeze(0), audio_cfg.sample_rate)
    torchaudio.save(os.path.join(output_dir, "04_vocoded_B.wav"), vocoded_B.unsqueeze(0), audio_cfg.sample_rate)
    print("Saved vocoded GT mels for A and B")

    ref_mel_A = extract_mel_spectrogram(ref_A, sample_rate=audio_cfg.sample_rate)
    ref_mel_B = extract_mel_spectrogram(ref_B, sample_rate=audio_cfg.sample_rate)
    src_sig_A = _band_signature(gt_mel_A.cpu())
    src_sig_B = _band_signature(gt_mel_B.cpu())
    tgt_sig_A = _band_signature(ref_mel_A.cpu())
    tgt_sig_B = _band_signature(ref_mel_B.cpu())

    print(f"\n[4] Loading Model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}. Skipping model tests.")
        return

    model = HubertVCModel(audio_cfg, model_cfg, train_cfg).to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    print("Model loaded.")
    print(f"[DEBUG] Trained alpha value: {model.cross_attn.alpha.item():.6f}")

    print("\n[SPEAKER SPACE DIAGNOSTIC]")
    with torch.no_grad():
        spk_A_tokens = model.mel_encoder(ref_A.unsqueeze(0))
        spk_B_tokens = model.mel_encoder(ref_B.unsqueeze(0))

        vec_A = spk_A_tokens.reshape(1, -1)
        vec_B = spk_B_tokens.reshape(1, -1)

        cos_sim = F.cosine_similarity(vec_A, vec_B, dim=-1)
        l2_dist = torch.norm(vec_A - vec_B, p=2)
        diff = (vec_A - vec_B).abs()
        top5 = diff.squeeze().topk(5)

        print(f"  Cosine similarity (Man vs Woman): {cos_sim.item():.4f}")
        print(f"  L2 distance       (Man vs Woman): {l2_dist.item():.4f}")
        print(f"  Top-5 most different dims values:  {[round(v, 4) for v in top5.values.tolist()]}")
        print(f"  Top-5 most different dims indices:  {top5.indices.tolist()}")
        print("  [VERDICT] ", end="")
        if cos_sim.item() > 0.90:
            print("Speaker space is COLLAPSED.")
        elif cos_sim.item() > 0.50:
            print("Partial separation. Speaker info is weak but present.")
        else:
            print("Speaker space is WELL-SEPARATED.")

    print("\n[INTERPOLATION PROBE]")
    with torch.no_grad():
        spk_A = model.mel_encoder([ref_A.to(device)])
        spk_B = model.mel_encoder([ref_B.to(device)])
        blended_spk = 0.5 * spk_A + 0.5 * spk_B

        pred_blend_A, _, _ = model(
            precomputed_speaker_feats=blended_spk,
            content_audio=[content_A.to(device)],
        )
        _print_cross_attention_debug(model, "blend_A_50_50")
        wave_blend = vocoder(pred_blend_A).squeeze(0).squeeze(0).cpu()
        torchaudio.save(
            os.path.join(output_dir, "09_blend_A_spk50.wav"),
            wave_blend.unsqueeze(0),
            audio_cfg.sample_rate,
        )
        print(f"  Blended mel mean: {pred_blend_A.mean():.4f}")
        print("Saved 09_blend_A_spk50.wav")

    with torch.no_grad():
        print("\n[5] Self-reconstruction: A content + A voice...")
        pred_mel_AA, _, _ = model(ref_A.unsqueeze(0), [content_A])
        _print_cross_attention_debug(model, "A_to_A")
        wave_AA_dev = vocoder(pred_mel_AA).squeeze(0).squeeze(0)
        wave_AA = wave_AA_dev.cpu()
        torchaudio.save(os.path.join(output_dir, "05_self_recon_A.wav"), wave_AA.unsqueeze(0), audio_cfg.sample_rate)
        print(f"   Pred mel: {pred_mel_AA.shape}, mean={pred_mel_AA.mean():.4f}")
        print("Saved A->A")

        print("\n[6] Self-reconstruction: B content + B voice...")
        pred_mel_BB, _, _ = model(ref_B.unsqueeze(0), [content_B])
        _print_cross_attention_debug(model, "B_to_B")
        wave_BB_dev = vocoder(pred_mel_BB).squeeze(0).squeeze(0)
        wave_BB = wave_BB_dev.cpu()
        torchaudio.save(os.path.join(output_dir, "06_self_recon_B.wav"), wave_BB.unsqueeze(0), audio_cfg.sample_rate)
        print(f"   Pred mel: {pred_mel_BB.shape}, mean={pred_mel_BB.mean():.4f}")
        print("Saved B->B")

        print("\n[7] Cross-conversion: A content + B voice...")
        pred_mel_AB, _, _ = model(ref_B.unsqueeze(0), [content_A])
        _print_cross_attention_debug(model, "A_to_B")
        _print_band_metric("A_to_B", pred_mel_AB, src_sig_A, tgt_sig_B)
        wave_AB_dev = vocoder(pred_mel_AB).squeeze(0).squeeze(0)
        _print_speaker_similarity(spk_eval, wave_AB_dev, ref_A.to(device), ref_B.to(device), tag="A_to_B", device=device, sample_rate=audio_cfg.sample_rate)
        wave_AB = wave_AB_dev.cpu()
        torchaudio.save(os.path.join(output_dir, "07_cross_AtoB.wav"), wave_AB.unsqueeze(0), audio_cfg.sample_rate)
        print(f"   Pred mel: {pred_mel_AB.shape}, mean={pred_mel_AB.mean():.4f}")
        print("Saved A->B")

        print("\n[8] Cross-conversion: B content + A voice...")
        pred_mel_BA, _, _ = model(ref_A.unsqueeze(0), [content_B])
        _print_cross_attention_debug(model, "B_to_A")
        _print_band_metric("B_to_A", pred_mel_BA, src_sig_B, tgt_sig_A)
        wave_BA_dev = vocoder(pred_mel_BA).squeeze(0).squeeze(0)
        _print_speaker_similarity(spk_eval, wave_BA_dev, ref_B.to(device), ref_A.to(device), tag="B_to_A", device=device, sample_rate=audio_cfg.sample_rate)
        wave_BA = wave_BA_dev.cpu()
        torchaudio.save(os.path.join(output_dir, "08_cross_BtoA.wav"), wave_BA.unsqueeze(0), audio_cfg.sample_rate)
        print(f"   Pred mel: {pred_mel_BA.shape}, mean={pred_mel_BA.mean():.4f}")
        print("Saved B->A")

    print("\n" + "=" * 60)
    print("LISTENING GUIDE:")
    print("=" * 60)
    print("  01_raw_content_A.wav   -> Ground truth Man (what was said)")
    print("  02_raw_content_B.wav   -> Ground truth Woman (what was said)")
    print("  05_self_recon_A.wav    -> Model: Man content + Man voice")
    print("  06_self_recon_B.wav    -> Model: Woman content + Woman voice")
    print("  07_cross_AtoB.wav      -> Model: Man words in Woman voice")
    print("  08_cross_BtoA.wav      -> Model: Woman words in Man voice")
    print("=" * 60)


def test_generalization_N_pairs(checkpoint_path: str, N: int):
    print("=" * 60)
    print(f"RUNNING {N}-UTTERANCE RANDOM PAIR TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()

    print("[1] Loading Dataset index...")
    dataset = VoiceConversionDataset(split="val", audio_config=audio_cfg, data_config=data_cfg, training_config=train_cfg)
    
    print(f"[2] Loading Model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found.")
        return

    model = HubertVCModel(audio_cfg, model_cfg, train_cfg).to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    
    vocoder = load_vocoder(None, device=str(device))
    vocoder.eval()
    spk_eval = _load_eval_speaker_encoder(device)

    all_sim_deltas = []
    all_band_metrics = []

    valid_utts = list(dataset.valid_utterances)
    random.shuffle(valid_utts)
    
    pairs = []
    while len(pairs) < N and len(valid_utts) >= 2:
        src_info = valid_utts.pop()
        src_spk = src_info[0]

        match_idx = None
        for i in range(len(valid_utts) - 1, -1, -1):
            if valid_utts[i][0] != src_spk:
                match_idx = i
                break

        if match_idx is None:
            continue

        tgt_info = valid_utts.pop(match_idx)
        pairs.append((src_info, tgt_info))

    if len(pairs) == 0:
        print("[WARN] Could not sample cross-speaker pairs from validation set.")
        return

    print(f"[3] Running over {len(pairs)} randomized pairs...")
    for idx, (src_info, tgt_info) in enumerate(pairs):
        src_spk, src_path = src_info
        tgt_spk, tgt_path = tgt_info
        
        full_audio_src = load_audio(src_path, sample_rate=audio_cfg.sample_rate).to(device)
        full_audio_tgt = load_audio(tgt_path, sample_rate=audio_cfg.sample_rate).to(device)

        ref_src, content_src = split_utterance_for_training(full_audio_src, (1.0, 2.0), audio_cfg.sample_rate, 0.5, True)
        ref_tgt, _ = split_utterance_for_training(full_audio_tgt, (1.0, 2.0), audio_cfg.sample_rate, 0.5, True)

        with torch.no_grad():
            pred_mel_cross, _, _ = model(ref_tgt.unsqueeze(0), [content_src])
            wave_cross = vocoder(pred_mel_cross).squeeze(0).squeeze(0)

            # Metrics
            emb_conv = _extract_eval_embedding(spk_eval, wave_cross, device)
            emb_src = _extract_eval_embedding(spk_eval, ref_src, device)
            emb_tgt = _extract_eval_embedding(spk_eval, ref_tgt, device)
            cos_tgt = F.cosine_similarity(emb_conv, emb_tgt, dim=-1).item()
            cos_src = F.cosine_similarity(emb_conv, emb_src, dim=-1).item()
            sim_delta = cos_tgt - cos_src
            all_sim_deltas.append(sim_delta)

            # Band
            gt_mel_src = extract_mel_spectrogram(content_src, audio_cfg.sample_rate)
            ref_mel_tgt = extract_mel_spectrogram(ref_tgt, audio_cfg.sample_rate)
            sig_pred = _band_signature(pred_mel_cross.cpu())
            sig_src = _band_signature(gt_mel_src.cpu())
            sig_tgt = _band_signature(ref_mel_tgt.cpu())
            d_pred_tgt = _band_distance(sig_pred, sig_tgt)
            d_pred_src = _band_distance(sig_pred, sig_src)
            band_metric = d_pred_tgt - d_pred_src
            all_band_metrics.append(band_metric)

        print(f"  [{idx+1}/{len(pairs)}] SPK_SIM delta={sim_delta:+.4f} | BAND_METRIC={band_metric:+.4f}")

    print("\n" + "=" * 40)
    print("RESULTS SUMMARY (N PAIRS CROSS):")
    print("=" * 40)
    print(f"  SPK_SIM delta: mean={np.mean(all_sim_deltas):+.4f}, std={np.std(all_sim_deltas):.4f}")
    print(f"  BAND_METRIC  : mean={np.mean(all_band_metrics):+.4f}, std={np.std(all_band_metrics):.4f}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt", help="Path to your trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="generalization_test", help="Folder to save output audio files")
    parser.add_argument("--n_pairs", type=int, default=0, help="If >0, runs N random combinations for bulk metrics.")
    args = parser.parse_args()
    
    if args.n_pairs > 0:
        test_generalization_N_pairs(args.checkpoint, args.n_pairs)
    else:
        test_generalization(args.checkpoint, args.output_dir)
