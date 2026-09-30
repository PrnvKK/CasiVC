# cache_features_fast.py
"""
Fast feature caching (Colab GPU) — exact numerics, parallel I/O.
================================================================

Produces the SAME per-utterance .pt cache format as cache_features.py
(content_feats fp16 [T,768], speaker_feats fp32 [1,192], gt_mel fp32 [80,Tm],
keyed by utt_id).

Why it is fast (the original's real bottlenecks were CPU-side):
  - audio decode + resample + mel run in a thread pool, overlapping disk I/O
    with GPU compute (the original did everything sequentially)
  - Resample / MelSpectrogram modules are created ONCE (the original
    re-built the resample kernel and mel filterbank for every single file)
  - HuBERT runs per file on GPU under fp16 autocast, ECAPA per file on GPU.

Why HuBERT is NOT batched: the conv frontend GroupNorm normalizes over the
padded length, so batched (even attention-masked) features differ from the
per-file path (measured max|diff| ~0.9 on features with mean |x| ~0.19).
Per-file is the exact same computation as cache_features.py — verified
bitwise on mel and ECAPA, exact-by-construction on HuBERT.

Usage (Colab):
  !python cache_features_fast.py --dataset_root /content/LibriTTS \
      --split train-clean-100 --cache_dir /content/tc100_cache \
      --device cuda --max_per_speaker 40 --max_seconds 20
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import soundfile as sf
from torchaudio.transforms import Resample, MelSpectrogram
from tqdm import tqdm

import vc.data.compat
from vc.config import AudioConfig, ModelConfig
from vc.content.hubert_encoder import HuBERTEncoder, HUBERT_HIDDEN_STATE_INDEX
from vc.speaker.mel_encoder import MelEncoder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MIN_SAMPLES = 1600  # skip utterances < 0.1 s (same as cache_features.py)
N_FFT = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="/content/LibriTTS")
    ap.add_argument("--split", default="train-clean-100")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_per_speaker", type=int, default=None)
    ap.add_argument("--max_seconds", type=float, default=None,
                    help="truncate utterances longer than this (RAM/disk control; "
                         "None = full length)")
    ap.add_argument("--workers", type=int, default=8, help="audio loader threads")
    args = ap.parse_args()

    device = args.device
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    os.makedirs(args.cache_dir, exist_ok=True)
    a_cfg = AudioConfig()
    m_cfg = ModelConfig()
    sr = a_cfg.sample_rate

    target_dir = Path(args.dataset_root) / args.split
    if not target_dir.exists():
        raise FileNotFoundError(f"{target_dir} not found - download the split first")

    # ---- file selection (same policy as cache_features.py) ----
    wav_files = sorted(target_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files under {target_dir}")
    if args.max_per_speaker is not None:
        from collections import defaultdict as _dd
        by_spk = _dd(list)
        for w in wav_files:
            by_spk[w.stem.split("_")[0]].append(w)
        wav_files = [w for spk in sorted(by_spk) for w in by_spk[spk][:args.max_per_speaker]]
        print(f"[fast-cache] max_per_speaker={args.max_per_speaker}: "
              f"{len(by_spk)} speakers -> {len(wav_files)} utterances")

    # keep only not-yet-cached files (resume-safe)
    todo = [w for w in wav_files if not (Path(args.cache_dir) / f"{w.stem}.pt").exists()]
    print(f"[fast-cache] {len(todo)} to process "
          f"({len(wav_files) - len(todo)} already cached), device={device}")
    if not todo:
        return

    # ---- frozen extractors ----
    print("[fast-cache] loading HuBERT + ECAPA...")
    hubert = HuBERTEncoder(
        model_name=m_cfg.hubert_model_name,
        cache_dir=m_cfg.hubert_cache_dir,
        enable_caching=False,
    ).to(device)
    hubert.eval()
    hubert_model = hubert.hubert_model
    mel_encoder = MelEncoder(device=device)
    mel_encoder.eval()

    # ---- persistent CPU transforms (created ONCE, not per file) ----
    resamplers = {}

    def get_resample(orig_sr):
        if orig_sr not in resamplers:
            resamplers[orig_sr] = Resample(orig_freq=orig_sr, new_freq=sr)
        return resamplers[orig_sr]

    mel_transform = MelSpectrogram(
        sample_rate=sr, n_fft=N_FFT, win_length=1024, hop_length=256,
        f_min=a_cfg.mel_fmin, f_max=a_cfg.mel_fmax, n_mels=a_cfg.n_mel_bands,
        window_fn=torch.hann_window, power=1.0, normalized=True,
        center=False, pad_mode="constant", mel_scale="slaney",
    )

    def load_one(path: Path):
        """Runs in a worker thread: returns (wav16k [T], mel [80,Tm]) or None."""
        wav_np, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np).mean(dim=1)
        if orig_sr != sr:
            wav = get_resample(orig_sr)(wav)
        if args.max_seconds is not None and wav.shape[0] > args.max_seconds * sr:
            wav = wav[: int(args.max_seconds * sr)]
        if wav.shape[0] < MIN_SAMPLES:
            return None
        m = wav
        if m.shape[0] < N_FFT:
            m = torch.nn.functional.pad(m, (0, N_FFT - m.shape[0]))
        mel = torch.log(torch.clamp(mel_transform(m), min=1e-5))
        return wav, mel

    def save_one(path: Path, wav_raw: torch.Tensor, mel: torch.Tensor):
        """GPU: HuBERT (per-file, exact per-utterance path) + ECAPA, then save."""
        with torch.no_grad():
            w = wav_raw.to(device)
            w = w / w.abs().max().clamp(min=1e-8)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = hubert_model(w.unsqueeze(0), output_hidden_states=True,
                                   return_dict=True)
            feats = out.hidden_states[HUBERT_HIDDEN_STATE_INDEX][0].float()
            ec = mel_encoder._safe_encode(wav_raw.unsqueeze(0).to(device),
                                          torch.ones(1, device=device))
        torch.save({
            "utt_id": path.stem,
            "content_feats": feats.half().cpu(),
            "speaker_feats": ec.reshape(1, 192).float().cpu(),
            "gt_mel": mel.float(),
        }, Path(cache_dir := args.cache_dir) / f"{path.stem}.pt")

    cached, failed = 0, 0
    chunk = args.workers * 4
    pbar = tqdm(total=len(todo), desc="fast-cache", unit="utt")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i in range(0, len(todo), chunk):
            paths = todo[i:i + chunk]
            try:
                results = list(ex.map(load_one, paths))
            except Exception as e:
                print(f"\n[fast-cache] chunk decode failure ({e}); per-file fallback")
                results = []
                for p in paths:
                    try:
                        results.append(load_one(p))
                    except Exception:
                        results.append(None)
            for p, r in zip(paths, results):
                if r is None:
                    pbar.update(1)
                    continue
                try:
                    save_one(p, r[0], r[1])
                    cached += 1
                except Exception as e:
                    failed += 1
                    print(f"\n[fast-cache] failed {p.name}: {e}")
                pbar.update(1)
    pbar.close()
    print(f"\n[fast-cache] done: cached={cached}, failed={failed}, "
          f"cache_dir={args.cache_dir}")


if __name__ == "__main__":
    main()
