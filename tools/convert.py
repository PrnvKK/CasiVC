"""Thin CLI: end-to-end zero-shot voice conversion.

  python tools/convert.py --source in.wav --target ref.wav --output out.wav `
      --checkpoint checkpoints/cf_km100_idP2.ckpt
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from vc.vocoder.inference import convert_voice


def main():
    ap = argparse.ArgumentParser(description="CasiVC zero-shot voice conversion")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--output", default="converted.wav")
    ap.add_argument("--checkpoint", default="checkpoints/cf_km100_idP2.ckpt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    convert_voice(
        source_audio_path=args.source,
        target_speaker_path=args.target,
        output_audio_path=args.output,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
