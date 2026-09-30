"""Thin CLI: content factorial train / id / eval.

Usage:
  python tools/content_factorial.py --stage eval --content vq `
      --init checkpoints/cf_km100_idP2.ckpt --n_content 2 --n_ref 2 `
      --results eval/EXP-XXX/results.json
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from vc.experiments.content_factorial import main

if __name__ == "__main__":
    main()
