"""Thin CLI: V0 objective-identifiability trainer (repaired validity arm)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from vc.experiments.train_v0 import main

if __name__ == "__main__":
    main()
