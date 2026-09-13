"""Materialize `results` symlinks for every experiment from its pointer.

Windows without Developer Mode / admin cannot create symlinks; in that case the
committed `results.pointer.json` remains the authoritative link and this script
prints a notice. On Linux/macOS (or Windows with privilege) it creates a
relative directory symlink next to each pointer.

  python tools/link_results.py [--check]
"""
import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO / "vc_research" / "changes" / "experiments"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report status only")
    args = ap.parse_args()

    made = skipped = failed = 0
    for pointer in sorted(EXP_ROOT.glob("EXP-*/results.pointer.json")):
        exp_dir = pointer.parent
        data = json.loads(pointer.read_text(encoding="utf-8"))
        target = (exp_dir / data["rel_from_here"]).resolve()
        link = exp_dir / "results"
        if link.exists() or link.is_symlink():
            skipped += 1
            print(f"[ok]      {exp_dir.name}/results already exists")
            continue
        if not target.exists():
            failed += 1
            print(f"[missing] {exp_dir.name}: target {data['target']} not found")
            continue
        if args.check:
            print(f"[todo]    {exp_dir.name}/results -> {data['target']}")
            continue
        rel = os.path.relpath(target, exp_dir)
        try:
            os.symlink(rel, link, target_is_directory=True)
            made += 1
            print(f"[link]    {exp_dir.name}/results -> {rel}")
        except OSError as e:
            failed += 1
            print(f"[fallback] {exp_dir.name}: symlink unavailable ({e.strerror}); "
                  f"use results.pointer.json")
    print(f"\nmade={made} already={skipped} fallback/failed={failed}")


if __name__ == "__main__":
    main()
