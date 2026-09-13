r"""Experiment runner with the mandatory hypothesis gate.

Refuses to launch a run unless the target experiment folder has a pre-registered
``predictions.md`` and a linked IDEA. This makes "no gate, no run" structural.

  python tools/run_exp.py --exp vc_research/changes/experiments/EXP-005-xyz `
      --command ".\.venv\Scripts\python.exe tools\content_factorial.py --stage id ..."
"""
import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="changes/experiments/EXP-XXX dir")
    ap.add_argument("--command", required=True, help="command line to run")
    ap.add_argument("--eval_dir", default="", help="eval output dir (default eval/<EXP name>)")
    ap.add_argument("--skip_gate", action="store_true", help="director override only")
    args = ap.parse_args()

    exp_dir = Path(args.exp)
    if not exp_dir.is_absolute():
        exp_dir = REPO / exp_dir
    if not exp_dir.exists():
        sys.exit(f"[run_exp] experiment dir not found: {exp_dir}")

    if not args.skip_gate:
        preds = exp_dir / "predictions.md"
        if not preds.exists():
            sys.exit(f"[run_exp] GATE: {preds} missing. Pre-register predictions first.")
        text = preds.read_text(encoding="utf-8").strip()
        if len(text) < 40:
            sys.exit(f"[run_exp] GATE: {preds} is empty/placeholder.")
        if "IDEA-" not in text:
            sys.exit(f"[run_exp] GATE: {preds} does not link an IDEA-XXX.")
        print(f"[run_exp] gate passed (predictions + idea linked)")

    eval_dir = Path(args.eval_dir) if args.eval_dir else REPO / "eval" / exp_dir.name
    logs = eval_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log = logs / f"{ts}.log"

    print(f"[run_exp] exp={exp_dir.name}")
    print(f"[run_exp] command={args.command}")
    print(f"[run_exp] log -> {log}")
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(args.command, shell=True, cwd=str(REPO),
                              stdout=f, stderr=subprocess.STDOUT)
    print(f"[run_exp] exit={proc.returncode}")
    print("[run_exp] next: write audit.md and update ledger.md status.")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
