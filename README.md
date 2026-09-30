# CasiVC

Cross-Attentive Semantic Identity Voice Conversion — a **<500k-trainable-parameter**
zero-shot VC system (frozen HuBERT L9 content, frozen ECAPA-TDNN speaker, frozen
HiFi-GAN vocoder).

## Where to start

| If you are... | Read this |
|---|---|
| **Human** | `vc_research/CURRENT_STATE.md` |
| **@director** (SOTA, Claude) | `vc_research/Research.md` → `STATUS.md` → `EVIDENCE_LOG.md` → `ledger.md` |
| **@executor** (opencode) | `vc_research/program.md` → your brief → your `EXP-XXX/` |

Agent behavior rules: `AGENTS.md`. Role names/permissions: `vc_research/roles.yaml`.

## Layout

```
vc_research/   control plane: strategy, contract, ledger, experiment records
src/vc/        library: config, model, content, speaker, fusion, decoder,
               vocoder, losses, data, eval, experiments
tools/         thin CLIs (train / id / eval / cache / analyze / convert)
probes/        diagnostics      baselines/   trivial controls
tests/         validity + metric guards       eval/   frozen evidence (read-only)
```

Repository history and the retired architecture are archived in `_archive/` and
`docs/Journal.md`; treat those as history, not current fact.

## Quick start (local CPU)

```powershell
$py = ".\.venv\Scripts\python.exe"
# first time: make the package importable
& $py -m pip install -e .

# evaluate the current best arm
& $py tools\content_factorial.py --stage eval `
    --content vq --init checkpoints/cf_km100_idP2.ckpt `
    --n_content 2 --n_ref 2 --results eval/EXP-XXX/results.json
```

Headline result: `km100_idP2`, margin **+0.1173** [+0.0925, +0.1395],
40 held-out targets, n=624 (see `vc_research/STATUS.md`).
