# EXP-004 — Phase-2 identity supervision (backfilled)

- date: 2026-09-26
- idea: IDEA-003
- evidence: `eval/EXP-004-phase2-id/`
- protocol: n624-v1 (40 held-out targets)

## Frozen config
| knob | value |
|---|---|
| init | `checkpoints/cf_km100.ckpt` (raw-frame codebook; no mean-norm) |
| id mode | full-length BOTH sides (`--crop 0`) + anti-source hinge |
| w_id / w_neg / neg_margin | 20 / (see run) / 0.1 |
| steps / device | 400 / cuda (per-sample backward) |

## Exact command
```powershell
.\.venv\Scripts\python.exe tools\content_factorial.py --stage id --content vq `
    --init checkpoints/cf_km100.ckpt --steps 400 --crop 0 --w_neg 20 `
    --neg_margin 0.1 --device cuda --out checkpoints/cf_km100_idP2.ckpt `
    --results eval/EXP-004-phase2-id/cf_km100_idP2_n620.json
```
