# EXP-003 — Colab scale-up ladder (backfilled)

- date: 2026-09-26
- idea: IDEA-001, IDEA-002 at scale
- evidence: `eval/EXP-003-scaleup-ladder/`
- protocol: n624-v1 (40 held-out targets)

## Frozen config
| knob | value |
|---|---|
| data | train-clean-100, 247 spk, 30 utt/spk cap, 20 s max |
| content | continuous / K-means(256, raw-frame fit) |
| id_loss | P1 crop-64 (km100_id only) |
| split_seed / num_val_spk | 0 / 40 |
| n_content / n_ref | 2 / 2 |
| device | cuda (Colab) |

## Exact commands
```powershell
.\.venv\Scripts\python.exe tools\content_factorial.py --stage train --content vq `
    --codebook 256 --epochs 20 --device cuda --out checkpoints/cf_km100.ckpt `
    --results eval/EXP-003-scaleup-ladder/cf_km100_n620.json
.\.venv\Scripts\python.exe tools\content_factorial.py --stage id --content vq `
    --init checkpoints/cf_km100.ckpt --steps 400 --crop 64 --device cuda `
    --out checkpoints/cf_km100_id.ckpt `
    --results eval/EXP-003-scaleup-ladder/cf_km100_id_n620.json
```
Note: raw-frame codebook → do **not** pass `--mean_norm` (hard error by design).
