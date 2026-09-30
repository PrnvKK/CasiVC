# EXP-001 — local content suppression (backfilled)

- date: 2026-09
- idea: IDEA-001 (content VQ), IDEA-000 (validity fix)
- executor: @executor
- evidence: `eval/EXP-001-local-content-suppression/`

## Frozen config
| knob | value |
|---|---|
| content | continuous / vq (mean-norm + K-means 256) |
| id_loss | 0 |
| split_seed / train_seed | 0 / {0,1,2} |
| num_val_spk | 16 |
| n_content / n_ref / n_src | 2 / 2 / 4 |
| epochs | 20 |

## Exact command (V0)
```powershell
.\.venv\Scripts\python.exe tools\train_v0.py --arms valid --epochs 20 `
    --skip_eval --out eval/EXP-001-local-content-suppression/v0_results.json
```
(VQ arms: `tools\content_factorial.py --stage train --content vq --mean_norm ...`.)
