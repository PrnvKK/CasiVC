# EXP-002 — local identity supervision (backfilled)

- date: 2026-09
- idea: IDEA-002
- evidence: `eval/EXP-002-local-identity-supervision/`

## Frozen config
| knob | value |
|---|---|
| content | vq (mean-norm + K-means 256) |
| id_loss | true HiFi-GAN→ECAPA, crop 64 |
| split_seed / train_seed | 0 / {0,1,2} |
| num_val_spk | 16 |
| steps / id_lr / w_id | 400 / 5e-5 / 20 |
| n_content / n_ref | 2 / 2 |

## Exact command
```powershell
.\.venv\Scripts\python.exe tools\content_factorial.py --stage id --content vq --mean_norm `
    --init checkpoints/cf_vq_fixed.ckpt --steps 200 --n_content 2 --n_ref 2 `
    --out checkpoints/cf_vq_fixed_id.ckpt `
    --results eval/EXP-002-local-identity-supervision/cf_vq_fixed_id_n240.json
```
