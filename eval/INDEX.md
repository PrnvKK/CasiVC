# eval/INDEX.md — frozen evidence index

> **Read-only.** Maps every result artifact to its experiment, arm, checkpoint,
> and split. `rows` are retained in all files.

## EXP-001 — local content suppression (protocol n240-v1, split 0/val16)

| file | arm | checkpoint |
|---|---|---|
| `EXP-001-local-content-suppression/v0_full_eval.json` | V0 continuous | `checkpoints/ident_valid.ckpt` |
| `EXP-001-local-content-suppression/cf_cont_s1_n240.json` | continuous seed 1 | `checkpoints/cf_cont_s1.ckpt` |
| `EXP-001-local-content-suppression/cf_cont_s2_n240.json` | continuous seed 2 | `checkpoints/cf_cont_s2.ckpt` |
| `EXP-001-local-content-suppression/cf_km_n240.json` | K-means only | `checkpoints/cf_km.ckpt` |
| `EXP-001-local-content-suppression/cf_km_id_n240.json` | K-means + id | `checkpoints/cf_km_id.ckpt` |
| `EXP-001-local-content-suppression/cf_vq_fixed_n240.json` | VQ (corrected) | `checkpoints/cf_vq_fixed.ckpt` |
| `EXP-001-local-content-suppression/cf_vq_fixed_s1_n240.json` | VQ seed 1 | `checkpoints/cf_vq_fixed_s1.ckpt` |
| `EXP-001-local-content-suppression/cf_vq_fixed_s2_n240.json` | VQ seed 2 | `checkpoints/cf_vq_fixed_s2.ckpt` |

## EXP-002 — local identity supervision (protocol n240-v1)

| file | arm | checkpoint |
|---|---|---|
| `EXP-002-local-identity-supervision/cf_vq_fixed_id_n240.json` | VQ + id | `checkpoints/cf_vq_fixed_id.ckpt` |
| `EXP-002-local-identity-supervision/cf_vq_fixed_id_s1_n240.json` | VQ + id seed 1 | `checkpoints/cf_vq_fixed_id_s1.ckpt` |
| `EXP-002-local-identity-supervision/cf_vq_fixed_id_s2_n240.json` | VQ + id seed 2 | `checkpoints/cf_vq_fixed_id_s2.ckpt` |
| `EXP-002-local-identity-supervision/cf_vq_fixed_id_eval_rows.json` | VQ + id (rows) | `checkpoints/cf_vq_fixed_id.ckpt` |
| `EXP-002-local-identity-supervision/v1_true_ecapa_results.json` | V0/V1 true-ECAPA | `checkpoints/ident_v1.ckpt` |

## EXP-003 — Colab scale-up ladder (protocol n624-v1, split 0/val40)

| file | arm | checkpoint |
|---|---|---|
| `EXP-003-scaleup-ladder/cf_cont100_n620.json` | continuous | `checkpoints/cf_cont100.ckpt` |
| `EXP-003-scaleup-ladder/cf_km100_n620.json` | K-means | `checkpoints/cf_km100.ckpt` |
| `EXP-003-scaleup-ladder/cf_km100_id_n620.json` | K-means + id (P1) | `checkpoints/cf_km100_id.ckpt` |

## EXP-004 — Phase-2 identity (protocol n624-v1) — CURRENT BEST

| file | arm | checkpoint |
|---|---|---|
| `EXP-004-phase2-id/cf_km100_idP2_n620.json` | K-means + id P2 | `checkpoints/cf_km100_idP2.ckpt` |

## Adding evidence

Evidence is append-only. New results go in a **new** `eval/EXP-XXX/` directory
created by `tools/run_exp.py`; never overwrite an existing file. A pointer
(`results.pointer.json`) lives in the matching `changes/experiments/EXP-XXX/`.
