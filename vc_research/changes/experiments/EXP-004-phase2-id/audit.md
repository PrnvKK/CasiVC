# EXP-004 audit

- status: **pass (numeric); audibly saturated**
- artifacts: `eval/EXP-004-phase2-id/`

## Results (40 targets, n=624)
| arm | margin [95% CI] | sim_src | sim_tgt | L1 |
|---|---|---|---|---|
| km100_id (P1) | +0.0941 [+0.069,+0.117] | 0.1341 | 0.2283 | 0.9941 |
| **km100_idP2** | **+0.1173 [+0.092,+0.140]** | 0.1339 | 0.2512 | 1.0009 |

Paired: idP2 − id = +0.0231 [+0.016,+0.030], **34/40**. `sim_src` flat → all of
the gain is `sim_tgt`; L1 cost +0.007.

## Verdict
Numerically the best arm and now canonical. **Audition: P1 vs P2 "exactly the
same"** — no perceived identity gain. Three significant +0.02-margin id-shaping
steps all landed in the same perceptual dead zone. Keep P2 for the numbers; stop
pushing id-loss shape (kill K-003). The remaining gap is structural (Q1/Q3).
