# EXP-003 audit

- status: **pass**
- artifacts: `eval/EXP-003-scaleup-ladder/`

## Results (40 targets, n=624)
| arm | margin [95% CI] | sim_src | sim_tgt | L1 |
|---|---|---|---|---|
| cont100 | +0.0035 [−0.020,+0.026] | 0.2253 | 0.2289 | 0.8279 |
| km100 | +0.0643 [+0.039,+0.089] | 0.1258 | 0.1901 | 0.9840 |
| km100_id | +0.0941 [+0.069,+0.117] | 0.1341 | 0.2283 | 0.9941 |

Paired: km100 − cont100 = +0.0616 [+0.044,+0.079], **36/40**;
km100_id − km100 = +0.0293 [+0.021,+0.038], **35/40**.
Buzz jitter ratio 0.68 vs the 0.70 gate (near pass).

## Verdict
All gates pass. The id effect doubled at scale. `sim_tgt` still ~0.23 vs the
0.92 vocoder ceiling → Phase-2 is the next lever (EXP-004).
