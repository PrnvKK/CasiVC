# EXP-001 audit

- status: **pass** (established)
- idea: IDEA-001
- artifacts: `eval/EXP-001-local-content-suppression/`

## Results (seed 0, n=240)
| arm | margin [95% CI] | sim_src | sim_tgt | content_L1 |
|---|---|---|---|---|
| V0 continuous | −0.0916 [−0.149,−0.026] | 0.3140 | 0.2224 | 0.8454 |
| K-means only | +0.0516 [+0.002,+0.103] | 0.1515 | — | 1.0234 |
| VQ (corrected) | +0.0470 [−0.005,+0.100] | 0.1530 | 0.1999 | 1.0126 |

Paired: K-means-only − V0 = +0.1373 [+0.111,+0.169], **16/16**.

## Verdict
Content K-means is the dominant source-suppression lever; margin flips positive.
Mean-norm contributes nothing to margin (attribution resolved). Fidelity tax
(`content_L1`) is real and structural (see kill K-012).
