# EXP-002 — predictions (pre-registered)

## Hypothesis
True frozen HiFi-GAN→ECAPA identity supervision raises the margin on top of VQ,
without increasing `sim_src`.

## Gates
| metric | pass condition |
|---|---|
| (VQ+id) − VQ paired | CI excludes 0, ≥ 12/16 targets |
| sim_src | not increased |
| sim_tgt | increased |

## Kill condition
If the paired id effect CI contains 0, or `sim_src` rises materially → falsified.
