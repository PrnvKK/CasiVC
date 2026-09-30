# EXP-003 — predictions (pre-registered)

## Hypothesis
At 247 speakers, both levers hold: K-means suppresses source identity, and
true-ECAPA identity supervision adds further margin (with a larger effect than
the local +0.015 because more train speakers).

## Gates
| metric | pass condition |
|---|---|
| km100 margin | CI excludes 0 without id loss |
| km100 − cont100 paired | CI excludes 0, ≥ 30/40 |
| km100_id − km100 paired | CI excludes 0, ≥ 30/40 |
| buzz jitter ratio | ≥ 0.70 vs GT |

## Kill condition
If K-means does not suppress at scale, or the id effect vanishes → falsified.
