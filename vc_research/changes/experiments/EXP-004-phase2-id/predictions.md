# EXP-004 — predictions (pre-registered)

## Hypothesis
Full-length both-sides `L_id` + anti-source hinge raises `sim_tgt` (and thus
margin) over the crop-64 P1 recipe, at no artifact cost.

## Gates
| metric | pass condition |
|---|---|
| idP2 − id paired | CI excludes 0, ≥ 30/40 |
| sim_src | flat (Δ ≈ 0) |
| content_L1 | cost ≤ +0.02 |
| jitter | no regression |

## Kill condition
If the paired margin CI contains 0 → the Phase-2 shaping is not worth it.

## Known caveat
Same-band audibility: prior id-shaping gains were inaudible; this must be
auditioned, and treated as numeric-only if not heard.
