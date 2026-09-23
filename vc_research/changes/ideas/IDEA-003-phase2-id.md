# IDEA-003 — Phase-2 full-length id + anti-source hinge

- date: 2026-09-26
- author: @director
- from: OBS-003
- status: pass (numeric), audibly saturated

## Claim
Supervising both sides of `L_id` at full length and adding an anti-source hinge
raises `sim_tgt` (and margin) over the crop-64 recipe.

## Mechanism
Full-length targets remove the crop cap; the hinge explicitly penalizes residual
source similarity.

## Gate
idP2 − id paired CI excludes 0, `sim_src` flat, L1 cost ≤ +0.02.

## Non-goals
Do not re-tune the shape further after this; the lever is saturated (OBS-003).
