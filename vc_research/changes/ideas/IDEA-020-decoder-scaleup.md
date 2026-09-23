# IDEA-020 — decoder + training-budget scale-up

- date: 2026-09-30
- author: @director
- from: OBS-003, kill K-013
- status: proposed

## Claim
Scaling decoder capacity and training budget closes buzz/over-smoothing (jitter
ratio → ≥0.70) and lifts the paper margin (~+0.12 → ~+0.15), without breaking the
parameter budget.

## Mechanism
Buzz was diagnosed as decoder over-smoothing from the tiny local budget; more
data/steps already moved jitter 0.42 → 0.68.

## Gate
Listening test (human) + jitter ratio ≥ 0.70 + margin not regressed; requires a
director decision before the compute.

## Non-goals
Band-weighted low-band L1 (kill K-013).
