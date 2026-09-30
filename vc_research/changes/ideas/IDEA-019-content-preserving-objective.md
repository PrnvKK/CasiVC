# IDEA-019 — content-preserving cross-speaker objective

- date: 2026-09-30
- author: @director
- from: OBS-002, OBS-003
- status: proposed

## Claim
A cheap content teacher/loss can let the converted mel depart from source
acoustics while preserving phonetics, breaking the current content-vs-timbre
trade-off.

## Mechanism
Supervise phonetic content in a representation that is invariant to timbre, so
the generator is free to move the mel toward the target without losing words.

## Gate
`sim_tgt`↑ / `sim_src`↓ plus a non-worsening intelligibility proxy; must be
cheap enough for local iteration.

## Non-goals
Gradient-reversal adversary on the bottleneck (kill K-002).
