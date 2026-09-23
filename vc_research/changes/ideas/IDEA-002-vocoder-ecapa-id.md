# IDEA-002 — true frozen HiFi-GAN→ECAPA identity supervision

- date: 2026-09
- author: @director
- from: OBS-002
- status: established

## Claim
`L_id = 1 − cos(ECAPA(HiFi-GAN(G)), ECAPA(HiFi-GAN(real mel)))` raises the
margin on top of VQ, without increasing `sim_src`.

## Mechanism
The loss is computed in the exact evaluation domain, so it teaches the generator
to move converted audio toward the target's measured embedding.

## Gate
(VQ+id) − VQ paired CI excludes 0, ≥12/16 wins, `sim_src` not increased.

## Non-goals
Mel→ECAPA teachers — dead (K-001).
