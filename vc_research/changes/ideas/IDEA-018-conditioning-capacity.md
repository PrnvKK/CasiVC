# IDEA-018 — conditioning capacity is the identity bottleneck

- date: 2026-09-30
- author: @director
- from: OBS-003, kill K-004..K-008
- status: proposed

## Claim
The `192-d ECAPA → 1 Linear → 4×64-d tokens` path is the bandwidth ceiling;
a richer conditioning mechanism raises margin *without* raising `sim_src`.

## Mechanism
Current pooling of 4 tokens into one 64-D vector discards signal (mean pairwise
token cosine 0.19). A better route from the 192-D embedding to the decoder FiLM
should carry more target identity per parameter.

## Gate
Paired margin Δ CI excludes 0, `sim_src` not increased (Δ ≤ 0.003), L1 not worse
by > 0.02, params < 500k. **Must lower `sim_src` locally before any scale-up.**

## Non-goals
Global token gain, zero-init token router, reference-frame / residual routing
(all falsified — kill K-004..K-008).
