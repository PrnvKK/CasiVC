# EVIDENCE_LOG.md — compressed justification of current beliefs

> Owner: @director. One entry per belief, each with the strongest supporting
> evidence path. Distilled; raw numbers live in `eval/`. If a belief changes,
> edit this file and note the superseding evidence.

## B1 — Training validity was the root cause of earlier failures
The batch-max mel resampler warp + unmasked mel loss + source-length warp
corrupted pre-fix runs; fixing them restored conditioning leverage.
Evidence: conditioning control on V0 (correct−zero +0.062, CI [+0.030,+0.099]).
Refs: `eval/EXP-001/`, `vc_research/ledger.md` IDEA-000.

## B2 — K-means content quantization collapses source identity and flips margin
K-means(256) on HuBERT L9 is the dominant lever: `sim_src` 0.31 → 0.13 locally,
0.225 → 0.126 at scale; K-means-only margin is significantly positive at scale
(+0.0643). Mean-norm contributes nothing to margin.
Refs: `eval/EXP-001/`, `eval/EXP-003/`.

## B3 — True frozen HiFi-GAN→ECAPA identity supervision works; mel→ECAPA teachers do not
`L_id = 1 − cos(ECAPA(HiFi-GAN(G)), ECAPA(HiFi-GAN(real mel)))` adds +0.015
local, +0.029 at 247 speakers. Mel→ECAPA teachers fail both acceptance gates.
Refs: `eval/EXP-002/`, `eval/EXP-003/`, `kill_log.md` K-001.

## B4 — Id-loss shaping is audibly saturated in the sim_tgt 0.22–0.25 band
Three significant +0.02-margin steps (local id, scale-up id, Phase-2) were all
inaudible on matched pairs. Phase-2 kept for numbers, not for perceived identity.
Refs: `eval/EXP-004/`, `kill_log.md` K-002.

## B5 — Remaining bottleneck is structural, not a disconnected signal path
Layerwise trace shows the ECAPA signal reaches mel; global gain, token routing,
reference frames, residual routing, and inference domain swaps all fail the
identity gate. The trade-off is content-vs-timbre in the generator/objective,
plus decoder over-smoothing under the tiny local budget.
Refs: `kill_log.md` K-003..K-009, `Research.md` §3.

## B6 — Buzz is arm-independent decoder over-smoothing
Predicted mel is smoother than GT (jitter ≈ half); quantization/codebook size and
smoothing change nothing audibly. Scale-up largely fixes it (jitter ratio
0.42 → 0.68 vs the 0.70 gate).
Refs: `probes/diag_buzz.py`, `eval/EXP-003/`.
