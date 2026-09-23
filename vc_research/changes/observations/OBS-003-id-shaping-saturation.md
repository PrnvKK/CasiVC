# OBS-003 — id-loss shaping is audibly saturated

- date: 2026-09-26
- found_by: @director
- links: IDEA-003, EXP-004

## Observation
Three consecutive significant +0.02-margin changes (local id, scale-up id,
Phase-2) all landed in the sim_tgt 0.22–0.25 band and were judged inaudible on
matched pairs.

## Evidence
`eval/EXP-004-phase2-id/`; audition pairs in `samples_phase2/`.

## Implication
Stop pushing `w_id`/crop/step count. Remaining identity gap is structural
(conditioning capacity, decoder rendering).
