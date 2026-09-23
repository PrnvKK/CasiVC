# OBS-002 — content K-means collapses source identity

- date: 2026-09
- found_by: @director
- links: IDEA-001, EXP-001

## Observation
K-means(256) of HuBERT L9 collapses `sim_src` 0.314 → 0.153 locally and flips
the margin positive (+0.137 vs V0, 16/16 targets). Mean-norm alone does nothing.

## Evidence
`eval/EXP-001-local-content-suppression/`.

## Implication
Content suppression, not identity loss, is the primary lever. It carries a real
`content_L1` fidelity tax.
