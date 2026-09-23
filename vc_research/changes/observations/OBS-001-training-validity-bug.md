# OBS-001 — training-validity bug corrupted earlier runs

- date: 2026-09
- found_by: @director
- links: IDEA-000, EXP-001

## Observation
The mel resampler used the batch-max length and the mel loss was unmasked, so
short utterances were stretched (median 3.31×) and zero-padding dominated the
loss. The source-side content tail was also warped in.

## Evidence
`eval/EXP-001-local-content-suppression/`; conditioning control on V0:
correct−zero margin +0.062 [+0.030,+0.099].

## Implication
Every pre-fix conclusion is invalid. Fixes are mandatory and guarded by `tests/`.
