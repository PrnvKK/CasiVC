# EXP-XXX — predictions (PRE-REGISTERED)

> Written and hashed BEFORE the run. If any of this changes, create a new EXP id.

## Hypothesis under test
<IDEA-XXX restated>

## Baselines / controls
- baseline arm: <name + evidence path>
- matched control: <same steps/lr, only the lever differs>

## Numeric predictions (gates)
| metric | predicted | pass condition |
|---|---|---|
| paired margin Δ vs baseline | | CI excludes 0, ≥ +0.010 |
| sim_src | | not increased (Δ ≤ 0.003) |
| sim_tgt | | increased |
| content_L1 | | not worse by > 0.02 |

## Kill condition
If <X>, the idea is falsified → `kill_log.md`.

## Anti-confounds
- identical eval pairs / seeds / steps as baseline?
- any bundled changes (mean-norm etc.) that would confound attribution?
