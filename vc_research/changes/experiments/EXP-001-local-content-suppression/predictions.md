# EXP-001 — predictions (pre-registered)

## Hypothesis
K-means(256) quantization of HuBERT L9 content collapses source identity and
moves the zero-shot margin positive, without id supervision.

## Gates
| metric | pass condition |
|---|---|
| VQ margin | target-level CI excludes 0 |
| VQ − V0 paired | CI excludes 0, ≥ 15/16 targets |
| sim_src | drops sharply vs V0 |
| content_L1 | allowed to rise (fidelity tax acknowledged) |

## Kill condition
If VQ does not lower `sim_src` or does not move the margin vs V0 → falsified.
