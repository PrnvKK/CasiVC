# EXP-002 audit

- status: **pass** (established)
- artifacts: `eval/EXP-002-local-identity-supervision/`

## Results (seed 0, n=240)
| arm | margin [95% CI] | sim_src | sim_tgt |
|---|---|---|---|
| VQ | +0.0470 [−0.005,+0.100] | 0.1530 | 0.1999 |
| **VQ+id** | **+0.0650 [+0.013,+0.118]** | 0.1593 | 0.2243 |

Paired: VQ+id − VQ = +0.0182 [+0.0095,+0.0267], **14/16**. Replicated 3/3 seeds
(target-level CI excludes 0 each seed). Crop-length control: 400 extra steps
alone contribute −0.0002 (not significant).

## Verdict
The identity effect is real and attributable to `L_id`, not extra steps. The
mel→ECAPA teacher alternative is dead (kill K-001).
