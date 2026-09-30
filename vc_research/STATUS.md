# STATUS.md — distilled index for @director

> Machine-maintainable one-screen dashboard. Updated from `ledger.md` + the
> newest `eval/` metrics. The director reads this instead of raw logs.

## Canonical ladder (scale-up, 40 held-out targets, n=624)

| arm | content | id | margin [95% CI] | sim_src | sim_tgt | L1 | evidence |
|---|---|---|---|---:|---:|---:|---|
| cont100 | continuous | – | +0.0035 [−0.0202,+0.0264] | 0.2253 | 0.2289 | 0.8279 | `eval/EXP-003/` |
| km100 | K-means | – | +0.0643 [+0.0390,+0.0891] | 0.1258 | 0.1901 | 0.9840 | `eval/EXP-003/` |
| km100_id | K-means | P1 | +0.0941 [+0.0690,+0.1173] | 0.1341 | 0.2283 | 0.9941 | `eval/EXP-003/` |
| **km100_idP2** | K-means | P2 | **+0.1173 [+0.0925,+0.1395]** | 0.1339 | 0.2512 | 1.0009 | `eval/EXP-004/` |

## Local replication (16 held-out targets, n=240)

| arm | margin [95% CI] | evidence |
|---|---|---|
| V0 continuous | −0.0916 [−0.149,−0.026] | `eval/EXP-001/` |
| VQ (fixed codebook) | +0.0470 [−0.005,+0.100] | `eval/EXP-001/` |
| VQ+id (fixed) | +0.0650 [+0.013,+0.118] | `eval/EXP-002/` |

## Open levers (from `Research.md` §4)

| Q | lever | status | gate |
|---|---|---|---|
| Q1 | conditioning capacity | open | margin↑, `sim_src`↓, <500k, L1 preserved |
| Q2 | content-preserving cross-spk objective | open | `sim_tgt`↑ / `sim_src`↓ + intelligibility |
| Q3 | decoder + budget scale-up | open | listening test + jitter ratio ≥ 0.70 |
| Q4 | learned VQ (EMA+commit) | open | match suppression, reduce L1 tax |

## Next action

`<fill in per director decision>` — see `claude_decisions/`.
