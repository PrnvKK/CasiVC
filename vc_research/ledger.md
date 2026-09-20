# ledger.md — hypothesis & evidence index

> Append-only. Every hypothesis gets an ID; never reuse an ID. The executor
> updates the **status** cell only; the director owns interpretation and the
> `kill_log.md` entries.
>
> Status: `proposed` · `running` · `pass` · `fail` · `inconclusive` ·
> `established` · `killed`
>
> Linked artifacts: `changes/observations/OBS-XXX.md`,
> `changes/ideas/IDEA-XXX.md`, `changes/experiments/EXP-XXX/`, `eval/EXP-XXX/`,
> `kill_log.md` for dead levers.

| ID | statement (falsifiable) | status | gate | evidence |
|---|---|---|---|---|
| IDEA-000 | Fixing per-sample resampler lengths + masked loss restores conditioning leverage | established | correct−zero margin CI excludes 0 | `eval/EXP-001/` |
| IDEA-001 | K-means(256) quantization of HuBERT L9 suppresses source identity and makes margin positive | established | margin CI excludes 0 without id loss | `eval/EXP-001/`, `eval/EXP-003/` |
| IDEA-002 | Frozen HiFi-GAN→ECAPA identity supervision raises margin | established | paired Δ CI excludes 0 | `eval/EXP-002/`, `eval/EXP-003/` |
| IDEA-003 | Full-length both-sides L_id + anti-source hinge raises margin | pass (audibly saturated) | paired Δ CI excludes 0 | `eval/EXP-004/` |
| IDEA-004 | Per-utterance mean-norm of content contributes to margin | fail | Δ margin CI excludes 0 | `eval/EXP-001/` |
| IDEA-005 | Mel→ECAPA teacher can provide a differentiable identity loss | killed | beaten by mean baseline; r ≥ 0.6 | `kill_log.md` K-001 |
| IDEA-006 | Gradient-reversal adversary on the 64-d bottleneck suppresses source identity | killed | `sim_src` drops | `kill_log.md` K-002 |
| IDEA-007 | More id weight/steps improve identity | killed | margin Δ CI excludes 0 | `kill_log.md` K-003 |
| IDEA-008 | Reference-frame conditioning improves target identity | fail | margin↑, `sim_src` not↑ | `kill_log.md` K-004 |
| IDEA-009 | Cross-reference residual routing improves target identity | fail | margin↑, `sim_src` not↑ | `kill_log.md` K-005 |
| IDEA-010 | Swapping raw→vocoded ECAPA at inference improves conversion | killed | margin Δ ≥ 0 | `kill_log.md` K-006 |
| IDEA-011 | Global speaker-token gain improves identity | killed | margin↑ without L1 cost | `kill_log.md` K-007 |
| IDEA-012 | Zero-init per-block token router routes timbre better | killed | matched-control Δ CI excludes 0 | `kill_log.md` K-008 |
| IDEA-013 | Cross-utterance training (vs self-recon) is the missing content lever | killed | L1 gap large / margin moves | `kill_log.md` K-009 |
| IDEA-014 | Source pitch leaks through HuBERT L9 | killed | contour corr(pred,src) high | `kill_log.md` K-010 |
| IDEA-015 | ECAPA→token channel collapses for unseen speakers | killed | held-out dispersion ≪ train | `kill_log.md` K-011 |
| IDEA-016 | Soft/residual quantization removes the VQ fidelity tax | killed | margin retained, L1 tax reduced | `kill_log.md` K-012 |
| IDEA-017 | Band-weighted low-band L1 fixes buzz/over-smoothing | killed | jitter improves | `kill_log.md` K-013 |
| IDEA-018 | Conditioning capacity is the identity bottleneck (mechanism TBD) | proposed | margin↑, `sim_src` not↑, <500k, L1 preserved | — |
| IDEA-019 | A content-preserving cross-speaker objective can separate content from timbre | proposed | `sim_tgt`↑ / `sim_src`↓ + intelligibility proxy | — |
| IDEA-020 | Decoder + training-budget scale-up closes buzz and lifts margin | proposed | listening test + jitter ratio ≥ 0.70 | — |
| IDEA-021 | Learned VQ (EMA + commitment) matches suppression with less L1 tax | proposed | suppression retained, L1 tax ↓ | — |

> Scale-up arms (cont100/km100/km100_id/idP2) are the canonical evidence for
> IDEA-001/002/003 at 247 speakers; see `eval/INDEX.md`.
