# kill_log.md — falsified / ditched ideas

> Owner: @director. One block per dead lever. These are **falsified with
> evidence**; do not re-run any of them. Each block: what was tried, the test,
> the result, and why it is dead. Superseded *methods* (not hypotheses) are not
> logged here — this is for scientific dead-ends.

---

## K-001 — Mel→ECAPA identity teachers (raw or vocoded targets)
**Idea:** train a differentiable `mel → ECAPA` regressor as a target-identity
loss. **Test:** two gates — held-out `cos(pred,true)` must beat the mean
embedding; teacher-space margin vs real ECAPA margin `r ≥ 0.6`. **Result:**
raw targets Δ=−0.040 vs mean, r=0.22 (0.36 train-overlap); vocoded targets
Δ=−0.063, r=0.29 (0.41 train-overlap). **Dead:** a shared mel→ECAPA map cannot
recover unseen-speaker ECAPA direction; the mean embedding predicts better.
Use true HiFi-GAN→ECAPA supervision instead. See `eval/EXP-002/`.

## K-002 — Content-bottleneck gradient-reversal adversary
**Idea:** suppress source identity via a reversal loss on the 64-d bottleneck.
**Result:** `sim_src` 0.3448 → 0.3444 (no change). **Dead.**

## K-003 — Scaling id weight / step count alone
**Idea:** more `w_id`/steps will move identity. **Result:** −0.0741 → −0.0735
(saturates; audibly inaudible steps). **Dead.**

## K-004 — Reference-frame conditioning
**Idea:** add reference-mel frame tokens as timbre K/V. **Result:** margin
+0.0030, CI crosses 0; `sim_tgt` +0.015 but `sim_src` +0.012. Improves
reconstruction, not target-vs-source identity. **Dead as an identity lever.**

## K-005 — Cross-reference residual routing
**Idea:** route the target reference's mean residual through a rank-16 adapter.
**Result:** margin +0.0075, `sim_tgt` +0.015, `sim_src` also +0.008. **Dead**;
a follow-up must show lower `sim_src` first.

## K-006 — Raw→vocoded ECAPA conditioning swap at inference
**Idea:** swap in `ECAPA(vocoder(ref mel))` at inference. **Result:** full-ref
margin −0.0073 (neutral/negative) + L1 cost; crop160 −0.0130 (significant).
**Dead.**

## K-007 — Global speaker-token gain increase
**Idea:** scale post-LayerNorm speaker tokens (1.5×, 2×). **Result:** `sim_src`
falls but `sim_tgt` falls as much or more; L1 worsens. **Dead.**

## K-008 — Zero-init per-block token router
**Idea:** 4×4 softmax route from 4 tokens to 4 blocks (16 params). **Result:**
matched 400-step no-router control shows routing Δ = −0.0011 CI [−0.0055,+0.0034];
logits stayed near-uniform. **Dead.**

## K-009 — Cross-utterance training as the content lever
**Idea:** self-recon-only training is why content leaks; switch to cross-utt.
**Result:** in-distribution same-speaker conditioning swap costs only +0.028 L1;
`sim` flat. **Dead.**

## K-010 — Source pitch leaks through HuBERT L9
**Idea:** pitch copying explains source identity. **Result:** contour
corr(pred,src) ≈ 0; drift −5..−17 st toward target. **Dead.**

## K-011 — ECAPA→token channel collapses for unseen speakers
**Idea:** held-out speakers are not represented. **Result:** val dispersion
contraction 0.456 ≈ train 0.492. **Dead.**

## K-012 — Soft / residual quantization removes the VQ fidelity tax
**Idea:** residual quantization recovers detail while keeping suppression.
**Result:** α∈{0.25,0.5} drops margin to −0.057/−0.080 and `sim_src` returns to
0.28–0.31. Suppression ≡ detail abandonment. **Dead.**

## K-013 — Band-weighted low-band L1 fixes buzz
**Idea:** weight the low band to stop over-smoothing. **Result:** jitter
0.2201 vs 0.2203 (control); L1_low unchanged. **Dead at local dose.**

---

### Process dead-ends (also do not repeat)
- **Trusting `docs/Journal.md` invariants** — derived from bug-corrupted runs.
- **4-speaker / 12-pair evaluation** — underpowered; baseline CI includes 0.
- **Any training with batch-max resampler length or unmasked mel loss.**
- **Batched full-length `L_id` on free Colab** — OOMs; keep per-sample backward.
- **`--mean_norm` with the raw-frame km100 codebooks** — hard error by design.
