# Research.md

> **Owner: @director (Claude) + human.** The executor must never edit this file.
> This is the *strategic* surface: the thesis, the current best result, the open
> questions, and what counts as progress. Tactical execution lives in
> `program.md`; the full hypothesis history lives in `ledger.md` and
> `kill_log.md`.

---

## 1. Problem

Zero-shot voice conversion at a hard **<500k trainable-parameter** budget.
Frozen backbone: HuBERT-base layer 9 (content), ECAPA-TDNN (speaker), SpeechBrain
HiFi-GAN 16 kHz (vocoder). Trainable: content bottleneck, ECAPA→speaker-token
projection, position-agnostic cross-attention, temporal resampler, ConvNeXt-1D
mel decoder.

**Success = `margin = cos(ECAPA(conv), ECAPA(tgt_ref)) − cos(ECAPA(conv), ECAPA(src))`**
positive and significant, with **content fidelity preserved** and **no audible
buzz**. Report `sim_src`, `sim_tgt`, `content_L1` alongside every margin.

## 2. Current best (canonical)

`km100_idP2` — content K-means(256) quantization of HuBERT L9 + Phase-2
full-length HiFi-GAN→ECAPA identity supervision:

- margin **+0.1173** [+0.0925, +0.1395], 40 held-out targets, n=624.
- Ladder: cont100 (+0.0035) → km100 (+0.0643) → km100_id (+0.0941) → idP2 (+0.1173).
- Vocoder/ECAPA ceiling for `sim_tgt` ≈ 0.92; achieved `sim_tgt` ≈ 0.25.

## 3. Thesis / what we have established

1. **Training validity was the root cause of years of false conclusions.** The
   batch-max resampler warp + unmasked mel loss + source-length warp corrupted
   every earlier run. Fixes are mandatory (see `program.md` §Invariants).
2. **Content suppression via K-means quantization is the dominant lever** for
   source-identity collapse (sim_src 0.31 → 0.13). It is the only lever that
   flips margin positive.
3. **Metric-aligned identity supervision works** (frozen HiFi-GAN→ECAPA); the
   mel→ECAPA teachers were rejected. The id effect scales with train-speaker
   count (+0.015 local → +0.029 at 247 spk).
4. **Id-loss shaping is audibly saturated** in the sim_tgt 0.22–0.25 band: three
   significant +0.02-margin steps were all inaudible. Stop pushing loss shape.
5. **The remaining bottleneck is structural:** a content-vs-timbre trade-off and
   decoder over-smoothing (capacity/budget-bound), not a missing signal path.

## 4. Open questions (the only things worth spending SOTA tokens on)

- **Q1 — Conditioning capacity.** `192-d ECAPA → 1 Linear → 4×64-d tokens` may be
  the bandwidth ceiling. Any new mechanism must raise margin *without raising
  `sim_src`* and stay <500k params. (Reference-frame / residual / token-router
  and global-gain levers are already falsified — see `kill_log.md`.)
- **Q2 — Content-preserving cross-speaker objective.** Can we let converted mel
  depart from source acoustics while preserving phonetics? Needs a cheap
  intelligibility proxy + local gate before scale-up.
- **Q3 — Decoder rendering.** Does scaling the decoder + training budget close
  buzz/over-smoothing and lift the paper number (~+0.15)? This is a compute
  question, gated by a listening test.
- **Q4 — Learned VQ** (EMA + commitment) vs fixed K-means, if it buys fidelity
  without losing suppression.

## 5. Evidence / decision rules

- Every claim is backed by a paired, target-level bootstrap CI over **≥16
  held-out targets**. No 4-speaker / 12-pair evidence.
- Every candidate lever gets a **numeric gate written before the run**
  (`changes/experiments/EXP-XXX/predictions.md`). Fail → `kill_log.md`.
- New conditioning must lower `sim_src` locally before any scale-up.
- Audition every identity step; metric deltas in the 0.22–0.25 `sim_tgt` band do
  not count as progress.
- See `EVIDENCE_LOG.md` for the compressed justification of current beliefs.

## 6. Where things live

- Execution contract for the executor → `program.md`
- Hypothesis/evidence rows → `ledger.md`; failed ideas → `kill_log.md`
- One-pager for a fresh session → `CURRENT_STATE.md`; distilled index → `STATUS.md`
- Frozen numbers → `eval/` (immutable); experiment records → `changes/experiments/`
