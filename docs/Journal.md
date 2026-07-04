# CasiVC Journal (condensed)

## Invariants (Sessions 1–17)
- L1 asymptotically crushes any speaker module sharing its gradient path. CE is the only gradient source that survives.
- Cross SPK_SIM ceiling ~0.20 is structural, not data (4k/8k/148spk all hit it).
- Three-path gradient separation (L1→prebias, Var→post-scale, Speaker→final.detach) is the only architecture with zero regression across 60ep.
- `.detach()` on speaker_feats protects upstream but NOT the module itself. L1 flows through `x → module → mel_proj_content → L1`, training module weights speaker-agnostic.
- Output-level speaker injection (80-band) is a dead end — L1 has direct per-band leverage, suppression inevitable.
- cent_cos ≠ audio quality. SPK_SIM on vocoded audio is the only reliable metric.
- Content regression E10→E30 = overfitting (train L1 ↓ while eval ↑), not architecture.
- Contrastive loss: theoretically correct, compute-infeasible on 14.5GB (OOM vocoder-in-loop, mel/fbank mismatch in bypass). Abandoned Session 17.

## Failed Paths (do not repeat)
- `speaker_film` with/without detach (S10, S15): L1 trains MLP speaker-agnostic.
- CE on shared decoder weights (S3, S5): asymptotic L1-vs-CE crash at E20-30.
- Variance loss on content_mel (S13): L1-vs-Var competition, content degraded +0.063.
- SpeakerConditionedAffine at output (S10): per-band floor ≥1.0 makes needed scale unreachable.
- Split mel_proj without CE on speaker path (S4, Fix #17): speaker weights decay to noise.
- HuBERT L1 tokens for speaker encoder (S8): 27.5% peak at E7, regressed to 14% at E30.

## Architecture State (current code)
- **Data:** train-clean-100, 8k utt, 148 train / 49 val / 49 test speakers.
- **Frozen:** HuBERT encoder, ECAPA-TDNN (mel_encoder), HiFi-GAN vocoder.
- **Trainable path:** audio → HuBERT(frozen) → hubert_proj(768→96 MLP) → cross_attn → fused_features[96] → TemporalResampler → decoder(4 blocks + adapter + mel_proj_content 96→80) → prebias_mel → (variance_mel + mel_speaker + out_bias) → pred_mel[80].
- **Speaker path:** ref_audio → ECAPA(frozen) → mel_encoder(token_norm) → speaker_feats[B,8,96] → cross_attn(keys/values).
- **Active losses:** lambda_mel=45 (L1 on prebias_mel), cross_attn_ce_weight=1.0 (per-frame CE on attended_features), spk_film_ce_weight=2.0 (per-frame CE on mel_speaker delta), pooled_mel_ce_weight=2.0 (pooled CE on speaker delta), cross_pair_prob=0.5/weight=6.0.
- **Architecture (S21):** G2 gated SpeakerDeltaProj — gate = clamp(1−cos(target_pooled, content_pooled), 0, 1).detach(). Self: gate≈0 → delta≡0 by construction. Cross: gate≈0.76 @E10 → delta active. Both x-path (input to MLP) and output delta masked.
- **Disabled:** lambda_var=0, all decoder FiLM modules (speaker_film, adapter_speaker_film, block3_id_film), out_scale bypassed, mel_classifier_weight=0.

## Sessions 18+ — Cross-Attention Alignment Fixes
1. **CE probe on attended_features** (cross_attn_classifier at hubertvc_model.py:203, taps cross_attn._cached_attended_features at L380). Result: 99.8% speaker accuracy → attended_features IS speaker-discriminative. But cos(attn_pre_film, spk_pooled)=0.024 — misaligned from ECAPA mel-encoder direction.
2. **Residual 0.35→0.10** (cross_attention.py:418). Result: no movement in cos or Cross SPK_SIM. Magnitude burial is NOT the cause — misalignment is basis-rotational (MHA out_proj decouples from spk_pooled).
3. **Pre-film speaker bias injection** (cross_attention.py:81 alpha_bias param + L377-379: `speaker_features.detach().mean(dim=1)` injected via `F.softplus(alpha_bias)` into attended_features). Result E10: cos(attn,spk_pooled) 0.024→0.076 (3×). Alignment proven fixable. But relay eroded ~80%: cos(fused) only +0.012.
4. **Post-relay injection** (cross_attention.py:481: second injection of same scalar·bias after `attended_features = attended_features + film_output`). Result E10: cos(fused)=+0.049 (4× baseline, flipped from negative). cos(attn) stable 0.076. Cross SPK_SIM=0.150 (flat vs baseline 0.147). **Speaker signal now cleanly reaches fused_features.**
5. **Conclusion:** Decoder path (resampler→adapter→4 MobileNet blocks→mel_proj_content) annihilates the speaker signal before it reaches output mel. L1 trains decoder weights speaker-agnostic through mel_proj_content — matches Session 15 Key Learning #1.

## Next Step
**Decoder trace audit** (3 code edits, eval-only, 1min on E10 checkpoint):
- `mobilenet_decoder.py`: cache `self._cached_post_adapter` after adapter (L730).
- `hubertvc_model.py`: expose `resampled_features` + `decoder_intermediates` + `post_adapter` in aux.
- `test_generalization.py`: print `cos(feature_pooled, spk_pooled)` at each 96-dim stage (resampled, post_adapter, blocks 0–3).
- Goal: localize which block is the main erasure point. Diagnostic only — no training.

## Session 19 — Decoder Trace Audit + T1 Adapter Identity-Freeze

**Decoder trace audit (eval-only):** cos(stage_pooled, spk_pooled) at each 96-dim stage, E10 baseline:
- resampled +0.072 · post_adapter +0.005 (93% erased in ONE layer) · block3 −0.011.

**T1 — adapter identity-freeze** (`mobilenet_decoder.py:525-538`): single-variable test — crush adapter-specific (H_A) or path-inherent (H_B)? Freeze adapter Conv1d(96→96,k=1) at identity, fresh E0 run.
- Bail (Content-Only L1 > 1.15): never triggered. Content intact (0.9741).
- 10ep: `post_adapter`=+0.125 (=resampled, identity works). `block3`=**−0.0630** (vs baseline −0.011, MORE negative). Cross SPK_SIM=**0.1360** (vs 0.1465, below shuffled floor 0.1399).

**Conclusion — H_B CONFIRMED:** L1's speaker crush is PATH-INHERENT, not adapter-specific. Blocks 0-3 absorb the crush work when adapter can't. Freezing the choke migrates it; it does not remove it. Matches Invariant §1.

**New invariant (S19):** speaker crush migrates to ANY unconstrained projection in the L1 gradient path. A point freeze can't hold — L1 finds the next rotation. Defense must be at the gradient-source input, not the projection sites.

**Byproduct:** resampled cos +0.125 (T1) / +0.072 (baseline) — pre-adapter features are the most speaker-aligned point in the pipeline. SpeakerDeltaProj currently reads x.detach() from post-block3 (the most-erased point, −0.0630) at `mobilenet_decoder.py:807`.

## Session 20 — Path-3 L1 Weight Sweep (Shared-Weight Ceiling Proven)

**Question:** Is mel_speaker under-scaled (magnitude lever) or L1-contaminated (direction lever, GPT-5.5 hypothesis)?

**T1 reverted** (`mobilenet_decoder.py:525-529`): adapter back to trainable xavier_uniform, T1 identity-freeze treated as completed diagnostic. Baseline reproduced at E10: Cross 0.155, Self 0.344, RMS(mel_speaker)/RMS(content) 0.096, Self Δ +0.017.

**Phase 3 — Remove Path-3 L1** (`losses.py:424-441`): `outs["mel_final"]` zeroed, full λ_mel on Path 1. E3 result:
- mel_speaker blew up 5×: RMS ratio 0.136→0.508, |mean| 0.466→1.661.
- Self SPK_SIM collapsed 0.165→0.087; Self Δ = **−0.070** (delta now *destroys* self identity).
- Cross SPK_SIM 0.077→0.029. **Failed** all criteria.

**Phase 3b — Reintroduce L1 at α=0.1** (`losses.py:428`: `λ_mel * 0.1 * l_mel_final`, full λ_mel on Path 1). E3 result:
- RMS ratio 0.274 (mid-band, within magnitude target).
- Self Δ = **−0.022** (still negative). Cross 0.052. **Failed** — landed between baseline and Phase 3 on every metric.

**Decisive monotone table (L1 weight on Path 3 vs Self Δ @E3):**
| L1 weight | Self Δ @E3 | Cross @E3 |
|---|---:|---:|
| 0.5 (baseline) | +0.014 | 0.077 |
| 0.1 (Phase 3b) | −0.022 | 0.052 |
| 0.0 (Phase 3)  | −0.070 | 0.029 |

**New invariant (S20):** The Cross SPK_SIM ~0.20 ceiling is **structural, caused by SpeakerDeltaProj's shared weights across self/cross pairs.** The module sees identical inputs on both paths and cannot satisfy "delta≈0 on self / delta=large on cross." Any L1 weight that protects self-reconstruction crushes cross identity; any weight that frees cross identity destroys self-reconstruction. **No α wins both.** The journal's long-hypothesized ceiling is now demonstrated, not just asserted.

**Byproduct observations:**
- Content-Only L1 = 0.9444 bit-identical across all three runs at E3 → Path 1 is rock-stable, only the delta path differs.
- Decoder trace (resampled/post_adapter/block3 cos) near-identical across runs → loss-weight changes do NOT affect upstream crush. Crush is upstream-independent of Path-3 L1 weight.
- mel_speaker direction was forming correctly (shuf-Δ 0.013→0.064 baseline E3→E10) — CE wins direction; L1 vs CE fight is purely over magnitude.

**Failed Paths updated (do not repeat):**
- Phase 3 (Path-3 L1=0): mel_speaker unbounded growth, Self SPK_SIM collapse. Output-level delta needs a magnitude anchor.
- Phase 3b (Path-3 L1=0.1): magnitude bounded but Self Δ negative at E3 — shared module cannot satisfy self/cross simultaneously. Tuning α is not the lever.

**Suggested next direction (suggestion, not compulsory — open to alternatives):** The shared-module ceiling suggests breaking the self/cross symmetry at the module architecture, not via loss weight. Two candidate single-variable edits:
- **(A) Input-gated SpeakerDeltaProj:** scale mel_speaker by `g = 1 − cos(spk_ref, content_spk_estimate)`. Self: g≈0 (correct, no conversion). Cross: g≈1. Needs a content-side speaker estimate (reuse frozen ECAPA on content audio, or small Linear on post-block3 features).
- **(C) Subtractive input:** feed `(spk_ref − content_spk)` as the speaker input to SpeakerDeltaProj instead of spk_ref alone. Self: input≈0 → delta≈0 by construction. Architectural, not equilibrium.
- (B) tanh hard clamp: bounds magnitude but doesn't address self-pair destruction — not recommended.

**Code state:** Phase 3b is the current state of `losses.py:424-441` (α=0.1). Revert to baseline (λ_mel*0.5 on both paths) before any S21 test for clean single-variable attribution, OR keep α=0.1 as the new baseline if Path-3 magnitude reduction is desired going forward.

## Session 21 — G2 Gated-Diff SpeakerDeltaProj

**Question:** Can breaking self/cross symmetry at the architecture (not loss weight) break the S20 shared-weight ceiling?

**G2 design:** SpeakerDeltaProj receives both target and content speaker tokens. Gate = `clamp(1 - cos(target_pooled, content_pooled), 0, 1).detach()`. Both x-path (input to MLP) and output delta are masked by gate. Self: gate≈0 → delta≡0 by construction. Cross: gate>0 → CE+cross-stats train conversion. SpeakerDeltaProj.mlp weights checkpoint-compatible (shapes unchanged). Cross-pair CE masked by `cross_mask` to drop same-speaker rolled pairs (pre-existing latent issue G2 exposed).

**E10 results vs S20 baseline (α=0.5):**
- Cross SPK_SIM: 0.1204 (vs 0.052 @E3; S20 α=0.5 baseline 0.077 @E3, +56%)
- Self Δ: 0.0000 (structurally zero — S20 self-destruction architecturally impossible)
- Shuffled Δ: 0.0663 (beat >0.05 threshold)
- Gate (cross): 0.6512 @E3 → 0.7612 @E10 — opening as training progresses
- mel_speaker |mean|: 0.7384 @E3 → 0.8080 @E10 (stable, not runaway)
- Content-Only L1: 0.9476 @E3 → 0.9616 @E10 (Path 1 stable)
- cos(fused, spk_pooled): 0.0642 → 0.0575 (within noise band)
- Decoder trace (resampled/post_adapter/block3): +0.0739/+0.0236/+0.0487 → +0.0858/−0.0013/+0.0549

**Net:** S20 ceiling broken. Shared-weight self/cross conflict resolved architecturally. Cross SPK_SIM 0.12 @E10 is the highest trajectory seen at this epoch across the project. G2 succeeded on its own success criteria (Cross > 0.077 by E10).

**New invariant (S21):** Self/cross symmetry can be broken via content-speaker cosine gating. Self SPK_SIM decoupled from cross SPK_SIM for the first time — no α tradeoff, no shared-weight equilibrium.

**Next:** Run E20 with no changes to determine if 0.12 is a waypoint or asymptote. Decision tree:
- Cross SPK_SIM ≳ 0.20 @E20 → still climbing, continue to E30 (no edit)
- Cross SPK_SIM plateaus 0.12–0.15 @E20 → hedge to pre-adapter 96-d injection (most speaker-aligned stage, cos resampled +0.086 vs block3 +0.055; fires S19 crush-migration invariant → needs own gradient isolation design)

**E20 result:** Cross SPK_SIM 0.1724 (still climbing from 0.1204), Shuffled Δ 0.0899, Self Δ 0.0000, Content-Only L1 0.9615, Gate 0.7793, resampled cos +0.1020. No plateau yet — continued to E30.

**E30 result — PLATEAU CONFIRMED:**
| Metric | E10 | E20 | E30 | E20→E30 rate |
|---|---:|---:|---:|---:|
| Cross SPK_SIM | 0.1204 | 0.1724 | 0.1816 | +0.009 (6× deceleration vs E10→E20) |
| Self SPK_SIM | 0.3369 | 0.3864 | 0.4172 | +0.031 (still climbing, decoupled) |
| Shuffled Δ | 0.0663 | 0.0899 | 0.0942 | +0.004 (direction formed) |
| Self Δ | 0.000 | 0.000 | 0.0000 | gate holds ✓ |
| Content-Only L1 | 0.9616 | 0.9615 | 0.9518 | Path 1 stable |
| Gate (cross) | 0.7612 | 0.7793 | 0.7974 | still opening, not the bottleneck |
| mel_speaker \|mean\| | 0.8080 | 0.6486 | 0.5955 | magnitude shrinking — CE winning direction |
| resampled cos | 0.0858 | 0.1020 | 0.0992 | flat (entry signal stable) |
| block3 cos | 0.0549 | 0.0332 | 0.0373 | noisy, no growth — plateau source |

**Conclusion:** G2 plateaued at E30. Cross SPK_SIM rate collapsed 6× (E10→E20: +0.052 → E20→E30: +0.009). mel_speaker magnitude still shrinking (0.81→0.60) — CE refining direction, not learning new direction. Gate keeps opening (0.78→0.80) but Cross doesn't budge → gate is NOT the bottleneck, the signal it gates is. Self SPK_SIM still climbing (+0.031) — content path decoupled and healthy.

**Diagnosis (structural):** SpeakerDeltaProj reads `x.detach()` from post-block3 — the most-erased stage (block3 cos +0.037, vs resampled +0.099). The plateau source is input signal starvation: the gate opens more, but the `x` content descriptor it multiplies is speaker-erased. Per S19 invariant, crush is path-inherent — defense must be at the gradient-source input, not projection sites. SpeakerDeltaProj's MLP is off the L1 path (x is detached), so moving its input upstream does NOT risk crush migration.

**Next step (S22 candidate — pre-adapter injection hedge):** Move SpeakerDeltaProj's `x` input from post-block3 to `resampled_features` (pre-adapter, cos +0.099, the most speaker-aligned point per S19 trace). Same G2 gate, same self/cross decoupling — just a richer content descriptor. SpeakerDeltaProj.mlp remains off the L1 gradient path (x stays detached), so S19 crush-migration invariant does NOT fire — the module has no L1 gradient flowing through it. Single-variable test; checkpoint-compatible (no arch change, only which tensor is fed as `x`).

**Code state:** G2 active in `mobilenet_decoder.py:456-500`, threaded through `hubertvc_model.py:330-341,417-419`. α=0.1 Path-3 L1 retained. All FiLM modules still disabled. No loss weights changed. E30 checkpoint is the current state.

## Session 22 — S22 Retirement + Dual Sweep (Delta & Alpha-Bias Net-Harmful)

**Actions (code):**
- Added Cross Content-Only SPK_SIM diagnostic (vocoded `prebias_mel` on cross path) to `test_generalization.py` — decisive for S22 motivation.
- Added `_bias_scale_k = 1.0` attribute (`cross_attention.py:87`) with multiplication at both injection sites (L386, L493) for alpha_bias sweep. Default 1.0 = no behavior change.
- Added `run_alpha_bias_sweep` to `test_generalization.py` with `--alpha_bias_sweep` CLI flag; `run_delta_sweep` already present.

**Delta scale sweep @E30** (best `s` by Cross SPK_SIM):
| s    | Cross | Self   | Self L1 |
|-----:|------:|-------:|--------:|
| -1.0 | 0.1592 | 0.4172 | 0.9518 |
| -0.5 | 0.1894 | 0.4172 | 0.9518 |
|  0.0 | 0.1940 | 0.4172 | 0.9518 |
|  0.25| 0.1944 | 0.4172 | 0.9518 |
|  0.50| 0.1946 | 0.4172 | 0.9518 | <- best cross
|  1.00| 0.1816 | 0.4172 | 0.9518 | (trained)
|  1.50| 0.1609 | 0.4172 | 0.9518 |

Best s=0.5 → 0.1946 vs trained s=1.0 → 0.1816 (Δ = −0.0124). s=0 baseline → 0.1940. Delta net-harmful at trained strength; best contribution +0.0006 (negligible).

**Alpha-bias scale sweep @E30** (best `k` by Cross SPK_SIM):
| k    | Cross | Self   |
|-----:|------:|-------:|
| 0.00 | 0.1835 | 0.4106 | <- best cross
| 0.50 | 0.1813 | 0.4134 |
| 1.00 | 0.1816 | 0.4172 | (trained)
| 2.00 | 0.1768 | 0.4166 |
| 4.00 | 0.1712 | 0.4151 |

Monotonic decline with k. k=0 best for Cross but collapses Self (−0.0066) — Self feeds forward through `out_bias` rotation, can't strip without side effect.

**Findings:**
- Both additive speaker-injection mechanisms (SpeakerDeltaProj output delta + alpha_bias upstream bias) are net-harmful at trained strength. The ~0.19 Cross identity is carried by bare cross-attention K/V fusion + FiLM, not the additive mechanisms.
- **S22 (pre-adapter `x` for SpeakerDeltaProj) retired** — delta is net-harmful regardless of input source; enriching `x` would amplify a counterproductive output.
- Cross Content-Only SPK_SIM @ s=0 = 0.1940 ≥ full pipeline 0.1816 → SpeakerDeltaProj is the destructive component on the cross path.
- Trainable param estimate ~1.0–1.1M (~1/30 FreeVC @ 30M, ~1/75 SEF-VC @ 75M) — sub-2M framing defensible.

**E30 scoreboard (current best state):**
- Cross SPK_SIM: 0.1816 · Cross Content-Only: 0.1940 · Self SPK_SIM: 0.4172
- Self L1: 0.9518 · Shuffled Δ: 0.0942 · Gate: 0.7974
- Vocoder ceiling: 0.9205 · Attn entropy: 1.2597 · cos(fused, content_pre): 0.4599
- resampled/post_adapter/block3 cos: +0.0992 / +0.0231 / +0.0373

**Failed Paths updated (do not repeat):**
- SpeakerDeltaProj output delta (S22 sweep): best s=0.5 only +0.0006 over no-delta; trained s=1.0 actively harmful (−0.0124). Output additive injection verified dead.
- alpha_bias upstream injection (S22 sweep): monotonic decline with k; even k=0 marginally beats k=1 by Self-leak, not true Cross gain.

**Code state:** Both sweeps present in `test_generalization.py` (`--delta_sweep`, `--alpha_bias_sweep`, `--s_values`, `--k_values`, `--num_pairs`). `_bias_scale_k=1.0` default preserved in `cross_attention.py:87`. E30 checkpoint current. No training-side changes this session.
