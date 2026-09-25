# Evaluation Protocol (FROZEN)

> **Read-only.** This file and `eval/splits/` define the benchmark. The executor
> may not edit them. Changing the protocol requires a new `protocol` id and a
> director decision.

## Metric

```
margin = cos(ECAPA(conv), ECAPA(tgt_ref)) − cos(ECAPA(conv), ECAPA(src))
sim_tgt = cos(ECAPA(conv), ECAPA(tgt_ref))
sim_src = cos(ECAPA(conv), ECAPA(src))
content_L1 = masked mel L1 (converted mel vs source GT mel)
```

- Vocoder: SpeechBrain HiFi-GAN 16 kHz. Speaker metric: SpeechBrain ECAPA
  (`spkrec-ecapa-voxceleb`). Both frozen.
- Converted audio is vocoded before ECAPA (matches deployment).

## Protocol `n240-v1` (local)

- Split: `make_split(seed=0, num_val_spk=16)` → 24 train / 16 held-out speakers.
- 4 sources × 2 refs × 2 contents × 16 targets = **240 conversions**.
- `n_src=4`, `n_content=2`, `n_ref=2`.
- Statistics: **paired bootstrap by target speaker**, 2000–4000 resamples.
- Report margin + 95% CI + `sim_src` + `sim_tgt` + `content_L1` + wins.

## Protocol `n624-v1` (Colab scale-up)

- Split: `make_split(seed=0, num_val_spk=40)` → 207 train / 40 held-out targets.
- 624 conversions; same statistics.

## Validity requirements (mandatory)

- Per-sample `target_lengths` **and** `content_lengths` passed to the model.
- Masked mel loss. Batch-max resampling + unmasked loss is forbidden.
- `evaluate_arm` is deterministic given `(n_src, n_content, n_ref)`; only compare
  arms at identical settings.

## Result manifest (mandatory)

Every result JSON carries: `exp_id, arm, checkpoint, ckpt_sha256, config_hash,
split_id, protocol, seed, git_commit, created_utc`, plus `metrics`/`per_target`
and the full `rows`.

Split files: `eval/splits/split-0-val16.txt` (local held-out speakers).
