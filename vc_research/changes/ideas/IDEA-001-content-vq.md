# IDEA-001 — K-means content quantization suppresses source identity

- date: 2026-09
- author: @director
- from: OBS-002
- status: established

## Claim
Quantizing HuBERT L9 frames to a fixed K-means codebook removes source timbre
detail, collapses `sim_src`, and moves the zero-shot margin positive.

## Mechanism
The codebook is source-agnostic; quantization discards the speaker-specific
residual that leaks source identity through the content path.

## Gate
VQ margin target-level CI excludes 0; VQ − V0 CI excludes 0 with ≥15/16 wins.

## Non-goals
Not a fidelity improvement — a deliberate detail/fidelity tax (see kill K-012).
