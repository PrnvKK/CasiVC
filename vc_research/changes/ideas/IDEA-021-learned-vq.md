# IDEA-021 — learned VQ (EMA + commitment)

- date: 2026-09-30
- author: @director
- from: OBS-002
- status: proposed

## Claim
A learned VQ codebook (EMA updates + commitment loss) matches fixed K-means
source suppression while reducing the `content_L1` fidelity tax.

## Mechanism
Jointly adapting the codebook to the generator's objective should spend codes on
perceptually relevant content rather than raw feature variance.

## Gate
Suppression retained (sim_src ≈ K-means level) and L1 tax reduced; run as a
matched control vs fixed K-means.

## Non-goals
Residual/soft quantization — falsified (kill K-012).
