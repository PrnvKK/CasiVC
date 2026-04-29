# CasiVC Workflow & Optimization Protocol

## 1. Environment Architecture
- **Local Repository**: This workspace is the source of truth for code. All modifications are made here and then synced by the USER to Google Colab.
- **Execution**: All training and generalization tests run on **Google Colab**.
- **Observability**: All logs, gradient analysis, and diagnostic outputs from Colab are pasted into `output_store.txt`.

## 2. Decision Framework for the AI (Antigravity)
- **Log-First Analysis**: Before proposing any change, perform a deep audit of the numerical distributions in `output_store.txt` (mean, std, min/max of mel predictions and FiLM parameters).
- **Mathematical Basis**: Every architectural or loss-function change must be justified with mathematical reasoning (e.g., "The gradient signal is too weak because of $X$", or "The attention mechanism is collapsed due to $Y$").
- **Structural over Scalar**: Prioritize fixing structural bottlenecks (how features flow) over simple weight tuning (lambda values).

## 3. Current Project State (April 2026)
- **Loss Function**: `BandStatsSpeakerLoss` (Masked per-band L1 on mel means/stds). This replaced the ineffective cosine loss and is successfully producing 3x more speaker differentiation.
- **Critical Diagnostics**:
    - **Metric**: $D(pred, tgt) - D(pred, src)$. Currently trending negative (good), but conversion is still perceptually weak.
    - **Robotic Quality**: Attributed to the mapping network's tendency to attenuate the residual (scale < 1.0) to satisfy the cross-speaker loss, coupled with a 1-token attention bottleneck.

## 4. Pending Technical Next Steps
1. **Increase Speaker Token Count**: Shift from `num_speaker_tokens = 1` to `8`.
    - **Mathematical Goal**: Break the uniform attention collapse. Allow the model to attend to different speaker "aspects" (pitch vs. resonance) per frame.
2. **Rebuild Cache**: Any change to speaker token shapes requires a full cache wipe (`rm -rf /content/hubertvc_cache`).
