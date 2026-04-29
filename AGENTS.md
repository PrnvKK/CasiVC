# CasiVC: AI Agent Collaboration Guide

Welcome to the CasiVC project. This document serves as the primary orientation for any AI agent tasked with debugging, optimizing, or expanding this zero-shot voice conversion system.

## 1. Project Objective
CasiVC is a high-fidelity, zero-shot voice conversion system designed to decouple content from timbre. It uses a frozen HuBERT encoder for semantic features and a multi-token ECAPA-TDNN based encoder for speaker characteristics, fused via a position-agnostic cross-attention mechanism.

**The primary goal:** Achieving stable, artifact-free timbre transfer without robotic buzzing or source-speaker leakage.


## 2. Design Constraints & Compute Optimization

Due to significant compute restrictions, this project follows a strict efficiency-first philosophy:

- **Minimal Trainable Parameters:** We aggressively prioritize lightweight architectures. The model is designed to be lean, primarily using efficient components like MobileNet-style blocks.
- **No Active Large Models:** Large-scale components (such as the HuBERT encoder) must remain **frozen**. We do not have the compute overhead to fine-tune or active-train heavy backbone architectures.
- **Parameter Efficiency:** All proposed modifications should aim to improve fidelity through mathematical refinement rather than increasing parameter count or model depth.

## 3. File System Overview

### Core Configuration
- `config.py`: Central repository for all hyperparameters (Audio, Model, Training). Avoid hardcoding values in modules; reference this file instead.

### Model Architecture (`models/`)
- `hubertvc_model.py`: The main orchestrator that connects all sub-modules.
- `hubert_encoder.py`: Extracts semantic content (Frozen HuBERT).
- `mel_encoder.py`: Learns a set of speaker tokens (Timbre dictionary) from reference audio.
- `cross_attention.py`: The fusion "brain." Contains the Attention mechanism and the Mapping Network (FiLM) that conditions the decoder.
- `mobilenet_decoder.py`: A lightweight, efficient Mel-spectrogram generator.

### Training & Logic (`training/`)
- `trainer.py`: Handles the training loop, validation, and specialized logging (Audit logs for gradients, entropy, and scales).
- `losses.py`: Implementation of Mel-reconstruction loss and Multi-Resolution STFT penalties.

### Evaluation & Inference
- `test_generalization.py`: The primary benchmarking tool. It tests the model on unseen speaker pairs and outputs metrics like `SPK_SIM` (Speaker Similarity) and `BAND_METRIC` (Timbre Bias).
- `inference.py`: Script for one-off conversions and production use.

## 4. Operational Workflow

- **Execution Environment:** The project is designed to be trained and evaluated in a **Google Colab** environment.
- **Data Persistence:** All meaningful training logs, generalization results, and diagnostic audits must be redirected or saved to `output_store.txt`, `epoch_summary.txt`, or `eval_summary.txt`.
- **Feedback Loop:** 
    1. Agent proposes changes.
    2. User runs training/eval in Colab.
    3. User provides the contents of the summary files.
    4. Agent analyzes the data and iterates.

## 5. Analysis Principles for Agents

All observations and recommendations made by an agent **MUST** follow these criteria:

1.  **Analytical & Log-Based:** Never guess. Base every conclusion on the numerical data provided in the summary files (e.g., standard deviations, min/max gradients, loss curves).
2.  **Mathematical Basis:** Use concepts like Attention Entropy (to detect mushiness), Variance analysis (to detect FiLM explosions), and Gradient magnitude (to detect vanishing/exploding updates).
3.  **Surgical Precision:** When recommending changes, specify the exact mathematical transformation (e.g., changing an activation function or adding a temperature scalar).
4.  **Debug Requests:** If the current logs are insufficient, the agent is encouraged to request the insertion of specific "Audit Prints" into the codebase to capture hidden internal states.

---
*Note: This project prioritizes signal-level stability over brute-force training duration. If the math doesn't align in the first 20 epochs, it won't align in 100.*
