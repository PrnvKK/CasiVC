# hubertvc/training/trainer.py
# ------------------------------------------------------------------
#  Classic PyTorch training loop for HuBERT-VC
#
#  Features
#    • configurable via config.py + CLI overrides
#    • checkpointing:  last.ckpt (periodic)  &  best.ckpt (lowest val loss)
#    • early-stopping on total validation loss
#    • tqdm progress bar with mel / stft / aux / total / lr
#    • frozen HiFi-GAN vocoder inside the loss loop (no grad)
# ------------------------------------------------------------------

from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch.nn.functional as F

from torch.cuda.amp import GradScaler, autocast




import torch
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ───────────────────────────────────────────────────────────────────
#  Project imports
# ───────────────────────────────────────────────────────────────────
from config import (
    AudioConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    PathConfig,
)
from data.dataset import (
    VoiceConversionDataset,
    collate_training_pairs,
)
from data.audio_utils import extract_mel_spectrogram
from models.hubertvc_model import HubertVCModel
from training.losses import VCGeneratorLoss, SpeakerClassifierLoss
from inference import load_vocoder, load_speaker_encoder  # thin wrapper to HiFi-GAN
from models.mobilenet_decoder import MobileNetDecoder

def print_gradients(module, module_name):
    print(f"\n[GRADIENT CHECK] {module_name}")
    for name, param in module.named_parameters():
        if param.grad is None:
            print(f"  {name}: grad is None")
        else:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            grad_min = param.grad.abs().min().item()
            print(f"  {name}: mean={grad_mean:.6e}, max={grad_max:.6e}, min={grad_min:.6e}")


# ==================================================================
#                           Trainer
# ==================================================================
class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        # ----------------------------------------------------------
        # 1. configs (instantiate once; may be mutated by CLI flags)
        # ----------------------------------------------------------
        self.audio_cfg = AudioConfig()
        self.data_cfg = DataConfig()
        self.model_cfg = ModelConfig()
        self.train_cfg = TrainingConfig()
        self.path_cfg = PathConfig()

        #  – optional new attributes added here for save/stop freq
        self.train_cfg.save_every_steps   = getattr(self.train_cfg, "save_every_steps", 0)
        self.train_cfg.save_every_epochs  = getattr(self.train_cfg, "save_every_epochs", 1)
        self.train_cfg.early_stop_patience= getattr(self.train_cfg, "early_stop_patience", 1000)

        # CLI overrides
        if args.save_every_steps is not None:
            self.train_cfg.save_every_steps = args.save_every_steps
        if args.save_every_epochs is not None:
            self.train_cfg.save_every_epochs = args.save_every_epochs
        if args.early_stop_patience is not None:
            self.train_cfg.early_stop_patience = args.early_stop_patience

        # ----------------------------------------------------------
        # 2. device
        # ----------------------------------------------------------
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # ----------------------------------------------------------
        # 3. data
        # ----------------------------------------------------------
        self._init_dataloaders()

        # ----------------------------------------------------------
        # 4. model, vocoder, loss fn
        # ----------------------------------------------------------
        num_speakers = len(self.train_dataset.speaker_to_idx)
        self.model: HubertVCModel = HubertVCModel(
            audio_cfg=self.audio_cfg,
            model_cfg=self.model_cfg,
            training_cfg=self.train_cfg,
            num_speakers=num_speakers
        ).to(self.device)

        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)

        #self.vocoder = load_vocoder(self.path_cfg.vocoder_path, device=self.device, trainable=True)
        
        
        self.vocoder = load_vocoder(self.path_cfg.vocoder_path, device=self.device)
        self.vocoder.eval()
        for p in self.vocoder.parameters():
            p.requires_grad = False
        
        
        self.speaker_encoder = load_speaker_encoder(
            self.path_cfg.speaker_encoder_path, device=self.device
        )
        self.speaker_encoder.eval()
        for p in self.speaker_encoder.parameters():
            p.requires_grad = False

        self.loss_fn = VCGeneratorLoss(self.train_cfg, self.speaker_encoder).to(self.device)
        self.classifier_loss_fn = SpeakerClassifierLoss().to(self.device)

        # ----------------------------------------------------------
        # 5. optimiser & scheduler
        # ----------------------------------------------------------
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.train_cfg.learning_rate,
            betas=(self.train_cfg.adam_beta1, self.train_cfg.adam_beta2),
        )

        
        # Mixed precision training
        self.scaler = GradScaler()


        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.train_cfg.lr_decay_steps,
            gamma=self.train_cfg.lr_decay_factor,
        )

        # ----------------------------------------------------------
        # 6. training state
        # ----------------------------------------------------------
        self.start_epoch      = 0
        self.global_step      = 0
        self.best_val_loss    = float("inf")
        self.no_improve_epochs= 0

        #  (optional) gradient clipping
        self.grad_clip = getattr(self.train_cfg, "grad_clip", None)

        # ----------------------------------------------------------
        # 7. checkpoint resume
        # ----------------------------------------------------------
        if args.checkpoint:
            self._load_checkpoint(args.checkpoint)

        # pretty print config once
        if self.global_step == 0:
            self._log_config()
            
        self._register_decoder_forward_hooks()
    
    def _register_decoder_forward_hooks(self):
        """Register forward hooks to detect NaN/Inf in decoder blocks"""
        def make_hook(module_name):
            def hook_fn(module, input, output):
                # Decoder can return (mel, intermediate) tuple when return_intermediate=True
                check = output[0] if isinstance(output, tuple) else output
                if torch.isnan(check).any() or torch.isinf(check).any():
                    print(f"❌ NaN/Inf detected in {module_name}")
                    raise RuntimeError(f"Numerical instability in {module_name}")
            return hook_fn
        
        for name, module in self.model.named_modules():
            if 'decoder' in name:
                module.register_forward_hook(make_hook(name))
        
        print("✅ Registered NaN/Inf detection hooks on decoder modules")

    # ==============================================================
    #  Data helpers
    # ==============================================================
    def _init_dataloaders(self) -> None:
        batch_size = getattr(self.train_cfg, "batch_size", 16)

        total_max = self.data_cfg.max_items
        if total_max is not None:
            train_ratio = self.data_cfg.train_speaker_ratio
            val_ratio = self.data_cfg.val_speaker_ratio
            ratio_sum = train_ratio + val_ratio
            train_max = int(total_max * train_ratio / ratio_sum)
            val_max = int(total_max * val_ratio / ratio_sum)
        else:
            train_max = None
            val_max = None

        self.train_dataset = VoiceConversionDataset(
            split="train",
            audio_config=self.audio_cfg,
            data_config=self.data_cfg,
            training_config=self.train_cfg,
            max_items=train_max
        )
        self.val_dataset = VoiceConversionDataset(
            split="val",
            audio_config=self.audio_cfg,
            data_config=self.data_cfg,
            training_config=self.train_cfg,
            max_items=val_max
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_training_pairs,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_training_pairs,
        )

    # ==============================================================
    #  Checkpoint I/O
    # ==============================================================
    def _checkpoint_path(self, name: str) -> str:
        Path(self.path_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(self.path_cfg.checkpoint_dir) / name)

    def _save_checkpoint(self, name: str) -> None:
        path = self._checkpoint_path(name)
        torch.save(
            {
                "epoch": self.start_epoch,
                "global_step": self.global_step,
                "model_state": self.model.state_dict(),
                "vocoder_state": self.vocoder.state_dict(),  # ADD vocoder state
                "optim_state": self.optimizer.state_dict(),
                "sched_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),    # ADD scaler state
                "best_val_loss": self.best_val_loss,
                "no_improve_epochs": self.no_improve_epochs,
                "config": self._collect_config(),
            },
            path,
        )

        print(f"[checkpoint] saved → {path}")

    def _load_checkpoint(self, ckpt_path: str) -> None:
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        try:
            self.model.load_state_dict(ckpt["model_state"], strict=False)
        except RuntimeError as e:
            print(f"\n[WARNING] Strict loading failed. This means your model architecture "
                  f"has changed since this checkpoint was saved.\n{e}\n"
                  f"Please start a fresh training run.\n")
            raise e
        try:
            self.optimizer.load_state_dict(ckpt["optim_state"])
            self.scheduler.load_state_dict(ckpt["sched_state"])
        except ValueError:
            print("[Warning] Optimizer/Scheduler state mismatch (expected when adding new parameters). Continuing with fresh optimizer states for new weights.")
        self.start_epoch       = ckpt.get("epoch", 0)
        self.global_step       = ckpt.get("global_step", 0)
        self.best_val_loss     = ckpt.get("best_val_loss", float("inf"))
        self.no_improve_epochs = ckpt.get("no_improve_epochs", 0)

        # ADD vocoder state loading with backward compatibility
        if "vocoder_state" in ckpt:
            self.vocoder.load_state_dict(ckpt["vocoder_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.scheduler.load_state_dict(ckpt["sched_state"])
        # ADD scaler state loading with backward compatibility
        if "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])

        print(f"[checkpoint] resumed from {ckpt_path} at epoch {self.start_epoch}")


    def analyze_gradient_flow(self):
        """
        Prints a summary of the gradient flow for the major components of the HubertVCModel.
        Call this function immediately after loss.backward() in your training loop.
        """

        model = self.model
        print("\n" + "="*60)
        print("           GRADIENT FLOW ANALYSIS")
        print("="*60)
        
        # Define the key components of your model to inspect
        # The keys are descriptive names, the values are keywords in the parameter names
        components_to_check = {
            "Mel Encoder": "mel_encoder",
            "Content Projection": "content_proj",
            "Speaker Projection": "speaker_proj",
            "Cross-Attention": "cross_attn",
            "Decoder Adapter": "decoder.adapter",
            "Decoder Block 0": "decoder.blocks.0",
            "Decoder Mel Proj Content": "decoder.mel_proj_content",
            "Decoder Speaker Delta Proj": "decoder.speaker_delta_proj",
        }
        
        # Add the final decoder block dynamically if it exists
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'blocks') and len(model.decoder.blocks) > 1:
            final_block_idx = len(model.decoder.blocks) - 1
            components_to_check[f"Decoder Block {final_block_idx}"] = f"decoder.blocks.{final_block_idx}"

        # Iterate through all trainable parameters
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
                
            for component_name, keyword in components_to_check.items():
                if keyword in name:
                    grad = param.grad
                    grad_mean_abs = grad.abs().mean().item()
                    grad_max_abs = grad.abs().max().item()
                    
                    # Identify if it's a weight or bias parameter for clarity
                    param_type = "bias" if "bias" in name else "weight"
                    
                    print(f"[{component_name:<22s} ({param_type:^6s})] \t"
                          f"| Grad Mean Abs: {grad_mean_abs:.2e} \t"
                          f"| Grad Max Abs: {grad_max_abs:.2e}")
                          
        print("="*60 + "\n")



      # ==============================================================
      #  Training step / epoch
      # ==============================================================
    def _train_epoch(self) -> dict:
        self.model.train()
        loss_accum = {"mel_content": 0.0, "mel_final": 0.0, "stft": 0.0, "speaker": 0.0, "classifier": 0.0, "var_content": 0.0, "entropy": 0.0, "total": 0.0, "pooled_ce_acc": 0.0, "pooled_ce_cnt": 0, "spk_film_acc": 0.0, "spk_film_cnt": 0, "cross_attn_acc": 0.0, "cross_attn_cnt": 0}
        num_batches = 0

        # Ensure exact reproducibility for this epoch's shuffling and data augmentation (like random crops)
        epoch_seed = 42 + self.start_epoch
        torch.manual_seed(epoch_seed)
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        
        # Override dataloader sampler generator for strict reproducibility
        if hasattr(self.train_loader.sampler, 'generator') and self.train_loader.sampler.generator is None:
            g = torch.Generator()
            g.manual_seed(epoch_seed)
            self.train_loader.sampler.generator = g
        elif hasattr(self.train_loader.sampler, 'generator'):
            self.train_loader.sampler.generator.manual_seed(epoch_seed)

        pbar = tqdm(self.train_loader, desc=f"Train | epoch {self.start_epoch}", leave=False)
        for batch in pbar:

            self.optimizer.zero_grad(set_to_none=True)

            # Extract batch data - UPDATED FOR ECAPA-TDNN AND CACHING
            content_audio = [wav.to(self.device) for wav in batch["content_audio"]] if batch.get("content_audio") is not None and batch["content_audio"][0] is not None else None
            ref_audio = batch["ref_audio"].to(self.device) if batch.get("ref_audio") is not None else None
            ref_mel = batch["ref_mel"].to(self.device) if batch.get("ref_mel") is not None else None
            
            # Caching feature tensors
            if batch.get("content_feats") is not None:
                batch["content_feats"] = batch["content_feats"].to(self.device)
            if batch.get("speaker_feats") is not None:
                batch["speaker_feats"] = batch["speaker_feats"].to(self.device)
                
            gt_mel = batch["gt_mel"].to(self.device)
            gt_wave = batch["gt_wave"].to(self.device)
            gt_lengths = batch["gt_lengths"].to(self.device)    # (B,) real mel frame counts

            # Forward pass - NOW USES ref_audio and optionally precomputed features
            classifier_weight = getattr(self.train_cfg, 'classifier_weight', 0.0)
            mel_classifier_weight = 0.0   # per-frame mel Conv1d CE remains disabled
            pooled_mel_ce_weight = getattr(self.train_cfg, 'pooled_mel_ce_weight', 0.0)
            spk_film_ce_weight = getattr(self.train_cfg, 'spk_film_ce_weight', 0.0)
            cross_attn_ce_weight = getattr(self.train_cfg, 'cross_attn_ce_weight', 0.0)
            need_bottleneck = (classifier_weight > 0 or pooled_mel_ce_weight > 0 or spk_film_ce_weight > 0 or cross_attn_ce_weight > 0 or getattr(self.train_cfg, 'lambda_var', 0) > 0) and (self.model.speaker_classifier is not None or self.model.pooled_mel_classifier is not None or self.model.spk_film_classifier is not None or self.model.cross_attn_classifier is not None or getattr(self.train_cfg, 'lambda_var', 0) > 0)
            
            pred_mel, _, aux = self.model(
                ref_audio=ref_audio,
                content_audio=content_audio,
                gt_mels=gt_mel,
                compute_losses=False,
                return_aux=need_bottleneck,
                return_bottleneck=need_bottleneck,
                precomputed_speaker_feats=batch.get("speaker_feats"),
                precomputed_content_feats=batch.get("content_feats"),
                precomputed_content_speaker_feats=batch.get("speaker_feats")  # S21: same raw ECAPA as target → self gate≈0
            )

            # 🔍 TEMPORAL ALIGNMENT CHECK (print once)
            if self.global_step == 0 and num_batches == 0:
                print("\n" + "="*60)
                print("⏱️ TEMPORAL ALIGNMENT CHECK & UTTERANCE INFO")
                print("="*60)
                print(f"Training on utterance: {batch.get('utterance_id', ['Unknown'])[0]}")
                print(f"Speaker ID: {batch.get('speaker_id', ['Unknown'])[0]}")
                print(f"Predicted mel shape: {pred_mel.shape}")   # (B, 80, T_pred)
                print(f"Target mel shape:    {gt_mel.shape}")     # (B, 80, T_gt)")
                print("="*60 + "\n")

            # We skip HiFi-GAN forward pass entirely to prevent VRAM OOM on Colab.
            # STFT and Speaker losses are disabled for training.
            pred_wave = None
            gt_wave_vocoded = None

            current_epoch = self.start_epoch
            stft_weight = 0.0 # Disabled STFT loss optimization completely
            
            pbar.set_description(f"Train | ep {current_epoch} | Mel-Only")

            pred_mel = pred_mel.to(torch.float32)

            # Calculate Losses
            # Three-path gradient separation:
            #   content_mel  (raw, pre-scale)        → L1 → mel_proj → upstream
            #   variance_mel (post-scale, pre-affine) → Variance → out_scale → upstream
            #   pred_mel     (post-affine, final)     → Speaker → [DETACH] → speaker_affine ONLY
            content_mel = aux.get("prebias_mel", None) if aux is not None else None
            variance_mel = aux.get("variance_mel", None) if aux is not None else None
            losses = self.loss_fn(pred_mel, gt_mel, pred_wave, gt_wave_vocoded, gt_lengths=gt_lengths, content_mel=content_mel, variance_mel=variance_mel)
            
            # Use ALL losses (Mel + Speaker Stats) for actual backpropagation!
            total = losses.total()

            # --- Entropy hinge: penalizes attention outside [1.0, 1.5] (uniform=2.08). Two-sided pocket. ---
            lambda_entropy = getattr(self.train_cfg, 'lambda_entropy', 0.0)
            if lambda_entropy > 0 and hasattr(self.model.cross_attn, '_cached_entropy'):
                entropy_val = self.model.cross_attn._cached_entropy
                entropy_hinge = torch.clamp(1.0 - entropy_val, min=0.0) + torch.clamp(entropy_val - 1.5, min=0.0)
                total = total + lambda_entropy * entropy_hinge
                loss_accum["entropy"] += entropy_hinge.item()
                if num_batches == 0:
                    print(f"[ENTROPY_HINGE] raw_entropy={entropy_val.item():.4f}, hinge={entropy_hinge.item():.4f}, weighted={lambda_entropy * entropy_hinge.item():.4f}")

            
            # --- Speaker classifier CE loss (self-pair, per-frame, no dilution) ---
            if need_bottleneck and aux is not None and "classifier_logits" in aux:
                logits_self = aux["classifier_logits"]  # [B, N, T]
                target_idx = batch["target_speaker_idx"].to(self.device)  # [B]
                T = logits_self.size(-1)
                target_expanded = target_idx.unsqueeze(1).expand(-1, T)  # [B, T]
                ce_loss_self = self.classifier_loss_fn(logits_self, target_expanded)
                total = total + classifier_weight * ce_loss_self
                loss_accum["classifier"] += ce_loss_self.item()
                
                if num_batches == 0:
                    with torch.no_grad():
                        logit_entropy = - (logits_self.softmax(dim=1) * torch.log_softmax(logits_self, dim=1)).sum(dim=1).mean().item()
                        _, predicted = logits_self.max(dim=1)  # [B, T]
                        accuracy = (predicted == target_expanded).float().mean().item()
                        print(f"[CLASSIFIER] weight={classifier_weight}, logit_entropy={logit_entropy:.4f}, self_accuracy={accuracy:.4f}")
                        # Top-3 predicted speakers (first frame of first sample)
                        top3_vals, top3_idx = logits_self[0, :, 0].topk(min(3, logits_self.size(1)))
                        print(f"[CLASSIFIER] frame-0 top3 indices: {top3_idx.tolist()}, values: {[f'{v:.3f}' for v in top3_vals.tolist()]}")

            # --- Block3 classifier CE (self-pair): supervises block3_sum pre-spk_film ---
            if need_bottleneck and aux is not None and "block3_classifier_logits" in aux:
                b3_logits = aux["block3_classifier_logits"]  # [B, N, T]
                T_b3 = b3_logits.size(-1)
                target_expanded_b3 = target_idx.unsqueeze(1).expand(-1, T_b3)
                ce_b3 = self.classifier_loss_fn(b3_logits, target_expanded_b3)
                total = total + classifier_weight * ce_b3
                loss_accum["classifier"] += ce_b3.item()
                if num_batches == 0:
                    with torch.no_grad():
                        _, pred_b3 = b3_logits.max(dim=1)
                        acc_b3 = (pred_b3 == target_expanded_b3).float().mean().item()
                        print(f"[B3_CLASSIFIER] ce={ce_b3.item():.4f}, self_accuracy={acc_b3:.4f}")

            # --- Spk_film classifier CE (self-pair): supervises SpeakerDeltaProj output ---
            # Provides speaker-discriminative gradient to SpeakerDeltaProj —
            # the key ingredient that makes the residual branch speaker-specific
            # rather than a generic reconstructor.
            spk_film_ce_weight = getattr(self.train_cfg, 'spk_film_ce_weight', 0.0)
            if spk_film_ce_weight > 0 and need_bottleneck and aux is not None and "spk_film_classifier_logits" in aux:
                sf_logits = aux["spk_film_classifier_logits"]  # [B, N, T]
                T_sf = sf_logits.size(-1)
                target_expanded_sf = target_idx.unsqueeze(1).expand(-1, T_sf)
                ce_sf = self.classifier_loss_fn(sf_logits, target_expanded_sf)
                total = total + spk_film_ce_weight * ce_sf
                loss_accum["classifier"] += ce_sf.item()
                with torch.no_grad():
                    _, pred_sf = sf_logits.max(dim=1)
                    acc_sf = (pred_sf == target_expanded_sf).float().mean().item()
                    loss_accum["spk_film_acc"] += acc_sf
                    loss_accum["spk_film_cnt"] += 1
                    if num_batches == 0:
                        print(f"[SPK_FILM_CLASSIFIER] weight={spk_film_ce_weight}, ce={ce_sf.item():.4f}, self_accuracy={acc_sf:.4f}")

            # --- Cross-attn classifier CE (self-pair): supervises MHA value path ---
            if cross_attn_ce_weight > 0 and need_bottleneck and aux is not None and "cross_attn_classifier_logits" in aux:
                ca_logits = aux["cross_attn_classifier_logits"]  # [B, N, T]
                T_ca = ca_logits.size(-1)
                target_expanded_ca = target_idx.unsqueeze(1).expand(-1, T_ca)
                ce_ca = self.classifier_loss_fn(ca_logits, target_expanded_ca)
                total = total + cross_attn_ce_weight * ce_ca
                loss_accum["classifier"] += ce_ca.item()
                with torch.no_grad():
                    _, pred_ca = ca_logits.max(dim=1)
                    acc_ca = (pred_ca == target_expanded_ca).float().mean().item()
                    loss_accum["cross_attn_acc"] += acc_ca
                    loss_accum["cross_attn_cnt"] += 1
                    if num_batches == 0:
                        print(f"[CROSS_ATTN_CE] weight={cross_attn_ce_weight}, ce={ce_ca.item():.4f}, self_accuracy={acc_ca:.4f}")

            # --- Mel-output classifier CE (self-pair): prevents mel_proj erasure ---
            if mel_classifier_weight > 0 and aux is not None and "mel_classifier_logits" in aux:
                mel_logits = aux["mel_classifier_logits"]  # [B, N, T]
                T_mel = mel_logits.size(-1)
                target_expanded_mel = target_idx.unsqueeze(1).expand(-1, T_mel)
                ce_mel = self.classifier_loss_fn(mel_logits, target_expanded_mel)
                total = total + mel_classifier_weight * ce_mel
                if num_batches == 0:
                    with torch.no_grad():
                        _, pred_mel_cls = mel_logits.max(dim=1)
                        acc_mel = (pred_mel_cls == target_expanded_mel).float().mean().item()
                        print(f"[MEL_CLASSIFIER] ce={ce_mel.item():.4f}, self_accuracy={acc_mel:.4f}")

            # --- Pooled mel-bias CE (self-pair): gated gradient to bias MLP only ---
            if pooled_mel_ce_weight > 0 and aux is not None and "pooled_mel_logits" in aux:
                pooled_logits = aux["pooled_mel_logits"]  # [B, N] — already pooled over time
                ce_pooled_mel = F.cross_entropy(pooled_logits, target_idx)  # [B, N] vs [B]
                total = total + pooled_mel_ce_weight * ce_pooled_mel
                loss_accum["classifier"] += ce_pooled_mel.item()
                with torch.no_grad():
                    _, pred_pooled = pooled_logits.max(dim=1)
                    acc_pooled = (pred_pooled == target_idx).float().mean().item()
                    loss_accum["pooled_ce_acc"] += acc_pooled
                    loss_accum["pooled_ce_cnt"] += 1
                if num_batches == 0:
                    print(f"[POOLED_MEL_CE] weight={pooled_mel_ce_weight}, ce={ce_pooled_mel.item():.4f}, self_accuracy={acc_pooled:.4f}")
            
            print(f"[LOSS_DEBUG] stft_weight={stft_weight:.4f}, "
            f"raw_stft={losses.get('stft', 0):.4f}, "
            f"actual_total={total:.4f}")
            
            # =========================================================================
            # CROSS-PAIR TRAINING: Mel Spectral Stats only (no L1 on cross pairs)
            # =========================================================================
            # Mathematically safe because Mel Stats matches per-band spectral shape
            # without penalizing phoneme mismatch: ∂L_stats/∂γ encourages stronger γ.
            # Reuses precomputed content/speaker features — no HuBERT/ECAPA re-run.
            cross_pair_prob = getattr(self.train_cfg, 'cross_pair_prob', 0.0)
            cross_stats_weight = getattr(self.train_cfg, 'cross_pair_stats_weight', 1.0)
            
            if cross_pair_prob > 0:
                speaker_ids = batch.get("speaker_id", [])
                B = len(speaker_ids)
                
                # Build mask: cross-pair only for different-speaker neighbours
                cross_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
                if B >= 2:
                    for i in range(B):
                        j = (i - 1) % B  # matches torch.roll(shifts=1): item i pairs with i-1
                        if speaker_ids[i] != speaker_ids[j]:
                            cross_mask[i] = True
                
                valid_pairs = cross_mask.sum().item()
                print(f"[CROSS-PAIR-CHECK] prob={cross_pair_prob}, B={B}, valid={valid_pairs}")
                
                if valid_pairs > 0 and random.random() < cross_pair_prob:
                    # Roll speaker features and targets by 1
                    rolled_speaker_feats = torch.roll(batch["speaker_feats"], shifts=1, dims=0)
                    rolled_gt_mel = torch.roll(gt_mel, shifts=1, dims=0)
                    rolled_lengths = torch.roll(gt_lengths, shifts=1, dims=0)
                    
                    # Forward pass with rolled speaker (content unchanged)
                    pred_cross, _, aux_cross = self.model(
                        ref_audio=None,
                        content_audio=None,
                        gt_mels=None,
                        compute_losses=False,
                        return_aux=need_bottleneck,
                        return_bottleneck=need_bottleneck,
                        precomputed_speaker_feats=rolled_speaker_feats,
                        precomputed_content_feats=batch.get("content_feats"),
                        precomputed_content_speaker_feats=batch.get("speaker_feats")  # S21: original (unrolled) raw ECAPA as content speaker
                    )
                    pred_cross = pred_cross.to(torch.float32)
                    
                    # Cross-pair loss: Mel Spectral Stats ONLY (no L1)
                    # Uses independent pred/target lengths to avoid paddding corruption
                    cross_loss_tensor = self.loss_fn.mel_stats_loss(
                        pred_cross,
                        rolled_gt_mel,
                        pred_lengths=gt_lengths,        # content speaker's length
                        target_lengths=rolled_lengths    # target speaker's length
                    )
                    
                    # Average over valid cross-pairs only
                    cross_loss = cross_loss_tensor * cross_stats_weight
                    total = total + cross_loss
                    
                    # --- Classifier CE on cross-pair (target = rolled speaker, masked by cross_mask) ---
                    mask_f = cross_mask.float()
                    valid_count = mask_f.sum().clamp(min=1)

                    if need_bottleneck and aux_cross is not None and "classifier_logits" in aux_cross:
                        logits_cross = aux_cross["classifier_logits"]  # [B, N, T]
                        target_idx_rolled = torch.roll(batch["target_speaker_idx"], shifts=1, dims=0).to(self.device)
                        T_cross = logits_cross.size(-1)
                        target_cross_expanded = target_idx_rolled.unsqueeze(1).expand(-1, T_cross)
                        ce_raw = F.cross_entropy(logits_cross, target_cross_expanded, reduction='none')  # [B, T]
                        ce_per_item = ce_raw.mean(dim=-1)  # [B]
                        ce_loss_cross = (ce_per_item * mask_f).sum() / valid_count
                        total = total + classifier_weight * ce_loss_cross

                    # --- Block3 classifier CE on cross-pair ---
                    if need_bottleneck and aux_cross is not None and "block3_classifier_logits" in aux_cross:
                        b3_logits_cross = aux_cross["block3_classifier_logits"]  # [B, N, T]
                        T_b3c = b3_logits_cross.size(-1)
                        target_b3_rolled = target_idx_rolled.unsqueeze(1).expand(-1, T_b3c)
                        ce_b3_raw = F.cross_entropy(b3_logits_cross, target_b3_rolled, reduction='none')  # [B, T]
                        ce_b3_per_item = ce_b3_raw.mean(dim=-1)  # [B]
                        ce_b3_cross = (ce_b3_per_item * mask_f).sum() / valid_count
                        total = total + classifier_weight * ce_b3_cross
                        loss_accum["classifier"] += ce_b3_cross.item()

                    # --- Spk_film classifier CE on cross-pair ---
                    spk_film_ce_weight = getattr(self.train_cfg, 'spk_film_ce_weight', 0.0)
                    if spk_film_ce_weight > 0 and need_bottleneck and aux_cross is not None and "spk_film_classifier_logits" in aux_cross:
                        sf_logits_cross = aux_cross["spk_film_classifier_logits"]  # [B, N, T]
                        T_sfc = sf_logits_cross.size(-1)
                        target_sf_rolled = target_idx_rolled.unsqueeze(1).expand(-1, T_sfc)
                        ce_sf_raw = F.cross_entropy(sf_logits_cross, target_sf_rolled, reduction='none')  # [B, T]
                        ce_sf_per_item = ce_sf_raw.mean(dim=-1)  # [B]
                        ce_sf_cross = (ce_sf_per_item * mask_f).sum() / valid_count
                        total = total + spk_film_ce_weight * ce_sf_cross
                        loss_accum["classifier"] += ce_sf_cross.item()
                        with torch.no_grad():
                            _, pred_sf_cross = sf_logits_cross.max(dim=1)
                            acc_sf_cross = (pred_sf_cross == target_sf_rolled).float().mean().item()
                            loss_accum["spk_film_acc"] += acc_sf_cross
                            loss_accum["spk_film_cnt"] += 1

                    # --- Mel-output classifier CE on cross-pair ---
                    if mel_classifier_weight > 0 and aux_cross is not None and "mel_classifier_logits" in aux_cross:
                        mel_logits_cross = aux_cross["mel_classifier_logits"]  # [B, N, T]
                        T_melc = mel_logits_cross.size(-1)
                        target_mel_rolled = target_idx_rolled.unsqueeze(1).expand(-1, T_melc)
                        ce_mel_raw = F.cross_entropy(mel_logits_cross, target_mel_rolled, reduction='none')  # [B, T]
                        ce_mel_per_item = ce_mel_raw.mean(dim=-1)  # [B]
                        ce_mel_cross = (ce_mel_per_item * mask_f).sum() / valid_count
                        total = total + mel_classifier_weight * ce_mel_cross

                    # --- Pooled mel-bias CE on cross-pair (gated) ---
                    if pooled_mel_ce_weight > 0 and aux_cross is not None and "pooled_mel_logits" in aux_cross:
                        pooled_logits_cross = aux_cross["pooled_mel_logits"]  # [B, N]
                        ce_pooled_raw = F.cross_entropy(pooled_logits_cross, target_idx_rolled, reduction='none')  # [B]
                        ce_pooled_cross = (ce_pooled_raw * mask_f).sum() / valid_count
                        total = total + pooled_mel_ce_weight * ce_pooled_cross
                        loss_accum["classifier"] += ce_pooled_cross.item()
                        with torch.no_grad():
                            _, pred_pooled_cross = pooled_logits_cross.max(dim=1)
                            acc_pooled_cross = (pred_pooled_cross == target_idx_rolled).float().mean().item()
                            loss_accum["pooled_ce_acc"] += acc_pooled_cross
                            loss_accum["pooled_ce_cnt"] += 1
                    
                    # --- Cross-attn classifier CE on cross-pair ---
                    if cross_attn_ce_weight > 0 and need_bottleneck and aux_cross is not None and "cross_attn_classifier_logits" in aux_cross:
                        ca_logits_cross = aux_cross["cross_attn_classifier_logits"]  # [B, N, T]
                        T_cac = ca_logits_cross.size(-1)
                        target_ca_rolled = target_idx_rolled.unsqueeze(1).expand(-1, T_cac)
                        ce_ca_raw = F.cross_entropy(ca_logits_cross, target_ca_rolled, reduction='none')  # [B, T]
                        ce_ca_per_item = ce_ca_raw.mean(dim=-1)  # [B]
                        ce_ca_cross = (ce_ca_per_item * mask_f).sum() / valid_count
                        total = total + cross_attn_ce_weight * ce_ca_cross
                        loss_accum["classifier"] += ce_ca_cross.item()
                        with torch.no_grad():
                            _, pred_ca_cross = ca_logits_cross.max(dim=1)
                            acc_ca_cross = (pred_ca_cross == target_ca_rolled).float().mean().item()
                            loss_accum["cross_attn_acc"] += acc_ca_cross
                            loss_accum["cross_attn_cnt"] += 1

                    if num_batches == 0:
                        rolled_ids = [speaker_ids[(i+1) % B] for i in range(B)]
                        print(f"[CROSS-PAIR] prob={cross_pair_prob}, weight={cross_stats_weight}")
                        print(f"[CROSS-PAIR] Active: {valid_pairs}/{B} items cross-paired (mask={cross_mask.tolist()})")
                        print(f"[CROSS-PAIR] speakers: {speaker_ids} → {rolled_ids}")
                        print(f"[CROSS-PAIR] self_loss={losses.total():.4f}, cross_stats_loss={cross_loss_tensor.item():.4f}, "
                              f"weighted_cross={cross_loss.item():.4f}")
                        print(f"[CROSS-PAIR] total (self + cross) = {total:.4f}")
            
            # 🔍 ENHANCED SIGNAL AUDIT (Run on first batch of session + periodically)
            if num_batches == 0 and (self.global_step == 0 or self.global_step % 100 == 0):
                print("\n" + "█"*60)
                print(f"🕵️  DEEP SIGNAL AUDIT | Step {self.global_step}")
                print("█"*60)

                # Safe accessor to handle (B, C, T) vs (B, T)
                def get_sample(tensor):
                    t = tensor.detach().cpu()
                    if t.dim() == 3: return t[0, 0, :8].numpy() # (B, C, T) -> First channel
                    if t.dim() == 2: return t[0, :8].numpy()    # (B, T) or (B, C) -> First row
                    return t.flatten()[:8].numpy()              # Fallback

                t_sample = get_sample(gt_mel)
                p_sample = get_sample(pred_mel)
                
                print(f"\n[1] RAW SAMPLE VALUES (First 8 frames/points):")
                print(f"    Target: {[float(f'{x:.2f}') for x in t_sample]}") 
                print(f"    Pred:   {[float(f'{x:.2f}') for x in p_sample]}")

                # 2. QUANTILES
                print(f"\n[2] DISTRIBUTION SHAPE:")
                for name, tensor in [("Target", gt_mel), ("Pred  ", pred_mel)]:
                    t_flat = tensor.float().reshape(-1)
                    q = torch.quantile(t_flat, torch.tensor([0.05, 0.5, 0.95]).to(self.device))
                    mean, std = t_flat.mean().item(), t_flat.std().item()
                    print(f"    {name}: 5%={q[0]:.2f} | Med={q[1]:.2f} | 95%={q[2]:.2f} || μ={mean:.2f}, σ={std:.2f}")

                # 3. SCALE MISMATCH DIAGNOSIS
                l1_err = torch.abs(gt_mel - pred_mel).mean().item()
                

                print(f"\n[3] DIAGNOSIS:")
                print(f"    Mean L1 Error: {l1_err:.4f}")
                if abs(gt_mel.mean()) > 2.0 and abs(pred_mel.mean()) < 0.5:
                     print("    🚨 ALERT: Target looks like Unnormalized Log-Mel (negative offset), Pred looks Zero-Centered.")
                elif l1_err > 2.0:
                     print("    ⚠️  WARNING: Massive scale mismatch detected.")
                else:
                     print("    ✅ Scales look roughly compatible.")
                print("█"*60 + "\n")

                with torch.no_grad():
                    print(f"[DOMAIN GAP] STFT diagnostics are disabled to save Colab VRAM.")
                    print(f"[SAME DOMAIN] Skipping diagnostic STFT calculation...")

            
            # =============================================================================
            # 🩺 VOCODER & MEL CONSISTENCY DIAGNOSTIC (Epoch-level, non-invasive)
            # =============================================================================
            if self.start_epoch in (0, 8):
                print("\n" + "=" * 100)
                print(f"🔬 VOCODER DIAGNOSTIC | Epoch {self.start_epoch:03d} | Step {self.global_step}")
                print("=" * 100)

                # -------------------------------------------------------------------------
                # Select first sample in batch: [B, 80, T] → [80, T]
                # -------------------------------------------------------------------------
                gt_mel_sample = gt_mel[0].detach()

                # Prediction might have lost its batch dim
                if pred_mel.dim() == 3:  # [B, 80, T]
                    pred_mel_sample = pred_mel[0].detach()
                else:                    # [80, T] (Batch dim already squeezed)
                    pred_mel_sample = pred_mel.detach()

                # In the vocoder diagnostic section, compare:
                print(f"[MEL_SCALE] gt_mel: mean={gt_mel.mean():.3f}, std={gt_mel.std():.3f}")
                print(f"[MEL_SCALE] pred_mel: mean={pred_mel.mean():.3f}, std={pred_mel.std():.3f}")
                # Both should be in the range the vocoder expects
                # SpeechBrain HiFi-GAN typically expects unnormalized log-mel in [-11, 2] range

                # -------------------------------------------------------------------------
                # 1. MEL DISTRIBUTION STATS
                # -------------------------------------------------------------------------
                print("\n📊 1. MEL DISTRIBUTION")
                print("─" * 60)
                print(
                    f"GT   | min={gt_mel_sample.min():6.3f} "
                    f"max={gt_mel_sample.max():6.3f} "
                    f"mean={gt_mel_sample.mean():6.3f} "
                    f"std={gt_mel_sample.std():6.3f}"
                )
                print(
                    f"PRED | min={pred_mel_sample.min():6.3f} "
                    f"max={pred_mel_sample.max():6.3f} "
                    f"mean={pred_mel_sample.mean():6.3f} "
                    f"std={pred_mel_sample.std():6.3f}"
                )

                l1_val = torch.nn.functional.l1_loss(pred_mel_sample.unsqueeze(0), gt_mel_sample.unsqueeze(0)).item()
                print(f"L1(pred, gt) = {l1_val:.4f}")

                # Percentiles (distribution shape)
                gt_p5, gt_p95 = torch.quantile(gt_mel_sample, torch.tensor([0.05, 0.95], device=gt_mel_sample.device))
                pr_p5, pr_p95 = torch.quantile(pred_mel_sample, torch.tensor([0.05, 0.95], device=pred_mel_sample.device))
                print(f"GT   5–95%: [{gt_p5:6.3f}, {gt_p95:6.3f}]")
                print(f"PRED 5–95%: [{pr_p5:6.3f}, {pr_p95:6.3f}]")

                # -------------------------------------------------------------------------
                # 2. VOCODER HANDSHAKE TESTS (SpeechBrain HiFi-GAN fix)
                # -------------------------------------------------------------------------
                print("\n🔧 2. VOCODER INPUT / OUTPUT CHECK")
                print("─" * 60)

                vocoder_tests = [
                    ("GT Mel (baseline)", gt_mel_sample),
                    ("Pred Mel (raw)", pred_mel_sample),
                    ("Pred → match GT μ/σ", 
                    (pred_mel_sample - pred_mel_sample.mean()) / (pred_mel_sample.std() + 1e-6)
                    * gt_mel_sample.std() + gt_mel_sample.mean()),
                    ("Silence (-10)", torch.full_like(pred_mel_sample, -10.0)),
                ]

                with torch.no_grad():
                    for name, mel_in in vocoder_tests:
                        # FIX: SpeechBrain HiFi-GAN expects [B=1, C=80, T]
                        mel_in_correct = mel_in.unsqueeze(0)  # [1, 80, T] ✅ CORRECT
                        wav = self.vocoder(mel_in_correct).squeeze()  # [T]

                        wav_std = wav.std().item()
                        wav_peak = wav.abs().max().item()
                        clip_pct = (wav.abs() > 1.0).float().mean().item() * 100.0

                        print(
                            f"{name:22s} | "
                            f"wav σ={wav_std:6.3f} "
                            f"peak={wav_peak:6.3f} "
                            f"clip={clip_pct:5.1f}%"
                        )

                # -------------------------------------------------------------------------
                # 3. TEMPORAL SANITY CHECK (first mel bin, first 20 frames)
                # -------------------------------------------------------------------------
                print("\n⏱️ 3. TEMPORAL SNAPSHOT (mel bin 0)")
                print("─" * 60)
                T = min(20, gt_mel_sample.shape[1])
                print("t | GT      | PRED    | |Δ|")
                for t in range(T):
                    g = gt_mel_sample[0, t].item()
                    p = pred_mel_sample[0, t].item()
                    print(f"{t:2d} | {g:7.3f} | {p:7.3f} | {abs(g-p):7.3f}")

                print("=" * 100 + "\n")

            self.scaler.scale(total).backward()
            vocoder_has_grad = False
            for name, p in self.vocoder.named_parameters():
                if p.grad is not None and p.grad.abs().max() > 0:
                    vocoder_has_grad = True
                    break
            print(f"[VOCODER_GRAD] Gradients flowing through vocoder: {vocoder_has_grad}")

            self.scaler.unscale_(self.optimizer)

            if self.global_step % 1 == 0:  # Print every step for now
                print(f"\n[GRADIENT ANALYSIS]")
                
                # Content projection gradients
                if self.model.cross_attn.content_proj.weight.grad is not None:
                    content_grad = self.model.cross_attn.content_proj.weight.grad
                    print(f"content_proj grad: mean={content_grad.abs().mean():.6f}, "
                          f"max={content_grad.abs().max():.6f}")
                
                # Speaker projection gradients
                if self.model.cross_attn.speaker_proj.weight.grad is not None:
                    speaker_grad = self.model.cross_attn.speaker_proj.weight.grad
                    print(f"speaker_proj grad: mean={speaker_grad.abs().mean():.6f}, "
                          f"max={speaker_grad.abs().max():.6f}")
                
                # Alpha was replaced by the AdaIN Mapping Network

            self.analyze_gradient_flow()

            if self.grad_clip:
                params_to_clip = [p for p in self.model.parameters() if p.requires_grad]
                params_to_clip += [p for p in self.vocoder.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=2.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # accumulate losses
            for k in loss_accum:
                loss_accum[k] += losses.get(k, torch.tensor(0.)).item() if k != "total" else total.item()
            num_batches += 1

            postfix = {
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                "mel_c": f"{losses.get('mel_content', torch.tensor(0.)).item():.3f}",
                "mel_f": f"{losses.get('mel_final', torch.tensor(0.)).item():.3f}",
                "stft": f"{losses.get('stft', torch.tensor(0.)).item():.3f}",
                "spk": f"{losses.get('speaker', torch.tensor(0.)).item():.3f}",
                "var_c": f"{losses.get('var_content', torch.tensor(0.)).item():.3f}",
                "tot": f"{total.item():.3f}",
            }
            if need_bottleneck:
                postfix["cls"] = f"{ce_loss_self.item():.3f}"
            pbar.set_postfix(**postfix)

            if (
                self.train_cfg.save_every_steps > 0
                and self.global_step % self.train_cfg.save_every_steps == 0
            ):
                self._save_checkpoint("last.ckpt")

            self.global_step += 1

        for k in loss_accum:
            loss_accum[k] /= max(1, num_batches)

        return loss_accum


    # ==============================================================
    #  Validation
    # ==============================================================
    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        self.vocoder.eval()
        loss_accum = {"mel": 0.0, "mel_content": 0.0, "mel_final": 0.0, "stft": 0.0, "speaker": 0.0, "var_content": 0.0, "total": 0.0}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Valid", leave=False):

            # Extract batch data - UPDATED FOR ECAPA-TDNN AND CACHING
            content_audio = [wav.to(self.device) for wav in batch["content_audio"]] if batch.get("content_audio") is not None and batch["content_audio"][0] is not None else None
            ref_audio = batch["ref_audio"].to(self.device) if batch.get("ref_audio") is not None else None
            ref_mel = batch["ref_mel"].to(self.device) if batch.get("ref_mel") is not None else None
            
            # Caching feature tensors
            if batch.get("content_feats") is not None:
                batch["content_feats"] = batch["content_feats"].to(self.device)
            if batch.get("speaker_feats") is not None:
                batch["speaker_feats"] = batch["speaker_feats"].to(self.device)
                
            gt_mel = batch["gt_mel"].to(self.device)
            gt_wave = batch["gt_wave"].to(self.device)
            gt_lengths = batch["gt_lengths"].to(self.device) if batch.get("gt_lengths") is not None else None

            # Forward pass - NOW USES ref_audio and optionally precomputed features
            # return_bottleneck=True needed for content_mel (variance loss)
            pred_mel, _, aux_val = self.model(
                ref_audio=ref_audio,
                content_audio=content_audio,
                gt_mels=None,
                compute_losses=False,
                return_aux=True,
                return_bottleneck=True,
                precomputed_speaker_feats=batch.get("speaker_feats"),
                precomputed_content_feats=batch.get("content_feats")
            )
            pred_wave = None

            # Same-domain STFT ground truth
            gt_wave_vocoded = None
            content_mel_val = aux_val.get("prebias_mel", None) if aux_val is not None else None
            variance_mel_val = aux_val.get("variance_mel", None) if aux_val is not None else None
            losses = self.loss_fn(pred_mel, gt_mel, pred_wave, gt_wave_vocoded, gt_lengths=gt_lengths, content_mel=content_mel_val, variance_mel=variance_mel_val)

            total = losses.total()

            for k in loss_accum:
                loss_accum[k] += losses.get(k, torch.tensor(0.)).item() if k != "total" else total.item()
            num_batches += 1

        for k in loss_accum:
            loss_accum[k] /= max(1, num_batches)

        return loss_accum


    # ==============================================================
    #  Public runner
    # ==============================================================
    def _build_cache_if_needed(self):
        """Builds a local cache of HuBERT and ECAPA features to speed up Colab training."""
        cache_dir = Path("/content/hubertvc_cache")
        # To force recomputation on the first epoch of every run, we can wipe the folder
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("🚀 PRECOMPUTING FROZEN FEATURES (HuBERT + ECAPA-TDNN)")
        print("="*60)
        self.model.eval()
        
        for split, loader in [("Train", self.train_loader), ("Val", self.val_loader)]:
            print(f"Caching {split} dataset...")
            # Use raw dataset directly to bypass padding for single utterances
            dataset = loader.dataset
            for idx in tqdm(range(len(dataset)), desc=f"{split} Cache", leave=False):
                speaker_id, utterance_path = dataset.valid_utterances[idx]
                utterance_id = Path(utterance_path).stem
                
                # Check if this precise utterance is cached
                cache_file = cache_dir / f"{utterance_id}.pt"
                if cache_file.exists():
                    continue
                    
                # Run the standard audio loader manually to get the exact split
                result = dataset._load_and_process_audio(utterance_path)
                if result is None:
                    continue
                ref_audio, ref_mel, content_audio, _, _ = result
                
                # Send to GPU
                ref_audio_gpu = ref_audio.unsqueeze(0).to(self.device)
                content_audio_gpu = [content_audio.to(self.device)]
                
                with torch.inference_mode():
                    # 1. ECAPA-TDNN Speaker Features — cache raw 192D embedding
                    # (projection + token_norm applied during training so they receive gradients)
                    speaker_feats = self.model.mel_encoder.extract_speaker_features(ref_audio_gpu, apply_projection=False)[0]  # [192]
                    
                    # 2. HuBERT Content Features 
                    hubert_out = self.model.hubert(content_audio_gpu)
                    # Cache raw 768D features so hubert_proj receives gradients during training
                    content_feats = hubert_out[0] # [T, 768]
                    
                    # 3. GT Mel 
                    gt_mel = extract_mel_spectrogram(
                        content_audio, 
                        sample_rate=self.audio_cfg.sample_rate
                    )
                    
                # Save to disk
                torch.save({
                    'content_feats': content_feats.cpu(),
                    'speaker_feats': speaker_feats.cpu(),
                    'gt_mel': gt_mel.cpu(),
                    'gt_wave': content_audio.cpu(),
                }, cache_file)
                
        print("✅ Precompution complete! Training will now run exponentially faster.\n")
        self.model.train()


    def run(self, max_epochs: int) -> None:
        #print(f"[trainer] device: {self.device} | epochs: {max_epochs}")
        
        self._build_cache_if_needed()
        
        for epoch in range(self.start_epoch, max_epochs):
            self.start_epoch = epoch

            train_loss = self._train_epoch()
            self.scheduler.step() 
            val_loss   = self._validate()

            spk_film_acc = train_loss.get('spk_film_acc', 0.0) / max(1, train_loss.get('spk_film_cnt', 1))
            pooled_ce_acc = train_loss.get('pooled_ce_acc', 0.0) / max(1, train_loss.get('pooled_ce_cnt', 1))
            print(
                f"[epoch {epoch:03d}] train:   mel_c={train_loss.get('mel_content', 0.0):.3f} mel_f={train_loss.get('mel_final', 0.0):.3f} spk={train_loss['speaker']:.3f} var={train_loss.get('var', 0.0):.3f} cls={train_loss.get('classifier', 0.0):.3f} sf_acc={spk_film_acc:.3f} pl_acc={pooled_ce_acc:.3f} total={train_loss['total']:.3f}"
            )
            print(
                f"                 val:     mel_c={val_loss.get('mel_content', 0.0):.3f} mel_f={val_loss.get('mel_final', 0.0):.3f} spk={val_loss['speaker']:.3f} var={val_loss.get('var', 0.0):.3f} total={val_loss['total']:.3f}"
            )


            # ---- epoch-based saving
            if (
                self.train_cfg.save_every_epochs > 0
                and (epoch + 1) % self.train_cfg.save_every_epochs == 0
            ):
                self._save_checkpoint("last.ckpt")

            # ---- best model logic
            if val_loss['total'] < self.best_val_loss:
                self.best_val_loss = val_loss['total']
                self.no_improve_epochs = 0
                self._save_checkpoint("best.ckpt")
            else:
                self.no_improve_epochs += 1

            # ---- early stopping
            if self.no_improve_epochs >= self.train_cfg.early_stop_patience:
                print(
                    f"[early-stop] no improvement for "
                    f"{self.no_improve_epochs} epochs; stopping."
                )
                break


        print("[trainer] training completed")

    # ==============================================================
    #  utilities
    # ==============================================================
    def _collect_config(self) -> Dict[str, Any]:
        # serialisable snapshot (for reproducibility)
        return {
            "audio": vars(self.audio_cfg),
            "data": vars(self.data_cfg),
            "model": vars(self.model_cfg),
            "train": vars(self.train_cfg),
        }

    def _log_config(self) -> None:
        cfg_json = json.dumps(self._collect_config(), indent=2)
        print("═════════ CONFIG ═════════")
        print(cfg_json)
        print("══════════════════════════")
    

# ==================================================================
#                       CLI entry-point
# ==================================================================
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HuBERT-VC training loop (manual)")
    p.add_argument("--checkpoint", type=str, default=None, help="resume from .ckpt")
    p.add_argument("--device",     type=str, default="cuda", help="cpu | cuda | cuda:0")
    p.add_argument("--epochs",     type=int, default=100,    help="max epochs")

    # optional overrides
    p.add_argument("--save_every_steps",   type=int, help="save last.ckpt every N steps")
    p.add_argument("--save_every_epochs",  type=int, help="save last.ckpt every N epochs")
    p.add_argument("--early_stop_patience",type=int, help="epochs without val-improve")

    return p.parse_args()


def main() -> None:
    args = _parse_cli()
    trainer = Trainer(args)
    trainer.run(max_epochs=args.epochs)


if __name__ == "__main__":
    main()
