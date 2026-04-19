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
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

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
from training.losses import VCGeneratorLoss
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
        self.model: HubertVCModel = HubertVCModel(
            self.audio_cfg,
            self.model_cfg,
            self.train_cfg
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

        self.loss_fn = VCGeneratorLoss(self.train_cfg).to(self.device)

        # ----------------------------------------------------------
        # 5. optimiser & scheduler
        # ----------------------------------------------------------
        # BandStatsSpeakerLoss has NO trainable parameters — only model params needed.
        all_params = list(self.model.parameters())
        self.optimizer = Adam(
            all_params,
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
                if torch.isnan(output).any() or torch.isinf(output).any():
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

        self.train_dataset = VoiceConversionDataset(
            split="train",
            audio_config=self.audio_cfg,
            data_config=self.data_cfg,
            training_config=self.train_cfg,
            max_items=self.data_cfg.max_items
        )
        self.val_dataset = VoiceConversionDataset(
            split="val",
            audio_config=self.audio_cfg,
            data_config=self.data_cfg,
            training_config=self.train_cfg,
            max_items=self.data_cfg.max_items
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,
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
                "vocoder_state": self.vocoder.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "sched_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
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
        self.model.load_state_dict(ckpt["model_state"], strict=False)

        try:
            self.optimizer.load_state_dict(ckpt["optim_state"])
            self.scheduler.load_state_dict(ckpt["sched_state"])
        except ValueError:
            print("[Warning] Optimizer/Scheduler state mismatch. Continuing with fresh states.")
        self.start_epoch       = ckpt.get("epoch", 0)
        self.global_step       = ckpt.get("global_step", 0)
        self.best_val_loss     = ckpt.get("best_val_loss", float("inf"))
        self.no_improve_epochs = ckpt.get("no_improve_epochs", 0)

        if "vocoder_state" in ckpt:
            self.vocoder.load_state_dict(ckpt["vocoder_state"])
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
            "Decoder Mel Projection": "decoder.mel_proj",
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
        loss_accum_self = {"mel": 0.0, "stft": 0.0, "speaker": 0.0, "total": 0.0}
        loss_accum_cross = {"mel": 0.0, "stft": 0.0, "speaker": 0.0, "total": 0.0}
        num_self = 0
        num_cross = 0

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

            # ------------------------------------------------------------------
            # HYBRID TRAINING: decide conditioning BEFORE forward pass
            # P0 fix: model must be conditioned on TARGET speaker on cross steps
            # Correct math: L_spk( f(content, s_tgt), s_tgt )
            # ------------------------------------------------------------------
            is_cross_speaker = (self.global_step % 2 == 1)
            target_speaker_feats = None
            speaker_feats_for_forward = batch.get("speaker_feats")   # default: source
            target_gt_mel = None
            target_gt_lengths = None

            can_cross = (batch.get("speaker_feats") is not None) or (ref_audio is not None)

            if is_cross_speaker and can_cross:
                target_gt_mel     = torch.roll(gt_mel,     shifts=1, dims=0)
                target_gt_lengths = torch.roll(gt_lengths, shifts=1, dims=0)
                
                if batch.get("speaker_feats") is not None:
                    target_speaker_feats = torch.roll(batch["speaker_feats"], shifts=1, dims=0)
                    speaker_feats_for_forward = target_speaker_feats
                if ref_audio is not None:
                    ref_audio = torch.roll(ref_audio, shifts=1, dims=0)
                    
                print(f"[HYBRID] Cross-speaker | step={self.global_step} "
                    f"| speakers: {batch.get('speaker_id')}")
            else:
                is_cross_speaker = False
                print(f"[HYBRID] Self-recon    | step={self.global_step}")

            # Forward pass — conditioned on target speaker on cross-speaker steps
            pred_mel, _, _ = self.model(
                ref_audio=ref_audio,
                content_audio=content_audio,
                gt_mels=gt_mel,
                compute_losses=False,
                return_aux=False,
                precomputed_speaker_feats=speaker_feats_for_forward,
                precomputed_content_feats=batch.get("content_feats")
            )

            pred_mel = pred_mel.to(torch.float32)
            if hasattr(self.train_cfg, 'lambda_rec') and self.train_cfg.lambda_rec > 0.0:
                max_len = pred_mel.size(-1)
                # OPTIMIZATION: Reduce frames to exactly 32 (VITS standard) 
                # This is exactly 8192 samples, perfectly supporting a 2048 STFT
                slice_len = min(32, max_len)
                start_idx = random.randint(0, max_len - slice_len) if max_len > slice_len else 0
                
                # OPTIMIZATION: Slicing the batch dimension. Backpropagating 32 samples 
                # through a frozen vocoder destroys steps/sec. We apply STFT constraint 
                # to a random sub-batch of 8, which provides plenty of gradient signal.
                sub_batch_size = min(8, pred_mel.size(0))
                b_indices = torch.randperm(pred_mel.size(0))[:sub_batch_size]

                pred_mel_slice = pred_mel[b_indices, :, start_idx:start_idx+slice_len]
                gt_mel_slice = gt_mel[b_indices, :, start_idx:start_idx+slice_len]

                with torch.no_grad():
                    gt_wave_vocoded = self.vocoder(gt_mel_slice.float()).squeeze(1)
                pred_wave = self.vocoder(pred_mel_slice).squeeze(1)
            else:
                pred_wave = None
                gt_wave_vocoded = None

            current_epoch = self.start_epoch
            stft_weight = getattr(self.train_cfg, 'lambda_rec', 0.0)
            mode_str = "Cross-Spk" if is_cross_speaker else "Self-Recon"
            pbar.set_description(f"Train | ep {current_epoch} | {mode_str}")

            # Calculate Losses
            losses = self.loss_fn(
                pred_mel, gt_mel, pred_wave, gt_wave_vocoded,
                gt_lengths=gt_lengths,
                target_gt_mel=target_gt_mel,
                target_gt_lengths=target_gt_lengths,
                is_cross_speaker=is_cross_speaker,
            )
            
            # Use ALL losses (Mel + Speaker Stats) for actual backpropagation!
            total = losses.total()
            last_entropy = getattr(self.model.cross_attn, 'last_entropy', None)
            if last_entropy is not None and self.train_cfg.lambda_entropy > 0:
                total = total + (-self.train_cfg.lambda_entropy * last_entropy)
            last_div = getattr(self.model.mel_encoder, 'last_diversity_loss', None)
            if last_div is not None and self.train_cfg.lambda_diversity > 0:
                total = total + self.train_cfg.lambda_diversity * last_div
            
            print(f"[LOSS_DEBUG] stft_weight={stft_weight:.4f}, "
            f"raw_stft={losses.get('stft', 0):.4f}, "
            f"actual_total={total:.4f}")
            
            # 🔍 ENHANCED SIGNAL AUDIT (Run on first batch of session + periodically)
            if (num_self + num_cross) == 0 and (self.global_step == 0 or self.global_step % 100 == 0):
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

            torch.autograd.set_detect_anomaly(True)  

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
                
                # Mapping network gradients
                mapping_grad_found = False
                for name, param in self.model.cross_attn.mapping_network.named_parameters():
                    if param.grad is not None:
                        print(f"mapping_network.{name} grad: mean={param.grad.abs().mean():.6f}, "
                            f"max={param.grad.abs().max():.6f}")
                        mapping_grad_found = True
                if not mapping_grad_found:
                    print("mapping_network grad: None (no gradient flowing to mapping_network)")

            self.analyze_gradient_flow()

            all_params = list(self.model.parameters()) + list(self.vocoder.parameters())

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=2.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # accumulate losses
            accum = loss_accum_cross if is_cross_speaker else loss_accum_self
            for k in accum:
                accum[k] += losses.get(k, torch.tensor(0.)).item() if k != "total" else total.item()

            if is_cross_speaker:
                num_cross += 1
            else:
                num_self += 1

            pbar.set_postfix(
                mel=f"{losses.get('mel', torch.tensor(0.)).item():.3f}",
                stft=f"{losses.get('stft', torch.tensor(0.)).item():.3f}",
                spk=f"{losses.get('speaker', torch.tensor(0.)).item():.3f}",
                tot=f"{total.item():.3f}",
                lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
            )

            if (
                self.train_cfg.save_every_steps > 0
                and self.global_step % self.train_cfg.save_every_steps == 0
            ):
                self._save_checkpoint("last.ckpt")

            self.global_step += 1

        for k in loss_accum_self:
            if num_self > 0: loss_accum_self[k] /= num_self
            if num_cross > 0: loss_accum_cross[k] /= num_cross

        combined = {k: (loss_accum_self[k] + loss_accum_cross[k]) / 2.0 for k in loss_accum_self}
        combined["self_mel"] = loss_accum_self["mel"]
        combined["cross_spk"] = loss_accum_cross["speaker"]
        return combined


    # ==============================================================
    #  Validation
    # ==============================================================
    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        self.vocoder.eval()
        loss_accum_self = {"mel": 0.0, "stft": 0.0, "speaker": 0.0, "total": 0.0}
        loss_accum_cross = {"mel": 0.0, "stft": 0.0, "speaker": 0.0, "total": 0.0}
        num_self = 0
        num_cross = 0
        val_step = 0

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
            gt_lengths = batch["gt_lengths"].to(self.device)

            # Mirror training: alternate cross-speaker evaluation
            is_cross_speaker = (val_step % 2 == 1)
            speaker_feats_for_forward = batch.get("speaker_feats")
            target_gt_mel = None
            target_gt_lengths = None

            can_cross = (batch.get("speaker_feats") is not None) or (ref_audio is not None)

            if is_cross_speaker and can_cross:
                target_gt_mel     = torch.roll(gt_mel,     shifts=1, dims=0)
                target_gt_lengths = torch.roll(gt_lengths, shifts=1, dims=0)
                
                if batch.get("speaker_feats") is not None:
                    target_speaker_feats = torch.roll(batch["speaker_feats"], shifts=1, dims=0)
                    speaker_feats_for_forward = target_speaker_feats
                if ref_audio is not None:
                    ref_audio = torch.roll(ref_audio, shifts=1, dims=0)
            else:
                is_cross_speaker = False

            # Forward pass - NOW USES ref_audio and optionally precomputed features
            pred_mel, _, _ = self.model(
                ref_audio=ref_audio,
                content_audio=content_audio,
                gt_mels=gt_mel,  # mirrored identically to train pass
                compute_losses=False,
                return_aux=False,
                precomputed_speaker_feats=speaker_feats_for_forward,
                precomputed_content_feats=batch.get("content_feats")
            )
            pred_mel = pred_mel.to(torch.float32)
            if hasattr(self.train_cfg, 'lambda_rec') and self.train_cfg.lambda_rec > 0.0:
                max_len = pred_mel.size(-1)
                slice_len = min(60, max_len)
                start_idx = (max_len - slice_len) // 2 if max_len > slice_len else 0
                
                pred_mel_slice = pred_mel[:, :, start_idx:start_idx+slice_len]
                gt_mel_slice = gt_mel[:, :, start_idx:start_idx+slice_len]

                with torch.no_grad():
                    gt_wave_vocoded = self.vocoder(gt_mel_slice.float()).squeeze(1)
                    pred_wave = self.vocoder(pred_mel_slice).squeeze(1)
            else:
                pred_wave = None
                gt_wave_vocoded = None
            
            losses = self.loss_fn(
                pred_mel, gt_mel, pred_wave, gt_wave_vocoded,
                gt_lengths=gt_lengths,
                target_gt_mel=target_gt_mel,
                target_gt_lengths=target_gt_lengths,
                is_cross_speaker=is_cross_speaker,
            )

            total = losses.total()
            last_entropy = getattr(self.model.cross_attn, 'last_entropy', None)
            if last_entropy is not None and self.train_cfg.lambda_entropy > 0:
                total = total + (-self.train_cfg.lambda_entropy * last_entropy)
            last_div = getattr(self.model.mel_encoder, 'last_diversity_loss', None)
            if last_div is not None and self.train_cfg.lambda_diversity > 0:
                total = total + self.train_cfg.lambda_diversity * last_div

            accum = loss_accum_cross if is_cross_speaker else loss_accum_self
            for k in accum:
                accum[k] += losses.get(k, torch.tensor(0.)).item() if k != "total" else total.item()

            if is_cross_speaker:
                num_cross += 1
            else:
                num_self += 1
            val_step += 1

        for k in loss_accum_self:
            if num_self > 0: loss_accum_self[k] /= num_self
            if num_cross > 0: loss_accum_cross[k] /= num_cross

        combined = {k: (loss_accum_self[k] + loss_accum_cross[k]) / 2.0 for k in loss_accum_self}
        combined["self_mel"] = loss_accum_self["mel"]
        combined["cross_spk"] = loss_accum_cross["speaker"]
        return combined


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
                    # HuBERT Content Features (frozen, safe to cache permanently)
                    hubert_out = self.model.hubert(content_audio_gpu)
                    content_feats = self.model.hubert_proj(hubert_out)[0] # [T, 96]
                    
                    # GT Mel
                    gt_mel = extract_mel_spectrogram(
                        content_audio, 
                        sample_rate=self.audio_cfg.sample_rate
                    )
                    
                # speaker_feats NOT cached: projection is trainable and caching its output
                # would freeze those weights (no gradient ever reaches them). Save ref_audio.
                torch.save({
                    'content_feats': content_feats.cpu(),
                    'ref_audio':     ref_audio.cpu(),
                    'gt_mel':        gt_mel.cpu(),
                    'gt_wave':       content_audio.cpu(),
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

            print(
                f"[epoch {epoch:03d}] train:   self_mel={train_loss['self_mel']:.3f} | stft={train_loss['stft']:.3f} | cross_spk={train_loss['cross_spk']:.3f} | total={train_loss['total']:.3f}"
            )
            print(
                f"                 val:     self_mel={val_loss['self_mel']:.3f} | stft={val_loss['stft']:.3f} | cross_spk={val_loss['cross_spk']:.3f} | total={val_loss['total']:.3f}"
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
