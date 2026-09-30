"""Masked reconstruction losses.

The mask is mandatory: every training path must pass per-sample lengths so that
zero-padded frames never contribute to the objective (the batch-max warp +
unmasked loss was a catastrophic training-validity bug).
"""

import torch


def masked_l1(pred, tgt, lengths):
    tgt = tgt.to(pred.device).float()
    lengths = lengths.to(pred.device)
    T = pred.shape[-1]
    mask = (torch.arange(T, device=pred.device)[None, :] < lengths[:, None]).unsqueeze(1).expand_as(pred).float()
    denom = mask.sum().clamp(min=1.0)
    return ((pred - tgt).abs() * mask).sum() / denom


def masked_spec_conv(pred, tgt, lengths):
    tgt = tgt.to(pred.device).float()
    lengths = lengths.to(pred.device)
    T = pred.shape[-1]
    mask = (torch.arange(T, device=pred.device)[None, :] < lengths[:, None]).unsqueeze(1).expand_as(pred).float()
    num = torch.norm((pred - tgt) * mask, p="fro")
    den = torch.norm(tgt * mask, p="fro") + 1e-6
    return num / den
