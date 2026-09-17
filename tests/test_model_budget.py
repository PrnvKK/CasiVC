from vc.config import AudioConfig, ModelConfig, TrainingConfig
from vc.model import HubertVCModel


def test_trainable_param_budget():
    model = HubertVCModel(AudioConfig(), ModelConfig(), TrainingConfig(), load_encoders=False)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n < 500_000, f"trainable budget exceeded: {n}"


def test_forward_accepts_per_sample_lengths():
    import torch
    model = HubertVCModel(AudioConfig(), ModelConfig(), TrainingConfig(), load_encoders=False)
    model.eval()
    C = torch.randn(2, 10, 768)
    S = torch.randn(2, 192)
    ml = torch.tensor([5.0, 8.0])
    cl = torch.tensor([7.0, 10.0])
    with torch.no_grad():
        pred, _, _ = model(precomputed_content_feats=C, precomputed_speaker_feats=S,
                           target_lengths=ml, content_lengths=cl)
    assert pred.shape[0] == 2
    assert pred.shape[-1] == 8  # padded to the batch max of the true target lengths
