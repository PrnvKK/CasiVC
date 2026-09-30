import torch

from vc.decoder.convnext_decoder import TemporalResampler


def test_per_sample_lengths_match_isolated_forward():
    """Per-sample resampling must equal an isolated per-sample forward.

    This guards the batch-max warp bug: a short utterance must not be stretched
    using another item's length, and the padded tail must not be time-warped in.
    """
    torch.manual_seed(0)
    r = TemporalResampler(channels=4)
    x = torch.randn(2, 6, 4)
    target_lengths = torch.tensor([3.0, 6.0])
    source_lengths = torch.tensor([2.0, 6.0])
    out = r(x, target_lengths=target_lengths, source_lengths=source_lengths)
    assert out.shape[1] == 6
    for i in range(2):
        si = int(source_lengths[i])
        ti = int(target_lengths[i])
        oi = r(x[i:i + 1, :si], target_length=ti)
        assert torch.allclose(out[i, :ti], oi[0], atol=1e-5)


def test_short_item_not_stretched_to_batch_max():
    torch.manual_seed(1)
    r = TemporalResampler(channels=3)
    x = torch.randn(2, 8, 3)
    # item 0: 1 content frame -> 1 mel frame; item 1 stays long
    tl = torch.tensor([1.0, 8.0])
    sl = torch.tensor([1.0, 8.0])
    out = r(x, target_lengths=tl, source_lengths=sl)
    ref = r(x[0:1, :1], target_length=1)
    assert torch.allclose(out[0, :1], ref[0], atol=1e-5)
