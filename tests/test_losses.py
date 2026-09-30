import torch

from vc.losses.masked import masked_l1, masked_spec_conv


def test_masked_l1_ignores_padding():
    pred = torch.zeros(1, 2, 5)
    tgt = torch.zeros(1, 2, 5)
    tgt[:, :, 3:] = 100.0  # garbage in the padded tail
    lengths = torch.tensor([3.0])
    # only the first 3 frames are valid; padding must not contribute
    assert masked_l1(pred, tgt, lengths).item() == 0.0


def test_masked_l1_equals_manual_valid_region():
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 7)
    tgt = torch.randn(2, 3, 7)
    lengths = torch.tensor([4.0, 7.0])
    got = masked_l1(pred, tgt, lengths).item()
    manual = []
    for i in range(2):
        L = int(lengths[i])
        manual.append((pred[i, :, :L] - tgt[i, :, :L]).abs().mean().item())
    # weighted equally by frame count: sample1 contributes 7, sample2 contributes 4
    exp = (4 * manual[0] + 7 * manual[1]) / 11
    assert abs(got - exp) < 1e-6


def test_masked_spec_conv_is_zero_when_equal():
    pred = torch.randn(1, 3, 6)
    assert masked_spec_conv(pred, pred, torch.tensor([6.0])).item() < 1e-6
