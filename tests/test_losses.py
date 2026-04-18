import torch

from src.training.losses import bpr_loss


def test_bpr_positive_higher_is_lower_loss():
    pos = torch.tensor([1.0, 2.0])
    neg_high = torch.tensor([-1.0, -2.0])
    neg_low = torch.tensor([0.5, 1.5])
    assert bpr_loss(pos, neg_high) < bpr_loss(pos, neg_low)


def test_bpr_multi_negative():
    pos = torch.tensor([1.0, 2.0])
    neg = torch.tensor([[-1.0, 0.0], [-2.0, 1.0]])
    loss = bpr_loss(pos, neg)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_bpr_weights_scale_loss():
    pos = torch.tensor([1.0, 2.0])
    neg = torch.tensor([-1.0, -2.0])
    w1 = torch.tensor([1.0, 1.0])
    w2 = torch.tensor([2.0, 2.0])
    assert bpr_loss(pos, neg, w2).item() > bpr_loss(pos, neg, w1).item()


def test_bpr_rejects_bad_shape():
    import pytest

    pos = torch.tensor([1.0])
    neg = torch.zeros((1, 2, 3))
    with pytest.raises(ValueError):
        bpr_loss(pos, neg)
