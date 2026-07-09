"""The developer's plumbing, checked dry (no giants are summoned)."""

import torch

from darkroom.developer import Developer, cosine_lr


def test_one_negative_becomes_a_prefix_strip():
    developer = Developer(clip_dim=512, gpt2_dim=768, prefix_len=10, hidden=64)
    strip = developer(torch.randn(3, 512))
    assert strip.shape == (3, 10, 768)


def test_everything_in_the_bath_trains():
    developer = Developer(clip_dim=512, gpt2_dim=768, prefix_len=10, hidden=64)
    assert all(p.requires_grad for p in developer.parameters())


def test_cosine_lr_warms_up_then_dies():
    base = 3e-4
    assert cosine_lr(0, 1000, base, warmup=100) < base / 50
    assert abs(cosine_lr(100, 1000, base, warmup=100) - base) < base * 0.02
    assert cosine_lr(999, 1000, base, warmup=100) < base / 100
