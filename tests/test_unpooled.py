"""The second sitting's probes must read what is trivially there: a
one-hot position planted into fake tokens is perfectly legible to a
token glance, and the attentive probe keeps its shapes straight."""

import torch

from atelier.config import AtelierConfig
from atelier.interrogation_unpooled import Attend
from atelier.probing_common import build_probe, question


def _tiny_cfg() -> AtelierConfig:
    cfg = AtelierConfig()
    cfg.device = "cpu"
    cfg.probe_epochs = 30
    cfg.probe_batch = 64
    cfg.scrutiny_hidden = 32
    return cfg


def _planted_tokens(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake rooms of 9 tokens of width 8: noise everywhere, and a loud
    constant vector added at the token whose index is the label."""
    g = torch.Generator().manual_seed(seed)
    tokens = 0.1 * torch.randn(n, 9, 8, generator=g)
    labels = torch.randint(0, 9, (n,), generator=g)
    tokens[torch.arange(n), labels] += 2.0
    return tokens, labels


def test_attend_shapes_and_size():
    probe = Attend(dim=128, n_classes=9)
    assert probe(torch.randn(4, 64, 128)).shape == (4, 9)
    assert sum(p.numel() for p in probe.parameters()) < 60_000


def test_token_glance_reads_planted_position():
    tokens, labels = _planted_tokens(800, seed=0)
    flat = tokens.flatten(1)                      # (800, 72)
    acc = question((flat[:600], flat[600:]), (labels[:600], labels[600:]),
                   lambda: build_probe("glance", 72, 9, 32), seed=0,
                   cfg=_tiny_cfg())
    assert acc == 1.0                             # Gate B: trivially present, fully read


def test_attend_retrieves_planted_content():
    """Attentive pooling summarizes content, not index, so its smoke test
    plants a loud class vector at a random token: the probe must find
    the loud token and read the class written on it."""
    g = torch.Generator().manual_seed(2)
    tokens = 0.1 * torch.randn(800, 9, 16, generator=g)
    labels = torch.randint(0, 9, (800,), generator=g)
    spots = torch.randint(0, 9, (800,), generator=g)
    tokens[torch.arange(800), spots, labels] += 4.0
    acc = question((tokens[:600], tokens[600:]), (labels[:600], labels[600:]),
                   lambda: Attend(16, 9), seed=0, cfg=_tiny_cfg())
    assert acc > 0.9
