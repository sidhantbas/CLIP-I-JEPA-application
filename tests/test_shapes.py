"""Tensor-shape sanity for both encoders: a junior engineer should be able
to trace every shape here in one read, so the models must agree with them."""

import torch

from atelier.config import AtelierConfig
from atelier.namer import Gaze, Namer, Tongue
from atelier.sculptor import Imagination, Sculptor, veil
from atelier.world import Tokenizer, World


def test_gaze_shapes():
    gaze = Gaze(dim=128)
    images = torch.rand(4, 3, 64, 64)
    assert gaze(images).shape == (4, 128)


def test_tongue_shapes():
    world = World(seed=1)
    tokenizer = Tokenizer(world.vocabulary(), max_len=24)
    tongue = Tongue(tokenizer.vocab_size, max_len=24, dim=128)
    tokens = tokenizer.encode_batch([s.caption for s in world.conjure(4)])
    assert tokens.shape == (4, 24)
    assert tongue(tokens).shape == (4, 128)


def test_accord_is_finite_and_symmetric_in_shape():
    world = World(seed=2)
    tokenizer = Tokenizer(world.vocabulary(), max_len=24)
    namer = Namer(tokenizer.vocab_size, max_len=24, dim=128, temperature_init=0.07)
    scenes = world.conjure(8)
    images = torch.stack([s.canvas for s in scenes])
    tokens = tokenizer.encode_batch([s.caption for s in scenes])
    loss, acc = namer.accord(images, tokens)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert 0.0 <= acc <= 1.0


def test_veil_covers_the_right_fraction():
    rng = torch.Generator().manual_seed(0)
    for _ in range(50):
        visible, hidden = veil(grid=8, ratio=(0.30, 0.50), rng=rng)
        assert 0.30 * 64 <= len(hidden) <= 0.50 * 64
        assert len(visible) + len(hidden) == 64
        together = torch.cat([visible, hidden]).sort().values
        assert torch.equal(together, torch.arange(64))


def test_sculptor_dream_shapes():
    cfg = AtelierConfig()
    cfg.device = "cpu"
    sculptor = Sculptor(cfg)
    images = torch.rand(2, 3, 64, 64)
    rng = torch.Generator().manual_seed(3)
    visible, hidden = veil(8, (0.30, 0.50), rng)
    context = sculptor.eye(images, keep=visible)
    assert context.shape == (2, len(visible), 128)
    full = sculptor.memory(images)
    assert full.shape == (2, 64, 128)
    guess = sculptor.imagination(context, hidden)
    assert guess.shape == (2, len(hidden), 128)
    loss, relative, spread = sculptor.dream_step(images, visible, hidden)
    assert torch.isfinite(loss) and relative > 0.0 and spread > 0.0


def test_memory_never_learns_by_gradient():
    cfg = AtelierConfig()
    cfg.device = "cpu"
    sculptor = Sculptor(cfg)
    assert all(not p.requires_grad for p in sculptor.memory.parameters())
    before = [p.clone() for p in sculptor.memory.parameters()]
    sculptor.remember()
    after = list(sculptor.memory.parameters())
    for b, a in zip(before, after):
        assert torch.isfinite(a).all()
        assert b.shape == a.shape
