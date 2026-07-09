"""The world must speak truly, deterministically, and only in its own words."""

import torch

from atelier.world import Scene, Tokenizer, World, recite_back


def test_captions_match_factors():
    world = World(seed=3)
    for scene in world.conjure(100):
        fact = recite_back(scene.caption)
        primary = scene.factors[0]
        assert fact["form"] == primary.form
        assert fact["color"] == primary.color
        assert fact["grid"] == primary.grid
        assert fact["size_word"] == primary.size_word
        if fact["facing"] is not None:
            assert fact["facing"] == primary.facing
        assert len(fact["beside"]) == len(scene.factors) - 1
        for (size_word, color, form), shape in zip(fact["beside"], scene.factors[1:]):
            assert (size_word, color, form) == (shape.size_word, shape.color, shape.form)


def test_same_seed_same_world():
    first, second = World(seed=11).conjure(8), World(seed=11).conjure(8)
    for a, b in zip(first, second):
        assert a.caption == b.caption
        assert torch.equal(a.canvas, b.canvas)


def test_canvas_is_well_formed():
    for scene in World(seed=5).conjure(16):
        assert scene.canvas.shape == (3, 64, 64)
        assert scene.canvas.dtype == torch.float32
        assert 0.0 <= scene.canvas.min() <= scene.canvas.max() <= 1.0


def test_vocabulary_is_closed():
    world = World(seed=9)
    vocabulary = set(world.vocabulary())
    tokenizer = Tokenizer(world.vocabulary(), max_len=24)
    for scene in world.conjure(200):
        words = scene.caption.replace(",", "").split()
        assert set(words) <= vocabulary, f"unknown words in: {scene.caption}"
        ids = tokenizer.encode(scene.caption)
        assert ids.shape == (24,)
        assert int((ids == Tokenizer.UNK).sum()) == 0
        assert int((ids != Tokenizer.PAD).sum()) == len(words)
