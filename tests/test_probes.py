"""The probes must be able to find what is truly there: a glance learns a
linear boundary, a scrutiny learns a folded one, labels tell the truth."""

import torch

from atelier.config import AtelierConfig
from atelier.probing_common import build_probe, flatten_to_pca, question
from atelier.probing_common import testify as recall_labels
from atelier.world import COMPASS_8, FORMS, World


def _tiny_cfg() -> AtelierConfig:
    cfg = AtelierConfig()
    cfg.device = "cpu"
    cfg.probe_epochs = 30
    cfg.probe_batch = 64
    cfg.scrutiny_hidden = 32
    return cfg


def test_probe_shapes():
    glance = build_probe("glance", dim=16, n_classes=3, hidden=32)
    scrutiny = build_probe("scrutiny", dim=16, n_classes=3, hidden=32)
    x = torch.randn(5, 16)
    assert glance(x).shape == (5, 3)
    assert scrutiny(x).shape == (5, 3)


def test_glance_learns_linear_truth():
    torch.manual_seed(0)
    x = torch.randn(600, 8)
    y = (x[:, 0] + x[:, 1] > 0).long()
    acc = question((x[:400], x[400:]), (y[:400], y[400:]),
                   lambda: build_probe("glance", 8, 2, 32), seed=0, cfg=_tiny_cfg())
    assert acc > 0.9


def test_scrutiny_reaches_past_the_glance():
    torch.manual_seed(0)
    x = torch.randn(1200, 8)
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).long()   # XOR, invisible to a glance
    cfg = _tiny_cfg()
    split = 900
    features, labels = (x[:split], x[split:]), (y[:split], y[split:])
    glance = question(features, labels,
                      lambda: build_probe("glance", 8, 2, 32), 0, cfg)
    scrutiny = question(features, labels,
                        lambda: build_probe("scrutiny", 8, 2, 32), 0, cfg)
    assert glance < 0.65                          # a glance cannot see XOR
    assert scrutiny > 0.85                        # a scrutiny can


def test_testify_matches_ground_truth():
    scenes = World(seed=4).conjure(50)
    forms = recall_labels(scenes, "form")
    facing = recall_labels(scenes, "orientation")
    for scene, f, o in zip(scenes, forms, facing):
        assert FORMS[f] == scene.factors[0].form
        assert COMPASS_8[o] == scene.factors[0].facing


def test_pca_floor_shapes():
    train = torch.randint(0, 255, (64, 3, 64, 64), dtype=torch.uint8)
    test = torch.randint(0, 255, (16, 3, 64, 64), dtype=torch.uint8)
    x_tr, x_te = flatten_to_pca(train, test, dim=12)
    assert x_tr.shape == (64, 12)
    assert x_te.shape == (16, 12)
